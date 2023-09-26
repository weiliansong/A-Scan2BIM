import argparse
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import sqrtm
from scipy.spatial import distance
from scipy.stats import entropy
from tqdm import tqdm

import my_utils
from datasets.building_ae import normalize_edges
from datasets.building_order_enum import BuildingCornerDataset, collate_fn_seq
from models.order_class_models import EdgeTransformer as ClassifierModel
from models.order_metric_models import EdgeTransformer as MetricModel
from models.tcn import TCN

# set random seed?

DATA_PATH = "./data/bim_dataset_big_v5/"
# METRIC_DIR = "./ckpts_vml/04_21_order_metric_noisy"
METRIC_DIR = "./ckpts_edge/04_17_order_metric_working"
# METRIC_DIR = "./ckpts_vml/02_23_metric_reduce_sum"
CLASS_DIR = "./ckpts_edge/04_14_order_class"
TCN_DIR = "./ckpts_ae"

match_threshold = 30


def get_seq_heuristic(
    curr_seq,
    cand_edges,
    target_len,
):
    all_edges = np.concatenate([curr_seq, cand_edges])

    edge_order = np.zeros(len(all_edges))
    edge_order[: len(curr_seq)] = np.arange(len(curr_seq), 0, -1)
    query_i = len(curr_seq) - 1

    full_dists = distance.squareform(distance.pdist(all_edges, my_utils.nearest_dist))

    new_seq = curr_seq.copy()
    while len(new_seq) < target_len:
        dists = full_dists[query_i].copy()

        # prevent from using previously selected edges
        used_mask = edge_order > 0
        dists[used_mask] = float("inf")

        # pick the best next edge
        best_idx = dists.argmin()
        new_seq = np.concatenate([new_seq, all_edges[best_idx][None, :]], axis=0)
        edge_order[edge_order > 0] += 1
        edge_order = np.minimum(edge_order, 10)
        edge_order[best_idx] = 1
        query_i = best_idx

    return new_seq


def get_next_edge_prob_fast(
    cond_edges,
    curr_seq,
    cand_edges,
    edge_model,
    max_order=10,
):
    full_seq = np.concatenate([cond_edges, curr_seq], axis=0)

    cand_seqs = np.tile(full_seq, (cand_edges.shape[0], 1, 1))
    cand_seqs = np.concatenate([cand_seqs, np.expand_dims(cand_edges, 1)], axis=1)

    # per sequence normalization
    _cand_seqs = []

    for seq in cand_seqs:
        corners, edges = my_utils.corners_and_edges(seq)
        corners, scale = my_utils.normalize_corners(corners)
        seq = corners[edges].reshape(-1, 4)
        _cand_seqs.append(seq)

    cand_seqs = np.array(_cand_seqs)

    # data prep and inference
    edge_order = np.zeros(cand_seqs.shape[1])

    start_idx = len(cond_edges)
    end_idx = start_idx + len(curr_seq) + 1

    edge_order[:start_idx] = max_order
    edge_order[start_idx:end_idx] = list(range(end_idx - start_idx, 0, -1))
    edge_order = np.minimum(edge_order, max_order)

    examples = [{"edge_coords": edges, "edge_order": edge_order} for edges in cand_seqs]
    data = collate_fn_seq(examples)

    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].cuda(0)

    logits = edge_model(data)
    probs = logits.softmax(-1)[:, 1].detach().cpu().numpy()

    return probs


def get_next_edge_prob_fast_old(
    curr_seq,
    cand_edges,
    edge_model,
):
    cand_seqs = np.tile(curr_seq, (cand_edges.shape[0], 1, 1))
    cand_seqs = np.concatenate([cand_seqs, np.expand_dims(cand_edges, 1)], axis=1)

    # per sequence normalization
    cand_seqs = np.array([my_utils.normalize_edges(edges) for edges in cand_seqs])

    # data prep and inference
    edge_order = np.array(list(range(cand_seqs.shape[1], 0, -1)))
    examples = [{"edge_coords": edges, "edge_order": edge_order} for edges in cand_seqs]
    data = collate_fn_seq(examples)

    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].cuda(0)

    logits = edge_model(data)
    probs = logits.softmax(-1)[:, 1].detach().cpu().numpy()

    return probs


def get_seq_classifier(
    model,
    cond_edges,
    curr_seq,
    cand_edges,
    target_len,
):
    curr_seq = curr_seq.copy()
    cand_edges = cand_edges.copy()

    while len(curr_seq) < target_len:
        # obtain probabilities for each possible sequence
        probs = get_next_edge_prob_fast(cond_edges, curr_seq, cand_edges, model)

        # pick the most probable, update lists
        next_edge = cand_edges[probs.argmax()]
        curr_seq = np.append(curr_seq, next_edge[None, :], axis=0)
        cand_edges = np.delete(cand_edges, probs.argmax(), axis=0)

    return curr_seq


def vis_probs(density_full, all_edges, edge_order, dists):
    density_img = 0.5 * np.max(density_full, axis=2)
    plt.imshow(density_img, cmap="gray", vmin=0, vmax=1)

    cmap = plt.get_cmap("hot")
    dists = dists.detach().cpu().numpy()

    for edge_i, (x0, y0, x1, y1) in enumerate(all_edges):
        if edge_order[edge_i] != 0:
            plt.plot([x0, x1], [y0, y1], "-oy")
            plt.text((x0 + x1) / 2, (y0 + y1) / 2, "%d" % edge_order[edge_i], color="c")
        else:
            prob = 1 - (dists[edge_i] / 2)
            plt.plot([x0, x1], [y0, y1], "-o", color=cmap(prob))
            plt.text((x0 + x1) / 2, (y0 + y1) / 2, "%.2f" % prob, color="c")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_seq_metric(
    model,
    cond_edges,
    curr_seq,
    cand_edges,
    target_len,
    max_order=10,
    density_full=None,
):
    assert len(curr_seq) < 10

    dist_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)

    all_edges = np.concatenate([cond_edges, curr_seq, cand_edges])
    edge_coords, _ = my_utils.metric_normalize_edges(all_edges)

    edge_order = np.zeros(len(all_edges))

    start_idx = len(cond_edges)
    end_idx = start_idx + len(curr_seq)

    edge_order[:start_idx] = max_order
    edge_order[start_idx:end_idx] = list(range(end_idx - start_idx, 0, -1))
    edge_order = np.minimum(edge_order, max_order)

    query_i = end_idx - 1

    new_seq = curr_seq.copy()
    seq_probs = []
    while len(new_seq) < target_len:
        example = {"edge_coords": edge_coords, "edge_order": edge_order}
        data = my_utils.metric_collate_fn([example])

        for key in data.keys():
            if type(data[key]) is torch.Tensor:
                data[key] = data[key].cuda()

        embeddings = model(data)[0]
        dists = dist_fn(embeddings[query_i : query_i + 1], embeddings)

        # prevent from using previously selected edges
        used_mask = edge_order > 0
        dists[used_mask] = float("inf")

        probs = dists[~used_mask].detach().cpu().numpy()
        probs = 1 - (probs / 2)  # the distance is inverse to the prob
        # probs = my_utils.softmax(probs)
        seq_probs.append(probs)

        # probs_2 = -1 * probs
        # probs_2 = my_utils.softmax(probs_2)
        # probs_3 = -10 * probs
        # probs_3 = my_utils.softmax(probs_3)
        # plt.plot(probs_2, '-')
        # plt.plot(probs_3, '-')
        # plt.show()

        # vis_probs(density_full, all_edges, edge_order, dists)

        # pick the best next edge
        best_idx = dists.argmin()
        new_seq = np.concatenate([new_seq, all_edges[best_idx][None, :]], axis=0)
        edge_order[edge_order > 0] += 1
        edge_order = np.minimum(edge_order, max_order)
        edge_order[best_idx] = 1
        query_i = best_idx

    return new_seq, seq_probs


def vis_fid(density_full, gt_edges, tcn):
    seqs = {}
    hiddens = {}

    seq_len = 10

    for i in range(seq_len + 1):
        seqs[i] = []
        hiddens[i] = []

    for seq_i, start_idx in enumerate(range(len(gt_edges) - seq_len)):
        print("%d / %d" % (seq_i, len(gt_edges) - seq_len))

        end_idx = start_idx + seq_len

        seq_gt = gt_edges[start_idx:end_idx]
        hidden_gt = get_hidden(seq_gt, tcn, 6)

        seqs[0].append(seq_gt)
        hiddens[0].append(hidden_gt)

        # replace edges with out-of-sequence ones
        avail_mask = np.ones(len(gt_edges), dtype=bool)
        avail_mask[start_idx:end_idx] = False

        for num_replace in range(1, seq_len + 1):
            replace_inds = np.random.choice(
                range(seq_len), size=num_replace, replace=False
            )
            with_inds = np.random.choice(
                avail_mask.nonzero()[0], size=num_replace, replace=False
            )

            seq_damaged = seq_gt.copy()
            seq_damaged[replace_inds] = gt_edges[with_inds]
            seqs[num_replace].append(seq_damaged.copy())
            hiddens[num_replace].append(get_hidden(seq_damaged, tcn, 6))

    scores = []

    for i in range(seq_len + 1):
        hidden_gt = np.array(hiddens[0])
        hidden_dmg = np.array(hiddens[i])

        fid = calculate_fid(hidden_gt, hidden_dmg)
        scores.append(fid)

    plt.plot(range(seq_len + 1), scores, "-o")
    plt.xlabel("Number of replacements")
    plt.ylabel("Perceptual score")
    plt.show()
    plt.close()


def vis_seq(density_full, all_edges, cond_edges, seq, cond_len, title):
    font = ImageFont.truetype("SourceCodePro-Regular.ttf", 48)

    density_slice = (density_full[:, :, 1] * 255).round().astype(np.uint8)
    density_pil = Image.fromarray(density_slice).convert("RGB")

    draw = ImageDraw.Draw(density_pil)
    draw.text((10, 10), title, (0, 255, 255), font=font)
    for xyxy in all_edges.tolist():
        draw.line(xyxy, fill=(255, 0, 0), width=5)
    for xyxy in cond_edges.tolist():
        draw.line(xyxy, fill=(0, 255, 255), width=5)

    frames = [np.array(density_pil)]
    for edge_i, xyxy in enumerate(seq.tolist()):
        if edge_i < cond_len:
            draw.line(xyxy, fill=(0, 100, 100), width=10)
        else:
            draw.line(xyxy, fill=(255, 255, 0), width=10)
        frames.append(np.array(density_pil))

    return frames


def save_seq(density_full, all_edges, seq, cond_len, save_f):
    frames = vis_seq(density_full, all_edges, seq, cond_len)
    imageio.mimsave(save_f, frames, duration=0.2, loop=0)


def compare_seq_failure(
    density_full,
    cond_edges,
    curr_seq,
    seq_gt,
    seq_metric,
    gt_edges,
    pred_edges,
    save_f=None,
    show_steps=False,
):
    # for making plots square
    def on_press(event):
        if event.key != " ":
            return

        event_ax = event.inaxes
        (minx, maxx) = event_ax.get_xlim()
        (miny, maxy) = event_ax.get_ylim()

        w = maxx - minx
        h = miny - maxy

        if w > h:
            side_len = w
            cy = (miny + maxy) / 2

            new_miny = cy + side_len / 2
            new_maxy = cy - side_len / 2

            event_ax.set_ylim(new_miny, new_maxy)
        else:
            side_len = h
            cx = (minx + maxx) / 2

            new_minx = cx - side_len / 2
            new_maxx = cx + side_len / 2

            event_ax.set_xlim(new_minx, new_maxx)

        event.canvas.draw()

    GREEN = "#00ff7fff"
    PINK = "#ffafffff"

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    size = fig.get_size_inches()
    fig.set_size_inches(size * 4)

    image = 0.5 * np.max(density_full, axis=2)
    for ax in axes:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    # plot all the available candidates
    for x0, y0, x1, y1 in pred_edges:
        axes[0].plot([x0, x1], [y0, y1], "--o", color="c")

    for x0, y0, x1, y1 in gt_edges:
        axes[1].plot([x0, x1], [y0, y1], "--o", color="c")

    # plot the current edges
    for x0, y0, x1, y1 in cond_edges:
        for ax in axes:
            ax.plot([x0, x1], [y0, y1], "-c")

    for curr_i, (x0, y0, x1, y1) in enumerate(curr_seq):
        for ax in axes:
            ax.plot([x0, x1], [y0, y1], "-o", color=GREEN)
            if show_steps:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, "%d" % curr_i, color="c")

    for ax, seq in zip(axes, [seq_metric, seq_gt]):
        for curr_i, (x0, y0, x1, y1) in enumerate(seq):
            ax.plot([x0, x1], [y0, y1], "-o", color=PINK)
            if show_steps:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, "%d" % curr_i, color="c")

    # axes[0].set_title("Metric")
    # axes[1].set_title("GT")

    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")

    plt.tight_layout()
    if save_f:
        plt.savefig(save_f, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        fig.canvas.mpl_connect("key_press_event", on_press)
        plt.show()


def compare_seq_static(
    density_full,
    cond_edges,
    curr_seq,
    seq_gt,
    seq_heuristic,
    seq_class,
    seq_metric,
    scores,
    save_f=None,
    show_titles=False,
    show_steps=False,
):
    GREEN = "#00ff7fff"
    PINK = "#ffafffff"

    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
    size = fig.get_size_inches()
    fig.set_size_inches(size * 4)

    image = 0.5 * np.max(density_full, axis=2)
    for ax in axes:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    # plot the current edges
    for x0, y0, x1, y1 in cond_edges:
        for ax in axes:
            ax.plot([x0, x1], [y0, y1], "-c")

    for curr_i, (x0, y0, x1, y1) in enumerate(curr_seq):
        for ax in axes:
            ax.plot([x0, x1], [y0, y1], "-o", color=GREEN)
            if show_steps:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, "%d" % curr_i, color="c")

    for ax, seq in zip(axes, [seq_heuristic, seq_class, seq_metric, seq_gt]):
        for curr_i, (x0, y0, x1, y1) in enumerate(seq):
            ax.plot([x0, x1], [y0, y1], "-o", color=PINK)
            if show_steps:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, "%d" % curr_i, color="c")

    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")

    if show_titles:
        axes[0].set_title("Heuristics %.3f" % scores[0])
        axes[1].set_title("Classifier %.3f" % scores[1])
        axes[2].set_title("Metric %.3f" % scores[2])
        axes[3].set_title("Ground-truth")

    plt.tight_layout()
    if save_f:
        plt.savefig(save_f, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def compare_seq_fig(
    density_full,
    cond_edges,
    curr_seq,
    seq_gt,
    seq_heuristic,
    seq_class,
    seq_metric,
    scores,
):
    GREEN = "#00ff7fff"
    PINK = "#ffafffff"

    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
    size = fig.get_size_inches()
    fig.set_size_inches(size * 4)

    image = 0.5 * np.max(density_full, axis=2)
    for ax in axes:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    # plot the current edges
    for x0, y0, x1, y1 in cond_edges:
        for ax in axes:
            ax.plot([x0, x1], [y0, y1], "-c")

    for curr_i, (x0, y0, x1, y1) in enumerate(curr_seq):
        for ax in axes:
            ax.plot([x0, x1], [y0, y1], "-o", color=GREEN)

    for ax, seq in zip(axes, [seq_heuristic, seq_class, seq_metric, seq_gt]):
        for curr_i, (x0, y0, x1, y1) in enumerate(seq):
            ax.plot([x0, x1], [y0, y1], "-o", color=PINK)

    def on_xlims_change(event_ax):
        (minx, maxx) = event_ax.get_xlim()
        (miny, maxy) = event_ax.get_ylim()

        w = maxx - minx
        h = miny - maxy

        if w != h:
            side_len = w
            cy = (miny + maxy) / 2

            new_miny = cy + side_len / 2
            new_maxy = cy - side_len / 2

            event_ax.set_ylim(new_miny, new_maxy)

    axes[0].callbacks.connect("xlim_changed", on_xlims_change)

    for ax in axes:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def compare_seq(
    density_full,
    all_edges,
    save_f,
    cond_len,
    cond_edges,
    seq_gt,
    seq_heuristic,
    seq_class,
    seq_metric,
    tcn,
):
    hiddens_gt = get_hidden(seq_gt, tcn, 6)
    hiddens_class = get_hidden(seq_class, tcn, 6)
    hiddens_metric = get_hidden(seq_metric, tcn, 6)
    hiddens_heuristic = get_hidden(seq_heuristic, tcn, 6)

    gt2class = np.linalg.norm(hiddens_gt - hiddens_class)
    gt2metric = np.linalg.norm(hiddens_gt - hiddens_metric)
    gt2heuristic = np.linalg.norm(hiddens_gt - hiddens_heuristic)

    frames_gt = vis_seq(
        density_full,
        all_edges,
        cond_edges,
        seq_gt,
        cond_len,
        "GT",
    )
    frames_heuristic = vis_seq(
        density_full,
        all_edges,
        cond_edges,
        seq_heuristic,
        cond_len,
        "Heuristic: %.3f" % gt2heuristic,
    )
    frames_class = vis_seq(
        density_full,
        all_edges,
        cond_edges,
        seq_class,
        cond_len,
        "Classifier: %.3f" % gt2class,
    )
    frames_metric = vis_seq(
        density_full,
        all_edges,
        cond_edges,
        seq_metric,
        cond_len,
        "Metric: %.3f" % gt2metric,
    )

    frames = []
    for frame_i in range(len(frames_gt)):
        frame = np.concatenate(
            [
                frames_gt[frame_i],
                frames_heuristic[frame_i],
                frames_class[frame_i],
                frames_metric[frame_i],
            ],
            axis=1,
        )
        frames.append(frame)

    imageio.mimsave(save_f, frames, duration=0.2, loop=0)


def get_hidden(seq, tcn, layer_idx):
    seq, _ = normalize_edges(seq)
    tcn_inputs = torch.tensor(seq).cuda().float()
    tcn_inputs = tcn_inputs.reshape(1, 1, -1)
    hidden = tcn.get_hidden(tcn_inputs)[layer_idx]
    # hidden = hidden.mean(dim=2)  # [1, D, N] -> [1, D]
    return hidden.flatten().detach().cpu().numpy()


# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def eval_floor(floor_idx, target_len, edge_type):
    # load floor data
    class_dataset = BuildingCornerDataset(
        DATA_PATH,
        None,
        phase="test",
        rand_aug=False,
        test_idx=floor_idx,
        loss_type="class",
    )
    metric_dataset = BuildingCornerDataset(
        DATA_PATH,
        None,
        phase="test",
        rand_aug=False,
        test_idx=floor_idx,
        loss_type="metric",
    )

    floor_name = list(class_dataset.density_fulls.keys())[0]
    density_full = class_dataset.density_fulls[floor_name]
    gt_edges = class_dataset.ordered_edges[floor_name]
    heat_edges = class_dataset.heat_edges[floor_name]

    # load classifier model
    class_model = ClassifierModel(d_model=256)

    class_ckpt_f = "%s/%d/checkpoint_latest.pth" % (CLASS_DIR, floor_idx)
    class_ckpt = torch.load(class_ckpt_f)
    class_model.load_state_dict(class_ckpt["edge_model"])
    class_model = class_model.cuda()
    class_model.eval()

    # load metric model
    metric_model = MetricModel(d_model=256)

    metric_ckpt_f = "%s/%d/checkpoint_latest.pth" % (METRIC_DIR, floor_idx)
    metric_ckpt = torch.load(metric_ckpt_f)
    metric_model.load_state_dict(metric_ckpt["edge_model"])
    metric_model = metric_model.cuda()
    metric_model.eval()

    # load our evaluation model
    channel_sizes = [10] * 8
    kernel_size = 8
    dropout = 0
    tcn = TCN(1, 1, channel_sizes, kernel_size=kernel_size, dropout=dropout)

    tcn_ckpt_f = "%s/0/checkpoint_latest.pth" % (TCN_DIR)
    tcn_ckpt = torch.load(tcn_ckpt_f)
    tcn.load_state_dict(tcn_ckpt)
    tcn = tcn.cuda()
    tcn.eval()

    # visualize the quality of the FID score
    # vis_fid(density_full, gt_edges, tcn)

    print("Getting GT sequences")

    hiddens_gt = []
    for start_idx in tqdm(range(0, len(gt_edges) - target_len)):
        seq_gt = gt_edges[start_idx : start_idx + target_len]
        hiddens_gt.append(get_hidden(seq_gt, tcn, 6))

    # need to consider different provided history length

    print("Getting predicted sequences")

    hiddens_class = []
    hiddens_metric = []
    hiddens_heuristic = []

    print("Edge type: %s" % edge_type)
    if edge_type == "GT":
        src_edges = gt_edges
    elif edge_type == "HEAT":
        src_edges = heat_edges
    else:
        raise Exception("Unknown edge type")

    # _gt_edges, _ = my_utils.revectorize(gt_edges, np.zeros(len(gt_edges)))
    # labels, pred2gt = my_utils.compute_label(
    #     pred_coords=src_edges, gt_coords=_gt_edges, threshold=30
    # )

    # for seq_i, (start_idx, end_idx) in enumerate(tqdm(seqs)):
    for seq_i, start_idx in enumerate(tqdm(range(len(src_edges)))):
        # for start_idx in tqdm(range(0, len(gt_edges) - target_len)):
        # if labels[seq_i] != 1:
        #     continue

        cond_edges = src_edges[:0]

        curr_mask = np.zeros(len(src_edges), dtype=bool)
        curr_mask[start_idx] = True

        curr_seq = src_edges[curr_mask]
        cand_edges = src_edges[~curr_mask]

        rand_idx = np.arange(len(cand_edges))
        np.random.shuffle(rand_idx)
        cand_edges = cand_edges[rand_idx]

        # curr_seq = gt_edges[start_idx : start_idx + 1]
        # cand_edges = my_utils.find_candidates(curr_seq, src_edges)

        # seq_heuristic = get_seq_heuristic(
        #     curr_seq=curr_seq,
        #     cand_edges=cand_edges,
        #     target_len=target_len,
        # )

        seq_class = get_seq_classifier(
            model=class_model,
            cond_edges=cond_edges,
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=target_len,
        )
        seq_heuristic = seq_class

        seq_metric, _ = get_seq_metric(
            model=metric_model,
            cond_edges=cond_edges,
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=target_len,
            density_full=density_full,
        )

        # my_utils.vis_edges(density_full, [["-o", seq_metric]])

        # used to compute FID scores
        # seq_heuristic = seq_heuristic[len(curr_seq) :]
        # seq_class = seq_class[len(curr_seq) :]
        # seq_metric = seq_metric[len(curr_seq) :]

        hiddens_class.append(get_hidden(seq_class, tcn, 6))
        hiddens_metric.append(get_hidden(seq_metric, tcn, 6))
        hiddens_heuristic.append(get_hidden(seq_heuristic, tcn, 6))

        # save visualization
        # compare_seq_static(
        #     density_full,
        #     cond_edges,
        #     curr_seq,
        #     seq_gt,
        #     seq_heuristic,
        #     seq_class,
        #     seq_metric,
        #     [0, 0, 0],
        # )
        # if seq_i < 10:
        #     compare_seq(
        #         density_full=density_full,
        #         all_edges=gt_edges,
        #         save_f="vis/compare_%s_%03d.gif" % (floor_name, seq_i),
        #         cond_len=0,
        #         cond_edges=cond_edges,
        #         seq_gt=seq_gt,
        #         seq_heuristic=seq_heuristic,
        #         seq_class=seq_class,
        #         seq_metric=seq_metric,
        #         tcn=tcn,
        #     )

    # compute FID scores
    hiddens_gt = np.array(hiddens_gt)
    hiddens_class = np.array(hiddens_class)
    hiddens_metric = np.array(hiddens_metric)
    hiddens_heuristic = np.array(hiddens_heuristic)

    fid_gt = calculate_fid(hiddens_gt, hiddens_gt)
    fid_class = calculate_fid(hiddens_gt, hiddens_class)
    fid_metric = calculate_fid(hiddens_gt, hiddens_metric)
    fid_heuristic = calculate_fid(hiddens_gt, hiddens_heuristic)

    print("FID GT       : %.3f" % fid_gt)
    print("FID heuristic: %.3f" % fid_heuristic)
    print("FID class    : %.3f" % fid_class)
    print("FID metric   : %.3f" % fid_metric)

    with open("fid_csv/fid_%s_%02d.csv" % (edge_type, floor_idx), "a") as f:
        f.write(
            "%d,%d,%.3f,%.3f,%.3f,%.3f\n"
            % (floor_idx, target_len, fid_gt, fid_heuristic, fid_class, fid_metric)
        )


def eval_floor_2(floor_idx, target_len, edge_type):
    # load floor data
    class_dataset = BuildingCornerDataset(
        DATA_PATH,
        None,
        phase="test",
        rand_aug=False,
        test_idx=floor_idx,
        loss_type="class",
    )
    metric_dataset = BuildingCornerDataset(
        DATA_PATH,
        None,
        phase="test",
        rand_aug=False,
        test_idx=floor_idx,
        loss_type="metric",
    )

    floor_name = list(class_dataset.density_fulls.keys())[0]
    density_full = class_dataset.density_fulls[floor_name]
    gt_edges = class_dataset.ordered_edges[floor_name]
    heat_edges = class_dataset.heat_edges[floor_name]

    # load classifier model
    class_model = ClassifierModel(d_model=256)

    class_ckpt_f = "%s/%d/checkpoint_latest.pth" % (CLASS_DIR, floor_idx)
    class_ckpt = torch.load(class_ckpt_f)
    class_model.load_state_dict(class_ckpt["edge_model"])
    class_model = class_model.cuda()
    class_model.eval()

    # load metric model
    metric_model = MetricModel(d_model=256)

    metric_ckpt_f = "%s/%d/checkpoint_latest.pth" % (METRIC_DIR, floor_idx)
    metric_ckpt = torch.load(metric_ckpt_f)
    metric_model.load_state_dict(metric_ckpt["edge_model"])
    metric_model = metric_model.cuda()
    metric_model.eval()

    # load our evaluation model
    channel_sizes = [10] * 8
    kernel_size = 8
    dropout = 0
    tcn = TCN(1, 1, channel_sizes, kernel_size=kernel_size, dropout=dropout)

    tcn_ckpt_f = "%s/0/checkpoint_latest.pth" % (TCN_DIR)
    tcn_ckpt = torch.load(tcn_ckpt_f)
    tcn.load_state_dict(tcn_ckpt)
    tcn = tcn.cuda()
    tcn.eval()

    # visualize the quality of the FID score
    # vis_fid(density_full, gt_edges, tcn)

    print("Getting GT sequences")

    hiddens_gt = []
    for start_idx in tqdm(range(0, len(gt_edges) - target_len)):
        seq_gt = gt_edges[start_idx : start_idx + target_len]
        hiddens_gt.append(get_hidden(seq_gt, tcn, 6))

    # need to consider different provided history length

    print("Getting predicted sequences")

    hiddens_class = []
    hiddens_metric = []
    hiddens_heuristic = []

    print("Edge type: %s" % edge_type)
    if edge_type == "GT":
        src_edges = gt_edges
    elif edge_type == "HEAT":
        src_edges = heat_edges
    else:
        raise Exception("Unknown edge type")

    seqs = []
    cond_len = 10

    for start_idx in range(0, len(gt_edges) - cond_len):
        end_idx = start_idx + cond_len
        seqs.append((start_idx, end_idx))

    # for seq_i, start_idx in enumerate(tqdm(range(len(src_edges)))):
    for seq_i, (start_idx, end_idx) in enumerate(tqdm(seqs)):
        cond_edges = src_edges[:0]

        # curr_mask = np.zeros(len(src_edges), dtype=bool)
        # curr_mask[start_idx] = True

        # curr_seq = src_edges[curr_mask]
        # cand_edges = src_edges[~curr_mask]

        curr_seq = gt_edges[start_idx:end_idx]
        cand_edges = my_utils.find_candidates(curr_seq, src_edges)

        seq_class = get_seq_classifier(
            model=class_model,
            cond_edges=cond_edges,
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=len(curr_seq) + target_len,
        )

        # seq_heuristic = get_seq_heuristic(
        #     curr_seq=curr_seq,
        #     cand_edges=cand_edges,
        #     target_len=len(curr_seq) + target_len,
        # )
        seq_heuristic = seq_class

        seq_metric, _ = get_seq_metric(
            model=metric_model,
            cond_edges=cond_edges,
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=len(curr_seq) + target_len,
        )

        # my_utils.vis_edges(density_full, [["-o", seq_metric]])

        # used to compute FID scores
        seq_heuristic = seq_heuristic[len(curr_seq) :]
        seq_class = seq_class[len(curr_seq) :]
        seq_metric = seq_metric[len(curr_seq) :]

        hiddens_class.append(get_hidden(seq_class, tcn, 6))
        hiddens_metric.append(get_hidden(seq_metric, tcn, 6))
        hiddens_heuristic.append(get_hidden(seq_heuristic, tcn, 6))

        # save visualization
        # compare_seq_static(
        #     density_full,
        #     cond_edges,
        #     curr_seq,
        #     seq_gt,
        #     seq_heuristic,
        #     seq_class,
        #     seq_metric,
        #     [0, 0, 0],
        # )
        # if seq_i < 10:
        #     compare_seq(
        #         density_full=density_full,
        #         all_edges=gt_edges,
        #         save_f="vis/compare_%s_%03d.gif" % (floor_name, seq_i),
        #         cond_len=0,
        #         cond_edges=cond_edges,
        #         seq_gt=seq_gt,
        #         seq_heuristic=seq_heuristic,
        #         seq_class=seq_class,
        #         seq_metric=seq_metric,
        #         tcn=tcn,
        #     )

    # compute FID scores
    hiddens_gt = np.array(hiddens_gt)
    hiddens_class = np.array(hiddens_class)
    hiddens_metric = np.array(hiddens_metric)
    hiddens_heuristic = np.array(hiddens_heuristic)

    fid_gt = calculate_fid(hiddens_gt, hiddens_gt)
    fid_class = calculate_fid(hiddens_gt, hiddens_class)
    fid_metric = calculate_fid(hiddens_gt, hiddens_metric)
    fid_heuristic = calculate_fid(hiddens_gt, hiddens_heuristic)

    print("FID GT       : %.3f" % fid_gt)
    print("FID heuristic: %.3f" % fid_heuristic)
    print("FID class    : %.3f" % fid_class)
    print("FID metric   : %.3f" % fid_metric)

    with open("fid_csv/fid_%s_%02d.csv" % (edge_type, floor_idx), "a") as f:
        f.write(
            "%d,%d,%.3f,%.3f,%.3f,%.3f\n"
            % (floor_idx, target_len, fid_gt, fid_heuristic, fid_class, fid_metric)
        )


def qual_eval(floor_idx, target_len):
    # load floor data
    class_dataset = BuildingCornerDataset(
        DATA_PATH,
        None,
        phase="test",
        rand_aug=False,
        test_idx=floor_idx,
        loss_type="class",
    )
    metric_dataset = BuildingCornerDataset(
        DATA_PATH,
        None,
        phase="test",
        rand_aug=False,
        test_idx=floor_idx,
        loss_type="metric",
    )

    floor_name = list(class_dataset.density_fulls.keys())[0]
    density_full = class_dataset.density_fulls[floor_name]
    gt_edges = class_dataset.ordered_edges[floor_name]
    heat_edges = class_dataset.heat_edges[floor_name]

    # load classifier model
    class_model = ClassifierModel(d_model=256)

    class_ckpt_f = "%s/%d/checkpoint_latest.pth" % (CLASS_DIR, floor_idx)
    class_ckpt = torch.load(class_ckpt_f)
    class_model.load_state_dict(class_ckpt["edge_model"])
    class_model = class_model.cuda()
    class_model.eval()

    # load metric model
    metric_model = MetricModel(d_model=256)

    metric_ckpt_f = "%s/%d/checkpoint_latest.pth" % (METRIC_DIR, floor_idx)
    metric_ckpt = torch.load(metric_ckpt_f)
    metric_model.load_state_dict(metric_ckpt["edge_model"])
    metric_model = metric_model.cuda()
    metric_model.eval()

    print("Getting GT sequences")

    for start_idx in tqdm(range(5, len(gt_edges) - target_len)):
        # we consider the past 5 GT edges as condition
        curr_seq = gt_edges[start_idx - 5 : start_idx]
        seq_gt = gt_edges[start_idx : start_idx + target_len]

        # we find candidate edges
        cand_edges = my_utils.find_candidates(curr_seq, heat_edges)

        # color_coords = [
        #     ["-vg", curr_seq],
        #     ["-^y", heat_edges],
        # ]
        # my_utils.vis_edges(density_full, color_coords)

        # color_coords = [
        #     ["-vg", curr_seq],
        #     ["-^y", cand_edges],
        # ]
        # my_utils.vis_edges(density_full, color_coords)

        # for current sequence, we include up to 10 edges
        if len(curr_seq) >= 10:
            cond_edges = curr_seq[:-9]
            curr_seq = curr_seq[-9:]
        else:
            cond_edges = curr_seq[:0]

        seq_heuristic = get_seq_heuristic(
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=len(curr_seq) + target_len,
        )

        seq_class = get_seq_classifier(
            model=class_model,
            cond_edges=cond_edges,
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=len(curr_seq) + target_len,
        )

        seq_metric, _ = get_seq_metric(
            model=metric_model,
            cond_edges=cond_edges,
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=len(curr_seq) + target_len,
        )

        seq_heuristic = seq_heuristic[len(curr_seq) :]
        seq_class = seq_class[len(curr_seq) :]
        seq_metric = seq_metric[len(curr_seq) :]
        assert len(seq_heuristic) == len(seq_gt)

        # find examples where the metric framework outputs are better
        def find_dist(seq_a, seq_b):
            return np.sum(
                [my_utils.line_dist(ab, cd) for (ab, cd) in zip(seq_a, seq_b)]
            )

        gt2heuristic = find_dist(seq_gt, seq_heuristic)
        gt2class = find_dist(seq_gt, seq_class)
        gt2metric = find_dist(seq_gt, seq_metric)

        # if ((gt2metric + 200) < gt2class) and ((gt2metric + 200) < gt2heuristic):
        if True:
            save_f = "./vis/%03d_%03d.png" % (floor_idx, start_idx)
            # compare_seq_static(
            #     density_full,
            #     cond_edges,
            #     curr_seq,
            #     seq_gt,
            #     seq_heuristic,
            #     seq_class,
            #     seq_metric,
            #     [gt2heuristic, gt2class, gt2metric],
            #     save_f,
            # )

            compare_seq_failure(
                density_full,
                cond_edges,
                curr_seq,
                seq_gt,
                seq_metric,
                gt_edges,
                cand_edges,
                save_f,
            )

            save_dict = {
                "floor_idx": floor_idx,
                "cond_edges": cond_edges,
                "curr_seq": curr_seq,
                "seq_gt": seq_gt,
                "seq_heuristic": seq_heuristic,
                "seq_class": seq_class,
                "seq_metric": seq_metric,
                "scores": [gt2heuristic, gt2class, gt2metric],
                "gt_edges": gt_edges,
                "cand_edges": cand_edges,
            }
            np.save("./npy/%03d_%03d.npy" % (floor_idx, start_idx), save_dict)


def eval_acc_wrt_history(floor_idx, cond_len, target_len):
    # load floor data
    class_dataset = BuildingCornerDataset(
        DATA_PATH,
        None,
        phase="test",
        rand_aug=False,
        test_idx=floor_idx,
        loss_type="class",
    )

    floor_name = list(class_dataset.density_fulls.keys())[0]
    density_full = class_dataset.density_fulls[floor_name]
    gt_edges = class_dataset.ordered_edges[floor_name]

    # load classifier model
    class_model = ClassifierModel(d_model=256)

    class_ckpt_f = "%s/%d/checkpoint_latest.pth" % (CLASS_DIR, floor_idx)
    class_ckpt = torch.load(class_ckpt_f)
    class_model.load_state_dict(class_ckpt["edge_model"])
    class_model = class_model.cuda()
    class_model.eval()

    # load metric model
    metric_model = MetricModel(d_model=256)

    metric_ckpt_f = "%s/%d/checkpoint_latest.pth" % (METRIC_DIR, floor_idx)
    metric_ckpt = torch.load(metric_ckpt_f)
    metric_model.load_state_dict(metric_ckpt["edge_model"])
    metric_model = metric_model.cuda()
    metric_model.eval()

    # need to consider different provided history length
    # cond_len = 3

    # seqs = []
    # for start_idx in range(0, len(gt_edges) - target_len):
    #     end_idx = start_idx + target_len
    #     seqs.append((start_idx, end_idx))
    # np.random.shuffle(seqs)

    print("Getting predicted sequences")

    all_acc = []
    for start_idx in tqdm(range(cond_len, len(gt_edges) - target_len)):
        end_idx = start_idx + target_len - cond_len

        # obtain features for GT sequence
        seq_gt = gt_edges[start_idx:end_idx]

        # obtain features for predicted sequence
        cond_edges = gt_edges[:0]

        used_mask = np.zeros(len(gt_edges), dtype=bool)
        used_mask[start_idx - cond_len : start_idx] = True

        curr_seq = gt_edges[start_idx - cond_len : start_idx]
        cand_edges = gt_edges[~used_mask]

        seq_metric, _ = get_seq_metric(
            model=metric_model,
            cond_edges=cond_edges,
            curr_seq=curr_seq,
            cand_edges=cand_edges,
            target_len=target_len,
        )

        # remove the edge conditions
        seq_metric = seq_metric[cond_len:]

        # see how many of them we got right
        for coords in seq_metric:
            if (coords == seq_gt).all(axis=1).any():
                all_acc.append(1)
            else:
                all_acc.append(0)

    # compute accuracy
    all_acc = 100 * np.mean(all_acc)
    print("Acc: %.3f" % all_acc)

    with open("acc_history.csv", "a") as f:
        f.write("%d,%d,%d,%.3f\n" % (floor_idx, cond_len, target_len, all_acc))


def eval_all_floors():
    for target_len in range(1, 11):
        for floor_idx in range(16):
            eval_floor(floor_idx, target_len)


def eval_floor_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_idx", type=int, default=0)
    parser.add_argument("--edge_type", type=str, default="")
    args = parser.parse_args()

    # csv_f = "fid_csv/fid_%s_%02d.csv" % (args.edge_type, args.test_idx)
    # with open(csv_f, "r") as f:
    #     lines = f.readlines()

    for target_len in range(2, 11):
        # if target_len <= len(lines):
        #     continue
        eval_floor(args.test_idx, target_len, args.edge_type)


def eval_all_acc_wrt_history():
    for cond_len in range(1, 10 + 1):
        for floor_idx in range(16):
            eval_acc_wrt_history(floor_idx, cond_len, cond_len + 5)


def set_font_sizes():
    # sets font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_seq_FID():
    # for FID scores on GT edges
    scores_gt_h = [[] for _ in range(9)]
    scores_gt_c = [[] for _ in range(9)]
    scores_gt_m = [[] for _ in range(9)]

    for floor_idx in range(16):
        with open("fid_csv/fid_GT_%02d.csv" % floor_idx, "r") as f:
            for line in f:
                tokens = line.strip().split(",")

                seq_len = int(tokens[1])
                # score_h = float(tokens[3])
                score_c = float(tokens[4])
                score_m = float(tokens[5])

                # scores_gt_h[seq_len - 2].append(score_h)
                scores_gt_c[seq_len - 2].append(score_c)
                scores_gt_m[seq_len - 2].append(score_m)

    for floor_idx in range(16):
        with open("fid_csv/good_GT/fid_GT_%02d.csv" % floor_idx, "r") as f:
            f.readline()  # ignore the first 1-len one

            for line in f:
                tokens = line.strip().split(",")

                seq_len = int(tokens[1])
                score_h = float(tokens[3])
                scores_gt_h[seq_len - 2].append(score_h)

    assert np.array(scores_gt_h).shape[1] == 16
    assert np.array(scores_gt_c).shape[1] == 16
    assert np.array(scores_gt_m).shape[1] == 16

    scores_gt_h = np.array(scores_gt_h).mean(axis=1)
    scores_gt_c = np.array(scores_gt_c).mean(axis=1)
    scores_gt_m = np.array(scores_gt_m).mean(axis=1)

    # for FID scores on HEAT edges
    scores_heat_h = [[] for _ in range(9)]
    scores_heat_c = [[] for _ in range(9)]
    scores_heat_m = [[] for _ in range(9)]

    for floor_idx in range(16):
        with open("fid_csv/fid_HEAT_%02d.csv" % floor_idx, "r") as f:
            for line in f:
                tokens = line.strip().split(",")

                seq_len = int(tokens[1])
                score_h = float(tokens[3])
                score_c = float(tokens[4])
                score_m = float(tokens[5])

                scores_heat_h[seq_len - 2].append(score_h)
                scores_heat_c[seq_len - 2].append(score_c)
                scores_heat_m[seq_len - 2].append(score_m)

    assert np.array(scores_heat_h).shape[1] == 16
    assert np.array(scores_heat_c).shape[1] == 16
    assert np.array(scores_heat_m).shape[1] == 16

    scores_heat_h = np.array(scores_heat_h).mean(axis=1)
    scores_heat_c = np.array(scores_heat_c).mean(axis=1)
    scores_heat_m = np.array(scores_heat_m).mean(axis=1)

    fig, [ax1, ax2] = plt.subplots(ncols=2)

    YELLOW = "#FFD700"
    GREEN = "#32CD32"
    BLUE = "#4682B4"

    # GREEN = "#00FF7F"
    # BLUE = "#00FFFF"

    ax1.plot(range(2, 11), scores_gt_h, "-o", color=YELLOW)
    ax1.plot(range(2, 11), scores_gt_c, "-o", color=BLUE)
    ax1.plot(range(2, 11), scores_gt_m, "-o", color=GREEN)

    ax2.plot(range(2, 11), scores_heat_h, "-o", color=YELLOW)
    ax2.plot(range(2, 11), scores_heat_c, "-o", color=BLUE)
    ax2.plot(range(2, 11), scores_heat_m, "-o", color=GREEN)

    ax1.set_title("Predicted sequences with GT walls")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("FID score")
    ax2.set_title("Predicted sequences with predicted walls")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("FID score")

    handles = [
        Patch(label="Baseline (heuristic)", facecolor=YELLOW),
        Patch(label="Baseline (classifier)", facecolor=BLUE),
        Patch(label="Ours", facecolor=GREEN),
    ]
    ax1.legend(handles=handles)
    ax2.legend(handles=handles)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_history_acc():
    scores = [[] for _ in range(10)]

    with open("acc_history.csv", "r") as f:
        for line in f:
            tokens = line.strip().split(",")

            floor_idx = int(tokens[0])
            hist_len = int(tokens[1])
            score = float(tokens[3])

            scores[hist_len - 1].append(score)

    scores = np.array(scores).mean(axis=1)

    with open("acc.csv", "w") as f:
        f.write(",".join(["%.3f" % x for x in scores.tolist()]) + "\n")

    plt.plot(scores[:-1], "-o")

    plt.xticks(range(9), range(1, 10))
    plt.xlabel("History length")
    plt.ylabel("Accuracy")
    # plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def eval_entropy():
    target_len = 10

    all_Hs = []
    for floor_idx in range(16):
        # load floor data
        class_dataset = BuildingCornerDataset(
            DATA_PATH,
            None,
            phase="test",
            rand_aug=False,
            test_idx=floor_idx,
            loss_type="class",
        )

        floor_name = list(class_dataset.density_fulls.keys())[0]
        density_full = class_dataset.density_fulls[floor_name]
        gt_edges = class_dataset.ordered_edges[floor_name]

        # load metric model
        metric_model = MetricModel(d_model=256)

        metric_ckpt_f = "%s/%d/checkpoint_latest.pth" % (METRIC_DIR, floor_idx)
        metric_ckpt = torch.load(metric_ckpt_f)
        metric_model.load_state_dict(metric_ckpt["edge_model"])
        metric_model = metric_model.cuda()
        metric_model.eval()

        src_edges = gt_edges

        for tgt_idx in tqdm(range(target_len, len(src_edges))):
            Hs = []
            all_probs = []

            for cond_len in range(1, target_len):
                start_idx = tgt_idx - cond_len
                end_idx = start_idx + cond_len
                assert end_idx == tgt_idx

                cond_edges = gt_edges[:0]
                curr_seq = src_edges[start_idx:end_idx]
                cand_edges = src_edges[end_idx:]

                seq_metric, seq_probs = get_seq_metric(
                    model=metric_model,
                    cond_edges=cond_edges,
                    curr_seq=curr_seq,
                    cand_edges=cand_edges,
                    target_len=len(curr_seq) + 1,
                )

                assert len(seq_probs) == 1
                all_probs.append(seq_probs[0])
                H = entropy(seq_probs[0])
                Hs.append(H)

            all_Hs.append(Hs)
            continue

            for cond_len, probs in enumerate(all_probs):
                plt.plot(probs, "-", label=cond_len)
            plt.tight_layout()
            plt.legend()
            plt.show()
            plt.close()

            plt.plot(Hs, "-o")
            plt.show()
            plt.close()

    all_Hs = np.array(all_Hs).mean(axis=0)

    with open("entropy.csv", "w") as f:
        f.write(",".join(["%.3f" % x for x in all_Hs.tolist()]) + "\n")

    plt.plot(all_Hs, "-o")
    plt.xlabel("Number of history edges")
    plt.ylabel("Entropy")
    plt.xticks(range(9), range(1, 10))
    plt.tight_layout()
    plt.show()
    plt.close()


def all_qual_eval():
    for floor_idx in range(16):
        qual_eval(floor_idx, 5)


def make_qual_fig():
    pairs = [
        # "002_119",
        # "000_017",
        # "000_081",
        # "004_038",
        "000_089",
        "001_060",
        "002_064",
        "003_040",
    ]

    for pair in pairs:
        (floor_idx, start_idx) = [int(x) for x in pair.split("_")]

        class_dataset = BuildingCornerDataset(
            DATA_PATH,
            None,
            phase="test",
            rand_aug=False,
            test_idx=floor_idx,
            loss_type="class",
        )
        floor_name = list(class_dataset.density_fulls.keys())[0]
        density_full = class_dataset.density_fulls[floor_name]

        save_dict = np.load(
            "./npy/%03d_%03d.npy" % (floor_idx, start_idx), allow_pickle=True
        ).item()

        # compare_seq_fig(
        #     density_full=density_full,
        #     cond_edges=save_dict["cond_edges"],
        #     curr_seq=save_dict["curr_seq"],
        #     seq_gt=save_dict["seq_gt"],
        #     seq_heuristic=save_dict["seq_heuristic"],
        #     seq_class=save_dict["seq_class"],
        #     seq_metric=save_dict["seq_metric"],
        #     scores=save_dict["scores"],
        # )

        compare_seq_failure(
            density_full,
            cond_edges=save_dict["cond_edges"],
            curr_seq=save_dict["curr_seq"],
            seq_gt=save_dict["seq_gt"],
            seq_metric=save_dict["seq_metric"],
            gt_edges=save_dict["gt_edges"],
            pred_edges=save_dict["cand_edges"],
        )

        compare_seq_static(
            density_full=density_full,
            cond_edges=save_dict["cond_edges"],
            curr_seq=save_dict["curr_seq"],
            seq_gt=save_dict["seq_gt"],
            seq_heuristic=save_dict["seq_heuristic"],
            seq_class=save_dict["seq_class"],
            seq_metric=save_dict["seq_metric"],
            scores=save_dict["scores"],
            show_titles=False,
            show_steps=True,
        )


if __name__ == "__main__":
    set_font_sizes()

    # eval_all_floors()
    # eval_floor_worker()
    # eval_entropy()
    # eval_all_acc_wrt_history()
    # plot_seq_FID()
    # plot_history_acc()
    # all_qual_eval()
    make_qual_fig()
