import argparse
import json
import os
import time
from socket import *
from struct import unpack

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import shapely
import torch
import torch.nn.functional as F
from models.corner_models import CornerEnum
from models.edge_full_models import EdgeEnum
from models.order_metric_models import EdgeTransformer as MetricModel
from models.unet import ResNetBackbone
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import entropy
from shapely import affinity
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
)
from tqdm import tqdm

import my_utils
from datasets.building_corners_full import collate_fn_seq, get_pixel_features
from timer import Timer
import typer

app = typer.Typer()

# for 512 dual big dataset
density_mean = [0.18115416, 0.18115416, 0.18115416]
density_std = [0.27998772, 0.27998772, 0.27998772]

markers = ["v", "^", "<", ">"]
colors = [
    "lime",
    "red",
    "turquoise",
    "hotpink",
    "cyan",
    "orange",
    "darkcyan",
    "yellow",
    "steelblue",
    "lightcoral",
    "skyblue",
]


class Backend:
    def __init__(self, deform_type="DETR_dense", num_samples=16):
        self.data_path = "../../data"
        # self.corner_model_path = "../../ckpts/05_02_corners"
        # self.metric_model_path = "../../ckpts/04_17_order_metric_working"
        self.corner_model_path = "../../ckpts/corner"
        self.metric_model_path = "../../ckpts/order_metric"

        # GT matching hyperparameters
        self.match_threshold = 30

        # edge model hyperparameters
        self.deform_type = deform_type
        self.num_samples = num_samples
        self.pool_type = "max"
        self.revectorize = True

        if deform_type == "DETR_dense":
            if num_samples == 8:
                # self.edge_model_path = (
                #     "../../ckpts/05_07_edges_width_aug_new_4_loss_sample_8"
                # )
                self.edge_model_path = "../../ckpts/edge_sample_8"
            elif num_samples == 16:
                # self.edge_model_path = "../../ckpts/05_07_edges_width_aug_new_4_loss"
                self.edge_model_path = "../../ckpts/edge_sample_16"
            else:
                raise Exception

        elif deform_type == "DETR":
            # self.edge_model_path = "../../ckpts/05_07_edges_width_aug_new_4_loss_DETR"
            self.edge_model_path = "../../ckpts/edge_sample_DETR"

        else:
            raise Exception

    def init_corner_models(self, floor_idx):
        backbone = ResNetBackbone()
        strides = backbone.strides
        num_channels = backbone.num_channels
        backbone = backbone.cuda()
        backbone.eval()

        corner_model = CornerEnum(
            input_dim=128,
            hidden_dim=256,
            num_feature_levels=4,
            backbone_strides=strides,
            backbone_num_channels=num_channels,
        )
        corner_model = corner_model.cuda()
        corner_model.eval()

        ckpt_path = "%s/%d/checkpoint.pth" % (self.corner_model_path, floor_idx)
        ckpt = torch.load(ckpt_path)

        backbone_ckpt = {}
        for key, value in ckpt["backbone"].items():
            key = key.replace("module.", "")
            backbone_ckpt[key] = value
        backbone.load_state_dict(backbone_ckpt)

        corner_model_ckpt = {}
        for key, value in ckpt["corner_model"].items():
            key = key.replace("module.", "")
            corner_model_ckpt[key] = value
        corner_model.load_state_dict(corner_model_ckpt)

        self.corner_backbone = backbone
        self.corner_model = corner_model

        print("Loaded corner models")

    def init_edge_models(self, floor_idx):
        start = time.time()

        backbone = ResNetBackbone()
        strides = backbone.strides
        num_channels = backbone.num_channels
        backbone = backbone.cuda()
        backbone.eval()

        edge_model = EdgeEnum(
            input_dim=128,
            hidden_dim=256,
            num_feature_levels=4,
            backbone_strides=strides,
            backbone_num_channels=num_channels,
            deform_type=self.deform_type,
            num_samples=self.num_samples,
            pool_type=self.pool_type,
        )
        edge_model = edge_model.cuda()
        edge_model.eval()

        ckpt_path = "%s/%d/checkpoint.pth" % (self.edge_model_path, floor_idx)
        ckpt = torch.load(ckpt_path)
        print("Edge ckpt path: %d from %s" % (ckpt["epoch"], ckpt_path))
        backbone.load_state_dict(ckpt["backbone"])
        edge_model.load_state_dict(ckpt["edge_model"])

        self.edge_backbone = backbone
        self.edge_model = edge_model

        end = time.time()

        print("Loaded edge models (%.3f seconds)" % (end - start))

    def init_metric_model(self, floor_idx):
        start = time.time()

        metric_model = MetricModel(d_model=256)
        metric_model = metric_model.cuda()
        metric_model.eval()

        ckpt_path = "%s/%d/checkpoint_latest.pth" % (self.metric_model_path, floor_idx)
        ckpt = torch.load(ckpt_path)
        metric_model.load_state_dict(ckpt["edge_model"])

        self.metric_model = metric_model

        end = time.time()

        print("Loaded metric models (%.3f seconds)" % (end - start))

    def init_floor(self, floor_idx):
        start = time.time()

        floor_f = os.path.join(self.data_path, "all_floors.txt")
        with open(floor_f, "r") as f:
            floor_names = [x.strip() for x in f.readlines()]
        self.floor_name = floor_names[floor_idx]

        with open("%s/bounds/%s.csv" % (self.data_path, self.floor_name), "r") as f:
            self.bounds = [float(x) for x in f.readline().strip().split(",")]

        # load full density image
        density_slices = []
        for slice_i in range(7):
            slice_f = "%s/density/%s/density_%02d.npy" % (
                self.data_path,
                self.floor_name,
                slice_i,
            )
            density_slice = np.load(slice_f)
            density_slices.append(density_slice)

        density_full = [
            my_utils.normalize_density(np.sum(density_slices[:4], axis=0)),
            my_utils.normalize_density(density_slices[4]),
            my_utils.normalize_density(np.sum(density_slices[5:7], axis=0)),
        ]
        density_full = np.stack(density_full, axis=2)

        # we need to square pad the image
        (h, w, _) = density_full.shape
        side_len = max(h, w)

        pad_h_before = (side_len - h) // 2
        pad_h_after = side_len - h - pad_h_before
        pad_w_before = (side_len - w) // 2
        pad_w_after = side_len - w - pad_w_before
        self.square_pad = [pad_w_before, pad_h_before, pad_w_after, pad_h_after]

        density_full = np.pad(
            density_full,
            [[pad_h_before, pad_h_after], [pad_w_before, pad_w_after], [0, 0]],
        )
        self.density_full = density_full

        # load GT annotation
        annot_f = os.path.join(self.data_path, "annot/%s.json" % self.floor_name)
        with open(annot_f, "r") as f:
            annot = json.load(f)

        annot = np.array(list(annot.values()))
        gt_coords, gt_widths = annot[:, :4], annot[:, 4]

        gt_coords += [pad_w_before, pad_h_before, pad_w_before, pad_h_before]
        gt_widths = np.floor(gt_widths * 12).astype(int)

        if self.revectorize:
            gt_coords, gt_widths = my_utils.revectorize(gt_coords, gt_widths)
        gt_corners, gt_edge_ids = my_utils.corners_and_edges(gt_coords)

        self.gt_coords = gt_coords
        self.gt_corners = gt_corners
        self.gt_widths = gt_widths

        # load predicted corners
        corner_f = os.path.join(
            self.data_path, "pred_corners/%s.json" % self.floor_name
        )

        if os.path.exists(corner_f):
            with open(corner_f, "r") as f:
                cached_corners = json.load(f)
            cached_corners = np.array(cached_corners)
            # cached_corners -= 128
            cached_corners += [pad_w_before, pad_h_before]
            self.cached_corners = cached_corners

            # # augment corners for the sake of testing
            # aug_corners = gt_corners.copy()

            # cost = distance.cdist(gt_corners, cached_corners)
            # gt_ind, cached_ind = linear_sum_assignment(cost)

            # valid_mask = cost[gt_ind, cached_ind] <= self.match_threshold
            # gt_ind = gt_ind[valid_mask]
            # cached_ind = cached_ind[valid_mask]
            # aug_corners[gt_ind] = cached_corners[cached_ind]
            # self.aug_corners = aug_corners

        else:
            self.init_corner_models(floor_idx)
            self.cached_corners = self.get_pred_corners()

            # save the corners so we don't have to do this again
            [pad_w_before, pad_h_before, pad_w_after, pad_h_after] = self.square_pad
            pred_corners = self.cached_corners.copy()
            pred_corners -= [pad_w_before, pad_h_before]

            save_f = "%s/pred_corners/%s.json" % (self.data_path, self.floor_name)
            os.makedirs(os.path.dirname(save_f), exist_ok=True)
            with open(save_f, "w") as f:
                json.dump(pred_corners.tolist(), f)

        end = time.time()
        print("Loaded floor data (%.3f seconds)" % (end - start))

    def cache_image_feats(self, corners):
        image = self.density_full.copy()

        image, corners, scale = my_utils.normalize_floor(image, corners)
        image = my_utils.process_image(image)

        data = {"img": image}
        data = collate_fn_seq([data])

        with torch.no_grad():
            image = data["img"].cuda()
            image_feats, feat_mask, _ = self.edge_backbone(image)

        self.norm_scale = scale
        self.image_feats = image_feats
        self.feat_mask = feat_mask

    def get_pred_corners(self):
        # hyperparameters
        corner_thresh = 0.5
        c_padding = 16
        side_len = 256 - c_padding * 2
        stride = side_len // 4
        ignore_border = 16

        # pad the image so we get predictions near boundary
        density_full = self.density_full.copy()

        # determine overlapping crops
        (h, w, _) = density_full.shape

        bboxes = []
        for i in range(int(w / stride) + 1):
            for j in range(int(h / stride) + 1):
                minx = i * stride
                miny = j * stride
                maxx = minx + side_len
                maxy = miny + side_len

                if maxy > h:
                    miny = h - side_len
                    maxy = h

                if maxx > w:
                    minx = w - side_len
                    maxx = w

                bbox = [minx, miny, maxx, maxy]
                if (not len(bboxes)) or (bboxes[-1] != bbox):
                    bboxes.append(bbox)

        # for each crop:
        # 1. run corner detector
        # 2. ignore predictions near the border
        # 3. paste onto a full floorplan result
        corner_full = np.zeros((h, w), dtype=np.float32)
        pixels, pixel_features = get_pixel_features(image_size=256)
        pixel_features = pixel_features.unsqueeze(0).cuda()

        print("Running corner detector on crops")
        for minx, miny, maxx, maxy in tqdm(bboxes):
            density_crop = density_full[miny:maxy, minx:maxx, :].copy()
            density_crop = np.pad(
                density_crop, [[c_padding, c_padding], [c_padding, c_padding], [0, 0]]
            )
            assert density_crop.shape == (256, 256, 3)
            density_crop_512 = density_crop.copy()

            # resize and preprocess
            # density_crop = cv2.cvtColor(density_crop, cv2.COLOR_GRAY2RGB)
            # density_crop = cv2.resize(density_crop, (256, 256))
            density_crop = density_crop.transpose((2, 0, 1))
            density_crop -= np.array(density_mean)[:, np.newaxis, np.newaxis]
            density_crop /= np.array(density_std)[:, np.newaxis, np.newaxis]
            density_crop = density_crop.astype(np.float32)

            # run corner network
            density_crop = torch.tensor(density_crop).unsqueeze(0).cuda()
            with torch.no_grad():
                image_feats, feat_mask, all_image_feats = self.corner_backbone(
                    density_crop
                )
                corner_crop = self.corner_model(
                    image_feats, feat_mask, pixel_features, pixels, all_image_feats
                )
            corner_crop = corner_crop[0].detach().cpu().numpy()
            corner_crop = corner_crop[c_padding:-c_padding, c_padding:-c_padding]

            # img = np.max(density_crop_512, axis=2)
            # plt.imshow(img, cmap="gray")
            # plt.imshow(corner_crop, cmap="hot", alpha=0.5)
            # plt.show()

            # resize and ignore predictions near border
            # corner_crop = cv2.resize(corner_crop, (512, 512))

            keep_mask = np.zeros_like(corner_crop, dtype=bool)
            keep_mask[ignore_border:-ignore_border, ignore_border:-ignore_border] = True
            corner_crop[~keep_mask] = 0

            # paste prediction in full view
            corner_full[miny:maxy, minx:maxx] = np.maximum(
                corner_full[miny:maxy, minx:maxx], corner_crop
            )

        # run NMS to obtain corners in this floorplan
        (height, width) = corner_full.shape
        pixels_x = np.arange(0, width)
        pixels_y = np.arange(0, height)

        xv, yv = np.meshgrid(pixels_x, pixels_y)
        all_pixels = list()
        for i in range(xv.shape[0]):
            pixs = np.stack([xv[i], yv[i]], axis=-1)
            all_pixels.append(pixs)
        pixels_full = np.stack(all_pixels, axis=0)

        pos_indices = np.where(corner_full >= corner_thresh)
        pred_corners = pixels_full[pos_indices]
        pred_confs = corner_full[pos_indices]
        pred_corners, pred_confs = my_utils.corner_nms(
            pred_corners, pred_confs, image_shape=corner_full.shape
        )

        # remove padding from pixel coordinates
        # density_full = density_full[pad_border:-pad_border, pad_border:-pad_border]
        # corner_full = corner_full[pad_border:-pad_border, pad_border:-pad_border]
        # pred_corners -= pad_border

        if False:
            plt.imshow(density_full[:, :, 1], cmap="gray")
            plt.imshow(corner_full, cmap="hot", alpha=0.7)
            plt.plot(pred_corners[:, 0], pred_corners[:, 1], "*c")
            # plt.plot(self.cached_corners[:, 0], self.cached_corners[:, 1], "*m")
            plt.show()
            plt.close()

        return pred_corners

    def get_pred_coords(
        self, pred_corners, branch="relation", threshold=0.5, postprocess=True
    ):
        image = self.density_full.copy()
        pred_corners_raw = pred_corners.copy()

        # _, pred_corners, _ = my_utils.normalize_floor(image, pred_corners)
        # image = my_utils.process_image(image)
        pred_corners, _ = my_utils.normalize_corners(pred_corners, self.norm_scale)

        all_edges = my_utils.all_combinations[len(pred_corners)]
        edge_coords = pred_corners[all_edges].reshape(-1, 4)

        data = {
            "floor_name": self.floor_name,
            # "img": image,
            "edge_coords": edge_coords,
            "processed_corners_lengths": len(pred_corners),
        }
        data = collate_fn_seq([data])

        edge_coords = data["edge_coords"].cuda()
        edge_mask = data["edge_coords_mask"].cuda()
        blank_labels = torch.full_like(edge_mask, fill_value=2, dtype=torch.long)

        corner_nums = data["processed_corners_lengths"]
        max_candidates = torch.stack([corner_nums.max() * 3] * len(corner_nums), dim=0)

        # network inference
        with torch.no_grad():
            # image = data["img"].cuda()
            # image_feats, feat_mask, _ = self.edge_backbone(image)

            (
                logits_s1,
                logits_edge_hb,
                logits_edge_rel,
                logits_width_hb,
                logits_width_rel,
                s2_ids,
                s2_edge_mask,
                s2_gt_values,
                ref_dict,
            ) = self.edge_model(
                self.image_feats,
                self.feat_mask,
                edge_coords,
                edge_mask,
                blank_labels,
                corner_nums,
                max_candidates,
                mask_gt=False,
            )

        # my_utils.vis_ref(image, edge_coords, ref_dict, s2_ids)

        if branch == "relation":
            s2_edges = all_edges[s2_ids[0].cpu().numpy()]
            edge_probs = logits_edge_rel.softmax(1)[0, 1, :].cpu().numpy()
            pred_edges = s2_edges[edge_probs >= threshold]
            # edge_preds = logits_edge_rel.argmax(1)[0].cpu().numpy()
            # pred_edges = s2_edges[edge_preds == 1]
            pred_coords = pred_corners_raw[pred_edges].reshape(-1, 4)

            pred_widths = logits_width_rel.argmax(1)[0].cpu().numpy()
            pred_widths = pred_widths[edge_probs >= threshold]
            # pred_widths = pred_widths[edge_preds == 1]

        elif branch == "hybrid":
            s2_edges = all_edges[s2_ids[0].cpu().numpy()]
            edge_probs = logits_edge_hb.softmax(1)[0, 1, :].cpu().numpy()
            pred_edges = s2_edges[edge_probs >= threshold]
            # edge_preds = logits_edge_hb.argmax(1)[0].cpu().numpy()
            # pred_edges = s2_edges[edge_preds == 1]
            pred_coords = pred_corners_raw[pred_edges].reshape(-1, 4)

            pred_widths = logits_width_hb.argmax(1)[0].cpu().numpy()
            pred_widths = pred_widths[edge_probs >= threshold]
            # pred_widths = pred_widths[edge_preds == 1]

        elif branch == "image":
            raise Exception("Nope")
            edge_probs = logits_s1.softmax(1)[0, 1, :].cpu().numpy()
            pred_edges = all_edges[edge_probs >= threshold]
            pred_coords = pred_corners_raw[pred_edges].reshape(-1, 4)

        else:
            raise Exception("Unknown branch")

        # post-processing to:
        # 1. merge overlapping lines
        # 2. snap almost horizontal or vertical lines
        if postprocess:
            pred_coords, pred_widths = my_utils.remove_overlaps(
                pred_coords, pred_widths
            )
            # pred_coords = my_utils.merge_edges(pred_coords)
            pred_coords = my_utils.snap_to_axis(pred_coords)

        # visualize prediction
        # color_coords = [["-v", pred_coords]]
        # my_utils.vis_edges(self.density_full, color_coords)
        # my_utils.vis_edges_idx(self.density_full, pred_coords)

        return pred_coords, pred_edges, pred_widths

    def run_edge_model(self, data, branch="image"):
        edge_coords = data["edge_coords"].cuda()
        edge_mask = data["edge_coords_mask"].cuda()
        edge_labels = data["edge_labels"].cuda()

        corner_nums = data["processed_corners_lengths"]
        # max_candidates = torch.stack([corner_nums.max() * 3] * len(corner_nums), dim=0)
        max_candidates = 20

        # network inference
        with torch.no_grad():
            # image = data["img"].cuda()
            # image_feats, feat_mask, _ = self.edge_backbone(image)

            (
                logits_s1,
                logits_s2_hb,
                logits_s2_rel,
                logits_width_hb,
                logits_width_rel,
                s2_ids,
                s2_edge_mask,
                s2_gt_values,
                ref_pts,
            ) = self.edge_model(
                self.image_feats,
                self.feat_mask,
                edge_coords,
                edge_mask,
                edge_labels,
                corner_nums,
                max_candidates,
                do_inference=True,
            )

        # per-edge classifier will serve as the base prediction
        assert branch != "image"
        probs = logits_s1[0].softmax(0)[1].detach().cpu().numpy()
        probs = np.zeros_like(probs)

        # and other filtered edges are given higher priority
        if branch == "hybrid":
            filtered_logits = logits_s2_hb[0].softmax(0)[1].detach().cpu().numpy()
            filtered_ids = s2_ids[0].cpu().numpy()
            probs[filtered_ids] = filtered_logits

        elif branch == "relation":
            filtered_logits = logits_s2_rel[0].softmax(0)[1].detach().cpu().numpy()
            filtered_ids = s2_ids[0].cpu().numpy()
            probs[filtered_ids] = filtered_logits

        else:
            raise Exception("Unknown branch")

        assert (0 <= probs).all() and (probs <= 1).all()
        return probs

    def run_metric_model(self, edge_coords, edge_order):
        _edge_coords = my_utils.normalize_edges(edge_coords)

        # run metric network
        example = {"edge_coords": _edge_coords, "edge_order": edge_order}
        data = my_utils.metric_collate_fn([example])

        for key in data.keys():
            if type(data[key]) is torch.Tensor:
                data[key] = data[key].cuda()

        embeddings = self.metric_model(data)[0]

        # compute distance from query edge to candidates
        query_mask = edge_order == 1
        assert query_mask.sum() == 1
        query_i = query_mask.argmax()

        assert edge_order[query_i] == 1
        dist_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        dists = dist_fn(embeddings[query_i : query_i + 1], embeddings)

        # prevent from using existing edges
        used_mask = edge_order > 0
        dists[used_mask] = float("inf")

        return dists

    # 1. simulate a user adding a corner/edge
    # 2. run inference to obtain proposals
    # 3. keep correct ones, and reset incorrect ones
    def fix_auto(self, branch="relation", pos_threshold=0.5):
        pred_corners = self.cached_corners.copy()
        pred_coords, pred_edges = self.get_pred_coords(pred_corners)
        gt_coords = self.gt_coords.copy()
        _, gt_mask = self.get_fixes(pred_coords)

        step_i = 0
        all_step_stats = []

        while (gt_mask != 1).any():
            # user takes an action, for now it's just adding corners
            missing_gt_ids = (gt_mask != 1).nonzero()[0]
            missing_gt_id = np.random.choice(missing_gt_ids)
            missing_gt_coord = gt_coords[missing_gt_id]

            # enumerate candidate edges
            new_corners = missing_gt_coord.reshape(2, 2)
            all_corners = np.concatenate([new_corners, pred_corners], axis=0)

            cand_edges = np.stack(
                [
                    np.zeros(len(all_corners) - 1, dtype=int),
                    np.arange(1, len(all_corners), dtype=int),
                ],
                axis=1,
            )
            cand_coords = all_corners[cand_edges].reshape(-1, 4)
            assert (cand_coords[0] == missing_gt_coord).all()

            # combine them, but note that we appended two new corners
            all_edges = np.concatenate([pred_edges + 2, cand_edges], axis=0)

            # mark the edges we want to query, all others are "GT"
            edge_labels = np.ones(len(all_edges), dtype=int)
            edge_labels[len(pred_edges) :] = 2

            # network inference
            image = self.density_full.copy()
            _image, _all_corners = my_utils.normalize_floor(image, all_corners)
            _image = my_utils.process_image(_image)

            data = {
                "img": _image,
                "edge_coords": _all_corners[all_edges].reshape(-1, 4),
                "edge_labels": edge_labels,
                "processed_corners_lengths": len(_all_corners),
            }
            data = collate_fn_seq([data])
            probs = self.run_edge_model(data, branch=branch)
            probs[: len(pred_edges)] = 0  # ignore condition edges

            # check all predicted edges
            pos_edges = all_edges[probs > pos_threshold]
            pos_coords = all_corners[pos_edges].reshape(-1, 4)
            pos_label, pos2gt = my_utils.compute_label(
                pred_coords=pos_coords,
                gt_coords=gt_coords,
                threshold=self.match_threshold,
            )

            # collect stats on the fixes
            step_stats = {
                "fixed_target": False,
                "steps_auto": 0,
                "steps_bad": 0,
            }

            for pos_i, gt_i in pos2gt:
                # this means we recovered the intended edge
                if gt_i == missing_gt_id:
                    step_stats["fixed_target"] = True

                # this means we fixed an additional edge
                elif gt_mask[gt_i] != 1:
                    step_stats["steps_auto"] += 1

                # this means a previously good edge was reclassified
                else:
                    assert gt_mask[gt_i] == 1

            step_stats["steps_bad"] += (pos_label == 0).sum()
            all_step_stats.append(step_stats)

            # update with new corners and edges and coords
            pred_corners = all_corners
            pred_edges = all_edges[: len(pred_edges)]

            for pos_i, gt_i in pos2gt:
                if gt_mask[gt_i] != 1:
                    pred_edges = np.append(pred_edges, [pos_edges[pos_i]], axis=0)
                    gt_mask[gt_i] = 1

            old_coords = pred_coords.copy()
            pred_coords = pred_corners[pred_edges].reshape(-1, 4)

            # manually fix target edge if needed
            if not step_stats["fixed_target"]:
                assert (pred_corners[[0, 1]].reshape(-1, 4) == missing_gt_coord).all()
                pred_edges = np.append(pred_edges, [[0, 1]], axis=0)
                pred_coords = pred_corners[pred_edges].reshape(-1, 4)
                gt_mask[missing_gt_id] = 1

            if False:
                color_coords = []
                color_coords.append(["--y", old_coords])
                color_coords.append(["-or", pos_coords[pos_label == 0]])
                color_coords.append(["-og", pos_coords[pos_label == 1]])
                color_coords.append(["*m", self.cached_corners])
                color_coords.append(["*c", [missing_gt_coord]])

                save_f = "./vis/%s_%03d.png" % (self.floor_name, step_i)
                my_utils.vis_edges(self.density_full, color_coords, save_f=save_f)

            # print out the stats for this step
            print(
                "%3d | GT: %d / %d | Target: %s Auto: %d"
                % (
                    step_i,
                    (gt_mask == 1).sum(),
                    len(gt_mask),
                    step_stats["fixed_target"],
                    step_stats["steps_auto"],
                )
            )

            # increment counter
            step_i += 1

        # check results at the end
        labels, pred2gt = my_utils.compute_label(
            pred_coords=pred_coords, gt_coords=gt_coords, threshold=self.match_threshold
        )
        precision, recall = my_utils.compute_precision_recall(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            threshold=self.match_threshold,
        )
        assert recall == 100

        if False:
            fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

            image = self.density_full[:, :, 1]
            ax1.imshow(image, cmap="gray")
            ax2.imshow(image, cmap="gray")

            pred2gt = np.array(pred2gt)

            for pred_i, (x0, y0, x1, y1) in enumerate(pred_coords):
                if pred_i in pred2gt[:, 0]:
                    ax1.plot([x0, x1], [y0, y1], "-og", zorder=5)
                else:
                    ax1.plot([x0, x1], [y0, y1], "-or", zorder=10)

            for gt_i, (x0, y0, x1, y1) in enumerate(gt_coords):
                if gt_i in pred2gt[:, 1]:
                    ax2.plot([x0, x1], [y0, y1], "-og", zorder=5)
                else:
                    print(gt_i)
                    ax2.plot([x0, x1], [y0, y1], "-or", zorder=10)

            ax1.set_title("Fixed edges")
            ax2.set_title("GT edges")

            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.suptitle("P: %.3f R: %.3f" % (precision, recall))
            plt.tight_layout()
            plt.show()

        return all_step_stats

    # manually add corners and see where we are
    def fix_manual(self, branch="hybrid"):
        fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

        image = np.max(self.density_full, axis=2)
        ax1.imshow(0.5 * image, cmap="gray", vmin=0, vmax=1)
        ax2.imshow(0.5 * image, cmap="gray", vmin=0, vmax=1)
        # ax3.imshow(0.2 * image, cmap="gray", vmin=0, vmax=1)

        heat_coords, heat_edges = self.get_pred_coords(self.cached_corners)
        heat_corners = self.cached_corners[np.unique(heat_edges.flatten())]
        heat_labels, heat2gt = my_utils.compute_label(
            heat_coords, self.gt_coords, threshold=self.match_threshold
        )

        self.heat_lines = []
        for heat_i, (x0, y0, x1, y1) in enumerate(heat_coords):
            if heat_labels[heat_i] == 1:
                [heat_line] = ax1.plot([x0, x1], [y0, y1], "-og", picker=True)
            else:
                [heat_line] = ax1.plot([x0, x1], [y0, y1], "-or", picker=True)

            self.heat_lines.append(heat_line)

        # determine what GT edges and corners are left
        # gt_corners, _ = my_utils.corners_and_edges(self.gt_coords)
        # c_labels, _ = my_utils.compute_label(
        #     pred_coords=gt_corners,
        #     gt_coords=self.cached_corners,
        #     threshold=self.match_threshold,
        #     dist_fn="l2",
        # )

        # for (x0, y0) in gt_corners[c_labels == 0]:
        #     ax1.plot(x0, y0, "*c", picker=True, pickradius=3, markersize=7)

        # also plot GT edges
        self.gt_lines = []
        for gt_i, (x0, y0, x1, y1) in enumerate(self.gt_coords):
            # color = np.random.choice(colors)
            if gt_i in heat2gt[:, 1]:
                color = "g"
            else:
                color = "r"
            marker = np.random.choice(markers)
            [gt_line] = ax2.plot([x0, x1], [y0, y1], "-" + marker, color=color)
            self.gt_lines.append(gt_line)

        # allow the user to pick multiple corners
        self.curr_corners = heat_corners.copy()
        self.picked_corners = []
        self.deleted_coords = []

        def on_click(event):
            # right click adds corner
            if event.button == 3:
                # for some reason plotting corners reset zoom level
                # so we save it here and reset it afterwards
                xlim = ax1.get_xlim()
                ylim = ax1.get_ylim()

                # see if there are predicted corners there, and if so,
                # use that one instead
                x = event.xdata
                y = event.ydata
                dists = distance.cdist([[x, y]], self.curr_corners)[0]
                if dists.min() < self.match_threshold:
                    (x, y) = self.curr_corners[dists.argmin()]

                [corner] = ax1.plot(x, y, "*c", markersize=7)
                self.picked_corners.append(corner)

                ax1.set_xlim(xlim)
                ax1.set_ylim(ylim)
                fig.canvas.draw()

        # click on lines to delete them
        # also save the deleted coordinates so we don't add them again
        # also that we only delete incorrect lines
        def on_pick(event):
            if event.mouseevent.button != 1:
                return

            # check if this line is correct or not
            thisline = event.artist
            this_coords = thisline.get_xydata().reshape(-1, 4)
            label, _ = my_utils.compute_label(
                this_coords, self.gt_coords, threshold=self.match_threshold
            )
            if label[0] == 1:
                return

            self.deleted_coords.append(this_coords[0])
            self.heat_lines.remove(thisline)  # this also makes sure it contains it
            thisline.remove()

            # update current corners in the plot
            curr_coords = np.array([line.get_xydata() for line in self.heat_lines])
            curr_coords = curr_coords.reshape(-1, 2)
            self.curr_corners = np.unique(curr_coords, axis=0)

            fig.canvas.draw()

        def on_press(event):
            # only register space key
            if event.key not in [" ", "backspace"]:
                return

            # don't do anything if we haven't added any new corners
            if len(self.picked_corners) == 0:
                return

            # backspace removes the last corner added
            if event.key == "backspace":
                last_corner = self.picked_corners[-1]
                self.picked_corners.remove(last_corner)
                last_corner.remove()
                fig.canvas.draw()
                return

            # reset GT edge styles, we changed them to indicate ones to fix
            # for gt_line in self.gt_lines:
            #     gt_line.set_linestyle("-")
            #     gt_line.set_color(np.random.choice(colors))

            # turned picked corner objects into actual coordinates
            picked_corners = [corner.get_xydata()[0] for corner in self.picked_corners]
            picked_corners = np.array(picked_corners).round()

            # we sometimes pick corners that exist already, so remove them from
            # the other group we enumerate with
            curr_corners = self.curr_corners.copy()
            for picked_corner in picked_corners:
                ignore_mask = (picked_corner == curr_corners).all(axis=1)
                curr_corners = curr_corners[~ignore_mask]

            # using the picked corners, enumerate more edge candidates
            cand_corners = np.concatenate([picked_corners, curr_corners])

            combs = []
            for i in range(len(picked_corners)):
                for j in range(i + 1, len(cand_corners)):
                    combs.append((i, j))
            combs = np.array(combs)

            cand_coords = cand_corners[combs].reshape(-1, 4)

            # we also include the current edges as GT condition
            curr_coords = np.array([line.get_xydata() for line in self.heat_lines])
            curr_coords = curr_coords.reshape(-1, 4)

            # filter candidate edges to not include any existing ones
            repeat_mask, cand2curr = my_utils.compute_label(
                cand_coords, curr_coords, threshold=self.match_threshold
            )
            cand_coords = cand_coords[~repeat_mask.astype(bool)]

            # prepare edge condition flags
            edge_coords = np.concatenate([curr_coords, cand_coords], axis=0)
            edge_labels = np.ones(len(edge_coords), dtype=int)
            edge_labels[len(curr_coords) :] = 2

            # network inference
            image = self.density_full.copy()
            corners, edges = my_utils.corners_and_edges(edge_coords)

            image, corners = my_utils.normalize_floor(image, corners)
            image = my_utils.process_image(image)
            edge_coords = corners[edges].reshape(-1, 4)

            data = {
                "img": image,
                "edge_coords": edge_coords,
                "edge_labels": edge_labels,
                "processed_corners_lengths": len(corners),
            }
            data = collate_fn_seq([data])
            probs = self.run_edge_model(data, branch=branch)
            probs = probs[len(curr_coords) :]

            # we want to pick at least one edge for each new corner
            used_mask = np.zeros(len(picked_corners), dtype=bool)
            new_coords = []

            # find the first edge that contains an unused corner
            rank = probs.argsort()[::-1]
            for rank_i in rank:
                # ignore if we deleted this edge before
                if len(self.deleted_coords):
                    matches, _ = my_utils.compute_label(
                        pred_coords=[cand_coords[rank_i]],
                        gt_coords=np.array(self.deleted_coords),
                        threshold=self.match_threshold,
                    )
                    if matches[0] == 1:
                        continue

                # ignore if the corners are already used
                (ax, ay, bx, by) = cand_coords[rank_i]

                a_idx = my_utils.find_idx((ax, ay), picked_corners)
                b_idx = my_utils.find_idx((bx, by), picked_corners)

                assert (a_idx >= 0) or (b_idx >= 0)

                # if this is classified as a valid edge, add it
                if probs[rank_i] > 0.5:
                    new_coords.append(cand_coords[rank_i])

                    if a_idx >= 0:
                        used_mask[a_idx] = True
                    if b_idx >= 0:
                        used_mask[b_idx] = True

                # if it isn't, we add if one of the corners have not been used
                else:
                    if (not used_mask[a_idx]) or (not used_mask[b_idx]):
                        new_coords.append(cand_coords[rank_i])

                        if a_idx >= 0:
                            used_mask[a_idx] = True
                        if b_idx >= 0:
                            used_mask[b_idx] = True

            assert used_mask.all()
            new_coords = np.array(new_coords)

            # update HEAT plot with new edges
            new_labels, new2gt = my_utils.compute_label(
                new_coords, self.gt_coords, threshold=self.match_threshold
            )

            for new_i, (x0, y0, x1, y1) in enumerate(new_coords):
                if new_labels[new_i] == 1:
                    [new_line] = ax1.plot([x0, x1], [y0, y1], "-og", picker=True)
                else:
                    [new_line] = ax1.plot([x0, x1], [y0, y1], "-or", picker=True)

                self.heat_lines.append(new_line)

            # update GT plot with edges we have gotten right
            # _, cand2gt = my_utils.compute_label(
            #     pred_coords=cand_coords,
            #     gt_coords=self.gt_coords,
            #     threshold=self.match_threshold,
            #     dist_fn="line_dist",
            # )

            if len(new2gt):
                for gt_i in new2gt[:, 1]:
                    # self.gt_lines[gt_i].set_linestyle("-")
                    self.gt_lines[gt_i].set_color("g")

            # update the third plot with current predictions
            # for child in ax3.get_children():
            #     if type(child) == Line2D:
            #         child.remove()

            # cmap = plt.get_cmap("hot")

            # for cand_i, (x0, y0, x1, y1) in enumerate(cand_coords):
            #     color = cmap(probs[cand_i])

            #     if probs[cand_i] > 0.5:
            #         zorder = 10
            #     else:
            #         zorder = 5

            #     ax3.plot([x0, x1], [y0, y1], color=color, zorder=zorder)

            # remove all picked corners
            for corner in self.picked_corners:
                corner.remove()
            self.picked_corners = []

            # update current corners in the plot
            curr_coords = np.array([line.get_xydata() for line in self.heat_lines])
            curr_coords = curr_coords.reshape(-1, 2)
            self.curr_corners = np.unique(curr_coords, axis=0)

            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_press)

        ax1.set_axis_off()
        ax2.set_axis_off()
        # ax3.set_axis_off()

        ax1.set_title("HEAT")
        ax2.set_title("GT")
        # ax3.set_title("Pred")

        plt.tight_layout()
        plt.show()
        plt.close()

    def fig_helper(self):
        fig, ax1 = plt.subplots(ncols=1)

        image = np.max(self.density_full, axis=2)
        ax1.imshow(0.5 * image, cmap="gray", vmin=0, vmax=1)

        # also plot GT edges
        line2idx = {}
        for gt_i, (x0, y0, x1, y1) in enumerate(self.gt_coords):
            [gt_line] = ax1.plot([x0, x1], [y0, y1], "-g", picker=True)
            line2idx[gt_line] = gt_i

        picked_inds = []

        def on_pick(event):
            # check if this line is correct or not
            thisline = event.artist
            remove_i = line2idx[thisline]
            picked_inds.append(remove_i)
            del line2idx[thisline]
            thisline.remove()

            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)

        ax1.set_axis_off()
        plt.tight_layout()
        plt.show()

        print("")
        np.save("picked.npy", np.array(picked_inds))

        plt.imshow(0.5 * image, cmap="gray", vmin=0, vmax=1)

        gt_corners = self.gt_corners

        for gt_i, (x0, y0, x1, y1) in enumerate(self.gt_coords):
            if gt_i not in picked_inds:
                plt.plot([x0, x1], [y0, y1], "-g")
            else:
                plt.plot([x0, x1], [y0, y1], "--c")

        plt.plot(
            gt_corners[:, 0], gt_corners[:, 1], "o", color="tab:orange", markersize=3
        )

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def get_fixes(self, pred_coords):
        gt_coords = self.gt_coords.copy()

        # first compute labels
        labels, pred2gt = my_utils.compute_label(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            threshold=self.match_threshold,
        )

        # use two masks to keep track of which edges are matched
        gt_mask = np.zeros(len(gt_coords), dtype=int)
        pred_mask = np.zeros(len(pred_coords), dtype=int)

        for pred_i, gt_i in pred2gt:
            gt_mask[gt_i] = 1
            pred_mask[pred_i] = 1

        # compute matching costs, this is looser than metric computation
        dists = distance.cdist(
            gt_coords,
            pred_coords,
            my_utils.match_heuristics,
        )

        # ignore all correct edges, since they are matched already
        for pred_i, gt_i in pred2gt:
            dists[:, pred_i] = float("inf")

        # then for each GT edge, find a predicted edge to modify from
        modify_pairs = []
        order = dists.min(axis=1).argsort()

        for gt_i in order:
            # ignore GT edges that are matched already
            if gt_mask[gt_i] > 0:
                continue

            # match with the closest predicted edge
            if dists[gt_i].min() < 1_000_000:
                pred_i = dists[gt_i].argmin()
                dists[:, pred_i] = float("inf")
                modify_pairs.append((pred_i, gt_i))
                gt_mask[gt_i] = 2
                pred_mask[pred_i] = 2

        # leftover GT edges we need to add
        leftover_gt_coords = gt_coords[~gt_mask]

        # leftover predicted edges we need to delete
        leftover_pred_coords = pred_coords[~pred_mask]

        # need to visualize
        if False:
            fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

            ax1.imshow(self.density_full[:, :, 1], cmap="gray")
            ax2.imshow(self.density_full[:, :, 1], cmap="gray")

            cmap = {0: "r", 1: "g", 2: "y"}

            ax1.plot(self.cached_corners[:, 0], self.cached_corners[:, 1], "*c")

            for pred_i, (x0, y0, x1, y1) in enumerate(pred_coords):
                ax1.plot([x0, x1], [y0, y1], color=cmap[pred_mask[pred_i]])

            for gt_i, (x0, y0, x1, y1) in enumerate(gt_coords):
                ax2.plot([x0, x1], [y0, y1], color=cmap[gt_mask[gt_i]])

            ax1.set_title("HEAT edges")
            ax2.set_title("GT edges")

            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.tight_layout()
            plt.show()
            plt.close()

        return pred_mask, gt_mask

    def try_with_new_edge(self, new_coord, pred_corners, pred_edges, branch="hybrid"):
        pred_corners = pred_corners.copy()

        # enumerate candidate edges
        new_corners = new_coord.reshape(2, 2)
        all_corners = np.concatenate([new_corners, pred_corners], axis=0)
        all_corners_raw = all_corners.copy()

        cand_edges = np.stack(
            [
                np.zeros(len(all_corners) - 1, dtype=int),
                np.arange(1, len(all_corners), dtype=int),
            ],
            axis=1,
        )
        cand_coords = all_corners[cand_edges].reshape(-1, 4)
        assert (cand_coords[0] == new_coord).all()

        # combine them, but note that we appended two new corners
        all_edges = np.concatenate([pred_edges + 2, cand_edges], axis=0)

        # mark the edges we want to query, all others are "GT"
        edge_labels = np.ones(len(all_edges), dtype=int)
        edge_labels[len(pred_edges) :] = 2

        # network inference
        image = self.density_full.copy()
        image, all_corners = my_utils.normalize_floor(image, all_corners)
        image = my_utils.process_image(image)

        data = {
            "img": image,
            "edge_coords": all_corners[all_edges].reshape(-1, 4),
            "edge_labels": edge_labels,
            "processed_corners_lengths": len(all_corners),
        }
        data = collate_fn_seq([data])
        probs = self.run_edge_model(data, branch=branch)

        cond_edges = all_edges[edge_labels == 1]
        cand_edges = all_edges[(edge_labels == 2) & (probs > -10)]
        cand_probs = probs[(edge_labels == 2) & (probs > -10)]

        cond_coords = all_corners_raw[cond_edges].reshape(-1, 4)
        cand_coords = all_corners_raw[cand_edges].reshape(-1, 4)

        return cond_coords, cand_coords, cand_probs

    def try_fix_with_new_corners(self, branch="hybrid"):
        pred_corners = self.cached_corners.copy()
        pred_coords, pred_edges = self.get_pred_coords(pred_corners)
        gt_coords = self.gt_coords.copy()
        pred_mask, gt_mask = self.get_fixes(pred_coords)

        # for each missing GT edge, try adding the corners and see if we can recover it
        all_accs = []
        missing_gt_ids = (gt_mask != 1).nonzero()[0]

        for missing_gt_id in tqdm(missing_gt_ids):
            missing_gt_coord = gt_coords[missing_gt_id]

            # enumerate candidate edges
            new_corners = missing_gt_coord.reshape(2, 2)
            all_corners = np.concatenate([new_corners, pred_corners], axis=0)

            cand_edges = np.stack(
                [
                    np.zeros(len(all_corners) - 1, dtype=int),
                    np.arange(1, len(all_corners), dtype=int),
                ],
                axis=1,
            )
            cand_coords = all_corners[cand_edges].reshape(-1, 4)
            assert (cand_coords[0] == missing_gt_coord).all()

            # combine them, but note that we appended two new corners
            all_edges = np.concatenate([pred_edges + 2, cand_edges], axis=0)

            # mark the edges we want to query, all others are "GT"
            edge_labels = np.ones(len(all_edges), dtype=int)
            edge_labels[len(pred_edges) :] = 2

            if False:
                all_coords = all_corners[all_edges].reshape(-1, 4)

                color_coords = []
                color_coords.append(["-oc", all_coords[edge_labels == 1]])
                color_coords.append(["-oy", all_coords[edge_labels == 2]])
                color_coords.append(["--g", [missing_gt_coord]])
                my_utils.vis_edges(self.density_full, color_coords)

            # network inference
            image = self.density_full.copy()
            image, all_corners = my_utils.normalize_floor(image, all_corners)
            image = my_utils.process_image(image)

            data = {
                "img": image,
                "edge_coords": all_corners[all_edges].reshape(-1, 4),
                "edge_labels": edge_labels,
                "processed_corners_lengths": len(all_corners),
            }
            data = collate_fn_seq([data])
            logits = self.run_edge_model(data, branch=branch)

            # check if we recovered the missing edge
            query_i = len(pred_edges)

            # labels = np.zeros(len(cand_edges), dtype=int)
            # labels[0] == 1
            # preds = (logits[len(pred_edges):] > 0.5).astype(int)

            if logits[query_i] > 0.5:
                all_accs.append(1)
            else:
                all_accs.append(0)

            if True:
                all_coords = all_corners[all_edges].reshape(-1, 4)

                color_coords = []
                color_coords.append(["-oc", all_coords[:query_i]])

                if logits[query_i] > 0.5:
                    color_coords.append(["-og", [all_coords[query_i]]])
                    save_f = "./vis/%s_%s_%03d_1.png" % (
                        self.floor_name,
                        branch,
                        missing_gt_id,
                    )
                else:
                    color_coords.append(["-or", [all_coords[query_i]]])
                    save_f = "./vis/%s_%s_%03d_0.png" % (
                        self.floor_name,
                        branch,
                        missing_gt_id,
                    )

                my_utils.vis_edges(image.transpose(1, 2, 0), color_coords, save_f)

        print(
            "Recovery acc: %d / %d = %.3f"
            % (np.sum(all_accs), len(all_accs), np.mean(all_accs) * 100)
        )

    def interactive_corners(self, branch="image"):
        # ax1: HEAT results
        # ax2: GT corners and edges
        # ax3: show edge repicking results
        fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, sharex=True, sharey=True)

        image = self.density_full[:, :, 1]
        ax1.imshow(image, cmap="gray")
        ax2.imshow(image, cmap="gray")
        ax3.imshow(image, cmap="gray")

        gt_coords = self.gt_coords
        heat_coords, heat_edges = self.get_pred_coords(self.cached_corners)
        heat_mask, gt_mask = self.get_fixes(heat_coords)

        for heat_i, (x0, y0, x1, y1) in enumerate(heat_coords):
            if heat_mask[heat_i] == 1:
                ax1.plot([x0, x1], [y0, y1], "-og")
            elif heat_mask[heat_i] == 2:
                ax1.plot([x0, x1], [y0, y1], "-oy")
            elif heat_mask[heat_i] == 0:
                ax1.plot([x0, x1], [y0, y1], "-or")
            else:
                raise Exception("Unknown edge label")

        lines = []

        for gt_i, (x0, y0, x1, y1) in enumerate(gt_coords):
            if gt_mask[gt_i] == 1:
                ax2.plot([x0, x1], [y0, y1], "-og")
            elif gt_mask[gt_i] == 2:
                line = ax2.plot([x0, x1], [y0, y1], "-oy", picker=True, pickradius=1)
                lines.append(line)
            elif gt_mask[gt_i] == 0:
                line = ax2.plot([x0, x1], [y0, y1], "-oc", picker=True, pickradius=1)
                lines.append(line)
            else:
                raise Exception("Unknown edge label")

        def onpick(event):
            thisline = event.artist
            new_coords = thisline.get_xydata().reshape(-1, 4)
            cond_coords, cand_coords, cand_probs = self.try_with_new_edge(
                new_coords, self.cached_corners, heat_edges, branch
            )

            # first remove all previously plotted lines
            ax2 = plt.gcf().axes[2]
            for child in ax2.get_children():
                if type(child) == mlines.Line2D:
                    child.remove()

            # plot input conditions
            for x0, y0, x1, y1 in cond_coords:
                ax2.plot([x0, x1], [y0, y1], "-oy")

            # plot candidate edge predictions
            for cand_i, (x0, y0, x1, y1) in enumerate(cand_coords):
                if cand_probs[cand_i] > 0.5:
                    ax2.plot([x0, x1], [y0, y1], "-og")
                else:
                    ax2.plot([x0, x1], [y0, y1], "-or")

            # plot the GT edge and corners
            [(x0, y0, x1, y1)] = new_coords
            ax2.plot([x0, x1], [y0, y1], "*c")

            plt.gcf().canvas.draw()

        fig.canvas.mpl_connect("pick_event", onpick)

        ax1.set_title("HEAT edges")
        ax2.set_title("GT edges")
        ax3.set_title("Repick results")

        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()

        plt.tight_layout()
        plt.show()
        plt.close()

    def start_server(self, server_ip="127.0.0.1", server_port=33333):
        # so that we don't add back deleted coordinates
        self.deleted_coords = []

        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)
        print("Server listening")

        try:
            while True:
                (connection, addr) = self.socket.accept()
                print("Client connected")

                try:
                    bs = connection.recv(4)
                    (msg_type,) = unpack("<i", bs)

                    bs = connection.recv(4)
                    (length,) = unpack("<i", bs)
                    print("Message type %d, length: %d" % (msg_type, length))

                    data = b""
                    while len(data) < length:
                        # doing it in batches is generally better than trying
                        # to do it all in one go, so I believe.
                        to_read = length - len(data)
                        data += connection.recv(4096 if to_read > 4096 else to_read)

                    if msg_type == 0:
                        print("Sending HEAT predictions")
                        heat_coords, _, heat_widths = self.get_pred_coords(
                            self.cached_corners, branch="hybrid", postprocess=True
                        )
                        heat_coords = self.local2global(heat_coords)

                        # heat_coords = self.local2global(self.gt_coords)
                        # heat_widths = self.gt_widths

                        # return the edges
                        data = np.concatenate(
                            [heat_coords, np.expand_dims(heat_widths, 1)], axis=-1
                        )
                        data = json.dumps(data.tolist()).encode("utf-8")
                        length = len(data)
                        bs = length.to_bytes(4, byteorder="little")

                        print("Sending payload of length %d" % length)
                        connection.sendall(bs)
                        connection.sendall(data)

                    elif msg_type == 1:
                        with Timer("Autocomplete"):
                            modify_container = json.loads(data)
                            modify_container = self.run_autocomplete(modify_container)

                        # return the container
                        data = json.dumps(modify_container).encode("utf-8")
                        length = len(data)
                        bs = length.to_bytes(4, byteorder="little")

                        print("Sending payload of length %d" % length)
                        connection.sendall(bs)
                        connection.sendall(data)

                    elif msg_type == 2:
                        with Timer("Suggestion"):
                            modify_container = json.loads(data)
                            modify_container = self.run_with_new_corners(
                                modify_container
                            )

                        # return the container
                        data = json.dumps(modify_container).encode("utf-8")
                        length = len(data)
                        bs = length.to_bytes(4, byteorder="little")

                        print("Sending payload of length %d" % length)
                        connection.sendall(bs)
                        connection.sendall(data)

                        # except Exception:
                        #     print('Error!')

                    elif msg_type == 3:
                        print("Sending GT walls")
                        edges = self.local2global(self.gt_coords)

                        # return the edges
                        data = json.dumps(edges.tolist()).encode("utf-8")
                        length = len(data)
                        bs = length.to_bytes(4, byteorder="little")

                        print("Sending payload of length %d" % length)
                        connection.sendall(bs)
                        connection.sendall(data)

                    else:
                        raise Exception("Unknown message type %d" % msg_type)

                finally:
                    connection.shutdown(SHUT_WR)
                    connection.close()

        finally:
            self.socket.close()

    def run_with_new_corners(self, container, branch="hybrid"):
        # get the current edges
        eids = []
        edges = []
        for eid, edge in container["walls"].items():
            eids.append(eid)
            edges.append(edge)
        eid2idx = dict(zip(eids, range(len(eids))))
        idx2eid = {v: k for (k, v) in eid2idx.items()}

        edges = np.array(edges)
        curr_coords = self.global2local(edges)

        # get the new corners
        picked_corners = []
        for corner in container["new_corners"]:
            picked_corner = [corner["X"], corner["Y"]]
            picked_corners.append(picked_corner)

        picked_corners = np.array(picked_corners)
        picked_corners = self.global2local(picked_corners)

        # snap the picked corners to existing corners if they are within threshold
        curr_corners, _ = my_utils.corners_and_edges(curr_coords)
        _, picked2curr = my_utils.compute_label(
            picked_corners, curr_corners, threshold=10, dist_fn="l2"
        )

        keep_mask = np.ones(len(curr_corners), dtype=bool)
        for picked_i, curr_i in picked2curr:
            picked_corners[picked_i] = curr_corners[curr_i]
            keep_mask[curr_i] = False

        curr_corners = curr_corners[keep_mask]

        # using the picked corners, enumerate more edge candidates
        cand_corners = np.concatenate([picked_corners, curr_corners])

        combs = []
        for i in range(len(picked_corners)):
            for j in range(i + 1, len(cand_corners)):
                combs.append((i, j))
        combs = np.array(combs)

        cand_coords = cand_corners[combs].reshape(-1, 4)

        # NOTE for some reason this step is not necessary anymore?
        # filter candidate edges to not include any existing ones
        # repeat_mask, cand2curr = my_utils.compute_label(
        #     cand_coords, curr_coords, threshold=self.match_threshold
        # )
        # cand_coords = cand_coords[~repeat_mask.astype(bool)]

        # prepare edge condition flags
        edge_coords = np.concatenate([curr_coords, cand_coords], axis=0)
        edge_labels = np.ones(len(edge_coords), dtype=int)
        edge_labels[len(curr_coords) :] = 2

        # network inference
        # image = self.density_full.copy()
        corners, edges = my_utils.corners_and_edges(edge_coords)

        # image, corners = my_utils.normalize_floor(image, corners)
        # image = my_utils.process_image(image)
        corners, _ = my_utils.normalize_corners(corners, self.norm_scale)
        edge_coords = corners[edges].reshape(-1, 4)

        with Timer("New corner model"):
            data = {
                # "img": image,
                "edge_coords": edge_coords,
                "edge_labels": edge_labels,
                "processed_corners_lengths": len(corners),
            }
            data = collate_fn_seq([data])
            probs = self.run_edge_model(data, branch=branch)
            probs = probs[len(curr_coords) :]

        # we want to pick at least one edge for each new corner
        used_mask = np.zeros(len(picked_corners), dtype=bool)
        new_coords = []

        # find the first edge that contains an unused corner
        rank = probs.argsort()[::-1]
        for rank_i in rank:
            # ignore if we deleted this edge before
            if len(self.deleted_coords):
                matches, _ = my_utils.compute_label(
                    pred_coords=[cand_coords[rank_i]],
                    gt_coords=np.array(self.deleted_coords),
                    threshold=self.match_threshold,
                )
                if matches[0] == 1:
                    continue

            # ignore if the corners are already used
            (ax, ay, bx, by) = cand_coords[rank_i]

            a_idx = my_utils.find_idx((ax, ay), picked_corners)
            b_idx = my_utils.find_idx((bx, by), picked_corners)

            assert (a_idx >= 0) or (b_idx >= 0)

            # if this is classified as a valid edge, add it
            if probs[rank_i] > 0.5:
                new_coords.append(cand_coords[rank_i])

                if a_idx >= 0:
                    used_mask[a_idx] = True
                if b_idx >= 0:
                    used_mask[b_idx] = True

            # if it isn't, we add if one of the corners have not been used
            else:
                if (not used_mask[a_idx]) or (not used_mask[b_idx]):
                    new_coords.append(cand_coords[rank_i])

                    if a_idx >= 0:
                        used_mask[a_idx] = True
                    if b_idx >= 0:
                        used_mask[b_idx] = True

        assert used_mask.all()
        new_coords = np.array(new_coords)
        new_coords = my_utils.snap_to_axis(new_coords)
        # new_coords = my_utils.remove_overlaps(new_coords, curr_coords)

        # visualize
        if False:
            color_coords = []
            color_coords.append(("-og", curr_coords))
            color_coords.append(("-oy", new_coords))
            color_coords.append(("*c", picked_corners))
            my_utils.vis_edges(self.density_full, color_coords)

        # send back the new coordinates to be added
        new_coords = my_utils.find_candidates(curr_coords, new_coords)
        new_coords = self.local2global(new_coords)
        container["suggestions"] = [new_coords.tolist()]
        return container

    def run_autocomplete(self, container, max_order=10, top_k=3, num_steps=3):
        # get the current edges
        eids = []
        edges = []
        for eid, edge in container["walls"].items():
            eids.append(int(eid))
            edges.append(edge)
        eid2idx = dict(zip(eids, range(len(eids))))
        idx2eid = {v: k for (k, v) in eid2idx.items()}

        edges = np.array(edges)
        curr_coords = self.global2local(edges)

        bad_coords = np.array(container["bad_suggestions"])
        if len(bad_coords):
            bad_coords = self.global2local(bad_coords)

        # get all the corners, including the ones we predicted
        with Timer("Obtain candidates"):
            assert self.cached_corners.dtype == curr_coords.dtype == "int64"

            curr_corners, curr_edges = my_utils.corners_and_edges(curr_coords)

            cached2curr = distance.cdist(self.cached_corners, curr_corners)
            new_mask = cached2curr.min(axis=1) >= 5
            new_corners = self.cached_corners[new_mask]

            all_corners = np.concatenate([curr_corners, new_corners], axis=0)

            # obtain all NEW walls that we can add
            all_coords, _, all_widths = self.get_pred_coords(
                all_corners, branch="hybrid", postprocess=True
            )
            # all_edges = {(a, b) for (a, b) in all_edges.tolist()}
            # curr_edges = {(a, b) for (a, b) in curr_edges.tolist()}
            # cand_edges = list(all_edges - curr_edges)
            # cand_coords = all_corners[cand_edges].reshape(-1, 4)
            # cand_coords_copy = cand_coords.copy()
            # cand_coords = my_utils.remove_overlaps(cand_coords, curr_coords)
            cand_coords, cand_widths = my_utils.find_candidates_fast(
                curr_coords, all_coords, all_widths
            )
            if len(bad_coords):
                cand_coords, cand_widths = my_utils.find_candidates_fast(
                    bad_coords, cand_coords, cand_widths
                )

            # color_coords = [["rv", curr_corners], ["g^", new_corners]]
            # color_coords = [["-rv", all_coords], ["-g^", cand_coords]]
            # color_coords = [["-rv", all_coords]]
            # my_utils.vis_edges(self.density_full, color_coords)
            # my_utils.vis_edges_idx(self.density_full, all_coords)

        # it's possible that we don't have any suggestions, so we return empty
        if not len(cand_coords):
            container["suggestions"] = []
            return container

        if False:
            fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

            density_img = np.max(self.density_full, axis=2)
            ax1.imshow(density_img, cmap="gray")
            ax2.imshow(density_img, cmap="gray")

            for x0, y0, x1, y1 in cand_coords_copy:
                ax1.plot([x0, x1], [y0, y1], "-vr")

            for x0, y0, x1, y1 in cand_coords:
                ax2.plot([x0, x1], [y0, y1], "-vr")

            for x0, y0, x1, y1 in curr_coords:
                ax1.plot([x0, x1], [y0, y1], "-^g")
                ax2.plot([x0, x1], [y0, y1], "-^g")

            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.tight_layout()
            plt.show()
            plt.close()

        # prepare flags for all walls
        edge_coords = np.concatenate([curr_coords, cand_coords], axis=0)
        edge_order = np.zeros(len(edge_coords), dtype=int)
        edge_order[: len(curr_coords)] = max_order

        # we reverse the order because the order is actually chronological,
        # but we mark the flag as last = 1, second to last = 2, etc.
        wall_order = container["wall_order"][::-1]
        for order_i, eid in enumerate(wall_order[:max_order]):
            edge_order[eid2idx[eid]] = order_i + 1

        assert edge_order.max() <= max_order

        if False:
            density_img = np.max(self.density_full, axis=2)
            plt.imshow(density_img, cmap="gray")

            max_order = edge_order.max()

            for edge_i, (x0, y0, x1, y1) in enumerate(edge_coords):
                if edge_order[edge_i] == max_order:
                    plt.plot([x0, x1], [y0, y1], "-or")
                elif edge_order[edge_i] == 0:
                    plt.plot([x0, x1], [y0, y1], "-oc")
                else:
                    plt.plot([x0, x1], [y0, y1], "-oy")
                    plt.text(
                        (x0 + x1) / 2, (y0 + y1) / 2, str(edge_order[edge_i]), color="y"
                    )

            plt.axis("off")
            plt.tight_layout()
            plt.show()

        with Timer("Order predict"):
            # pick the top-k edges
            dists = self.run_metric_model(edge_coords, edge_order)
            top_k_inds = dists.argsort()[:top_k].cpu().numpy()
            top_k_coords = edge_coords[top_k_inds]

            # for each of the top-k, run them a number of steps
            rollouts = []

            for top_k in top_k_inds:
                rollout = [top_k]
                rollout_order = edge_order.copy()

                while len(rollout) < num_steps:
                    rollout_order[rollout_order > 0] += 1
                    rollout_order[top_k] = 1
                    rollout_order = np.minimum(rollout_order, max_order)

                    rollout_dists = self.run_metric_model(edge_coords, rollout_order)
                    top_k = rollout_dists.argmin().cpu().numpy().tolist()
                    rollout.append(top_k)

                rollouts.append(rollout)

            rollouts = np.array(rollouts)
            rollout_coords = edge_coords[rollouts]
            rollout_widths = cand_widths[np.maximum(rollouts - len(curr_coords), 0)]

        if False:
            color_coords = [
                ["-oc", curr_coords],
                ["-vg", rollout_coords[0]],
                ["->y", rollout_coords[1]],
                ["-^r", rollout_coords[2]],
            ]
            my_utils.vis_edges(self.density_full, color_coords)

        # return as suggestion
        with Timer("Post-process"):
            suggestions = []

            for coords, widths in zip(rollout_coords, rollout_widths):
                coords = self.local2global(coords)
                data = np.concatenate([coords, np.expand_dims(widths, 1)], axis=-1)
                suggestions.append(data.tolist())

            container["suggestions"] = suggestions

        return container

    # from local pixel space to global in feet
    def local2global(self, edges):
        # thank goodness for shapely
        edges = [((x0, y0), (x1, y1)) for (x0, y0, x1, y1) in edges]
        edges = MultiLineString(edges)

        # undo scaling caused by feet to inches
        # negative yfact because we flip the image upside down for prediction
        xfact = 1.0 / 12.0
        yfact = 1.0 / 12.0

        # offset by the bounds and also padding
        [pad_w_before, pad_h_before, _, _] = self.square_pad

        xoff = self.bounds[0] - pad_w_before
        yoff = self.bounds[1] - pad_h_before

        edges = affinity.translate(edges, xoff=xoff, yoff=yoff)
        edges = affinity.scale(edges, xfact=xfact, yfact=yfact, origin=(0, 0))

        edges = np.array([list(edge.coords) for edge in edges.geoms]).reshape(-1, 4)
        edges = edges.round(3)

        return edges

    def global2local(self, coords):
        # thank goodness for shapely
        if coords.shape[1] == 4:
            coords = [((x0, y0), (x1, y1)) for (x0, y0, x1, y1) in coords]
            coords = MultiLineString(coords)
        elif coords.shape[1] == 5:
            coords = [((x0, y0), (x1, y1)) for (x0, y0, x1, y1, _) in coords]
            coords = MultiLineString(coords)
        elif coords.shape[1] == 2:
            coords = MultiPoint(coords)
        else:
            raise Exception("Unknown coordinate type")

        # undo scaling caused by feet to inches
        # negative yfact because we flip the image upside down for prediction
        xfact = 12.0
        yfact = 12.0

        # offset by the revit crop corner, and also padding
        [pad_w_before, pad_h_before, _, _] = self.square_pad

        xoff = -self.bounds[0] + pad_w_before
        yoff = -self.bounds[1] + pad_h_before

        coords = affinity.scale(coords, xfact=xfact, yfact=yfact, origin=(0, 0))
        coords = affinity.translate(coords, xoff=xoff, yoff=yoff)

        coords = np.array([list(edge.coords) for edge in coords.geoms])
        coords = coords.astype(int)

        if coords.shape[1] == 2:
            coords = coords.reshape(-1, 4)
        else:
            coords = coords.reshape(-1, 2)

        return coords

    def vis_gt(self):
        pred_coords, _ = self.get_pred_coords(self.cached_corners)
        density_img = np.max(self.density_full, axis=2)
        color_coords = [["-vc", pred_coords], ["--^m", self.gt_coords]]
        my_utils.vis_edges(density_img, color_coords)


def compute_metrics():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deform_type", type=str, default="DETR_dense")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--branch", type=str, default="hybrid")
    parser.add_argument("--postprocess", action="store_true", default=True)
    args = parser.parse_args()

    all_precision = []
    all_recall = []
    all_f_scores = []
    all_iou = []
    all_accs_width = []

    for floor_idx in range(16):
        backend = Backend(deform_type=args.deform_type, num_samples=args.num_samples)
        backend.init_edge_models(floor_idx)
        backend.init_floor(floor_idx)

        corners = backend.cached_corners
        backend.cache_image_feats(corners)

        gt_coords = backend.gt_coords
        gt_widths = backend.gt_widths

        pred_coords, _, pred_widths = backend.get_pred_coords(
            corners,
            branch=args.branch,
            postprocess=args.postprocess,
        )

        # compute P/R
        precisions = []
        recalls = []
        f_scores = []

        # also for computing width accuracy
        accs_width = []

        for threshold in [5, 15, 30]:
            labels, pred2gt = my_utils.compute_label(
                pred_coords=pred_coords,
                gt_coords=gt_coords,
                threshold=threshold,
            )

            precision = (labels == 1).sum() / len(labels) * 100
            recall = (labels == 1).sum() / len(gt_coords) * 100
            f_score = 2.0 * precision * recall / (recall + precision + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(f_score)

            _accs_width = [[] for _ in range(1, 10)]
            for pred_i, gt_i in pred2gt:
                for tolerance in range(1, 10):
                    if abs(pred_widths[pred_i] - gt_widths[gt_i]) < tolerance:
                        _accs_width[tolerance - 1].append(1)
                    else:
                        _accs_width[tolerance - 1].append(0)

            _accs_width = np.array(_accs_width).mean(axis=1)
            accs_width.append(_accs_width)

        all_precision.append(precisions)
        all_recall.append(recalls)
        all_f_scores.append(f_scores)
        all_accs_width.append(accs_width)

        # compute IoU
        gt_boxes = []
        for (x0, y0, x1, y1), width in zip(gt_coords, gt_widths):
            line_shp = LineString([(x0, y0), (x1, y1)])
            box_shp = line_shp.buffer(width / 2, cap_style=2)
            gt_boxes.append(box_shp)
        gt_boxes = shapely.unary_union(gt_boxes)

        pred_boxes = []
        for (x0, y0, x1, y1), width in zip(pred_coords, pred_widths):
            line_shp = LineString([(x0, y0), (x1, y1)])
            box_shp = line_shp.buffer(width / 2, cap_style=2)
            pred_boxes.append(box_shp)
        pred_boxes = shapely.unary_union(pred_boxes)

        intersection = gt_boxes.intersection(pred_boxes)
        union = gt_boxes.union(pred_boxes)
        iou = intersection.area / union.area
        all_iou.append(iou)

        if False:
            fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, sharex=True, sharey=True)

            image = np.max(backend.density_full, axis=2)
            ax1.imshow(image, cmap="gray")
            ax2.imshow(image, cmap="gray")
            ax3.imshow(image, cmap="gray")

            # plot GT
            for poly in gt_boxes.geoms:
                (x, y) = poly.exterior.xy
                ax1.fill(x, y, facecolor="lightsalmon")

            for (x0, y0, x1, y1), width in zip(gt_coords, gt_widths):
                line_shp = LineString([(x0, y0), (x1, y1)])
                box_shp = line_shp.buffer(width / 2, cap_style=2)
                (x, y) = box_shp.exterior.xy
                ax1.plot(x, y, "-c")

            # plot pred
            for poly in pred_boxes.geoms:
                (x, y) = poly.exterior.xy
                ax2.fill(x, y, facecolor="lightsalmon")

            for (x0, y0, x1, y1), width in zip(pred_coords, pred_widths):
                line_shp = LineString([(x0, y0), (x1, y1)])
                box_shp = line_shp.buffer(width / 2, cap_style=2)
                (x, y) = box_shp.exterior.xy
                ax2.plot(x, y, "-c")

            # plot intersection over union
            for poly in intersection.geoms:
                (x, y) = poly.exterior.xy
                ax3.fill(x, y, facecolor="lightsalmon")

            for poly in union.geoms:
                (x, y) = poly.exterior.xy
                ax3.fill(x, y, facecolor="lightblue")

            ax1.set_axis_off()
            ax2.set_axis_off()
            ax3.set_axis_off()

            plt.show()
            plt.close()

        # color_coords = [
        #     ["-og", pred_coords],
        #     ["*c", corners],
        # ]
        # my_utils.vis_edges(backend.density_full, color_coords)

        # color_coords = [
        #     ["-og", pred_coords, pred_widths],
        # ]
        # my_utils.vis_edges_width(backend.density_full, color_coords)

    all_precision = np.array(all_precision)
    all_recall = np.array(all_recall)
    all_f_scores = np.array(all_f_scores)
    all_iou = np.array(all_iou)
    all_accs_width = np.array(all_accs_width)

    # print(" 5: P %.3f R %.3f" % (all_precision[:, 0].mean(), all_recall[:, 0].mean()))
    # print("15: P %.3f R %.3f" % (all_precision[:, 1].mean(), all_recall[:, 1].mean()))
    # print("30: P %.3f R %.3f" % (all_precision[:, 2].mean(), all_recall[:, 2].mean()))

    with open("pr_paper_updated.csv", "a") as f:
        f.write(
            "%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n"
            % (
                args.deform_type,
                args.num_samples,
                all_precision[:, 0].mean(),
                all_recall[:, 0].mean(),
                all_f_scores[:, 0].mean(),
                all_accs_width[:, 0].mean(),
                all_precision[:, 1].mean(),
                all_recall[:, 1].mean(),
                all_f_scores[:, 1].mean(),
                all_accs_width[:, 1].mean(),
                all_precision[:, 2].mean(),
                all_recall[:, 2].mean(),
                all_f_scores[:, 2].mean(),
                all_accs_width[:, 2].mean(),
                all_iou.mean(),
            )
        )


def compute_metrics_corner():
    all_precisions = []
    all_recalls = []

    for floor_idx in range(16):
        backend = Backend()
        backend.init_floor(floor_idx)

        [pad_w_before, pad_h_before, pad_w_after, pad_h_after] = backend.square_pad

        gt_corners = backend.gt_corners

        corner_f = os.path.join(
            backend.data_path, "pred_corners_4/%s.json" % backend.floor_name
        )
        with open(corner_f, "r") as f:
            pred_corners = json.load(f)
        pred_corners = np.array(pred_corners)
        # pred_corners -= 128
        pred_corners += [pad_w_before, pad_h_before]

        labels, _ = my_utils.compute_label(
            pred_corners, gt_corners, threshold=backend.match_threshold, dist_fn="l2"
        )
        precision = (labels == 1).sum() / len(labels) * 100
        recall = (labels == 1).sum() / len(gt_corners) * 100

        all_precisions.append(precision)
        all_recalls.append(recall)

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)

    print("P: %.3f  R: %.3f" % (avg_precision, avg_recall))


def export_corners():
    all_precisions = []
    all_recalls = []

    for floor_idx in range(16):
        backend = Backend()
        backend.init_floor(floor_idx)
        backend.init_corner_models(floor_idx)

        gt_corners = backend.gt_corners
        pred_corners = backend.get_pred_corners()

        labels, _ = my_utils.compute_label(
            pred_corners, gt_corners, threshold=backend.match_threshold, dist_fn="l2"
        )
        precision = (labels == 1).sum() / len(labels) * 100
        recall = (labels == 1).sum() / len(gt_corners) * 100

        all_precisions.append(precision)
        all_recalls.append(recall)

        # need to remove padding before saving
        [pad_w_before, pad_h_before, pad_w_after, pad_h_after] = backend.square_pad
        pred_corners -= [pad_w_before, pad_h_before]

        save_f = "%s/pred_corners/%s.json" % (backend.data_path, backend.floor_name)
        with open(save_f, "w") as f:
            json.dump(pred_corners.tolist(), f)

        # color_coords = [
        #     ["vy", pred_corners],
        #     ["^g", gt_corners],
        # ]
        # my_utils.vis_edges(backend.density_full, color_coords)

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)

    print("P: %.3f  R: %.3f" % (avg_precision, avg_recall))


def save_edge_preds():
    for floor_idx in range(16):
        backend = Backend(deform_type="DETR_dense", num_samples=16)
        backend.init_edge_models(floor_idx)
        backend.init_floor(floor_idx)

        corners = backend.cached_corners
        backend.cache_image_feats(corners)

        pred_coords, _, _ = backend.get_pred_coords(
            corners, branch="hybrid", postprocess=True
        )

        print(backend.floor_name)
        my_utils.vis_edges(
            backend.density_full,
            [["-oc", pred_coords]],
            title=backend.floor_name,
            save_f="./vis_pred_full/%s.png" % backend.floor_name,
        )

        continue

        # need to remove padding before saving
        [pad_w_before, pad_h_before, pad_w_after, pad_h_after] = backend.square_pad
        pred_coords -= [pad_w_before, pad_h_before, pad_w_before, pad_h_before]

        np.save(
            "./data/bim_dataset_big_v5/pred_full_paper/%s.npy" % backend.floor_name,
            pred_coords,
        )


@app.command()
def demo_floor():
    floor_idx = 11

    backend = Backend()
    backend.init_floor(floor_idx)
    backend.init_corner_models(floor_idx)
    backend.init_edge_models(floor_idx)
    backend.init_metric_model(floor_idx)
    backend.cache_image_feats(backend.cached_corners)

    backend.start_server()


if __name__ == "__main__":
    app()
