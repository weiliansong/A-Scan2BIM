import json
import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import distance
from shapely import affinity
from skimage.transform import SimilarityTransform
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import my_utils
from timer import Timer
from utils.nn_utils import positional_encoding_2d

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

EPS = 1e-6

# for 512 dual big dataset
density_mean = [0.18115416, 0.18115416, 0.18115416]
density_std = [0.27998772, 0.27998772, 0.27998772]

# combined
# combined_mean = [0.06896243, 0.06896243, 0.06896243]
# combined_std = [0.16101032, 0.16101032, 0.16101032]

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


def line_dist(ab, cd):
    ax, ay, bx, by = ab
    cx, cy, dx, dy = cd

    l2_ac = np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2)
    l2_ad = np.sqrt((dx - ax) ** 2 + (dy - ay) ** 2)
    l2_bc = np.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
    l2_bd = np.sqrt((dx - bx) ** 2 + (dy - by) ** 2)

    if l2_ac + l2_bd < l2_ad + l2_bc:
        return max(l2_ac, l2_bd)
    else:
        return max(l2_ad, l2_bc)


def get_scores(pred_coords, gt_coords, threshold=15):
    dists = distance.cdist(pred_coords, gt_coords, line_dist)
    label = ~((dists < threshold).any(axis=1))

    scores = {
        "edge_tp": (label == 0).sum(),
        "edge_fp": (label == 1).sum(),
        "edge_length": len(gt_coords),
        "false_edge_ids": (label == 1).nonzero()[0],
    }
    return scores


class BuildingCornerDataset(Dataset):
    def __init__(
        self,
        data_path,
        revit_root,
        phase="train",
        image_size=256,
        rand_aug=True,
        test_idx=-1,
        multiplier=1,
        loss_type=None,
    ):
        super(BuildingCornerDataset, self).__init__()
        self.data_path = data_path
        self.revit_root = revit_root
        self.phase = phase
        self.rand_aug = rand_aug
        self.image_size = image_size
        self.loss_type = loss_type

        print("Random augmentation: %s" % str(self.rand_aug))
        print("Loss type: %s" % self.loss_type)

        # assert image_size == 512
        assert test_idx > -1
        assert loss_type

        floor_f = os.path.join(data_path, "all_floors.txt")
        with open(floor_f, "r") as f:
            floors = [x.strip().split(",") for x in f.readlines()]

        # remove the testing floor right now
        test_floor = floors[test_idx]
        del floors[test_idx]

        # find the index to the smallest floor, used for validation
        fewest_idx = -1
        fewest_views = float("inf")

        for floor_idx, (floor, num_views) in enumerate(floors):
            if int(num_views) < fewest_views:
                fewest_idx = floor_idx
                fewest_views = int(num_views)

        assert fewest_idx > -1
        val_floor = floors[fewest_idx]
        del floors[fewest_idx]

        # prepare splits
        if phase == "train":
            assert len(floors) == 14
        elif phase == "train_eval":
            assert len(floors) == 14
        elif phase == "valid":
            floors = [val_floor]
        elif phase == "test":
            floors = [test_floor]
        elif phase == "data":
            floors = [test_floor]
        else:
            raise ValueError("Invalid phase {}".format(phase))

        print("%s: %s" % (phase, str(floors)))

        # for each floor
        self.density_fulls = {}
        self.orders = {}
        self.ordered_edges = {}
        self.heat_edges = {}
        self.comb_edges = {}
        self.examples = []

        for floor_name in floors:
            floor_name = floor_name[0]
            print(floor_name)

            # load full density image
            tokens = floor_name.split("_")
            first = "_".join(tokens[:-1])
            second = tokens[-1]

            # load full density image
            density_slices = []
            for slice_i in range(7):
                slice_f = "../../../revit_projects/%s/%s/density_%02d.npy" % (
                    first,
                    second,
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

            # pad the image so it's a square
            (h, w, _) = density_full.shape
            side_len = max(h, w)

            pad_h_before = (side_len - h) // 2
            pad_h_after = side_len - h - pad_h_before
            pad_w_before = (side_len - w) // 2
            pad_w_after = side_len - w - pad_w_before

            density_full = np.pad(
                density_full,
                [[pad_h_before, pad_h_after], [pad_w_before, pad_w_after], [0, 0]],
            )
            self.density_fulls[floor_name] = density_full

            # load GT annotation and pad them
            annot_f = os.path.join(
                data_path, "annot/%s_one_shot_full.json" % floor_name
            )
            with open(annot_f, "r") as f:
                annot = json.load(f)

            gt_eids = list(annot.keys())
            gt_edges = np.array(list(annot.values()))[:, :4]
            gt_edges += [pad_w_before, pad_h_before, pad_w_before, pad_h_before]

            # load predicted HEAT edges and pad them
            heat_f = os.path.join(data_path, "pred_full_paper/%s.npy" % floor_name)
            heat_edges = np.load(heat_f)
            heat_edges += [pad_w_before, pad_h_before, pad_w_before, pad_h_before]
            self.heat_edges[floor_name] = heat_edges

            # load GT order information
            order = []

            actions_f = os.path.join(data_path, "actions/%s.json" % floor_name)
            with open(actions_f, "r") as f:
                actions = json.load(f)

            for action in actions:
                for (eid, _) in action["added"]:
                    eid = str(eid)
                    if (eid in gt_eids) and (eid not in order):
                        order.append(eid)

            # order edges
            eid2idx = dict(zip(gt_eids, range(len(gt_eids))))

            def order2idx(order_i):
                return eid2idx[order[order_i]]

            edge_ordering = [order2idx(order_i) for order_i in range(len(order))]
            ordered_edges = gt_edges[edge_ordering]
            self.ordered_edges[floor_name] = ordered_edges

            # combine edges
            labels, heat2ordered = my_utils.compute_label(
                heat_edges, ordered_edges, threshold=30
            )

            comb_edges = ordered_edges.copy()
            for (heat_i, ordered_i) in heat2ordered:
                comb_edges[ordered_i] = heat_edges[heat_i]

            comb_edges = np.concatenate([comb_edges, heat_edges[labels == 0]])
            self.comb_edges[floor_name] = comb_edges

            # now ready to generate examples
            num_steps = len(ordered_edges)
            min_len = 2
            max_len = 10

            if phase == "test":
                directions = ["forward"]
            else:
                directions = ["forward", "reverse"]

            if loss_type == "metric":
                examples = self.prepare_metric_examples(
                    floor_name, directions, num_steps, min_len, max_len
                )
            elif loss_type == "class":
                examples = self.prepare_class_examples(
                    floor_name, directions, num_steps, min_len, max_len
                )
            else:
                raise Exception("Unknown loss type to prepare examples for")

            self.examples.extend(examples)

        self.examples *= multiplier

    def prepare_class_examples(
        self, floor_name, directions, num_steps, min_len, max_len
    ):
        pos_examples = []
        neg_examples = []

        for direction in directions:
            for start_idx in range(0, num_steps - max_len):
                for end_idx in range(start_idx + min_len, start_idx + max_len):
                    assert (start_idx < num_steps) and (end_idx < num_steps)

                    # positive example would be without any damages
                    neg_idx = -1
                    pos_examples.append(
                        [floor_name, direction, start_idx, end_idx, neg_idx]
                    )

                    neg_inds = list(range(end_idx, num_steps))
                    # if len(neg_inds) > 20:
                    #     neg_inds = neg_inds[:20]

                    for neg_idx in neg_inds:
                        neg_examples.append(
                            [floor_name, direction, start_idx, end_idx, neg_idx]
                        )

        # balance the two classes
        pos_multipler = len(neg_examples) // len(pos_examples)
        pos_examples *= pos_multipler

        return pos_examples + neg_examples

    def prepare_metric_examples(
        self, floor_name, directions, num_steps, min_len, max_len
    ):
        examples = []

        # for direction in directions:
        #     for start_idx in range(0, num_steps - max_len):
        #         for end_idx in range(start_idx + min_len, start_idx + max_len):
        #             assert (start_idx < num_steps) and (end_idx < num_steps)
        #             examples.append([floor_name, direction, start_idx, end_idx])

        for direction in directions:
            for start_idx in range(0, num_steps - min_len):
                for end_idx in range(start_idx + min_len, num_steps):
                    examples.append([floor_name, direction, start_idx, end_idx])

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.loss_type == "class":
            example = self.gen_class_example(self.examples[idx])
        elif self.loss_type == "metric":
            example = self.gen_metric_example(self.examples[idx])
        else:
            raise Exception("Unknown loss type")

        # random dropping of condition edges, if there are any
        if self.rand_aug:
            edge_order = example["edge_order"]
            keep_mask = np.ones_like(edge_order, dtype=bool)
            num_total = (edge_order == 10).sum()

            if num_total > 0:
                num_ignore = np.random.choice(num_total)
                ignore_inds = np.random.choice(num_total, num_ignore, replace=False)
                keep_mask[ignore_inds] = False

                example["edge_coords"] = example["edge_coords"][keep_mask]
                example["edge_order"] = example["edge_order"][keep_mask]
                if self.loss_type == "metric":
                    example["edge_label"] = example["edge_label"][keep_mask]

        # normalize and augment
        corners, edges = my_utils.corners_and_edges(example["edge_coords"])
        corners, scale = my_utils.normalize_corners(corners)

        if self.rand_aug:
            # random rotate and scale of edges
            corners = self.aug_example(corners)

        edge_coords = corners[edges].reshape(-1, 4)
        example["edge_coords"] = edge_coords
        example["normalize_param"] = scale

        # self.vis_example(example, aug_param, 0)

        return example

    def gen_class_example(self, example_tup, max_order=10):
        (floor_name, direction, start_idx, end_idx, neg_idx) = example_tup

        ordered_edges = self.ordered_edges[floor_name].copy()

        assert direction in ["forward", "reverse"]
        if direction == "reverse":
            ordered_edges = ordered_edges[::-1]

        edge_coords = ordered_edges[:end_idx]
        edge_order = np.full(len(edge_coords), max_order)
        edge_order[start_idx:end_idx] = np.arange(end_idx - start_idx, 0, -1)

        if neg_idx == -1:
            label = 1
        else:
            label = 0
            edge_coords[-1] = ordered_edges[neg_idx]

        example = {
            "floor_name": floor_name,
            "edge_coords": edge_coords,
            "edge_order": edge_order,
            "label": label,
        }
        return example

    def gen_metric_example(self, example_tup, max_order=10):
        (floor_name, direction, start_idx, end_idx) = example_tup

        comb_edges = self.comb_edges[floor_name].copy()

        assert direction in ["forward", "reverse"]
        if direction == "reverse":
            num_edges = len(self.ordered_edges[floor_name])
            comb_edges[:num_edges] = comb_edges[:num_edges][::-1]

        # edge_order = np.zeros(len(ordered_edges))
        # order_inds = list(range(end_idx - start_idx - 1, 0, -1))
        # edge_order[:start_idx] = max_order
        # edge_order[start_idx : end_idx - 1] = order_inds
        # assert edge_order.max() <= max_order

        edge_order = np.zeros(len(comb_edges))
        order_inds = list(range(end_idx - start_idx - 1, 0, -1))
        edge_order[start_idx : end_idx - 1] = order_inds
        edge_order = np.minimum(edge_order, max_order)

        # we only keep relevant edges
        label = np.zeros_like(edge_order)
        label[edge_order != 0] = 3  # edges to ignore, since they are conditions
        label[end_idx - 2] = 1  # the last provided edge, anchor
        label[end_idx - 1] = 2  # should be the next edge, positive

        example = {
            "floor_name": floor_name,
            "edge_coords": comb_edges,
            "edge_order": edge_order,
            "edge_label": label,
        }
        return example

    # random rotate and scaling
    def aug_example(self, corners):
        # random rotate
        angle = np.random.choice(range(360))
        angle = np.deg2rad(angle)

        minx = corners[:, 0].min()
        miny = corners[:, 1].min()
        maxx = corners[:, 0].max()
        maxy = corners[:, 1].max()
        cx = -1 * (minx + maxx) / 2
        cy = -1 * (miny + maxy) / 2

        T_center = SimilarityTransform(translation=(cx, cy))
        T_rot = SimilarityTransform(rotation=angle)
        new_corners = T_center(corners)
        new_corners = T_rot(new_corners)
        new_corners = T_center.inverse(new_corners)

        # random scale
        xfact = np.random.choice(range(75, 100)) / 100
        yfact = np.random.choice(range(75, 100)) / 100

        new_corners[:, 0] *= xfact
        new_corners[:, 1] *= yfact

        return new_corners

    def vis_example(self, example, aug_param, start_idx, guess=False):
        # undo augmentation to edges
        floor_name = example["floor_name"]
        edge_coords = example["edge_coords"]
        gt_coords = example["gt_coords"]
        edge_order = example["edge_order"]

        if aug_param:
            edge_coords = self.unnormalize_edges(edge_coords, aug_param)
        edge_coords = self.unnormalize_edges(edge_coords, example["normalize_param"])

        fig, [ax1, ax2] = plt.subplots(ncols=2)
        size = fig.get_size_inches()
        fig.set_size_inches(size * 2)

        density_full = self.density_fulls[example["floor_name"]]
        ax1.imshow(density_full[:, :, 1], cmap="gray")
        ax2.imshow(density_full[:, :, 1], cmap="gray")

        max_order = 10

        for edge_i, (x0, y0, x1, y1) in enumerate(edge_coords):
            if edge_order[edge_i] == max_order:
                ax1.plot([x0, x1], [y0, y1], "-", color="red", linewidth=4)
            else:
                ax1.plot([x0, x1], [y0, y1], "-oy")
                ax1.text(
                    (x0 + x1) / 2, (y0 + y1) / 2, str(edge_order[edge_i]), color="c"
                )

        for edge_i, (x0, y0, x1, y1) in enumerate(gt_coords):
            if edge_order[edge_i] == max_order:
                ax2.plot([x0, x1], [y0, y1], "-", color="red", linewidth=4)
            else:
                ax2.plot([x0, x1], [y0, y1], "-oy")
                ax2.text(
                    (x0 + x1) / 2, (y0 + y1) / 2, str(edge_order[edge_i]), color="c"
                )

        ax1.set_axis_off()
        ax2.set_axis_off()

        ax1.set_title("Label: %d" % example["label"])

        plt.tight_layout()

        plt.show()
        # plt.savefig(save_f, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    def vis_actions(
        self,
        actions,
        density_full,
        floor_name,
        padding,
        final_db,
    ):
        density_slice = (density_full[:, :, 1] * 255).round().astype(np.uint8)
        density_pil = Image.fromarray(density_slice).convert("RGB")

        # only the add order is drawn here
        add_img = density_pil.copy()
        draw_add = ImageDraw.Draw(add_img)

        bounds_f = "%s/bounds/%s.csv" % (self.data_path, floor_name)
        with open(bounds_f, "r") as f:
            (minx, miny, _, _) = [float(x) for x in f.readline().strip().split(",")]
        minx -= padding[0]
        miny -= padding[1]

        db = {}
        frames = []
        zs = []

        for action in actions:
            for (eid, coords) in action["added"]:
                (x0, y0, z0, x1, y1, z1) = coords
                zs.append(z0)
                zs.append(z1)

                if eid in db.keys():
                    print("Already in DB?")
                db[eid] = [x0 - minx, y0 - miny, x1 - minx, y1 - miny]

                if str(eid) in final_db.keys():
                    xyxy = final_db[str(eid)].tolist()
                    draw_add.line(xyxy, fill=(255, 255, 0), width=3)

            for (eid, before, after) in action["modified"]:
                (x0, y0, z0, x1, y1, z1) = after
                zs.append(z0)
                zs.append(z1)

                if eid not in db.keys():
                    print("Not in DB?")
                db[eid] = [x0 - minx, y0 - miny, x1 - minx, y1 - miny]

            for eid in action["deleted"]:
                assert eid in db.keys()
                del db[eid]

            # visualize this step
            img = density_pil.copy()
            draw = ImageDraw.Draw(img)

            for xyxy in db.values():
                draw.line(xyxy, fill=(255, 255, 0), width=3)

            frame = np.concatenate([np.array(add_img), np.array(img)], axis=1)
            frames.append(frame)

        imageio.mimsave(
            "vis_steps/%s.gif" % floor_name,
            frames,
            duration=0.2,
            loop=0,
        )
        exit(0)

    def vis_intersections(self, modify_shp, intersect_shps, all_shps, floor_name):
        density_full = self.density_fulls[floor_name]

        fig, [ax1, ax2] = plt.subplots(ncols=2)

        ax1.imshow(density_full[:, :, 1], cmap="gray")
        ax2.imshow(density_full[:, :, 1], cmap="gray")

        for shp in intersect_shps:
            poly = shp.buffer(5)
            ax1.plot(*poly.exterior.xy)

        ((x0, y0), (x1, y1)) = list(modify_shp.coords)
        ax1.plot([x0, x1], [y0, y1], "-*")

        for shp in all_shps:
            poly = shp.buffer(5)
            ax2.plot(*poly.exterior.xy)

        ax1.set_axis_off()
        ax2.set_axis_off()

        plt.tight_layout()
        plt.show()
        plt.close()

    def normalize_edges(self, edges, normalize_param=None):
        raise Exception("DO NOT USE!")

        lines = [((x0, y0), (x1, y1)) for (x0, y0, x1, y1) in edges]
        lines = MultiLineString(lines)

        if not normalize_param:
            # center edges around 0
            (minx, miny, maxx, maxy) = lines.bounds
            w = maxx - minx
            h = maxy - miny
            xoff = -1 * (minx + maxx) / 2
            yoff = -1 * (miny + maxy) / 2

            # normalize so longest edge is 1000, and to not change aspect ratio
            if max(w, h) > 1000:
                if w > h:
                    xfact = yfact = 1000 / w
                else:
                    xfact = yfact = 1000 / h
            else:
                xfact = yfact = 1

        else:
            (xoff, yoff) = normalize_param["translation"]
            (xfact, yfact) = normalize_param["scale"]

        lines = affinity.translate(lines, xoff=xoff, yoff=yoff)
        lines = affinity.scale(lines, xfact=xfact, yfact=yfact, origin=(0, 0))

        new_coords = [list(line.coords) for line in lines.geoms]
        new_coords = np.array(new_coords).reshape(-1, 4)

        normalize_param = {
            "translation": (xoff, yoff),
            "scale": (xfact, yfact),
        }

        return new_coords, normalize_param

    def unnormalize_edges(self, edges, normalize_param):
        raise Exception("DO NOT USE!")

        lines = [((x0, y0), (x1, y1)) for (x0, y0, x1, y1) in edges]
        lines = MultiLineString(lines)

        # normalize so longest edge is 1000, and to not change aspect ratio
        (xfact, yfact) = normalize_param["scale"]
        xfact = 1 / xfact
        yfact = 1 / yfact
        lines = affinity.scale(lines, xfact=xfact, yfact=yfact, origin=(0, 0))

        # center edges around 0
        (xoff, yoff) = normalize_param["translation"]
        lines = affinity.translate(lines, xoff=-xoff, yoff=-yoff)

        # rotation
        if "rotation" in normalize_param.keys():
            angle = normalize_param["rotation"]
            lines = affinity.rotate(lines, -angle, origin=(0, 0))

        new_coords = [list(line.coords) for line in lines.geoms]
        new_coords = np.array(new_coords).reshape(-1, 4)

        return new_coords


def collate_fn_corner(data):
    batched_data = {}
    for field in data[0].keys():
        if field == "floor_name":
            batch_values = [item[field] for item in data]
        else:
            batch_values = default_collate([d[field] for d in data])
        if field in ["edge_coords", "label"]:
            batch_values = batch_values.long()
        batched_data[field] = batch_values

    return batched_data


def pad_sequence(seq, length, pad_value=0):
    if len(seq) == length:
        return seq
    else:
        pad_len = length - len(seq)
        if len(seq.shape) == 1:
            if pad_value == 0:
                paddings = np.zeros(
                    [
                        pad_len,
                    ]
                )
            else:
                paddings = (
                    np.ones(
                        [
                            pad_len,
                        ]
                    )
                    * pad_value
                )
        else:
            if pad_value == 0:
                paddings = np.zeros(
                    [
                        pad_len,
                    ]
                    + list(seq.shape[1:])
                )
            else:
                paddings = (
                    np.ones(
                        [
                            pad_len,
                        ]
                        + list(seq.shape[1:])
                    )
                    * pad_value
                )
        paddings = paddings.astype(seq.dtype)
        padded_seq = np.concatenate([seq, paddings], axis=0)
        return padded_seq


def collate_fn_seq(data):
    batched_data = {}
    lengths_info = {}

    for field in data[0].keys():
        batch_values = [example[field] for example in data]

        if field in ["floor_name", "normalize_param"]:
            batched_data[field] = batch_values

        elif ("_coords" in field) or ("_order" in field) or ("_label" in field):
            all_lens = [len(value) for value in batch_values]
            max_len = max(all_lens)
            pad_value = 0

            batch_values = [
                pad_sequence(value, max_len, pad_value) for value in batch_values
            ]
            batch_values = np.stack(batch_values, axis=0)
            batch_values = torch.Tensor(batch_values)

            if "_coords" in field:
                lengths_info[field] = all_lens

            batched_data[field] = batch_values

        else:
            batched_data[field] = default_collate(batch_values).long()

    # Add length and mask into the data, the mask if for Transformers' input format, True means padding
    for field, lengths in lengths_info.items():
        lengths_str = field + "_lengths"
        batched_data[lengths_str] = torch.Tensor(lengths).long()
        mask = torch.arange(max(lengths))
        mask = mask.unsqueeze(0).repeat(batched_data[field].shape[0], 1)
        mask = mask >= batched_data[lengths_str].unsqueeze(-1)
        mask_str = field + "_mask"
        batched_data[mask_str] = mask

    return batched_data


def get_pixel_features(image_size, d_pe=128):
    all_pe = positional_encoding_2d(d_pe, image_size, image_size)
    pixels_x = np.arange(0, image_size)
    pixels_y = np.arange(0, image_size)

    xv, yv = np.meshgrid(pixels_x, pixels_y)
    all_pixels = list()
    for i in range(xv.shape[0]):
        pixs = np.stack([xv[i], yv[i]], axis=-1)
        all_pixels.append(pixs)
    pixels = np.stack(all_pixels, axis=0)

    pixel_features = all_pe[:, pixels[:, :, 1], pixels[:, :, 0]]
    pixel_features = pixel_features.permute(1, 2, 0)
    return pixels, pixel_features


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    DATAPATH = "./data/cities_dataset"
    DET_PATH = "./data/det_final"
    train_dataset = BuildingCornerDataset(DATAPATH, DET_PATH, phase="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_corner,
    )
    for i, item in enumerate(train_dataloader):
        import pdb

        pdb.set_trace()
        print(item)
