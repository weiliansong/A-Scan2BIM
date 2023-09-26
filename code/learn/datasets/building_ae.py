import time
from email.policy import default
import glob
from hashlib import new
import json
from timeit import repeat
from tkinter import Y
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import skimage
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm
from utils.nn_utils import positional_encoding_2d, pos_encode_2d
from torchvision import transforms
from PIL import Image, ImageDraw
from skimage.transform import AffineTransform, warp, rotate
import itertools
import imageio

# from rtree import index
from torch.utils.data.dataloader import default_collate
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString, box, MultiLineString
from shapely import affinity
from metrics.get_metric import compute_metrics
from scipy.spatial import distance

# from infer_full import FloorHEAT
import my_utils

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

EPS = 1e-6


def normalize_edges(edges, normalize_param=None):
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
        if w > h:
            xfact = yfact = 1 / w
        else:
            xfact = yfact = 1 / h

    else:
        (xoff, yoff) = normalize_param["translation"]
        (xfact, yfact) = normalize_param["scale"]

    lines = affinity.translate(lines, xoff=xoff, yoff=yoff)
    lines = affinity.scale(lines, xfact=xfact, yfact=yfact, origin=(0, 0))

    new_coords = [list(line.coords) for line in lines.geoms]
    new_coords = np.array(new_coords).reshape(-1, 4)

    assert new_coords.max() <= 0.5 + 1e-6
    assert new_coords.min() >= -0.5 - 1e-6

    normalize_param = {
        "translation": (xoff, yoff),
        "scale": (xfact, yfact),
    }

    return new_coords, normalize_param


class BuildingCornerDataset(Dataset):
    def __init__(
        self,
        data_path,
        phase="train",
        rand_aug=True,
        test_idx=-1,
        min_seq_len=2,
        max_seq_len=10,
        epoch_size=1000,
    ):
        super(BuildingCornerDataset, self).__init__()
        self.data_path = data_path
        self.phase = phase
        self.rand_aug = rand_aug
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.epoch_size = epoch_size

        print("Random augmentation: %s" % str(self.rand_aug))

        assert test_idx > -1

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
        self.heat_shps = {}
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
            gt_edges = np.array(list(annot.values()))
            gt_edges += [pad_w_before, pad_h_before, pad_w_before, pad_h_before]

            # load predicted HEAT edges and pad them
            heat_f = os.path.join(data_path, "pred_full_2/%s.npy" % floor_name)
            heat_edges = np.load(heat_f)
            heat_edges -= 128
            heat_edges += [pad_w_before, pad_h_before, pad_w_before, pad_h_before]
            self.heat_edges[floor_name] = heat_edges

            # also convert HEAT edges to Shapely format, for ease later on
            heat_lines = [[(x0, y0), (x1, y1)] for (x0, y0, x1, y1) in heat_edges]
            heat_shps = [LineString(line) for line in heat_lines]
            self.heat_shps[floor_name] = heat_shps

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

            # now ready to generate examples
            eid2idx = dict(zip(gt_eids, range(len(gt_eids))))

            def order2idx(order_i):
                return eid2idx[order[order_i]]

            edge_ordering = [order2idx(order_i) for order_i in range(len(order))]
            ordered_edges = gt_edges[edge_ordering]
            self.ordered_edges[floor_name] = ordered_edges

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        floor_names = list(self.ordered_edges.keys())
        floor_name = np.random.choice(floor_names)

        all_edges = self.ordered_edges[floor_name]
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len + 1)
        rand_inds = np.random.choice(range(len(all_edges)), size=seq_len, replace=False)
        edge_coords = all_edges[rand_inds].copy()
        edge_coords, _ = normalize_edges(edge_coords)
        example = {"edge_coords": edge_coords}

        aug_param = {}
        if self.rand_aug:
            raise Exception
            edge_coords = example["edge_coords"]
            lines = [[(x0, y0), (x1, y1)] for (x0, y0, x1, y1) in edge_coords]
            lines = MultiLineString(lines)

            # rotation
            # angle = np.random.choice([0, 90, 180, 270])
            angle = np.random.choice(range(0, 360))
            lines = affinity.rotate(lines, angle, origin=(0, 0))
            aug_param["rotation"] = angle

            # translation
            xoff = np.random.choice(range(-100, 100))
            yoff = np.random.choice(range(-100, 100))
            lines = affinity.translate(lines, xoff=xoff, yoff=yoff)
            aug_param["translation"] = (xoff, yoff)

            # scaling
            xfact = np.random.choice(range(100, 150)) / 100
            yfact = np.random.choice(range(100, 150)) / 100
            lines = affinity.scale(lines, xfact=xfact, yfact=yfact, origin=(0, 0))
            aug_param["scale"] = (xfact, yfact)

            edge_coords = [list(line.coords) for line in lines.geoms]
            edge_coords = np.array(edge_coords).reshape(-1, 4)

            example["edge_coords"] = edge_coords

        # self.vis_example(example, aug_param, 0)

        example["edge_coords"] = example["edge_coords"].flatten().astype(np.float32)

        return example

    def vis_example(self, example, aug_param, start_idx, guess=False):
        # undo augmentation to edges
        floor_name = example["floor_name"]
        edge_coords = example["edge_coords"]
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

        max_order = edge_order.max()

        for edge_i, (x0, y0, x1, y1) in enumerate(edge_coords):
            if edge_order[edge_i] == max_order:
                ax1.plot([x0, x1], [y0, y1], "-", color="red", linewidth=4)
            else:
                ax1.plot([x0, x1], [y0, y1], "-oy")
                ax1.text(
                    (x0 + x1) / 2, (y0 + y1) / 2, str(edge_order[edge_i]), color="c"
                )

        # if not guess:
        #     for edge_i, (x0, y0, x1, y1) in enumerate(edge_coords_gt):
        #         ax2.plot([x0, x1], [y0, y1], "-oy")
        #         ax2.text((x0 + x1) / 2, (y0 + y1) / 2, str(edge_i), color="c")

        ax1.set_axis_off()
        ax2.set_axis_off()

        if not guess:
            if example["label"] == 0:
                ax1.set_title("Real")
            else:
                if example["damage_type"] == 1:
                    ax1.set_title("Fake (replace)")
                elif example["damage_type"] == 2:
                    ax1.set_title("Fake (shuffle)")
                elif example["damage_type"] == 3:
                    ax1.set_title("Fake (shift)")
                elif example["damage_type"] == 4:
                    ax1.set_title("Fake (HEAT replace)")
                else:
                    raise Exception("Unknown damage type")

        plt.tight_layout()

        if guess:
            save_f = "./vis/%s_%02d_%06d_0.png" % (
                floor_name,
                len(edge_coords),
                start_idx,
            )
        else:
            save_f = "./vis/%s_%02d_%06d_1.png" % (
                floor_name,
                len(edge_coords),
                start_idx,
            )

        plt.show()
        # plt.savefig(save_f, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    def unnormalize_edges(self, edges, normalize_param):
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

        elif ("_coords" in field) or ("_mask" in field) or ("_order" in field):
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
