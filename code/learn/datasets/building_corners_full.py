import glob
import itertools
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from shapely.geometry import LineString
from skimage.transform import SimilarityTransform, resize, rotate
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm

import my_utils
from metrics.get_metric import Metric, compute_metrics
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


class BuildingCornerDataset(Dataset):
    def __init__(
        self,
        data_path,
        phase="train",
        batch_size=64,
        rand_aug=True,
        test_idx=-1,
        multiplier=1,
        threshold=8,
        vis_labels=False,
    ):
        super(BuildingCornerDataset, self).__init__()
        self.data_path = data_path
        self.phase = phase
        self.rand_aug = rand_aug
        self.batch_size = batch_size
        self.multiplier = multiplier
        self.threshold = threshold

        all_combinations = dict()
        for length in range(2, 400):
            ids = np.arange(length)
            combs = np.array(list(itertools.combinations(ids, 2)))
            all_combinations[length] = combs

        print("GT matching threshold: %d" % threshold)

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
        self.gt_coords = {}
        self.gt_widths = {}
        self.gt_corners = {}
        self.pred_corners = {}
        self.aug_corners = {}
        self.pos_edges = {}
        self.neg_edges = {}
        self.pos_widths = {}

        for floor_name in floors:
            floor_name = floor_name[0]
            print(floor_name)

            # load full density image
            density_slices = []
            for slice_i in range(7):
                slice_f = "%s/density/%s/density_%02d.npy" % (
                    data_path,
                    floor_name,
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
            self.padding = [pad_w_before, pad_h_before, pad_w_after, pad_h_after]

            density_full = np.pad(
                density_full,
                [[pad_h_before, pad_h_after], [pad_w_before, pad_w_after], [0, 0]],
            )
            self.density_fulls[floor_name] = density_full

            # load GT annotation
            annot_f = os.path.join(data_path, "annot/%s.json" % floor_name)
            with open(annot_f, "r") as f:
                annot = json.load(f)

            annot = np.array(list(annot.values()))
            gt_coords, gt_widths = annot[:, :4], annot[:, 4]

            gt_coords += [pad_w_before, pad_h_before, pad_w_before, pad_h_before]
            gt_widths = np.floor(gt_widths * 12).astype(int)

            gt_coords, gt_widths = my_utils.revectorize(gt_coords, gt_widths)
            self.gt_coords[floor_name] = gt_coords
            self.gt_widths[floor_name] = gt_widths

            # temporarily visualize old and revectorized edges
            if False:
                image = density_full.max(axis=2)

                fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

                ax1.imshow(0.5 * image, cmap="gray", vmin=0, vmax=1)
                ax2.imshow(0.5 * image, cmap="gray", vmin=0, vmax=1)

                for x0, y0, x1, y1 in gt_coords:
                    color = np.random.choice(colors)
                    marker = np.random.choice(markers)
                    ax1.plot([x0, x1], [y0, y1], "-" + marker, color=color)

                for x0, y0, x1, y1 in new_gt_coords:
                    color = np.random.choice(colors)
                    marker = np.random.choice(markers)
                    ax2.plot([x0, x1], [y0, y1], "-" + marker, color=color)

                ax1.set_axis_off()
                ax2.set_axis_off()

                plt.tight_layout()
                plt.show()

            # load predicted corners
            corner_f = os.path.join(data_path, "pred_corners/%s.json" % floor_name)
            with open(corner_f, "r") as f:
                pred_corners = json.load(f)
            pred_corners = np.array(pred_corners)
            pred_corners += [pad_w_before, pad_h_before]
            self.pred_corners[floor_name] = pred_corners

            # augment GT corners with predicted ones
            gt_corners, gt_edge_ids = self.corners_and_edges(gt_coords)
            self.gt_corners[floor_name] = gt_corners
            aug_corners = gt_corners.copy()

            cost = distance.cdist(gt_corners, pred_corners)
            gt_ind, pred_ind = linear_sum_assignment(cost)

            valid_mask = cost[gt_ind, pred_ind] <= threshold
            gt_ind = gt_ind[valid_mask]
            pred_ind = pred_ind[valid_mask]
            aug_corners[gt_ind] = pred_corners[pred_ind]
            self.aug_corners[floor_name] = aug_corners

            # generate true and false edges
            all_edges = all_combinations[len(aug_corners)]

            if False:
                gt_data = {"corners": gt_corners, "edges": gt_edge_ids}
                pred_data = {"corners": pred_corners, "edges": all_edges}
                scores = compute_metrics(gt_data, pred_data, thresh=threshold)

                pos_edges = all_edges[scores["matched_pred_edge_ids"]]
                neg_edges = all_edges[scores["false_pred_edge_ids"]]

                self.pos_edges[floor_name] = pos_edges
                self.neg_edges[floor_name] = neg_edges
            else:
                aug_coords = aug_corners[all_edges].reshape(-1, 4)
                labels, matches = my_utils.compute_label(
                    pred_coords=aug_coords, gt_coords=gt_coords, threshold=threshold
                )

                # pos_edges = all_edges[labels == 1]
                # neg_edges = all_edges[labels == 0]
                # self.pos_edges[floor_name] = pos_edges
                # self.neg_edges[floor_name] = neg_edges

                neg_edges = all_edges[labels == 0]

                pos_edges = []
                pos_widths = []

                for pred_i, gt_i in matches:
                    pos_edges.append(all_edges[pred_i])
                    pos_widths.append(gt_widths[gt_i])

                self.pos_edges[floor_name] = pos_edges
                self.neg_edges[floor_name] = neg_edges
                self.pos_widths[floor_name] = pos_widths

                # self.vis_matches(density_full, pred_coords, gt_coords, matches)

            if False:
                self.vis_edges(gt_coords, density_full)

            # visualize corner results
            if False:
                metric = Metric()
                scores_corner = metric.calc_corner(gt_data, pred_data)
                self.vis_corner_results(
                    gt_corners, pred_corners, scores_corner, density_full
                )

            if vis_labels:
                example = {
                    "floor_name": floor_name,
                    "img": density_full.transpose(2, 0, 1),
                    "pred_corners": aug_corners,
                    "edge_coords": aug_corners[pos_edges].reshape(-1, 4),
                    "edge_labels": np.ones(len(pos_edges)),
                    "edge_widths": pos_widths,
                }
                self.vis_example(example)

            # metric = Metric()

            # gt_corners, gt_edge_ids = self.corners_and_edges(gt_coords)
            # gt_data = {"corners": gt_corners, "edges": gt_edge_ids}

            # all_edges = all_combinations[len(pred_corners)]
            # pred_data = {"corners": pred_corners, "edges": all_edges}

            # scores = metric.calc(gt_data, pred_data)

            # pos_edges = all_edges[scores["matched_pred_edge_ids"]]
            # neg_edges = all_edges[scores["false_pred_edge_ids"]]
            # assert len(pos_edges) + len(neg_edges) == len(all_edges)

            # visualize corners
            if False:
                plt.imshow(density_full[:, :, 1], cmap="gray")
                plt.plot(gt_corners[:, 0], gt_corners[:, 1], "og")
                plt.plot(pred_corners[:, 0], pred_corners[:, 1], "or")
                plt.axis("off")
                plt.savefig("test.png")
                plt.close()

    def __len__(self):
        return len(self.density_fulls) * self.multiplier

    def __getitem__(self, idx):
        floor_names = list(self.density_fulls.keys())
        floor_name = np.random.choice(floor_names)

        # depending on batch size, sample half positive and half negative
        all_pos_edges = self.pos_edges[floor_name]
        all_neg_edges = self.neg_edges[floor_name]

        # if self.phase == "train":
        #     num_pos = min(self.batch_size // 2, len(all_pos_edges))
        #     num_neg = self.batch_size - num_pos

        #     pos_inds = np.random.choice(
        #         range(len(all_pos_edges)), size=num_pos, replace=False
        #     )
        #     neg_inds = np.random.choice(
        #         range(len(all_neg_edges)), size=num_neg, replace=False
        #     )
        # else:
        #     pos_inds = list(range(len(all_pos_edges)))
        #     neg_inds = list(range(len(all_neg_edges)))

        pos_edges = all_pos_edges
        neg_edges = all_neg_edges
        edges = np.concatenate([pos_edges, neg_edges], axis=0)

        # prepare edge classification labels
        pos_e_labels = np.ones(len(pos_edges))
        neg_e_labels = np.zeros(len(neg_edges))
        edge_labels = np.concatenate([pos_e_labels, neg_e_labels], axis=0)

        # prepare width classification labels
        pos_w_labels = self.pos_widths[floor_name]
        neg_w_labels = np.zeros(len(neg_edges))
        width_labels = np.concatenate([pos_w_labels, neg_w_labels], axis=0)

        # normalization and random augmentation
        image = self.density_fulls[floor_name].copy()
        corners = self.aug_corners[floor_name].copy()

        image, corners = self.normalize_floor(image, corners)
        if self.rand_aug:
            image, corners = self.aug_example(image, corners)

        # preprocess image
        image = self.process_image(image)

        # prepare example dictionary
        edge_coords = corners[edges].reshape(-1, 4)

        example = {
            "floor_name": floor_name,
            "img": image,
            "edge_coords": edge_coords,
            "edge_labels": edge_labels,
            "width_labels": width_labels,
            "processed_corners_lengths": len(corners),
        }
        # self.vis_example(example)

        return example

    def normalize_floor(self, image, corners, max_side_len=1000):
        minx = corners[:, 0].min()
        miny = corners[:, 1].min()
        maxx = corners[:, 0].max()
        maxy = corners[:, 1].max()

        # normalize so longest edge is 1000, and to not change aspect ratio
        w = maxx - minx
        h = maxy - miny

        if max(w, h) > max_side_len:
            if w > h:
                scale = max_side_len / w
            else:
                scale = max_side_len / h
        else:
            scale = 1

        tform = SimilarityTransform(scale=scale)
        new_corners = tform(corners)

        # also scale the image the same way
        (h, w, _) = image.shape
        new_h = round(h * scale)
        new_w = round(w * scale)
        new_image = resize(image, (new_h, new_w))

        return new_image, new_corners

    def process_image(self, image):
        image = image.transpose((2, 0, 1))
        image -= np.array(density_mean)[:, np.newaxis, np.newaxis]
        image /= np.array(density_std)[:, np.newaxis, np.newaxis]
        image = image.astype(np.float32)

        return image

    # random rotate and scaling
    def aug_example(self, image, corners):
        # random rotate
        angle = np.random.choice([0, 90, 180, 270])
        k = -1 * angle / 90
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

        image = np.rot90(image, k=k)

        # random scale
        xfact = np.random.choice(range(75, 100)) / 100
        yfact = np.random.choice(range(75, 100)) / 100

        new_corners[:, 0] *= xfact
        new_corners[:, 1] *= yfact

        (h, w, _) = image.shape
        new_h = h * yfact
        new_w = w * xfact
        image = resize(image, (new_h, new_w))

        return image, new_corners

    def vis_edges(self, coords, image):
        image -= image.min()
        image /= image.max()
        plt.imshow(image[:, :, 1], cmap="gray")

        for x0, y0, x1, y1 in coords:
            plt.plot([x0, x1], [y0, y1], "-o")

        plt.axis("off")
        plt.show()
        plt.close()

    def vis_corner_results(self, gt_corners, pred_corners, scores, image):
        fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)
        # size = fig.get_size_inches()
        # fig.set_size_inches(size * 4)

        image -= image.min()
        image /= image.max()
        ax1.imshow(image[:, :, 1], cmap="gray")
        ax2.imshow(image[:, :, 1], cmap="gray")

        matched_pred = list(scores["c_det_annot"].keys())
        matched_gt = list(scores["c_det_annot"].values())

        for pred_i, (x, y) in enumerate(pred_corners):
            if pred_i in matched_pred:
                ax1.plot(x, y, "og")
            else:
                ax1.plot(x, y, "or")

        for gt_i, (x, y) in enumerate(gt_corners):
            if gt_i in matched_gt:
                ax2.plot(x, y, "og")
            else:
                ax2.plot(x, y, "or")

        plt.axis("off")
        plt.show()
        plt.close()

    def vis_example(self, example):
        fig, [ax1, ax2] = plt.subplots(ncols=2)
        size = fig.get_size_inches()
        fig.set_size_inches(size * 4)

        image = example["img"]
        image -= image.min()
        image /= image.max()
        ax1.imshow(image[1, :, :], cmap="gray")
        ax2.imshow(image[1, :, :], cmap="gray")

        for label, (x0, y0, x1, y1), width in zip(
            example["edge_labels"], example["edge_coords"], example["edge_widths"]
        ):
            if label == 1:
                line_shp = LineString([(x0, y0), (x1, y1)])
                box_shp = line_shp.buffer(width / 2, cap_style=2)
                (x, y) = box_shp.exterior.xy
                ax1.plot([x0, x1], [y0, y1], "--g")
                ax1.plot(x, y, "-g")
            # else:
            #     ax1.plot([x0, x1], [y0, y1], "-or", alpha=0.2)

        gt_coords = self.gt_coords[example["floor_name"]]
        gt_widths = self.gt_widths[example["floor_name"]]
        for (x0, y0, x1, y1), width in zip(gt_coords, gt_widths):
            line_shp = LineString([(x0, y0), (x1, y1)])
            box_shp = line_shp.buffer(width / 2, cap_style=2)
            (x, y) = box_shp.exterior.xy
            ax2.plot([x0, x1], [y0, y1], "--g")
            ax2.plot(x, y, "-g")

        plt.axis("off")
        plt.show()
        # plt.savefig(
        #     "vis/%s_%02d.png" % (example["floor_name"], self.threshold),
        #     bbox_inches="tight",
        #     pad_inches=0.1,
        # )
        plt.close()

    def vis_matches(self, image, pred_coords, gt_coords, matches):
        fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

        image -= image.min()
        image /= image.max()
        ax1.imshow(image[:, :, 1], cmap="gray")
        ax2.imshow(image[:, :, 1], cmap="gray")

        matches = np.array(matches)
        for match_i, (gt_i, pred_i) in enumerate(matches):
            (x0, y0, x1, y1) = pred_coords[pred_i]
            ax1.plot([x0, x1], [y0, y1], "-og")
            ax1.text((x0 + x1) / 2, (y0 + y1) / 2, str(match_i), color="c")

            (x0, y0, x1, y1) = gt_coords[gt_i]
            ax2.plot([x0, x1], [y0, y1], "-og")
            ax2.text((x0 + x1) / 2, (y0 + y1) / 2, str(match_i), color="c")

        # for pred_i, (x0, y0, x1, y1) in enumerate(pred_coords):
        #     if pred_i not in matches[:, 0]:
        #         ax1.plot([x0, x1], [y0, y1], "-og")

        for gt_i, (x0, y0, x1, y1) in enumerate(gt_coords):
            if gt_i not in matches[:, 1]:
                ax2.plot([x0, x1], [y0, y1], "-og")

        ax1.set_axis_off()
        ax2.set_axis_off()

        plt.show()
        plt.close()

    def corners_and_edges(self, edge_coords):
        corners = np.concatenate([edge_coords[:, :2], edge_coords[:, 2:]])
        corners = np.unique(corners, axis=0)

        edge_ids = set()
        for ax, ay, bx, by in edge_coords:
            assert (corners == np.array([ax, ay])).all(axis=1).sum() == 1
            a_idx = (corners == np.array([ax, ay])).all(axis=1).argmax()

            assert (corners == np.array([bx, by])).all(axis=1).sum() == 1
            b_idx = (corners == np.array([bx, by])).all(axis=1).argmax()

            # assert (a_idx, b_idx) not in edge_ids
            # assert (b_idx, a_idx) not in edge_ids
            if a_idx < b_idx:
                edge_ids.add((a_idx, b_idx))
            else:
                edge_ids.add((b_idx, a_idx))

        edge_ids = np.array(list(edge_ids))

        return corners, edge_ids


def collate_fn_corner(data):
    batched_data = {}
    for field in data[0].keys():
        if field in ["annot", "rec_mat"]:
            batch_values = [item[field] for item in data]
        else:
            batch_values = default_collate([d[field] for d in data])
        if field in ["pixel_features", "pixel_labels", "gauss_labels"]:
            batch_values = batch_values.float()
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

        if field in ["floor_name", "normalize_param", "aug_param"]:
            batched_data[field] = batch_values

        elif ("_coords" in field) or ("_mask" in field) or ("_labels" in field):
            all_lens = [len(value) for value in batch_values]
            max_len = max(all_lens)
            pad_value = 0

            batch_values = [
                pad_sequence(value, max_len, pad_value) for value in batch_values
            ]
            batch_values = np.stack(batch_values, axis=0)
            batch_values = torch.Tensor(batch_values)

            if "_labels" in field:
                batch_values = batch_values.long()

            if "_coords" in field:
                lengths_info[field] = all_lens

            batched_data[field] = batch_values

        else:
            batched_data[field] = default_collate(batch_values)

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
