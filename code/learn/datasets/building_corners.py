import glob
import itertools
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiLineString, box
from skimage.transform import SimilarityTransform, resize
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm

import my_utils
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

all_combibations = dict()
for length in range(2, 351):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combibations[length] = combs


class BuildingCornerDataset(Dataset):
    def __init__(
        self,
        data_path,
        det_path,
        phase="train",
        image_size=256,
        rand_aug=True,
        d_pe=128,
        training_split=None,
        inference=False,
        use_combined=False,
        test_idx=-1,
    ):
        super(BuildingCornerDataset, self).__init__()
        self.data_path = data_path
        self.det_path = det_path
        self.phase = phase
        self.d_pe = d_pe
        self.rand_aug = rand_aug
        self.image_size = image_size
        self.inference = inference
        self.use_combined = use_combined

        # assert image_size == 512
        assert not self.use_combined
        assert not training_split
        assert test_idx > -1

        # blur_transform = RandomBlur()
        # self.train_transform = transforms.Compose([
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #     transforms.RandomGrayscale(p=0.3),
        #     blur_transform])

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
            self.training = True
            assert len(floors) == 14
        elif phase == "valid":
            self.training = False
            floors = [val_floor]
        elif phase == "test":
            self.training = False
            floors = [test_floor]
        else:
            raise ValueError("Invalid phase {}".format(phase))

        print("%s: %s" % (phase, str(floors)))

        self.density_fulls = {}
        self.annots = {}
        self._data_names = []

        for (floor_name, _) in floors:
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

            # we need to square pad the image
            (h, w, _) = density_full.shape
            side_len = max(h, w)

            pad_h_before = 0
            pad_h_after = 0
            pad_w_before = 0
            pad_w_after = 0

            density_full = np.pad(
                density_full,
                [[pad_h_before, pad_h_after], [pad_w_before, pad_w_after], [0, 0]],
            )
            self.density_fulls[floor_name] = density_full

            # load GT annotation
            annot_f = os.path.join(
                data_path, "annot/%s_one_shot_full.json" % floor_name
            )
            with open(annot_f, "r") as f:
                annot = json.load(f)

            gt_coords = np.array(list(annot.values()))
            gt_coords += [pad_w_before, pad_h_before, pad_w_before, pad_h_before]
            # gt_coords = my_utils.revectorize(gt_coords)

            # convert the GT coordinates
            gt_corners, _ = my_utils.corners_and_edges(gt_coords)

            # compute crop coordinates
            bboxes = []
            (h, w, _) = density_full.shape
            self.c_padding = 16
            side_len = 256 - self.c_padding * 2
            stride = side_len // 8

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

            # for each crop, cache the edge coordinates
            for bbox_i, bbox in enumerate(bboxes):
                data_name = "%s_%03d" % (floor_name, bbox_i)
                (minx, miny, maxx, maxy) = bbox

                crop_corners = []
                for (x, y) in gt_corners:
                    if (minx < x < maxx) and (miny < y < maxy):
                        crop_corners.append([x, y])

                if not len(crop_corners):
                    continue
                crop_corners = np.array(crop_corners).copy()
                crop_corners -= np.array([minx, miny])
                crop_corners += np.array([self.c_padding, self.c_padding])
                self.annots[data_name] = (floor_name, bbox, crop_corners)

                # finally save this example
                # self.vis_example(rgb, crop_corners)
                self._data_names.append(data_name)

        print("Number of examples: %d" % len(self._data_names))

    def vis_example(self, rgb, annot):
        img = np.max(rgb, axis=2)
        plt.imshow(img, cmap="gray")
        plt.plot(annot[:, 0], annot[:, 1], "*c")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

    def __len__(self):
        return len(self._data_names)

    def __getitem__(self, idx):
        data_name = self._data_names[idx]
        (floor_name, bbox, corners) = self.annots[data_name]

        # crop the image
        (minx, miny, maxx, maxy) = bbox
        image = self.density_fulls[floor_name][miny:maxy, minx:maxx, :].copy()
        image = np.pad(
            image,
            [
                [self.c_padding, self.c_padding],
                [self.c_padding, self.c_padding],
                [0, 0],
            ],
        )
        assert image.shape == (self.image_size, self.image_size, 3)

        annot = None
        annot_path = ""
        det_path = ""
        det_corners = None
        img_path = ""
        rec_mat = None

        if self.rand_aug:
            image, corners = self.aug_example(image, corners)

        image = skimage.img_as_float(image)

        raw_data = {
            "name": data_name,
            "corners": corners,
            "annot": annot,
            "image": image,
            "rec_mat": rec_mat,
            "annot_path": annot_path,
            "det_path": det_path,
            "img_path": img_path,
        }

        return self.process_data(raw_data)

    def process_data(self, data):
        img = data["image"]
        corners = data["corners"]
        annot = data["annot"]
        rec_mat = data["rec_mat"]

        assert not self.use_combined
        mean = density_mean
        std = density_std

        # pre-process the image to use ImageNet-pretrained backbones
        img = img.transpose((2, 0, 1))
        raw_img = img.copy()
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[
            :, np.newaxis, np.newaxis
        ]
        img = img.astype(np.float32)

        corners = np.array(corners)

        # corner labels
        pixel_labels, gauss_labels = self.get_corner_labels(img.shape[1:], corners)

        # visualize corner labels
        # plt.imshow(raw_img.transpose(1, 2, 0), cmap="gray")
        # plt.imshow(gauss_labels, alpha=0.5)
        # plt.tight_layout()
        # plt.show()
        # plt.close()

        return {
            "pixel_labels": pixel_labels,
            "gauss_labels": gauss_labels,
            "annot": annot,
            "name": data["name"],
            "img": img,
            "raw_img": raw_img,
            "rec_mat": rec_mat,
            "annot_path": data["annot_path"],
            "det_path": data["det_path"],
            "img_path": data["img_path"],
        }

    # random rotate and scaling
    def aug_example(self, image, corners):
        # random rotate
        angle = np.random.choice([0, 90, 180, 270])
        k = -1 * angle / 90
        angle = np.deg2rad(angle)

        if len(corners):
            cx = cy = -1 * self.image_size / 2
            T_center = SimilarityTransform(translation=(cx, cy))
            T_rot = SimilarityTransform(rotation=angle)
            new_corners = T_center(corners)
            new_corners = T_rot(new_corners)
            new_corners = T_center.inverse(new_corners)
        else:
            new_corners = []

        image = np.rot90(image, k=k)

        # NOTE disabled
        # random scale
        # xfact = np.random.choice(range(75, 100)) / 100
        # yfact = np.random.choice(range(75, 100)) / 100

        # if len(new_corners):
        #     new_corners[:, 0] *= xfact
        #     new_corners[:, 1] *= yfact

        # (h, w, _) = image.shape
        # new_h = h * yfact
        # new_w = w * xfact
        # image = resize(image, (new_h, new_w))

        return image, new_corners

    def get_corner_labels(self, image_shape, corners):
        labels = np.zeros(image_shape)
        if len(corners):
            corners = corners.round()
            xint, yint = corners[:, 0].astype(np.int), corners[:, 1].astype(np.int)
            labels[yint, xint] = 1

        # pad the labels so that the edge doesn't get too high
        labels = np.pad(labels, [[30, 30], [30, 30]])

        gauss_labels = gaussian_filter(labels, sigma=2)
        if len(corners):
            gauss_labels = gauss_labels / gauss_labels.max()

        # remove padding now
        labels = labels[30:-30, 30:-30]
        gauss_labels = gauss_labels[30:-30, 30:-30]

        # # double-check Gaussian label gets us to the one-hot label
        # preds_s1 = (gauss_labels >= 0.5).astype(float)
        # pos_target_ids = np.where(labels == 1)
        # correct = (preds_s1[pos_target_ids] == labels[pos_target_ids]).astype(float).sum()
        # recall_s1 = correct / len(pos_target_ids[0])

        return labels, gauss_labels

    def resize_data(self, image, annot, det_corners):
        new_image = cv2.resize(image, (self.image_size, self.image_size))
        new_annot = {}
        r = self.image_size / 256
        for c, connections in annot.items():
            new_c = tuple(np.array(c) * r)
            new_connections = [other_c * r for other_c in connections]
            new_annot[new_c] = new_connections
        new_dets = det_corners * r
        return new_image, new_annot, new_dets

    def coin_flip(self, pos_prob=0.5):
        return np.random.choice([0, 1], p=[1 - pos_prob, pos_prob])

    def random_crop(self, img, annot):
        if self.coin_flip(pos_prob=0.2):
            return img, annot

        # remove padding that is there
        padding = 0  # NOTE was 8 when things were 512
        _img = img.copy()
        # img = img[padding:-padding, padding:-padding, :]

        # turn annotations into a list of edges
        edges = []

        for (a, bs) in annot.items():
            for b in bs:
                edges.append(a + b)  # (x, y) + (x, y) = (x, y, x, y)

        edges = np.array(edges) - padding
        assert (edges >= 0).all()

        # find a random crop view
        assert img.shape[0] == img.shape[1]
        side_len = img.shape[0]
        center = side_len // 2
        jitter = side_len // 8

        x = np.random.choice(range(center - jitter, center + jitter))
        y = np.random.choice(range(center - jitter, center + jitter))
        w = np.random.choice(range(center - jitter, center)) * 2
        h = np.random.choice(range(center - jitter, center)) * 2

        minx = max(x - w // 2, 0)
        maxx = min(minx + w, side_len)
        miny = max(y - h // 2, 0)
        maxy = min(miny + h, side_len)

        img_crop = img[miny:maxy, minx:maxx, :]
        new_h, new_w, _ = img_crop.shape

        # trim edges in view
        edges -= np.array([minx, miny, minx, miny])
        edges = np.maximum(edges, np.array([0, 0, 0, 0]))
        edges = np.minimum(edges, np.array([new_w, new_h, new_w, new_h]))

        # filter out walls out-of-bounds
        keep_edges = []
        for edge in edges:
            x0, y0, x1, y1 = edge

            if (x0 < EPS) and (x1 < EPS):
                continue
            elif (x0 > new_w - EPS) and (x1 > new_w - EPS):
                continue
            elif (y0 < EPS) and (y1 < EPS):
                continue
            elif (y0 > new_h - EPS) and (y1 > new_h - EPS):
                continue
            else:
                keep_edges.append(edge)

        # if we somehow cropped all edges out, then don't do augmentation
        if not len(keep_edges):
            return _img, annot
        keep_edges = np.array(keep_edges)

        # resize image and also edges back upto original resolution
        scale_h = img.shape[0] / img_crop.shape[0]
        scale_w = img.shape[1] / img_crop.shape[1]
        img_crop = resize(img_crop, img.shape)
        img_crop = (img_crop * 255.0).astype(np.uint8)
        keep_edges *= np.array([scale_w, scale_h, scale_w, scale_h])
        keep_edges = np.minimum(keep_edges, 255)

        # add back padding
        img_crop = np.pad(img_crop, [[padding, padding], [padding, padding], [0, 0]])
        keep_edges += padding
        assert img_crop.shape == (self.image_size, self.image_size, 3)
        assert keep_edges.max() <= 255

        # turn back into annot object
        new_annot = {}
        for (x0, y0, x1, y1) in keep_edges:
            a = (x0, y0)
            b = (x1, y1)

            if a not in new_annot.keys():
                new_annot[a] = []
            if b not in new_annot.keys():
                new_annot[b] = []

            new_annot[a].append(b)
            new_annot[b].append(a)

        # inspect
        if False:
            import matplotlib.pyplot as plt

            fig, [ax1, ax2] = plt.subplots(ncols=2)

            ax1.imshow(_img)

            xx = [minx, maxx, maxx, minx, minx]
            yy = [miny, miny, maxy, maxy, miny]
            ax1.plot(xx, yy, "-")

            for (a, bs) in annot.items():
                for b in bs:
                    x0, y0 = a
                    x1, y1 = b
                    xx = [x0, x1, x1, x0, x0]
                    yy = [y0, y0, y1, y1, y0]
                    ax1.plot(xx, yy, "-")

            ax2.imshow(img_crop)

            for (a, bs) in new_annot.items():
                for b in bs:
                    x0, y0 = a
                    x1, y1 = b
                    xx = [x0, x1, x1, x0, x0]
                    yy = [y0, y0, y1, y1, y0]
                    ax2.plot(xx, yy, "-")

            plt.show()
            plt.close()

        return img_crop, new_annot

    def my_random_flip(self, img, target):
        # horizontal flip
        if self.coin_flip():
            img = np.fliplr(img)
            target["boxes"][:, 0] = 1 - target["boxes"][:, 0]

        # vertical flip
        if self.coin_flip():
            img = torch.flipud(img)
            target["boxes"][:, 1] = 1 - target["boxes"][:, 1]

        img = img.permute([2, 0, 1])

        return img, target

    def verify_augmentation(self, img, target):
        boxes = box_cxcywh_to_xyxy(target["boxes"])

        plt.imshow(img.permute([1, 2, 0]), cmap="gray")

        for (x0, y0, x1, y1) in boxes * 256:
            xx = [x0, x1, x1, x0, x0]
            yy = [y0, y0, y1, y1, y0]
            plt.plot(xx, yy, "-")

        plt.show()
        plt.close()

    def random_aug_annot(self, img, annot, det_corners=None):
        # do random flipping
        img, annot, det_corners = self.random_flip(img, annot, det_corners)

        # prepare random augmentation parameters (only do random rotation for now)
        theta = np.random.randint(0, 360) / 360 * np.pi * 2
        r = self.image_size / 256
        origin = [127 * r, 127 * r]
        p1_new = [127 * r + 100 * np.sin(theta) * r, 127 * r - 100 * np.cos(theta) * r]
        p2_new = [127 * r + 100 * np.cos(theta) * r, 127 * r + 100 * np.sin(theta) * r]
        p1_old = [127 * r, 127 * r - 100 * r]  # y_axis
        p2_old = [127 * r + 100 * r, 127 * r]  # x_axis
        pts1 = np.array([origin, p1_old, p2_old]).astype(np.float32)
        pts2 = np.array([origin, p1_new, p2_new]).astype(np.float32)
        M_rot = cv2.getAffineTransform(pts1, pts2)

        # Combine annotation corners and detection corners
        all_corners = list(annot.keys())
        if det_corners is not None:
            for i in range(det_corners.shape[0]):
                all_corners.append(tuple(det_corners[i]))
        all_corners_ = np.array(all_corners)

        # Do the per-corner transform
        # Done in a big matrix transformation to save processing time.
        corner_mapping = dict()
        ones = np.ones([all_corners_.shape[0], 1])
        all_corners_ = np.concatenate([all_corners_, ones], axis=-1)
        aug_corners = np.matmul(M_rot, all_corners_.T).T

        for idx, corner in enumerate(all_corners):
            corner_mapping[corner] = aug_corners[idx]

        # If the transformed geometry goes beyond image boundary, we simply re-do the augmentation
        new_corners = np.array(list(corner_mapping.values()))
        if new_corners.min() <= 0 or new_corners.max() >= (self.image_size - 1):
            # return self.random_aug_annot(img, annot, det_corners)
            return img, annot, None, det_corners

        # build the new annot dict
        aug_annot = dict()
        for corner, connections in annot.items():
            new_corner = corner_mapping[corner]
            tuple_new_corner = tuple(new_corner)
            aug_annot[tuple_new_corner] = list()
            for to_corner in connections:
                aug_annot[tuple_new_corner].append(corner_mapping[tuple(to_corner)])

        # Also transform the image correspondingly
        rows, cols, ch = img.shape
        new_img = cv2.warpAffine(img, M_rot, (cols, rows), borderValue=(255, 255, 255))

        y_start = (new_img.shape[0] - self.image_size) // 2
        x_start = (new_img.shape[1] - self.image_size) // 2
        aug_img = new_img[
            y_start : y_start + self.image_size, x_start : x_start + self.image_size, :
        ]

        if det_corners is None:
            return aug_img, aug_annot, corner_mapping, None
        else:
            aug_det_corners = list()
            for corner in det_corners:
                new_corner = corner_mapping[tuple(corner)]
                aug_det_corners.append(new_corner)
            aug_det_corners = np.array(aug_det_corners)
            return aug_img, aug_annot, corner_mapping, aug_det_corners

    def random_flip(self, img, annot, det_corners):
        height, width, _ = img.shape
        rand_int = np.random.randint(0, 4)
        if rand_int == 0:
            return img, annot, det_corners

        all_corners = list(annot.keys())
        if det_corners is not None:
            for i in range(det_corners.shape[0]):
                all_corners.append(tuple(det_corners[i]))
        new_corners = np.array(all_corners)

        if rand_int == 1:
            img = img[:, ::-1, :]
            new_corners[:, 0] = width - new_corners[:, 0]
        elif rand_int == 2:
            img = img[::-1, :, :]
            new_corners[:, 1] = height - new_corners[:, 1]
        else:
            img = img[::-1, ::-1, :]
            new_corners[:, 0] = width - new_corners[:, 0]
            new_corners[:, 1] = height - new_corners[:, 1]

        new_corners = np.clip(new_corners, 0, self.image_size - 1)  # clip into [0, 255]
        corner_mapping = dict()
        for idx, corner in enumerate(all_corners):
            corner_mapping[corner] = new_corners[idx]

        aug_annot = dict()
        for corner, connections in annot.items():
            new_corner = corner_mapping[corner]
            tuple_new_corner = tuple(new_corner)
            aug_annot[tuple_new_corner] = list()
            for to_corner in connections:
                aug_annot[tuple_new_corner].append(corner_mapping[tuple(to_corner)])

        if det_corners is not None:
            aug_det_corners = list()
            for corner in det_corners:
                new_corner = corner_mapping[tuple(corner)]
                aug_det_corners.append(new_corner)
            det_corners = np.array(aug_det_corners)

        return img, aug_annot, det_corners


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
