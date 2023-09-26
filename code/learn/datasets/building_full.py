import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import skimage
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm
from utils.nn_utils import positional_encoding_2d
from torchvision import transforms
from PIL import Image
from skimage.transform import resize
import itertools
from rtree import index
from torch.utils.data.dataloader import default_collate
from shapely.geometry import LineString, box

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

EPS = 1e-6

# for 512 dual big dataset
density_mean = [0.18115416, 0.18115416, 0.18115416]
density_std = [0.27998772, 0.27998772, 0.27998772]

# combined
# combined_mean = [0.06896243, 0.06896243, 0.06896243]
# combined_std = [0.16101032, 0.16101032, 0.16101032]

all_combinations = dict()
for length in range(2, 400):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combinations[length] = combs


class BuildingCornerDataset(Dataset):
    def __init__(self, data_path, revit_root, phase='train', image_size=256,
                 rand_aug=True, d_pe=128, training_split=None,
                 inference=False, use_combined=False, test_idx=-1):
        super(BuildingCornerDataset, self).__init__()
        self.data_path = data_path
        self.revit_root = revit_root
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

        floor_f = os.path.join(data_path, 'all_floors.txt')
        with open(floor_f, 'r') as f:
            floors = [x.strip().split(',') for x in f.readlines()]

        # remove the testing floor right now
        test_floor = floors[test_idx]
        del floors[test_idx]

        # find the index to the smallest floor, used for validation
        fewest_idx = -1
        fewest_views = float('inf')

        for floor_idx, (floor, num_views) in enumerate(floors):
          if int(num_views) < fewest_views:
            fewest_idx = floor_idx
            fewest_views = int(num_views)

        assert fewest_idx > -1
        val_floor = floors[fewest_idx]
        del floors[fewest_idx]

        # prepare splits
        if phase == 'train':
          self.training = True
          assert len(floors) == 15
        elif phase == 'valid':
          self.training = False
          floors = [val_floor,]
        elif phase == 'test':
          self.training = False
          floors = [test_floor,]
        else:
            raise ValueError('Invalid phase {}'.format(phase))

        print('%s: %s' % (phase, str(floors)))

        # for each floor
        self.density_fulls = {}
        self.annots = {}
        self.all_bboxes = {}
        self.examples = []

        for floor_name in tqdm(floors):
            floor_name = floor_name[0]

            # load full density image
            tokens = floor_name.split('_')
            first = '_'.join(tokens[:-1])
            second = tokens[-1]

            density_f = revit_root + '%s/%s/density.npy' % (first, second)
            density_full = np.load(density_f)
            density_full /= density_full.max()

            # drop bottom and top 5 percentile for density map
            counts = sorted(density_full[density_full > 0])
            lower = np.percentile(counts, q=10)
            upper = np.percentile(counts, q=90)

            density_full = np.maximum(density_full, lower)
            density_full = np.minimum(density_full, upper)
            density_full -= lower
            density_full /= (upper - lower)

            # pad so we can detect outside edges
            padding = 128
            density_full = np.pad(density_full, [[padding, padding], [padding, padding]])
            self.density_fulls[floor_name] = density_full

            # determine the bbox
            side_len = self.image_size
            stride = self.image_size // 2

            (h, w) = density_full.shape

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

                    bbox = (minx, miny, maxx, maxy)
                    if (not len(bboxes)) or (bboxes[-1] != bbox):
                        bboxes.append(bbox)

            self.all_bboxes[floor_name] = np.array(bboxes)

            # load annotation
            annot_f = data_path + 'annot_full/%s_one_shot_full.npy' % floor_name
            annot = np.load(annot_f, allow_pickle=True).item()

            # pad annotations
            new_annots = {}
            for (a, bs) in annot.items():
                new_a = (a[0] + padding, a[1] + padding)
                new_annots[new_a] = []

                for b in bs:
                    new_b = (b[0] + padding, b[1] + padding)
                    new_annots[new_a].append(new_b)

            annot = new_annots
            self.annots[floor_name] = annot

            # enumerate edges
            gt_corners = np.array(list(annot.keys()))
            all_ids = all_combinations[len(gt_corners)]
            all_edges = gt_corners[all_ids].reshape(-1, 4)

            # orient edges
            for i in range(len(all_edges)):
                x0, y0, x1, y1 = all_edges[i]

                if x0 > x1:
                    all_edges[i] = [x1, y1, x0, y0]

                elif x0 == x1:
                    if y0 > y1:
                        all_edges[i] = [x1, y1, x0, y0]

            # add examples
            for edge in all_edges:
                self.examples.append((floor_name, edge))

            break


    def __len__(self):
        return len(self.examples)


    def get_edge_bbox(self, edge_coord):
        (x0, y0, x1, y1) = edge_coord

        c_x = (x0 + x1) / 2
        c_y = (y0 + y1) / 2
        w = max(abs(x1-x0), 4)
        h = max(abs(y1-y0), 4)

        l = c_x - w/2
        r = c_x + w/2
        b = c_y - h/2
        t = c_y + h/2

        return (l, b, r, t)


    def __getitem__(self, idx):
        floor_name, edge = self.examples[idx]

        density_full = self.density_fulls[floor_name]
        annot = self.annots[floor_name]

        # determine if this is a good edge
        (x0, y0, x1, y1) = edge
        assert ((x0, y0) in annot.keys()) and ((x1, y1) in annot.keys())

        if (x1, y1) in annot[(x0, y0)]:
            assert (x0, y0) in annot[(x1, y1)]
            label = 1
        else:
            assert (x0, y0) not in annot[(x1, y1)]
            label = 0

        # determine which crops intersect with this edge
        bboxes = self.all_bboxes[floor_name]
        edge_shp = LineString([(x0, y0), (x1, y1)])

        intersect_bboxes = []
        for (minx, miny, maxx, maxy) in bboxes:
            bbox_shp = box(minx, miny, maxx, maxy)
            if bbox_shp.intersects(edge_shp):
                intersect_bboxes.append((minx, miny, maxx, maxy))

        if False:
            plt.imshow(density_full, cmap='gray')

            (x0, y0, x1, y1) = edge
            plt.plot([x0, x1], [y0, y1], '-or')

            for (minx, miny, maxx, maxy) in intersect_bboxes:
                xx = [minx, maxx, maxx, minx, minx]
                yy = [miny, miny, maxy, maxy, miny]
                plt.plot(xx, yy, '--c')

            gt_corners = np.array(list(annot.keys()))
            plt.plot(gt_corners[:,0], gt_corners[:,1], 'or')

            plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.close()

        # for each crop, crop the line in the bbox
        crops = []

        for (minx, miny, maxx, maxy) in intersect_bboxes:
            # get the crop image
            density_crop = density_full[miny:maxy, minx:maxx]

            # get the cropped line coordinates
            bbox_shp = box(minx, miny, maxx, maxy)
            intersection = bbox_shp.intersection(edge_shp)
            assert intersection.length
            ((x_0, y_0), (x_1, y_1)) = intersection.coords

            x_0 = min(x_0 - minx, self.image_size-1)
            x_1 = min(x_1 - minx, self.image_size-1)
            y_0 = min(y_0 - miny, self.image_size-1)
            y_1 = min(y_1 - miny, self.image_size-1)

            coords = np.array([x_0, y_0, x_1, y_1]).astype(int)
            assert (coords.min() >= 0) and (coords.max() < self.image_size)

            crop = {
                'image': self.process_image(density_crop),
                'coords': coords
            }
            crops.append(crop)

            if False:
                plt.imshow(density_crop, cmap='gray')
                plt.plot([x_0, x_1], [y_0, y_1], '-or')
                plt.tight_layout()
                plt.show()
                plt.close()

        example = {
            'edge': np.array(edge).astype(int),
            'label': label,
            'crops': crops
        }
        return example

        data_name = self._data_names[idx]
        annot = self.annots[data_name]
        annot_path = os.path.join(self.data_path, 'annot', data_name + '_one_shot.npy')
        # annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()

        # need to swap xy...
        # _annot = {}

        # for key, values in annot.items():
        #   _key = (float(key[1]), float(key[0]))
        #   _annot[_key] = []
        #   for value in values:
        #     _annot[_key].append((float(value[1]), float(value[0])))

        # annot = _annot

        assert not self.det_path
        if self.det_path:
          det_path = os.path.join(self.det_path, data_name + '.npy')
          det_corners = np.array(np.load(det_path, allow_pickle=True))  # [N, 2]
          det_corners = det_corners[:, ::-1]  # turn into x,y format
        else:
          det_path = ''
          det_corners = None

        assert not self.use_combined
        if self.use_combined:
            img_path = os.path.join(self.data_path, 'rgb', data_name + '_combined.png')
        else:
            img_path = os.path.join(self.data_path, 'rgb', data_name + '.png')

        # rgb = np.array(Image.open(img_path))
        # rgb = cv2.imread(img_path)
        rgb = self.rgbs[data_name]

        # need to downscale the image and annotation
        assert rgb.shape == (512, 512, 3)
        rgb = cv2.resize(rgb, (256, 256))

        _annot = {}
        for (a, bs) in annot.items():
          a_x, a_y = a[0] // 2, a[1] // 2
          if (a_x, a_y) not in _annot.keys():
              _annot[(a_x, a_y)] = []

          for b in bs:
            b_x, b_y = b[0] // 2, b[1] // 2
            if (b_x, b_y) not in _annot[(a_x, a_y)]:
                _annot[(a_x, a_y)].append((b_x, b_y))

          assert len(_annot[(a_x, a_y)])

        annot = _annot

        # if self.image_size != 256:
        #     rgb, annot, det_corners = self.resize_data(rgb, annot, det_corners)

        if self.rand_aug:
            image, annot, corner_mapping, det_corners = self.random_aug_annot(rgb, annot, det_corners=det_corners)
            # rgb, annot = self.random_crop(rgb, annot)
            # rgb, annot, det_corners = self.random_flip(rgb, annot, det_corners)
            # image = rgb
        else:
            image = rgb

        ###
        if False:
          print(annot_path)

          plt.imshow(image)
          for (a, bs) in annot.items():
            for b in bs:
              plt.plot([a[0], b[0]], [a[1], b[1]], 'o-')

          plt.show()
          plt.close()
        ###

        rec_mat = None

        corners = np.array(list(annot.keys()))[:, [1, 0]]

        if not self.inference and len(corners) > 100:
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        if self.training:
            # Add some randomness for g.t. corners
            corners += np.random.normal(0, 0, size=corners.shape)
            pil_img = Image.fromarray(image)
            # image = self.train_transform(pil_img)
            image = np.array(image)
        image = skimage.img_as_float(image)

        # sort by the second value and then the first value, here the corners are in the format of (y, x)
        sort_idx = np.lexsort(corners.T)
        corners = corners[sort_idx]

        corner_list = []
        for corner_i in range(corners.shape[0]):
            corner_list.append((corners[corner_i][1], corners[corner_i][0]))  # to (x, y) format

        raw_data = {
            'name': data_name,
            'corners': corner_list,
            'annot': annot,
            'image': image,
            'rec_mat': rec_mat,
            'annot_path': annot_path,
            'det_path': det_path,
            'img_path': img_path,
        }

        return self.process_data(raw_data)

    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.transpose((2, 0, 1))
        img -= np.array(density_mean)[:, np.newaxis, np.newaxis]
        img /= np.array(density_std)[:, np.newaxis, np.newaxis]
        img = img.astype(np.float32)

        return img

    def process_data(self, data):
        img = data['image']
        corners = data['corners']
        annot = data['annot']
        rec_mat = data['rec_mat']

        if self.use_combined:
          mean = combined_mean
          std = combined_std
        else:
          mean = density_mean
          std = density_std

        # pre-process the image to use ImageNet-pretrained backbones
        img = img.transpose((2, 0, 1))
        raw_img = img.copy()
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = img.astype(np.float32)

        corners = np.array(corners)

        # corner labels
        pixel_labels, gauss_labels = self.get_corner_labels(corners)

        return {
            'pixel_labels': pixel_labels,
            'gauss_labels': gauss_labels,
            "annot": annot,
            "name": data['name'],
            'img': img,
            'raw_img': raw_img,
            'rec_mat': rec_mat,
            'annot_path': data['annot_path'],
            'det_path': data['det_path'],
            'img_path': data['img_path'],
        }

    def get_corner_labels(self, corners):
        labels = np.zeros((self.image_size, self.image_size))
        corners = corners.round()
        xint, yint = corners[:, 0].astype(np.int), corners[:, 1].astype(np.int)
        labels[yint, xint] = 1

        # pad the labels so that the edge doesn't get too high
        labels = np.pad(labels, [[30, 30], [30, 30]])

        gauss_labels = gaussian_filter(labels, sigma=2)
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
        return np.random.choice([0,1], p=[1-pos_prob, pos_prob])

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

        x = np.random.choice(range(center-jitter, center+jitter))
        y = np.random.choice(range(center-jitter, center+jitter))
        w = np.random.choice(range(center-jitter, center)) * 2
        h = np.random.choice(range(center-jitter, center)) * 2

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

          if ((x0 < EPS) and (x1 < EPS)):
            continue
          elif ((x0 > new_w-EPS) and (x1 > new_w-EPS)):
            continue
          elif ((y0 < EPS) and (y1 < EPS)):
            continue
          elif ((y0 > new_h-EPS) and (y1 > new_h-EPS)):
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
        img_crop = (img_crop * 255.).astype(np.uint8)
        keep_edges *= np.array([scale_w, scale_h, scale_w, scale_h])
        keep_edges = np.minimum(keep_edges, 255)

        # add back padding
        img_crop = np.pad(img_crop, [[padding,padding], [padding,padding], [0,0]])
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
            fig, [ax1, ax2] = plt.subplots(ncols=2)

            ax1.imshow(_img)

            xx = [minx, maxx, maxx, minx, minx]
            yy = [miny, miny, maxy, maxy, miny]
            ax1.plot(xx, yy, '-')

            for (a, bs) in annot.items():
              for b in bs:
                x0, y0 = a
                x1, y1 = b
                xx = [x0, x1, x1, x0, x0]
                yy = [y0, y0, y1, y1, y0]
                ax1.plot(xx, yy, '-')

            ax2.imshow(img_crop)

            for (a, bs) in new_annot.items():
              for b in bs:
                x0, y0 = a
                x1, y1 = b
                xx = [x0, x1, x1, x0, x0]
                yy = [y0, y0, y1, y1, y0]
                ax2.plot(xx, yy, '-')

            plt.show()
            plt.close()

        return img_crop, new_annot


    def my_random_flip(self, img, target):
        # horizontal flip
        if self.coin_flip():
            img = np.fliplr(img)
            target['boxes'][:,0] = 1 - target['boxes'][:,0]

        # vertical flip
        if self.coin_flip():
            img = torch.flipud(img)
            target['boxes'][:,1] = 1 - target['boxes'][:,1]

        img = img.permute([2,0,1])

        return img, target


    def verify_augmentation(self, img, target):
        boxes = box_cxcywh_to_xyxy(target['boxes'])

        plt.imshow(img.permute([1,2,0]), cmap='gray')

        for (x0, y0, x1, y1) in boxes * 256:
            xx = [x0, x1, x1, x0, x0]
            yy = [y0, y0, y1, y1, y0]
            plt.plot(xx, yy, '-')

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
        aug_img = new_img[y_start:y_start + self.image_size, x_start:x_start + self.image_size, :]

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
        if field == 'crops':
            batch_values = [item[field] for item in data]
        else:
            batch_values = default_collate([d[field] for d in data])
        if field in ['edge', 'label']:
            batch_values = batch_values.long()
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATAPATH = './data/cities_dataset'
    DET_PATH = './data/det_final'
    train_dataset = BuildingCornerDataset(DATAPATH, DET_PATH, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                                  collate_fn=collate_fn_corner)
    for i, item in enumerate(train_dataloader):
        import pdb;

        pdb.set_trace()
        print(item)
