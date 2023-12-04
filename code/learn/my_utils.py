import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters
import torch
from rtree import index
from scipy.spatial import distance
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, box
from shapely.ops import nearest_points, split
from skimage.transform import SimilarityTransform, resize
from torch.utils.data.dataloader import default_collate

all_combinations = dict()
for length in range(2, 500):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combinations[length] = combs

all_triplets = dict()
for length in range(2, 100):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 3)))
    all_triplets[length] = combs


density_mean = [0.18115416, 0.18115416, 0.18115416]
density_std = [0.27998772, 0.27998772, 0.27998772]


def find_intersects(all_shps, bbox):
    (minx, miny, maxx, maxy) = bbox
    bbox_shp = box(minx, miny, maxx, maxy)
    intersects = bbox_shp.intersection(all_shps)

    coords = []

    if type(intersects) == MultiLineString:
        for intersect in intersects.geoms:
            if type(intersect) == LineString:
                coords.append(np.array(intersect.coords))

    elif type(intersects) == LineString:
        coords.append(np.array(intersects.coords))

    else:
        return []

    coords = np.array(coords).copy().reshape(-1, 4)
    coords -= np.array([minx, miny, minx, miny])

    return coords


def vis_edges_idx(density_full, coords):
    density_img = np.max(density_full, axis=2)
    plt.imshow(density_img, cmap="gray")

    for idx, coord in enumerate(coords):
        (x0, y0, x1, y1) = coord
        plt.plot([x0, x1], [y0, y1], "-og")
        plt.text((x0 + x1) / 2, (y0 + y1) / 2, "%d" % idx, color="c")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def vis_edges_width(density_full, color_coords):
    density_img = np.max(density_full, axis=2)
    plt.imshow(density_img, cmap="gray")

    for flags, coords, widths in color_coords:
        for coord, width in zip(coords, widths):
            assert len(coord) == 4

            (x0, y0, x1, y1) = coord
            line_shp = LineString([(x0, y0), (x1, y1)])
            box_shp = line_shp.buffer(width / 2, cap_style=2)
            (x, y) = box_shp.exterior.xy
            plt.plot(x, y, flags)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def revectorize(coords, widths):
    corners = np.unique(coords.reshape(-1, 2), axis=0)
    corners = MultiPoint(corners)

    coords = coords.reshape(-1, 2, 2)

    new_coords = []
    new_widths = []
    for coord, width in zip(coords, widths):
        _coord = LineString(coord.tolist())

        new_coord = split(_coord, corners)
        for edge in new_coord.geoms:
            new_coords.append(np.array(edge.coords))
            new_widths.append(width)

    new_coords = np.array(new_coords).reshape(-1, 4)
    new_widths = np.array(new_widths)

    return new_coords, new_widths


def normalize_vec(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


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


def metric_normalize_edges(edges, normalize_param=None):
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


def metric_collate_fn(data):
    batched_data = {}
    lengths_info = {}

    for field in data[0].keys():
        batch_values = [example[field] for example in data]

        if field in ["floor_name", "normalize_param"]:
            batched_data[field] = batch_values

        elif (
            ("_coords" in field)
            or ("_mask" in field)
            or ("_order" in field)
            or ("label" in field)
        ):
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


def vis_edges(density_full, color_coords, title="", save_f=None):
    density_img = np.max(density_full, axis=2)
    plt.imshow(density_img, cmap="gray")

    for flags, coords in color_coords:
        for coord in coords:
            if len(coord) == 4:
                (x0, y0, x1, y1) = coord
                plt.plot([x0, x1], [y0, y1], flags)
            elif len(coord) == 2:
                (x0, y0) = coord
                plt.plot(x0, y0, flags)
            else:
                raise Exception("Unknown type of coordinate")

    plt.axis("off")

    (ymin, ymax) = plt.ylim()
    plt.ylim((ymax, ymin))

    if title:
        plt.title(title)
    plt.tight_layout()

    if save_f:
        fig = plt.gcf()
        size = fig.get_size_inches()
        fig.set_size_inches(size * 2)
        plt.savefig(save_f, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.gca().set_aspect("equal")
        plt.show()


def corner_nms(preds, confs, image_shape):
    data = np.zeros(image_shape)
    neighborhood_size = 5
    threshold = 0

    for i in range(len(preds)):
        data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = data == data_max
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = list()
    for i, pred in enumerate(filtered_preds):
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)

    return filtered_preds, new_confs


def snap_to_axis(all_edges, threshold=10):
    corners, edges = corners_and_edges(all_edges)

    for a, b in edges:
        edge = corners[[a, b]].reshape(4)
        (ax, ay, bx, by) = edge

        # this edge is already aligned, so ignore
        if (ax == bx) or (ay == by):
            continue

        # compute angle closest to axis
        angle2x = get_angle(ab=edge, cd=[0, 0, 1, 0])
        angle2y = get_angle(ab=edge, cd=[0, 0, 0, 1])

        close_to_x = ~(threshold < angle2x < (180 - threshold))
        close_to_y = ~(threshold < angle2y < (180 - threshold))
        assert not (close_to_x and close_to_y)

        # only rotate edges if we're close enough to one of the axis
        if close_to_x:
            ay = by = round((ay + by) / 2)
        elif close_to_y:
            ax = bx = round((ax + bx) / 2)

        corners[a] = [ax, ay]
        corners[b] = [bx, by]

    return corners[edges].reshape(-1, 4)


def remove_short_edges(all_edges):
    # remove edges that are shorter but close to others
    while True:
        simplified_edges = []
        skipped_mask = np.zeros(len(all_edges), dtype=bool)

        # build a database and shapely lines for faster lookup
        db = index.Index()
        edge_shps = []

        for edge_i, edge_coord in enumerate(all_edges):
            db.insert(edge_i, get_edge_bbox(edge_coord))
            (x0, y0, x1, y1) = edge_coord
            edge_shps.append(LineString([(x0, y0), (x1, y1)]))

        # go from shortest to longest edge
        edge_lengths = np.array([edge.length for edge in edge_shps])
        iter_order = np.argsort(edge_lengths)

        for edge_i in iter_order:
            edge_shp = edge_shps[edge_i]
            edge_coord = all_edges[edge_i]
            cand_ids = db.intersection(get_edge_bbox(edge_coord))
            skip = False

            for cand_i in cand_ids:
                if (edge_i == cand_i) or skipped_mask[cand_i]:
                    continue

                # ignore this candidate if it is shorter
                cand_shp = edge_shps[cand_i]
                if cand_shp.length < edge_shp.length:
                    continue

                # ignore this candidate if angle between is too large
                if not angle_matches(edge_coord, all_edges[cand_i]):
                    continue

                # check each point of the edge, if they are close to the candidate
                ((ax, ay), (bx, by)) = list(edge_shp.coords)
                a = Point(ax, ay)
                b = Point(bx, by)
                a_dist = LineString(nearest_points(a, cand_shp)).length
                b_dist = LineString(nearest_points(b, cand_shp)).length

                if (a_dist < 5) and (b_dist < 5):
                    skip = True
                    break

            if skip:
                skipped_mask[edge_i] = True
            else:
                simplified_edges.append(edge_coord)

        if len(simplified_edges) == len(all_edges):
            print("No more actions to do")
            all_edges = simplified_edges
            break
        else:
            print("%d -> %d" % (len(all_edges), len(simplified_edges)))
            all_edges = simplified_edges

    return all_edges


def find_candidates_fast(
    curr_seq, heat_edges, heat_widths, return_mask=False, threshold=10
):
    # cache shapely lines and build a database for faster lookup
    db = index.Index()
    curr_shps = []

    for curr_i, curr_coord in enumerate(curr_seq):
        db.insert(curr_i, get_edge_bbox(curr_coord))
        (x0, y0, x1, y1) = curr_coord
        curr_shps.append(LineString([(x0, y0), (x1, y1)]))

    # now find candidates
    cand_edges = []
    cand_widths = []
    keep_mask = np.zeros(len(heat_edges), dtype=bool)

    for heat_i, heat_coord in enumerate(heat_edges):
        (ax, ay, bx, by) = heat_coord
        heat_shp = LineString([(ax, ay), (bx, by)])
        keep = True

        cand_ids = db.intersection(get_edge_bbox(heat_coord))

        for cand_i in cand_ids:
            curr_coord = curr_seq[cand_i]
            (cx, cy, dx, dy) = curr_coord
            curr_shp = curr_shps[cand_i]

            if not angle_matches(heat_coord, curr_coord):
                continue

            # check each point of the edge, if they are close to the candidate
            a = Point(ax, ay)
            b = Point(bx, by)
            a_dist = LineString(nearest_points(a, curr_shp)).length
            b_dist = LineString(nearest_points(b, curr_shp)).length

            c = Point(cx, cy)
            d = Point(dx, dy)
            c_dist = LineString(nearest_points(c, heat_shp)).length
            d_dist = LineString(nearest_points(d, heat_shp)).length

            if (a_dist < threshold) and (b_dist < threshold):
                keep = False
                break

            if (c_dist < threshold) and (d_dist < threshold):
                keep = False
                break

        if keep:
            cand_edges.append(heat_coord)
            cand_widths.append(heat_widths[heat_i])
            keep_mask[heat_i] = True

    if return_mask:
        return keep_mask
    else:
        return np.array(cand_edges), np.array(cand_widths)


def find_candidates(curr_seq, heat_edges, threshold=5):
    cand_edges = []

    for edge_coord in heat_edges:
        (ax, ay, bx, by) = edge_coord
        edge_shp = LineString([(ax, ay), (bx, by)])
        keep = True

        for curr_coord in curr_seq:
            (cx, cy, dx, dy) = curr_coord
            curr_shp = LineString([(cx, cy), (dx, dy)])

            if not angle_matches(edge_coord, curr_coord):
                continue

            # check each point of the edge, if they are close to the candidate
            a = Point(ax, ay)
            b = Point(bx, by)
            a_dist = LineString(nearest_points(a, curr_shp)).length
            b_dist = LineString(nearest_points(b, curr_shp)).length

            c = Point(cx, cy)
            d = Point(dx, dy)
            c_dist = LineString(nearest_points(c, edge_shp)).length
            d_dist = LineString(nearest_points(d, edge_shp)).length

            if (a_dist < threshold) and (b_dist < threshold):
                keep = False
                break

            if (c_dist < threshold) and (d_dist < threshold):
                keep = False
                break

        if keep:
            cand_edges.append(edge_coord)

    return np.array(cand_edges)


def remove_overlaps(all_edges, all_widths, threshold=10):
    good_edges = []
    good_widths = []

    # build a database and shapely lines for faster lookup
    db = index.Index()
    edge_shps = []

    for edge_i, edge_coord in enumerate(all_edges):
        db.insert(edge_i, get_edge_bbox(edge_coord))
        (x0, y0, x1, y1) = edge_coord
        edge_shps.append(LineString([(x0, y0), (x1, y1)]))

    # go from longest to shortest edge
    edge_lengths = np.array([edge.length for edge in edge_shps])
    iter_order = np.argsort(edge_lengths)[::-1]
    keep_mask = np.ones(len(all_edges), dtype=bool)

    for edge_i in iter_order:
        if not keep_mask[edge_i]:
            continue

        edge_shp = edge_shps[edge_i]
        edge_coord = all_edges[edge_i]
        cand_ids = list(db.intersection(get_edge_bbox(edge_coord)))

        if edge_shp.length < threshold:
            keep_mask[edge_i] = False
            continue

        for cand_i in cand_ids:
            if cand_i == edge_i:
                continue

            # ignore this candidate if angle between is too large
            cand_shp = edge_shps[cand_i]
            cand_coord = all_edges[cand_i]
            if not angle_matches(edge_coord, cand_coord):
                continue
            if cand_shp.length < threshold:
                continue
            if cand_shp.length > edge_shp.length:
                continue

            # check each point of the edge, if they are close to the candidate
            ((ax, ay), (bx, by)) = list(cand_shp.coords)
            a = Point(ax, ay)
            b = Point(bx, by)
            a_dist = LineString(nearest_points(a, edge_shp)).length
            b_dist = LineString(nearest_points(b, edge_shp)).length

            if (a_dist < threshold) and (b_dist < threshold):
                keep_mask[edge_i] = False
                break

        # if we do keep, then remove bad next edges
        # if keep_mask[edge_i]:
        #     for cand_i in cand_ids:
        #         if cand_i == edge_i:
        #             continue

        #         # ignore this candidate if angle between is too large
        #         cand_shp = edge_shps[cand_i]
        #         cand_coord = all_edges[cand_i]
        #         if not angle_matches(edge_coord, cand_coord):
        #             continue

        #         # check each point of the edge, if they are close to the candidate
        #         ((ax, ay), (bx, by)) = list(cand_shp.coords)
        #         a = Point(ax, ay)
        #         b = Point(bx, by)
        #         a_dist = LineString(nearest_points(a, edge_shp)).length
        #         b_dist = LineString(nearest_points(b, edge_shp)).length

        #         if (a_dist < threshold) and (b_dist < threshold):
        #             keep_mask[cand_i] = False

    good_edges = all_edges[keep_mask]
    good_widths = all_widths[keep_mask]

    return good_edges, good_widths


def merge_edges(all_edges, threshold=5):
    # actually merge edges
    while True:
        merged_edges = []
        used_mask = np.zeros(len(all_edges), dtype=bool)

        # build a database and shapely lines for faster lookup
        db = index.Index()
        edge_shps = []

        for edge_i, edge_coord in enumerate(all_edges):
            db.insert(edge_i, get_edge_bbox(edge_coord))
            (x0, y0, x1, y1) = edge_coord
            edge_shps.append(LineString([(x0, y0), (x1, y1)]))

        # go from shortest to longest edge
        edge_lengths = np.array([edge.length for edge in edge_shps])
        iter_order = np.argsort(edge_lengths)[::-1]

        for edge_i in iter_order:
            if used_mask[edge_i]:
                continue

            edge_shp = edge_shps[edge_i]
            edge_coord = all_edges[edge_i]
            cand_ids = db.intersection(get_edge_bbox(edge_coord))
            merge_i = -1

            for cand_i in cand_ids:
                if (edge_i == cand_i) or used_mask[cand_i]:
                    continue

                # ignore this candidate if angle between is too large
                cand_shp = edge_shps[cand_i]
                cand_coord = all_edges[cand_i]
                if not angle_matches(edge_coord, cand_coord):
                    continue

                # check each point of the edge, if they are close to the candidate
                ((ax, ay), (bx, by)) = list(edge_shp.coords)
                a = Point(ax, ay)
                b = Point(bx, by)
                a_dist = LineString(nearest_points(a, cand_shp)).length
                b_dist = LineString(nearest_points(b, cand_shp)).length

                if (a_dist < threshold) and (b_dist < threshold):
                    merge_i = cand_i
                    break

            if merge_i == -1:
                used_mask[edge_i] = True
                merged_edges.append(edge_coord)

            else:
                used_mask[edge_i] = True
                used_mask[merge_i] = True
                merged_edge = merge_edges_helper(edge_coord, all_edges[merge_i])
                merged_edges.append(merged_edge)

        if len(merged_edges) == len(all_edges):
            print("No more actions to do")
            all_edges = merged_edges
            break
        else:
            print("%d -> %d" % (len(all_edges), len(merged_edges)))
            all_edges = merged_edges

    return np.array(all_edges)


def merge_edges_helper(ab, cd):
    (ax, ay, bx, by) = ab
    (cx, cy, dx, dy) = cd

    ab_len = (ax - bx) ** 2 + (ay - by) ** 2
    ac_len = (ax - cx) ** 2 + (ay - cy) ** 2
    ad_len = (ax - dx) ** 2 + (ay - dy) ** 2
    bc_len = (bx - cx) ** 2 + (by - cy) ** 2
    bd_len = (bx - dx) ** 2 + (by - dy) ** 2
    cd_len = (cx - dx) ** 2 + (cy - dy) ** 2

    cand_lines = [
        [ax, ay, bx, by],
        [ax, ay, cx, cy],
        [ax, ay, dx, dy],
        [bx, by, cx, cy],
        [bx, by, dx, dy],
        [cx, cy, dx, dy],
    ]
    lens = np.array([ab_len, ac_len, ad_len, bc_len, bd_len, cd_len])
    new_line = cand_lines[np.argmax(lens)]

    # determine the two edges' orientation, and make sure they match
    # points = np.array([[ax,ay], [bx,by], [cx,cy], [dx,dy]])

    # # horizontal
    # if abs(bx-ax) > abs(by-ay):
    #   assert abs(dx-cx) > abs(dy-cy)

    #   min_idx = points[:,0].argmin()
    #   max_idx = points[:,0].argmax()

    # # vertical
    # else:
    #   assert abs(dx-cx) <= abs(dy-cy)

    #   min_idx = points[:,1].argmin()
    #   max_idx = points[:,1].argmax()

    return np.array(new_line)


def process_image(image):
    image = image.transpose((2, 0, 1))
    image -= np.array(density_mean)[:, np.newaxis, np.newaxis]
    image /= np.array(density_std)[:, np.newaxis, np.newaxis]
    image = image.astype(np.float32)

    return image


# def normalize_corners(corners, scale):
#     corners = corners.copy()
#     tform = SimilarityTransform(scale=scale)
#     new_corners = tform(corners)
#
#     return new_corners


def normalize_corners(corners, scale=None, max_side_len=1000):
    corners = corners.copy()

    if scale == None:
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

    return new_corners, scale


def normalize_floor(image, corners, max_side_len=1000):
    corners = corners.copy()

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

    return new_image, new_corners, scale


def convert_edges(pred_edges, combine_threshold=3.0):
    # we do pixel-level precision
    corners = pred_edges.reshape(-1, 2).astype(int)
    corners = np.unique(corners, axis=0)

    new_corners = []
    corner_map = {}
    corner_used = np.zeros(len(corners), dtype=bool)

    for corner_i in range(len(corners)):
        if corner_used[corner_i]:
            continue

        dists = distance.cdist(corners[corner_i : corner_i + 1], corners)
        similar_inds = (dists[0] < combine_threshold).nonzero()[0]
        new_corner = corners[similar_inds].mean(axis=0)
        new_corners.append(new_corner)
        new_corner_i = len(new_corners) - 1

        for similar_i in similar_inds:
            corner_map[similar_i] = new_corner_i
            corner_used[similar_i] = True

    new_corners = np.array(new_corners).round().astype(int)

    # edges use corner index
    edges = []

    for edge in pred_edges:
        (x0, y0, x1, y1) = edge.astype(int)

        a = (corners == [x0, y0]).all(axis=1)
        assert a.sum() == 1
        a = corner_map[a.argmax()]

        b = (corners == [x1, y1]).all(axis=1)
        assert b.sum() == 1
        b = corner_map[b.argmax()]

        if ((a, b) not in edges) and ((b, a) not in edges):
            edges.append((a, b))

    return {"corners": np.array(new_corners), "edges": np.array(edges)}


def normalize_edges(edge_coords):
    corners, edges = corners_and_edges(edge_coords)

    corners, _ = normalize_corners(corners)
    edge_coords = corners[edges].reshape(-1, 4)

    return edge_coords


def unnormalize_edges(edges, clip_edges=True):
    edges = (edges + 1) / 2 * 255.0
    if clip_edges:
        edges = torch.clip(edges, 0, 255)

    return edges


def get_vis_bounds(modified_coords, image_shape):
    minx = modified_coords[:, [0, 2]].min() - 15
    miny = modified_coords[:, [1, 3]].min() - 15
    maxx = modified_coords[:, [0, 2]].max() + 15
    maxy = modified_coords[:, [1, 3]].max() + 15

    side_len = max([maxx - minx, maxy - miny, 512])

    (h, w) = image_shape
    if (side_len > h) or (side_len > w):
        minx = 0
        miny = 0
        maxx = w
        maxy = h
    else:
        centerx = (minx + maxx) // 2
        centery = (miny + maxy) // 2

        minx = max(centerx - side_len // 2, 0)
        miny = max(centery - side_len // 2, 0)
        maxx = minx + side_len
        maxy = miny + side_len

        if maxx > w:
            maxx = w
            minx = maxx - side_len
        if maxy > h:
            maxy = h
            miny = maxy - side_len

    assert 0 <= minx <= maxx <= w
    assert 0 <= miny <= maxy <= h

    return minx, miny, maxx, maxy


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


# to keep walls consistent, sort them so that it's always x0 <= x1
def make_xyxy_consistent(walls):
    new_walls = walls

    for i in range(len(walls)):
        x0, y0, x1, y1 = walls[i]

        if x0 > x1:
            new_walls[i] = [x1, y1, x0, y0]

        elif x0 == x1:
            if y0 > y1:
                new_walls[i] = [x1, y1, x0, y0]

    return new_walls


def random_range(low, high, size=None):
    return np.random.choice(range(int(low), int(high + 1)), size=size)
    # return (high-low) * np.random.random_sample(size) + low


def get_edge_len(edge):
    (ax, ay, bx, by) = edge
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def nearest_dist(ab, cd):
    ax, ay, bx, by = ab
    cx, cy, dx, dy = cd

    ab_shp = LineString([(ax, ay), (bx, by)])
    cd_shp = LineString([(cx, cy), (dx, dy)])

    return LineString(nearest_points(ab_shp, cd_shp)).length


def perpen_dist(ab, cd):
    ax, ay, bx, by = ab
    cx, cy, dx, dy = cd

    ab_shp = LineString([(ax, ay), (bx, by)])
    cd_shp = LineString([(cx, cy), (dx, dy)])
    c_shp = Point(cx, cy)
    d_shp = Point(dx, dy)

    ab_c = LineString(nearest_points(ab_shp, c_shp)).length
    ab_d = LineString(nearest_points(ab_shp, d_shp)).length

    return max(ab_c, ab_d)


# every GT edge can only be matched with the closest pred edge
def compute_label(pred_coords, gt_coords, threshold=0, dist_fn="line_dist"):
    if dist_fn == "line_dist":
        full_dists = distance.cdist(pred_coords, gt_coords, line_dist)
    elif dist_fn == "l2":
        full_dists = distance.cdist(pred_coords, gt_coords, metric="euclidean")
    else:
        raise Exception("Unknown distance function")

    labels = np.zeros(len(pred_coords))
    matches = []
    iter_order = full_dists.min(axis=1).argsort()

    for pred_i in iter_order:
        edge_dists = full_dists[pred_i]
        if edge_dists.min() <= threshold:
            gt_i = edge_dists.argmin()
            labels[pred_i] = True
            full_dists[:, gt_i] = float("inf")  # remove GT edge from pool
            matches.append((pred_i, gt_i))

    return labels, np.array(matches)


def compute_precision_recall(pred_coords, gt_coords, threshold=0):
    labels, _ = compute_label(
        pred_coords=pred_coords, gt_coords=gt_coords, threshold=threshold
    )

    precision = (labels == 1).sum() / len(labels) * 100
    recall = (labels == 1).sum() / len(gt_coords) * 100

    return precision, recall


# find a's index within b
def find_idx(a, b):
    mask = (a == b).all(axis=1)
    assert mask.sum() <= 1

    if mask.sum() == 0:
        return -1
    else:
        return mask.argmax()


def corners_and_edges(edge_coords):
    corners = np.concatenate([edge_coords[:, :2], edge_coords[:, 2:]])
    corners = np.unique(corners, axis=0)

    edge_ids = []
    for ax, ay, bx, by in edge_coords:
        assert (corners == np.array([ax, ay])).all(axis=1).sum() == 1
        a_idx = (corners == np.array([ax, ay])).all(axis=1).argmax()

        assert (corners == np.array([bx, by])).all(axis=1).sum() == 1
        b_idx = (corners == np.array([bx, by])).all(axis=1).argmax()

        # assert (a_idx, b_idx) not in edge_ids
        # assert (b_idx, a_idx) not in edge_ids
        if a_idx < b_idx:
            edge_ids.append((a_idx, b_idx))
        else:
            edge_ids.append((b_idx, a_idx))

    edge_ids = np.array(edge_ids)

    return corners, edge_ids


def get_nearby(edges_from, edges_to, threshold=5):
    dists = distance.cdist(edges_from, edges_to, nearest_dist)

    nearby = dists.copy()
    nearby[nearby <= threshold] = 1
    nearby[nearby > threshold] = 0
    nearby = nearby.astype(int)
    np.fill_diagonal(nearby, 0)

    return nearby, dists


def get_connect_ends(edges, threshold=5):
    shps = []
    for x0, y0, x1, y1 in edges:
        shps.append(LineString([(x0, y0), (x1, y1)]))

    connect = {}
    for i in range(len(shps)):
        pt_a = Point(shps[i].coords[0])
        pt_b = Point(shps[i].coords[1])

        for j in range(len(shps)):
            if i == j:
                continue

            if i not in connect.keys():
                connect[i] = []

            # the intersection point has to be one of the ends
            dist_aj = LineString(nearest_points(pt_a, shps[j])).length
            dist_bj = LineString(nearest_points(pt_b, shps[j])).length

            if (dist_aj <= threshold) or (dist_bj <= threshold):
                connect[i].append(j)

    return connect


def line_dist(ab, cd, angle_threshold=5):
    # check if angles are close
    # angle = get_angle(ab, cd)
    # if angle_threshold < angle < (180 - angle_threshold):
    #     return float("inf")

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


def perturb_edges(edges, image_size, modify_i=-1):
    (img_h, img_w) = image_size
    new_edges = edges.copy()

    for edge_i, edge in enumerate(new_edges):
        if (modify_i > -1) and (edge_i != modify_i):
            continue

        edge = edge.copy()

        # chance to not modify edge
        if np.random.random() < 0.5:
            continue

        # random jitter endpoints
        if np.random.random() < 0.5:
            edge[:2] += random_range(-3, 3, size=2)
        if np.random.random() < 0.5:
            edge[2:] += random_range(-3, 3, size=2)

        # don't modify if it's too short
        if get_edge_len(edge) <= 5:
            continue

        # random tilt

        # random shift
        if np.random.random() < 0.25:
            shift = random_range(-3, 3)
            # shift horizontally
            if np.random.random() < 0.5:
                edge[0] += shift
                edge[2] += shift
            # shift vertically
            else:
                edge[1] += shift
                edge[3] += shift

        if get_edge_len(edge) <= 5:  # don't modify if it's too short
            continue

        # random shorten/extend edge in either/both directions
        (ax, ay, bx, by) = edge
        ab_len = get_edge_len(edge)
        cx = (ax + bx) / 2
        cy = (ay + by) / 2

        # c -> a
        if np.random.random() < 0.5:
            dx = ax - cx
            dy = ay - cy

            dist = random_range(-1 * ab_len / 2, ab_len / 2)
            # dist = np.clip(dist, a_min=-10, a_max=10)
            ratio = 1 + dist / ab_len

            ax = cx + ratio * dx
            ay = cy + ratio * dy

        # c -> b
        if np.random.random() < 0.5:
            dx = bx - cx
            dy = by - cy

            dist = random_range(-1 * ab_len / 2, ab_len / 2)
            # dist = np.clip(dist, a_min=-10, a_max=10)
            ratio = 1 + dist / ab_len

            bx = cx + ratio * dx
            by = cy + ratio * dy

        edge = [ax, ay, bx, by]
        if get_edge_len(edge) <= 5:  # don't modify if it's too short
            continue
        else:
            new_edges[edge_i] = edge

    new_edges[:, 0] = np.clip(new_edges[:, 0], 0, img_w)
    new_edges[:, 1] = np.clip(new_edges[:, 1], 0, img_h)
    new_edges[:, 2] = np.clip(new_edges[:, 2], 0, img_w)
    new_edges[:, 3] = np.clip(new_edges[:, 3], 0, img_h)

    return make_xyxy_consistent(new_edges)


def get_angle(ab, cd):
    ax, ay, bx, by = ab
    cx, cy, dx, dy = cd

    m1 = (by - ay) / (bx - ax + 1e-6)
    m2 = (dy - cy) / (dx - cx + 1e-6)
    angle = abs(np.arctan(m1) - np.arctan(m2)) * 180 / np.pi
    assert 0 <= angle <= 180

    return angle


# check to make sure angle from x-axis is less than threshold
def angle_matches(ab, cd, angle_threshold=10):
    angle = get_angle(ab, cd)
    if angle_threshold < angle < (180 - angle_threshold):
        return False
    else:
        return True


def get_edge_bbox(edge_coord):
    (x0, y0, x1, y1) = edge_coord

    c_x = (x0 + x1) / 2
    c_y = (y0 + y1) / 2
    w = abs(x1 - x0) + 10
    h = abs(y1 - y0) + 10

    l = c_x - w / 2
    r = c_x + w / 2
    b = c_y - h / 2
    t = c_y + h / 2

    return (l, b, r, t)


def match_heuristics(ab, cd, angle_threshold=15, dist_threshold=15):
    # check if angles are close
    angle = get_angle(ab, cd)
    if angle_threshold < angle < (180 - angle_threshold):
        return 1_000_000

    # check if closest distance is small
    (ax, ay, bx, by) = ab
    (cx, cy, dx, dy) = cd

    ab_shp = LineString([(ax, ay), (bx, by)])
    cd_shp = LineString([(cx, cy), (dx, dy)])

    min_dist = LineString(nearest_points(ab_shp, cd_shp)).length
    if min_dist >= dist_threshold:
        return 2_000_000

    # return the line distance
    ac_dist = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
    bd_dist = np.sqrt((bx - dx) ** 2 + (by - dy) ** 2)
    ad_dist = np.sqrt((ax - dx) ** 2 + (ay - dy) ** 2)
    bc_dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)

    dist = min(ac_dist + bd_dist, ad_dist + bc_dist)
    assert dist <= 1_000_000

    if dist > max(ab_shp.length, cd_shp.length):
        return 3_000_000
    else:
        return dist


def normalize_density(density_slice):
    # drop bottom and top 5 percentile for density map
    counts = sorted(density_slice[density_slice > 0])
    lower = np.percentile(counts, q=10)
    upper = np.percentile(counts, q=90)

    density_slice = np.maximum(density_slice, lower)
    density_slice = np.minimum(density_slice, upper)
    density_slice -= lower
    density_slice /= upper - lower

    return density_slice


def normalize_example(example):
    # normalize edge coordinates
    assert example["ref_mask"].sum() == 1
    edge_coords = example["edge_coords"].copy().astype(float)
    ref_edge = edge_coords[example["ref_mask"].argmax()]

    # normalize by reference edge
    (x0, y0, x1, y1) = ref_edge
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    edge_coords -= [cx, cy, cx, cy]

    # normalize all edges to 1000 x 1000 canvas
    minx = edge_coords[:, [0, 2]].min()
    maxx = edge_coords[:, [0, 2]].max()
    miny = edge_coords[:, [1, 3]].min()
    maxy = edge_coords[:, [1, 3]].max()

    scale_x = 1000 / (maxx - minx)
    scale_y = 1000 / (maxy - miny)
    edge_coords[:, [0, 2]] *= scale_x
    edge_coords[:, [1, 3]] *= scale_y

    example["original_coords"] = example["edge_coords"].copy()
    example["edge_coords"] = edge_coords

    return example


# reference shape: (s b n k h l p d)
# s: number of decoder layers
# b: batch size
# n: number of edges
# k: number of samples per edge
# h: number of heads
# l: number of levels
# p: number of additional sample points
# d: dimension, should be 2 for xy
def vis_ref(image, edge_coords, ref_dict, s2_ids):
    image = image[0].permute([1, 2, 0]).cpu().numpy()
    image -= image.min()
    image /= image.max()
    image = np.max(image, axis=2)

    edge_coords = edge_coords[0].cpu().numpy()
    s2_ids = s2_ids[0].cpu().numpy()
    per_ref_in = ref_dict["per_ref_in"].cpu().numpy()
    per_ref_out = ref_dict["per_ref_out"].cpu().numpy()
    hb_ref_in = ref_dict["hb_ref_in"].cpu().numpy()
    hb_ref_out = ref_dict["hb_ref_out"].cpu().numpy()

    (h, w) = image.shape
    per_ref_in[..., 0] *= w
    per_ref_in[..., 1] *= h
    per_ref_out[..., 0] *= w
    per_ref_out[..., 1] *= h
    hb_ref_in[..., 0] *= w
    hb_ref_in[..., 1] *= h
    hb_ref_out[..., 0] *= w
    hb_ref_out[..., 1] *= h

    # randomly pick one of the edges and plot reference points
    edge_inds = list(range(len(edge_coords)))
    np.random.shuffle(edge_inds)

    for edge_i in edge_inds:
        plt.imshow(image, cmap="gray")

        (x0, y0, x1, y1) = edge_coords[edge_i]
        plt.plot([x0, x1], [y0, y1], "-c")

        # only plot the classifier points
        ref_pts_in = per_ref_in[0, 0, edge_i].reshape(-1, 2)
        ref_pts_out = per_ref_out[0, 0, edge_i].reshape(-1, 2)

        plt.plot(ref_pts_out[:, 0], ref_pts_out[:, 1], "*m")
        plt.plot(ref_pts_in[:, 0], ref_pts_in[:, 1], "*g")

        # for src_i in range(len(ref_pts_in)):
        #     (ax, ay) = ref_pts_in[src_i]
        #     for (bx, by) in ref_pts_out[src_i]:
        #         plt.arrow(ax, ay, bx - ax, by - ay, color="g")
        #         plt.plot(bx, by, "*g")

        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
