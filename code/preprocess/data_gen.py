import os
import json
import glob
import shutil

import laspy
import imageio
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from os.path import join as path_join

from tqdm import tqdm
from PIL import Image
from shapely.geometry import LineString, MultiLineString, box
from scipy.signal import convolve2d
from skimage.draw import line as draw_line
from skimage.transform import resize
from skimage.morphology import dilation, erosion, thin, square

get_pc = True


# these should exist already
history_root = "../../data/history"
laz_root = "../../data/laz"
tform_root = "../../data/transforms"

# to be created
pcd_cache_root = "../../data/pcd"
bounds_root = "../../data/bounds"
density_root = "../../data/density"
action_root = "../../data/actions"
annot_root = "../../data/annot"


element_types = [
    "Autodesk.Revit.DB.Wall",
    "Autodesk.Revit.DB.Floor",
    "Autodesk.Revit.DB.Ceiling",
    "Autodesk.Revit.DB.Panel",
    "Autodesk.Revit.DB.Mullion",
    "Autodesk.Revit.DB.FamilyInstance",
    "Autodesk.Revit.DB.Level",
    "Autodesk.Revit.DB.CurtainGridLine",
    "Autodesk.Revit.DB.WallType",
    "Autodesk.Revit.DB.FloorType",
    "Autodesk.Revit.DB.CeilingType",
    "Autodesk.Revit.DB.PanelType",
    "Autodesk.Revit.DB.MullionType",
    "Autodesk.Revit.DB.FamilySymbol",
]

type_map = {
    "Autodesk.Revit.DB.Wall": 0,
    "Autodesk.Revit.DB.WallType": 1,
    "door": 2,
    "window": 3,
    "column": 4,
    "door_type": 5,
    "window_type": 6,
    "column_type": 7,
}

# point cloud slice heights
slice_intervals = np.array([-1.0, 1.64, 3.28, 4.92, 6.56, 8.2, 9.84, 12.0])


def set_font_sizes():
    # sets font sizes
    SMALL_SIZE = 8 * 2
    MEDIUM_SIZE = 10 * 2
    BIGGER_SIZE = 12 * 2

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_type(element):
    element_type = element["element_type"]
    element_members = element["members"]

    # skip beams for now
    if element_members[".Id.IntegerValue"] == 414019:
        element_type = ""

    elif element_type == "Autodesk.Revit.DB.Panel":
        element_type = "window"

    elif element_type == "Autodesk.Revit.DB.FamilyInstance":
        type_name = element_members[
            ".Symbol.BIP.OmniClass Title_-1002503_ReadOnly_String"
        ]

        if "Door" in type_name:
            element_type = "door"
        elif "Window" in type_name:
            element_type = "window"
        elif "Column" in type_name:
            element_type = "column"
        else:
            element_type = ""

    elif element_type == "Autodesk.Revit.DB.FamilySymbol":
        type_name = element_members[".BIP.OmniClass Title_-1002503_ReadOnly_String"]

        if "Door" in type_name:
            element_type = "door_type"
        elif "Window" in type_name:
            element_type = "window_type"
        elif "Column" in type_name:
            element_type = "column_type"
        else:
            element_type = ""

    if element_type not in type_map.keys():
        return -1
    else:
        return type_map[element_type]


def get_unique_name(freelancer_root):
    floor_name, architect_name = freelancer_root.split("/")[-3:-1]
    unique_name = "%s_%s" % (floor_name, architect_name)
    return unique_name


# expecting box in [0,1]
def crop_pc(pcd, bbox):
    minx, miny, maxx, maxy = bbox
    ring = [
        [minx, miny, 0],
        [minx, maxy, 0],
        [maxx, maxy, 0],
        [maxx, miny, 0],
        [minx, miny, 0],
    ]

    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_min = pcd.get_min_bound()[2] - 1
    vol.axis_max = pcd.get_max_bound()[2] + 1
    vol.bounding_polygon = o3d.utility.Vector3dVector(ring)

    return vol.crop_point_cloud(pcd)


def crop_pc_z(pcd, minz, maxz):
    minx, miny, _ = pcd.get_min_bound() - 1
    maxx, maxy, _ = pcd.get_max_bound() + 1

    ring = [
        [minx, miny, 0],
        [minx, maxy, 0],
        [maxx, maxy, 0],
        [maxx, miny, 0],
        [minx, miny, 0],
    ]

    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_min = minz
    vol.axis_max = maxz
    vol.bounding_polygon = o3d.utility.Vector3dVector(ring)

    return vol.crop_point_cloud(pcd)


def pcd2las(pcd):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    header = laspy.LasHeader(point_format=3)
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.red = colors[:, 0] * 65280.0
    las.green = colors[:, 1] * 65280.0
    las.blue = colors[:, 2] * 65280.0

    return las


def get_density(pcd, bbox, width=256, height=256, percentile=90):
    ps = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    image_res = np.array((width, height))

    minx, miny, maxx, maxy = bbox
    min_coords = np.array([minx, miny, np.min(ps[:, 2])])
    max_coords = np.array([maxx, maxy, np.max(ps[:, 2])])

    # adds a border to the density image
    # max_m_min = max_coords - min_coords
    # max_coords = max_coords + 0.1 * max_m_min
    # min_coords = min_coords - 0.1 * max_m_min

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res

    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    coordinates = np.round(
        (ps[:, :2] - min_coords[None, :2])
        / (max_coords[None, :2] - min_coords[None, :2])
        * image_res[None]
    )
    coordinates = np.minimum(
        np.maximum(coordinates, np.zeros_like(image_res)), image_res - 1
    )

    density = np.zeros((height, width), dtype=np.float32)
    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    unique_coordinates = unique_coordinates.astype(np.int32)
    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts

    # normalize by the 50th percentile
    # upper = np.percentile(sorted(counts), q=percentile)
    # density = np.minimum(density / upper, 1.0)
    # density = np.flipud(density)

    return density


def get_height(pcd, bbox, width=512, height=512):
    ps = np.asarray(pcd.points)

    image_res = np.array((width, height))

    minx, miny, maxx, maxy = bbox
    min_coords = np.array([minx, miny, np.min(ps[:, 2])])
    max_coords = np.array([maxx, maxy, np.max(ps[:, 2])])

    # adds a border to the density image
    # max_m_min = max_coords - min_coords
    # max_coords = max_coords + 0.1 * max_m_min
    # min_coords = min_coords - 0.1 * max_m_min

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res

    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    coordinates = np.round(
        (ps[:, :2] - min_coords[None, :2])
        / (max_coords[None, :2] - min_coords[None, :2])
        * image_res[None]
    )
    coordinates = np.minimum(
        np.maximum(coordinates, np.zeros_like(image_res)), image_res - 1
    )

    # unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    # unique_coordinates = unique_coordinates.astype(np.int32)

    coordinates = coordinates.astype(np.int32)

    # also get top-down color
    height = np.zeros((height, width), dtype=np.float32)

    groups = {}
    for (x, y), pt in zip(coordinates, ps):
        height[y, x] = max(pt[2], height[y, x])
        # key = tuple(coord_2d.tolist())
        # if key not in groups.keys():
        #   groups[key] = []
        # groups[key].append(pt)

    # height /= height.max()
    height = np.flipud(height)

    return height


def scale_coords(coords, old_w, old_h, side_len):
    scale_x = old_w / side_len
    scale_y = old_h / side_len
    center_x = old_w / 2
    center_y = old_h / 2

    coords = coords.copy()
    coords -= np.array([[center_x, center_y], [center_x, center_y]])
    coords /= np.array([[scale_x, scale_y], [scale_x, scale_y]])
    coords += np.array([[side_len / 2, side_len / 2], [side_len / 2, side_len / 2]])

    return coords


def process_model(floor_root, pcd):
    floor_name = floor_root.split("/")[-2]

    ###########################################
    ### Get transactions from journal files ###
    ###########################################
    print("Loading journal files from", floor_root)

    # load transactions from journal files
    transactions = []
    num_total = 0
    num_undo = 0
    num_rollback = 0
    journal_files = glob.glob(path_join(floor_root, "my_journal*.txt"))

    for journal_f in sorted(journal_files):
        with open(journal_f, "r", encoding="utf-8") as f:
            for line in f:
                if "===" not in line:
                    line = line.replace("\x00", "")  # one file has this issue
                    transaction = json.loads(line)
                    num_total += 1

                    # # skip if we don't have any affected elements
                    # if len(transaction['deleted_elements']) == 0:
                    #   # skip if the affected elements are not of concern
                    #   skip = True
                    #   for element in transaction['added_elements'] + transaction['modified_elements']:
                    #     if element['element_type'] in element_types:
                    #       skip = False

                    #   if skip:
                    #     continue

                    # skip rollback operations, which don't do anything
                    if transaction["operation_type"] == "TransactionRolledBack":
                        assert (
                            (len(transaction["added_elements"]) == 0)
                            and (len(transaction["modified_elements"]) == 0)
                            and (len(transaction["deleted_elements"]) == 0)
                        )
                        num_rollback += 1
                        continue

                    # skip this one and the one before if it's an undo transaction
                    if transaction["operation_type"] == "TransactionUndone":
                        for action_name in transaction["action_names"]:
                            assert len(transactions[-1]["action_names"]) == 1
                            last_action_name = transactions[-1]["action_names"][0]

                            if action_name == "Modify element attributes":
                                # NOTE this action seems to be causing a lot of issues, here
                                # we check if there are any relevant elements being modified
                                # or added. If not, we can simply ignore this undo action
                                skip = True
                                for element in (
                                    transaction["added_elements"]
                                    + transaction["modified_elements"]
                                ):
                                    if element["element_type"] in element_types:
                                        skip = False

                                if not skip:
                                    # NOTE a special case where the undo transaction can have a
                                    # different name
                                    assert (
                                        last_action_name == "Modify type attributes"
                                    ) or (action_name == last_action_name)
                            else:
                                assert action_name == last_action_name

                            num_undo += 1
                            del transactions[-1]

                        continue

                    transactions.append(transaction)

    print("Total number of transactions: %d" % num_total)
    print("Number of undo operations: %d" % num_undo)
    print("Number of rollback operations: %d" % num_rollback)

    ############################################
    ### See how many of each element we have ###
    ############################################
    if False:
        counter = {}
        for type_id in type_map.values():
            counter[type_id] = set()

        for transaction in transactions:
            for element in transaction["added_elements"]:
                eid = element["members"][".Id.IntegerValue"]
                type_id = get_type(element)

                if type_id > -1:
                    counter[type_id].add(eid)

            for element in transaction["modified_elements"]:
                eid = element["members"][".Id.IntegerValue"]
                type_id = get_type(element)

                if (type_id > -1) and (eid not in counter[type_id]):
                    counter[type_id].add(eid)

            for eid in transaction["deleted_elements"]:
                for type_id in counter.keys():
                    if eid in counter[type_id]:
                        counter[type_id].remove(eid)

        with open("general_stats.csv", "a") as f:
            f.write(
                "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"
                % (
                    floor_root,
                    num_total,
                    len(counter[0]),
                    len(counter[1]),
                    len(counter[2]),
                    len(counter[3]),
                    len(counter[4]),
                    len(counter[5]),
                    len(counter[6]),
                    len(counter[7]),
                )
            )

        return

    #################################
    ### Number of actions by type ###
    #################################
    if False:
        counter = {}
        type_db = {}

        for type_id in type_map.values():
            counter[type_id] = {}
            for action_id in [0, 1, 2]:
                counter[type_id][action_id] = 0

        for transaction in transactions:
            for element in transaction["added_elements"]:
                eid = element["members"][".Id.IntegerValue"]
                type_id = get_type(element)

                if type_id > -1:
                    counter[type_id][0] += 1
                    type_db[eid] = type_id

            for element in transaction["modified_elements"]:
                eid = element["members"][".Id.IntegerValue"]
                type_id = get_type(element)

                if type_id > -1:
                    counter[type_id][1] += 1
                    type_db[eid] = type_id

            for eid in transaction["deleted_elements"]:
                if eid in type_db.keys():
                    type_id = type_db[eid]
                    counter[type_id][2] += 1

        labels = ["walls", "doors", "windows", "columns"]
        colors = ["steelblue", "gold", "mediumseagreen", "lightcoral"]
        sizes = [
            sum(counter[0].values()),
            sum(counter[1].values()),
            sum(counter[2].values()),
            sum(counter[3].values()),
        ]

        plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
        plt.title("Percentage of actions by element type")
        plt.tight_layout()

        unique_name = get_unique_name(floor_root)
        plt.savefig("figs/%s_action_dist.png" % unique_name)
        plt.close()

        return

    ###################################################################
    ### Filter walls so we ignore curtain and non-floor-level walls ###
    ###################################################################
    def is_wall(element):
        BIP_FAMILY_NAME = ".WallType.BIP.Family Name_-1002002_ReadOnly_String"
        BIP_BASE_OFFSET = ".BIP.Base Offset_-1001108_ReadWrite_Double"

        if (
            (element["element_type"] == "Autodesk.Revit.DB.Wall")
            and ("Curtain Wall" not in element["members"][BIP_FAMILY_NAME])
            and (element["members"][BIP_BASE_OFFSET] < 1e-6)
        ):
            return True
        else:
            return False

    concern_ids = []

    for transaction in transactions:
        for element in transaction["added_elements"] + transaction["modified_elements"]:
            eid = element["members"][".Id.IntegerValue"]

            if is_wall(element):
                concern_ids.append(eid)
            else:
                if eid in concern_ids:
                    concern_ids.remove(eid)

    ###############################
    ### Collect actions globaly ###
    ###############################
    scale = 12

    def get_centerline(element):
        # NOTE we do conversion here already, converting units to inches
        x0 = element["members"][".Location.Curve.GetEndPoint.X"] * scale
        y0 = element["members"][".Location.Curve.GetEndPoint.Y"] * scale
        z0 = element["members"][".Location.Curve.GetEndPoint.Z"] * scale
        x1 = element["members"][".Location.Curve.GetEndPoint_1.X"] * scale
        y1 = element["members"][".Location.Curve.GetEndPoint_1.Y"] * scale
        z1 = element["members"][".Location.Curve.GetEndPoint_1.Z"] * scale
        width = element["members"][".WallType.Width"]

        return [x0, y0, z0, x1, y1, z1, width]

    actions = []
    final_db = {}

    for t_idx, transaction in enumerate(transactions):
        action = {"added": [], "modified": [], "deleted": []}

        for element in transaction["added_elements"]:
            eid = element["members"][".Id.IntegerValue"]

            if is_wall(element) and (eid in concern_ids):
                assert eid not in final_db.keys()
                centerline = get_centerline(element)
                action["added"].append([eid, centerline])
                final_db[eid] = centerline

        for element in transaction["modified_elements"]:
            eid = element["members"][".Id.IntegerValue"]

            if is_wall(element) and (eid in concern_ids):
                # 372784 for avinash
                if eid not in final_db.keys():
                    print("%d not in db" % eid)
                    final_db[eid] = [0, 0, 0, 0, 0, 0, 0]

                # check to see if the wall location actually got modified
                old_centerline = final_db[eid]
                new_centerline = get_centerline(element)
                diff = np.abs(np.array(new_centerline) - np.array(old_centerline))

                # we don't do filtering here
                if diff.max() > 1e-6:
                    action["modified"].append([eid, old_centerline, new_centerline])
                    final_db[eid] = new_centerline

        for eid in transaction["deleted_elements"]:
            if eid in final_db.keys():
                action["deleted"].append(eid)
                del final_db[eid]

        if len(action["added"]) or len(action["modified"]) or len(action["deleted"]):
            actions.append(action)

    os.makedirs(action_root, exist_ok=True)
    action_f = path_join(action_root, "%s.json" % floor_name)
    with open(action_f, "w") as f:
        json.dump(actions, f)

    ############################################################
    ### Generate density maps based on building bounding box ###
    ############################################################
    lines = []
    for eid in final_db.keys():
        x0, y0, z0, x1, y1, z1, width = final_db[eid]
        lines.append(LineString([(x0, y0), (x1, y1)]))

    lines = MultiLineString(lines)
    (bounds_minx, bounds_miny, bounds_maxx, bounds_maxy) = lines.bounds

    bounds_minx -= 15
    bounds_miny -= 15
    bounds_maxx += 15
    bounds_maxy += 15

    # save the boundary information
    os.makedirs(bounds_root, exist_ok=True)
    bounds_f = path_join(bounds_root, "%s.csv" % floor_name)
    with open(bounds_f, "w") as f:
        f.write("%s,%s,%s,%s\n" % (bounds_minx, bounds_miny, bounds_maxx, bounds_maxy))

    # decide on the resolution of things
    h = round(bounds_maxy - bounds_miny)
    w = round(bounds_maxx - bounds_minx)

    # get density map
    floor_density_root = path_join(density_root, floor_name)
    os.makedirs(floor_density_root, exist_ok=True)

    if get_pc:
        print("Obtaining full density map")

        # crop point cloud so we work with relevant area
        # also we divide by scale since our point cloud is in feet still
        bounds = np.array([bounds_minx, bounds_miny, bounds_maxx, bounds_maxy]) / scale
        cropped_pcd = crop_pc(pcd, bounds)

        # slice by every half a meter, sort of I think
        print("Slicing point cloud")

        density_slices = []
        for slice_i, (slice_minz, slice_maxz) in enumerate(
            tqdm(list(zip(slice_intervals[:-1], slice_intervals[1:])))
        ):
            sliced_pcd = crop_pc_z(cropped_pcd, slice_minz, slice_maxz)
            density_slice = get_density(sliced_pcd, bounds, width=w, height=h)

            density_f = path_join(floor_density_root, "density_%02d.npy" % slice_i)
            np.save(density_f, density_slice)
            density_slices.append(density_slice)

            if False:
                counts = sorted(density_slice[density_slice > 0])
                lower = np.percentile(counts, q=10)
                upper = np.percentile(counts, q=90)

                density_slice = np.maximum(density_slice, lower)
                density_slice = np.minimum(density_slice, upper)
                density_slice -= lower
                density_slice /= upper - lower

                slice_img = (density_slice * 255.0).astype(np.uint8)
                slice_img = Image.fromarray(slice_img).convert("RGBA")

                unique_name = get_unique_name(floor_root)
                slice_img.save(
                    "./vis/%s_%02d_%.3f_%.3f.png"
                    % (unique_name, slice_i, slice_minz, slice_maxz)
                )

    else:
        density_slices = []
        for slice_i in range(7):
            density_f = path_join(floor_density_root, "density_%02d.npy" % slice_i)
            density_slice = np.load(density_f)
            density_slices.append(density_slice)

    # first channel is slice 1-4, then 5, then 6
    # and we normalize for each channel
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

    density_full = [
        normalize_density(np.sum(density_slices[:4], axis=0)),
        normalize_density(density_slices[4]),
        normalize_density(np.sum(density_slices[5:7], axis=0)),
    ]
    density_full = np.stack(density_full, axis=2)

    ##############################################
    ### (optional) Full visualization of walls ###
    ##############################################
    if False:
        print("Full visualization of walls")

        density_img = (density_full / 2 * 255.0).astype(np.uint8)
        density_img = Image.fromarray(density_img).convert("RGBA")

        wall_mask = np.zeros([h, w, 4], dtype=np.uint8)
        for line in lines:
            (x0, y0), (x1, y1) = line.coords

            x0 = round(x0 - bounds_minx)
            x1 = round(x1 - bounds_minx)
            y0 = round(y0 - bounds_miny)
            y1 = round(y1 - bounds_miny)

            rr, cc = draw_line(y0, x0, y1, x1)
            wall_mask[rr, cc, 0] = 255
            wall_mask[rr, cc, 3] = 255

        wall_mask = Image.fromarray(wall_mask)

        two_img = Image.alpha_composite(density_img, wall_mask)
        two_img = two_img.convert("RGB")

        if not os.path.exists("./full_walls"):
            os.makedirs("./full_walls")

        unique_name = get_unique_name(floor_root)
        two_img.save("./full_walls/%s.png" % unique_name)

        return

    ###########################################
    ### (optional) Visualize steps globally ###
    ###########################################
    if False:
        print("Visualizing steps globally")

        if not os.path.exists("./global"):
            os.makedirs("./global")

        unique_name = get_unique_name(floor_root)

        db = {}

        frame_fs = []
        for step_i, step in enumerate(tqdm(actions)):
            for eid, centerline in step["added"]:
                db[eid] = centerline
            for eid, old_centerline, new_centerline in step["modified"]:
                if eid not in db.keys():
                    print(eid, floor_root)
                db[eid] = new_centerline
            for eid in step["deleted"]:
                assert eid in db.keys()
                del db[eid]

            canvas = np.zeros([h, w], dtype=np.uint8)
            for eid, centerline in db.items():
                (x0, y0, z0, x1, y1, z1) = centerline.copy()
                x0 = round(x0 - bounds_minx)
                y0 = round(y0 - bounds_miny)
                x1 = round(x1 - bounds_minx)
                y1 = round(y1 - bounds_miny)

                if (max(x0, x1) > w) or (max(y0, y1) > h):
                    print("Line out of bounds")
                    continue

                rr, cc = draw_line(y0, x0, y1, x1)
                canvas[rr, cc] = 255

            canvas = dilation(canvas, square(3))
            canvas = Image.fromarray(canvas)

            frame_f = "./global/%s_%04d.png" % (unique_name, step_i)
            canvas.save(frame_f)
            frame_fs.append(frame_f)

        print("Saving global visualization as a gif")

        # chunk_size = 400
        # for i in tqdm(range(0, len(frame_fs), chunk_size)):
        #   chunk_fs = frame_fs[i:i+chunk_size]

        frames = []
        for frame_i, frame_f in enumerate(frame_fs):
            frame = np.array(Image.open(frame_f))
            frames.append(frame)
            os.remove(frame_f)

        imageio.mimsave("./global/%s.gif" % unique_name, frames, duration=0.1)

    #####################################
    ### Save out the floor-scale data ###
    #####################################
    annot = {}
    edges = {}

    for eid in final_db.keys():
        x0, y0, _, x1, y1, _, width = final_db[eid]

        x0 = round(x0 - bounds_minx)
        x1 = round(x1 - bounds_minx)
        y0 = round(y0 - bounds_miny)
        y1 = round(y1 - bounds_miny)

        a = tuple((x0, y0))
        b = tuple((x1, y1))

        edge = [x0, y0, x1, y1, width]
        edges[eid] = edge

        if a not in annot.keys():
            annot[a] = []
        if b not in annot.keys():
            annot[b] = []

        annot[a].append(b)
        annot[b].append(a)

    os.makedirs(annot_root, exist_ok=True)
    annot_f = path_join(annot_root, "%s.json" % floor_name)
    with open(annot_f, "w") as f:
        json.dump(edges, f)


def density_to_img(density_slice, freelancer_root):
    counts = sorted(density_slice[density_slice > 0])
    lower = np.percentile(counts, q=10)
    upper = np.percentile(counts, q=90)

    density_slice = np.maximum(density_slice, lower)
    density_slice = np.minimum(density_slice, upper)
    density_slice -= lower
    density_slice /= upper - lower

    density_img = (density_slice * 255.0).astype(np.uint8)
    density_img = Image.fromarray(density_img).convert("RGBA")

    unique_name = get_unique_name(freelancer_root)
    annot = np.load(
        "annot/%s_one_shot_full.npy" % unique_name, allow_pickle=True
    ).item()

    (w, h) = density_img.size
    wall_mask = np.zeros([h, w, 4], dtype=np.uint8)
    for (ax, ay), bs in annot.items():
        for bx, by in bs:
            rr, cc = draw_line(ay, ax, by, bx)
            wall_mask[rr, cc, 0] = 255
            wall_mask[rr, cc, 3] = 255

    wall_mask = Image.fromarray(wall_mask)

    two_img = Image.alpha_composite(density_img, wall_mask)
    two_img = two_img.convert("RGB")

    return two_img


def vis_slices(floor_root):
    for freelancer_root in glob.glob(path_join(floor_root, "*/")):
        if not os.path.exists(path_join(freelancer_root, "intervals.npy")):
            continue

        unique_name = get_unique_name(freelancer_root)

        # slice 1-4
        density_slices = []
        for slice_i in [0, 1, 2, 3]:
            density_slice = np.load(
                path_join(freelancer_root, "density_%02d.npy" % slice_i)
            )
            density_slices.append(density_slice)

        density_slice = np.sum(density_slices, axis=0)
        two_img = density_to_img(density_slice, freelancer_root)
        two_img.save("./vis/%s_0.png" % unique_name)

        # slice 5
        density_slice = np.load(path_join(freelancer_root, "density_04.npy"))
        two_img = density_to_img(density_slice, freelancer_root)
        two_img.save("./vis/%s_1.png" % unique_name)

        # slice 6
        density_slices = []
        for slice_i in [5, 6]:
            density_slice = np.load(
                path_join(freelancer_root, "density_%02d.npy" % slice_i)
            )
            density_slices.append(density_slice)
        density_slice = np.sum(density_slices, axis=0)
        two_img = density_to_img(density_slice, freelancer_root)
        two_img.save("./vis/%s_2.png" % unique_name)


def process_floor(floor_root):
    floor_name = floor_root.split("/")[-2]

    # load cached point cloud if we have it
    cached_path = path_join(pcd_cache_root, "%s.pcd" % floor_name)

    pcd = None

    if get_pc:
        if os.path.exists(cached_path):
            print("Loading cached point cloud for floor %s" % floor_root)
            pcd = o3d.io.read_point_cloud(cached_path)

        else:
            print("Loading LAZ for floor %s" % floor_root)
            laz_path = path_join(laz_root, "%s.laz" % floor_name)

            # load whole point cloud as o3d object
            with laspy.open(laz_path) as fh:
                las = fh.read()
                scale_x, scale_y, scale_z = las.header.scale
                offset_x, offset_y, offset_z = las.header.offset

                X = las.X * scale_x + offset_x
                Y = las.Y * scale_y + offset_y
                Z = las.Z * scale_z + offset_z

                points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
                # points /= .3048  # meters to feet

                R, G, B = las.red, las.green, las.blue
                colors = np.stack([R, G, B], axis=-1).astype(np.float64)
                colors = colors / 65280.0

            # perform architect transforms
            print("Using point cloud transformation from freelancer")

            # obtain transformation matrix
            tform_path = path_join(tform_root, "%s.txt" % floor_name)
            with open(tform_path, "r") as f:
                b0 = [float(x) for x in f.readline().strip().split(",")[1:]]
                b1 = [float(x) for x in f.readline().strip().split(",")[1:]]
                b2 = [float(x) for x in f.readline().strip().split(",")[1:]]
                origin = [float(x) for x in f.readline().strip().split(",")[1:]]
                scale = float(f.readline().strip().split(",")[1])

                t = np.array([b0, b1, b2, origin]) / scale

            # do the transform
            points = np.concatenate([points, np.ones([len(points), 1])], axis=1)
            points = points @ t

            # also meters to feet
            points /= 0.3048

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.io.write_point_cloud(cached_path, pcd)

    process_model(floor_root, pcd)


if __name__ == "__main__":
    set_font_sizes()

    skip_floors = [
        # '05_MedOffice_01_F2',
        # '06_MedOffice_02_F1',
        "07_MedOffice_03_F4",
        "07_MedOffice_03_F5",
        # '11_MedOffice_05_F4',
        # '19_MedOffice_07_F4',
        # '32_ShortOffice_05_F1',
        # '32_ShortOffice_05_F2',
        # '32_ShortOffice_05_F3',
        # '33_SmallBuilding_03_F1'
    ]

    floor_roots = sorted(glob.glob(path_join(history_root, "*/")))
    for floor_idx, floor_root in enumerate(floor_roots):
        skip = False
        for skip_floor in skip_floors:
            if skip_floor in floor_root:
                skip = True

        if not skip:
            process_floor(floor_root)

            # try:
            #     process_floor(floor_root)
            #     vis_slices(floor_root)
            # except:
            #     print("Something went wrong... :(")
            # print("")
