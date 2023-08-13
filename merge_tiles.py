import json, cv2, numpy as np, pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path
from pathlib import Path

from skimage import measure
from skimage import filters
from tqdm.auto import tqdm


def load_jsonl(path):
    infos = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            infos.append(json.loads(line))

    return infos

def save_jsonl(info_dicts, path):
    with open(path, mode="w", encoding="utf-8") as f:
        for entry in info_dicts:
            json.dump(entry, f)
            f.write('\n')

def load_json(path):
    with open(path, mode='r', encoding='utf-8') as f:
        json_dict = json.load(f)
        f.close()
    return json_dict

def show_image(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.show()


def check_mergeable(bbox, img_shape, margin):
    im_h, im_w = img_shape
    x_min = margin
    x_max = im_w - margin

    y_min = margin
    y_max = im_h - margin

    x1, y1, x2, y2 = bbox

    if x1 >= x_min and x2 <= x_max and y1 >= y_min and y2 <= y_max:
        return False
    return True

def coordinates_to_masks(coordinates, shape):
    masks = []
    for coord in coordinates:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coord)], 1)
        masks.append(mask)
    return masks


def lstsegm2mask(lstsegm, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for coordinates in lstsegm:
        for coord in coordinates:
            cv2.fillPoly(mask, [np.array(coord)], 1)
    return mask


def get_coords(center_i, center_j, step_size):
    
    center_x, center_y =center_i, center_j

    coord_11 = (center_x - step_size, center_y - step_size)
    coord_12 = (center_x            , center_y - step_size)
    coord_13 = (center_x + step_size, center_y - step_size)

    coord_21 = (center_x - step_size, center_y)
    coord_22 = (center_x            , center_y)
    coord_23 = (center_x + step_size, center_y)

    coord_31 = (center_x - step_size, center_y + step_size)
    coord_32 = (center_x            , center_y + step_size)
    coord_33 = (center_x + step_size, center_y + step_size)

    all_coords = [[coord_11, coord_12, coord_13], [coord_21, coord_22, coord_23], [coord_31, coord_32, coord_33]]
    base_coords = [[(0, 0),             (step_size, 0),             (step_size*2, 0)], 
                [(0, step_size),     (step_size, step_size),     (step_size*2, step_size)], 
                [(0, step_size * 2), (step_size, step_size * 2), (step_size*2, step_size * 2)]]
    
    return all_coords, base_coords

def create_anno(list_segms, category):
    annos = []
    for segm in list_segms:
        seg_info = dict(type=category, coordinates=segm)
        annos.append(seg_info)
    return annos

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def process(ref_path, tgt_path, ignore_path, output_dir):

    img_root = Path("/data/ocr/linhpv10/Research/general_object_detection/datasets/HHV/train/")
    img_ext = ".tif"

    categories_list = ['blood_vessel', 'glomerulus', 'unsure']

    step_size = 512
    target_shape = (512 * 3, 512 * 3)
    default_shape = (512, 512)
    MARGIN_PIXEL = 2

    ensure_dir(output_dir)

    df_ref = pd.read_csv(ref_path)
    df_tgt = pd.read_csv(tgt_path)

    df_ignore = pd.read_csv(ignore_path)
    ignored_ids = df_ignore["id"].tolist()

    data_infos = []
    jsonl_file_path = "cleaned_polygons.jsonl"
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data_infos.append(json.loads(line))

    infos = dict(map(lambda x: (x["id"], x["annotations"]), data_infos))

    df_ref["coord"] = df_ref.apply(lambda x: (x["i"], x["j"]), axis = 1)
    df_ref = df_ref[["coord", "id"]]
    df_ref.set_index("coord", drop=True, inplace=True)
    ref_coords = df_ref.to_dict()["id"]

    merged_tile_polygons = []

    for idx, row in df_tgt.iterrows():

        print(f"{idx} => {row['id']}")
        center_image_path = img_root.joinpath(f"{row['id']}{img_ext}")
        center_img = cv2.imread(str(center_image_path))

        all_coords, base_coords = get_coords(row["i"], row["j"], step_size)

        train_regions = []
        val_regions = []
        train_ids_tiles = []
        val_ids_tiles = []
        has_ignored = False
        for coords, bases in zip(all_coords, base_coords):
            train_tiles = []
            val_tiles = []
            for coord, base_o in zip(coords, bases):
                ids = ref_coords.get(coord, None)
                train_info = (ids, base_o)
                val_info = (ids, base_o) if ids not in ignored_ids else (None, base_o)
                train_ids_tiles.append(train_info)
                val_ids_tiles.append(val_info)
                if ids is None:
                    train_tile = np.ones((*default_shape, 3), dtype=center_img.dtype) * 128
                    val_tile = np.ones((*default_shape, 3), dtype=center_img.dtype) * 128
                else:
                    img_path = img_root.joinpath(f"{ids}{img_ext}")
                    if ids in ignored_ids:
                        has_ignored = True
                        val_tile = np.ones((*default_shape, 3), dtype=center_img.dtype) * 128
                    else:
                        val_tile = cv2.imread(str(img_path))
                    train_tile = cv2.imread(str(img_path))  

                train_tiles.append(train_tile)
                val_tiles.append(val_tile)
            
            train_sub_region = cv2.hconcat(train_tiles)
            val_sub_region = cv2.hconcat(val_tiles)
            
            train_regions.append(train_sub_region)
            val_regions.append(val_sub_region)

        merged_train_tile = cv2.vconcat(train_regions) ## SAVED IMAGE
        merged_val_tile = cv2.vconcat(val_regions) ## SAVED IMAGE

        if has_ignored:
            list_ids_tiles = [train_ids_tiles, val_ids_tiles]
            list_postfix = ["train", "val"]
            list_merged_tile = [merged_train_tile, merged_val_tile]
        else:
            list_ids_tiles = [train_ids_tiles]
            list_postfix = ["all"]
            list_merged_tile = [merged_train_tile]

        merged_tile_infos = []

        for idx in range(len(list_ids_tiles)):
            ids_tiles = list_ids_tiles[idx]
            post_fix = list_postfix[idx]
            merged_tile_img = list_merged_tile[idx]

            for category in categories_list:
                all_no_merge_ins, all_mergable_ins = [], []

                for ids_tile in ids_tiles:
                    ids, base_o = ids_tile
                    base_o = np.array(base_o)

                    if ids is None:
                        continue

                    tile_info = infos.get(ids, None)

                    if tile_info is None:
                        continue

                    no_merge_ins, mergeable_ins = [], []

                    for anno in tile_info:
                        cat_type = anno["type"]
                        if cat_type == category:
                            segmentation = anno["coordinates"]
                            mask_img = coordinates_to_masks(segmentation, default_shape)[0]
                            ys, xs = np.where(mask_img)
                            x1, x2 = min(xs), max(xs)
                            y1, y2 = min(ys), max(ys)
                            bbox = [x1, y1, x2, y2]
                            if check_mergeable(bbox, mask_img.shape[:2], MARGIN_PIXEL):
                                mergeable_ins.append(segmentation)
                            else:
                                no_merge_ins.append(segmentation)

                    abs_no_merge_ins = []
                    for no_merge_seg in no_merge_ins:
                        abs_coord = np.array(no_merge_seg) + base_o
                        abs_coord = abs_coord.tolist()
                        abs_no_merge_ins.append(abs_coord)

                    abs_mergeable_ins = []
                    for merge_seg in mergeable_ins:
                        abs_coord = np.array(merge_seg) + base_o
                        abs_coord = abs_coord.tolist()
                        abs_mergeable_ins.append(abs_coord)

                    all_no_merge_ins.extend(abs_no_merge_ins)
                    all_mergable_ins.extend(abs_mergeable_ins)

                no_merge_annos = create_anno(all_no_merge_ins, category)
                merged_tile_infos.extend(no_merge_annos)

                mergeable_mask = lstsegm2mask(all_mergable_ins, target_shape)

                contours, _ = cv2.findContours(mergeable_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                merged_insts = [np.transpose(inst, (1, 0, 2)).tolist() for inst in contours]
                
                merged_annos = create_anno(merged_insts, category)

                merged_tile_infos.extend(merged_annos)

            merged_id = f"{row['id']}_{post_fix}"
            print(merged_id)
            tile_info = dict(id=merged_id, annotations=merged_tile_infos)
            merged_tile_polygons.append(tile_info)

            saved_img_path = output_dir.joinpath(f"{merged_id}{img_ext}")

            cv2.imwrite(str(saved_img_path), merged_tile_img)

    save_jsonl(merged_tile_polygons, f"merged_tile_{tgt_path.stem}.jsonl")
    

def main():
    output_dir = Path("/data/ocr/linhpv10/Research/general_object_detection/datasets/HHV/train_merged_tiles_ds1/")
    
    ref_path = Path("ds1/ds1_wsi1.csv")
    tgt_path = Path("ds1_kfold/ds1_wsi1_right.csv")
    ignore_path = Path("ds1_kfold/ds1_wsi1_ignore.csv")
    print(tgt_path)
    process(ref_path, tgt_path, ignore_path, output_dir)

    ref_path = Path("ds1/ds1_wsi1.csv")
    tgt_path = Path("ds1_kfold/ds1_wsi1_left.csv")
    ignore_path = Path("ds1_kfold/ds1_wsi1_ignore.csv")
    print(tgt_path)
    process(ref_path, tgt_path, ignore_path, output_dir)

    ref_path = Path("ds1/ds1_wsi2.csv")
    tgt_path = Path("ds1_kfold/ds1_wsi2_left.csv")
    ignore_path = Path("ds1_kfold/ds1_wsi2_ignore.csv")
    print(tgt_path)
    process(ref_path, tgt_path, ignore_path, output_dir)

    ref_path = Path("ds1/ds1_wsi2.csv")
    tgt_path = Path("ds1_kfold/ds1_wsi2_right.csv")
    ignore_path = Path("ds1_kfold/ds1_wsi2_ignore.csv")
    print(tgt_path)
    process(ref_path, tgt_path, ignore_path, output_dir)


if __name__ == "__main__":
    main()
