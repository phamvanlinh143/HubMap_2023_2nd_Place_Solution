import json, cv2, numpy as np, itertools, random, pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from sklearn import model_selection

from skimage import io
from pycocotools.coco import COCO
import matplotlib.patches as mpatches

from pathlib import Path

from skimage import measure
from skimage import filters


def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

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

def show_mask(img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
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

def create_anno(list_segms, category):
    annos = []
    for segm in list_segms:
        seg_info = dict(type=category, coordinates=segm)
        annos.append(seg_info)
    return annos

def crop_full_tile(img, margin=128):
    base_size = 512
    img_h, img_w = img.shape[:2]
    assert img_h == base_size * 3
    assert img_w == base_size * 3
    crop_h, crop_w = base_size + margin * 2, base_size + margin * 2
    x0 = base_size - margin
    y0 = base_size - margin

    x1 = x0 + crop_w
    y1 = y0 + crop_h
    center_region = img[y0:y1, x0:x1]
    bbox = [x0, y0, x1, y1]
    return center_region, bbox

def crop_mask(mask, bbox):
    x0, y0, x1, y1 = bbox
    mask_region = mask[y0:y1, x0:x1]
    return mask_region

categories_list = ['blood_vessel', 'glomerulus', 'unsure']

img_dir = Path("../datasets/train_9tiles/")
img_ext = ".tif"

default_shape = (512 * 3, 512 * 3)
MARGIN_PIXEL = 2
mask_pixel_thresh = 32

def process(jsonl_path, out_img_dir, out_jsonl_dir, margin=128):
    jsonl_path = Path(jsonl_path)
    data_infos = []
    with open(jsonl_path, "r") as file:
        for line in file:
            data_infos.append(json.loads(line))

    merged_tile_polygons = []

    for data_info in tqdm(data_infos):
        ids = data_info["id"]

        img_path = img_dir.joinpath(f"{ids}{img_ext}")
        img = cv2.imread(str(img_path))

        center_region, bbox = crop_full_tile(img, margin=margin)

        base_o = np.array(bbox[:2])

        annos = data_info["annotations"]
        
        crop_annos = []
        for anno in annos:
            cat_type = anno["type"]

            segmentation = anno["coordinates"]

            mask_img = coordinates_to_masks(segmentation, default_shape)[0]

            mask_img = crop_mask(mask_img, bbox)

            num_fg = np.sum(mask_img.astype(np.uint8))
            if num_fg > 0:
                ys, xs = np.where(mask_img)
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                mask_bbox = [x1, y1, x2, y2]

                if check_mergeable(mask_bbox, mask_img.shape[:2], MARGIN_PIXEL):
                    contours, hierarchy = cv2.findContours(mask_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    merged_insts = [np.transpose(inst, (1, 0, 2)).tolist() for inst in contours]
                    mask_size = list(map(lambda x: np.array(x).size, merged_insts))
                    new_segmentations = []
                    for idx, size in enumerate(mask_size):
                        if size > mask_pixel_thresh:
                            new_segmentations.append(merged_insts[idx])
                else:
                    new_segmentation = np.array(segmentation) - base_o
                    new_segmentations = [new_segmentation.tolist()]
            else:
                continue
            
            for new_segmentation in new_segmentations:
                crop_anno = {
                    "type": cat_type,
                    'coordinates': new_segmentation
                }
                crop_annos.append(crop_anno)

        if len(crop_annos) == 0:
            continue

        tile_info = dict(id=ids, annotations=crop_annos)
                
        merged_tile_polygons.append(tile_info)

        saved_img_path = out_img_dir.joinpath(f"{ids}{img_ext}")

        cv2.imwrite(str(saved_img_path), center_region)


    save_jsonl(merged_tile_polygons, f"{out_jsonl_dir.joinpath(jsonl_path.stem)}_{margin}.jsonl")


def main():
    margin = 128
    jsonl_dir = Path("anno_9tiles/")
    jsonl_paths = list(jsonl_dir.rglob("*.jsonl"))
    
    out_img_dir = Path(f"../datasets/train_9tiles_crop{margin}")
    ensure_dir(out_img_dir)
    
    out_jsonl_dir = Path(f"annos_9tiles_crop{margin}")
    ensure_dir(out_jsonl_dir)
    
    for jsonl_path in jsonl_paths:
        print(jsonl_path)
        process(jsonl_path, out_img_dir, out_jsonl_dir, margin=margin)


if __name__ == "__main__":
    main()