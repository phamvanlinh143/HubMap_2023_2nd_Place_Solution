import json, cv2, numpy as np, itertools, random, pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from sklearn import model_selection

import matplotlib.pyplot as plt
from skimage import io
from pycocotools.coco import COCO
import matplotlib.patches as mpatches

from pathlib import Path


DATASET_ROOT = Path("../datasets/")


def load_jsonl(path):
    infos = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            infos.append(json.loads(line))

    return infos

def load_json(path):
    with open(path, mode='r', encoding='utf-8') as f:
        json_dict = json.load(f)
        f.close()
    return json_dict

def coordinates_to_masks(coordinates, shape):
    masks = []
    for coord in coordinates:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coord)], 1)
        masks.append(mask)
    return masks

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle_to_binary_mask(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) 
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

# hm_1cls
def get_num_vessel(info):
    vessels = [x for x in info if x["type"] == 'blood_vessel']
    return len(vessels)


######################### hm_1cls ######################### 
categories_list=['blood_vessel']
#------------------------------------------------------------------------------
categories_ids = {name:id+1 for id, name in enumerate(categories_list)}  
ids_categories = {id+1:name for id, name in enumerate(categories_list)}  
categories =[{'id':id,'name':name} for name,id in categories_ids.items()]

print(categories_ids)
print(ids_categories)
print(categories)

# hm_1cls
margin = 128
img_w = 512 + margin * 2
img_h = 512 + margin * 2

def coco_structure(images_ids, info_datas):
    idx = 1
    annotations = []
    images = []

    for item in tqdm(info_datas, total=int(len(images_ids))):
        image_id = item["id"]
        if image_id in images_ids:
            image = {"id": image_id, "file_name": image_id + ".tif", "height": img_h, "width": img_w}
            images.append(image)
        else:
            continue
        #-----------------------------
        anns = item["annotations"]
        for an in anns:
            category_type = an["type"]
            if category_type =="blood_vessel":
                category_id = categories_ids[category_type]
                segmentation = an["coordinates"]
                mask_img = coordinates_to_masks(segmentation, (img_h, img_w))[0]
                ys, xs = np.where(mask_img)
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                rle = binary_mask_to_rle(mask_img)

                seg = {
                    "id": idx,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],
                    "area": int(np.sum(mask_img)),
                    "iscrowd": 0,
                }
                if image_id in images_ids:
                    annotations.append(seg)
                    idx = idx + 1
                
    return {"info": {}, "licenses": [], "categories": categories, "images": images, "annotations": annotations}

def gen_coco_ds1():
    out_dir = DATASET_ROOT.joinpath("hm_9tiles_crop128_1cls/ds1")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds1_wsi1_left_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds1_wsi1_left_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))


    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    data_train_ids = list(filter(lambda x: "val" not in x, data_ids))
    data_val_ids = list(filter(lambda x: "train" not in x, data_ids))
    print(len(data_train_ids), len(data_val_ids))

    coco_data = coco_structure(data_train_ids, data)
    ds_name = "ds1_wsi1_left_train"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)

    coco_data = coco_structure(data_val_ids, data)
    ds_name = "ds1_wsi1_left_val"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
    
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds1_wsi1_right_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds1_wsi1_right_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))


    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    data_train_ids = list(filter(lambda x: "val" not in x, data_ids))
    data_val_ids = list(filter(lambda x: "train" not in x, data_ids))
    print(len(data_train_ids), len(data_val_ids))

    coco_data = coco_structure(data_train_ids, data)
    ds_name = "ds1_wsi1_right_train"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)

    coco_data = coco_structure(data_val_ids, data)
    ds_name = "ds1_wsi1_right_val"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
        
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds1_wsi2_left_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds1_wsi2_left_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))


    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    data_train_ids = list(filter(lambda x: "val" not in x, data_ids))
    data_val_ids = list(filter(lambda x: "train" not in x, data_ids))
    print(len(data_train_ids), len(data_val_ids))

    coco_data = coco_structure(data_train_ids, data)
    ds_name = "ds1_wsi2_left_train"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)

    coco_data = coco_structure(data_val_ids, data)
    ds_name = "ds1_wsi2_left_val"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
        
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds1_wsi2_right_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))


    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds1_wsi2_right_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))


    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    data_train_ids = list(filter(lambda x: "val" not in x, data_ids))
    data_val_ids = list(filter(lambda x: "train" not in x, data_ids))
    print(len(data_train_ids), len(data_val_ids))

    coco_data = coco_structure(data_train_ids, data)
    ds_name = "ds1_wsi2_right_train"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)

    coco_data = coco_structure(data_val_ids, data)
    ds_name = "ds1_wsi2_right_val"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
        
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds1_wsi1_ignore_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds1_wsi1_ignore_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    coco_data = coco_structure(data_ids, data)
    ds_name = "ds1_wsi1_ignore"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds1_wsi2_ignore_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds1_wsi2_ignore_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    coco_data = coco_structure(data_ids, data)
    ds_name = "ds1_wsi2_ignore"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
def gen_coco_ds2():
    out_dir = DATASET_ROOT.joinpath("hm_9tiles_crop128_1cls/ds2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds2_wsi1_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds2_wsi1_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    coco_data = coco_structure(data_ids, data)
    ds_name = "ds2_wsi1"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds2_wsi2_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds2_wsi2_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    coco_data = coco_structure(data_ids, data)
    ds_name = "ds2_wsi2"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds2_wsi3_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))


    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds2_wsi3_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    coco_data = coco_structure(data_ids, data)
    ds_name = "ds2_wsi3"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
        
    polygon_infos = load_jsonl("annos_9tiles_crop128/merged_tile_ds2_wsi4_128.jsonl")

    count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

    jsonl_file_path = "annos_9tiles_crop128/merged_tile_ds2_wsi4_128.jsonl"
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    data_ids = list(map(lambda x: x["id"], data))
    print(len(data_ids))
    data_ids = list(filter(lambda x: count_infos[x] > 0, data_ids))
    print(len(data_ids))

    coco_data = coco_structure(data_ids, data)
    ds_name = "ds2_wsi4"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        
def gen_coco_oof_ds1():
    out_dir = DATASET_ROOT.joinpath("hm_9tiles_crop128_1cls/ds1")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_dir = Path("annos_9tiles_crop128/")
    jsonl_files = list(jsonl_dir.rglob("*t_*.jsonl"))
    len(jsonl_files)
    
    data = []
    for jsonl_file_path in jsonl_files:
        with open(jsonl_file_path, "r") as file:
            for line in file:
                line = json.loads(line)
                ids = line["id"]
                if "train" in ids:
                    continue
                else:
                    data.append(line)
                    
    list_ids = [item["id"] for item in data]
    
    coco_data = coco_structure(list_ids, data)
    ds_name = "oof"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        


def main():
    gen_coco_ds1()
    gen_coco_ds2()
    gen_coco_oof_ds1()


if __name__ == "__main__":
    main()
