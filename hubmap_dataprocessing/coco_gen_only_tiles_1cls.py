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

polygon_infos = load_jsonl(DATASET_ROOT.joinpath('cleaned_polygons.jsonl'))

count_infos = dict(map(lambda x: (x["id"], get_num_vessel(x['annotations'])), polygon_infos))

def get_num_instance(id_name):
    return count_infos[id_name]


jsonl_file_path = DATASET_ROOT.joinpath('cleaned_polygons.jsonl')
data = []
with open(jsonl_file_path, "r") as file:
    for line in file:
        data.append(json.loads(line))
        
        
# ######################### hm_1cls ######################### 
categories_list=['blood_vessel']
#------------------------------------------------------------------------------
categories_ids = {name:id+1 for id, name in enumerate(categories_list)}  
ids_categories = {id+1:name for id, name in enumerate(categories_list)}  
categories =[{'id':id,'name':name} for name,id in categories_ids.items()]

print(categories_ids)
print(ids_categories)
print(categories)


# hm_1cls
def coco_structure(images_ids):
    idx=1
    annotations=[]
    images=[]
    for item in tqdm(data,total=int(len(images_ids))):
        image_id=item["id"]
        if image_id in images_ids:
            image = {"id": image_id, "file_name": image_id + ".tif", "height": 512, "width": 512}
            images.append(image)
        else:
            continue
        #-----------------------------
        anns=item["annotations"]
        for an in anns:
            category_type=an["type"]
            if category_type =="blood_vessel":
                category_id=categories_ids[category_type]
                segmentation=an["coordinates"]
                mask_img = coordinates_to_masks(segmentation, (512, 512))[0]
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
                    idx=idx+1
                
    return {"info": {}, "licenses": [], "categories": categories, "images": images, "annotations": annotations}


def gen_coco_ds1():
    ds_dir = Path("dataset_splits/ds1_kfold/")
    ds_paths = list(ds_dir.rglob("*.csv"))
    
    out_dir = DATASET_ROOT.joinpath("hm_1cls/ds1")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ds_path in ds_paths:
        print(ds_path)
        df = pd.read_csv(ds_path)
        df.reset_index(inplace=True,drop=True)
        df.head()
        print(f"before: # {len(df)} samples")
        df["num_ins"] = df["id"].apply(get_num_instance)
        df = df[df["num_ins"] > 0]
        df.reset_index(inplace=True,drop=True)
        print(f"after: # {len(df)} samples")
        ids = df['id'].values.tolist()
        coco_data = coco_structure(ids)
        ds_name = ds_path.stem
        output_file_path = out_dir.joinpath(f"{ds_name}.json")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
            
def gen_coco_oof_ds1():
    ds_dir = Path("dataset_splits/ds1_kfold/")
    ds_paths = list(ds_dir.rglob("*.csv"))
    
    ds_paths = list(filter(lambda x: "ignore" not in str(x), ds_paths))
    ds_paths
    
    out_dir = DATASET_ROOT.joinpath("hm_1cls/ds1")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    list_ids = []
    for ds_path in ds_paths:
        print(ds_path)
        df = pd.read_csv(ds_path)
        df.reset_index(inplace=True,drop=True)
        df.head()
        print(f"before: # {len(df)} samples")
        df["num_ins"] = df["id"].apply(get_num_instance)
        df = df[df["num_ins"] > 0]
        df.reset_index(inplace=True,drop=True)
        print(f"after: # {len(df)} samples")
        ids = df['id'].values.tolist()
        list_ids.extend(ids)
        
    len(list_ids)
    
    coco_data = coco_structure(list_ids)
    ds_name = "oof"
    output_file_path = out_dir.joinpath(f"{ds_name}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
        

def gen_coco_ds2():
    ds_dir = Path("dataset_splits/ds2/")
    ds_paths = list(ds_dir.rglob("*.csv"))
    out_dir = DATASET_ROOT.joinpath("hm_1cls/ds2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ds_path in ds_paths:
        print(ds_path)
        df = pd.read_csv(ds_path)
        df.reset_index(inplace=True,drop=True)
        df.head()
        print(f"before: # {len(df)} samples")
        df["num_ins"] = df["id"].apply(get_num_instance)
        df = df[df["num_ins"] > 0]
        df.reset_index(inplace=True,drop=True)
        print(f"after: # {len(df)} samples")
        ids = df['id'].values.tolist()
        coco_data = coco_structure(ids)
        ds_name = ds_path.stem
        output_file_path = out_dir.joinpath(f"{ds_name}.json")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump(coco_data, output_file, ensure_ascii=True, indent=4)
            

def main():
    gen_coco_ds1()
    gen_coco_ds2()
    gen_coco_oof_ds1()


if __name__ == "__main__":
    main()