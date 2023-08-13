import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from natsort import natsorted
import staintools

from tqdm.auto import tqdm

def ensure_dir(data_dir: Path):
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        

class StainTransform:
    def __init__(self, img_root, img_ext ,df_ref_path, std_p=0.5, stn_p=0.5, prob=1.0):
        self.img_root = Path(img_root)
        self.img_ext = img_ext
        self.df_refs = pd.read_csv(df_ref_path)
        self.prob = prob
        self.std_p = std_p
        self.std_percentile = (85, 96)
        self.stn_p = stn_p
        self.stn_method = ("macenko", "vahadane")
        self._prepare_img_refs()

    def _prepare_img_refs(self):
        list_wsis = [df for df in list(self.df_refs.groupby("source_wsi"))]
        list_wsis = list(filter(lambda x: x[0] > 5, list_wsis))

        img_refs_by_wsi = []
        for wsi in list_wsis:
            _, val_df = wsi
            val_ids = val_df["id"].tolist()

            img_refs_by_wsi.append(np.array(list(self.img_root.joinpath(f"{ids}{self.img_ext}") for ids in val_ids)))

        self._img_refs_by_wsi = img_refs_by_wsi

    
    def random_ref_paths(self, num_samples):
        ref_paths = []
        for ref_by_wsi in self._img_refs_by_wsi:
            ref_pths = np.random.choice(ref_by_wsi, size=num_samples, replace=False).tolist()
            ref_paths.extend(ref_pths)
        return ref_paths
    

    def stain_process(self, src_img, ref_img):
        condition = True
        num_try = 0
        while condition and num_try < 10:
            try:
                if np.random.rand() < self.std_p:
                    percentile = np.random.randint(*self.std_percentile)
                    target = staintools.LuminosityStandardizer.standardize(ref_img, percentile)
                    to_trans = staintools.LuminosityStandardizer.standardize(src_img, percentile)
                else:
                    target = ref_img
                    to_trans = src_img

                if np.random.rand() < self.stn_p:
                    stn_mtd = np.random.choice(self.stn_method)
                    normalizer = staintools.StainNormalizer(method=stn_mtd)
                else:
                    normalizer = staintools.ReinhardColorNormalizer()

                normalizer.fit(target)
                to_trans = normalizer.transform(to_trans)
                condition = False
            except:
                num_try += 1
                pass
        if num_try >= 10:
            to_trans = src_img
        return to_trans

    
    def gen_stain(self, src_img, src_img_name, output_dir, num_samples):

        src_img_name = Path(src_img_name)
        src_img_stem = src_img_name.stem

        out_stain_dir = Path(output_dir.joinpath(src_img_stem))
        ensure_dir(out_stain_dir)

        ref_img_paths = self.random_ref_paths(num_samples)
        idx = 0
        for ref_path in ref_img_paths:
            ref_img = cv2.imread(str(ref_path))
            stain_img = self.stain_process(src_img, ref_img)
            out_path = out_stain_dir.joinpath(f"{src_img_stem}_{idx:03d}{self.img_ext}")
            cv2.imwrite(str(out_path), stain_img)
            idx += 1


def filter_img_paths(list_img_paths, ds_ids):
    filtered_img_paths = []

    for img_path in list_img_paths:
        img_stem = img_path.stem
        img_ids = img_stem.split("_")[0]
        if img_ids in ds_ids:
            filtered_img_paths.append(img_path)

    return filtered_img_paths


def stain_ds1():
    img_root = Path("../datasets/train/")
    img_ext = ".tif"
    df_ref_path = Path("refs_stain.csv")

    stain_img_dir = Path("../datasets/train_9tiles")
    list_img_paths = list(stain_img_dir.rglob(f"*{img_ext}"))

    ds1w1_src = Path("dataset_splits/ds1/ds1_wsi1.csv")
    ds1w1 = pd.read_csv(ds1w1_src)
    ds1w1_ids = ds1w1["id"].tolist()

    ds1w2_src = Path("dataset_splits/ds1/ds1_wsi2.csv")
    ds1w2 = pd.read_csv(ds1w2_src)
    ds1w2_ids = ds1w2["id"].tolist()

    ds1w12_ids = ds1w1_ids + ds1w2_ids

    print(f"# Before: {len(list_img_paths)}")
    list_img_paths = filter_img_paths(list_img_paths, ds1w12_ids)
    print(f"# After: {len(list_img_paths)}")

    num_samples = 1

    std_p=0.05
    stn_p=0.75
    prob=1.0
    stain_transform = StainTransform(img_root, img_ext, df_ref_path, std_p=std_p, stn_p=stn_p, prob=prob)

    stain_out_dir = Path("../datasets/stain_9tiles_augs")
    ensure_dir(stain_out_dir)

    for src_img_path in tqdm(list_img_paths):
        src_img = cv2.imread(str(src_img_path))
        img_name = src_img_path.name
        stain_transform.gen_stain(src_img, img_name, stain_out_dir, num_samples)
        

def stain_ds2():
    img_root = Path("../datasets/train/")
    img_ext = ".tif"
    df_ref_path = Path("refs_stain.csv")

    stain_img_dir = Path("../datasets/train_9tiles")
    list_img_paths = list(stain_img_dir.rglob(f"*{img_ext}"))

    ds2w1_src = Path("dataset_splits/ds2/ds2_wsi1.csv")
    ds2w1 = pd.read_csv(ds2w1_src)
    ds2w1_ids = ds2w1["id"].tolist()

    ds2w2_src = Path("dataset_splits/ds2/ds2_wsi2.csv")
    ds2w2 = pd.read_csv(ds2w2_src)
    ds2w2_ids = ds2w2["id"].tolist()
    
    ds2w3_src = Path("dataset_splits/ds2/ds2_wsi3.csv")
    ds2w3 = pd.read_csv(ds2w3_src)
    ds2w3_ids = ds2w3["id"].tolist()
    
    ds2w4_src = Path("dataset_splits/ds2/ds2_wsi4.csv")
    ds2w4 = pd.read_csv(ds2w4_src)
    ds2w4_ids = ds2w4["id"].tolist()

    ds2w1234_ids = ds2w1_ids + ds2w2_ids + ds2w3_ids + ds2w4_ids


    print(f"# Before: {len(list_img_paths)}")
    list_img_paths = filter_img_paths(list_img_paths, ds2w1234_ids)
    print(f"# After: {len(list_img_paths)}")

    num_samples = 1

    std_p=0.05
    stn_p=0.75
    prob=1.0
    stain_transform = StainTransform(img_root, img_ext, df_ref_path, std_p=std_p, stn_p=stn_p, prob=prob)

    stain_out_dir = Path("../datasets/stain_9tiles_augs")
    ensure_dir(stain_out_dir)

    for src_img_path in tqdm(list_img_paths):
        src_img = cv2.imread(str(src_img_path))
        img_name = src_img_path.name
        stain_transform.gen_stain(src_img, img_name, stain_out_dir, num_samples)
        
        
def main():
    stain_ds1()
    stain_ds2()


if __name__ == "__main__":
    main()