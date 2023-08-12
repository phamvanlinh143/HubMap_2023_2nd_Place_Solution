import numpy as np
import pandas as pd
from pathlib import Path
import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class StainTransform:
    def __init__(self, img_aug_root, img_ext, margin=0, prob=1.0):
        self.img_aug_root = Path(img_aug_root)
        self.img_ext = img_ext
        self.prob = prob
        self._prepare_img_refs()
        self.margin = margin

    def _prepare_img_refs(self):
        ids_dirs = list(self.img_aug_root.iterdir()) 
        ids_dirs = list(filter(lambda x: x.is_dir(), ids_dirs))
        self._stain_infos = {}
        for ids_dir in ids_dirs:
            ids_stem = ids_dir.stem
            stain_img_paths = list(Path(ids_dir).rglob(f"*{self.img_ext}"))
            self._stain_infos[ids_stem] = stain_img_paths
            
    @staticmethod
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

        return center_region

    def _get_img_ref(self, img_path):
        ref_img = cv2.imread(str(img_path))

        if self.margin > 0:
            ref_img = self.crop_full_tile(ref_img, self.margin)

        return ref_img
    
    def _get_stain_offline(self, org_image, img_stem):
        ref_candidates = self._stain_infos.get(img_stem, None)

        if ref_candidates is None:
            return org_image
        else:
            if isinstance(ref_candidates, list) and len(ref_candidates) > 0:
                ref_path = np.random.choice(ref_candidates)
                stain_img = self._get_img_ref(ref_path)
                return stain_img
            else:
                return org_image

    def _adjust_img(self, results):
        for key in results.get('img_fields', ['img']):
            img_stem = Path(results['filename']).stem
            img = results[key]
            results[key] = self._get_stain_offline(img, img_stem).astype(img.dtype)
    
    def __call__(self, results):

        if np.random.rand() > self.prob:
            return results
        self._adjust_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
    

# @PIPELINES.register_module()
# class StainTransform:
#     def __init__(self, img_root, img_ext , df_ref_path, std_p=0.5, stn_p=0.5, prob=1.0):
#         self.img_root = Path(img_root)
#         self.img_ext = img_ext
#         self.df_refs = pd.read_csv(df_ref_path)
#         self.prob = prob
#         self.std_p = std_p
#         self.std_percentile = (85, 96)
#         self.stn_p = stn_p
#         self.stn_method = ("macenko", "vahadane")
#         self._prepare_img_refs()

#     def _prepare_img_refs(self):
#         lst_ids = self.df_refs["id"].tolist()
#         self._lst_img_refs = [self.img_root.joinpath(f"{ids}{self.img_ext}") for ids in lst_ids]

#     def _get_img_ref(self):
#         img_path = np.random.choice(self._lst_img_refs)
#         ref_img = cv2.imread(str(img_path))
#         return ref_img

#     def _process(self, image):
#         condition = True
#         num_try = 0
#         while condition and num_try < 5:
#             try:
#                 ref_img = self._get_img_ref()
#                 if np.random.rand() < self.std_p:
#                     percentile = np.random.randint(*self.std_percentile)
#                     target = staintools.LuminosityStandardizer.standardize(ref_img, percentile)
#                     to_trans = staintools.LuminosityStandardizer.standardize(image, percentile)
#                 else:
#                     target = ref_img
#                     to_trans = image

#                 if np.random.rand() < self.stn_p:
#                     stn_mtd = np.random.choice(self.stn_method)
#                     normalizer = staintools.StainNormalizer(method=stn_mtd)
#                 else:
#                     normalizer = staintools.ReinhardColorNormalizer()

#                 normalizer.fit(target)
#                 to_trans = normalizer.transform(to_trans)
#                 condition = False
#             except:
#                 num_try += 1
#                 pass
#         if num_try >= 5:
#             to_trans = image
#         return to_trans
    
#     def _get_stain_offline(self, img_stem):

    
#     def _adjust_img(self, results):
#         for key in results.get('img_fields', ['img']):
#             img = results[key]
#             results[key] = self._process(img).astype(img.dtype)
    
#     def __call__(self, results):

#         if np.random.rand() > self.prob:
#             return results
#         self._adjust_img(results)
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         return repr_str