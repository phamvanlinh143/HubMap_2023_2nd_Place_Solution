# HuBMAP - Hacking the Human Vasculature

[Competition](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature)

[Solution Write-up](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429240)

## 2nd place solution for HubMap 2023 Challenge hosted on Kaggle

This documentation outlines how to reproduce the 2nd place solution for HubMap - Hacking the Human Vasculature

## Conda environment setup
```bash
conda create --name hubmap python=3.8 -y
conda activate hubmap

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install mmcv-full
pip install mmcv-full==1.6.0

pip install einops==0.6.1 timm==0.5.4 pandas PyYaml natsort 

# install staintools
conda install -c conda-forge python-spams
pip install staintools

git clone https://github.com/phamvanlinh143/HubMap_2023_2nd_Place_Solution.git
cd HubMap_2023_2nd_Place_Solution
pip install -r requirements.txt
pip install -v -e .

# install mmcls and albumentations==1.3.0 scipy==1.8.1
pip install mmcls==0.25.0 albumentations==1.3.0 scipy==1.8.1
```
## Backbone References:

| Architecture | Backbone | Reference | Pretrained-Weight |
| --- | --- | --- | --- | 
| Cascade Mask R-CNN | CoaT Small | [CoaT](https://github.com/mlpc-ucsd/CoaT) | [coat_small_pretrained](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_4f7a069e.pth) |
| Cascade Mask R-CNN | Swin-T | [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) | [swin_t_pretrained](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth) |
| Cascade Mask R-CNN | ConvNeXt-T | [ConvNeXt](https://github.com/open-mmlab/mmpretrain/blob/mmcls-0.x/mmcls/models/backbones/convnext.py) | [convnext_t_pretrained](https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth) |
| Cascade Mask R-CNN | ConvNeXt-S | [ConvNeXt](https://github.com/open-mmlab/mmpretrain/blob/mmcls-0.x/mmcls/models/backbones/convnext.py) | [convnext_s_pretrained](https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth) |

***Download pretained-weights and save to `HubMap_2023_2nd_Place_Solution/pretrained_weights` folder***

## Data Preparation

Please download the following datasets from the Kaggle:

- `kaggle competitions download -c hubmap-hacking-the-human-vasculature`
- `kaggle datasets download -d phamvanlinh143/hubmap-train-9tiles-crop128`
- `kaggle datasets download -d phamvanlinh143/hubmap-stain-augs`
- `kaggle datasets download -d phamvanlinh143/hubmap-stain-9tiles-augs`
- `kaggle datasets download -d phamvanlinh143/hubmap-coco-datasets`

***Download datasets and save to `HubMap_2023_2nd_Place_Solution/datasets` folder***

Directory structure should be as follows.

```
HubMap_2023_2nd_Place_Solution
├── pretrained_weights
│   ├── cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_4f7a069e.pth
│   ├── cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth
│   ├── cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth
│   └── cascade_mask_rcnn_swin_tiny_patch4_window7.pth
├── datasets
│   ├── hm_1cls                     # extracted phamvanlinh143/hubmap-coco-datasets dataset
│   ├── hm_9tiles_crop128_1cls      # extracted phamvanlinh143/hubmap-coco-datasets dataset
│   ├── stain_9tiles_augs           # extracted phamvanlinh143/hubmap-stain-9tiles-augs dataset
│   ├── stain_augs                  # extracted phamvanlinh143/hubmap-stain-augs dataset
│   ├── test                        # extracted hubmap-hacking-the-human-vasculature dataset
│   ├── train                       # extracted hubmap-hacking-the-human-vasculature dataset
│   ├── train_9tiles_crop128        # extracted phamvanlinh143/hubmap-train-9tiles-crop128 dataset
│   ├── cleaned_polygons.jsonl      # ref: https://www.kaggle.com/code/fnands/de-duplicate-labels
│   ├── polygons.jsonl              # extracted hubmap-hacking-the-human-vasculature dataset
│   ├── tile_meta.csv               # extracted hubmap-hacking-the-human-vasculature dataset
│   └── wsi_meta.csv                # extracted hubmap-hacking-the-human-vasculature dataset 
└── other folders (folked from mmdetection)
``` 

***You can create folders in `datasets` (it is not necessary to download datasets from phamvanlinh143/\*)***

***Requirement: Directory structure of `datasets` should be as follows.***
```
HubMap_2023_2nd_Place_Solution
└── datasets
    ├── test                        # extracted hubmap-hacking-the-human-vasculature dataset
    ├── train                       # extracted hubmap-hacking-the-human-vasculature dataset
    ├── cleaned_polygons.jsonl      # ref: https://www.kaggle.com/code/fnands/de-duplicate-labels
    ├── polygons.jsonl              # extracted hubmap-hacking-the-human-vasculature dataset
    ├── tile_meta.csv               # extracted hubmap-hacking-the-human-vasculature dataset
    └── wsi_meta.csv                # extracted hubmap-hacking-the-human-vasculature dataset 
``` 
### How to create `hm_1cls` (COCO Format) and `stain_augs`
```
    # create hm_1cls
    cd hubmap_dataprocessing/
    python coco_gen_only_tiles_1cls.py

    # Note: refs_stain.csv => (filtered from tile_meta.csv - ignore dataset 1)
    # create stain_augs
    python gen_stain_only_tiles.py # it will take much time

    cd ..
```
### How to create `train_9tiles_crop128`, `hm_9tiles_crop128_1cls` (COCO Format), and `stain_9tiles_augs`
```
    # create train_9tiles_crop128
    cd hubmap_dataprocessing/
    python merge_9tiles.py
    python crop128_9tiles.py

    # create hm_9tiles_crop128_1cls
    python coco_gen_9tiles_crop128_1cls.py

    # create stain_9tiles_augs
    python gen_stain_9tiles.py # it will take much time

    # remove anno_9tiles and annos_9tiles_crop128
    rm -rf anno_9tiles
    rm -rf annos_9tiles_crop128
    # remove train_9tiles folder in datasets
    rm -rf ../datasets/train_9tiles
    
    cd ..
```

## Models Training:

Once all the datasets are downloaded and unzipped. You can training each of the models in following steps:

```
Notes to model configs
    - *_pt.py : pretraining config
    - *_ft.py : finetune config
Notes to naming:
    - only_tiles     : defaut dataset from competition (tile with shape 512x512)
    - 9tiles_crop128 : merging 8 tiles around the original tile. Padding and cropping 128 pixels around the original tile (tile with shape 768x768). 
```

1. Training only_tiles models (4 backbone: Coat-Small, Swin-T, ConvneXt-T, ConvneXt-S)

    ***Example: Training fold 0 Swin-T***

    - Step 1: Pretraining for fold 0
        ```
        CUDA_VISIBLE_DEVICES=0 python tools/train.py hubmap_configs/only_tiles/swin_t/cascade_mask_rcnn_swin_t_1cls_ds1_w1l_pt.py
        ```
    - Step 2: Do SWA for pretrained checkpoints (checkpoints were saved at workdir: `workdirs/only_tiles/swin_t/ds1_w1l_pt/`)
        ```
        python do_swa.py --workdir workdirs/only_tiles/swin_t/ds1_w1l_pt/
        ```
    - Step 3: Finetune for fold 0 (remember to verify pretrained checkpoint after doing swa at step 2)
        ```
        CUDA_VISIBLE_DEVICES=0 python tools/train.py hubmap_configs/only_tiles/swin_t/cascade_mask_rcnn_swin_t_1cls_ds1_w1l_ft.py
        ```
    - Step 4: Do SWA for finetune checkpoints (checkpoints were saved at workdir: `workdirs/only_tiles/swin_t/ds1_w1l_ft/`)
        ```
        python do_swa.py --workdir workdirs/only_tiles/swin_t/ds1_w1l_ft/
        ```
        Final weight of Swin-T fold 0: `workdirs/only_tiles/swin_t/ds1_w1l_ft/swa_last.pth`


2. Training 9tiles_crop128 model (2 backbone: Coat-Small, Swin-T)

    ***Example: Training fold 1 Coat-Small***

    - Step 1: Pretraining for fold 1
        ```
        CUDA_VISIBLE_DEVICES=0 python tools/train.py hubmap_configs/9tiles_crop128/coat_small/cascade_mask_rcnn_coat_small_1cls_crop128_ds1_w1r_pt.py
        ```
    - Step 2: Do SWA for pretrained checkpoints (checkpoints were saved at workdir: `workdirs/9tiles_crop128/coat_small/ds1_w1r_pt/`)
        ```
        python do_swa.py --workdir workdirs/9tiles_crop128/coat_small/ds1_w1r_pt/
        ```
    - Step 3: Finetune for fold 1 (remember to verify pretrained checkpoint after doing swa at step 2)
        ```
        CUDA_VISIBLE_DEVICES=0 python tools/train.py hubmap_configs/9tiles_crop128/coat_small/cascade_mask_rcnn_coat_small_1cls_crop128_ds1_w1r_ft.py
        ```
    - Step 4: Do SWA for finetune checkpoints (checkpoints were saved at workdir: `workdirs/9tiles_crop128/coat_small/ds1_w1r_ft/`)
        ```
        python do_swa.py --workdir workdirs/9tiles_crop128/coat_small/ds1_w1r_ft/
        ```
        Final weight of Swin-T fold 1: `workdirs/9tiles_crop128/coat_small/ds1_w1r_ft/swa_last.pth`

## Inference
Inference and ensemble could be found [here](https://www.kaggle.com/code/phamvanlinh143/hubmap-2nd-place-inference).

## References
* https://github.com/open-mmlab/mmdetection