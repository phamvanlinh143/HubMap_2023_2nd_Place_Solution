# HuBMAP - Hacking the Human Vasculature

https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature

## 2nd place solution for HubMap 2023 Challenge hosted on Kaggle

This documentation outlines how to reproduce the 2nd place solution for HubMap - Hacking the Human Vasculature

## Conda environment setup
```bash
conda create --name hubmap python=3.10 -y
conda activate hubmap

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install mmcv-full
pip install mmcv-full==1.6.0

pip install einops==0.6.1 timm==0.5.4 pandas PyYaml 

git clone https://github.com/phamvanlinh143/mmdetection.git
cd mmdetection
pip install -r requirements.txt
pip install -v -e .

# install mmcls and albumentations==1.3.0
pip install mmcls==0.25.0 albumentations==1.3.0
```
## Backbone References:

| Architecture | Backbone | Reference | Pretrained-Weight |
| --- | --- | --- | --- | 
| Cascade Mask R-CNN | CoaT Small | [CoaT](https://github.com/mlpc-ucsd/CoaT) | [coat_small_pretrained](https://vcl.ucsd.edu/coat/pretrained/tasks/mmdet/cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_4f7a069e.pth) |
| Cascade Mask R-CNN | Swin-T | [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) | [swin_t_pretrained](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth) |
| Cascade Mask R-CNN | ConvNeXt-T | [ConvNeXt](https://github.com/open-mmlab/mmpretrain/blob/mmcls-0.x/mmcls/models/backbones/convnext.py) | [convnext_t_pretrained](https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth) |
| Cascade Mask R-CNN | ConvNeXt-S | [ConvNeXt](https://github.com/open-mmlab/mmpretrain/blob/mmcls-0.x/mmcls/models/backbones/convnext.py) | [convnext_s_pretrained](https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth) |

***Download pretained-weights and save to `mmdetection/pretrained_weights` folder***

## Data Preparation


## References
* https://github.com/open-mmlab/mmdetection