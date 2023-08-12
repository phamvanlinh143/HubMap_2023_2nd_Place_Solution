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

## References
* https://github.com/open-mmlab/mmdetection