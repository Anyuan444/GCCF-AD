# GCCF-AD

Official implementation of **GCCF-AD: Cross-modal Feature Consistency and Adaptive Fusion Framework for Multimodal Industrial Anomaly Detection**.

GCCF-AD is an unsupervised multimodal industrial anomaly detection framework that jointly exploits RGB appearance information and 3D geometric information. By establishing bidirectional cross-modal feature consistency and adaptive anomaly response fusion, GCCF-AD improves anomaly detection and localization performance on complex industrial surfaces.

The framework mainly contains:

- Bidirectional 2D↔3D cross-modal feature mapping for appearance–geometry alignment
- Cross-modal Contrastive Consistency Learning (CCCL) for semantic feature alignment
- Dynamic Cross-modal Feature Fusion (DCFU) for adaptive anomaly response integration
- Patch-level Refinement and Localization Head (PRLH) for anomaly map refinement
- End-to-end multimodal anomaly detection and localization on MVTec 3D-AD
- # 1. Environment
- Python 3.9
CUDA 11.8
PyTorch 2.5.1+cu118
Torchvision 0.20.1+cu118

Install dependencies:

pip install -r requirements.txt

A recommended requirements.txt is:

torch==2.5.1+cu118
torchvision==0.20.1+cu118
numpy
opencv-python
scikit-learn
scipy
matplotlib
tqdm
Pillow
wandb

If PyTorch cannot be installed directly through pip, please install the CUDA version from the official PyTorch website.
# 2. Dataset
This project uses the MVTec 3D-AD dataset.
Dataset download link:
https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad
After downloading, organize the dataset as follows:
datasets/
└── mvtec_3d_anomaly_detection/
    ├── bagel/
    ├── cable_gland/
    ├── carrot/
    ├── cookie/
    ├── dowel/
    ├── foam/
    ├── peach/
    ├── potato/
    ├── rope/
    └── tire/

Each category should contain the official training and testing folders from MVTec 3D-AD.

Supported categories:

bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire
