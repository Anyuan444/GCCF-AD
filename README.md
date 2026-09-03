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
```text- Python 3.9
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
```
# 2. Dataset
This project uses the MVTec 3D-AD dataset.
```text Dataset download link:
https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad
After downloading, organize the dataset as follows:
'''datasets/
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
```
# 3. Project Structure
```text ## 
DCMF-AD/
├── train.py
├── test.py
├── models/
│   ├── dataset.py
│   ├── features.py
│   ├── feature_transfer_nets.py
│   ├── full_models.py
│   ├── pointnet2_utils.py
│   └── ...
├── utils/
│   ├── general_utils.py
│   ├── metrics_utils.py
│   └── ...
├── results/
├── checkpoints_CFM_mvtec/
├── requirements.txt
└── README.md
```
# 4. Training
Run the following command to train the model on one category:
```text python train.py \
  --dataset_path ./datasets/mvtec_3d_anomaly_detection \
  --checkpoint_savepath ./checkpoints_CFM_mvtec \
  --class_name foam \
  --epochs_no 50 \
  --batch_size 4

Example for another category:

python train.py \
  --dataset_path ./datasets/mvtec_3d_anomaly_detection \
  --checkpoint_savepath ./checkpoints_CFM_mvtec \
  --class_name bagel \
  --epochs_no 50 \
  --batch_size 4

During training, the model progressively optimizes:

2D→3D feature mapping
3D→2D feature mapping
Deep feature contrastive loss
Adaptive fusion layer
Lightweight detection head

The training checkpoints will be saved to:

checkpoints_CFM_mvtec/
└── foam/
    ├── CFM_2Dto3D_foam_50ep_4bs.pth
    ├── CFM_3Dto2D_foam_50ep_4bs.pth
    ├── FusionLayer_foam_50ep_4bs.pth
    ├── JointDet_foam_50ep_4bs.pth
    ├── training_steps_foam_50ep_4bs.csv
    └── training_epochs_foam_50ep_4bs.csv
```
# 5. Testing
After training, run inference with:
```text python test.py \
  --dataset_path ./datasets/mvtec_3d_anomaly_detection \
  --checkpoint_folder ./checkpoints_CFM_mvtec \
  --class_name foam \
  --epochs_no 50 \
  --batch_size 4

The testing script loads the trained CFM modules, FusionLayer, and JointDet. It then generates anomaly maps and computes evaluation metrics.
```
# 6.Important Notes
```textMake sure the dataset path is correct.
If the following error occurs:
ValueError: num_samples should be a positive integer value, but got num_samples=0

it usually means that the dataset path or folder structure is incorrect.

The checkpoint path used for testing must be the same as the path used during training.
The class_name, epochs_no, and batch_size used during testing should match the training configuration.

For example, if training uses:

--class_name foam --epochs_no 50 --batch_size 4

then testing should also use:

--class_name foam --epochs_no 50 --batch_size 4

Otherwise, the checkpoint file names may not match.
```
