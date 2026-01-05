# Large Medical Image Foundation Model (LaMIM)
https://github.com/MAI-Lab-West-China-Hospital/LaMIM
Chen, M., Zhang, M., Yin, L. et al. Medical image foundation models in assisting diagnosis of brain tumors: a pilot study. Eur Radiol 34, 6667–6679 (2024). https://doi.org/10.1007/s00330-024-10728-1

依赖 monai
pip install monai==1.4.0  与 torch 2.3.1+cu121 适配  


import sys
import os
import torch
# 添加MONAI源码路径到Python路径
monai_path = r'D:\share_code\tools-advanced\brain_mri_FM\MONAI'
sys.path.insert(0, monai_path)
from monai.networks.nets import ViTAutoEnc

model = ViTAutoEnc(
			in_channels=4,
			out_channels=4,
			img_size=(96, 96, 96),
			patch_size=(16, 16, 16),
			proj_type='conv',
			num_heads=32,
			num_layers=16,
			hidden_size=2048,
			mlp_dim=3072,
)
for k,v in model.named_parameters(): print(f"{k}, {v.shape}")



## Model Overview
This repository provides self-supervised pre-trained vision transformer autoencoder (ViTAutoEnc), which was trained on 57,000 brain enhanced MRI scans, for multi-contrast brain MRI. The training process is illustrated below
![workflow](workflow.png)
Two mask schemes are used: mask block size 16*16*16 with 86 blocks (A); mask block size 4*4*4 with 6000 blocks (B)
![dropout_scheme](dropout_scheme.png)

## How to use

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Download pretrained models
We provide the self-supervised pre-trained weights in [SSL_ViT_Block16](https://drive.google.com/file/d/1x1VI-0AoMqQZYVcbNoTQxe5ac-t3Ia5R/view?usp=drive_link) and [SSL_ViT-Block4](https://drive.google.com/file/d/1ttHL3IeZwuhjLPKS6SeLYjRQW-p6dD1U/view?usp=drive_link). Download and place them into the **Pretrained_models** folder.

### Data Prepare
1. Convert DICOM in NIFTI. [dcm2niix](https://github.com/rordenlab/dcm2niix) is recommended.
2. Skull strip. Choose T1 weight or enhanced T1w volume with higher resolutin for skull stripping. [HD-BET](https://github.com/MIC-DKFZ/HD-BET) is recommended. 
3. Remove redundant blank area using ExtractRegionFromImageByMask from [ANTs](https://github.com/ANTsX/ANTs).
4. Co-register other contrasts, e.g. T2w and FLAIR, to skull stripped T1w or T1c, then multiply with the brain mask generated in previous step.
5. Merge the co-registered and skull stripped volumes into a 4D volume in the order of T1w, T1c, T2w and FLAIR.

### Fine tune the weights on downstream task
1. Modify ```train_files``` and ```val_files``` in finetune_train.py to point to the pre-processed train and validation images, and run finetune_train.py
2. Modify ```test_files``` in finetune_test.py to point the preprocessed test images, and run finetune_test.py to evaluate downstream model.
3. (Optional) Modify Model.py to test more classifier configurations.

## Pre-train from scratch with own data
It is highly recommended to perform pre-train with [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) using NVIDIA Container on server equiped with multiple GPUs.
1. Pull the MONAI docker image
`docker pull projectmonai/monai`
2. Modify PATHs in run_docker.sh accordingly, then run ```run_docker.sh```
3. Run ```run_pretraining.sh``` insider docker
