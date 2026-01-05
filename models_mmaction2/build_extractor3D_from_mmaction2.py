"""
sufeng, sufeng@pku.edu.cn
conda env omimri
build model:
    input shape: 6D: B, S, C=3, D/T, H, W (C=RGB)
    output shape: 3D: B, S*fea_num, fea_dim
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np

import sys
# add OmiMRI_path to sys_path
OmiMRI_path = '/gpfs/share/home/2306393443/sufengdata/tools-advanced/set_transformer/OmiMRI-V1.0'
if OmiMRI_path not in sys.path:
    sys.path.insert(0, OmiMRI_path)

# mmlab, mmaction2 tools
from mmaction.apis import init_recognizer
from mmengine import Config # from mmcv import Config

MMACTION_PATH = os.path.join(OmiMRI_path, 'mmaction2')

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}; {param.shape}")

# # two kinds of loading model weights in mmaction2
# create model first, then load checkpoint
# original_model = init_recognizer(cfg, device=device)
# state_dict = torch.load(path_ckpt, map_location='cpu')['state_dict']
# original_model.load_state_dict(state_dict, strict=False)
# create model & initialize weights from checkpoint simualtaneously
# original_model = init_recognizer(cfg, checkpoint=path_ckpt, device=device)

def build_feature_extractor(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = config['model_name']
    C, D, H, W = config['CDHW']  # (3, 48, 192, 192)
    # ---------- resnet50_slowonly_tpn
    if model_name == 'resnet50_slowonly_tpn':
        path_config = os.path.join(MMACTION_PATH, 'configs/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py')
        path_ckpt = os.path.join(MMACTION_PATH, 'modelzoo/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth')
        cfg = Config.fromfile(path_config)
        cfg.model.backbone.conv1_kernel = (1, 7, 7)  # default (1, 7, 7)
        cfg.model.backbone.conv1_stride_t = 3   # default 1
        cfg.model.backbone.pool1_stride_t = 2   # default 1
        # initialize the model
        original_model = init_recognizer(cfg, checkpoint=path_ckpt, device=device)
        del original_model.data_preprocessor  # use preprocess outside the model, in data_loader
        del original_model.cls_head
        # check the output feature shape
        original_model.to(device)
        with torch.no_grad(): 
            x = torch.randn(4, 3, C, D, H, W)
            fea = original_model(x.to(device)) # tuple

        print(f"input shape: {x.shape}")
        if isinstance(fea, tuple) or isinstance(fea, list):
            for idx, value in enumerate(fea):
                print(f"out {idx}, shape {value.shape}") # ([12, 1024, 8, 12, 12], [12, 2048, 8, 6, 6])
        else:
            print(f" out shape {value.shape}")

        fea_num_dim = [((D//6)*(H//16)*(W//16), 1024), 
                       ((D//6)*(H//32)*(W//32), 2048)]

    return original_model, fea_num_dim

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    # models
    config= {'model_name': 'resnet50_slowonly_tpn', 
             'CDHW': (3, 48, 192, 192),}

    for k,v in config.items():
        print(f"{k}: {v}")
    
    original_model, fea_num_dim = build_feature_extractor(config)
    original_model.to(device)
    
    # ------ forward test
    C, D, H, W = config['CDHW']
    with torch.no_grad():
        fea_original = original_model(torch.randn(4, 3, C, D, H, W).to(device))
        # print(fea_original.shape) 
    
    summary(original_model, input_size=(4, 3, C, D, H, W), col_names=["input_size", "output_size", "num_params"]) 

    # print(original_model)
    # print_model_parameters(original_model)

