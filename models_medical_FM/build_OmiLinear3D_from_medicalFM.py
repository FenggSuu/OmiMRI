"""
sufeng, sufeng@pku.edu.cn
conda env: omimri

build models:
    input shape: 6D, BSCDHW (C=1)
    output shape: BC'

model scale:
input_size: 'BSCDHW' (1, 1, 1, 50, 200, 200), 
                     (1, 1, 1, 48, 192, 192), 
                     (1, 1, 1, 96, 96, 96)

"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import math
import numpy as np

import sys
# add OmiMRI_path to sys_path
OmiMRI_path = '/gpfs/share/home/2306393443/sufengdata/tools-advanced/set_transformer/OmiMRI-V1.0'
if OmiMRI_path not in sys.path:
    sys.path.insert(0, OmiMRI_path)
    
class DeviceAwareModule(nn.Module):
    """
    auto device control
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __init__(self):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @property
    def device(self):
        return self._device
        
    def to(self, *args, **kwargs):
        self._device = torch._C._nn._parse_to(*args, **kwargs)[0]
        return super().to(*args, **kwargs)
        
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if x.device != self.device:
                x = x.to(self.device)
        return x

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}; {param.shape}")
        # print("-" * 50)


class AttentionPoolingEnhanced(DeviceAwareModule):
    def __init__(self, in_dim=512, out_seq=1, out_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.out_seq = out_seq
        
        assert out_dim % num_heads == 0, f"out_dim {out_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = out_dim // num_heads
        
        # query
        self.queries = nn.Parameter(torch.randn(1, out_seq, out_dim))
        # projection
        self.q_proj = nn.Linear(out_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        # norm
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        # res+ffn
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim)
        )
        # scale
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        # x shape: [batch, seq_len, in_dim]
        x = super().forward(x)
        
        batch_size, seq_len, _ = x.shape
        
        # repeat queries
        queries = self.queries.repeat(batch_size, 1, 1)  # [batch, out_seq, out_dim]
        # projection
        q = self.q_proj(queries)  # [batch, out_seq, out_dim]
        k = self.k_proj(x)        # [batch, seq_len, out_dim]
        v = self.v_proj(x)        # [batch, seq_len, out_dim]
        # multi-head
        q = q.view(batch_size, self.out_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # scale
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        # output
        attn_output = attn_probs @ v  # [batch, num_heads, out_seq, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, self.out_seq, self.out_dim
        )
        
        # out_proj
        attn_output = self.out_proj(attn_output)
        # res+nrom
        attn_output = self.norm1(attn_output + queries)
        # FFN
        ffn_output = self.ffn(attn_output)
        # res+nrom
        output = self.norm2(attn_output + ffn_output)
        
        return output  # [batch, out_seq, out_dim]
    
class OmiLinear3D_MedFM(DeviceAwareModule):
    def __init__(self, config):
        super().__init__()
        # ResNet50_MedicalNet, ResNet_CTSSL, CTViT_CTCLIP, SwinViT_BrainSegFounder
        # ViTAutoEnc_BrainIAC, ViTAutoEnc_BrainLaMIM
        if config['model_name'] == 'CTViT_CTCLIP':
            from models_medical_FM.Linear3d_CTViT_CTCLIP import Extractor3D_CTCLIP
            config['CDHW'] = (1, 50, 200, 200)
            original_model = Extractor3D_CTCLIP(CDHW=config['CDHW'], fea_num_dim=config['fea_num_dim'], Omi=config['Omi'])
        
        if config['model_name'] == 'ResNet50_MedicalNet':
            from models_medical_FM.LinearSeg3d_ResNet50_MedicalNet import Extractor3D_MedNetRes
            # config['CDHW'] = (1, 48, 192, 192)
            original_model = Extractor3D_MedNetRes(CDHW=config['CDHW'], fea_num_dim=config['fea_num_dim'], Omi=config['Omi'])

        if config['model_name'] == 'ResNet_CTSSL':
            from models_medical_FM.LinearSeg3d_ResNet_CTSSL import Extractor3D_ResCTSSL
            # config['CDHW'] = (1, 48, 192, 192)
            original_model = Extractor3D_ResCTSSL(CDHW=config['CDHW'], fea_num_dim=config['fea_num_dim'], Omi=config['Omi'])

        if config['model_name'] == 'SwinViT_BrainSegFounder':
            from models_medical_FM.LinearSeg3d_SwinViT_BrainSegFounder import Extractor3D_BrainSegFounder
            # config['CDHW'] = (1, 48, 192, 192) # (1, 96, 96, 96)
            original_model = Extractor3D_BrainSegFounder(CDHW=config['CDHW'], fea_num_dim=config['fea_num_dim'], Omi=config['Omi'])
        
        if config['model_name'] == 'ViTAutoEnc_BrainIAC':
            from models_medical_FM.LinearSeg3d_ViTAutoEnc_BrainIAC import Extractor3D_BrainIAC
            # config['CDHW'] = (1, 48, 192, 192) # (1, 96, 96, 96)
            original_model = Extractor3D_BrainIAC(CDHW=config['CDHW'], fea_num_dim=config['fea_num_dim'], Omi=config['Omi'])

        if config['model_name'] == 'ViTAutoEnc_BrainLaMIM':
            from models_medical_FM.LinearSeg3d_ViTAutoEnc_BrainLaMIM import Extractor3D_BrainLaMIM
            # config['CDHW'] = (1, 48, 192, 192) # (1, 96, 96, 96)
            original_model = Extractor3D_BrainLaMIM(CDHW=config['CDHW'], fea_num_dim=config['fea_num_dim'], Omi=config['Omi'])

        self.model_name = config['model_name']
        self.config = config
        self.image_encoder = original_model
        # feature norm
        self.feature_norm = nn.LayerNorm(config['fea_num_dim'][1])
        if config['Omi']:
            # attention
            self.set_attention = AttentionPoolingEnhanced(in_dim=self.config['fea_num_dim'][1], 
                    out_seq=1, out_dim=config['omi_dim'])
            # linear: classification/regression
            self.linear_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.GELU(),
                nn.Linear(self.config['omi_dim'], self.config['out_dimension']),
            )
        else:
            # linear: classification/regression
            self.linear_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.GELU(),
                nn.Linear(self.config['fea_num_dim'][1] * self.config['set_size'], self.config['out_dimension']),
            )
    
    def forward(self, x):
        assert len(x.shape)==6, f"input shape is {x.shape} and should be B, S, C, D, H, W."
        x = super().forward(x)
        
        B, S, C, D, H, W = x.shape
        if not self.config['Omi']:
            assert self.config['set_size']==S, f"set_size is {S} but should be {self.config['set_size']}"

        x = x.view(B*S, C, D, H, W)

        # feature extraction
        fea = self.image_encoder(x)
        # norm
        fea = self.feature_norm(fea)
        if self.config['Omi']:
            # attention [batch, omi_dim]
            fea = fea.view(B, -1, self.config['fea_num_dim'][1])
            fea_final = self.set_attention(fea)
            fea_final = fea_final.view(B, -1)
            logits = self.linear_head(fea_final)
            return logits
        else:
            fea = fea.view(B, -1)
            logits = self.linear_head(fea)
            return logits

def count_parameters(model):
    """count parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%% test
# example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    # ResNet50_MedicalNet, ResNet_CTSSL, CTViT_CTCLIP
    # SwinViT_BrainSegFounder, ViTAutoEnc_BrainIAC, ViTAutoEnc_BrainLaMIM
    config= {'model_name': 'ResNet_CTSSL', 
            'CDHW': (1, 48, 192, 192),
            'Omi': False,
            'set_size': 4,
            'fea_num_dim': (32, 512),
            'omi_dim': 128,
            'out_dimension': 2
            }

    OmiModel = OmiLinear3D_MedFM(config)
    OmiModel.to(device)
    
    # ------ forward 
    config = OmiModel.config
    for k,v in config.items():
        print(f"{k}: {v}")
    C, D, H, W = config['CDHW']
    set_size = config['set_size']
    with torch.no_grad():
        x = OmiModel(torch.randn(4, set_size, C, D, H, W).to(device))
        print(x.shape) 
    
    summary(OmiModel, input_size=(1, set_size, C, D, H, W), col_names=["input_size", "output_size", "num_params"]) 

    # print(OmiModel)
    # print_model_parameters(OmiModel)

