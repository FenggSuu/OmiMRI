"""
sufeng, sufeng@pku.edu.cn
conda env: omimri

build models:
    input shape: 6D, BSCDHW (C=3)
    output shape: BC'

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

from models_mmaction2.build_extractor3D_from_mmaction2 import build_feature_extractor

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
        # res+FFN
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
    
class OmiLinear3D_mma2(DeviceAwareModule):
    def __init__(self, config):
        super().__init__()
        
        original_model, fea_shape = build_feature_extractor(config)

        self.model_name = config['model_name']
        self.config = config
        self.image_encoder = original_model
        # norm
        self.feature_norm = nn.LayerNorm(config['fea_num_dim'][1])
        # feature reshape
        fea_num, fea_dim = self.config['fea_num_dim']
        self.feature_adjust = nn.ModuleDict()  # PyTorch track the modules
        if self.config['Omi']:
            # one feature layer
            self.feature_adjust['fea_num_conv1d_1'] = nn.Conv1d(
                in_channels=fea_shape[0][0],  
                out_channels=fea_num, 
                kernel_size=1, stride=1 )
            self.feature_adjust['fea_dim_linear_1'] = nn.Linear(fea_shape[0][1], fea_dim)
            # two feature layers
            if len(fea_shape) == 2:
                # out_1
                self.feature_adjust['fea_num_conv1d_1'] = nn.Conv1d(
                    in_channels=fea_shape[0][0],  
                    out_channels=fea_num//2, 
                    kernel_size=1, stride=1 )
                self.feature_adjust['fea_dim_linear_1'] = nn.Linear(fea_shape[0][1], fea_dim)
                # out_2
                self.feature_adjust['fea_num_conv1d_2'] = nn.Conv1d(
                    in_channels=fea_shape[1][0],  
                    out_channels=fea_num//2, 
                    kernel_size=1, stride=1 )
                self.feature_adjust['fea_dim_linear_2'] = nn.Linear(fea_shape[1][1], fea_dim)
            
            # attention
            self.set_attention = AttentionPoolingEnhanced(in_dim=fea_dim, 
                    out_seq=1, out_dim=self.config['omi_dim']).to(self.device)
            # linear: classification/regression
            self.linear_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.GELU(),
                nn.Linear(self.config['omi_dim'], self.config['out_dimension']),
            )
        else:
            self.feature_adjust['pool'] = nn.AdaptiveAvgPool3d((1, 1, 1))
            # one feature layer
            if len(fea_shape) == 1:
                self.feature_adjust['fea_dim_linear_1'] = nn.Linear(self.config['set_size'] * fea_shape[0][1], fea_dim)
                # linear: classification/regression
                self.linear_head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.GELU(),
                    nn.Linear(fea_dim, self.config['out_dimension']),
                )
            # two feature layers
            if len(fea_shape) == 2:
                # out_1
                self.feature_adjust['fea_dim_linear_1'] = nn.Linear(self.config['set_size'] * fea_shape[0][1], fea_dim//2)
                # out_2
                self.feature_adjust['fea_dim_linear_2'] = nn.Linear(self.config['set_size']* fea_shape[1][1], fea_dim//2)
                # linear: classification/regression
                self.linear_head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.GELU(),
                    nn.Linear(fea_dim, self.config['out_dimension']),
                )
    
    def forward(self, x):
        assert x.dim()==6, f"input shape is {B, S, C, D, H, W} and should be B, S, C, D, H, W."
        x = super().forward(x)
        
        B, S, C, D, H, W = x.shape
        if not self.config['Omi']:
            assert S == self.config['set_size'], f"set_size is {S} but should be {self.config['set_size']}"

        # 1. feature extraction
        fea_original = self.image_encoder(x)
        if self.config['Omi']:
            if 'fea_dim_linear_2' not in self.feature_adjust:
                # fea_1
                fea = fea_original
                BS, c, d, h, w = fea.shape
                fea = fea.view(BS, c, -1).transpose(1, 2)  # [BS, dhw, c]
                fea = self.feature_adjust['fea_num_conv1d_1'](fea)  # [BS, fea_num, hidden_size]
                fea_1 = self.feature_adjust['fea_dim_linear_1'](fea)  # [BS, fea_num, fea_dim]
                fea_comb = fea_1
            else:
                # fea_1
                fea = fea_original[0]
                BS, c, d, h, w = fea.shape
                fea = fea.view(BS, c, -1).transpose(1, 2)  # [BS, dhw, c]
                fea = self.feature_adjust['fea_num_conv1d_1'](fea)  # [BS, fea_num, hidden_size]
                fea_1 = self.feature_adjust['fea_dim_linear_1'](fea)  # [BS, fea_num, fea_dim]
                # fea_2
                fea = fea_original[1]
                BS, c, d, h, w = fea.shape
                fea = fea.view(BS, c, -1).transpose(1, 2)  # [BS, dhw, c]
                fea = self.feature_adjust['fea_num_conv1d_2'](fea)  # [BS, fea_num, hidden_size]
                fea_2 = self.feature_adjust['fea_dim_linear_2'](fea)  # [BS, fea_num, fea_dim]
                # fea_comb
                fea_comb = torch.cat((fea_1, fea_2), dim=1)  # [BS, fea_num*k, fea_dim], k=1/2
                
            # 2. attention fusion [batch, omi_dim]
            fea_comb = self.feature_norm(fea_comb)
            fea_comb = fea_comb.view(B, -1, self.config['fea_num_dim'][1])
            fea_final = self.set_attention(fea_comb)
            fea_final = fea_final.view(B, -1)
        else:
            if 'fea_dim_linear_2' not in self.feature_adjust:
                # fea_1
                fea = fea_original
                fea = self.feature_adjust['pool'](fea)
                fea = fea.view(B, -1)  # [B, S*c]
                fea_1 = self.feature_adjust['fea_dim_linear_1'](fea)  # [B, fea_dim]
                fea_1 = self.feature_norm(fea_1)
                fea_final = fea_1
                fea_final = self.feature_norm(fea_final)
            else:
                # fea_1
                fea = fea_original[0]
                fea = self.feature_adjust['pool'](fea)
                fea = fea.view(B, -1)  # [B, S*c]
                fea_1 = self.feature_adjust['fea_dim_linear_1'](fea)  # [B, fea_dim//2]
                # fea_2
                fea = fea_original[1]
                fea = self.feature_adjust['pool'](fea)
                fea = fea.view(B, -1)  # [B, S*c]
                fea_2 = self.feature_adjust['fea_dim_linear_2'](fea)  # [B, fea_dim//2]
                # fea_final
                fea_final = torch.cat((fea_1, fea_2), dim=1)  # [B, fea_dim]
                fea_final = self.feature_norm(fea_final)

        # 3. task head
        logits = self.linear_head(fea_final)
        return logits

def count_parameters(model):
    """count parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%% test
# examples
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    # model configs
    config= {'model_name': 'resnet50_slowonly_tpn', 
            'CDHW': (3, 48, 192, 192),
            'Omi': True,
            'set_size': 2,
            'fea_num_dim': (8, 512),
            'omi_dim': 128,
            'out_dimension': 2
            }

    for k,v in config.items():
        print(f"{k}: {v}")
    
    OmiModel = OmiLinear3D_mma2(config)
    OmiModel.to(device)
    
    # ------ forward
    # OmiModel 6D input [N, num_crops/set, C, T, H, W] 
    C, D, H, W = config['CDHW']
    set_size = config['set_size']
    with torch.no_grad():
        x = OmiModel(torch.randn(4, set_size, C, D, H, W).to(device))
        print(x.shape) 
    
    summary(OmiModel, input_size=(1, set_size, C, D, H, W), col_names=["input_size", "output_size", "num_params"]) 

    # print(OmiModel)
    # print_model_parameters(OmiModel)
