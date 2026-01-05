# OmiMRI
OmiMRI: A Clinical-adaptive AI Framework for Format-Free Interpretation of Heterogeneous Brain MRIs    
Clinical brain MRI analysis faces a fundamental challenge: bridging the gap between oversimplified research developments and the inherent heterogeneity of real-world clinical practice. Quantifying this gap, our analysis of 26 MRI attributes across 22 clinical datasets reveals substantial heterogeneity across institutions and patients. Current AI tools typically require rigid input formats, necessitating extensive data exclusion or preprocessing that severely limits their real-world utility. Here we present OmiMRI, a unified, format-free framework designed to bridge this gap by enabling adaptive processing of arbitrary MRI combinations. Rather than strictly defining a standalone architecture, OmiMRI functions as a universal framework that integrates diverse pretrained 2D/3D convolutional and Transformer-based networks as feature encoders. Through a self-attention mechanism and dynamic weighting to fuse features from variable inputs, OmiMRI decouples clinical performance from rigid input specifications and enables adaptive processing of arbitrary MRI combinations.
Across 15 diverse classification, segmentation, and regression tasks, OmiMRI demonstrates robust input-scaling capabilities, yielding significant improvements over traditional fixed-input models. Notably, OmiMRI outperforms advanced medical imaging foundation models (e.g., BrainIAC and BrainSegFounder) in 94.4% of comparisons involving 2–4 input MRIs under consistent experimental conditions. Furthermore, the framework exhibits continuous performance gains through the incremental incorporation of multi-center data. In a clinically demanding, data-limited task distinguishing glioblastoma from metastasis, OmiMRI achieved diagnostic performance matching or exceeding that of senior neuroradiologists (AUROC 0.931 vs. 0.907, P > 0.05; AUPRC 0.973 vs. 0.930, P < 0.05), while providing interpretable attention maps aligned with radiological landmarks. Together, these results establish OmiMRI as a clinically adaptive AI paradigm that transform format-rigid modeling into flexible, expert-level systems capable of embracing the heterogeneity of real-world patient data.

## Setup & Install
### Install key tools
```
conda create -n omimri python=3.10
conda activate omimri
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmengine==0.10.3
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# process conflicts
pip install numpy==1.25.1 
pip install opencv-contrib-python==4.9.0.80
pip install opencv-python==4.9.0.80

pip install torchinfo
pip install timm==1.0.13
pip install importlib-metadata==7.1.0
```

### Install mmaction2 from source
Reference: https://mmaction2.readthedocs.io/en/latest/get_started/installation.html
```
cd path/to/OmiMRI-V1.0
# cd /gpfs/share/home/2306393443/sufengdata/tools-advanced/set_transformer/OmiMRI-V1.0
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without re-installation.
```

## Quick Start
### Pretrained backbone: resnet50_slowonly_tpn 
Reference: https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html#tpn   
Download pretrained resnet50_slowonly_tpn checkpoint.
```
cd path/to/OmiMRI-V1.0
cd mmaction2
mkdir modelzoo
cd modelzoo
wget https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth
```
File structure as follows:
```
cd path/to/OmiMRI-V1.0
    |—— mmaction2
        |—— configs
        |—— modelzoo
            |—— tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth
        |—— ...
    |—— models_mmaction2
        |—— build_extractor3D_from_mmaction2.py
        |—— build_OmiLinear3D_from_mmaction2.py
```
### Create and test backbone
```
cd path/to/OmiMRI-V1.0
cd models_mmaction2
python build_extractor3D_from_mmaction2.py
```
run and test build_extractor3D_from_mmaction2.py   
Get outputs and summary informations:   
    input shape: torch.Size([4, 3, 3, 48, 192, 192])    
    out 0, shape torch.Size([12, 1024, 8, 12, 12])    
    out 1, shape torch.Size([12, 2048, 8, 6, 6])   

### Create and test OmiMRI model
```
cd path/to/OmiMRI-V1.0
cd models_mmaction2
python build_OmiLinear3D_from_mmaction2.py
```
run and test build_OmiLinear3D_from_mmaction2.py    
Get outputs and summary informations:      
    input shape: torch.Size([4, 3, 3, 48, 192, 192])    
    out 0, shape torch.Size([12, 1024, 8, 12, 12])    
    out 1, shape torch.Size([12, 2048, 8, 6, 6])    
    torch.Size([4, 2])   
