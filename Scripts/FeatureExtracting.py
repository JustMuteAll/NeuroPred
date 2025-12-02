import os
import sys
module_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(module_path)

import numpy as np
from scipy.io import savemat
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from NeuroPredictor.FeatExtractor import (
    TimmFeatureExtractor,
    TorchvisionFeatureExtractor,
    CLIPFeatureExtractor,
    OpenCLIPFeatureExtractor
)
from NeuroPredictor.Encoder import Encoder
from tqdm import tqdm
from NeuroPredictor.Utils import ImageFolderDataset, get_extractor, ensure_batch_first

# Model params
backbone_type = 'ssl'  # 'timm', 'torchvision', 'clip', 'open_clip' or 'ssl'
model_name = 'simclr_rn50'
ckpt_path = r"D:\Dataset\ssl_weights" # or None
batch_size = 64
layer_to_test = 'layer4.1.relu'
use_amp = True
device = 'cuda'

# Path setting
save_fmt = '.npy' # '.npy' or '.mat'
save_dir = r"D:\Analysis\results\LayerSearch"
data_name = '_'.join([backbone_type, model_name, layer_to_test])
save_path = os.path.join(save_dir, data_name + save_fmt)
image_folder = r"D:\Analysis\NSD_Alignment\NSD_shared1000"

default_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
# ---- Main processing ----
# Initialize feature extractor
extractor = get_extractor(backbone_type, model_name, device, use_amp, ckpt_path=ckpt_path)
preprocess = extractor.get_preprocess() or default_preprocess

# Prepare image dataset and loader
dataset = ImageFolderDataset(image_folder, transform=preprocess)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 按 layer 分批提取特征并拼接
print(f"Processing layer: {layer_to_test}")
# 对当前 layer 分批提取
feat_batches = []
for batch in tqdm(loader):
    cur_batch_size = batch.shape[0]
    imgs_batch = batch.to(device)                  
    with torch.no_grad():
        feats_batch = extractor(imgs_batch, layer_to_test)
        feats_batch = ensure_batch_first(feats_batch, batch_size=cur_batch_size)
    feat_batches.append(feats_batch.cpu())

# 将所有 batch 拼接成 (N, D)
feats_all = torch.cat(feat_batches, dim=0)
# Process
if len(feats_all.shape) > 2:
    feats_np = feats_all.numpy().reshape(feats_all.shape[0], -1)
else:
    feats_np = feats_all.numpy()
print(feats_np.shape)

if save_fmt == '.npy':
    np.save(save_path, feats_np)
elif save_fmt == '.mat':
    savemat(save_path, {'feat': feats_np})
else:
    raise ValueError("fmt 仅支持 'npy' 或 'mat'")

print("Saving done.")




