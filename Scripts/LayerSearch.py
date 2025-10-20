import os
import sys
module_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(module_path)

import numpy as np
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
backbone_type = 'clip'  # 'timm', 'torchvision', 'clip', or 'open_clip'
model_name = 'ViT-B/16'
# ckpt_path = r"D:\Dataset\robust_weights\resnet50\resnet50_l2_eps0.ckpt"
ckpt_path = None
batch_size = 64
# vit_layer = 17
# layers_to_test = [f'blocks.{vit_layer}.norm1', f'blocks.{vit_layer}.attn.qkv', f'blocks.{vit_layer}.attn.q_norm', f'blocks.{vit_layer}.attn.k_norm', f'blocks.{vit_layer}.attn.proj', f'blocks.{vit_layer}.ls1', f'blocks.{vit_layer}.norm2', f'blocks.{vit_layer}.mlp.fc1', f'blocks.{vit_layer}.mlp.act', f'blocks.{vit_layer}.mlp.norm', f'blocks.{vit_layer}.mlp.fc2', f'blocks.{vit_layer}.ls2']
layers_to_test = ['transformer.resblocks.6.ln_1', 'transformer.resblocks.6.mlp.c_fc', 'transformer.resblocks.6.mlp.gelu',  'transformer.resblocks.7.ln_1', 'transformer.resblocks.7.mlp.c_fc', 'transformer.resblocks.7.mlp.gelu', 'transformer.resblocks.7.mlp.c_proj', 'transformer.resblocks.7.ln_2', 'transformer.resblocks.8.ln_1', 'transformer.resblocks.8.mlp.c_fc', 'transformer.resblocks.8.mlp.gelu', 'transformer.resblocks.8.mlp.c_proj', 'transformer.resblocks.8.ln_2']
use_amp = True
device = 'cuda'

# Encoding params
cv_folds = 5
cv_mode = 'simple' # 'simple' or 'nested'
len_chunk = None  # set to None to disable chunk
pca_comps = 50  # set to None to disable PCA

# Path setting
save_dir = r"D:\Analysis\results\LayerSearch"
image_folder = r"D:\Analysis\NSD_Alignment\NSD_shared1000"
neural_path = r"D:\Analysis\Ephys_data_59ver.npz"

default_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

seed = 42
np.random.seed(seed)
# ---- Main processing ----
# Load neural responses
data = np.load(neural_path)
y, nc = data['data'], data['nc']
n_units = nc.shape[0]
# n_random_units = 1000
# random_idx = np.random.permutation(n_units)[:n_random_units]
# y, nc = y[:, random_idx], nc[random_idx]
print(y.shape, nc.shape)

# Initialize feature extractor
extractor = get_extractor(backbone_type, model_name, device, use_amp)
preprocess = extractor.get_preprocess() or default_preprocess

# Prepare image dataset and loader
dataset = ImageFolderDataset(image_folder, transform=preprocess)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 按 layer 分批提取特征并拼接，然后训练 Encoder
scores, best_unit_list = [], []
for layer in layers_to_test:
    print(f"Processing layer: {layer}")
    # 对当前 layer 分批提取
    feat_batches = []
    for batch in tqdm(loader):
        cur_batch_size = batch.shape[0]
        imgs_batch = batch.to(device)                  
        with torch.no_grad():
            feats_batch = extractor(imgs_batch, layer)
            feats_batch = ensure_batch_first(feats_batch, batch_size=cur_batch_size)
        feat_batches.append(feats_batch.cpu())

    # 将所有 batch 拼接成 (N, D)
    feats_all = torch.cat(feat_batches, dim=0)
    # Process
    if len(feats_all.shape) > 2:
        feats_np = feats_all.numpy().reshape(feats_all.shape[0], -1)
    else:
        feats_np = feats_all.numpy()

    enc = Encoder(method='Ridge', cv_splits=cv_folds, pca_components=pca_comps)
    if not len_chunk:
        if cv_mode == 'simple':
            enc.fit(feats_np, y, nc)
            score = np.mean(enc.best_scores_)
            best_unit_idx = np.argmax(enc.best_scores_)
        elif cv_mode == 'nested':
            _, best_score = enc.fit_nested_cv(feats_np, y, nc)
            score = np.mean(best_score)
            best_unit_idx = np.argmax(best_score)
    else:
        n_chunk = int(np.ceil(n_units / len_chunk))
        score = np.array([])
        best_unit_idx = 0
        for i in range(n_chunk):
            print("Chunk idx:", i)
            start_idx, end_idx = i * len_chunk, (i+1) * len_chunk
            cur_feats, cur_y, cur_nc = feats_np, y[:, start_idx:end_idx], nc[start_idx:end_idx]
            print(cur_feats.shape, cur_y.shape, cur_nc.shape)
            if cv_mode == 'simple':
                enc.fit(cur_feats, cur_y, cur_nc)
                score = enc.best_scores_ if score.size == 0 else np.concatenate([score, enc.best_scores_])
                best_unit_idx = np.argmax(enc.best_scores_)
            elif cv_mode == 'nested':
                _, best_score = enc.fit_nested_cv(feats_np, y, nc)
                score = best_score if score.size == 0 else np.concatenate([score, best_score])
                best_unit_idx = np.argmax(best_score)
        score = np.mean(score)
        
    print(f"Layer {layer}: CV Pearson r = {score:.4f}")
    scores.append(score)
    best_unit_list.append(best_unit_idx)

# Find and report best layer
best_idx = int(np.argmax(scores))
best_layer = layers_to_test[best_idx]
best_score = scores[best_idx]
print(f"\nBest layer: {best_layer} with CV Pearson r = {best_score:.4f}")
print(f"\nBest unit idx: {best_unit_list[best_idx]}")

# Plot results
save_path = os.path.join(save_dir, "vit-"+best_layer+".png")
plt.figure()
plt.plot(layers_to_test, scores, marker='o')
plt.xlabel('Layer')
plt.ylabel('CV Pearson r')
plt.title('Encoding performance by layer')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
plt.close()