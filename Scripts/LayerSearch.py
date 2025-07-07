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


# ==== User-defined settings (replace these with your own paths and parameters) ====
backbone_type = 'clip'  # 'timm', 'torchvision', 'clip', or 'open_clip'
model_name = 'RN50' # 
layers_to_test = ['avgpool','layer1','layer2','layer3','layer4']
batch_size = 32
use_amp = True
device = 'cuda'

save_dir = r"D:\Analysis\results\LayerSearch"
image_folder = r"D:\Analysis\NSD_Alignment\NSD_shared1000"
neural_path = r"D:\Analysis\NeuroPredictor\Ephys_data_59ver.npz"
# ==============================================================================
def load_responses(path):
    """
    Load neural responses from a .npy or .npz file or CSV with numpy.
    Assumes shape (N,) or (N, D).
    """
    return NotImplemented


class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_extractor(backbone_type, model_name, device, amp):
    backbone_type = backbone_type.lower()
    if backbone_type == 'timm':
        return TimmFeatureExtractor(model_name, pretrained=True, device=device, amp=amp)
    if backbone_type == 'torchvision':
        return TorchvisionFeatureExtractor(model_name, pretrained=True, device=device, amp=amp)
    if backbone_type == 'clip':
        return CLIPFeatureExtractor(model_name, device=device, amp=amp)
    if backbone_type == 'open_clip':
        return OpenCLIPFeatureExtractor(model_name, pretrained=True, device=device)
    raise ValueError(f"Unknown network type {backbone_type}")

default_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

# ---- Main processing ----
# 1. Load neural responses
data = np.load(neural_path)
y, nc = data['data'], data['nc']
print(y.shape, nc.shape)

# 2. Initialize feature extractor
extractor = get_extractor(backbone_type, model_name, device, use_amp)
preprocess = extractor.get_preprocess() or default_preprocess

# 3. Prepare image dataset and loader
dataset = ImageFolderDataset(image_folder, transform=preprocess)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 4–5. 按 layer 分批提取特征并拼接，然后训练 Encoder
scores = []
for layer in layers_to_test:
    print(f"Processing layer: {layer}")

    # 对当前 layer 分批提取
    feat_batches = []
    for batch in tqdm(loader):
        # 假设 loader 只返回图像张量
        imgs_batch = batch.to(device)                   # 将这一批数据放到正确设备
        with torch.no_grad():
            feats_batch = extractor(imgs_batch, layer)  # (B, D) 或 {…}
        # 如果 extractor 返回 dict，改为 feats_batch = feats_batch[layer]
        feat_batches.append(feats_batch.cpu())

    # 将所有 batch 拼接成 (N, D)
    feats_all = torch.cat(feat_batches, dim=0)
    # Process
    if len(feats_all.shape) > 2:
        feats_np = feats_all.numpy().reshape(feats_all.shape[0], -1)
    else:
        feats_np = feats_all.numpy()
    print(feats_np.shape)
    # 用 Encoder 做 5 折 CV PLS
    enc = Encoder(method='Ridge', cv_folds=5, pca_comps=500)
    enc.fit(feats_np, y, nc)
    score = np.mean(enc.cv_r_)
    print(f"Layer {layer}: CV Pearson r = {score:.4f}")
    # print(enc.best_params_, np.sum(enc.pca_.explained_variance_ratio_))
    scores.append(score)

# 6. Find and report best layer
best_idx = int(np.argmax(scores))
best_layer = layers_to_test[best_idx]
best_score = scores[best_idx]
print(f"\nBest layer: {best_layer} with CV Pearson r = {best_score:.4f}")

# 7. Plot results
save_path = os.path.join(save_dir, model_name+"-"+best_layer+".png")
plt.figure()
plt.plot(layers_to_test, scores, marker='o')
plt.xlabel('Layer')
plt.ylabel('CV Pearson r')
plt.title('Encoding performance by layer')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
plt.close()