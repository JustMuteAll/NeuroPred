import os
import h5py
import torch
import numpy as np
from scipy.stats import pearsonr
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from NeuroPredictor.FeatExtractor import (
    TimmFeatureExtractor,
    TorchvisionFeatureExtractor,
    CLIPFeatureExtractor,
    OpenCLIPFeatureExtractor,
    SSLFeatureExtractor
)

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def get_extractor(backbone_type, model_name, device, amp, ckpt_path):
    backbone_type = backbone_type.lower()
    if backbone_type == 'timm':
        return TimmFeatureExtractor(model_name, pretrained=True, device=device, amp=amp)
    elif backbone_type == 'torchvision':
        return TorchvisionFeatureExtractor(model_name, ckpt_path=ckpt_path, device=device, amp=amp)
    elif backbone_type == 'clip':
        return CLIPFeatureExtractor(model_name, device=device, amp=amp)
    elif backbone_type == 'open_clip':
        return OpenCLIPFeatureExtractor(model_name, pretrained=True, device=device)
    elif backbone_type == 'ssl':
        return SSLFeatureExtractor(model_name, device=device, amp=amp, weights_dir=ckpt_path)
    raise ValueError(f"Unknown network type {backbone_type}")

def ensure_batch_first(feats, batch_size=None, seq_len_expected=None):
    """
    将 3D 特征标准化为 (B, S, C) 形式。
    支持输入 (B,S,C) 或 (S,B,C) 两种常见格式。
    如果不是 3D tensor 则原样返回（你可能拿到 cls vector -> (B,C)）。
    """
    if not torch.is_tensor(feats):
        feats = torch.as_tensor(feats)

    if feats.ndim != 3:
        return feats

    a, b, c = feats.shape

    # 如果给了 batch_size，优先用它判断
    if batch_size is not None:
        if a == batch_size:
            return feats  # already (B, S, C)
        if b == batch_size:
            return feats.permute(1, 0, 2).contiguous()  # (S,B,C) -> (B,S,C)
        
def visualize_tuning(resp, unit, img_dir, save_dir):
    # 获取所有图像路径（排序以保证和 resp 顺序一致）
    img_files = sorted(os.listdir(img_dir))
    img_paths = [os.path.join(img_dir, f) for f in img_files]

    # 取该 unit 的反应
    unit_resp = resp[:, unit]

    # 按反应排序
    sorted_idx = np.argsort(unit_resp)
    top_idx = sorted_idx[-5:][::-1]  # 最大的5个
    bottom_idx = sorted_idx[:5]      # 最小的5个

    # --- figure layout ---
    fig = plt.figure(figsize=(18, 12))

    # (1) 折线图
    ax_line = plt.subplot2grid((3, 5), (0, 0), colspan=5)
    ax_line.plot(np.sort(unit_resp), marker='o')
    ax_line.set_title(f'Unit {unit} Response (original order)')
    ax_line.set_xlabel('Image index')
    ax_line.set_ylabel('Response')

    # (2) Top-5 images
    for i, idx in enumerate(top_idx):
        ax = plt.subplot2grid((3, 5), (1, i))
        img = plt.imread(img_paths[idx])
        ax.imshow(img)
        ax.set_title(f"Top {i+1}\n{img_files[idx]}\nResp={unit_resp[idx]:.2f}")
        ax.axis('off')

    # (3) Bottom-5 images
    for i, idx in enumerate(bottom_idx):
        ax = plt.subplot2grid((3, 5), (2, i))
        img = plt.imread(img_paths[idx])
        ax.imshow(img)
        ax.set_title(f"Bottom {i+1}\n{img_files[idx]}\nResp={unit_resp[idx]:.2f}")
        ax.axis('off')

    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"unit_{unit}_tuning.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")