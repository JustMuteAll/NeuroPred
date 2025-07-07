import os
import sys
module_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(module_path)

import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from NeuroPredictor.FeatExtractor import (
    TimmFeatureExtractor,
    TorchvisionFeatureExtractor,
    CLIPFeatureExtractor,
    OpenCLIPFeatureExtractor
)
from NeuroPredictor.Encoder import Encoder
from NeuroPredictor.brain_guide_pipeline import mypipelineSAG
from diffusers import DPMSolverMultistepScheduler

# ==============================
#      Hyperparameters
# ==============================
# Backbone & feature extraction
backbone_type   = 'open_clip'                 # 'timm', 'torchvision', 'clip', 'open_clip'
model_name   = 'ViT-B/32'
model_layer  = 'transformer.resblocks.9'                   # layer index or module name
batch_size     = 32
device          = 'cuda'

# Neural data & ROI
roi_name        = 'face'                 # for output folder naming
image_folder    = r"D:\Analysis\NSD_Alignment\NSD_shared1000"
neural_path  = r"D:\Analysis\NeuroPredictor\Ephys_data_Face.npz"
save_dir      = rf"D:\Analysis\results\figures_{roi_name}_{backbone_type}"

# Encoder (ridge) & PCA
enc_method      = 'Ridge'
cv_folds        = 5
ridge_alphas    = np.logspace(-3, 3, 7)
pca_comps       = None                   # set to None to disable PCA

# Diffusion / MEI generation
num_to_generate = 10
num_steps       = 50
clip_scale      = 30
seed_low        = 1
seed_high       = 10000
diffusion_model = "stabilityai/stable-diffusion-2-1-base"

# ==============================
#      Helper Functions
# ==============================
class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png','jpg','jpeg','bmp','tiff'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def standardize(arr: np.ndarray, axis: int) -> np.ndarray:
    mean = arr.mean(axis=axis, keepdims=True)
    std  = arr.std(axis=axis, keepdims=True)
    std_safe = np.where(std==0, 1, std)
    return (arr - mean) / std_safe

def normalize(arr: np.ndarray, axis: int) -> np.ndarray:
    norm = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)
    norm[norm==0] = 1
    return arr / norm

# ==============================
#   Feature Extraction Logic
# ==============================
BACKBONE_CONFIG = {
    'timm': {
        'class': TimmFeatureExtractor,
        'init_kwargs': {'model_name':model_name if model_name is not None else 'vit_base_patch16_clip_224.laion2b'},
        'hook_args':    {'layer_or_names': model_layer, 'embedtype':'spatial'},
        'transform':    transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466,0.4578275,0.40821073],
                std =[0.26862954,0.26130258,0.27577711]
            )
        ])
    },
    'torchvision': {
        'class': TorchvisionFeatureExtractor,
        'init_kwargs': {'model_name':model_name if model_name is not None else 'resnet50'},
        'hook_args':    {'module_names':model_layer},
        'transform': None
    },
    'clip': {
        'class': CLIPFeatureExtractor,
        'init_kwargs': {'model_name':model_name if model_name is not None else 'RN50'},
        'hook_args':    {'module_names':model_layer},
        'transform': None
    },
    'open_clip': {
        'class': OpenCLIPFeatureExtractor,
        'init_kwargs': {'model_name':model_name if model_name is not None else 'ViT-B/32','pretrained': r'D:\neuropredictor\checkpoints\CLIPAG_ViTB32.pt'},
        'hook_args':    {'module_names':model_layer},
        'transform': None
    }
}

def extract_features(backbone_type, img_folder, batch_size, device):
    cfg = BACKBONE_CONFIG[backbone_type]
    model = cfg['class'](**cfg['init_kwargs']).to(device).eval()
    for p in model.parameters(): p.requires_grad=False

    transform = cfg['transform'] or model.get_preprocess()
    dataset   = ImageFolderDataset(img_folder, transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feats = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            out  = model(imgs, **cfg['hook_args'])
            feats.append(out.cpu().numpy())

    features = np.concatenate(feats, axis=0)
    return features, model

# ==============================
#          Main Script
# ==============================
def main():
    os.makedirs(save_dir, exist_ok=True)

    print("Start feature extraction")
    # 1) Extract image features
    start = time.time()
    features, backbone = extract_features(
        backbone_type, image_folder, batch_size, device
    )
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    print("End, Total time:", time.time() - start)

    # 2) Load & preprocess neural data
    merged = np.load(neural_path)
    # concatenate only roi_name entries
    y = merged['data']
    y = standardize(y, axis=0)
    X = normalize(features, axis=1)

    # 3) Fit Encoder
    print("Start Encoder fitting")
    start = time.time()
    enc = Encoder(
        method=enc_method,
        ridge_alphas=ridge_alphas,
        pca_comps=pca_comps,
        cv_folds=cv_folds
    )
    enc.fit(X, y)
    coef, intercept = enc.coef_, enc.intercept_
    print("End, Total time:", time.time() - start)

    # 4) Prepare diffusion pipeline
    pipe = mypipelineSAG.from_pretrained(
        diffusion_model, torch_dtype=torch.float16, revision="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    coeff_t   = torch.from_numpy(coef.astype(np.float16)).to(device)
    inter_t   = torch.from_numpy(intercept.astype(np.float16)).to(device)

    # 5) Generate
    seeds = np.random.RandomState(42).randint(seed_low, seed_high, size=num_to_generate)
    pipe.brain_tweak = lambda img: brain_loss(img, backbone, backbone_type,
                                              model_layer, coeff_t, inter_t)
    print("Start MEI generating")
    start = time.time()
    for i, sd in enumerate(seeds):
        g     = torch.Generator(device=device).manual_seed(int(sd))
        out   = pipe(
            "",
            sag_scale=0,
            guidance_scale=0,
            num_inference_steps=num_steps,
            generator=g,
            clip_guidance_scale=clip_scale
        )
        out.images[0].save(
            os.path.join(save_dir, f"mei_{i:05d}.png"),
            format="PNG", compress_level=6
        )
        print(f"Saved image {i+1}/{num_to_generate}")
    print("End, Total time:", time.time() - start)

def brain_loss(image_input, backbone, btype, layer, coef, intercept):
    img_feat = (
        backbone(image_input, layer_or_names=layer, embedtype='spatial')
        if btype=='timm' else
        backbone(image_input, module_names=layer)
    )
    img_feat = img_feat.half()
    if img_feat.dim() > 2:
        img_feat = img_feat.reshape(img_feat.shape[0], -1)
    normed   = img_feat / img_feat.norm(p=2, dim=1, keepdim=True)
    pred     = normed @ coef.T + intercept
    return -pred.mean()

if __name__ == "__main__":
    main()
