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
from tqdm import tqdm

from NeuroPredictor.Encoder import Encoder
from NeuroPredictor.FeatVis import ModelforVis, ModelInterpreter, heatmap_validation
from NeuroPredictor.Utils import ImageFolderDataset, get_extractor

# Model params
backbone_type = 'torchvision'  # 'timm', 'torchvision', 'clip', or 'open_clip'
model_name = 'vit_l_16'
# ckpt_path = r"D:\Dataset\ss_weights\resnet50\dino_resnet50_pretrain.pth"
ckpt_path = None
# layers_to_test = ['layer2','layer3','layer4']
target_layer = 'encoder.layers.encoder_layer_16.mlp.4'
batch_size = 32
use_amp = True
device = 'cuda'
unit_idx = 15132

# Encoding params
cv_folds = 5
pca_comps = None     # set to None to disable PCA

# Path setting
save_dir = r"D:\Analysis\results\Visualization"
model_dir = model_name + target_layer
cur_save_dir = os.path.join(save_dir, model_dir)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(cur_save_dir, exist_ok=True)
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

# ---- Main processing ----
data = np.load(neural_path)
y, nc = data['data'], data['nc']
y, nc = y[:, unit_idx], nc[unit_idx]
print(y.shape, nc.shape)

# Initialize feature extractor
extractor = get_extractor(backbone_type, model_name, device, use_amp, ckpt_path=None)
preprocess = extractor.get_preprocess() or default_preprocess

# Prepare image dataset and loader
dataset = ImageFolderDataset(image_folder, transform=preprocess)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 按 layer 分批提取特征并拼接，然后训练 Encoder
scores = []
print(f"Processing layer: {target_layer}")
# 对当前 layer 分批提取
feat_batches = []
for batch in tqdm(loader):
    imgs_batch = batch.to(device)                  
    with torch.no_grad():
        feats_batch = extractor(imgs_batch, target_layer)  
    feat_batches.append(feats_batch.cpu())

# 将所有 batch 拼接成 (N, D)
feats_all = torch.cat(feat_batches, dim=0)
# Process
if len(feats_all.shape) > 2:
    feats_np = feats_all.numpy().reshape(feats_all.shape[0], -1)
else:
    feats_np = feats_all.numpy()

enc = Encoder(method='Ridge', cv_splits=cv_folds, pca_components=pca_comps)
cv_mode = 'simple' # 'simple' or 'nested'
if cv_mode == 'simple':
    enc.fit(feats_np, y, nc)
    score = np.mean(enc.best_scores_)
elif cv_mode == 'nested':
    _, best_score = enc.fit_nested_cv(feats_np, y, nc)
    score = np.mean(best_score)
print(f"Layer {target_layer}: CV Pearson r = {score:.4f}")
print(enc.coef_.shape, enc.intercept_.shape)

# 构建 hook-based backbone 到 AlexNet 的 FC6（classifier.1）
mv = ModelforVis(extractor)
backbone = mv.backbone_constructing(target_layer)
# 如果你的 model expects 224x224, 不用传 dummy_input；否则传入合适的 dummy (e.g. 1x3xH xW)
dummy = torch.randn(1, 3, 224, 224, device=mv.device, requires_grad=True)

model_vis = mv.readout_concat(enc.coef_, enc.intercept_, dummy_input=dummy)
model_vis.eval()

# Load neural responses
num_images = 1000
img_path_list = os.listdir(image_folder)
heatmap_list = []
mean_pred_resp = np.array([])

for img_idx, img_path in tqdm(enumerate(img_path_list[:num_images])):
    # 你可以换成自己的图片路径
    img_path = os.path.join(image_folder, img_path)
    img = Image.open(img_path).convert("RGB")
    inputs = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]

    interp = ModelInterpreter(model_vis)
    method = "smoothgrad" # "smoothgrad", "smoothgrad_sq", "gradcam"
    if method == "gradcam":
        attr = interp.interpret(method=method, inputs=inputs, target=None, target_layer=model_vis.layer4[-1])
    else:
        attr = interp.interpret(method=method, inputs=inputs, target=None)
    
    # ====== 可视化结果 ======
    save_path = os.path.join(cur_save_dir, f"{img_idx + 1}.png")
    cur_heatmap = interp.visualize(img, attr, alpha=0.7, cmap='plasma',save_path=save_path)
    heatmap_list.append(cur_heatmap)

    pred_resp = np.array(heatmap_validation(attr=cur_heatmap, img=inputs, model=model_vis))
    if mean_pred_resp.size == 0:
        mean_pred_resp = pred_resp
    else:
        mean_pred_resp += pred_resp
    print(f"image {img_idx + 1} completed")

heatmap_array = np.array(heatmap_list)
print(heatmap_array.shape)
# np.save(os.path.join(cur_save_dir, "heatmap.npy"), heatmap_array)

mean_pred_resp /= num_images
plt.plot(mean_pred_resp)
axes = plt.gca()
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.xlabel('Batch ranking')
plt.ylabel('Predicted response')
plt.savefig(os.path.join(cur_save_dir, "mean_pred_resp.png"))
plt.close()