import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Union, Optional
from captum.attr import IntegratedGradients, GradientShap, Saliency, NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import LayerGradCam

class ModelInterpreter:
    def __init__(self, model):
        self.model = model.eval()

    def interpret(self, inputs, target, method="smoothgrad", target_layer=None):
        if method == "smoothgrad":
            saliency = Saliency(self.model)
            nt = NoiseTunnel(saliency)
            # ig = IntegratedGradients(self.model)
            # nt = NoiseTunnel(ig)
            attr = nt.attribute(inputs, nt_type="smoothgrad", target=target, nt_samples=25, stdevs=0.2)
        elif method == "smoothgrad_sq":
            saliency = Saliency(self.model)
            nt = NoiseTunnel(saliency)
            # ig = IntegratedGradients(self.model)
            # nt = NoiseTunnel(ig)
            attr = nt.attribute(inputs, nt_type="smoothgrad_sq", target=target, nt_samples=25, stdevs=0.2)
        elif method == "gradcam":
            if target_layer is None:
                raise ValueError("GradCAM requires target_layer argument")
            gradcam = LayerGradCam(self.model, target_layer)
            attr = gradcam.attribute(inputs, target=target)
            attr = torch.nn.functional.interpolate(attr, size=inputs.shape[2:], mode="bilinear")
        else:
            raise ValueError(f"Unknown method {method}")
        return attr

    def visualize(self, orig_img, attr, alpha=0.5, cmap='jet', save_path=None):
        """
        orig_img: 原始未标准化图像 (PIL.Image 或 numpy array)，范围 [0,255]
        attr: captum 生成的 attribution (tensor)
        method: 方法名称
        alpha: 热力图透明度
        """
        if orig_img is not None:
            if isinstance(orig_img, Image.Image):
                orig_img = np.array(orig_img.convert("RGB"))
            # if orig_img.max() > 1:  # 如果是 0-255，归一化到 0-1
            #     orig_img = orig_img.astype(np.float32) / 255.0

        # 转 numpy
        ori_heatmap = attr.squeeze().detach().cpu().numpy()

        if ori_heatmap.ndim == 3:  # GradCAM shape (C,H,W)
            ori_heatmap = np.mean(ori_heatmap, axis=0)
            
        if save_path:
            heatmap = np.maximum(ori_heatmap, 0)
            heatmap /= (heatmap.max() + 1e-8)
            heatmap = np.uint8(255 * heatmap)

            heatmap = cv2.applyColorMap(heatmap, getattr(cv2, f'COLORMAP_{cmap.upper()}') 
                                    if hasattr(cv2, f'COLORMAP_{cmap.upper()}') else cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))

            # # 转为彩色热力图
            # heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig_img.astype(np.uint8), 1 - alpha, heatmap, alpha, 0)
            # overlay = alpha *heatmap + (1-alpha) * orig_img
            overlay = overlay / 255.0
        
            plt.figure(figsize=(6,6))
            plt.imshow(overlay)
            plt.axis("off")
            plt.savefig(save_path)
            plt.close()

        return ori_heatmap

class BackboneExtractor(nn.Module):
    """
    Hook-based extractor: runs full model forward and captures activation at target_module
    using a forward hook. Returns the captured activation tensor.
    """
    def __init__(self, full_model: nn.Module, target_module: nn.Module):
        super().__init__()
        self.full_model = full_model
        self.target_module = target_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activ = {}

        def _hook(module, inp, outp):
            # detach to avoid keeping unnecessary graph; keep on same device
            activ['out'] = outp.clone()

        handle = self.target_module.register_forward_hook(_hook)
        # Run full forward — hook will capture activation
        _ = self.full_model(x)
        handle.remove()

        if 'out' not in activ:
            raise RuntimeError("Target module did not produce activation during forward pass.")
        return activ['out']


class ModelWithReadout(nn.Module):
    """
    Combined module: backbone (returns features) -> flatten (if needed) -> linear readout.
    """
    def __init__(self, backbone: nn.Module, linear: nn.Linear):
        super().__init__()
        self.backbone = backbone
        self.readout = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, start_dim=1)
        return self.readout(feats)


class ModelforVis(nn.Module):
    """
    ModelforVis expects `feature_extractor` to be an instance of your FeatureExtractor (has `.model` attribute).
    This class ONLY uses a hook-based extractor (no sequential re-assembly).
    """

    def __init__(self, feature_extractor):
        """
        feature_extractor: your FeatureExtractor instance (has .model and .device ideally)
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        # underlying nn.Module (actual model)
        self.model = getattr(feature_extractor, 'model', None)
        if self.model is None:
            raise ValueError("Provided feature_extractor does not expose `.model` attribute (expected nn.Module).")

        # backbone will be set after backbone_constructing()
        self.backbone: Optional[nn.Module] = None
        self._target_module = None

        # device inference
        self.device = None
        if hasattr(feature_extractor, 'device'):
            try:
                self.device = torch.device(feature_extractor.device)
            except Exception:
                # feature_extractor.device might already be torch.device or string
                self.device = torch.device(str(feature_extractor.device))
        else:
            # fallback to model param device
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')

    def _get_module_by_dotted_name(self, root: nn.Module, dotted: str) -> nn.Module:
        """
        Resolve dotted path to a submodule (e.g. 'features.10' or 'layer4.1.conv2').
        """
        cur = root
        if dotted == '':
            return cur
        for tok in dotted.split('.'):
            if tok.isdigit():
                idx = int(tok)
                children = list(cur.children())
                if idx < 0 or idx >= len(children):
                    raise AttributeError(f"Index {idx} out of range for module {cur}")
                cur = children[idx]
            else:
                if not hasattr(cur, tok):
                    raise AttributeError(f"Module {cur} has no attribute '{tok}'")
                cur = getattr(cur, tok)
        return cur

    def backbone_constructing(self, layer_name: str) -> nn.Module:
        """
        Build a hook-based backbone that returns the activation at `layer_name`.
        Always uses a hook-based extractor (no re-assembling).

        Returns:
            backbone module (callable): calling backbone(input) returns activation tensor at that layer.
        """
        if not isinstance(layer_name, str) or len(layer_name) == 0:
            raise ValueError("layer_name must be a non-empty dotted-path string (e.g. 'features.10')")

        # resolve submodule
        try:
            target_mod = self._get_module_by_dotted_name(self.model, layer_name)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve layer_name '{layer_name}': {e}")

        # create hook-based backbone extractor
        self.backbone = BackboneExtractor(self.model, target_mod)
        self._target_module = target_mod

        return self.backbone

    def readout_concat(
        self,
        coef: Union[np.ndarray, torch.Tensor, list, tuple],
        intercept: Union[float, np.ndarray, torch.Tensor, list, tuple],
        dummy_input: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Attach a linear readout created from coef & intercept onto the hook-based backbone.
        - coef: 1D (in_features,) or 2D (out_features, in_features)
        - intercept: scalar or 1D (out_features,)
        - dummy_input: optional tensor used to infer backbone output shape. If None, try sensible default (1,3,224,224).
        Returns:
            ModelWithReadout instance ready for inference.
        """

        if self.backbone is None:
            raise RuntimeError("Backbone not constructed. Call backbone_constructing(layer_name) first.")

        # decide device
        dev = self.device if self.device is not None else torch.device('cpu')

        # convert coef / intercept to tensors on device
        if isinstance(coef, np.ndarray) or isinstance(coef, (list, tuple)):
            coef_t = torch.tensor(np.asarray(coef), dtype=torch.float32, device=dev)
        elif isinstance(coef, torch.Tensor):
            coef_t = coef.to(dev).float()
        else:
            raise ValueError("Unsupported coef type")

        if isinstance(intercept, np.ndarray) or isinstance(intercept, (list, tuple)):
            bias_t = torch.tensor(np.asarray(intercept), dtype=torch.float32, device=dev)
        elif isinstance(intercept, torch.Tensor):
            bias_t = intercept.to(dev).float()
        else:
            # scalar
            bias_t = torch.tensor([float(intercept)], dtype=torch.float32, device=dev)

        # normalize coef shape
        if coef_t.ndim == 1:
            out_features = 1
            in_features = int(coef_t.shape[0])
            weight = coef_t.unsqueeze(0)  # (1, in_features)
        elif coef_t.ndim == 2:
            out_features, in_features = int(coef_t.shape[0]), int(coef_t.shape[1])
            weight = coef_t
        else:
            raise ValueError("coef must be 1D or 2D")

        # normalize bias shape
        if bias_t.ndim == 0:
            bias = bias_t.view(1)
        elif bias_t.ndim == 1:
            bias = bias_t
        else:
            raise ValueError("intercept must be scalar or 1D")

        if bias.shape[0] != out_features:
            if bias.shape[0] == 1 and out_features > 1:
                bias = bias.expand(out_features)
            else:
                raise ValueError(f"intercept length {bias.shape[0]} incompatible with out_features {out_features}")

        # create linear layer and load weights
        linear = nn.Linear(in_features, out_features).to(dev)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        # prepare dummy input to probe backbone output dimension
        if dummy_input is None:
            dummy_input = torch.zeros(1, 3, 224, 224, device=dev)
        else:
            dummy_input = dummy_input.to(dev)

        # run backbone to get feature shape (note: this will run full model forward and capture hook)
        with torch.no_grad():
            feats = self.backbone(dummy_input)

        # compute flattened feature dim
        if feats.dim() > 2:
            feat_dim = int(torch.prod(torch.tensor(feats.shape[1:], device=dev)).item())
        else:
            # feats expected shape [B, D]
            feat_dim = int(feats.shape[1])

        if feat_dim != in_features:
            raise ValueError(
                f"Backbone produced flattened feature dim {feat_dim}, but coef expects in_features {in_features}. "
                "If your model uses a different input size, provide a matching dummy_input to readout_concat."
            )

        # build ModelWithReadout and return
        model_vis = ModelWithReadout(self.backbone, linear).to(dev).eval()
        return model_vis


def heatmap_validation(attr, img, model, mask_value=0):
    """
    attr: (224, 224) numpy array, attribution heatmap
    img: (3, 224, 224) torch.Tensor, 原始图像，范围已适配模型
    model: PyTorch model
    mask_value: 用来替换被 mask 掉的像素，默认为 0
    """
    
    # Step 1: 计算原始预测类别
    model.eval()
    with torch.no_grad():
        ori_out = model(img).item()

    # Step 2: 把热图分成 16×16 patch，每个 patch 大小 14×14
    patch_size = 56
    H, W = attr.shape
    patch_scores = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch_attr = attr[i:i+patch_size, j:j+patch_size]
            score = patch_attr.sum()
            patch_scores.append(((i, j), score))
    
    # Step 3: 根据热图贡献排序 (从大到小)
    patch_scores = sorted(patch_scores, key=lambda x: x[1], reverse=True)
    pred_resp = []

    # Step 4: 依次 mask 每个 patch 并计算模型预测
    for (i, j), _ in patch_scores:
        masked_img = img.clone()
        masked_img[:, :, i:i+patch_size, j:j+patch_size] = mask_value
        with torch.no_grad():
            out = model(masked_img).item()
        pred_resp.append(ori_out - out)

    return pred_resp
    