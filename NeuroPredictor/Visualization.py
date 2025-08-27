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

    def visualize(self, orig_img, attr, alpha=0.5, cmap='jet'):
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
        heatmap = attr.squeeze().detach().cpu().numpy()

        if heatmap.ndim == 3:  # GradCAM shape (C,H,W)
            heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
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
        plt.show()

class BackboneExtractor(nn.Module):
    """
    Safe fallback extractor: run full model forward and capture activation of target module
    via a forward hook. Returns the activation tensor.
    """
    def __init__(self, model: nn.Module, target_module: nn.Module):
        super().__init__()
        self.model = model
        self.target_module = target_module

    def forward(self, x):
        activ = {}

        def _hook(module, _inp, outp):
            activ['out'] = outp

        handle = self.target_module.register_forward_hook(_hook)
        _ = self.model(x)
        handle.remove()
        if 'out' not in activ:
            raise RuntimeError("Target module did not produce an activation during forward.")
        return activ['out']


class ModelWithReadout(nn.Module):
    """
    Simple wrapper: backbone -> flatten (if needed) -> linear readout
    """
    def __init__(self, backbone: nn.Module, linear: nn.Linear):
        super().__init__()
        self.backbone = backbone
        self.readout = linear

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, start_dim=1)
        return self.readout(feats)


class ModelforVis(nn.Module):
    """
    ModelforVis expects `model` to be a FeatureExtractor instance (subclass of FeatureExtractorBase).
    Methods:
      - backbone_constructing(layer_name): build efficient truncated backbone if possible,
          otherwise fallback to hook-based BackboneExtractor.
      - readout_concat(coef, intercept, dummy_input=None): attach linear readout built from coef/bias.
    """
    def __init__(self, model):
        super().__init__()
        # model is expected to be a FeatureExtractorBase instance (with .model attribute)
        self.feature_extractor = model
        self.model = model.model  # underlying nn.Module
        self.backbone: Optional[nn.Module] = None
        self._target_module = None
        self._truncated_via_sequential = False
        # device inference
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')

    def _get_module_by_dotted_name(self, root: nn.Module, dotted: str) -> nn.Module:
        """
        Resolve dotted path (e.g. 'features.10' or 'layer4.1.conv2') into a submodule object.
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

    def _attempt_build_sequential_trunc(self, root: nn.Module, dotted: str) -> Optional[nn.Module]:
        """
        Try to build an explicit truncated nn.Sequential that executes only up to `dotted`.
        Returns nn.Sequential if successful, otherwise None.

        Strategy (best-effort):
        - Walk down attribute tokens.
        - Whenever encountering an attribute that is nn.Sequential (or ModuleList), we include
          its children up to the desired index.
        - For simple common cases (alexnet.features[:k], vgg.features[:k], resnet.layerX[:k]), this works.
        - If we encounter arbitrary attribute/methods that can't be represented as simple sequential
          concatenation, we bail out and return None (so fallback will use hook).
        """
        tokens = dotted.split('.')
        cur = root
        seq_parts = []  # parts to include in final Sequential

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            # If token refers to attribute on cur
            if not tok.isdigit():
                if not hasattr(cur, tok):
                    return None
                attr = getattr(cur, tok)
                # If attr is Sequential-like and next token is index or we need the whole attr
                if isinstance(attr, (nn.Sequential, nn.ModuleList)):
                    # If next token is index, consume index and include children up to that index
                    next_i = i + 1
                    if next_i < len(tokens) and tokens[next_i].isdigit():
                        target_idx = int(tokens[next_i])
                        # include children up to target_idx (inclusive)
                        children = list(attr.children())[:target_idx + 1]
                        if len(children) == 0:
                            return None
                        seq_parts.append(nn.Sequential(*children))
                        # set cur to the chosen child (the module at target_idx)
                        cur = list(attr.children())[target_idx]
                        i += 2
                        continue
                    else:
                        # include whole attr (safe), and continue descending into it
                        seq_parts.append(attr)
                        cur = attr
                        i += 1
                        continue
                else:
                    # attr is a module but not sequential; include it as a single block
                    seq_parts.append(attr)
                    cur = attr
                    i += 1
                    continue
            else:
                # token is digit, but cur must be indexable (e.g., cur is Sequential)
                idx = int(tok)
                if isinstance(cur, (nn.Sequential, nn.ModuleList)):
                    children = list(cur.children())
                    if idx >= len(children):
                        return None
                    # include children up to idx
                    seq_parts.append(nn.Sequential(*children[:idx + 1]))
                    cur = children[idx]
                    i += 1
                    continue
                else:
                    return None

        # Now combine seq_parts into one nn.Sequential (flatten any nested Sequentials)
        flat_modules = []
        for part in seq_parts:
            if isinstance(part, nn.Sequential):
                for m in part.children():
                    flat_modules.append(m)
            else:
                flat_modules.append(part)
        if len(flat_modules) == 0:
            return None
        try:
            truncated = nn.Sequential(*flat_modules)
            return truncated
        except Exception:
            return None

    def backbone_constructing(self, layer_name: str) -> nn.Module:
        """
        Build self.backbone truncated at `layer_name`.
        Attempts efficient truncation (building nn.Sequential). If not possible, falls back to hook-based extractor.
        Returns the backbone module.
        """
        if not isinstance(layer_name, str) or len(layer_name) == 0:
            raise ValueError("layer_name must be a non-empty string path (e.g. 'features.10')")

        # Resolve the target module
        try:
            target_mod = self._get_module_by_dotted_name(self.model, layer_name)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve layer_name '{layer_name}': {e}")

        # Try building efficient Sequential truncation
        truncated = self._attempt_build_sequential_trunc(self.model, layer_name)
        if truncated is not None:
            # Ensure module on correct device
            truncated = truncated.to(self.device).eval()
            self.backbone = truncated
            self._target_module = target_mod
            self._truncated_via_sequential = True
            return self.backbone

        # Fallback: hook-based extractor (safe)
        self.backbone = BackboneExtractor(self.model, target_mod)
        self._target_module = target_mod
        self._truncated_via_sequential = False
        return self.backbone

    def readout_concat(
        self,
        coef: Union[np.ndarray, torch.Tensor, list, tuple],
        intercept: Union[float, np.ndarray, torch.Tensor, list, tuple],
        dummy_input: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """
        Attach linear readout from coef & intercept onto the current backbone.
        - coef: shape (in_features,) or (out_features, in_features)
        - intercept: scalar or shape (out_features,)
        - dummy_input: optional tensor to probe backbone output shape (if None, will try sensible defaults)
        Returns combined model (ModelWithReadout).
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not constructed. Call backbone_constructing(layer_name) first.")

        # convert coef & intercept to tensors
        if isinstance(coef, np.ndarray) or isinstance(coef, list) or isinstance(coef, tuple):
            coef_t = torch.tensor(np.asarray(coef), dtype=torch.float32, device=self.device)
        else:
            coef_t = coef.to(self.device).float()

        if isinstance(intercept, np.ndarray) or isinstance(intercept, list) or isinstance(intercept, tuple):
            bias_t = torch.tensor(np.asarray(intercept), dtype=torch.float32, device=self.device)
        elif isinstance(intercept, torch.Tensor):
            bias_t = intercept.to(self.device).float()
        else:
            bias_t = torch.tensor([float(intercept)], dtype=torch.float32, device=self.device)

        # reshape coef into (out_features, in_features)
        if coef_t.ndim == 1:
            out_features = 1
            in_features = int(coef_t.shape[0])
            weight = coef_t.unsqueeze(0)  # (1, in_features)
        elif coef_t.ndim == 2:
            out_features, in_features = int(coef_t.shape[0]), int(coef_t.shape[1])
            weight = coef_t
        else:
            raise ValueError("coef must be 1D or 2D (numpy or torch tensor)")

        # bias shape
        if bias_t.ndim == 0:
            bias = bias_t.view(1)
        elif bias_t.ndim == 1:
            bias = bias_t
        else:
            raise ValueError("intercept must be scalar or 1D array/tensor")

        if bias.shape[0] != out_features:
            # allow broadcast if single bias provided
            if bias.shape[0] == 1 and out_features > 1:
                bias = bias.expand(out_features)
            else:
                raise ValueError(f"intercept length {bias.shape[0]} incompatible with out_features {out_features}")

        # Create linear layer and load weights
        linear = nn.Linear(in_features, out_features).to(self.device)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        # Infer backbone output feature dimension via dummy input
        if dummy_input is None:
            # Try to create sensible dummy using feature_extractor preprocess / size=224
            # If feature_extractor has preprocess that can produce tensor, we can't easily call it here,
            # so fallback to (1,3,224,224) on the same device.
            dummy_input = torch.zeros(1, 3, 224, 224, device=self.device)

        dummy_input = dummy_input.to(self.device)
        # Pass through backbone to get feature shape
        with torch.no_grad():
            feats = self.backbone(dummy_input)

        if feats.dim() > 2:
            feat_dim = int(torch.prod(torch.tensor(feats.shape[1:], device=self.device)).item())
        else:
            feat_dim = feats.shape[1]

        if feat_dim != in_features:
            raise ValueError(
                f"Backbone produced flattened feature dim {feat_dim}, but coef expects in_features {in_features}. "
                "If your model expects a different input size, pass a matching dummy_input to readout_concat."
            )

        # Build final model
        model_vis = ModelWithReadout(self.backbone, linear).to(self.device).eval()
        return model_vis
