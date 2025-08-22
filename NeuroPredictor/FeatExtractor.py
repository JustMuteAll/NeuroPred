import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Union, List

class FeatureExtractorBase(nn.Module):
    """
    Base class for feature extractors with mixed-precision (AMP) support.
    Subclasses must set:
      - self.model (nn.Module) on the correct device
      - self.preprocess (callable) or None
    """
    def __init__(self, device: str = 'cuda', amp: bool = True):
        super().__init__()
        self.device = device
        self.amp = amp
        # model and preprocess set in subclass
        # hook storage
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._features = {}

    def list_hookable_layers(self) -> List[str]:
        """
        List leaf modules available for hooking.
        """
        return [name for name, m in self.model.named_modules() if len(list(m.children())) == 0]

    def get_preprocess(self):
        """
        Return preprocessing transform or None if not defined.
        """
        return getattr(self, 'preprocess', None)

    def _register_hooks(self, module_names: List[str]) -> None:
        """
        Attach forward hooks to capture outputs.
        """
        modules = dict(self.model.named_modules())
        for name in module_names:
            if name not in modules:
                raise ValueError(f"Module '{name}' not found in model")
            def hook_fn(module, _inp, outp, key=name):
                self._features[key] = outp
            self._hooks.append(modules[name].register_forward_hook(hook_fn))

    def _clear_hooks(self) -> None:
        """
        Remove all registered hooks.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _extract_features(
        self,
        x: torch.Tensor,
        module_names: Union[str, List[str]]
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass with hooks, returning flattened features.
        """
        names = [module_names] if isinstance(module_names, str) else module_names
        self._features = {}
        self._register_hooks(names)
        x = x.to(self.device)
        if self.amp:
            with torch.amp.autocast(device_type='cuda'):
                _ = self.model(x)
        else:
            _ = self.model(x)
        self._clear_hooks()
        # flatten outputs
        outputs = {}
        for name in names:
            feat = self._features[name]
            # outputs[name] = feat.view(feat.size(0), -1)
            outputs[name] = feat
        return outputs[names[0]] if len(names) == 1 else outputs
    
    def get_feature_shapes(self, layer_names: List[str], image_size: int = 224) -> dict:
        """
        Get feature output shapes of given layers using a dummy image.

        Args:
            layer_names (List[str]): names of layers to hook.
            image_size (int): input image height and width (default: 224).

        Returns:
            dict: mapping from layer name to feature shape (tuple).
        """
        # Create dummy image: batch_size=1, 3-channel RGB
        dummy = torch.randn(1, 3, image_size, image_size).to(self.device)
        print(type(dummy))

        # Register hooks and run forward
        self._features = {}
        self._register_hooks(layer_names)

        with torch.no_grad():
            if self.amp:
                with torch.amp.autocast(device_type='cuda'):
                    _ = self.model(dummy)
            else:
                _ = self.model(dummy)

        self._clear_hooks()

        # Return shape of each feature
        feature_shapes = {}
        for name in layer_names:
            feat = self._features[name]
            feature_shapes[name] = tuple(feat.shape)

        return feature_shapes


    def forward(self, x: torch.Tensor, module_names: Union[str, List[str]]):
        return self._extract_features(x, module_names)

# Subclasses
import timm
import torchvision.models as tv_models
import clip
import open_clip
from torchvision.transforms import Compose

class TimmFeatureExtractor(FeatureExtractorBase):
    """
    Feature extractor for timm models, supporting ViT patch embeddings and embedtypes.
    """
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_clip_224.laion2b',
        pretrained: bool = True,
        device: str = 'cuda',
        amp: bool = True
    ):
        super().__init__(device, amp)
        # Load and freeze
        self.model = timm.create_model(model_name, pretrained=pretrained).to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad = False

        # ViT-specific patch info
        if hasattr(self.model, 'patch_embed'):
            self.patch_size = self.model.patch_embed.patch_size[0]
            self.stride = self.model.patch_embed.proj.stride
        else:
            self.patch_size = None
            self.stride = None
        self._features = []

    def list_hookable_layers(self) -> List[str]:
        # For ViT, allow index-based layer names: 'blocks.0', etc.
        names = super().list_hookable_layers()
        if hasattr(self.model, 'blocks'):
            for idx in range(len(self.model.blocks)):
                names.append(f'blocks.{idx}')
        return names

    def _register_hook(self, layer: int, embedtype: str) -> None:
        """
        Attach hook on ViT block index with embedtype 'CLS', 'spatial', or 'mean'.
        """
        def hook_fn(module, _inp, outp):
            if embedtype == 'CLS':
                feat = outp[:, 0, :]
            elif embedtype == 'spatial':
                feat = outp[:, 1:, :].reshape(outp.size(0), -1)
            elif embedtype == 'mean':
                feat = outp[:, 1:, :].mean(dim=1)
            else:
                raise ValueError(f"Unsupported embedtype: {embedtype}")
            self._features.append(feat)

        block = self.model.blocks[layer]
        self._hooks.append(block.register_forward_hook(hook_fn))

    def extract_vit(
        self,
        x: torch.Tensor,
        layer: int,
        embedtype: str = 'CLS'
    ) -> torch.Tensor:
        """
        Extract features from a ViT block by index and embed type, returning (B, D).
        """
        self._features = []
        self._register_hook(layer, embedtype)
        if self.amp:
            with torch.amp.autocast(device_type='cuda'):
                _ = self.model(x.to(self.device))
        else:
            _ = self.model(x.to(self.device))
        self._clear_hooks()
        return self._features[0]

    def forward(
        self,
        x: torch.Tensor,
        layer_or_names: Union[int, str, List[str]],
        embedtype: str = 'CLS'
    ) -> Union[torch.Tensor, dict]:
        if isinstance(layer_or_names, int):
            return self.extract_vit(x, layer_or_names, embedtype)
        else:
            return self._extract_features(x, layer_or_names)

class TorchvisionFeatureExtractor(FeatureExtractorBase):
    def __init__(self, model_name: str = 'resnet50', ckpt_path=False, device='cuda', amp=True):
        super().__init__(device, amp)
        if hasattr(tv_models, model_name):
            if not ckpt_path:
                self.model = getattr(tv_models, model_name)(pretrained=True).to(self.device).eval()
            else:
                self.model = getattr(tv_models, model_name)(weights=None).to(self.device).eval()
                ckpt = torch.load(ckpt_path, map_location="cpu")

                if isinstance(ckpt, dict):
                    # 尝试常见键名
                    if "state_dict" in ckpt:
                        sd = ckpt["state_dict"]
                    elif "model_state_dict" in ckpt:
                        sd = ckpt["model_state_dict"]
                    elif "model" in ckpt and isinstance(ckpt["model"], dict):
                        sd = ckpt["model"]
                    else:
                        # 否则假设整个对象就是 state_dict
                        sd = ckpt
                else:
                    sd = ckpt

                new_sd = {}
                for k, v in sd.items():
                    new_k = k
                    if k.startswith("module."):
                        new_k = k[len("module."):]
                    if new_k.startswith("model."):
                        new_k = new_k[len("model."):]
                    new_sd[new_k] = v
                missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
                
        else:
            raise ValueError(f"Unknown torchvision model {model_name}")
        
        for p in self.model.parameters(): p.requires_grad = False
        try:
            weights = tv_models.get_model_weights(model_name).DEFAULT
            self.preprocess = weights.transforms()
        except Exception:
            self.preprocess = None

class CLIPFeatureExtractor(FeatureExtractorBase):
    def __init__(self, model_name='ViT-B/32', device='cuda', amp=True):
        super().__init__(device, amp)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.visual = self.model.visual.to(self.device).eval()
        for p in self.model.visual.parameters(): p.requires_grad = False
        self.model = self.model.visual

class OpenCLIPFeatureExtractor(FeatureExtractorBase):
    def __init__(
        self,
        model_name: str = 'ViT-B/32',
        pretrained: Union[str, bool] = 'laion2b_s34b_b79k',
        device: str = 'cuda',
        precision: str = 'amp',
        load_weights_only: bool = False
    ):
        super().__init__(device, amp=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
            load_weights_only=load_weights_only
        )
        self.model = model.visual.to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.preprocess = preprocess
