import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(self, model: nn.Module, layers: list = None, processing: str = "none"):
        """
        model: PyTorch model
        layers: list of layer names (strings) to hook
                if None → hooks all layers
        processing:
            "none"    → raw features
            "flatten" → flatten to (B, -1)
            "gap"     → global average pooling (if applicable)
        """
        self.model = model
        self.layers = layers
        self.processing = processing

        self.features = {}
        self.handles = []

        self._register_hooks()

    def _get_layer_dict(self):
        return dict(self.model.named_modules())

    def _process(self, x):
        if self.processing == "none":
            return x

        elif self.processing == "flatten":
            return x.reshape(x.size(0), -1)

        elif self.processing == "gap":
            # apply GAP only if feature map is 4D
            if x.dim() == 4:
                return F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.size(0), -1)
            else:
                # fallback to flatten for non-CNN layers
                return x.reshape(x.size(0), -1)

        else:
            raise ValueError(f"Unknown processing: {self.processing}")

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features[name] = self._process(output.detach())
        return hook

    def _register_hooks(self):
        layer_dict = self._get_layer_dict()

        # Auto-select all layers if None
        if self.layers is None:
            self.layers = [
                name for name in layer_dict.keys()
                if name != ""  # skip root module
            ]

        for name in self.layers:
            if name not in layer_dict:
                raise ValueError(f"Layer {name} not found in model")

            handle = layer_dict[name].register_forward_hook(
                self._hook_fn(name)
            )
            self.handles.append(handle)

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, **inputs):
        """
        Supports:
            model(x=...)
            model(input=...)
            model(**kwargs)
        """
        self.clear()

        with torch.no_grad():
            _ = self.model(**inputs)

        return self.features