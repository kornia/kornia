# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class FeatureExtractor:
    def __init__(self, model: nn.Module, layers: Optional[List[str]] = None, processing: str = "none"):
        """Initialize feature extractor for selected layers."""
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
                name
                for name in layer_dict.keys()
                if name != ""  # skip root module
            ]

        for name in self.layers:
            if name not in layer_dict:
                raise ValueError(f"Layer {name} not found in model")

            handle = layer_dict[name].register_forward_hook(self._hook_fn(name))
            self.handles.append(handle)

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, **inputs):
        """Run model forward pass and collect features."""
        self.clear()

        with torch.no_grad():
            _ = self.model(**inputs)

        return self.features
