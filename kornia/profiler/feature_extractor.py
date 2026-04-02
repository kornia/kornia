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
        r"""Initialize a feature extractor for selected model layers.

        This utility registers forward hooks on specified layers of a model
        and captures their outputs during the forward pass. Optionally, the
        extracted features can be post-processed.

        Args:
            model: PyTorch model from which features will be extracted.
            layers: List of layer names to hook. If ``None``, all layers
                (except the root module) are selected.
            processing: Feature processing method. Supported options:
                - ``"none"``: No processing.
                - ``"flatten"``: Flatten features to shape :math:`(B, D)`.
                - ``"gap"``: Apply global average pooling for 4D tensors,
                  otherwise fallback to flatten.

        Raises:
            ValueError: If an unknown processing mode is provided.
        """
        self.model = model
        self.layers = layers
        self.processing = processing

        self.features = {}
        self.handles = []

        self._register_hooks()

    def _get_layer_dict(self):
        r"""Retrieve a dictionary mapping layer names to modules.

        Returns:
            Dictionary where keys are layer names and values are modules.
        """
        return dict(self.model.named_modules())

    def _process(self, x):
        r"""Process extracted features according to the selected mode.

        Args:
            x: Feature tensor output from a hooked layer.

        Returns:
            Processed feature tensor.

        Raises:
            ValueError: If an unknown processing mode is specified.
        """
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
        r"""Create a forward hook function for a given layer.

        Args:
            name: Name of the layer being hooked.

        Returns:
            A forward hook function that stores processed outputs.
        """

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features[name] = self._process(output.detach())

        return hook

    def _register_hooks(self):
        r"""Register forward hooks on the selected layers.

        Automatically selects all layers if none are provided.

        Raises:
            ValueError: If a specified layer name is not found in the model.
        """
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
        r"""Clear stored features from previous forward passes."""
        self.features = {}

    def remove_hooks(self):
        r"""Remove all registered forward hooks from the model."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, **inputs):
        r"""Run a forward pass and collect features from hooked layers.

        Args:
            **inputs: Keyword arguments passed to the model's forward method.

        Returns:
            Dictionary mapping layer names to extracted feature tensors.
        """
        self.clear()

        with torch.no_grad():
            _ = self.model(**inputs)

        return self.features
