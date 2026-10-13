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

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from kornia.profiler.collectors.base import Collector


class LayerCollector(Collector):
    """Collect intermediate tensor representations from selected model layers.

    Forward hooks capture tensor outputs from the selected modules. Captured
    tensors are detached from the computation graph and optionally processed
    before being stored.

    Args:
        layers: Names of the model modules from which to collect outputs.
            If ``None``, all named modules except the root module are selected.
        processing: Processing applied to captured tensors. Supported values
            are ``"none"``, ``"flatten"``, and ``"gap"``. Global average
            pooling is applied to four-dimensional tensors in ``"gap"`` mode;
            other tensors are flattened.
    """

    def __init__(
        self,
        layers: Optional[List[str]] = None,
        processing: str = "none",
    ) -> None:
        self.layers: Optional[List[str]] = layers
        self.processing: str = processing
        self.model: Optional[nn.Module] = None
        self.features: Dict[str, Tensor] = {}
        self.handles: List[RemovableHandle] = []

        if processing not in {"none", "flatten", "gap"}:
            raise ValueError(f"Unknown processing: {processing}")

    def attach(self, model: nn.Module) -> None:
        """Attach forward hooks to the selected modules of a model.

        Existing hooks are removed before the new model is attached. If no
        layer names were specified, hooks are registered on all named modules
        except the root module.

        Args:
            model: PyTorch model whose intermediate representations are
                collected.

        Raises:
            ValueError: If any explicitly selected layer name is absent
                from the model.
        """
        self.remove_hooks()
        self.model = model
        layer_dict = dict(model.named_modules())

        layers = self.layers
        if layers is None:
            layers = [name for name in layer_dict if name != ""]
        else:
            missing_layers = [name for name in layers if name not in layer_dict]
            if missing_layers:
                raise ValueError(f"Layers not found in model: {missing_layers}")

        self.layers = layers

        try:
            for name in layers:
                handle = layer_dict[name].register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)
        except Exception:
            self.remove_hooks()
            raise

    def _process(self, x: Tensor) -> Tensor:
        """Apply the configured processing to a captured tensor.

        Args:
            x: Tensor produced by a hooked module.

        Returns:
            The original tensor for ``"none"``, a flattened tensor for
            ``"flatten"``, or a pooled or flattened tensor for ``"gap"``.

        Raises:
            ValueError: If the processing mode is unsupported.
        """
        if self.processing == "none":
            return x
        if self.processing == "flatten":
            return x.reshape(x.size(0), -1)
        if self.processing == "gap":
            if x.dim() == 4:
                return F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.size(0), -1)
            return x.reshape(x.size(0), -1)

        raise ValueError(f"Unknown processing: {self.processing}")

    def _hook_fn(
        self, name: str
    ) -> Callable[[nn.Module, Tuple[Any, ...], Any], None]:
        """Create a forward hook that stores a module's tensor output.

        Args:
            name: Name used as the key for the captured representation.

        Returns:
            A forward-hook callback that processes and stores tensor outputs.
            Non-tensor outputs are ignored.
        """

        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if isinstance(output, Tensor):
                self.features[name] = self._process(output.detach())

        return hook

    def clear(self) -> None:
        """Clear representations collected during a previous forward pass."""
        self.features = {}

    def remove_hooks(self) -> None:
        """Remove registered forward hooks and detach from the current model.

        All stored representations are cleared. The collector can subsequently
        be attached to another model.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.clear()
        self.model = None

    def __call__(self, **inputs: Tensor) -> Dict[str, Tensor]:
        """Run the attached model and return its collected representations.

        The stored representations are cleared before each forward pass.
        Model execution occurs under ``torch.no_grad()``.

        Args:
            **inputs: Keyword arguments forwarded to the attached model.

        Returns:
            A dictionary mapping selected module names to their processed
            tensor outputs. Modules producing non-tensor outputs are omitted.

        Raises:
            RuntimeError: If no model is currently attached.
        """
        if self.model is None:
            raise RuntimeError(
                "Collector is not attached to a model. Call attach(model) first."
            )

        self.clear()
        with torch.no_grad():
            self.model(**inputs)
        return self.features
