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

from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from kornia.feature._layout import _mark_patch_convolutions, _patch_conv_layout


def _keep_float32_statistics(
    fn: Callable[[torch.Tensor], torch.Tensor], statistics: tuple[torch.Tensor | None, ...]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a module conversion ``fn`` so that it leaves ``statistics`` in float32 where it would make them float16.

    Pretrained running statistics can exceed float16's largest value, 65504, and converting them gives inf.
    """

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        converted = fn(tensor)
        if converted.dtype == torch.float16 and any(tensor is statistic for statistic in statistics):
            # Recover from the original buffer: promoting an already-overflowed value cannot restore it.
            return tensor.to(device=converted.device, dtype=torch.float32)
        return converted

    return convert


class _DescriptorSequential(nn.Sequential):
    """Keep a descriptor's final BatchNorm statistics and computation in float32 for half inputs.

    Each convolution also receives its input in the layout ``_patch_conv_layout`` picks for the device.
    """

    def __init__(self, *args: nn.Module | OrderedDict[str, nn.Module]) -> None:
        super().__init__(*args)
        # Sequential slices share modules. Only the original descriptor's final norm is protected;
        # a slice ending at an earlier BatchNorm must not change that shared layer's dtype.
        if (
            len(self)
            and isinstance(self[-1], nn.BatchNorm2d)
            and not (len(args) == 1 and isinstance(args[0], OrderedDict))
        ):
            self[-1]._descriptor_float32_statistics = True
        for module in self:
            if getattr(module, "_descriptor_float32_statistics", False):
                for name in ("running_mean", "running_var"):
                    tensor = getattr(module, name)
                    if tensor is not None and tensor.dtype == torch.float16:
                        setattr(module, name, tensor.float())

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "_DescriptorSequential":
        statistics = tuple(
            tensor
            for module in self
            if getattr(module, "_descriptor_float32_statistics", False)
            for tensor in (module.running_mean, module.running_var)
        )
        return super()._apply(_keep_float32_statistics(fn, statistics), recurse=recurse)

    def __prepare_scriptable__(self) -> "_DescriptorSequential":
        _mark_patch_convolutions(self)
        return self

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for module in self:
            # SOSNet's leading InstanceNorm2d returns the default layout, so the layout is set per convolution; a
            # convolution's output already has it, which makes the later calls free. A scripted loop also needs the
            # mark (see _mark_patch_convolutions). Ruff's SIM114 would merge the two tests with ``or``, which
            # TorchScript rejects.
            if isinstance(module, nn.Conv2d):  # noqa: SIM114
                input = _patch_conv_layout(input)
            elif getattr(module, "_patch_conv_input", False):
                input = _patch_conv_layout(input)
            if getattr(module, "_descriptor_float32_statistics", False) and input.dtype == torch.float16:
                # MPS also requires the input and running statistics to have the same dtype.
                input = module(input.float()).to(input.dtype)
            else:
                input = module(input)
        return input


class _Float32BatchNormSequential(nn.Sequential):
    """Keep the running statistics of a ``(convolution, BatchNorm2d, activation)`` block in float32 under float16.

    Pretrained running statistics can exceed float16's largest value, 65504; converting them gives inf, and the channel
    then normalizes to 0. A float16 conversion through this block therefore leaves the BatchNorm's running statistics in
    float32, converted from the original values, while its parameters convert as usual, so the model keeps one
    parameter dtype. While the statistics are float32 and the BatchNorm's weight float16, which is what that conversion
    leaves, the BatchNorm normalizes in float32 with float32 copies of its weight and bias and returns float16. Any
    other block, such as one never converted to float16, also under ``torch.autocast``, runs exactly as an
    ``nn.Sequential``. The layers stay plain modules.
    """

    def __init__(self, *args: nn.Module) -> None:
        super().__init__(*args)
        for module in self:
            if isinstance(module, nn.BatchNorm2d):
                module._float32_statistics = True

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "_Float32BatchNormSequential":
        statistics = tuple(
            tensor
            for module in self
            if getattr(module, "_float32_statistics", False)
            for tensor in (module.running_mean, module.running_var)
        )
        return super()._apply(_keep_float32_statistics(fn, statistics), recurse=recurse)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Decided from the BatchNorm's own tensors on every call: float32 statistics with a float16 weight is what a
        # float16 conversion leaves. Only a marked, plain BatchNorm2d without hooks of its own takes the float32 path;
        # a fused, converted or wrapped layer runs as in an nn.Sequential. hasattr is resolved per layer by TorchScript.
        for module in self:
            if hasattr(module, "_float32_statistics"):
                upcast = False
                running_var = module.running_var
                if running_var is not None and running_var.dtype == torch.float32:
                    if torch.jit.is_scripting():
                        weight = module.weight
                        upcast = weight is not None and weight.dtype == torch.float16
                    else:
                        # torch.fx traces a parameter read as an attribute, and a traced dtype cannot decide a branch.
                        # FSDP and DataParallel replicas hold the weight as a plain tensor instead of a parameter.
                        weight = next(iter(module.parameters()), None)
                        if weight is None:
                            weight = getattr(module, "weight", None)
                        hooks = module._forward_hooks or module._forward_pre_hooks
                        upcast = (
                            type(module) is nn.BatchNorm2d
                            and not hooks
                            and isinstance(weight, torch.Tensor)
                            and weight.dtype == torch.float16
                        )
                if upcast:
                    # nn.BatchNorm2d.forward in float32, with float32 copies of the float16 weight and bias. MPS also
                    # requires the input and the running statistics to have the same dtype.
                    factor = 0.0 if module.momentum is None else module.momentum
                    num_batches_tracked = module.num_batches_tracked
                    if module.training and module.track_running_stats and num_batches_tracked is not None:
                        num_batches_tracked.add_(1)
                        momentum = module.momentum
                        factor = 1.0 / float(num_batches_tracked) if momentum is None else momentum
                    weight, bias = module.weight, module.bias
                    running_mean = module.running_mean
                    use_statistics = not module.training or module.track_running_stats
                    input = F.batch_norm(
                        input.float(),
                        running_mean if use_statistics else None,
                        running_var if use_statistics else None,
                        None if weight is None else weight.float(),
                        None if bias is None else bias.float(),
                        module.training or (running_mean is None and running_var is None),
                        factor,
                        module.eps,
                    ).to(input.dtype)
                else:
                    input = module(input)
            else:
                input = module(input)
        return input
