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

from collections.abc import Iterable

import torch
from torch import nn

# A defaulted argument rather than a closed-over global: TorchScript cannot read module-level Python values.
_CPU_HAS_ONEDNN: bool = torch.backends.mkldnn.is_available()


def _mark_patch_convolutions(layers: Iterable[nn.Module]) -> None:
    """Mark the ``nn.Conv2d`` layers among ``layers`` for a scripted layer loop that calls :func:`_patch_conv_layout`.

    ``_mark_patch_convolutions(layers)`` sets ``_patch_conv_input = True`` on each ``nn.Conv2d`` in ``layers`` (not on
    nested ones). A ``forward`` that loops over its layers sets the layout before each layer that passes
    ``isinstance(layer, nn.Conv2d)``, else before one with ``getattr(layer, "_patch_conv_input", False)``. Eager needs
    only the ``isinstance``. TorchScript resolves it against the one ``Conv2d`` class type it compiled first in the
    process, and gives every ``Conv2d`` with other constants a mangled type that fails it, so a scripted loop would
    skip the layout before some or all of its convolutions, depending on what had been scripted before it, and convolve
    in a different layout from eager. TorchScript reads the attribute of each submodule instead. The networks call this
    from ``__prepare_scriptable__``, which ``torch.jit.script`` runs on every submodule before compiling, so the mark
    also reaches a convolution swapped in after construction and a model unpickled from an earlier kornia.
    """
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            layer._patch_conv_input = True


def _patch_conv_layout(x: torch.Tensor, cpu_has_onednn: bool = _CPU_HAS_ONEDNN) -> torch.Tensor:
    r"""Return a patch network's convolution input in the memory format its device convolves fastest in.

    The patch descriptor and shape networks (HardNet, HardNet8, SOSNet, TFeat, HyNet, AffNet, OriNet) call this where
    their convolutions start. It changes the strides only, never the values; the networks return the same numbers in
    either layout up to rounding, because a backend may pick other convolution kernels for channels-last (bitwise equal
    on MPS; within float32 rounding on x86 oneDNN).

    - MPS, float32 and float16: channels-last. The convolution bodies run 1.04-1.38x faster on an Apple M1 with torch
      2.14, and extracting 2048 patches and describing them with HardNet 1.07x (float32) and 1.11x (float16).
    - CUDA, float16 and bfloat16: channels-last. On an RTX 4090 with torch 2.14 the forwards on 2048 patches run
      1.08-1.44x faster; on 512 patches the result is mixed (0.83-1.30x across runs, the losses in float16).
    - CPU float32 with oneDNN: channels-last, which avoids oneDNN's activation reorders: 1.2-1.5x on an x86 i7-14700K,
      except HardNet8, which gains at the default thread count and is 0.95x on 8 threads. A CPU build without oneDNN,
      such as the macOS arm64 wheels, convolves with NNPACK, which takes only the default layout, so the input is left
      alone instead of paying a copy that changes nothing.
    - Everything else keeps the input's layout. CUDA float32 is among them: under cuDNN's default TF32 convolutions,
      six of the seven bodies are 0.87-1.00x in channels-last at 2048 patches and more.

    Args:
        x: convolution input ``(B, C, H, W)``.
        cpu_has_onednn: whether CPU convolutions run on oneDNN. Leave the default, which asks the torch build.

    Returns:
        ``x``, as a channels-last copy where that layout is faster.
    """
    device = x.device.type
    if (
        (device == "cpu" and cpu_has_onednn and x.dtype == torch.float32)
        or (device == "mps" and x.dtype in (torch.float32, torch.float16))
        or (device == "cuda" and x.dtype in (torch.float16, torch.bfloat16))
    ):
        return x.to(memory_format=torch.channels_last)
    return x
