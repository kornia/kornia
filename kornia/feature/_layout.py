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

import torch

# A defaulted argument rather than a closed-over global: TorchScript cannot read module-level Python values.
_CPU_HAS_ONEDNN: bool = torch.backends.mkldnn.is_available()


def _patch_conv_layout(x: torch.Tensor, cpu_has_onednn: bool = _CPU_HAS_ONEDNN) -> torch.Tensor:
    r"""Return a patch network's convolution input in the memory format its device convolves fastest in.

    The patch descriptor and shape networks (HardNet, HardNet8, SOSNet, TFeat, HyNet, AffNet, OriNet) call this where
    their convolutions start. It changes the strides only, never the values: the networks return the same numbers in
    either layout.

    - MPS, float32 and float16: channels-last. The convolution bodies run 1.04-1.38x faster on an Apple M1 with torch
      2.14, and extracting 2048 patches and describing them with HardNet 1.07x (float32) and 1.11x (float16).
    - CPU float32 with oneDNN: channels-last, which avoids oneDNN's activation reorders. A CPU build without oneDNN,
      such as the macOS arm64 wheels, convolves with NNPACK, which takes only the default layout, so the input is left
      alone instead of paying a copy that changes nothing.
    - Everything else, CUDA included, keeps the input's layout.

    Args:
        x: convolution input ``(B, C, H, W)``.
        cpu_has_onednn: whether CPU convolutions run on oneDNN. Leave the default, which asks the torch build.

    Returns:
        ``x``, as a channels-last copy where that layout is faster.
    """
    device = x.device.type
    if (device == "cpu" and cpu_has_onednn and x.dtype == torch.float32) or (
        device == "mps" and x.dtype in (torch.float32, torch.float16)
    ):
        return x.to(memory_format=torch.channels_last)
    return x
