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
from torch.nn import functional as F


def _apply_linear_transformation(
    image: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply a 3x3 linear color transformation with device-aware optimization.

    Args:
        image: Input image tensor with shape :math:`(*, 3, H, W)`.
        kernel: Transformation matrix with shape :math:`(3, 3)` applied along the channel
            dimension.
        bias: Bias vector with shape :math:`(3,)` to be added to the output.

    Returns:
        Tensor with the same shape as ``image`` containing the transformed values.
    """
    # Handle Integer inputs by casting to float safely
    if image.is_floating_point():
        image_compute = image
    else:
        image_compute = image.float()

    # Match kernel dtype to the image (propagates float64 if needed)
    kernel_compute = kernel.to(dtype=image_compute.dtype, device=image_compute.device)
    input_shape = image_compute.shape

    # Empirical benchmarks show that einsum is faster on CPU for this specific pattern,
    # while conv2d offers significant speedups on GPU/CUDA.
    # We branch to ensure optimal performance on both devices.
    # BRANCH 1: CPU (Einsum)
    if image.device.type == "cpu":
        out = torch.einsum("oi, ...ihw -> ...ohw", kernel, image)

        if bias is not None:
            out = out + bias.view(-1, 1, 1)

        return out.contiguous()

    # BRANCH 2: GPU/Accelerators (Conv2d)
    else:
        # Reshape for conv2d: (B*..., C, H, W)
        input_flat = image_compute.reshape(-1, 3, input_shape[-2], input_shape[-1])

        weight = kernel_compute.view(3, 3, 1, 1)
        out_flat = F.conv2d(input_flat, weight, bias=bias)

        # Unflatten back to original shape
        out = out_flat.reshape(input_shape)

    return out
