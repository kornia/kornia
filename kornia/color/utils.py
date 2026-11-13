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

from typing import Optional

import torch


def _apply_linear_transformation(
    image: torch.Tensor, kernel: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Apply a 3x3 linear color transformation with device-aware optimization.

    Args:
        image: Input image tensor with shape ``(*, 3, H, W)``.
        kernel: Transformation matrix with shape ``(3, 3)`` applied along the channel
            dimension.
        bias: Bias vector with shape ``(3,)`` to be added to the output.

    Returns:
        Tensor with the same shape as ``image`` containing the transformed values.
    """
    # Handle Integer inputs by casting to float safely
    image_compute = image if image.is_floating_point() else image.float()

    # Match kernel dtype to the image (propagates float64 if needed)
    kernel_compute = kernel.to(dtype=image_compute.dtype, device=image_compute.device)
    bias_compute = bias.to(dtype=image_compute.dtype, device=image_compute.device) if bias is not None else None
    input_shape = image_compute.shape

    # One matmul over the flattened pixels on every device. It replaces a 1x1 conv2d off the cpu, which
    # followed cuDNN's TF32 switch (on by default in PyTorch) with the precision chosen per shape: on CUDA
    # a single (1, 3, 224, 224) float32 image, and every shape under torch.compile, came back with a
    # 10-bit mantissa (3.6e-4 error in rgb_to_xyz, 5.9e-2 in rgb_to_lab). A matmul follows
    # torch.backends.cuda.matmul.allow_tf32 (off by default) like every other matrix product. It also
    # replaces the einsum on the cpu, where it gives bitwise the same result about 2x faster.
    # The leading size is spelled out rather than -1 so that empty shapes reshape too.
    batch = 1
    for size in input_shape[:-3]:
        batch *= size
    out = torch.matmul(kernel_compute, image_compute.reshape(batch, 3, input_shape[-2] * input_shape[-1]))

    if bias_compute is not None:
        out = out + bias_compute.view(-1, 1)

    return out.reshape(input_shape)
