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

    # A matmul on every device, not a 1x1 conv2d off the cpu: a conv follows cuDNN's TF32 switch (on by
    # default in PyTorch), and with older cuDNN (seen with torch 2.5.1's cuDNN 9.1) the conv picked TF32
    # per shape, e.g. for a single (1, 3, 224, 224) float32 image and under torch.compile. A matmul follows
    # torch.backends.cuda.matmul.allow_tf32 (off by default) like every other matrix product. On the cpu
    # it gives bitwise the einsum's result, faster.
    if image_compute.device.type == "cuda" and image_compute.stride(-3) == 1 and input_shape[-2] * input_shape[-1] > 1:
        # Channel-innermost input (channels_last, or HWC permuted to CHW): cuBLAS takes a slow strided path
        # for the flattened view below, 2-3x slower than the conv2d this replaced (#6768). Contract over the
        # last axis instead, which keeps the caller's channels_last layout. Not on the cpu, where the view
        # below is already fast and bitwise equal to the einsum.
        out_cl = torch.matmul(image_compute.movedim(-3, -1), kernel_compute.t())
        if bias_compute is not None:
            out_cl = out_cl + bias_compute
        return out_cl.movedim(-1, -3)

    # The leading size is spelled out rather than -1 so that empty shapes reshape too.
    batch = 1
    for size in input_shape[:-3]:
        batch *= size
    out = torch.matmul(kernel_compute, image_compute.reshape(batch, 3, input_shape[-2] * input_shape[-1]))

    if bias_compute is not None:
        out = out + bias_compute.view(-1, 1)

    return out.reshape(input_shape)
