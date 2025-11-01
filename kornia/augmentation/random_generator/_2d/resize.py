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

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _common_param_check
from kornia.core import Device, Tensor
from kornia.geometry.bbox import bbox_generator
from kornia.geometry.transform.affwarp import _side_to_image_size

__all__ = ["ResizeGenerator"]


class ResizeGenerator(RandomGeneratorBase):
    r"""Get parameters for ```resize``` transformation for resize transform.

    Args:
        resize_to: Desired output size of the crop, like (h, w).
        side: Which side to resize if `resize_to` is only of type int.

    Returns:
        parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (Tensor): output bounding boxes with a shape (B, 4, 2).
            - input_size (Tensor): (h, w) from batch input.
            - resize_to (tuple): new (h, w) for batch input.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, resize_to: Union[int, Tuple[int, int]], side: str = "short") -> None:
        super().__init__()
        self.output_size = resize_to
        self.side = side

    def __repr__(self) -> str:
        repr = f"output_size={self.output_size}"
        return repr

    def make_samplers(self, device: Device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        pass

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device = self.device
        _dtype = self.dtype

        if batch_size == 0:
            return {
                "src": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                "dst": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
            }

        h, w = batch_shape[-2], batch_shape[-1]

        # Avoid tensor() for scalars, use torch.full for batch if batch_size > 1.
        xs = torch.zeros(batch_size, device=_device, dtype=_dtype)
        ys = torch.zeros(batch_size, device=_device, dtype=_dtype)
        ws = torch.full((batch_size,), w, device=_device, dtype=_dtype)
        hs = torch.full((batch_size,), h, device=_device, dtype=_dtype)

        src = bbox_generator(xs, ys, ws, hs)

        if isinstance(self.output_size, int):
            aspect_ratio = w / h
            output_size = _side_to_image_size(self.output_size, aspect_ratio, self.side)
        else:
            output_size = self.output_size

        if not (
            len(output_size) == 2
            and isinstance(output_size[0], (int,))
            and isinstance(output_size[1], (int,))
            and output_size[0] > 0
            and output_size[1] > 0
        ):
            raise AssertionError(f"`resize_to` must be a tuple of 2 positive integers. Got {output_size}.")
        ow, oh = output_size[1], output_size[0]
        wdst = torch.full((batch_size,), ow, device=_device, dtype=_dtype)
        hdst = torch.full((batch_size,), oh, device=_device, dtype=_dtype)
        dst = bbox_generator(xs, ys, wdst, hdst)

        # Vectorized and efficient
        _input_size = torch.tensor([h, w], device=_device, dtype=torch.long)
        _output_size = torch.tensor([oh, ow], device=_device, dtype=torch.long)
        _input_size = _input_size.unsqueeze(0).expand(batch_size, -1)
        _output_size = _output_size.unsqueeze(0).expand(batch_size, -1)

        return {"src": src, "dst": dst, "input_size": _input_size, "output_size": _output_size}
