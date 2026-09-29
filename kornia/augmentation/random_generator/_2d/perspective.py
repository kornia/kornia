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

from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _check_positive_int_or_traced, _common_param_check
from kornia.augmentation.utils.helpers import _constant_tensor
from kornia.core.utils import _extract_device_dtype

__all__ = ["PerspectiveGenerator"]


def _source_end_and_extent(
    size: Union[int, torch.Tensor],
) -> Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]]:
    """Return the last source coordinate and the corner-offset extent along one spatial axis.

    A size-1 axis would make two source corners coincide and the homography singular. It gets a
    unit extent and no offset instead, so its single row or column maps onto itself. The legacy
    ONNX tracer passes sizes as 0-d tensors; the rule then has to be tensor arithmetic, because
    a Python comparison is evaluated once at trace time and never reaches the exported graph.
    """
    if isinstance(size, torch.Tensor):
        return (size - 1).clamp(min=1), size * (size > 1)
    return (1, 0) if size == 1 else (size - 1, size)


def _as_scalar_tensor(value: Union[int, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.full((), value, device=device, dtype=dtype)


class PerspectiveGenerator(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        distortion_scale: the degree of distortion, ranged from 0 to 1.
        sampling_method: ``'basic'`` | ``'area_preserving'``. Default: ``'basic'``
            If ``'basic'``, samples by translating the image corners randomly inwards.
            If ``'area_preserving'``, samples by randomly translating the image corners in any direction.
            Preserves area on average. See https://arxiv.org/abs/2104.03308 for further details.

    Returns:
        A dict of parameters to be passed for transformation.
            - start_points (torch.Tensor): element-wise perspective source areas with a shape of (B, 4, 2).
            - end_points (torch.Tensor): element-wise perspective target areas with a shape of (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, distortion_scale: Union[torch.Tensor, float] = 0.5, sampling_method: str = "basic") -> None:
        super().__init__()
        if sampling_method not in ("basic", "area_preserving"):
            raise NotImplementedError(f"Sampling method {sampling_method} not yet implemented.")
        self.distortion_scale = distortion_scale
        self.sampling_method = sampling_method

    def __repr__(self) -> str:
        return f"distortion_scale={self.distortion_scale}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = torch.as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}.")
        self.rand_val_sampler = Uniform(
            torch.tensor(0, device=device, dtype=dtype),
            torch.tensor(1, device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.distortion_scale])
        _common_param_check(batch_size, same_on_batch)
        _check_positive_int_or_traced(height, "height")
        _check_positive_int_or_traced(width, "width")

        x_end, x_extent = _source_end_and_extent(width)
        y_end, y_extent = _source_end_and_extent(height)

        # Subtract before casting: bbox_generator subtracts in the tensor dtype,
        # which changes large half-precision image coordinates.
        if isinstance(x_end, torch.Tensor) or isinstance(y_end, torch.Tensor):
            # _constant_tensor is specified for Python scalars only, so build the corners from the traced sizes.
            x = _as_scalar_tensor(x_end, _device, _dtype)
            y = _as_scalar_tensor(y_end, _device, _dtype)
            zero = torch.zeros((), device=_device, dtype=_dtype)
            start_points = torch.stack([torch.stack([zero, x, x, zero]), torch.stack([zero, zero, y, y])], dim=-1)
            start_points = start_points.unsqueeze(0)
        else:
            start_points = _constant_tensor(
                [[[0, 0], [x_end, 0], [x_end, y_end], [0, y_end]]],
                device=_device,
                dtype=_dtype,
            )
        start_points = start_points.expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = self._distortion_scale * x_extent / 2
        fy = self._distortion_scale * y_extent / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2).to(device=_device, dtype=_dtype)

        # TODO: This line somehow breaks the gradcheck
        rand_val: torch.Tensor = _adapted_rsampling(start_points.shape, self.rand_val_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        if self.sampling_method == "basic":
            pts_norm = _constant_tensor([[[1, 1], [-1, 1], [-1, -1], [1, -1]]], device=_device, dtype=_dtype)
            offset = factor * rand_val * pts_norm
        elif self.sampling_method == "area_preserving":
            offset = 2 * factor * (rand_val - 0.5)

        end_points = start_points + offset

        return {"start_points": start_points, "end_points": end_points}
