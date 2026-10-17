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

from kornia.augmentation.random_generator._2d.perspective import _as_scalar_tensor, _source_end_and_extent
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.augmentation.utils.helpers import _constant_tensor
from kornia.core.utils import _extract_device_dtype


class PerspectiveGenerator3D(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    See the Convention block on :class:`~kornia.augmentation.RandomPerspective3D`.

    Args:
        distortion_scale: controls the degree of distortion and ranges from 0 to 1.

    Returns:
        A dict of parameters to be passed for transformation.
            - start_points (torch.Tensor): perspective source bounding boxes with a shape of (B, 8, 3).
            - end_points (torch.Tensor): perspective target bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, distortion_scale: Union[torch.Tensor, float] = 0.5) -> None:
        super().__init__()
        self.distortion_scale = distortion_scale

    def __repr__(self) -> str:
        return f"distortion_scale={self.distortion_scale}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = torch.as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}")
        self.rand_sampler = Uniform(
            torch.tensor(0, device=device, dtype=dtype),
            torch.tensor(1, device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        depth = batch_shape[-3]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.distortion_scale])

        # Coincident source corners make the perspective solve singular. Give singleton axes a
        # unit extent and no offset, as in the 2D generator. torch.jit.trace passes the sizes as
        # 0-d tensors, and the helper keeps the rule as tensor arithmetic there (#5110).
        x_end, x_extent = _source_end_and_extent(width)
        y_end, y_extent = _source_end_and_extent(height)
        z_end, z_extent = _source_end_and_extent(depth)

        if isinstance(x_end, torch.Tensor) or isinstance(y_end, torch.Tensor) or isinstance(z_end, torch.Tensor):
            # _constant_tensor is specified for Python scalars only, so build the corners from the traced sizes.
            x = _as_scalar_tensor(x_end, _device, _dtype)
            y = _as_scalar_tensor(y_end, _device, _dtype)
            z = _as_scalar_tensor(z_end, _device, _dtype)
            zero = torch.zeros((), device=_device, dtype=_dtype)
            start_points = torch.stack(
                [
                    torch.stack([zero, x, x, zero, zero, x, x, zero]),
                    torch.stack([zero, zero, y, y, zero, zero, y, y]),
                    torch.stack([zero, zero, zero, zero, z, z, z, z]),
                ],
                dim=-1,
            ).unsqueeze(0)
        else:
            start_points = _constant_tensor(
                [
                    [
                        [0, 0, 0],
                        [x_end, 0, 0],
                        [x_end, y_end, 0],
                        [0, y_end, 0],
                        [0, 0, z_end],
                        [x_end, 0, z_end],
                        [x_end, y_end, z_end],
                        [0, y_end, z_end],
                    ]
                ],
                device=_device,
                dtype=_dtype,
            )
        start_points = start_points.expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = self._distortion_scale * x_extent / 2
        fy = self._distortion_scale * y_extent / 2
        fz = self._distortion_scale * z_extent / 2

        factor = torch.stack([fx, fy, fz], 0).view(-1, 1, 3).to(device=_device, dtype=_dtype)

        rand_val: torch.Tensor = _adapted_rsampling(start_points.shape, self.rand_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )

        pts_norm = _constant_tensor(
            [[[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]],
            device=_device,
            dtype=_dtype,
        )
        end_points = start_points + factor * rand_val * pts_norm

        return {"start_points": start_points, "end_points": end_points}
