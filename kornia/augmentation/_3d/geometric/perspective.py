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

from typing import Any, Dict, Optional, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.augmentation.utils.helpers import _constant_tensor
from kornia.constants import Resample
from kornia.geometry import get_perspective_transform3d, warp_perspective3d


class RandomPerspective3D(GeometricAugmentationBase3D):
    r"""Apply random perspective transformation to 3D volumes (5D torch.Tensor).

    Args:
        p: probability of the image being perspectively transformed.
        distortion_scale: it controls the degree of distortion and ranges from 0 to 1.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`
        - Output: :math:`(B, C, D, H, W)`

    Note:
        Input torch.Tensor must be float and normalized into [0, 1] for the best differentiability support.

    Convention:
        See :class:`~kornia.augmentation.GeometricAugmentationBase3D` for the shared 3D geometry contract.

        - ``distortion_scale=0`` generates identical source and destination corners. The default bilinear,
          ``align_corners=False`` perspective warp nevertheless does not reproduce its input, even a constant one;
          use ``align_corners=True`` for an identity warp up to floating-point roundoff in ``float32`` and
          ``float64``. The false-setting normalization defect is tracked in
          `#4503 <https://github.com/kornia/kornia/issues/4503>`_.
        - the default interpolation is bilinear and the default ``align_corners`` is ``False``.
        - a spatial dimension of 1 gets a unit source extent and no corner offset, so its single
          slice, row or column maps onto itself. At ``distortion_scale=0`` the transform matrix is
          the identity; output identity still requires ``align_corners=True`` as described above.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> inputs= torch.tensor([[[
        ...    [[1., 0., 0.],
        ...     [0., 1., 0.],
        ...     [0., 0., 1.]],
        ...    [[1., 0., 0.],
        ...     [0., 1., 0.],
        ...     [0., 0., 1.]],
        ...    [[1., 0., 0.],
        ...     [0., 1., 0.],
        ...     [0., 0., 1.]]
        ... ]]])
        >>> aug = RandomPerspective3D(0.5, p=1., align_corners=True)
        >>> aug(inputs)
        tensor([[[[[0.3976, 0.2651, 0.0000],
                   [0.5507, 0.4657, 0.1153],
                   [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                  [[0.0901, 0.1390, 0.0000],
                   [0.3668, 0.5174, 0.0000],
                   [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                  [[0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomPerspective3D(0.5, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        distortion_scale: Union[torch.Tensor, float] = 0.5,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.flags = {"resample": Resample.get(resample), "align_corners": align_corners}
        self._param_generator = rg.PerspectiveGenerator3D(distortion_scale)

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        start_points, end_points = params["start_points"], params["end_points"]
        width, depth = input.shape[-1], input.shape[-3]
        # The solver uses corners 0, 1, 2, 5, 7: only two lie on x=0. On a singleton x axis, swap the x/z
        # or x/y corner ordering so three constrain each singleton plane to map onto itself.
        if isinstance(width, torch.Tensor):
            # torch.jit.trace passes the sizes as 0-d tensors, so the order is selected in the graph (#5110).
            device = start_points.device
            singleton_order = torch.where(
                torch.as_tensor(depth, device=device) > 1,
                _constant_tensor([0, 4, 7, 3, 1, 5, 6, 2], device=device, dtype=torch.long),
                _constant_tensor([0, 3, 2, 1, 4, 7, 6, 5], device=device, dtype=torch.long),
            )
            order = torch.where(width.to(device) == 1, singleton_order, torch.arange(8, device=device))
            start_points, end_points = start_points[:, order], end_points[:, order]
        elif isinstance(width, int) and width == 1:
            order = [0, 4, 7, 3, 1, 5, 6, 2] if depth > 1 else [0, 3, 2, 1, 4, 7, 6, 5]
            start_points, end_points = start_points[:, order], end_points[:, order]
        return get_perspective_transform3d(start_points, end_points).to(input)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the transform to be a torch.Tensor. Got {type(transform)}")

        return warp_perspective3d(
            input,
            transform,
            (input.shape[-3], input.shape[-2], input.shape[-1]),
            flags=flags["resample"].name.lower(),
            align_corners=flags["align_corners"],
        )
