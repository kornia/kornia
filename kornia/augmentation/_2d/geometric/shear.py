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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample, SamplePadding
from kornia.geometry.transform import get_shear_matrix2d, warp_affine


class RandomShear(GeometricAugmentationBase2D):
    r"""Apply a random 2D shear transformation to a torch.tensor image.

    The transformation is computed so that the image center is kept invariant.

    Args:
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "torch.zeros" (0), "border" (1) or "reflection" (2).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_affine`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomShear((-5., 2., 5., 10.), p=1.)
        >>> out = aug(input)
        >>> out, aug.transform_matrix
        (tensor([[[[0.4403, 0.7614, 0.1516],
                  [0.1753, 0.3074, 0.6127],
                  [0.4438, 0.8924, 0.4061]]]]), tensor([[[ 1.0000,  0.0100, -0.0100],
                 [-0.1183,  0.9988,  0.1194],
                 [ 0.0000,  0.0000,  1.0000]]]))
        >>> aug.inverse(out)
        tensor([[[[0.4045, 0.7577, 0.1393],
                  [0.2071, 0.3074, 0.5582],
                  [0.3958, 0.8868, 0.4265]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomShear((-15., 20.), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        shear: Union[torch.Tensor, float, Tuple[float, float], Tuple[float, float, float, float]],
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.shear = shear

        # Parse and store shear bounds as tensors for auto augment compatibility
        _shear = torch.as_tensor(shear)
        if _shear.dim() == 0:
            self.shear_x = torch.stack([-_shear, _shear])
            self.shear_y = torch.stack([-_shear, _shear])
        elif _shear.shape == torch.Size([2]):
            self.shear_x = _shear[:2].clone()
            self.shear_y = torch.tensor([0.0, 0.0])
        elif _shear.shape == torch.Size([4]):
            self.shear_x = _shear[:2].clone()
            self.shear_y = _shear[2:].clone()
        else:
            self.shear_x = _shear[:2].clone() if _shear.shape[0] >= 2 else _shear.clone()
            self.shear_y = _shear[2:].clone() if _shear.shape[0] >= 4 else torch.tensor([0.0, 0.0])

        self.flags = {
            "resample": Resample.get(resample),
            "padding_mode": SamplePadding.get(padding_mode),
            "align_corners": align_corners,
        }

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        _device, _dtype = self.device, self.dtype

        # Parse shear - (2,) means (min, max) for x, (4,) means (x_min, x_max, y_min, y_max)
        _shear = torch.as_tensor(self.shear, device=_device, dtype=_dtype)
        if _shear.dim() == 0:
            # Scalar: symmetric shear in both x and y
            _shear = torch.stack([-_shear, _shear, -_shear, _shear])
        elif _shear.shape == torch.Size([2]):
            # (min, max) for x only, y is 0
            _shear = torch.stack(
                [
                    _shear[0],
                    _shear[1],
                    torch.tensor(0.0, device=_device, dtype=_dtype),
                    torch.tensor(0.0, device=_device, dtype=_dtype),
                ]
            )
        elif _shear.shape != torch.Size([4]):
            raise ValueError(f"'shear' shall be either a scalar, (2,) or (4,). Got {self.shear}.")

        # Sample shear
        if self.same_on_batch:
            shear_x = (
                torch.empty(1, device=_device, dtype=_dtype)
                .uniform_(_shear[0].item(), _shear[1].item())
                .expand(batch_size)
            )
            shear_y = (
                torch.empty(1, device=_device, dtype=_dtype)
                .uniform_(_shear[2].item(), _shear[3].item())
                .expand(batch_size)
            )
        else:
            shear_x = torch.empty(batch_size, device=_device, dtype=_dtype).uniform_(_shear[0].item(), _shear[1].item())
            shear_y = torch.empty(batch_size, device=_device, dtype=_dtype).uniform_(_shear[2].item(), _shear[3].item())

        # Center
        center = torch.tensor([width, height], device=_device, dtype=_dtype).view(1, 2) / 2.0 - 0.5
        center = center.expand(batch_size, -1)

        return {
            "shear_x": shear_x,
            "shear_y": shear_y,
            "center": center,
        }

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        return get_shear_matrix2d(
            torch.as_tensor(params["center"], device=input.device, dtype=input.dtype),
            torch.deg2rad(torch.as_tensor(params["shear_x"], device=input.device, dtype=input.dtype)),
            torch.deg2rad(torch.as_tensor(params["shear_y"], device=input.device, dtype=input.dtype)),
        )

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

        return warp_affine(
            input,
            transform[:, :2, :],
            (height, width),
            flags["resample"].name.lower(),
            align_corners=flags["align_corners"],
            padding_mode=flags["padding_mode"].name.lower(),
        )

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=torch.as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
