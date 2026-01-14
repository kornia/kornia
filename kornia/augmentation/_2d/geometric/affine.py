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
from kornia.augmentation.utils import _range_bound, _singular_range_check
from kornia.constants import Resample, SamplePadding
from kornia.geometry.transform import get_affine_matrix2d, warp_affine


class RandomAffine(GeometricAugmentationBase2D):
    r"""Apply a random 2D affine transformation to a torch.tensor image.

    .. image:: _static/img/RandomAffine.png

    The transformation is computed so that the image center is kept invariant.

    Args:
        degrees: Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale: scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If (a, b, c, d), the scale is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d.
            Will keep original scale by default.
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "torch.zeros" (0), "border" (1), "reflection" (2) or "fill" (3).
        fill_value: the value to be filled in the padding area when padding_mode="fill".
            Can be a float, int, or a torch.tensor of shape (C) or (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_affine`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> out = aug(input)
        >>> out, aug.transform_matrix
        (tensor([[[[0.3961, 0.7310, 0.1574],
                  [0.1781, 0.3074, 0.5648],
                  [0.4804, 0.8379, 0.4234]]]]), tensor([[[ 0.9923, -0.1241,  0.1319],
                 [ 0.1241,  0.9923, -0.1164],
                 [ 0.0000,  0.0000,  1.0000]]]))
        >>> aug.inverse(out)
        tensor([[[[0.3890, 0.6573, 0.1865],
                  [0.2063, 0.3074, 0.5459],
                  [0.3892, 0.7896, 0.4224]]]])
        >>> input
        tensor([[[[0.4963, 0.7682, 0.0885],
                  [0.1320, 0.3074, 0.6341],
                  [0.4901, 0.8964, 0.4556]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        degrees: Union[torch.Tensor, float, Tuple[float, float]],
        translate: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        scale: Optional[Union[torch.Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[torch.Tensor, float, Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        fill_value: Optional[Union[float, int, torch.Tensor]] = None,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        # Store parameters for generate_parameters
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        if fill_value is not None and not isinstance(fill_value, torch.Tensor):
            fill_value = torch.as_tensor(fill_value)

        self.flags = {
            "resample": Resample.get(resample),
            "padding_mode": SamplePadding.get(padding_mode),
            "align_corners": align_corners,
            "fill_value": fill_value,
        }

    def _parse_shear(self, _device: torch.device, _dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Parse shear parameter into a 2x2 tensor."""
        if self.shear is None:
            return None
        _shear = torch.as_tensor(self.shear, device=_device, dtype=_dtype)
        zero = torch.tensor(0.0, device=_device, dtype=_dtype)
        if _shear.dim() == 0:
            _shear = torch.stack([-_shear, _shear, zero, zero]).reshape(2, 2)
        elif _shear.shape == torch.Size([2]):
            _shear = torch.stack([_shear[0], _shear[1], zero, zero]).reshape(2, 2)
        elif _shear.shape == torch.Size([4]):
            _shear = _shear.reshape(2, 2)
        if _shear.shape != torch.Size([2, 2]):
            raise ValueError(f"'shear' shall be either a scalar, (2,), (4,) or (2, 2). Got {self.shear}.")
        return _shear

    def _parse_scale(self, _device: torch.device, _dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Parse scale parameter into a 2x2 tensor."""
        if self.scale is None:
            return None
        _scale = torch.as_tensor(self.scale, device=_device, dtype=_dtype)
        if _scale.shape == torch.Size([2]):
            _scale = _scale.unsqueeze(0).repeat(2, 1)
        elif _scale.shape == torch.Size([4]):
            _scale = _scale.reshape(2, 2)
        elif _scale.shape != torch.Size([2, 2]):
            raise ValueError(f"'scale' shall be either shape (2), (4), or (2, 2). Got {self.scale}.")
        _singular_range_check(_scale[0], "scale-x", bounds=(0, float("inf")), mode="2d")
        _singular_range_check(_scale[1], "scale-y", bounds=(0, float("inf")), mode="2d")
        return _scale

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        _device, _dtype = self.device, self.dtype
        n = 1 if self.same_on_batch else batch_size

        # Parse parameters
        angle_range = _range_bound(self.degrees, "degrees", 0, (-360, 360)).to(device=_device, dtype=_dtype)
        _translate: Optional[torch.Tensor] = None
        if self.translate is not None:
            _translate = torch.as_tensor(self.translate, device=_device, dtype=_dtype)
            _singular_range_check(_translate, "translate", bounds=(0, 1), mode="2d")
        _scale = self._parse_scale(_device, _dtype)
        _shear = self._parse_shear(_device, _dtype)

        # Sample angle
        angle = torch.empty(n, device=_device, dtype=_dtype).uniform_(angle_range[0].item(), angle_range[1].item())
        if self.same_on_batch:
            angle = angle.expand(batch_size)

        # Sample translations
        if _translate is not None:
            max_dx, max_dy = _translate[0].item() * width, _translate[1].item() * height
            tx = (torch.rand(n, device=_device, dtype=_dtype) - 0.5) * 2.0 * max_dx
            ty = (torch.rand(n, device=_device, dtype=_dtype) - 0.5) * 2.0 * max_dy
            if self.same_on_batch:
                tx, ty = tx.expand(batch_size), ty.expand(batch_size)
            translations = torch.stack([tx, ty], dim=-1)
        else:
            translations = torch.zeros((batch_size, 2), device=_device, dtype=_dtype)

        # Center
        center = torch.tensor([width, height], device=_device, dtype=_dtype).view(1, 2) / 2.0 - 0.5
        center = center.expand(batch_size, -1)

        # Sample scale
        if _scale is not None:
            scale_x = torch.empty(n, device=_device, dtype=_dtype).uniform_(_scale[0, 0].item(), _scale[0, 1].item())
            scale_y = torch.empty(n, device=_device, dtype=_dtype).uniform_(_scale[1, 0].item(), _scale[1, 1].item())
            if self.same_on_batch:
                scale_x, scale_y = scale_x.expand(batch_size), scale_y.expand(batch_size)
            scale = torch.stack([scale_x, scale_y], dim=-1)
        else:
            scale = torch.ones((batch_size, 2), device=_device, dtype=_dtype)

        # Sample shear
        if _shear is not None:
            shear_x = torch.empty(n, device=_device, dtype=_dtype).uniform_(_shear[0, 0].item(), _shear[0, 1].item())
            shear_y = torch.empty(n, device=_device, dtype=_dtype).uniform_(_shear[1, 0].item(), _shear[1, 1].item())
            if self.same_on_batch:
                shear_x, shear_y = shear_x.expand(batch_size), shear_y.expand(batch_size)
        else:
            shear_x = shear_y = torch.zeros(batch_size, device=_device, dtype=_dtype)

        return {
            "translations": translations,
            "center": center,
            "scale": scale,
            "angle": angle,
            "shear_x": shear_x,
            "shear_y": shear_y,
        }

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        return get_affine_matrix2d(
            torch.as_tensor(params["translations"], device=input.device, dtype=input.dtype),
            torch.as_tensor(params["center"], device=input.device, dtype=input.dtype),
            torch.as_tensor(params["scale"], device=input.device, dtype=input.dtype),
            torch.as_tensor(params["angle"], device=input.device, dtype=input.dtype),
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
            fill_value=flags["fill_value"],
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
