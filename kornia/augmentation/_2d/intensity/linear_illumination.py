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

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core.check import KORNIA_CHECK
from kornia.enhance.normalize import normalize_min_max


class RandomLinearIllumination(IntensityAugmentationBase2D):
    r"""Applies random 2D Linear illumination patterns to a batch of images.

    .. image:: _static/img/RandomLinearIllumination.png

    Args:
        gain: Range for the gain factor (intensity) applied to the generated illumination.
        sign: Range for the sign of the distribution. If only one sign is needed,
        insert only as a tuple or float.
        p: Probability of applying the transformation.
        same_on_batch: If True, apply the same transformation across the entire batch. Default is False.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> input = torch.ones(1, 3, 3, 3) * 0.5
        >>> aug = RandomLinearIllumination(gain=0.25, p=1.)
        >>> aug(input)
        tensor([[[[0.2500, 0.2500, 0.2500],
                  [0.3750, 0.3750, 0.3750],
                  [0.5000, 0.5000, 0.5000]],
        <BLANKLINE>
                 [[0.2500, 0.2500, 0.2500],
                  [0.3750, 0.3750, 0.3750],
                  [0.5000, 0.5000, 0.5000]],
        <BLANKLINE>
                 [[0.2500, 0.2500, 0.2500],
                  [0.3750, 0.3750, 0.3750],
                  [0.5000, 0.5000, 0.5000]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomLinearIllumination(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        gain: Optional[Union[float, Tuple[float, float]]] = (0.01, 0.2),
        sign: Optional[Union[float, Tuple[float, float]]] = (-1.0, 1.0),
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        # Validation and initialization of gain parameter.
        if isinstance(gain, (tuple, float)):
            if isinstance(gain, float):
                gain = (gain, gain)
            elif len(gain) == 1:
                gain = (gain[0], gain[0])
            elif len(gain) > 2 or len(gain) <= 0:
                raise ValueError(
                    "The length of gain must be greater than 0 "
                    "and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("gain must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in gain),
            "gain values must be between 0 and 1. Recommended values less than 0.2.",
        )
        self.gain = gain

        # Validation and initialization of sign parameter.
        if isinstance(sign, (tuple, float)):
            if isinstance(sign, float):
                sign = (sign, sign)
            elif len(sign) == 1:
                sign = (sign[0], sign[0])
            elif len(sign) > 2 or len(sign) <= 0:
                raise ValueError(
                    "The length of sign must be greater than 0 "
                    "and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("sign must be a tuple or a float")
        KORNIA_CHECK(
            all(-1 <= el <= 1 for el in sign),
            "sign of linear value must be between -1 and 1.",
        )
        self.sign_range = sign

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size, channels, height, width = batch_shape
        _device, _dtype = self.device, self.dtype

        if self.same_on_batch:
            gain_factor = (
                torch.empty(1, 1, 1, 1, device=_device, dtype=_dtype)
                .uniform_(self.gain[0], self.gain[1])
                .expand(batch_size, 1, 1, 1)
            )
            sign_val = (
                torch.empty(1, 1, 1, 1, device=_device, dtype=_dtype)
                .uniform_(self.sign_range[0], self.sign_range[1])
                .expand(batch_size, 1, 1, 1)
            )
            directions = (
                torch.empty(1, 1, 1, 1, device=_device).uniform_(0, 4).to(torch.int8).expand(batch_size, 1, 1, 1)
            )
        else:
            gain_factor = torch.empty(batch_size, 1, 1, 1, device=_device, dtype=_dtype).uniform_(
                self.gain[0], self.gain[1]
            )
            sign_val = torch.empty(batch_size, 1, 1, 1, device=_device, dtype=_dtype).uniform_(
                self.sign_range[0], self.sign_range[1]
            )
            directions = torch.empty(batch_size, 1, 1, 1, device=_device).uniform_(0, 4).to(torch.int8)

        sign = torch.where(
            sign_val >= 0.0,
            torch.tensor(1, device=_device, dtype=_dtype),
            torch.tensor(-1, device=_device, dtype=_dtype),
        )

        # Precompute 1D ramps
        ramp_h = torch.linspace(0, 1, height, device=_device, dtype=_dtype).view(1, 1, height, 1)
        ramp_h_rev = torch.linspace(1, 0, height, device=_device, dtype=_dtype).view(1, 1, height, 1)
        ramp_w = torch.linspace(0, 1, width, device=_device, dtype=_dtype).view(1, 1, 1, width)
        ramp_w_rev = torch.linspace(1, 0, width, device=_device, dtype=_dtype).view(1, 1, 1, width)

        # Broadcast masks for each direction
        d = directions.to(torch.int64)  # [B,1,1,1]
        m0 = (d == 0).to(_dtype)  # lower
        m1 = (d == 1).to(_dtype)  # upper
        m2 = (d == 2).to(_dtype)  # left
        m3 = (d == 3).to(_dtype)  # right

        # Build [B,1,H,W] gradient in one shot
        grad_b1 = m0 * ramp_h + m1 * ramp_h_rev + m2 * ramp_w + m3 * ramp_w_rev

        # Expand to [B,C,H,W]
        gradient = grad_b1.expand(batch_size, channels, height, width)

        # Apply sign and gain
        gradient = sign * gain_factor * gradient

        return {"gradient": gradient.to(device=_device, dtype=_dtype)}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Apply random gaussian gradient illumination to the input image."""
        return input.add(params["gradient"].to(input)).clamp(0, 1)


class RandomLinearCornerIllumination(IntensityAugmentationBase2D):
    r"""Applies random 2D Linear from corner illumination patterns to a batch of images.

    .. image:: _static/img/RandomLinearCornerIllumination.png

    Args:
        gain: Range for the gain factor (intensity) applied to the generated illumination.
        sign: Range for the sign of the distribution. If only one sign is needed,
        insert only as a tuple or float.
        p: Probability of applying the transformation.
        same_on_batch: If True, apply the same transformation across the entire batch. Default is False.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> input = torch.ones(1, 3, 3, 3) * 0.5
        >>> aug = RandomLinearCornerIllumination(gain=0.25, p=1.)
        >>> aug(input)
        tensor([[[[0.3750, 0.4375, 0.5000],
                  [0.3125, 0.3750, 0.4375],
                  [0.2500, 0.3125, 0.3750]],
        <BLANKLINE>
                 [[0.3750, 0.4375, 0.5000],
                  [0.3125, 0.3750, 0.4375],
                  [0.2500, 0.3125, 0.3750]],
        <BLANKLINE>
                 [[0.3750, 0.4375, 0.5000],
                  [0.3125, 0.3750, 0.4375],
                  [0.2500, 0.3125, 0.3750]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomLinearCornerIllumination(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        gain: Optional[Union[float, Tuple[float, float]]] = (0.01, 0.2),
        sign: Optional[Union[float, Tuple[float, float]]] = (-1.0, 1.0),
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        # Validation and initialization of gain parameter.
        if isinstance(gain, (tuple, float)):
            if isinstance(gain, float):
                gain = (gain, gain)
            elif len(gain) == 1:
                gain = (gain[0], gain[0])
            elif len(gain) > 2 or len(gain) <= 0:
                raise ValueError(
                    "The length of gain must be greater than 0 "
                    "and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("gain must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in gain),
            "gain values must be between 0 and 1. Recommended values less than 0.2.",
        )
        self.gain = gain

        # Validation and initialization of sign parameter.
        if isinstance(sign, (tuple, float)):
            if isinstance(sign, float):
                sign = (sign, sign)
            elif len(sign) == 1:
                sign = (sign[0], sign[0])
            elif len(sign) > 2 or len(sign) <= 0:
                raise ValueError(
                    "The length of sign must be greater than 0 "
                    "and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("sign must be a tuple or a float")
        KORNIA_CHECK(
            all(-1 <= el <= 1 for el in sign),
            "sign of linear value must be between -1 and 1.",
        )
        self.sign_range = sign

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size, channels, height, width = batch_shape
        _device, _dtype = self.device, self.dtype

        if self.same_on_batch:
            gain_factor = (
                torch.empty(1, 1, 1, 1, device=_device, dtype=_dtype)
                .uniform_(self.gain[0], self.gain[1])
                .expand(batch_size, 1, 1, 1)
            )
            sign_val = (
                torch.empty(1, 1, 1, 1, device=_device, dtype=_dtype)
                .uniform_(self.sign_range[0], self.sign_range[1])
                .expand(batch_size, 1, 1, 1)
            )
            directions = (
                torch.empty(1, 1, 1, 1, device=_device).uniform_(0, 4).to(torch.long).expand(batch_size, 1, 1, 1)
            )
        else:
            gain_factor = torch.empty(batch_size, 1, 1, 1, device=_device, dtype=_dtype).uniform_(
                self.gain[0], self.gain[1]
            )
            sign_val = torch.empty(batch_size, 1, 1, 1, device=_device, dtype=_dtype).uniform_(
                self.sign_range[0], self.sign_range[1]
            )
            directions = torch.empty(batch_size, 1, 1, 1, device=_device).uniform_(0, 4).to(torch.long)

        sign = torch.where(
            sign_val >= 0.0,
            torch.tensor(1, device=_device, dtype=_dtype),
            torch.tensor(-1, device=_device, dtype=_dtype),
        )

        # Compute base gradients
        y_grad = torch.linspace(0, 1, height, device=_device, dtype=_dtype).unsqueeze(1).expand(height, width)
        x_grad = torch.linspace(0, 1, width, device=_device, dtype=_dtype).unsqueeze(0).expand(height, width)

        base = torch.stack(
            [
                x_grad + y_grad,  # 0: Bottom right
                -x_grad + y_grad,  # 1: Bottom left
                x_grad - y_grad,  # 2: Upper right
                1 - (x_grad + y_grad),  # 3: Upper left
            ],
            dim=0,
        )  # (4, H, W)

        # Expand to (4, C, H, W)
        base = base.unsqueeze(1).expand(-1, channels, -1, -1)

        # Index according to directions
        gradient = base[directions.view(-1), :, :, :]  # (B, C, H, W)

        gradient = sign * gain_factor * normalize_min_max(gradient)

        return {"gradient": gradient.to(device=_device, dtype=_dtype)}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Apply random gaussian gradient illumination to the input image."""
        return input.add(params["gradient"].to(input)).clamp(0, 1)
