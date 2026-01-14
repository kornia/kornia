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
from kornia.enhance import normalize_min_max
from kornia.filters.kernels import gaussian


class RandomGaussianIllumination(IntensityAugmentationBase2D):
    r"""Applies random 2D Gaussian illumination patterns to a batch of images.

    .. image:: _static/img/RandomGaussianIllumination.png

    Args:
        gain: Range for the gain factor (intensity) applied to the generated illumination.
        center: The center coordinates of the Gaussian distribution are expressed as a
        percentage of the spatial dimensions :math:(H, W).
        sigma: The sigma values (standard deviation) of the Gaussian distribution are expressed as a
        percentage of the spatial dimensions :math:(H, W).
        sign: Range for the sign of the Gaussian distribution. If only one sign is needed,
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
        >>> aug = RandomGaussianIllumination(gain=0.5, p=1.)
        >>> aug(input)
        tensor([[[[0.7266, 1.0000, 0.7266],
                  [0.6621, 0.9121, 0.6621],
                  [0.5000, 0.6911, 0.5000]],
        <BLANKLINE>
                 [[0.7266, 1.0000, 0.7266],
                  [0.6621, 0.9121, 0.6621],
                  [0.5000, 0.6911, 0.5000]],
        <BLANKLINE>
                 [[0.7266, 1.0000, 0.7266],
                  [0.6621, 0.9121, 0.6621],
                  [0.5000, 0.6911, 0.5000]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomGaussianIllumination(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        gain: Optional[Union[float, Tuple[float, float]]] = (0.01, 0.15),
        center: Optional[Union[float, Tuple[float, float]]] = (0.1, 0.9),
        sigma: Optional[Union[float, Tuple[float, float]]] = (0.2, 1.0),
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

        # Validation and initialization of center parameter.
        if isinstance(center, (tuple, float)):
            if isinstance(center, float):
                center = (center, center)
            elif len(center) == 1:
                center = (center[0], center[0])
            elif len(center) > 2 or len(center) <= 0:
                raise ValueError(
                    "The length of center must be greater than 0 "
                    "and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("center must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in center),
            "center of gaussian value must be between 0 and 1.",
        )
        self.center = center

        # Validation and initialization of sigma parameter.
        if isinstance(sigma, (tuple, float)):
            if isinstance(sigma, float):
                sigma = (sigma, sigma)
            elif len(sigma) == 1:
                sigma = (sigma[0], sigma[0])
            elif len(sigma) > 2 or len(sigma) <= 0:
                raise ValueError(
                    "The length of sigma must be greater than 0 "
                    "and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("sigma must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in sigma),
            "sigma of gaussian value must be between 0 and 1.",
        )
        self.sigma = sigma

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
            "sign of gaussian value must be between -1 and 1.",
        )
        self.sign_range = sign

        def _apply_transform(
            input: torch.Tensor,
            params: Dict[str, torch.Tensor],
            flags: Dict[str, Any],
            transform: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return input.add_(params["gradient"]).clamp_(0, 1)

        self._fn = _apply_transform

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size, channels, height, width = batch_shape
        _device, _dtype = self.device, self.dtype

        if self.same_on_batch:
            gain_factor = (
                torch.empty(1, 1, 1, 1, device=_device, dtype=_dtype)
                .uniform_(self.gain[0], self.gain[1])
                .expand(batch_size, 1, 1, 1)
            )
            sigma_x = width * torch.empty(1, 1, device=_device, dtype=_dtype).uniform_(
                self.sigma[0], self.sigma[1]
            ).expand(batch_size, 1)
            center_x = torch.round(
                width * torch.empty(1, 1, device=_device, dtype=_dtype).uniform_(self.center[0], self.center[1])
            ).expand(batch_size, 1)
            sigma_y = height * torch.empty(1, 1, device=_device, dtype=_dtype).uniform_(
                self.sigma[0], self.sigma[1]
            ).expand(batch_size, 1)
            center_y = torch.round(
                height * torch.empty(1, 1, device=_device, dtype=_dtype).uniform_(self.center[0], self.center[1])
            ).expand(batch_size, 1)
            sign_val = (
                torch.empty(1, 1, 1, 1, device=_device, dtype=_dtype)
                .uniform_(self.sign_range[0], self.sign_range[1])
                .expand(batch_size, 1, 1, 1)
            )
        else:
            gain_factor = torch.empty(batch_size, 1, 1, 1, device=_device, dtype=_dtype).uniform_(
                self.gain[0], self.gain[1]
            )
            sigma_x = width * torch.empty(batch_size, 1, device=_device, dtype=_dtype).uniform_(
                self.sigma[0], self.sigma[1]
            )
            center_x = torch.round(
                width
                * torch.empty(batch_size, 1, device=_device, dtype=_dtype).uniform_(self.center[0], self.center[1])
            )
            sigma_y = height * torch.empty(batch_size, 1, device=_device, dtype=_dtype).uniform_(
                self.sigma[0], self.sigma[1]
            )
            center_y = torch.round(
                height
                * torch.empty(batch_size, 1, device=_device, dtype=_dtype).uniform_(self.center[0], self.center[1])
            )
            sign_val = torch.empty(batch_size, 1, 1, 1, device=_device, dtype=_dtype).uniform_(
                self.sign_range[0], self.sign_range[1]
            )

        sign = torch.where(
            sign_val >= 0.0,
            torch.tensor(1.0, device=_device, dtype=_dtype),
            torch.tensor(-1.0, device=_device, dtype=_dtype),
        )

        # Generate random gaussian for create a 2D gaussian image.
        gauss_x = gaussian(width, sigma_x, mean=center_x, device=_device, dtype=_dtype).unsqueeze(1)
        gauss_y = gaussian(height, sigma_y, mean=center_y, device=_device, dtype=_dtype).unsqueeze(2)

        # gradient = (batch_size, channels, height, width)
        gradient = (gauss_y @ gauss_x).unsqueeze_(1).repeat(1, channels, 1, 1)

        # Normalize between 0-1 to apply the gain factor.
        gradient = normalize_min_max(gradient, min_val=0.0, max_val=1.0)
        gradient = sign.mul_(gain_factor).mul(gradient)

        return {"gradient": gradient}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Apply random gaussian gradient illumination to the input image."""
        return self._fn(input=input, params=params, flags=flags, transform=transform)

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
        mode: Optional[str] = None,
        options: Optional[Dict[Any, Any]] = None,
        disable: bool = False,
    ) -> "RandomGaussianIllumination":
        self._fn = torch.compile(
            self._fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )

        return self
