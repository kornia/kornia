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
from kornia.augmentation.random_generator._2d import LinearCornerIlluminationGenerator, LinearIlluminationGenerator
from kornia.core.check import KORNIA_CHECK


class RandomLinearIllumination(IntensityAugmentationBase2D):
    r"""Applies random 2D Linear illumination patterns to a batch of images.

    .. image:: _static/img/RandomLinearIllumination.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

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

    Convention:
        - the class draws ``_params["gradient"]``, a tensor of the input's shape, adds it to the image and
          clamps the sum into ``[0, 1]``, so the output stays inside that range even when the input does not.
        - ``sign`` is drawn per sample, from ``(-1.0, 1.0)`` by default, and fixes the direction of that
          sample's gradient, so one batch can hold both a darkened and a brightened image. A point range such
          as ``sign=1.0`` fixes the direction for every sample.
        - the clamp bites on in-range images too: once ``gain`` exceeds the headroom between the image and the
          bound, the sum is cut there rather than rescaled.
        - the module pickles, deep-copies and passes through ``torch.save``, and the copy reproduces the
          original's output under the same seed.

    .. warning::
        An input whose values are all negative comes back as an all-zero image. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

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

        # Validation and initialization of amount parameter.
        if isinstance(gain, (tuple, float)):
            if isinstance(gain, float):
                gain = (gain, gain)
            elif len(gain) == 1:
                gain = (gain[0], gain[0])
            elif len(gain) > 2 or len(gain) <= 0:
                raise ValueError(
                    "The length of gain must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("gain must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in gain),
            "gain values must be between 0 and 1. Recommended values less than 0.2.",
        )

        if isinstance(sign, (tuple, float)):
            if isinstance(sign, float):
                sign = (sign, sign)
            elif len(sign) == 1:
                sign = (sign[0], sign[0])
            elif len(sign) > 2 or len(sign) <= 0:
                raise ValueError(
                    "The length of sign must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("sign must be a tuple or a float")
        KORNIA_CHECK(
            all(-1 <= el <= 1 for el in sign),
            "sign of linear value must be between -1 and 1.",
        )

        # Generator of random parameters and masks.
        self._param_generator = LinearIlluminationGenerator(gain, sign)

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

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

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

    Convention:
        - the class draws ``_params["gradient"]``, a tensor of the input's shape, adds it to the image and
          clamps the sum into ``[0, 1]``, so the output stays inside that range even when the input does not.
        - ``sign`` is drawn per sample, from ``(-1.0, 1.0)`` by default, and fixes the direction of that
          sample's gradient, so one batch can hold both a darkened and a brightened image. A point range such
          as ``sign=1.0`` fixes the direction for every sample.
        - the clamp bites on in-range images too: once ``gain`` exceeds the headroom between the image and the
          bound, the sum is cut there rather than rescaled.
        - the module pickles, deep-copies and passes through ``torch.save``, and the copy reproduces the
          original's output under the same seed.

    .. warning::
        An input whose values are all negative comes back as an all-zero image. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

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

        # Validation and initialization of amount parameter.
        if isinstance(gain, (tuple, float)):
            if isinstance(gain, float):
                gain = (gain, gain)
            elif len(gain) == 1:
                gain = (gain[0], gain[0])
            elif len(gain) > 2 or len(gain) <= 0:
                raise ValueError(
                    "The length of gain must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("gain must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in gain),
            "gain values must be between 0 and 1. Recommended values less than 0.2.",
        )

        if isinstance(sign, (tuple, float)):
            if isinstance(sign, float):
                sign = (sign, sign)
            elif len(sign) == 1:
                sign = (sign[0], sign[0])
            elif len(sign) > 2 or len(sign) <= 0:
                raise ValueError(
                    "The length of sign must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("sign must be a tuple or a float")
        KORNIA_CHECK(
            all(-1 <= el <= 1 for el in sign),
            "sign of linear value must be between -1 and 1.",
        )

        # Generator of random parameters and masks.
        self._param_generator = LinearCornerIlluminationGenerator(gain, sign)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Apply random gaussian gradient illumination to the input image."""
        return input.add(params["gradient"].to(input)).clamp(0, 1)
