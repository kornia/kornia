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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator._2d import GaussianIlluminationGenerator
from kornia.core.check import KORNIA_CHECK


def _apply_gaussian_illumination(
    input: torch.Tensor,
    params: Dict[str, torch.Tensor],
    flags: Dict[str, Any],
    transform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a sampled illumination pattern through a picklable, compilable callable."""
    # The gradient comes from the generator on CPU.
    # We must explicitly move it to the input's device before adding.
    gradient = params["gradient"].to(input.device, input.dtype)

    # Out-of-place add: in-place would mutate the caller's tensor and corrupt
    # the non-transformed branch, which shares the same input.
    return input.add(gradient).clamp_(0, 1)


class RandomGaussianIllumination(IntensityAugmentationBase2D):
    r"""Applies random 2D Gaussian illumination patterns to a batch of images.

    .. image:: _static/img/RandomGaussianIllumination.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

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

    Convention:
        - the class draws ``_params["gradient"]``, a tensor with the original normalized ``(B, C, H, W)``
          input shape; a ``(C, H, W)`` input remains batched in stored parameters even when ``keepdim=True``.
          It adds the field to the image and
          clamps the sum into ``[0, 1]``, so the output stays inside that range even when the input does not --
          except where the Gaussian kernel itself is NaN, see the warning below.
        - ``sign`` is drawn per sample, from ``(-1.0, 1.0)`` by default, and only whether the draw is negative
          is used: it decides whether that sample's gradient darkens or brightens, so one batch can hold both
          a darkened and a brightened image. A point range such as ``sign=1.0`` brightens every sample.
        - the clamp bites on in-range images too: once ``gain`` exceeds the headroom between the image and the
          bound, the sum is cut there rather than rescaled.
        - ``sigma`` is a fraction of the axis length, not an absolute width: the generator draws it and
          multiplies by the image's width and height before building the kernel, so the same ``sigma`` is a
          narrower kernel on a smaller image.
        - the module pickles, deep-copies and passes through ``torch.save``, and the copy reproduces the
          original's output under the same seed. After ``.compile()``, which swaps in a compiled transform,
          it no longer pickles or passes through ``torch.save``, although it still deep-copies.

    .. warning::
        An all-negative input can come back as an all-zero image when the sampled gradient does not raise it
        above zero; a positive sampled gradient can recover values instead. Tracked in
        `#4430 <https://github.com/kornia/kornia/issues/4430>`_.

    .. warning::
        A ``sigma`` at or near ``0``, which the constructor admits, makes the whole output NaN in every dtype.
        :func:`kornia.filters.kernels.gaussian` normalizes by ``gauss.sum()``, which underflows to zero there,
        so the kernel is ``0 / 0``. ``sigma=0.0`` does it at any size; because an even-length axis carries a
        half-pixel offset, no sample sits at the mean and a small non-zero ``sigma`` does it too -- a ``4 x 4``
        image is NaN at ``sigma=0.01`` where a ``3 x 3`` one is not. The default ``sigma=(0.2, 1.0)`` is
        unaffected. Tracked in `#4589 <https://github.com/kornia/kornia/issues/4589>`_.

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

        if isinstance(center, (tuple, float)):
            if isinstance(center, float):
                center = (center, center)
            elif len(center) == 1:
                center = (center[0], center[0])
            elif len(center) > 2 or len(center) <= 0:
                raise ValueError(
                    "The length of center must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("center must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in center),
            "center of gaussian value must be between 0 and 1.",
        )

        if isinstance(sigma, (tuple, float)):
            if isinstance(sigma, float):
                sigma = (sigma, sigma)
            elif len(sigma) == 1:
                sigma = (sigma[0], sigma[0])
            elif len(sigma) > 2 or len(sigma) <= 0:
                raise ValueError(
                    "The length of sigma must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple or a float."
                )
        else:
            raise ValueError("sigma must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in sigma),
            "sigma of gaussian value must be between 0 and 1.",
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
            "sign of gaussian value must be between -1 and 1.",
        )

        # Generator of random parameters and masks.
        self._param_generator = GaussianIlluminationGenerator(gain, center, sigma, sign)

        self._fn = _apply_gaussian_illumination

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
    ) -> RandomGaussianIllumination:
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
