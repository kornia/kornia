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

from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import BorderType
from kornia.filters import box_blur


class RandomBoxBlur(IntensityAugmentationBase2D):
    """Add random blur with a box filter to an image tensor.

    .. image:: _static/img/RandomBoxBlur.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`.

    Args:
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``constant``, ``reflect``, ``replicate`` or ``circular``, given as a
          case-insensitive string, a ``BorderType`` member or its integer value
          (CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3).
        normalized: selects the implementation of :func:`kornia.filters.box_blur`: ``True`` computes the blur
          as two 1D passes (``separable=True``), ``False`` as one 2D pass. The kernel is L1-normalized either
          way, so both return the mean of each window, never its sum, and differ only by float rounding.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Convention:
        - ``kernel_size`` is ``(kH, kW)``: the first entry counts rows and the second counts columns, as in
          :func:`kornia.filters.box_blur`. An even entry is accepted and centres the window half a pixel toward
          the top-left rather than raising -- a ``(2, 2)`` window on an impulse spreads it over the impulse's
          own row and column and the one before each -- where :class:`RandomGaussianBlur` and
          :class:`RandomMedianBlur` raise on the forward pass.
        - ``normalized`` reaches that function as its ``separable`` argument, so it selects the implementation
          rather than a normalization: both settings are L1-normalized means, they agree to float rounding, and
          a constant image survives either up to the rounding of the kernel weights. ``border_type`` defaults
          to ``"reflect"``, as the function does, and ``normalized=True`` matches
          :func:`kornia.filters.box_blur`'s ``separable=True`` default.
        - the output is not clamped. At the default ``border_type="reflect"`` every output value is a weighted
          average of input values and stays between the input's own extremes, up to the rounding of the kernel
          weights; ``border_type="constant"`` pads with zeros, which pulls a border pixel toward ``0``: below
          the input's minimum for a positive image, and above its maximum for a negative one.

    .. warning::
        At the default ``border_type="reflect"``, an image with a spatial axis no longer than half the kernel's extent
        along that axis raises a raw torch ``RuntimeError`` about the padding rather than a kornia error naming the
        class or the shape. ``"constant"`` and ``"replicate"`` run on the same image; ``"circular"`` raises a padding
        error of its own, also raw, once the kernel radius exceeds that axis. Tracked in `#4559
        <https://github.com/kornia/kornia/issues/4559>`_.

    .. note::
        This function internally uses :func:`kornia.filters.box_blur`.

    Examples:
        >>> img = torch.ones(1, 1, 24, 24)
        >>> out = RandomBoxBlur((7, 7))(img)
        >>> out.shape
        torch.Size([1, 1, 24, 24])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomBoxBlur((7, 7), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        border_type: Union[int, str, BorderType] = "reflect",
        normalized: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {
            "kernel_size": kernel_size,
            "border_type": BorderType.get(border_type),
            "normalized": normalized,
        }

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return box_blur(
            input,
            flags["kernel_size"],
            # a per-call `border_type` override reaches `flags` unnormalized, so normalize here too
            border_type=BorderType.get(flags["border_type"]).name.lower(),
            separable=flags["normalized"],
        )
