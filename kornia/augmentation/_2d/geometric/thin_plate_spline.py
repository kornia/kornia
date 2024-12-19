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

from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.constants import SamplePadding
from kornia.core import Tensor, tensor
from kornia.geometry.transform import get_tps_transform, warp_image_tps


# NOTE: This NEEDS to be updated. It is out of the random generator controller.
class RandomThinPlateSpline(AugmentationBase2D):
    r"""Add random noise to the Thin Plate Spline algorithm.

    .. image:: _static/img/RandomThinPlateSpline.png

    Args:
        scale: the scale factor to apply to the destination points.
        align_corners: Interpolation flag used by ``grid_sample``.
        mode: Interpolation mode used by `grid_sample`. Either 'bilinear' or 'nearest'.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_image_tps`.

    Examples:
        >>> img = torch.ones(1, 1, 2, 2)
        >>> out = RandomThinPlateSpline()(img)
        >>> out.shape
        torch.Size([1, 1, 2, 2])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomThinPlateSpline(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        scale: float = 0.2,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {
            "align_corners": align_corners,
            "padding_mode": SamplePadding.get(padding_mode),
        }
        self.dist = torch.distributions.Uniform(-scale, scale)

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        B, _, _, _ = shape
        src = tensor([[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]]).expand(B, 5, 2)  # Bx5x2
        dst = src + self.dist.rsample(src.shape)
        return {"src": src, "dst": dst}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        src = params["src"].to(input)
        dst = params["dst"].to(input)
        # NOTE: warp_image_tps need to use inverse parameters
        kernel, affine = get_tps_transform(dst, src)
        return warp_image_tps(input, src, kernel, affine, flags["align_corners"], flags["padding_mode"].name.lower())
