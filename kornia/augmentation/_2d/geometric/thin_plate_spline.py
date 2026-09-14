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
from kornia.geometry.transform import get_tps_transform, warp_image_tps


# NOTE: This NEEDS to be updated. It is out of the random generator controller.
class RandomThinPlateSpline(AugmentationBase2D):
    r"""Add random noise to the Thin Plate Spline algorithm.

    .. image:: _static/img/RandomThinPlateSpline.png

    Convention:
        See the shared contract on :class:`~kornia.augmentation.AugmentationBase2D`.
        Five control points use normalized ``(x, y)`` coordinates: ``(-1, -1)``, ``(-1, 1)``,
        ``(1, -1)``, ``(1, 1)``, and ``(0, 0)``. Each destination coordinate receives uniform
        noise between ``-scale`` and ``scale``; zero scale leaves these points unchanged.
        Sampling is bilinear with zero padding and ``align_corners=False`` by default; no
        interpolation-mode argument is exposed. Output spatial size is unchanged, and there is
        no ``transform_matrix`` or ``inverse`` interface. Spatial labels in containers have
        additional limitations; see `#4420 <https://github.com/kornia/kornia/issues/4420>`_.

    .. warning::
        Unchanged control points do not give an identity image at the default
        ``align_corners=False`` because the sampling lattice is corner-aligned. Setting
        ``align_corners=True`` removes this mismatch for float32/float64 inputs, up to numerical precision.
        The solve has additional dtype limitations documented on
        :func:`~kornia.geometry.transform.get_tps_transform`. The sampling-grid mismatch is tracked in
        `#3928 <https://github.com/kornia/kornia/issues/3928>`_ and
        `#4411 <https://github.com/kornia/kornia/issues/4411>`_.

    Args:
        scale: the non-negative scale factor to apply to the destination points.
            Zero leaves the control points unchanged.
        align_corners: Interpolation flag used by ``grid_sample``.
        padding_mode: Padding used by ``grid_sample``: 'zeros', 'border' or 'reflection'.
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
        self.dist = None if scale == 0 else torch.distributions.Uniform(-scale, scale)

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        B, _, _, _ = shape

        device = self.device
        dtype = self.dtype

        # 5 TPS control points in normalized coordinates
        src = torch.tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).expand(B, 5, 2)

        if self.dist is None:
            noise = torch.zeros_like(src)
        elif self.same_on_batch:
            noise = self.dist.rsample((1, 5, 2)).to(device=device, dtype=dtype)
            noise = noise.expand(B, 5, 2)
        else:
            noise = self.dist.rsample((B, 5, 2)).to(device=device, dtype=dtype)

        dst = src + noise

        return {"src": src, "dst": dst}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = params["src"].to(input)
        dst = params["dst"].to(input)
        # NOTE: warp_image_tps need to use inverse parameters
        kernel, affine = get_tps_transform(dst, src)
        return warp_image_tps(input, src, kernel, affine, flags["align_corners"], flags["padding_mode"].name.lower())
