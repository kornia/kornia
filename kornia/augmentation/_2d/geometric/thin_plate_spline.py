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
from kornia.augmentation.utils.helpers import _constant_tensor
from kornia.augmentation.utils.random import _sample_distribution
from kornia.constants import SamplePadding
from kornia.core.utils import _to_device_nonblocking
from kornia.geometry.transform import get_tps_transform, warp_image_tps


# NOTE: This NEEDS to be updated. It is out of the random generator controller.
class RandomThinPlateSpline(AugmentationBase2D):
    r"""Add random noise to the Thin Plate Spline algorithm.

    .. image:: _static/img/RandomThinPlateSpline.png

    Convention:
        Five control points use normalized ``(x, y)`` coordinates: ``(-1, -1)``, ``(-1, 1)``,
        ``(1, -1)``, ``(1, 1)``, and ``(0, 0)``. Each destination coordinate receives uniform
        noise between ``-scale`` and ``scale``; zero scale leaves these points unchanged.
        Sampling is bilinear, with no interpolation-mode argument. Zero scale is an identity warp up to
        interpolation roundoff for either ``align_corners``. There is no ``transform_matrix`` or ``inverse``;
        spatial labels in containers have limitations (`#4420 <https://github.com/kornia/kornia/issues/4420>`_).
        A ``mask`` data key raises ``NotImplementedError`` identifying this augmentation, even at ``p=0``.
        Its defaults and the ``align_corners`` split across classes
        (`#4412 <https://github.com/kornia/kornia/issues/4412>`_) are listed on
        :class:`~kornia.augmentation.GeometricAugmentationBase2D`.

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
        dist = self.dist
        if dist is not None and self._generator is not None:
            # Explicit draws must use the configured sampling device. Keep the
            # historical CPU distribution and draw order on the global RNG path.
            dist = torch.distributions.Uniform(
                dist.low.to(device=device), dist.high.to(device=device), validate_args=False
            )

        # 5 TPS control points in normalized coordinates
        src = _constant_tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).expand(B, 5, 2)

        if dist is None:
            noise = torch.zeros_like(src)
        elif self.same_on_batch:
            noise = _sample_distribution(dist, torch.Size((1, 5, 2)), self._generator, True).to(
                device=device, dtype=dtype
            )
            noise = noise.expand(B, 5, 2)
        else:
            noise = _sample_distribution(dist, torch.Size((B, 5, 2)), self._generator, True).to(
                device=device, dtype=dtype
            )

        dst = src + noise

        return {"src": src, "dst": dst}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = _to_device_nonblocking(params["src"], input.device, input.dtype)
        dst = _to_device_nonblocking(params["dst"], input.device, input.dtype)
        # NOTE: warp_image_tps need to use inverse parameters
        kernel, affine = get_tps_transform(dst, src)
        return warp_image_tps(input, src, kernel, affine, flags["align_corners"], flags["padding_mode"].name.lower())

    def apply_transform_mask(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reject mask inputs, which this augmentation does not support."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support mask inputs. See https://github.com/kornia/kornia/issues/4420."
        )

    def apply_non_transform_mask(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reject mask inputs even when the transformation is not applied."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support mask inputs. See https://github.com/kornia/kornia/issues/4420."
        )
