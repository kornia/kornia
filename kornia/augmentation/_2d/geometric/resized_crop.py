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

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.base import _input_metadata_only
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.augmentation.utils._crop import _compiled_slice_resize
from kornia.constants import Resample
from kornia.core.utils import is_compiling
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform


class RandomResizedCrop(GeometricAugmentationBase2D):
    r"""Crop random patches in an image torch.Tensor and resizes to a given size.

    .. image:: _static/img/RandomResizedCrop.png

    Args:
        size: Desired output size (out_h, out_w) of each edge.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        scale: range of size of the origin size cropped.
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        resample: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of the augmentation been applied.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the torch.Tensor based
                       on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                       to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                       differentiability.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input torch.Tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation torch.Tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation torch.Tensor and returned.

    Convention:
        See :class:`~kornia.augmentation.GeometricAugmentationBase2D` for coordinates, defaults and inverse.
        ``size`` is an ``(height, width)`` tuple; a bare integer raises, unlike :class:`CenterCrop`
        (`#4417 <https://github.com/kornia/kornia/issues/4417>`_). ``p`` selects or skips the whole batch together.
        Within a selected batch, the generator tries ten candidate crops per image, sampling area fractions from
        ``scale`` and width/height ratios from ``ratio`` (shared with ``same_on_batch=True``), and resizes the first
        that fits to the requested output size. As in torchvision's ``get_params``, a candidate may equal the input,
        and when none fits the fallback size keeps an input whose width/height is within ``ratio`` whole, so
        ``scale=(1.0, 1.0)`` keeps the whole image; otherwise it keeps the full width (input narrower than
        ``min(ratio)``) or the full height (wider than ``max(ratio)``). Unlike torchvision, which centres the fallback
        crop, it is placed at a random position like any other crop.

        Both cropping modes use the configured interpolation and ``align_corners``, so slice mode raises for
        ``resample="nearest"`` unless ``align_corners=None``
        (`#4802 <https://github.com/kornia/kornia/issues/4802>`_). At ``align_corners=False`` the two modes give
        different images, and slice mode does not follow ``transform_matrix``
        (`#4804 <https://github.com/kornia/kornia/issues/4804>`_). Only resample mode supports :meth:`inverse`,
        which resamples onto the original canvas and cannot recover discarded information.

    Note:
        Compiled slice-mode interpolation matches eager execution to floating-point tolerance,
        not bitwise. Eager execution retains native slicing and resizing for performance;
        the tensorized compiled path avoids recompilation as crop coordinates change.

    Example:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.tensor([[[0., 1., 2.],
        ...                         [3., 4., 5.],
        ...                         [6., 7., 8.]]])
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[3.0000, 4.0000, 5.0000],
                  [4.5000, 5.5000, 6.5000],
                  [6.0000, 7.0000, 8.0000]]]])
        >>> aug.inverse(out, padding_mode="border")
        tensor([[[[3., 4., 5.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.08, 1.0),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (3.0 / 4.0, 4.0 / 3.0),
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ) -> None:
        # Since PyTorch does not support ragged torch.Tensor. So cropping function happens all the time.
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self._param_generator = rg.ResizedCropGenerator(size, scale, ratio)
        self.flags = {
            "size": size,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
            "cropping_mode": cropping_mode,
            "padding_mode": "zeros",
        }
        # In "slice" mode the image is cropped by index (``crop_by_indices``) and apply_transform
        # never reads the transform matrix — only box/keypoint propagation does. Defer building it
        # (a full get_perspective_transform) until ``.transform_matrix`` is actually accessed.
        # "resample" mode warps through the matrix, so it must be built eagerly.
        self._compute_matrix_lazily = cropping_mode == "slice"

    @_input_metadata_only
    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        if flags["cropping_mode"] in ("resample", "slice"):
            transform: torch.Tensor = get_perspective_transform(params["src"].to(input), params["dst"].to(input))
            return transform.expand(input.shape[0], -1, -1)
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            if not isinstance(transform, torch.Tensor):
                raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

            return crop_by_transform_mat(
                input,
                transform[:, :2, :],
                flags["size"],
                mode=flags["resample"].name.lower(),
                padding_mode="zeros",
                align_corners=flags["align_corners"],
            )
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            if is_compiling():
                return _compiled_slice_resize(
                    input, params["src"], flags["size"], flags["resample"].name.lower(), flags["align_corners"]
                )
            return crop_by_indices(
                input,
                params["src"],
                flags["size"],
                interpolation=flags["resample"].name.lower(),
                align_corners=flags["align_corners"],
            )
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        if not isinstance(size, tuple):
            raise TypeError(f"Expected the size be a tuple. Gotcha {type(size)}")

        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

        return crop_by_transform_mat(
            input,
            transform[:, :2, :],
            size,
            flags["resample"].name.lower(),
            flags["padding_mode"],
            flags["align_corners"],
        )
