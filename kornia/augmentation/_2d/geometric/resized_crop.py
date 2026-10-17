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
from kornia.constants import Resample
from kornia.core.utils import is_compiling
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform
from kornia.geometry.transform._crop import (
    _compiled_slice_resize,
    _corner_aligned_resize_transform,
    _half_pixel_resize_transform,
)
from kornia.geometry.transform.crop2d import _crop_inverse_by_transform_mat


def _one_pixel_axes(box: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``(B, 2)`` mask of the x and y axes along which an inclusive box is one pixel wide.

    Read from the box in its stored dtype: cast to a half-precision input's dtype first, neighbouring coordinates
    above 256 can round to one value in bfloat16 and make a two-pixel box look one pixel wide.
    """
    box = box.to(device)
    return (box[:, 2] - box[:, 0]) == 0


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
        and when none fits the fallback keeps an input whose width/height is within ``ratio`` whole, or else its full
        width (input narrower than ``min(ratio)``) or its full height (wider than ``max(ratio)``). So
        ``scale=(1.0, 1.0)`` keeps an input within ``ratio`` whole, except that a candidate rounded a pixel or two
        short on the longer side also fits (8x6 gives 7x6 in about 18% of draws, as in torchvision). Unlike
        torchvision, which centres the fallback crop, it is placed at a random position like any other crop.

        Both cropping modes use the configured interpolation and ``align_corners``. Slice mode ignores
        ``align_corners`` for ``resample="nearest"``. With bilinear or bicubic interpolation, each mode's image
        follows its ``transform_matrix``, and at ``align_corners=False`` the two modes give different images: slice
        mode resizes the crop on the half-pixel grid, ``x' = (x - x0 + 0.5) * W_out / W_crop - 0.5`` for a crop
        starting at column ``x0`` (likewise for ``y``), while resample mode maps the crop's corner pixel centres
        onto the output's. Nearest sampling in slice mode uses the half-pixel grid and PyTorch's
        ``nearest-exact`` interpolation. With corner alignment, an axis where the output is one pixel long samples
        the crop's first row or column, and one where the crop is one pixel long repeats it, as ``F.interpolate``
        does, in both modes. Both modes support
        :meth:`inverse`, which warps the output back onto the original canvas with the inverse of the mode's
        ``transform_matrix`` and cannot recover discarded information.

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
        if flags["cropping_mode"] == "slice" and (
            flags["resample"] == Resample.NEAREST
            or (not flags["align_corners"] and flags["resample"] in (Resample.BILINEAR, Resample.BICUBIC))
        ):
            return _half_pixel_resize_transform(params["src"].to(input), params["dst"].to(input)).expand(
                input.shape[0], -1, -1
            )
        if flags["cropping_mode"] in ("resample", "slice"):
            src, dst = params["src"].to(input), params["dst"].to(input)
            transform: torch.Tensor = get_perspective_transform(src, dst)
            # A one-pixel crop or output axis makes the box vertices collinear, and the solve returns NaN or a
            # singular matrix. Corner-aligned resizing samples the crop's first pixel along that axis, which the
            # extent matrix encodes; every other box keeps the solved matrix bit for bit.
            one_pixel = (
                _one_pixel_axes(params["src"], input.device) | _one_pixel_axes(params["dst"], input.device)
            ).any(-1)
            transform = torch.where(one_pixel[:, None, None], _corner_aligned_resize_transform(src, dst), transform)
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

            output = crop_by_transform_mat(
                input,
                transform[:, :2, :],
                flags["size"],
                mode=flags["resample"].name.lower(),
                padding_mode="zeros",
                align_corners=flags["align_corners"],
            )
            # A one-pixel crop axis has no scale to warp with: the matrix puts its pixel at output index 0, and the
            # rest of that axis repeats it, as slice mode's interpolation does.
            one_pixel = _one_pixel_axes(params["src"], input.device)
            output = torch.where(one_pixel[:, 0, None, None, None], output[..., :1].expand_as(output), output)
            return torch.where(one_pixel[:, 1, None, None, None], output[..., :1, :].expand_as(output), output)
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            mode = "nearest-exact" if flags["resample"] == Resample.NEAREST else flags["resample"].name.lower()
            # ``interpolate`` rejects ``align_corners`` for nearest resampling.
            align_corners = None if mode in ("nearest", "nearest-exact") else flags["align_corners"]
            if is_compiling():
                return _compiled_slice_resize(input, params["src"], flags["size"], mode, align_corners)
            return crop_by_indices(
                input,
                params["src"],
                flags["size"],
                interpolation=mode,
                align_corners=align_corners,
            )
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def inverse_inputs(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        output = super().inverse_inputs(input, params, flags, transform, **kwargs)
        # A one-pixel crop axis is a translation in the matrix, so the inverse warp also lays the repeated pixel
        # onto the canvas pixels after it. With zero padding, everything outside the crop is zero.
        if kwargs.get("padding_mode", flags["padding_mode"]) != "zeros" or output.shape[0] == 0:
            return output
        one_pixel = _one_pixel_axes(params["src"], output.device) & self._gate(params).to(output.device)[:, None]
        # Integer pixel indices: in bfloat16 neighbouring indices above 256 round to the same value.
        start = params["src"][:, 0].to(output.device).round().long()
        columns = torch.arange(output.shape[-1], device=output.device)
        rows = torch.arange(output.shape[-2], device=output.device)
        keep_columns = ~one_pixel[:, :1] | (columns == start[:, :1])
        keep_rows = ~one_pixel[:, 1:] | (rows == start[:, 1:])
        keep = keep_rows[:, None, :, None] & keep_columns[:, None, None, :]
        return torch.where(keep, output, torch.zeros_like(output))

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not isinstance(size, tuple):
            raise TypeError(f"Expected the size be a tuple. Got {type(size)}")

        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

        return _crop_inverse_by_transform_mat(
            input,
            transform,
            size,
            flags["resample"].name.lower(),
            flags["padding_mode"],
            flags["align_corners"],
        )
