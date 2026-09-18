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


def _resize_coordinates(
    start: torch.Tensor,
    end: torch.Tensor,
    input_size: int,
    output_size: int,
    mode: str,
    align_corners: Optional[bool],
) -> torch.Tensor:
    length = end - start
    positions = torch.arange(output_size, device=start.device, dtype=start.dtype)
    if mode == "nearest":
        # ATen computes the scale on the host (usually float32; see the CPU-double path). CUDA/Inductor
        # tensor division can use reciprocal multiplication instead, changing pixels
        # at integer boundaries (e.g. 26 -> 22). Materialize possible scales as graph constants;
        # only the sampled length indexes this table at runtime. No float64 GPU/MPS ops.
        scales = torch.tensor([n / output_size for n in range(input_size + 1)], device=start.device, dtype=start.dtype)
        return (positions * scales[length.long()]).float().floor()
    if align_corners:
        return positions * ((length - 1) / (output_size - 1) if output_size > 1 else length * 0.0)
    return (positions + 0.5) * (length / output_size) - 0.5


def _cubic_weights(fraction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # PyTorch interpolate's cubic convolution, A=-0.75 (ATen/native/UpSample.h).
    def inner(x: torch.Tensor) -> torch.Tensor:
        return ((1.25 * x - 2.25) * x) * x + 1.0

    def outer(x: torch.Tensor) -> torch.Tensor:
        return ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0

    return outer(fraction + 1), inner(fraction), inner(1 - fraction), outer(2 - fraction)


def _compiled_slice_resize(
    input: torch.Tensor, src: torch.Tensor, size: Tuple[int, int], mode: str, align_corners: Optional[bool]
) -> torch.Tensor:
    """Slice and resize without ragged intermediate tensors or Python crop coordinates.

    Match ``interpolate`` pixel locations and clamp each interpolation tap to the *crop*
    boundary, including bicubic's outer taps. A warp of the entire image would read outside
    that boundary. Integer box conversion matches ``crop_by_indices`` and stops box gradients.
    """
    batch, channels, height, width = input.shape
    # Nearest normally uses float32 index arithmetic; handle ATen's CPU-double exception below.
    coordinate_dtype = torch.float64 if input.dtype == torch.float64 and mode != "nearest" else torch.float32
    src = src.to(device=input.device, dtype=torch.long)
    # Match Python slicing for negative and out-of-bounds replay coordinates.
    x0, x1 = src[:, 0, 0:1], src[:, 1, 0:1] + 1
    y0, y1 = src[:, 0, 1:2], src[:, 3, 1:2] + 1
    x0 = torch.where(x0 < 0, x0 + width, x0).clamp(0, width).to(coordinate_dtype)
    x1 = torch.where(x1 < 0, x1 + width, x1).clamp(0, width).to(coordinate_dtype)
    y0 = torch.where(y0 < 0, y0 + height, y0).clamp(0, height).to(coordinate_dtype)
    y1 = torch.where(y1 < 0, y1 + height, y1).clamp(0, height).to(coordinate_dtype)
    x = _resize_coordinates(x0, x1, width, size[1], mode, align_corners)
    y = _resize_coordinates(y0, y1, height, size[0], mode, align_corners)
    if mode == "nearest" and input.device.type == "cpu" and input.dtype == torch.float64 and sum(size) > 128:
        # ATen's generic CPU nearest kernel uses double scales for double images,
        # then floorf; its channels-last kernel uses float scales. Match the layout
        # of the actual slices (including crop_by_indices' identical-box batch path).
        xd = _resize_coordinates(x0.double(), x1.double(), width, size[1], mode, align_corners)
        yd = _resize_coordinates(y0.double(), y1.double(), height, size[0], mode, align_corners)
        if channels > 3:
            h, w = y1 - y0, x1 - x0
            sn, sc, sh, sw = input.stride()
            channels_last = (sc == 1) & ((w == 1) | (sw == channels)) & ((h == 1) | (sh == channels * w))
            if batch > 1:
                corners = src[:, [0, 1, 0, 3], [0, 0, 1, 1]]
                identical = (corners == corners[:1]).all()
                channels_last = channels_last & (~identical | (sn == channels * h * w))
            x, y = torch.where(channels_last, x, xd), torch.where(channels_last, y, yd)
        else:
            x, y = xd, yd
    # Accumulate interpolation-tap gradients in opmath precision before casting to
    # half/bfloat16, rather than rounding separately at each gather's backward.
    work = input if mode == "nearest" else input.to(coordinate_dtype)
    flat = work.reshape(batch, channels, height * width)

    def gather(x_index: torch.Tensor, y_index: torch.Tensor) -> torch.Tensor:
        x_index = (x0 + x_index.clamp(min=0)).minimum(x1 - 1).long()
        y_index = (y0 + y_index.clamp(min=0)).minimum(y1 - 1).long()
        indices = (y_index.unsqueeze(-1) * width + x_index.unsqueeze(-2)).reshape(batch, 1, -1)
        return flat.gather(2, indices.expand(-1, channels, -1)).reshape(batch, channels, *size)

    if mode == "nearest":
        return gather(x, y)

    if mode == "bilinear":
        x, y = x.clamp(min=0), y.clamp(min=0)
    ix, iy = x.floor(), y.floor()
    fx, fy = x - ix, y - iy
    if mode == "bilinear":
        wx, wy = fx[:, None, None, :], fy[:, None, :, None]
        top = gather(ix, iy) * (1 - wx) + gather(ix + 1, iy) * wx
        bottom = gather(ix, iy + 1) * (1 - wx) + gather(ix + 1, iy + 1) * wx
        return (top * (1 - wy) + bottom * wy).to(input.dtype)

    wxs, wys = _cubic_weights(fx), _cubic_weights(fy)
    result = input.new_zeros((batch, channels, *size), dtype=coordinate_dtype)
    for j in range(4):
        row = input.new_zeros((batch, channels, *size), dtype=coordinate_dtype)
        for i in range(4):
            row = row + gather(ix + i - 1, iy + j - 1) * wxs[i][:, None, None, :]
        result = result + row * wys[j][:, None, :, None]
    return result.to(input.dtype)


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
        See :class:`~kornia.augmentation.AugmentationBase2D` for input, dtype, probability, and replay,
        :class:`~kornia.augmentation.RigidAffineAugmentationBase2D` for transformation matrices, and
        :class:`~kornia.augmentation.GeometricAugmentationBase2D` for inverse behavior.
        ``size`` is an ``(height, width)`` tuple. A bare integer
        is rejected, unlike :class:`CenterCrop`; the sibling split is tracked in
        `#4417 <https://github.com/kornia/kornia/issues/4417>`_. Here ``p`` selects or skips the whole batch together.
        Within a selected batch, the generator tries ten candidate crops per image, sampling area fractions from
        ``scale`` and width/height ratios from ``ratio`` (shared with ``same_on_batch=True``). Rounded candidate
        dimensions must be positive and strictly smaller than the input on both axes. If no candidate fits,
        a fallback chooses dimensions by comparing input height/width with ``min(ratio)``, then clamps them to
        the input size. This fallback can violate both requested ranges: on an 8x6 input, ``scale=(1.0, 1.0)``
        with the default ratio produces a 4x6 crop, with half the input area and width/height ratio 1.5.
        The selected crop is resized to the requested output size.

        Slice mode calls index cropping with the configured interpolation and ``align_corners``; resample mode
        calls ``crop_by_transform_mat`` with zero padding. Both default to bilinear sampling and
        ``align_corners=True``. Under ``torch.compile``, slice mode uses tensor indexing and
        interpolation so newly sampled crop coordinates do not trigger recompilation.
        Only resample mode supports :meth:`inverse`; its inverse
        resamples onto the original canvas and cannot recover information discarded by cropping or interpolation.

    Example:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.tensor([[[0., 1., 2.],
        ...                         [3., 4., 5.],
        ...                         [6., 7., 8.]]])
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[1.0000, 1.5000, 2.0000],
                  [4.0000, 4.5000, 5.0000],
                  [7.0000, 7.5000, 8.0000]]]])
        >>> aug.inverse(out, padding_mode="border")
        tensor([[[[1., 1., 2.],
                  [4., 4., 5.],
                  [7., 7., 8.]]]])

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
            transform = transform.expand(input.shape[0], -1, -1)
            return transform
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
