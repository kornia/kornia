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

import warnings
from math import ceil
from typing import Optional, Tuple, Union, cast
from warnings import warn

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

from kornia.core._compat import torch_version_lt
from kornia.core.exceptions import ValueCheckError

FullPadType = Tuple[int, int, int, int]
TuplePadType = Union[Tuple[int, int], FullPadType]
PadType = Union[int, TuplePadType]


def _check_patch_params(window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]) -> None:
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))
    # Raised directly rather than through KORNIA_CHECK, which python -O and disable_checks() switch off: a zero
    # window_size would then return empty patches instead of failing.
    if any(size <= 0 for size in window_size):
        raise ValueCheckError(f"window_size must contain positive values. Got {window_size}.")
    if any(step <= 0 for step in stride):
        raise ValueCheckError(f"stride must contain positive values. Got {stride}.")


def create_padding_tuple(padding: PadType, unpadding: bool = False) -> FullPadType:
    """Create argument for padding op."""
    padding = cast(TuplePadType, _pair(padding))

    if len(padding) not in [2, 4]:
        raise AssertionError(
            f"{'Unpadding' if unpadding else 'Padding'} must be either an int, tuple of two ints or tuple of four ints"
        )

    if len(padding) == 2:
        pad_vert = _pair(padding[0])
        pad_horz = _pair(padding[1])
    else:
        pad_vert = padding[:2]
        pad_horz = padding[2:]

    return cast(FullPadType, pad_horz + pad_vert)


def compute_padding(
    original_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
) -> FullPadType:
    """Compute the ``(top, bottom, left, right)`` padding that makes sliding windows fit.

    ``padding = compute_padding(original_size, window_size, stride)`` returns four ints that, passed as ``padding`` to
    :func:`extract_tensor_patches` and as ``unpadding`` to :func:`combine_tensor_patches`, make windows of
    ``window_size`` at ``stride`` cover the padded image with none left over. Each padding is split evenly, with the odd
    pixel at the bottom or right, and an image smaller than one window is padded to one window.

    Args:
        original_size: the image size ``(H, W)``, or one int for both.
        window_size: the size of the sliding window used while extracting patches.
        stride: the stride of the sliding window. Default: ``window_size``.

    Return:
        The required padding as a tuple of four ints ``(top, bottom, left, right)``.

    Raises:
        ValueCheckError: if a ``window_size`` or ``stride`` component is below 1.

    See Also:
        - :func:`extract_tensor_patches`: takes this as ``padding``.
        - :func:`combine_tensor_patches`: takes this as ``unpadding``.

    Example:
        >>> image = torch.arange(12).view(1, 1, 4, 3)
        >>> padding = compute_padding((4,3), (3,3))
        >>> out = extract_tensor_patches(image, window_size=(3, 3), stride=(3, 3), padding=padding)
        >>> combine_tensor_patches(out, original_size=(4, 3), window_size=(3, 3), stride=(3, 3), unpadding=padding)
        tensor([[[[ 0,  1,  2],
                  [ 3,  4,  5],
                  [ 6,  7,  8],
                  [ 9, 10, 11]]]])

    """
    original_size = cast(Tuple[int, int], _pair(original_size))
    window_size = cast(Tuple[int, int], _pair(window_size))
    if stride is None:
        stride = window_size
    stride = cast(Tuple[int, int], _pair(stride))
    _check_patch_params(window_size, stride)

    remainder_vertical = (original_size[0] - window_size[0]) % stride[0]
    remainder_horizontal = (original_size[1] - window_size[1]) % stride[1]
    # it might be best to apply padding only to the far edges (right, bottom), so
    # that fewer patches are affected by the padding.
    # For now, just use the default padding
    vertical_padding = stride[0] - remainder_vertical if remainder_vertical != 0 else 0
    # Stride alignment alone can leave the padded image smaller than one window.
    vertical_padding = max(vertical_padding, window_size[0] - original_size[0])

    horizontal_padding = stride[1] - remainder_horizontal if remainder_horizontal != 0 else 0
    horizontal_padding = max(horizontal_padding, window_size[1] - original_size[1])

    if vertical_padding % 2 == 0:
        top_padding = bottom_padding = vertical_padding // 2
    else:
        top_padding = vertical_padding // 2
        bottom_padding = ceil(vertical_padding / 2)

    if horizontal_padding % 2 == 0:
        left_padding = right_padding = horizontal_padding // 2
    else:
        left_padding = horizontal_padding // 2
        right_padding = ceil(horizontal_padding / 2)
    # the new implementation with unfolding requires symmetric padding
    return int(top_padding), int(bottom_padding), int(left_padding), int(right_padding)


class ExtractTensorPatches(nn.Module):
    """Cut an image into sliding-window patches with stored settings (module form).

    ``extract = ExtractTensorPatches(window_size, stride, padding)`` stores its arguments, and
    ``patches = extract(image)`` is :func:`extract_tensor_patches` on a ``(B, C, H, W)`` image, returning
    ``(B, N, C, h, w)`` patches.

    Convention:
        See the Convention block on :func:`extract_tensor_patches`.

    Args:
        window_size: the size ``(h, w)`` of the sliding window and of each patch, or one int for both.
        stride: stride of the sliding window, an int or ``(sh, sw)``.
        padding: zero padding, or cropping where negative: an int, ``(ph, pw)`` or ``(top, bottom, left, right)``.
        allow_auto_padding: whether to pad the image as :func:`compute_padding` computes when the windows do not fit.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output: ``(B, N, C, h, w)``

    Raises:
        ValueCheckError: if a ``window_size`` or ``stride`` component is below 1; raised from the constructor.

    See Also:
        - :func:`extract_tensor_patches`: the one-call form.
        - :class:`CombineTensorPatches`: puts the patches back into an image.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = ExtractTensorPatches((2, 3))(input)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3., 4., 5.],
                  [6., 7., 8.]]]])

    """

    def __init__(
        self,
        window_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: PadType = 0,
        allow_auto_padding: bool = False,
    ) -> None:
        super().__init__()
        _check_patch_params(window_size, stride)
        self.window_size: Union[int, Tuple[int, int]] = window_size
        self.stride: Union[int, Tuple[int, int]] = stride
        self.padding: PadType = padding
        self.allow_auto_padding: bool = allow_auto_padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Extract the sliding-window patches of a ``(B, C, H, W)`` image.

        Args:
            input: the image, ``(B, C, H, W)``.

        Returns:
            The patches, ``(B, N, C, h, w)``, with ``N`` window positions and ``(h, w)`` the window size.

        """
        return extract_tensor_patches(
            input,
            self.window_size,
            stride=self.stride,
            padding=self.padding,
            allow_auto_padding=self.allow_auto_padding,
        )


class CombineTensorPatches(nn.Module):
    """Reassemble sliding-window patches into an image with stored settings (module form).

    ``combine = CombineTensorPatches(original_size, window_size, stride, unpadding)`` stores its arguments, and
    ``image = combine(patches)`` is :func:`combine_tensor_patches` on ``(B, N, C, h, w)`` patches, returning a
    ``(B, C, H, W)`` image. Unlike the function, ``stride`` defaults to ``window_size``.

    Convention:
        See the Convention block on :func:`combine_tensor_patches`.

    Args:
        original_size: the size ``(H, W)`` of the image the patches were extracted from, and of the output.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window. Default: ``window_size``.
        unpadding: the padding used while extracting the patches, removed from the output.
        allow_auto_unpadding: whether to remove the padding :func:`compute_padding` computes when the windows do not
            fit ``original_size``; pass it when the patches were extracted with ``allow_auto_padding=True``.

    Shape:
        - Input: ``(B, N, C, h, w)``
        - Output: ``(B, C, H, W)``

    Raises:
        ValueCheckError: if a ``window_size`` or ``stride`` component is below 1; raised from the constructor.

    See Also:
        - :func:`combine_tensor_patches`: the one-call form.
        - :class:`ExtractTensorPatches`: cuts the patches this module puts back.

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> CombineTensorPatches(original_size=(4, 4), window_size=(2, 2))(out)
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])

    """

    def __init__(
        self,
        original_size: Tuple[int, int],
        window_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        unpadding: PadType = 0,
        allow_auto_unpadding: bool = False,
    ) -> None:
        super().__init__()
        _check_patch_params(window_size, stride if stride is not None else window_size)
        self.original_size: Tuple[int, int] = original_size
        self.window_size: Union[int, Tuple[int, int]] = window_size
        self.stride: Union[int, Tuple[int, int]] = stride if stride is not None else window_size
        self.unpadding: PadType = unpadding
        self.allow_auto_unpadding: bool = allow_auto_unpadding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Reassemble ``(B, N, C, h, w)`` patches into a ``(B, C, H, W)`` image.

        Args:
            input: the patches, ``(B, N, C, h, w)``.

        Returns:
            The image, ``(B, C, H, W)`` with ``(H, W)`` equal to ``original_size``.

        """
        return combine_tensor_patches(
            input,
            self.original_size,
            self.window_size,
            stride=self.stride,
            unpadding=self.unpadding,
            allow_auto_unpadding=self.allow_auto_unpadding,
        )


def _check_patch_fit(original_size: Tuple[int, int], window_size: Tuple[int, int], stride: Tuple[int, int]) -> bool:
    remainder_vertical = (original_size[0] - window_size[0]) % stride[0]
    remainder_horizontal = (original_size[1] - window_size[1]) % stride[1]
    # the remainder takes into account half a window on each side,
    # the rest of the image is divided based on the stride, not the window
    # size
    # we can fit a full number of patches in, based on the stride, only if nothing remains
    return (
        original_size[0] >= window_size[0]
        and original_size[1] >= window_size[1]
        and remainder_horizontal == 0
        and remainder_vertical == 0
    )


def combine_tensor_patches(
    patches: torch.Tensor,
    original_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    allow_auto_unpadding: bool = False,
    unpadding: PadType = 0,
    eps: Optional[float] = None,
) -> torch.Tensor:
    """Reassemble sliding-window patches into an image, averaging where they overlap.

    ``image = combine_tensor_patches(patches, original_size, window_size, stride)`` takes ``(B, N, C, h, w)``
    patches and returns the ``(B, C, H, W)`` image of size ``original_size``. Pass the ``stride`` used for the
    extraction and its ``padding`` as ``unpadding``. When the extraction padding is non-negative and makes the windows
    fit, or both auto flags are set, a round trip returns the input: exactly for integer and ``bool`` patches, up to
    rounding where floating patches overlap.

    Convention:
        - Overlapping patches are averaged: each pixel is divided by the number of patches that cover it, where
          :func:`torch.nn.functional.fold` sums them. A pixel that no patch covers (patches that do not fit) is 0.
        - The output keeps the dtype of the patches. Integer patches are averaged exactly in ``int64``, with the mean
          truncated toward zero (on MPS before torch 2.9 they go through ``float32``, exact up to :math:`2^{24}`);
          ``bool`` patches are True where any covering patch is True; ``float16`` and ``bfloat16`` patches are summed
          in ``float32``.
        - ``unpadding`` and ``allow_auto_unpadding`` follow the ``padding`` and ``allow_auto_padding`` rules of
          :func:`extract_tensor_patches`; without either, patches that do not fit ``original_size`` give a warning.
        - This function takes ``allow_auto_unpadding`` before ``unpadding``.
        - An empty batch is reconstructed with the same spatial size and unpadding as a non-empty one.

    Args:
        patches: the patches, ``(B, N, C, h, w)``.
        original_size: the size ``(H, W)`` of the image the patches were extracted from, and of the output.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window. Required.
        allow_auto_unpadding: whether to remove the padding :func:`compute_padding` computes when the windows do not
            fit ``original_size``.
        unpadding: the padding used while extracting the patches, removed from the output.
        eps: deprecated, has no effect. Each covered pixel is divided by its exact overlap count and a pixel no patch
            covers is 0.

    Return:
        The combined patches in an image tensor with shape ``(B, C, H, W)``.

    Raises:
        ValueError: if ``patches`` is not 5-D.
        ValueCheckError: if a ``window_size`` or ``stride`` component is below 1.
        AssertionError: if ``stride`` is larger than ``window_size``, which leaves pixels uncovered.
        AssertionError: if ``unpadding`` is a tuple of other than two or four ints.
        RuntimeError: if the number or size of the patches does not match ``original_size``, ``window_size``, ``stride``
            and ``unpadding``.

    See Also:
        - :func:`extract_tensor_patches`: cuts the patches this function puts back.
        - :class:`CombineTensorPatches`: the module form.
        - :func:`compute_padding`: the ``unpadding`` that makes the windows fit.

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> combine_tensor_patches(out, original_size=(4, 4), window_size=(2, 2), stride=(2, 2))
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])

    """
    if patches.ndim != 5:
        raise ValueError(f"Invalid input shape, we expect BxNxCxHxW. Got: {patches.shape}")

    original_size = cast(Tuple[int, int], _pair(original_size))
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))
    _check_patch_params(window_size, stride)

    if (stride[0] > window_size[0]) | (stride[1] > window_size[1]):
        raise AssertionError(
            f"Stride={stride} should be less than or equal to Window size={window_size}, information is missing"
        )

    # if padding is specified, we leave it up to the user to ensure it fits
    # otherwise we check here if it will fit and offer to calculate padding
    unpadding_tuple = create_padding_tuple(unpadding, unpadding=True)
    if not any(unpadding_tuple) and not _check_patch_fit(original_size, window_size, stride):
        if not allow_auto_unpadding:
            warn(
                f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\n"
                f"Image size: {original_size}\n"
                "This means we probably cannot correctly recombine patches. By enabling `allow_auto_unpadding`, "
                "the input will be unpadded to fit the window and stride.\n"
                "If the patches have been obtained through `extract_tensor_patches` with the correct padding or "
                "the argument `allow_auto_padding`, this will result in a correct reconstruction.",
                stacklevel=1,
            )
        else:
            unpadding_tuple = create_padding_tuple(
                compute_padding(original_size=original_size, window_size=window_size, stride=stride)
            )
    has_unpadding = any(unpadding_tuple)

    if eps is not None:
        warnings.warn(
            "`eps` is deprecated and has no effect: each covered pixel is divided by its exact overlap count and a "
            "pixel no patch covers is 0. It will be removed in a future release; stop passing it.",
            DeprecationWarning,
            stacklevel=2,
        )
    out_dtype = patches.dtype
    is_integer = not patches.is_floating_point() and not patches.is_complex()
    if is_integer:
        # Checked before the dtype/device branches so that one message holds on every path.
        target = _patch_target_index(patches, original_size, window_size, stride, unpadding_tuple, has_unpadding)
        # MPS before torch 2.9 raises "index_add(): Expected non int64 dtype for source." there, so integer and bool
        # patches keep the float32 fold: exact up to 2**24, then cast back.
        if not (patches.device.type == "mps" and torch_version_lt(2, 9, 0)):
            return _combine_integer_patches(patches, target)

    # Fold half precision in float32: float16 overflows past 65504 and bfloat16 rounds the overlap sums. Complex
    # patches are folded as two real tensors: MPS has no complex col2im kernel.
    half = patches.dtype in (torch.float16, torch.bfloat16)
    work_dtype = patches.real.dtype if patches.is_complex() else patches.dtype
    if is_integer or half:
        work_dtype = torch.float32
    ones_tensor = torch.ones(
        patches.shape[0], patches.shape[2], original_size[0], original_size[1], device=patches.device, dtype=work_dtype
    )
    if has_unpadding:
        ones_tensor = F.pad(ones_tensor, list(unpadding_tuple))
    restored_size = ones_tensor.shape[2:]
    patches = patches.permute(0, 2, 3, 4, 1).flatten(1, 3)
    unfold_ones = F.unfold(ones_tensor, kernel_size=window_size, stride=stride)
    norm_map = F.fold(input=unfold_ones, output_size=restored_size, kernel_size=window_size, stride=stride)
    fold_kwargs = {"output_size": restored_size, "kernel_size": window_size, "stride": stride}
    if patches.is_complex():
        folded = [F.fold(input=part, **fold_kwargs) for part in (patches.real, patches.imag)]
    else:
        folded = [F.fold(input=patches.to(work_dtype), **fold_kwargs)]
    if has_unpadding:
        norm_map = F.pad(norm_map, [-i for i in unpadding_tuple])
        folded = [F.pad(part, [-i for i in unpadding_tuple]) for part in folded]
    # A covered pixel has an integer overlap count >= 1 and is divided by it exactly. A pixel no patch covers (patches
    # that do not fit, the warned path) has count 0 and sum 0, and stays 0. Complex parts are divided separately: a
    # complex division by the real count rounds on some torch versions.
    count = torch.where(norm_map == 0, torch.ones_like(norm_map), norm_map)
    restored = torch.complex(folded[0] / count, folded[1] / count) if patches.is_complex() else folded[0] / count
    return restored.to(out_dtype)


def _patch_target_index(
    patches: torch.Tensor,
    original_size: Tuple[int, int],
    window_size: Tuple[int, int],
    stride: Tuple[int, int],
    unpadding_tuple: FullPadType,
    has_unpadding: bool,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Tuple[int, int]]:
    """Return the target pixel of every patch element, the (left, right, top, bottom) crop and the padded size."""
    N = patches.shape[1]
    pad_l, pad_r, pad_t, pad_b = unpadding_tuple if has_unpadding else (0, 0, 0, 0)  # F.pad order, as the float path
    Hr, Wr = original_size[0] + pad_t + pad_b, original_size[1] + pad_l + pad_r
    index = torch.arange(Hr * Wr, device=patches.device).view(1, 1, Hr, Wr)
    index_patches = _extract_tensor_patchesnd(index, window_size, stride)
    if (N, *patches.shape[3:]) != (index_patches.shape[1], *index_patches.shape[3:]):
        raise RuntimeError(
            f"Given patches of shape {tuple(patches.shape)}, expected {index_patches.shape[1]} patches of window size "
            f"{window_size} for output_size={(Hr, Wr)} and stride={stride}."
        )
    return index_patches.flatten(), (pad_l, pad_r, pad_t, pad_b), (Hr, Wr)


def _combine_integer_patches(
    patches: torch.Tensor, target: Tuple[torch.Tensor, Tuple[int, int, int, int], Tuple[int, int]]
) -> torch.Tensor:
    # F.fold has no integer kernel ("col2im_out_cpu" not implemented for 'Long'), and a float sum rounds above 2**24
    # (float32) or 2**53 (float64). Scatter-add in int64 instead. The overlap mean truncates toward zero, as a float
    # mean cast back to an integer dtype does.
    flat_index, pads, (Hr, Wr) = target
    B, C = patches.shape[0], patches.shape[2]
    values = patches.to(torch.int64).permute(0, 2, 1, 3, 4).flatten(2)
    # The int64 cast wraps uint64 values from 2**63 up to negatives. Flipping the sign bit maps [0, 2**64) onto
    # [-2**63, 2**63) in order (v - 2**63), where the mean below is exact; the bit is flipped back on the result.
    unsigned64 = patches.dtype == torch.uint64
    if unsigned64:
        values = torch.bitwise_xor(values, -(2**63))
    zeros = torch.zeros(B, C, Hr * Wr, dtype=torch.int64, device=patches.device)
    if patches.dtype == torch.bool:
        # Bool patches are True wherever any covering patch is True.
        out = zeros.index_add(-1, flat_index, values) != 0
    else:
        # An overlap sum of int64 values can leave int64, so each value v is split by the overlap count c of its pixel
        # as v = q * c + r (q truncated, |r| < c). The q sum to at most max |v| in magnitude and the r to less than
        # c**2, so the mean sum(q) + sum(r) / c is exact for every int64 input.
        counts = torch.zeros(Hr * Wr, dtype=torch.int64, device=patches.device)
        counts = counts.index_add_(0, flat_index, torch.ones_like(flat_index))
        covered = counts > 0
        counts = counts.clamp_min(1)
        count = counts[flat_index]
        q = torch.div(values, count, rounding_mode="trunc")
        sum_r = zeros.index_add(-1, flat_index, values - q * count)
        out = zeros.index_add(-1, flat_index, q) + torch.div(sum_r, counts, rounding_mode="floor")
        if unsigned64:
            # A uint64 mean is non-negative, so its truncation is the floor, which the shift by 2**63 preserves. A
            # pixel no patch covers stays 0 rather than flipping to 2**63.
            out = torch.where(covered, torch.bitwise_xor(out, -(2**63)), torch.zeros_like(out))
        else:
            # The floor of the mean, moved toward zero where the mean is negative and not an integer.
            out = out + ((out < 0) & (torch.remainder(sum_r, counts) != 0)).to(torch.int64)
    # Negative unpadding pads the result with zeros, as the float path's inverse F.pad does.
    out = F.pad(out.view(B, C, Hr, Wr), [-p for p in pads])
    return out.to(patches.dtype)


def _extract_tensor_patchesnd(
    input: torch.Tensor, window_sizes: Tuple[int, ...], strides: Tuple[int, ...]
) -> torch.Tensor:
    dims = range(2, input.dim())
    for dim, patch_size, stride in zip(dims, window_sizes, strides):
        input = input.unfold(dim, patch_size, stride)
    input = input.permute(0, *dims, 1, *(dim + len(dims) for dim in dims)).contiguous()
    return input.flatten(1, len(dims))


def extract_tensor_patches(
    input: torch.Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: PadType = 0,
    allow_auto_padding: bool = False,
) -> torch.Tensor:
    """Cut an image into sliding-window patches stacked along a new patch dimension.

    ``patches = extract_tensor_patches(image, window_size, stride)`` takes a ``(B, C, H, W)`` image and returns its
    ``(B, N, C, h, w)`` patches, where ``(h, w)`` is ``window_size`` and ``N`` is the number of window positions.

    Convention:
        - Patches are stacked row-major over the grid of window positions, left to right and then top to bottom, the
          order of :func:`torch.nn.functional.unfold`.
        - ``window_size``, ``stride`` and a 2-tuple ``padding`` are ``(height, width)``, or one int for both. A positive
          ``padding`` adds zeros and a negative one crops. A 4-tuple is ``(top, bottom, left, right)``, not the
          ``(left, right, top, bottom)`` of :func:`torch.nn.functional.pad`; a 2-tuple ``(ph, pw)`` adds ``ph`` rows
          above and below and ``pw`` columns left and right.
        - A window that does not fit warns and drops the last rows or columns; an image smaller than the window warns
          and then raises. ``allow_auto_padding=True`` pads instead, as :func:`compute_padding` computes.
        - A non-zero ``padding`` switches off both the fit warning and ``allow_auto_padding``, so a padding too small
          for the window drops the last rows or columns silently. An all-zero ``padding`` counts as none.
        - A ``window_size`` or ``stride`` component below 1 raises ``ValueCheckError``, a ``ValueError``, up front
          (at construction for the module classes), here and in :func:`combine_tensor_patches`,
          :func:`compute_padding`, :class:`ExtractTensorPatches` and :class:`CombineTensorPatches`, also under
          ``python -O`` and ``disable_checks()``.
        - An empty batch gives an empty output with the same ``N``, ``C``, ``h`` and ``w`` as a non-empty one.

    Args:
        input: the image to extract the patches from, ``(B, C, H, W)`` with ``C`` at least 1.
        window_size: the size ``(h, w)`` of the sliding window and of each patch, or one int for both.
        stride: stride of the sliding window, an int or ``(sh, sw)``.
        padding: zero padding, or cropping where negative: an int, ``(ph, pw)`` or ``(top, bottom, left, right)``.
        allow_auto_padding: whether to pad the image as :func:`compute_padding` computes when the windows do not fit.

    Returns:
        The extracted patches, ``(B, N, C, h, w)``.

    Raises:
        TypeError: if ``input`` is not a tensor.
        ValueError: if ``input`` is not 4-D or has no channel.
        ValueCheckError: if a ``window_size`` or ``stride`` component is below 1.
        AssertionError: if ``padding`` is a tuple of other than two or four ints.
        RuntimeError: if the image, after any padding, is smaller than the window.

    See Also:
        - :class:`ExtractTensorPatches`: the module form.
        - :func:`combine_tensor_patches`: puts the patches back into an image.
        - :func:`compute_padding`: the ``padding`` that makes the windows fit.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = extract_tensor_patches(input, (2, 3))
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3., 4., 5.],
                  [6., 7., 8.]]]])

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) != 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    # check if the window sliding over the image will fit into the image
    # torch's unfold drops the final patches that don't fit
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))
    _check_patch_params(window_size, stride)
    if input.shape[1] == 0:
        raise ValueError("extract_tensor_patches requires a non-empty channel dimension.")
    original_size = (input.shape[-2], input.shape[-1])

    # if padding is specified, we leave it up to the user to ensure it fits
    # otherwise we check here if it will fit and offer to calculate padding
    padding_tuple = create_padding_tuple(padding)
    if not any(padding_tuple) and not _check_patch_fit(original_size, window_size, stride):
        if not allow_auto_padding:
            warn(
                f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\n"
                f"Image size: {original_size}\n"
                "This means that the final incomplete patches will be dropped. By enabling `allow_auto_padding`, "
                "the input will be padded to fit the window and stride.",
                stacklevel=1,
            )
        else:
            padding_tuple = create_padding_tuple(
                compute_padding(original_size=original_size, window_size=window_size, stride=stride)
            )

    if any(padding_tuple):
        input = F.pad(input, list(padding_tuple))

    return _extract_tensor_patchesnd(input, window_size, stride)
