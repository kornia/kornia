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

FullPadType = Tuple[int, int, int, int]
TuplePadType = Union[Tuple[int, int], FullPadType]
PadType = Union[int, TuplePadType]


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
    r"""Compute required padding to ensure chaining of :func:`extract_tensor_patches` and
    :func:`combine_tensor_patches` produces expected result.

    Args:
        original_size: the size of the original torch.Tensor.
        window_size: the size of the sliding window used while extracting patches.
        stride: The stride of the sliding window. Optional: if not specified, window_size will be used.

    Return:
        The required padding as a tuple of four ints: (top, bottom, left, right)

    Convention:
        See the Convention block of :class:`~kornia.contrib.ExtractTensorPatches`, which states where the padding
        goes.

    Example:
        >>> image = torch.arange(12).view(1, 1, 4, 3)
        >>> padding = compute_padding((4,3), (3,3))
        >>> out = extract_tensor_patches(image, window_size=(3, 3), stride=(3, 3), padding=padding)
        >>> combine_tensor_patches(out, original_size=(4, 3), window_size=(3, 3), stride=(3, 3), unpadding=padding)
        tensor([[[[ 0,  1,  2],
                  [ 3,  4,  5],
                  [ 6,  7,  8],
                  [ 9, 10, 11]]]])

    .. note::
        This function will be implicitly used in :func:`extract_tensor_patches` and :func:`combine_tensor_patches` if
        `allow_auto_(un)padding` is set to True and no non-zero `(un)padding` is given.

    """  # noqa: D205
    original_size = cast(Tuple[int, int], _pair(original_size))
    window_size = cast(Tuple[int, int], _pair(window_size))
    if stride is None:
        stride = window_size
    stride = cast(Tuple[int, int], _pair(stride))

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
    r"""nn.Module that extracts patches from tensors and stacks them.

    In the simplest case, the output value of the operator with input size
    :math:`(B, C, H, W)` is :math:`(B, N, C, H_{out}, W_{out})`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches.
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the input height and width of the input in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote the patch size
        defined in the function signature.

    An empty batch (:math:`B = 0`) gives an empty output with the same :math:`N`, :math:`C`, :math:`H_{out}` and
    :math:`W_{out}` as a non-empty one. :math:`C` must be non-zero and the window sizes positive.

    * :attr:`window_size` is the size of the sliding window and controls the
      shape of the output torch.Tensor and defines the shape of the output patch.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`padding` controls the amount of implicit zero padding on each side
      of each dimension.
    * :attr:`allow_auto_padding` allows automatic calculation of the padding required
      to fit the window and stride into the image.

    The parameters :attr:`window_size`, :attr:`stride` and :attr:`padding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    :attr:`padding` can also be a ``tuple`` of four ints -- in which case, the
    first two ints are for the height dimension while the last two ints are for
    the width dimension.

    Convention:
        - Patches are stacked row-major over the grid of window positions, left to right and then top to bottom, the
          order of :func:`torch.nn.functional.unfold`.
        - ``padding`` adds zeros. A 4-tuple is ``(top, bottom, left, right)``, not the ``(left, right, top, bottom)``
          of :func:`torch.nn.functional.pad`; a 2-tuple ``(ph, pw)`` adds ``ph`` rows above and below and ``pw``
          columns left and right.
        - A window that does not fit warns and drops the last rows or columns. ``allow_auto_padding=True`` pads
          instead, as :func:`compute_padding` computes: split evenly with the odd pixel at the bottom or right, and an
          image smaller than the window padded to one window.
        - A non-zero ``padding`` switches off both the fit warning and ``allow_auto_padding``, so a padding too small
          for the window drops the last rows or columns silently. An all-zero ``padding`` counts as none.
        - Known defect: a ``stride`` of 0 is not rejected up front and fails with an unrelated error such as
          ``ZeroDivisionError``, here and in :func:`combine_tensor_patches`
          (`#5959 <https://github.com/kornia/kornia/issues/5959>`_).

    Args:
        input: torch.Tensor image where to extract the patches with shape :math:`(B, C, H, W)`.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: zero padding: an int, ``(ph, pw)`` or ``(top, bottom, left, right)``.
        allow_auto_padding: whether to allow automatic padding if the window and stride do not fit into the image.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, N, C, H_{out}, W_{out})`

    Returns:
        the torch.Tensor with the extracted patches.

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

    def __init__(
        self,
        window_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: PadType = 0,
        allow_auto_padding: bool = False,
    ) -> None:
        super().__init__()
        self.window_size: Union[int, Tuple[int, int]] = window_size
        self.stride: Union[int, Tuple[int, int]] = stride
        self.padding: PadType = padding
        self.allow_auto_padding: bool = allow_auto_padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Extract sliding-window patches from a batched image tensor.

        Args:
            input: Input tensor with shape :math:`(B, C, H, W)`, where:
                - ``B`` is the batch size (number of images),
                - ``C`` is the number of channels,
                - ``H`` is height,
                - ``W`` is width.

        Returns:
            A tensor containing stacked patches with shape
            :math:`(B, N, C, H_{out}, W_{out})`, where:
            - ``N`` is the number of extracted windows per image,
            - ``H_{out}`` and ``W_{out}`` are patch height and patch width.
        """
        return extract_tensor_patches(
            input,
            self.window_size,
            stride=self.stride,
            padding=self.padding,
            allow_auto_padding=self.allow_auto_padding,
        )


class CombineTensorPatches(nn.Module):
    r"""nn.Module that combines patches back into full tensors.

    In the simplest case, the output value of the operator with input size
    :math:`(B, N, C, H_{out}, W_{out})` is :math:`(B, C, H, W)`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches.
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the height and width of the output image in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote the patch size
        defined in the function signature.

    An empty batch (:math:`B = 0`) is reconstructed with the same spatial size and unpadding as a non-empty one.

    * :attr:`original_size` is the size of the original image prior to
      extracting torch.Tensor patches, which is the size of the output image.
    * :attr:`window_size` is the size of the sliding window used while
      extracting torch.Tensor patches.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`unpadding` is the amount of padding to be removed. If specified,
      this value must be the same as padding used while extracting torch.Tensor patches.
    * :attr:`allow_auto_unpadding` allows automatic calculation of the padding required
      to fit the window and stride into the image. This must be used if the
      `allow_auto_padding` flag was used for extracting the patches.


    The parameters :attr:`original_size`, :attr:`window_size`, :attr:`stride`, and :attr:`unpadding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    :attr:`unpadding` can also be a ``tuple`` of four ints -- in which case, the
    first two ints are for the height dimension while the last two ints are for
    the width dimension.

    Convention:
        - Overlapping patches are averaged: each pixel is divided by the number of patches that cover it, where
          :func:`torch.nn.functional.fold` sums them. A pixel that no patch covers (patches that do not fit, which
          warns) is 0.
        - The output keeps the dtype of the patches. Integer patches are averaged exactly in ``int64`` (on MPS from
          torch 2.9), with the mean truncated toward zero; ``float16`` and ``bfloat16`` patches are summed in
          ``float32``.
        - A ``stride`` larger than ``window_size`` raises, because the patches leave pixels uncovered.
        - ``unpadding`` and ``allow_auto_unpadding`` follow the ``padding`` and ``allow_auto_padding`` rules of
          :class:`ExtractTensorPatches`; pass the values used to extract the patches.
        - ``stride`` defaults to ``window_size`` here and in :func:`compute_padding`, to 1 in
          :class:`ExtractTensorPatches` and :func:`extract_tensor_patches`, and has no default in
          :func:`combine_tensor_patches`, which also takes ``allow_auto_unpadding`` before ``unpadding``.

    Args:
        patches: patched torch.Tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.
        original_size: the size of the original torch.Tensor and the output size.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window. Default: ``window_size``.
        unpadding: the padding used while extracting the patches, removed from the output.
        allow_auto_unpadding: whether to allow automatic unpadding of the input
            if the window and stride do not fit into the original_size.

    Shape:
        - Input: :math:`(B, N, C, H_{out}, W_{out})`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> combine_tensor_patches(out, original_size=(4, 4), window_size=(2, 2), stride=(2, 2))
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])

    .. note::
        This function is supposed to be used in conjunction with :class:`ExtractTensorPatches`.

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
        self.original_size: Tuple[int, int] = original_size
        self.window_size: Union[int, Tuple[int, int]] = window_size
        self.stride: Union[int, Tuple[int, int]] = stride if stride is not None else window_size
        self.unpadding: PadType = unpadding
        self.allow_auto_unpadding: bool = allow_auto_unpadding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Reconstruct full images from extracted patches.

        Args:
            input: Patch tensor with shape :math:`(B, N, C, H_{out}, W_{out})`, where:
                - ``B`` is the batch size,
                - ``N`` is the number of patches,
                - ``C`` is the number of channels,
                - ``H_{out}`` and ``W_{out}`` are patch height and width.

        Returns:
            A reconstructed tensor with shape :math:`(B, C, H, W)`, where ``H``
            and ``W`` correspond to ``original_size`` after optional unpadding.
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
    r"""Restore input from patches.

    Convention:
        See the Convention block of :class:`~kornia.contrib.CombineTensorPatches`.

    Args:
        patches: patched torch.Tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.
        original_size: the size of the original torch.Tensor and the output size.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window. Required.
        unpadding: the padding used while extracting the patches, removed from the output.
        allow_auto_unpadding: whether to allow automatic unpadding of the input
            if the window and stride do not fit into the original_size.
        eps: deprecated, has no effect. Each covered pixel is divided by its exact overlap count and a pixel no patch
            covers is 0.

    Return:
        The combined patches in an image torch.Tensor with shape :math:`(B, C, H, W)`.

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> combine_tensor_patches(out, original_size=(4, 4), window_size=(2, 2), stride=(2, 2))
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])

    .. note::
        This function is supposed to be used in conjunction with :func:`extract_tensor_patches`.

    """
    if patches.ndim != 5:
        raise ValueError(f"Invalid input shape, we expect BxNxCxHxW. Got: {patches.shape}")

    original_size = cast(Tuple[int, int], _pair(original_size))
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))

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
    r"""Extract patches from tensors and stacks them.

    Convention:
        See the Convention block of :class:`~kornia.contrib.ExtractTensorPatches`.

    Args:
        input: torch.Tensor image where to extract the patches with shape :math:`(B, C, H, W)`.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: zero padding: an int, ``(ph, pw)`` or ``(top, bottom, left, right)``.
        allow_auto_padding: whether to allow automatic padding if the window and stride do not fit into the image.

    Returns:
        the torch.Tensor with the extracted patches with shape :math:`(B, N, C, H_{out}, W_{out})`.

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
    if input.shape[1] == 0:
        raise ValueError("extract_tensor_patches requires a non-empty channel dimension.")
    if any(size <= 0 for size in window_size):
        raise ValueError(f"window_size must contain positive values. Got {window_size}.")
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
