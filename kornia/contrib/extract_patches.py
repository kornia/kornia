from math import ceil
from typing import Optional, Tuple, Union, cast
from warnings import warn

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from kornia.core import Module, Tensor, pad

FullPadType = Tuple[int, int, int, int]
TuplePadType = Union[Tuple[int, int], FullPadType]
PadType = Union[int, TuplePadType]


def create_padding_tuple(padding: PadType, unpadding: bool = False) -> FullPadType:
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
    padding = cast(FullPadType, pad_horz + pad_vert)

    return padding


def compute_padding(
    original_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
) -> FullPadType:
    r"""Compute required padding to ensure chaining of :func:`extract_tensor_patches` and
    :func:`combine_tensor_patches` produces expected result.

    Args:
        original_size: the size of the original tensor.
        window_size: the size of the sliding window used while extracting patches.
        stride: The stride of the sliding window. Optional: if not specified, window_size will be used.

    Return:
        The required padding as a tuple of four ints: (top, bottom, left, right)

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
        `allow_auto_(un)padding` is set to True.
    """
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
    if remainder_vertical != 0:
        vertical_padding = stride[0] - remainder_vertical
    else:
        vertical_padding = 0

    if remainder_horizontal != 0:
        horizontal_padding = stride[1] - remainder_horizontal
    else:
        horizontal_padding = 0

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
    padding = int(top_padding), int(bottom_padding), int(left_padding), int(right_padding)
    return cast(FullPadType, padding)


class ExtractTensorPatches(Module):
    r"""Module that extract patches from tensors and stack them.

    In the simplest case, the output value of the operator with input size
    :math:`(B, C, H, W)` is :math:`(B, N, C, H_{out}, W_{out})`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches stacked in
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the input height and width of the input in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote to denote to the patch size
        defined in the function signature.
        left-right and top-bottom order.

    * :attr:`window_size` is the size of the sliding window and controls the
      shape of the output tensor and defines the shape of the output patch.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`padding` controls the amount of implicit zeros-paddings on both
      sizes at each dimension.
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

    Args:
        input: tensor image where to extract the patches with shape :math:`(B, C, H, W)`.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: Zero-padding added to both side of the input.
        allow_auto_adding: whether to allow automatic padding if the window and stride do not fit into the image.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, N, C, H_{out}, W_{out})`

    Returns:
        the tensor with the extracted patches.

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

    def forward(self, input: Tensor) -> Tensor:
        return extract_tensor_patches(
            input,
            self.window_size,
            stride=self.stride,
            padding=self.padding,
            allow_auto_padding=self.allow_auto_padding,
        )


class CombineTensorPatches(Module):
    r"""Module that combines patches back into full tensors.

    In the simplest case, the output value of the operator with input size
    :math:`(B, N, C, H_{out}, W_{out})` is :math:`(B, C, H, W)`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches stacked in
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the input height and width of the input in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote to denote to the patch size
        defined in the function signature.
        left-right and top-bottom order.


    * :attr:`original_size` is the size of the original image prior to
      extracting tensor patches and defines the shape of the output patch.
    * :attr:`window_size` is the size of the sliding window used while
      extracting tensor patches.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`unpadding` is the amount of padding to be removed. If specified,
      this value must be the same as padding used while extracting tensor patches.
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

    Args:
        patches: patched tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.
        original_size: the size of the original tensor and the output size.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window.
        unpadding: remove the padding added to both side of the input.
        allow_auto_unpadding: whether to allow automatic unpadding of the input
            if the window and stride do not fit into the original_size.
        eps: small value used to prevent division by zero.

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

    def forward(self, input: Tensor) -> Tensor:
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
    if (remainder_horizontal != 0) or (remainder_vertical != 0):
        # needs padding to fit
        return False
    # we can fit a full number of patches in, based on the stride
    return True


def combine_tensor_patches(
    patches: Tensor,
    original_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    allow_auto_unpadding: bool = False,
    unpadding: PadType = 0,
    eps: float = 1e-8,
) -> Tensor:
    r"""Restore input from patches.

    See :class:`~kornia.contrib.CombineTensorPatches` for details.

    Args:
        patches: patched tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.
        original_size: the size of the original tensor and the output size.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window.
        unpadding: remove the padding added to both side of the input.
        allow_auto_unpadding: whether to allow automatic unpadding of the input
            if the window and stride do not fit into the original_size.
        eps: small value used to prevent division by zero.

    Return:
        The combined patches in an image tensor with shape :math:`(B, C, H, W)`.

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

    if not unpadding:
        # if padding is specified, we leave it up to the user to ensure it fits
        # otherwise we check here if it will fit and offer to calculate padding
        if not _check_patch_fit(original_size, window_size, stride):
            if not allow_auto_unpadding:
                warn(
                    f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\n"
                    f"Image size: {original_size}\n"
                    "This means we probably cannot correctly recombine patches. By enabling `allow_auto_unpadding`, "
                    "the input will be unpadded to fit the window and stride.\n"
                    "If the patches have been obtained through `extract_tensor_patches` with the correct padding or "
                    "the argument `allow_auto_padding`, this will result in a correct reconstruction."
                )
            else:
                unpadding = compute_padding(original_size=original_size, window_size=window_size, stride=stride)
                # TODO: Can't we just do actual size minus original size to get padding?

    if unpadding:
        unpadding = create_padding_tuple(unpadding)
        unpadding = cast(FullPadType, unpadding)

    ones = torch.ones(
        patches.shape[0],
        patches.shape[2],
        original_size[0],
        original_size[1],
        device=patches.device,
        dtype=patches.dtype,
    )

    if unpadding:
        ones = pad(ones, pad=unpadding)
    restored_size = ones.shape[2:]

    patches = patches.permute(0, 2, 3, 4, 1)
    patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
    int_flag = 0
    if not torch.is_floating_point(patches):
        int_flag = 1
        dtype = patches.dtype
        patches = patches.float()
        ones = ones.float()

    # Calculate normalization map
    unfold_ones = F.unfold(ones, kernel_size=window_size, stride=stride)
    norm_map = F.fold(input=unfold_ones, output_size=restored_size, kernel_size=window_size, stride=stride)
    if unpadding:
        norm_map = pad(norm_map, [-i for i in unpadding])

    # Restored tensor
    saturated_restored_tensor = F.fold(input=patches, output_size=restored_size, kernel_size=window_size, stride=stride)
    if unpadding:
        saturated_restored_tensor = pad(saturated_restored_tensor, [-i for i in unpadding])

    # Remove satuation effect due to multiple summations
    restored_tensor = saturated_restored_tensor / (norm_map + eps)
    if int_flag:
        restored_tensor = restored_tensor.to(dtype)
    return restored_tensor


def _extract_tensor_patchesnd(input: Tensor, window_sizes: Tuple[int, ...], strides: Tuple[int, ...]) -> Tensor:
    batch_size, num_channels = input.size()[:2]
    dims = range(2, input.dim())
    for dim, patch_size, stride in zip(dims, window_sizes, strides):
        input = input.unfold(dim, patch_size, stride)
    input = input.permute(0, *dims, 1, *(dim + len(dims) for dim in dims)).contiguous()
    return input.view(batch_size, -1, num_channels, *window_sizes)


def extract_tensor_patches(
    input: Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: PadType = 0,
    allow_auto_padding: bool = False,
) -> Tensor:
    r"""Function that extract patches from tensors and stacks them.

    See :class:`~kornia.contrib.ExtractTensorPatches` for details.

    Args:
        input: tensor image where to extract the patches with shape :math:`(B, C, H, W)`.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: Zero-padding added to both side of the input.
        allow_auto_adding: whether to allow automatic padding if the window and stride do not fit into the image.

    Returns:
        the tensor with the extracted patches with shape :math:`(B, N, C, H_{out}, W_{out})`.

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
        raise TypeError(f"Input input type is not a Tensor. Got {type(input)}")

    if len(input.shape) != 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    # check if the window sliding over the image will fit into the image
    # torch's unfold drops the final patches that don't fit
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))
    original_size = (input.shape[-2], input.shape[-1])

    if not padding:
        # if padding is specified, we leave it up to the user to ensure it fits
        # otherwise we check here if it will fit and offer to calculate padding
        if not _check_patch_fit(original_size, window_size, stride):
            if not allow_auto_padding:
                warn(
                    f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\n"
                    f"Image size: {original_size}\n"
                    "This means that the final incomplete patches will be dropped. By enabling `allow_auto_padding`, "
                    "the input will be padded to fit the window and stride."
                )
            else:
                padding = compute_padding(original_size=original_size, window_size=window_size, stride=stride)

    if padding:
        padding = create_padding_tuple(padding)
        input = pad(input, padding)

    return _extract_tensor_patchesnd(input, window_size, stride)
