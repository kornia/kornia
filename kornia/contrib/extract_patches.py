from typing import Optional, Tuple, Union, cast
from warnings import warn

import torch
from torch.nn.modules.utils import _pair

from kornia.core import Module, Tensor, concatenate, pad

PadType = Union[Tuple[int, int], Tuple[int, int, int, int]]


def compute_padding(
    original_size: Union[int, Tuple[int, int]], window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]]
) -> Tuple[int, int, int, int]:
    r"""Compute required padding to ensure chaining of :func:`extract_tensor_patches` and
    :func:`combine_tensor_patches` produces expected result.

    Args:
        original_size: the size of the original tensor.
        window_size: the size of the sliding window used while extracting patches.
        stride: The stride of the sliding window.

    Return:
        The required symmetric padding tuple (`vertical`, `horizontal`) as a tuple of 2 ints.

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
        This function is supposed to be used in conjunction with :func:`extract_tensor_patches`
        and :func:`combine_tensor_patches`.
    """
    original_size = cast(Tuple[int, int], _pair(original_size))
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))

    remainder_vertical = (original_size[0] - window_size[0] // 2) % stride[0]
    remainder_horizontal = (original_size[1] - window_size[1] // 2) % stride[1]
    # it might be best to apply padding only to the far edges (right, bottom), so
    # that fewer patches are affected by the padding.
    # For now, just use the default padding
    if remainder_vertical != (window_size[0] // 2):
        vertical_padding = window_size[0] // 2 - remainder_vertical
        if vertical_padding < 0:
            vertical_padding = (stride[0] - remainder_vertical) + window_size[0] // 2
    else:
        vertical_padding = 0

    if remainder_horizontal != (window_size[1] // 2):
        horizontal_padding = window_size[1] // 2 - remainder_horizontal  # floor division might drop one pixel
        if horizontal_padding < 0:
            horizontal_padding = (stride[1] - remainder_horizontal) + window_size[1] // 2
    else:
        horizontal_padding = 0

    padding = (vertical_padding // 2, horizontal_padding // 2)  # symmetric padding

    return padding


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
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: Zero-padding added to both side of the input.

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
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, PadType]] = 0,
        allow_auto_padding: bool = False,
    ) -> None:
        super().__init__()
        self.window_size: Tuple[int, int] = _pair(window_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.padding: PadType = _pair(padding)
        self.allow_auto_padding = allow_auto_padding

    def forward(self, input: Tensor) -> Tensor:
        return extract_tensor_patches(
            input, self.window_size, stride=self.stride,
            padding=self.padding, allow_auto_padding=self.allow_auto_padding
        )


class CombineTensorPatches(Module):
    r"""Module that combine patches from tensors.

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
    * :attr:`unpadding` is the amount of padding to be removed. This value
      must be the same as padding used while extracting tensor patches.

    The parameters :attr:`original_size`, :attr:`window_size`, and :attr:`unpadding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    :attr:`unpadding` can also be a ``tuple`` of four ints -- in which case, the
    first two ints are for the height dimension while the last two ints are for
    the width dimension.

    Args:
        patches: patched tensor.
        original_size: the size of the original tensor and the output patch size.
        window_size: the size of the sliding window used.
        unpadding: remove the padding added to both side of the input.

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
        original_size: Union[int, Tuple[int, int]],
        window_size: Union[int, Tuple[int, int]],
        unpadding: Union[int, PadType] = 0,
        allow_auto_unpadding: bool = False,
    ) -> None:
        super().__init__()
        self.original_size: Tuple[int, int] = _pair(original_size)
        self.window_size: Tuple[int, int] = _pair(window_size)
        self.unpadding: PadType = _pair(unpadding)
        self.allow_auto_unpadding: bool = allow_auto_unpadding

    def forward(self, input: Tensor) -> Tensor:
        return combine_tensor_patches(
            input, self.original_size, self.window_size, stride=self.window_size,
            unpadding=self.unpadding, allow_auto_unpadding=self.allow_auto_unpadding
        )


def _check_patch_fit(original_size, window_size, stride):
    remainder_vertical = (original_size[0] - window_size[0] // 2) % stride[0]
    remainder_horizontal = (original_size[1] - window_size[1] // 2) % stride[1]
    if (remainder_horizontal != (window_size[1] // 2)) or (remainder_vertical != (window_size[0] // 2)):
        # needs padding to fit
        # iif it's half, we can fit a full number of patches in, based on the stride
        return False


def combine_tensor_patches(
    patches: Tensor,
    original_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    unpadding: Union[int, PadType] = 0,
    allow_auto_unpadding: bool = False,
) -> Tensor:
    r"""Restore input from patches.

    See :class:`~kornia.contrib.CombineTensorPatches` for details.

    Args:
        patches: patched tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.
        original_size: the size of the original tensor and the output patch size.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window.
        unpadding: remove the padding added to both side of the input.
        allow_auto_unpadding: whether to allow automatic unpadding of the input if the window does not fit into the image.

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

    if len(patches.shape) != 5:
        raise ValueError(f"Invalid input shape, we expect BxNxCxHxW. Got: {patches.shape}")

    original_size = cast(Tuple[int, int], _pair(original_size))
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))

    if stride[0] != window_size[0] or stride[1] != window_size[1]:
        raise NotImplementedError(
            f"Only stride == window_size is supported. Got {stride} and {window_size}."
            "Please feel free to drop a PR to Kornia Github."
        )

    if not unpadding:
        # if padding is specified, we leave it up to the user to ensure it fits
        # otherwise we check here if it will fit and offer to calculate padding
        if not _check_patch_fit(original_size, window_size, stride):
            if not allow_auto_unpadding:
                warn(
                    f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\nImage size: {original_size}\n"
                    "This means that the final incomplete patches will be dropped. By enabling `allow_auto_padding`, the input will be padded to fit the window and stride."
                )
            else:
                unpadding = compute_padding(original_size=original_size, window_size=window_size, stride=stride)

    if unpadding:
        unpadding = cast(PadType, _pair(unpadding))

        if len(unpadding) not in [2, 4]:
            raise AssertionError("Unpadding must be either an int, tuple of two ints or tuple of four ints")

        if len(unpadding) == 2:
            pad_vert = _pair(unpadding[0])
            pad_horz = _pair(unpadding[1])
        else:
            pad_vert = unpadding[:2]
            pad_horz = unpadding[2:]
        unpadding = cast(Tuple[int, int, int, int], pad_horz + pad_vert)

        hpad_check = (original_size[0] + unpadding[2] + unpadding[3]) % window_size[0] == 0
        wpad_check = (original_size[1] + unpadding[0] + unpadding[1]) % window_size[1] == 0

        if not hpad_check or not wpad_check:
            raise NotImplementedError("Insufficient padding")

        vertical_patches = (original_size[0] + (unpadding[2] + unpadding[3])) // window_size[0]
        horizontal_patches = (original_size[1] + (unpadding[0] + unpadding[1])) // window_size[1]
    else:
        vertical_patches = original_size[0] // window_size[0]
        horizontal_patches = original_size[1] // window_size[1]

    patches_tensor = patches.view(-1, vertical_patches, horizontal_patches, *patches.shape[-3:])
    restored_tensor = concatenate(torch.chunk(patches_tensor, vertical_patches, 1), -2)
    restored_tensor = concatenate(torch.chunk(restored_tensor, horizontal_patches, 2), -1).squeeze(1).squeeze(1)
    print(f"Unpadding: {unpadding}")

    if unpadding:
        unpadding = cast(Tuple[int, int, int, int], unpadding)
        restored_tensor = pad(restored_tensor, [-i for i in unpadding])
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
    padding: Union[int, PadType] = 0,
    allow_auto_padding: bool = False,
) -> Tensor:
    r"""Function that extract patches from tensors and stack them.

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
    stride = _pair(stride)
    window_size = _pair(window_size)
    original_size = (input.shape[-2], input.shape[-1])
    
    if not padding:
        # if padding is specified, we leave it up to the user to ensure it fits
        # otherwise we check here if it will fit and offer to calculate padding
        if not _check_patch_fit(original_size, window_size, stride):
            if not allow_auto_padding:
                warn(
                    f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\nImage size: {original_size}\n"
                    "This means that the final incomplete patches will be dropped. By enabling `allow_auto_padding`, the input will be padded to fit the window and stride."
                )
            else:
                padding = compute_padding(original_size=original_size, window_size=window_size, stride=stride)

    if padding:
        padding = cast(PadType, _pair(padding))

        if len(padding) not in [2, 4]:
            raise AssertionError("Padding must be either an int, tuple of two ints or tuple of four ints")

        if len(padding) == 2:
            pad_vert = _pair(padding[0])
            pad_horz = _pair(padding[1])
        else:
            pad_vert = padding[:2]
            pad_horz = padding[2:]
        padding = cast(Tuple[int, int, int, int], pad_horz + pad_vert)
        input = pad(input, padding)

    return _extract_tensor_patchesnd(input, _pair(window_size), _pair(stride))
