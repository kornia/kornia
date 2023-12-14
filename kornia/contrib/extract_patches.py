import math
from typing import Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from kornia.core import Module, Tensor, pad


def compute_padding(
    original_size: Union[int, Tuple[int, int]], window_size: Union[int, Tuple[int, int]]
) -> Tuple[int, int]:
    r"""Compute required padding to ensure chaining of :func:`extract_tensor_patches` and
    :func:`combine_tensor_patches` produces expected result.

    Args:
        original_size: the size of the original tensor.
        window_size: the size of the sliding window used while extracting patches.

    Return:
        The required padding for `(vertical, horizontal)` as a tuple of 2 ints.

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

    if original_size[0] % window_size[0] != 0:
        padh = math.ceil((window_size[0] - original_size[0] % window_size[0]) / 2)
    else:
        padh = 0
    if original_size[1] % window_size[1] != 0:
        padw = math.ceil((window_size[1] - original_size[1] % window_size[1]) / 2)
    else:
        padw = 0

    return (padh, padw)


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
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
    ) -> None:
        super().__init__()
        self.window_size: Tuple[int, int] = _pair(window_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.padding: Tuple[int, int] = _pair(padding)

    def forward(self, input: Tensor) -> Tensor:
        return extract_tensor_patches(input, self.window_size, stride=self.stride, padding=self.padding)


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
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`unpadding` is the amount of padding to be removed. This value
      must be the same as padding used while extracting tensor patches.

    The parameters :attr:`original_size`, :attr:`window_size`, :attr:`stride`, and :attr:`unpadding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    Args:
        patches: patched tensor.
        original_size: the size of the original tensor and the output patch size.
        window_size: the size of the sliding window used.
        stride: stride of the sliding window.
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
        stride: Union[int, Tuple[int, int]],
        unpadding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        super().__init__()
        self.original_size: Tuple[int, int] = _pair(original_size)
        self.window_size: Tuple[int, int] = _pair(window_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.unpadding: Tuple[int, int] = _pair(unpadding)

    def forward(self, input: Tensor) -> Tensor:
        return combine_tensor_patches(
            input, self.original_size, self.window_size, stride=self.stride, unpadding=self.unpadding
        )


def combine_tensor_patches(
    patches: Tensor,
    original_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    unpadding: Union[int, Tuple[int, int]] = 0,
    eps: float = 1e-8,
) -> Tensor:
    r"""Restore input from patches.

    See :class:`~kornia.contrib.CombineTensorPatches` for details.

    Args:
        patches: patched tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.
        original_size: the size of the original tensor and the output patch size.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window.
        unpadding: remove the padding added to both side of the input.

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
    unpadding = cast(Tuple[int, int], _pair(unpadding))

    if len(unpadding) != 2:
        raise AssertionError("Unpadding must be either an int or a tuple of two ints")

    ones = torch.ones(
        patches.shape[0],
        patches.shape[2],
        original_size[0],
        original_size[1],
        device=patches.device,
        dtype=patches.dtype,
    )
    patches = patches.permute(0, 2, 3, 4, 1)
    patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
    int_flag = 0
    if not torch.is_floating_point(patches):
        int_flag = 1
        dtype = patches.dtype
        patches = patches.float()
        ones = ones.float()

    # Calculate normalization map
    unfold_ones = F.unfold(ones, kernel_size=window_size, stride=stride, padding=unpadding)
    norm_map = F.fold(
        input=unfold_ones, output_size=original_size, kernel_size=window_size, stride=stride, padding=unpadding
    )

    # Restored tensor
    saturated_restored_tensor = F.fold(
        input=patches, output_size=original_size, kernel_size=window_size, stride=stride, padding=unpadding
    )

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
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    r"""Function that extract patches from tensors and stack them.

    See :class:`~kornia.contrib.ExtractTensorPatches` for details.

    Args:
        input: tensor image where to extract the patches with shape :math:`(B, C, H, W)`.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: Zero-padding added to both side of the input.

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

    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))

    if (stride[0] > window_size[0]) | (stride[1] > window_size[1]):
        raise AssertionError(f"Stride={stride} should be less than or equal to Window size={window_size}")

    if padding:
        padding = cast(Tuple[int, int], _pair(padding))

        if len(padding) != 2:
            raise AssertionError("Padding must be either an int or a tuple of two ints")

        pad_vert = _pair(padding[0])
        pad_horz = _pair(padding[1])
        padding4 = cast(Tuple[int, int, int, int], pad_horz + pad_vert)
        input = pad(input, padding4)

    return _extract_tensor_patchesnd(input, window_size, stride)
