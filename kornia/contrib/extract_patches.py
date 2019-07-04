from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair


class ExtractTensorPatches(nn.Module):
    r"""Module that extract patches from tensors and stack them.

    Applies a 2D convolution over an input tensor to extract patches and stack
    them in the depth axis of the output tensor. The function applies a
    Depthwise Convolution by applying the same kernel for all the input planes.

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

    Arguments:
        window_size (Union[int, Tuple[int, int]]): the size of the convolving
          kernel and the output patch size.
        stride (Optional[Union[int, Tuple[int, int]]]): stride of the
          convolution. Default is 1.
        padding (Optional[Union[int, Tuple[int, int]]]): Zero-padding added to
          both side of the input. Default is 0.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, N, C, H_{out}, W_{out})`

    Returns:
        torch.Tensor: the tensor with the extracted patches.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = kornia.contrib.extract_tensor_patches(input, (2, 3))
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3.0000, 4.0000, 5.0000],
                  [6.0000, 7.0000, 8.0000]]]])
    """

    def __init__(
            self,
            window_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = 1,
            padding: Optional[Union[int, Tuple[int, int]]] = 0) -> None:
        super(ExtractTensorPatches, self).__init__()
        self.window_size: Tuple[int, int] = _pair(window_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.padding: Tuple[int, int] = _pair(padding)

        # create base kernel
        self.kernel: torch.Tensor = self.create_kernel(self.window_size)

    @staticmethod
    def create_kernel(
            window_size: Tuple[int, int]) -> torch.Tensor:
        r"""Creates a binary kernel to extract the patches. If the window size
        is HxW will create a (H*W)xHxW kernel.
        """
        window_range: int = window_size[0] * window_size[1]
        kernel: torch.Tensor = torch.zeros(window_range, window_range)
        for i in range(window_range):
            kernel[i, i] += 1.0
        return kernel.view(window_range, 1, window_size[0], window_size[1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes
        batch_size, channels, height, width = input.shape

        # run convolution 2d to extract patches
        kernel: torch.Tensor = self.kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(input.device).to(input.dtype)
        output_tmp: torch.Tensor = F.conv2d(
            input,
            kernel,
            stride=self.stride,
            padding=self.padding,
            groups=channels)

        # reshape the output tensor
        output: torch.Tensor = output_tmp.view(
            batch_size, channels, self.window_size[0], self.window_size[1], -1)
        return output.permute(0, 4, 1, 2, 3)  # BxNxCxhxw


######################
# functional interface
######################


def extract_tensor_patches(
        input: torch.Tensor,
        window_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0) -> torch.Tensor:
    r"""Function that extract patches from tensors and stack them.

    See :class:`~kornia.contrib.ExtractTensorPatches` for details.
    """
    return ExtractTensorPatches(window_size, stride, padding)(input)
