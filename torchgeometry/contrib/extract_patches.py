from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair


class ExtractTensorPatches(nn.Module):
    r"""Module that extract patches from tensors and stack them.
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
        self.eps: float = 1e-6

        # create base kernel
        self.kernel: torch.Tensor = self.create_kernel(self.window_size)

    @staticmethod
    def create_kernel(
            window_size: Tuple[int, int],
            eps: float = 1e-6) -> torch.Tensor:
        r"""Creates a binary kernel to extract the patches. If the window size
        is HxW will create a (H*W)xHxW kernel.
        """
        window_range: int = window_size[0] * window_size[1]
        kernel: torch.Tensor = torch.zeros(window_range, window_range) + eps
        for i in range(window_range):
            kernel[i, i] += 1.0
        return kernel.view(window_range, 1, window_size[0], window_size[1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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

    See :class:`~torchgeometry.contrib.ExtractTensorPatches` for details.
    """
    return ExtractTensorPatches(window_size, stride, padding)(input)
