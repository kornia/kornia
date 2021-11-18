import torch
import torch.nn as nn


def conv_distance_transform(
    input: torch.Tensor,
    kernel_size: int = 7
):
    pass


class ConvDistanceTransform(nn.Module):
    r"""Module that approximates the Manhattan (city block) distance transform of binary images with iterative convolutions.

    Args:
        input: input image tensor with shape :math:`(B,1,H,W)`.
        kernel_size: size of the convolution kernel.

    Returns:
        tensor with shape :math:`(B,1,H,W)`.

    """

    def __init__(
        self,
        kernel_size: int = 7
    ):
        super().__init__()
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass
