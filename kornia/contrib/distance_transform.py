"""The convolutional distance transform was proposed in: Karam, C.; Sugimoto, K.; Hirakawa, K., "Fast Convolutional
Distance Transform," Signal Processing Letters (SPL), 2019 IEEE Journal.

The algorithm implemented in conv_distance_transform is from     Pham et al, "A Differentiable Convolutional Distance
Transform Layer for Improved Image Segmentation"     Pattern Recognition, 2021 Conference Proceedings
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_cdt_kernel(
    kernel_size: int,
) -> torch.Tensor:
    # Value of h derived from the parameters used by Pham et. al in their proposal of the algorithm.
    h = -0.35
    grid_range = torch.Tensor(range(kernel_size))

    gridx, gridy = torch.meshgrid(grid_range, grid_range)
    gridx = gridx - math.floor(kernel_size / 2)
    gridy = gridy - math.floor(kernel_size / 2)

    kernel = torch.hypot(gridx, gridy)
    kernel = torch.exp(kernel / h)

    # for BCHW tensors
    kernel = torch.unsqueeze(kernel, 0)
    kernel = torch.unsqueeze(kernel, 0)

    return kernel


def conv_distance_transform(
    input: torch.Tensor,
    kernel_size: int = 7
) -> torch.Tensor:
    r"""Approximates the Manhattan distance transform of images using convolutions.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the input image.
    The transformation is applied independently across the channel dimension of the inputs.


    Args:
        input: Image with shape :math:`(B,C,H,W)`.
        kernel_size: size of the convolution kernel. Larger kernels are more accurate but less numerically stable.

    Returns:
        tensor with shape :math:`(B,C,H,W)`.

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    device: torch.device = input.device

    n_iters = math.ceil(max(input.shape[2], input.shape[3]) / math.floor(kernel_size / 2))
    kernel = make_cdt_kernel(kernel_size)

    out = torch.zeros(input.shape, dtype=torch.float32, device=device)

    # It is possible to avoid cloning the input if boundary = input, but this would require modifying the input tensor.
    boundary = input.clone().to(torch.float32)
    kernel.to(device)

    # If input images have multiple channels, view the channels in the batch dimension to match kernel shape.
    if input.shape[1] > 1:
        batch_channel_view_shape = (input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3])
        out = out.view(*batch_channel_view_shape)
        boundary = boundary.view(*batch_channel_view_shape)

    for i in range(n_iters):
        cdt = F.conv2d(boundary, kernel, padding='same')
        cdt = -0.35 * torch.log(cdt)

        # We are calculating log(0) above.
        torch.nan_to_num(cdt, out=cdt, posinf=0.0)

        mask = cdt > 0
        if mask.sum() == 0:
            break

        offset = i * kernel_size / 2
        out[mask] += offset + cdt[mask]
        boundary[mask] = 1

    # View channels in the channel dimension, if they were added to batch dimension during transform.
    if input.shape[1] > 1:
        out = out.view(input.shape)

    return out


class ConvDistanceTransform(nn.Module):
    r"""Module that approximates the Manhattan (city block) distance transform of images using convolutions.

    Args:
        kernel_size: size of the convolution kernel. Larger kernels are more accurate but less numerically stable.

    """
    def __init__(
        self,
        kernel_size: int = 7
    ):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return conv_distance_transform(input, self.kernel_size)
