import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


class DepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}


    Shape:
        - Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples::

        >>> depth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = tgm.losses.DepthSmoothnessLoss()
        >>> loss = smooth(depth, image)
    """

    def __init__(self) -> None:
        super(DepthSmoothnessLoss, self).__init__()

    @staticmethod
    def gradient_x(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    @staticmethod
    def gradient_y(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def forward(self, depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(depth):
            raise TypeError("Input depth type is not a torch.Tensor. Got {}"
                            .format(type(depth)))
        if not torch.is_tensor(image):
            raise TypeError("Input image type is not a torch.Tensor. Got {}"
                            .format(type(image)))
        if not len(depth.shape) == 4:
            raise ValueError("Invalid depth shape, we expect BxCxHxW. Got: {}"
                             .format(depth.shape))
        if not len(image.shape) == 4:
            raise ValueError("Invalid image shape, we expect BxCxHxW. Got: {}"
                             .format(image.shape))
        if not depth.shape[-2:] == image.shape[-2:]:
            raise ValueError("depth and image shapes must be the same. Got: {}"
                             .format(depth.shape, image.shape))
        if not depth.device == image.device:
            raise ValueError(
                "depth and image must be in the same device. Got: {}" .format(
                    depth.device, image.device))
        if not depth.dtype == image.dtype:
            raise ValueError(
                "depth and image must be in the same dtype. Got: {}" .format(
                    depth.dtype, image.dtype))
        # compute the gradients
        depth_dx: torch.Tensor = self.gradient_x(depth)
        depth_dy: torch.Tensor = self.gradient_y(depth)
        image_dx: torch.Tensor = self.gradient_x(image)
        image_dy: torch.Tensor = self.gradient_y(image)

        # compute image weights
        weights_x: torch.Tensor = torch.exp(
            -torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y: torch.Tensor = torch.exp(
            -torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        # apply image weights to depth
        smoothness_x: torch.Tensor = torch.abs(depth_dx * weights_x)
        smoothness_y: torch.Tensor = torch.abs(depth_dy * weights_y)
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)


######################
# functional interface
######################


def depth_smoothness_loss(
        depth: torch.Tensor,
        image: torch.Tensor) -> torch.Tensor:
    r"""Computes image-aware depth smoothness loss.

    See :class:`~torchgeometry.losses.DepthSmoothnessLoss` for details.
    """
    return DepthSmoothnessLoss()(depth, image)
