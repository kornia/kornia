import torch
import torch.nn as nn

# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    assert len(img.shape) == 4, img.shape
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    assert len(img.shape) == 4, img.shape
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def inverse_depth_smoothness_loss(idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}

    Args:
        idepth: tensor with the inverse depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.

    Return:
        a scalar with the computed loss.

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> loss = inverse_depth_smoothness_loss(idepth, image)
    """
    if not isinstance(idepth, torch.Tensor):
        raise TypeError("Input idepth type is not a torch.Tensor. Got {}".format(type(idepth)))

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image type is not a torch.Tensor. Got {}".format(type(image)))

    if not len(idepth.shape) == 4:
        raise ValueError("Invalid idepth shape, we expect BxCxHxW. Got: {}".format(idepth.shape))

    if not len(image.shape) == 4:
        raise ValueError("Invalid image shape, we expect BxCxHxW. Got: {}".format(image.shape))

    if not idepth.shape[-2:] == image.shape[-2:]:
        raise ValueError("idepth and image shapes must be the same. Got: {} and {}".format(idepth.shape, image.shape))

    if not idepth.device == image.device:
        raise ValueError(
            "idepth and image must be in the same device. Got: {} and {}".format(idepth.device, image.device)
        )

    if not idepth.dtype == image.dtype:
        raise ValueError("idepth and image must be in the same dtype. Got: {} and {}".format(idepth.dtype, image.dtype))

    # compute the gradients
    idepth_dx: torch.Tensor = _gradient_x(idepth)
    idepth_dy: torch.Tensor = _gradient_y(idepth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)

    # compute image weights
    weights_x: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # apply image weights to depth
    smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
    smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


class InverseDepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}

    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = InverseDepthSmoothnessLoss()
        >>> loss = smooth(idepth, image)
    """

    def forward(self, idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return inverse_depth_smoothness_loss(idepth, image)
