import torch
import torch.nn as nn

import kornia

__all__ = [
    "Vflip",
    "Hflip",
    "Rot180",
    "rot180",
    "hflip",
    "vflip",
]


class Vflip(kornia.nn.geometry.Vflip):
    r"""Vertically flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The vertically flipped image tensor

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> kornia.vflip(input)
        tensor([[[0, 1, 1],
                 [0, 0, 0],
                 [0, 0, 0]]])
    """

    def __init__(self) -> None:

        super(Vflip, self).__init__()
        kornia.deprecation_warning("kornia.geometry.Vflip", "kornia.nn.geometry.Vflip")


class Hflip(kornia.nn.geometry.Hflip):
    r"""Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The horizontally flipped image tensor

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> kornia.hflip(input)
        tensor([[[0, 0, 0],
                 [0, 0, 0],
                 [1, 1, 0]]])
    """

    def __init__(self) -> None:

        super(Hflip, self).__init__()
        kornia.deprecation_warning("kornia.geometry.Hflip", "kornia.nn.geometry.Hflip")


class Rot180(kornia.nn.geometry.Rot180):
    r"""Rotate a tensor image or a batch of tensor images
        180 degrees. Input must be a tensor of shape (C, H, W)
        or a batch of tensors :math:`(*, C, H, W)`.

        Args:
            input (torch.Tensor): input tensor

        Examples:
            >>> input = torch.tensor([[[
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 1., 1.]]]])
            >>> kornia.rot180(input)
            tensor([[[1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]]])
        """

    def __init__(self) -> None:

        super(Rot180, self).__init__()
        kornia.deprecation_warning("kornia.geometry.Rot180", "kornia.nn.geometry.Rot180")


def rot180(input: torch.Tensor) -> torch.Tensor:
    r"""Rotate a tensor image or a batch of tensor images
    180 degrees. Input must be a tensor of shape (C, H, W)
    or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The rotated image tensor

    """

    return torch.flip(input, [-2, -1])


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The horizontally flipped image tensor

    """
    w = input.shape[-1]
    return input[..., torch.arange(w - 1, -1, -1, device=input.device)]


def vflip(input: torch.Tensor) -> torch.Tensor:
    r"""Vertically flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The vertically flipped image tensor

    """

    h = input.shape[-2]
    return input[..., torch.arange(h - 1, -1, -1, device=input.device), :]
