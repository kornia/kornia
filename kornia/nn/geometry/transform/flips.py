import torch
import torch.nn as nn

import kornia

__all__ = [
    "Vflip",
    "Hflip",
    "Rot180"
]


class Vflip(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.geometry.vflip(input)

    def __repr__(self):
        return self.__class__.__name__


class Hflip(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.geometry.hflip(input)

    def __repr__(self):
        return self.__class__.__name__


class Rot180(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.geometry.rot180(input)

    def __repr__(self):
        return self.__class__.__name__
