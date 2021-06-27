import torch
import torch.nn as nn


class Vflip(nn.Module):
    r"""Vertically flip a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The vertically flipped image tensor.

    Examples:
        >>> vflip = Vflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> vflip(input)
        tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return vflip(input)

    def __repr__(self):
        return self.__class__.__name__


class Hflip(nn.Module):
    r"""Horizontally flip a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The horizontally flipped image tensor.

    Examples:
        >>> hflip = Hflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> hflip(input)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]])
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return hflip(input)

    def __repr__(self):
        return self.__class__.__name__


class Rot180(nn.Module):
    r"""Rotate a tensor image or a batch of tensor images 180 degrees.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Examples:
        >>> rot180 = Rot180()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> rot180(input)
        tensor([[[[1., 1., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rot180(input)

    def __repr__(self):
        return self.__class__.__name__


def rot180(input: torch.Tensor) -> torch.Tensor:
    r"""Rotate a tensor image or a batch of tensor images 180 degrees.

    .. image:: _static/img/rot180.png

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The rotated image tensor.

    """

    return torch.flip(input, [-2, -1])


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Horizontally flip a tensor image or a batch of tensor images.

    .. image:: _static/img/hflip.png

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The horizontally flipped image tensor.

    """
    w = input.shape[-1]
    return input[..., torch.arange(w - 1, -1, -1, device=input.device)]


def vflip(input: torch.Tensor) -> torch.Tensor:
    r"""Vertically flip a tensor image or a batch of tensor images.

    .. image:: _static/img/vflip.png

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The vertically flipped image tensor.

    """

    h = input.shape[-2]
    return input[..., torch.arange(h - 1, -1, -1, device=input.device), :]
