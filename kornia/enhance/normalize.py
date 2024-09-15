"""Module containing functionals for intensity normalisation."""

from typing import List, Tuple, Union

import torch

from kornia.core import ImageModule as Module
from kornia.core import Tensor

__all__ = ["normalize", "normalize_min_max", "denormalize", "Normalize", "Denormalize"]


class Normalize(Module):
    r"""Normalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Normalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Normalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = Normalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(
        self,
        mean: Union[Tensor, Tuple[float], List[float], float],
        std: Union[Tensor, Tuple[float], List[float], float],
    ) -> None:
        super().__init__()

        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean])

        if isinstance(std, (int, float)):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)[None]

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)[None]

        self.mean = mean
        self.std = std

    def forward(self, input: Tensor) -> Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def normalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    r"""Normalize an image/video tensor with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        data: Image tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Normalised tensor with same size as input :math:`(B, C, *)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = normalize(x, torch.tensor([0.0]), torch.tensor([255.]))
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = normalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """
    shape = data.shape

    if torch.onnx.is_in_onnx_export():
        if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
            raise ValueError("Only tensor is accepted when converting to ONNX.")
        if mean.shape[0] != 1 or std.shape[0] != 1:
            raise ValueError(
                "Batch dimension must be one for broadcasting when converting to ONNX."
                f"Try changing mean shape and std shape from ({mean.shape}, {std.shape}) to (1, C) or (1, C, 1, 1)."
            )
    else:
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=data.device, dtype=data.dtype)

        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=data.device, dtype=data.dtype)

        # Allow broadcast on channel dimension
        if mean.shape and mean.shape[0] != 1:
            if mean.shape[0] != data.shape[1] and mean.shape[:2] != data.shape[:2]:
                raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

        # Allow broadcast on channel dimension
        if std.shape and std.shape[0] != 1:
            if std.shape[0] != data.shape[1] and std.shape[:2] != data.shape[:2]:
                raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

        mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
        std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    mean = mean[..., None]
    std = std[..., None]

    out: Tensor = (data.view(shape[0], shape[1], -1) - mean) / std

    return out.view(shape)


class Denormalize(Module):
    r"""Denormalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Denormalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Denormalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = Denormalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """

    def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]) -> None:
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: Tensor) -> Tensor:
        return denormalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def denormalize(data: Tensor, mean: Union[Tensor, float], std: Union[Tensor, float]) -> Tensor:
    r"""Denormalize an image/video tensor with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        input: Image tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Denormalised tensor with same size as input :math:`(B, C, *)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = denormalize(x, 0.0, 255.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = denormalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """
    shape = data.shape

    if torch.onnx.is_in_onnx_export():
        if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
            raise ValueError("Only tensor is accepted when converting to ONNX.")
        if mean.shape[0] != 1 or std.shape[0] != 1:
            raise ValueError("Batch dimension must be one for broadcasting when converting to ONNX.")
    else:
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=data.device, dtype=data.dtype)

        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=data.device, dtype=data.dtype)

        # Allow broadcast on channel dimension
        if mean.shape and mean.shape[0] != 1:
            if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
                raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

        # Allow broadcast on channel dimension
        if std.shape and std.shape[0] != 1:
            if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
                raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

        mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
        std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    mean = mean[..., None]
    std = std[..., None]

    out: Tensor = (data.view(shape[0], shape[1], -1) * std) + mean

    return out.view(shape)


def normalize_min_max(x: Tensor, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-6) -> Tensor:
    r"""Normalise an image/video tensor by MinMax and re-scales the value between a range.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \frac{x_i - \text{min}(x)}{\text{max}(x) - \text{min}(x)} + a

    where :math:`a` is :math:`\text{min_val}` and :math:`b` is :math:`\text{max_val}`.

    Args:
        x: The image tensor to be normalised with shape :math:`(B, C, *)`.
        min_val: The minimum value for the new range.
        max_val: The maximum value for the new range.
        eps: Float number to avoid zero division.

    Returns:
        The normalised image tensor with same shape as input :math:`(B, C, *)`.

    Example:
        >>> x = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(x, min_val=-1., max_val=1.)
        >>> x_norm.min()
        tensor(-1.)
        >>> x_norm.max()
        tensor(1.0000)
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")

    if not isinstance(max_val, float):
        raise TypeError(f"'b' should be a float. Got: {type(max_val)}.")

    if len(x.shape) < 3:
        raise ValueError(f"Input shape must be at least a 3d tensor. Got: {x.shape}.")

    shape = x.shape
    B, C = shape[0], shape[1]

    x_min: Tensor = x.view(B, C, -1).min(-1)[0].view(B, C, 1)
    x_max: Tensor = x.view(B, C, -1).max(-1)[0].view(B, C, 1)

    x_out: Tensor = (max_val - min_val) * (x.view(B, C, -1) - x_min) / (x_max - x_min + eps) + min_val
    return x_out.view(shape)
