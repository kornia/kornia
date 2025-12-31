# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""nn.Module containing functionals for intensity normalisation."""

from typing import List, Tuple, Union

import torch
from torch import nn

from kornia.utils.image import perform_keep_shape_image

__all__ = ["Denormalize", "Normalize", "denormalize", "normalize", "normalize_min_max"]


class Normalize(nn.Module):
    r"""Normalize a torch.tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image torch.tensor of size :math:`(*, C, ...)`.
        - Output: Normalised torch.tensor with same size as input :math:`(*, C, ...)`.

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
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    r"""Normalize an image/video torch.tensor with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        data: Image torch.tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Normalised torch.tensor with same size as input :math:`(B, C, *)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = F.normalize(x, torch.tensor([0.0]), torch.tensor([255.]))
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = F.normalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

    """
    shape = data.shape

    if torch.onnx.is_in_onnx_export():
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise ValueError("Only torch.tensor is accepted when converting to ONNX.")
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

    out: torch.Tensor = (data.view(shape[0], shape[1], -1) - mean) / std

    return out.view(shape)


class Denormalize(nn.Module):
    r"""Denormalize a torch.tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image torch.tensor of size :math:`(*, C, ...)`.
        - Output: Denormalised torch.tensor with same size as input :math:`(*, C, ...)`.

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

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> None:
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return denormalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def denormalize(data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> torch.Tensor:
    r"""Denormalize an image/video torch.tensor with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        data: Image torch.tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Denormalised torch.tensor with same size as input :math:`(B, C, *)`.

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
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise ValueError("Only torch.tensor is accepted when converting to ONNX.")
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

    if mean.dim() == 1:
        mean = mean.view(1, -1, *([1] * (data.dim() - 2)))
    # If the torch.tensor is >1D (e.g., (B, C)), reshape to (B, C, 1, ...)
    else:
        while len(mean.shape) < data.dim():
            mean = mean.unsqueeze(-1)

    if std.dim() == 1:
        std = std.view(1, -1, *([1] * (data.dim() - 2)))
    else:
        while len(std.shape) < data.dim():
            std = std.unsqueeze(-1)

    return torch.addcmul(mean, data, std)


@perform_keep_shape_image
def normalize_min_max(x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    r"""Normalise an image/video torch.tensor by MinMax and re-scales the value between a range.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \frac{x_i - \text{min}(x)}{\text{max}(x) - \text{min}(x)} + a

    torch.where :math:`a` is :math:`\text{min_val}` and :math:`b` is :math:`\text{max_val}`.

    Args:
        x: The image torch.tensor to be normalised with shape :math:`(*, C, H, W)`.
        min_val: The minimum value for the new range.
        max_val: The maximum value for the new range.
        eps: Float number to avoid zero division.

    Returns:
        The normalised image torch.tensor with same shape as input :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(x, min_val=-1., max_val=1.)
        >>> x_norm.min()
        torch.tensor(-1.)
        >>> x_norm.max()
        torch.tensor(1.0000)

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a torch.tensor. Got: {type(x)}.")

    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")

    if not isinstance(max_val, float):
        raise TypeError(f"'max_val' should be a float. Got: {type(max_val)}.")

    shape = x.shape
    B, C = shape[0], shape[1]

    x_reshaped = x.view(B, C, -1)
    x_min = x_reshaped.min(-1, keepdim=True)[0]  # Shape: (B, C, 1)
    x_max = x_reshaped.max(-1, keepdim=True)[0]  # Shape: (B, C, 1)

    x_out = (max_val - min_val) * (x_reshaped - x_min) / (x_max - x_min + eps) + min_val
    return x_out.view(shape)
