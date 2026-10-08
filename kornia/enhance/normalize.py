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

__all__ = ["Denormalize", "Normalize", "denormalize", "normalize", "normalize_min_max"]


def _promote_integer_data(data: torch.Tensor) -> torch.Tensor:
    r"""Return ``data`` in torch's default floating dtype if it is an integer or bool tensor.

    :func:`normalize` and :func:`denormalize` cast ``mean`` and ``std`` to the dtype of ``data``. For an integer
    tensor that truncates a fractional statistic to an integer (``0.5`` becomes ``0``) and runs the subtraction and
    ``addcmul`` in the integer dtype, where the result wraps. Promoting the data first, as ``data * 0.5`` does, keeps
    a fractional statistic and computes in floating point. Floating and complex tensors are returned untouched.
    """
    if data.is_floating_point() or data.is_complex():
        return data
    return data.to(torch.get_default_dtype())


class Normalize(nn.Module):
    r"""Normalize a torch.Tensor image with mean and standard deviation.

    Convention:
        The channel axis is dimension 1: inputs are ``(B, C, *)`` and one statistic
        per channel has shape (C,) (or (B, C) for separate batch statistics).
        See :func:`normalize` for the functional contract.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image torch.Tensor of size :math:`(B, C, *)`.
        - Output: Normalised torch.Tensor with same size as input :math:`(B, C, *)`.

    Note:
        An integer or bool input is converted to torch's default floating dtype first; see :func:`normalize`.

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

        if isinstance(mean, torch.Tensor):
            self.register_buffer("mean", mean, persistent=False)
        else:
            self.mean = mean

        if isinstance(std, torch.Tensor):
            self.register_buffer("std", std, persistent=False)
        else:
            self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize an input tensor channel-wise with this module's statistics.

        This method is a thin wrapper over :func:`normalize`, reusing the
        ``mean`` and ``std`` values stored in the module constructor.

        Args:
            input: Tensor to normalize, typically with shape :math:`(*, C, ...)`,
                where ``*`` represents optional leading dimensions (for example
                batch) and ``C`` is the channel dimension.

        Returns:
            A tensor with the same shape as ``input`` whose channel values are
            normalized by ``(x - mean) / std``.
        """
        # Promote before the statistics are built in the input dtype: for an integer input a fractional Python
        # statistic would otherwise truncate here, before `normalize` sees it.
        input = _promote_integer_data(input)

        # A Python number becomes (1,) and a sequence (1, *shape), the shapes the
        # constructor used to store, built in the input dtype so float64 keeps its bits.
        mean = self.mean
        std = self.std
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean, device=input.device, dtype=input.dtype)[None]
        if not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std, device=input.device, dtype=input.dtype)[None]
        return normalize(input, mean, std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    r"""Normalize an image/video torch.Tensor with mean and standard deviation.

    Convention:
        This function treats dimension 0 as batch and dimension 1 as channel.
        mean and std may be (C,), (1, C), or (B, C); the output has the input shape.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        data: Image torch.Tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Normalised torch.Tensor with same size as input :math:`(B, C, *)`.

    Note:
        An integer or bool ``data`` is converted to torch's default floating dtype (float32 unless changed) before
        ``mean`` and ``std`` are applied, so a fractional statistic is not truncated and the result is floating.
        Floating and complex inputs keep their dtype. Pixel values are not rescaled: give ``mean`` and ``std`` in the
        0-255 scale for a uint8 image.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = normalize(x, torch.tensor([0.0]), torch.tensor([255.]))
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
        >>> normalize(x, 127.5, 127.5)
        tensor([[[[-1.,  1.]]]])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = normalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

    """
    data = _promote_integer_data(data)
    shape = data.shape

    if torch.onnx.is_in_onnx_export():
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise ValueError("Only torch.Tensor is accepted when converting to ONNX.")
        # A per-channel vector broadcasts the same way as its (1, C) view; accept it instead of raising.
        if mean.dim() == 1:
            mean = mean.view(1, -1)
        if std.dim() == 1:
            std = std.view(1, -1)
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
        if mean.shape and mean.shape[0] != 1 and mean.shape[0] != data.shape[1] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

        # Allow broadcast on channel dimension
        if std.shape and std.shape[0] != 1 and std.shape[0] != data.shape[1] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

        mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
        std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    mean = mean[..., None]
    std = std[..., None]

    numel = 1
    for dim in shape[2:]:
        numel *= dim

    out: torch.Tensor = (data.reshape(shape[0], shape[1], numel) - mean) / std

    return out.reshape(shape)


class Denormalize(nn.Module):
    r"""Denormalize a torch.Tensor image with mean and standard deviation.

    Convention:
        See :func:`denormalize`; the inverse uses dimension 1 as channel on
        ``(B, C, *)`` input, matching :class:`Normalize`.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image torch.Tensor of size :math:`(B, C, *)`.
        - Output: Denormalised torch.Tensor with same size as input :math:`(B, C, *)`.

    Note:
        An integer or bool input is converted to torch's default floating dtype first; see :func:`denormalize`.

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

        if isinstance(mean, torch.Tensor):
            self.register_buffer("mean", mean, persistent=False)
        else:
            self.mean = mean

        if isinstance(std, torch.Tensor):
            self.register_buffer("std", std, persistent=False)
        else:
            self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Restore scale/offset from a tensor normalized by mean and std.

        This method delegates to :func:`denormalize` using this module's stored
        ``mean`` and ``std`` parameters.

        Args:
            input: Tensor to denormalize, commonly shaped :math:`(*, C, ...)`
                with channel dimension ``C``.

        Returns:
            A tensor with the same shape as ``input`` where each channel is
            transformed by ``x * std + mean``.
        """
        # Promote before the statistics are built in the input dtype, as in `Normalize.forward`.
        input = _promote_integer_data(input)

        # A Python number becomes (1,) and a sequence keeps its own shape, as the
        # constructor used to store them: a (C,) list is still checked against the
        # channel count, a (B, C) list still gives per-sample statistics, and the
        # ONNX branch, which indexes ``mean.shape[0]``, never sees a 0-d tensor.
        mean = self.mean
        std = self.std
        if not isinstance(mean, torch.Tensor):
            mean = torch.atleast_1d(torch.as_tensor(mean, device=input.device, dtype=input.dtype))
        if not isinstance(std, torch.Tensor):
            std = torch.atleast_1d(torch.as_tensor(std, device=input.device, dtype=input.dtype))
        return denormalize(input, mean, std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def denormalize(data: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> torch.Tensor:
    r"""Denormalize an image/video torch.Tensor with mean and standard deviation.

    Convention:
        This is the elementwise inverse of :func:`normalize` for matching mean and std
        on ``(B, C, *)`` input.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        data: Image torch.Tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Return:
        Denormalised torch.Tensor with same size as input :math:`(B, C, *)`.

    Note:
        An integer or bool ``data`` is converted to torch's default floating dtype (float32 unless changed) before
        ``mean`` and ``std`` are applied, so the result is floating and cannot wrap. Floating and complex inputs keep
        their dtype.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = denormalize(x, 0.0, 255.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
        >>> denormalize(x, 10., 2.)
        tensor([[[[ 10., 520.]]]])

        >>> x = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = denormalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])

    """
    data = _promote_integer_data(data)
    shape = data.shape

    if torch.onnx.is_in_onnx_export():
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise ValueError("Only torch.Tensor is accepted when converting to ONNX.")
        # A per-channel vector broadcasts the same way as its (1, C) view; accept it instead of raising.
        if mean.dim() == 1:
            mean = mean.view(1, -1)
        if std.dim() == 1:
            std = std.view(1, -1)
        if mean.shape[0] != 1 or std.shape[0] != 1:
            raise ValueError("Batch dimension must be one for broadcasting when converting to ONNX.")
    else:
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=data.device, dtype=data.dtype)

        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=data.device, dtype=data.dtype)

        # Allow broadcast on channel dimension
        if mean.shape and mean.shape[0] != 1 and mean.shape[0] != data.shape[1] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

        # Allow broadcast on channel dimension
        if std.shape and std.shape[0] != 1 and std.shape[0] != data.shape[1] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

        mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
        std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    if mean.dim() == 1:
        mean = mean.view(1, -1, *([1] * (data.dim() - 2)))
    # If the torch.Tensor is >1D (e.g., (B, C)), reshape to (B, C, 1, ...)
    else:
        while len(mean.shape) < data.dim():
            mean = mean.unsqueeze(-1)

    if std.dim() == 1:
        std = std.view(1, -1, *([1] * (data.dim() - 2)))
    else:
        while len(std.shape) < data.dim():
            std = std.unsqueeze(-1)

    return torch.addcmul(mean, data, std)


def normalize_min_max(
    input: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    r"""Normalise an image/video torch.Tensor by MinMax and re-scales the value between a range.

    Convention:
        Input is ``(*, C, H, W)``: minima and maxima are taken over H and W separately for every
        leading index and channel. Constant planes map to min_val. Empty leading dimensions are
        preserved; channel and spatial dimensions must be nonzero.

    The data is normalised using the following formulation:

    .. math::
        y_i = (b - a) * \frac{x_i - \text{min}(x)}{\text{max}(x) - \text{min}(x)} + a

    where :math:`a` is :math:`\text{min_val}` and :math:`b` is :math:`\text{max_val}`.

    Args:
        input: The image torch.Tensor to be normalised with shape :math:`(*, C, H, W)`.
        min_val: The minimum value for the new range.
        max_val: The maximum value for the new range.
        eps: Float number to avoid zero division.

    Returns:
        The normalised image torch.Tensor with same shape as input :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(x, min_val=-1., max_val=1.)
        >>> x_norm.min()
        tensor(-1.)
        >>> x_norm.max()
        tensor(1.0000)

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"data should be a torch.Tensor. Got: {type(input)}.")

    if input.ndim < 2:
        raise ValueError(f"Input tensor must have at least two dimensions. Got {input.shape}")

    if 0 in input.shape[-3:]:
        raise ValueError("Invalid input tensor, channel and spatial dimensions must be nonzero.")

    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")

    if not isinstance(max_val, float):
        raise TypeError(f"'max_val' should be a float. Got: {type(max_val)}.")

    shape = input.shape
    x_reshaped = input.flatten(start_dim=-2)
    x_min = x_reshaped.min(-1, keepdim=True)[0]
    x_max = x_reshaped.max(-1, keepdim=True)[0]

    x_out = (max_val - min_val) * (x_reshaped - x_min) / (x_max - x_min + eps) + min_val
    return x_out.reshape(shape)
