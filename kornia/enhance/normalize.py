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

from kornia.core.utils import _as_tensor_nonblocking

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
    """Normalize an image with a stored per-channel mean and standard deviation (module form).

    ``op = Normalize(mean, std)`` stores the statistics and ``out = op(input)`` is :func:`normalize` with them. A tensor
    statistic is stored as a non-persistent buffer, and an ``nn.Parameter`` stays a parameter that optimizers and
    checkpoints include.

    Convention:
        - See the convention block of :func:`normalize`. Unlike the function, the module also takes lists and tuples.

    Args:
        mean: mean of each channel, a number, a list, a tuple or a tensor.
        std: standard deviation of each channel, in the same forms.

    Shape:
        - Input: image ``(B, C, *)``.
        - Output: same shape as the input.

    See Also:
        - :func:`normalize`: the one-call form.
        - :class:`Denormalize`: the inverse.
        - :class:`~kornia.augmentation.Normalize`: the same operation inside an augmentation pipeline.

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

        if isinstance(mean, torch.Tensor) and not isinstance(mean, nn.Parameter):
            self.register_buffer("mean", mean, persistent=False)
        else:
            self.mean = mean

        if isinstance(std, torch.Tensor) and not isinstance(std, nn.Parameter):
            self.register_buffer("std", std, persistent=False)
        else:
            self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize an input tensor channel-wise with this module's statistics.

        This method is a thin wrapper over :func:`normalize`, reusing the
        ``mean`` and ``std`` values stored in the module constructor.

        Args:
            input: Tensor to normalize, typically with shape ``(*, C, ...)``,
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
    """Normalize an image or video with a per-channel mean and standard deviation.

    ``out = normalize(data, mean, std)`` returns ``(data - mean) / std`` for each channel of ``data`` ``(B, C, *)``,
    with the channel on dimension 1.

    Convention:
        - ``mean`` and ``std`` are converted to the dtype and device of ``data``; the output keeps its dtype.
        - An integer or bool ``data`` is converted to torch's default floating dtype first, so a fractional statistic is
          not truncated. Pixel values are not rescaled: give ``mean`` and ``std`` in the 0-255 scale for a uint8 image.

    Args:
        data: image or video with shape ``(B, C, *)``.
        mean: mean of each channel: a number, a 0-d tensor, or a tensor with shape ``(C,)``, ``(1, C)`` for all samples
            or ``(B, C)`` for one per sample. A list raises.
        std: standard deviation of each channel, in the same forms.

    Returns:
        Normalized tensor with the shape of ``data``.

    See Also:
        - :func:`denormalize`: the inverse.
        - :func:`normalize_min_max`: scales each channel plane to a value range instead of using fixed statistics.
        - :class:`Normalize`: the module form.

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
        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean] * shape[1], device=data.device, dtype=data.dtype)

        if isinstance(std, (int, float)):
            std = torch.tensor([std] * shape[1], device=data.device, dtype=data.dtype)

        # Allow broadcast on channel dimension
        if mean.shape and mean.shape[0] != 1 and mean.shape[0] != data.shape[1] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

        # Allow broadcast on channel dimension
        if std.shape and std.shape[0] != 1 and std.shape[0] != data.shape[1] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

        mean = _as_tensor_nonblocking(mean, data.device, data.dtype)
        std = _as_tensor_nonblocking(std, data.device, data.dtype)

    mean = mean[..., None]
    std = std[..., None]

    numel = 1
    for dim in shape[2:]:
        numel *= dim

    out: torch.Tensor = (data.reshape(shape[0], shape[1], numel) - mean) / std

    return out.reshape(shape)


class Denormalize(nn.Module):
    """Undo a per-channel mean and standard-deviation normalization with stored statistics (module form).

    ``op = Denormalize(mean, std)`` stores the statistics and ``out = op(input)`` is :func:`denormalize` with them. A
    tensor statistic is stored as a non-persistent buffer, and an ``nn.Parameter`` stays a parameter that optimizers and
    checkpoints include.

    Convention:
        - See the convention block of :func:`denormalize`. Unlike the function, the module also takes lists and tuples.

    Args:
        mean: mean of each channel, a number, a list, a tuple or a tensor.
        std: standard deviation of each channel, in the same forms.

    Shape:
        - Input: image ``(B, C, *)``.
        - Output: same shape as the input.

    See Also:
        - :func:`denormalize`: the one-call form.
        - :class:`Normalize`: the inverse.
        - :class:`~kornia.augmentation.Denormalize`: the same operation inside an augmentation pipeline.

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

        if isinstance(mean, torch.Tensor) and not isinstance(mean, nn.Parameter):
            self.register_buffer("mean", mean, persistent=False)
        else:
            self.mean = mean

        if isinstance(std, torch.Tensor) and not isinstance(std, nn.Parameter):
            self.register_buffer("std", std, persistent=False)
        else:
            self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Restore scale/offset from a tensor normalized by mean and std.

        This method delegates to :func:`denormalize` using this module's stored
        ``mean`` and ``std`` parameters.

        Args:
            input: Tensor to denormalize, commonly shaped ``(*, C, ...)``
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
    """Undo a per-channel mean and standard-deviation normalization of an image or video.

    ``out = denormalize(data, mean, std)`` returns ``data * std + mean`` for each channel of ``data`` ``(B, C, *)``,
    which inverts :func:`normalize` for the same ``mean`` and ``std``.

    Convention:
        - ``mean`` and ``std`` take the forms of :func:`normalize`, are converted to the dtype and device of ``data``,
          and the output keeps its dtype.
        - An integer or bool ``data`` is converted to torch's default floating dtype first, so the result is floating
          and cannot wrap.

    Args:
        data: image or video with shape ``(B, C, *)``.
        mean: mean of each channel: a number, a 0-d tensor, or a tensor with shape ``(C,)``, ``(1, C)`` or ``(B, C)``.
        std: standard deviation of each channel, in the same forms.

    Returns:
        Denormalized tensor with the shape of ``data``.

    See Also:
        - :func:`normalize`: the inverse.
        - :class:`Denormalize`: the module form.

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
        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean] * shape[1], device=data.device, dtype=data.dtype)

        if isinstance(std, (int, float)):
            std = torch.tensor([std] * shape[1], device=data.device, dtype=data.dtype)

        # Allow broadcast on channel dimension
        if mean.shape and mean.shape[0] != 1 and mean.shape[0] != data.shape[1] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

        # Allow broadcast on channel dimension
        if std.shape and std.shape[0] != 1 and std.shape[0] != data.shape[1] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

        mean = _as_tensor_nonblocking(mean, data.device, data.dtype)
        std = _as_tensor_nonblocking(std, data.device, data.dtype)

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
    """Rescale each channel plane of an image linearly to a value range.

    ``out = normalize_min_max(input, min_val, max_val)`` maps the minimum of each ``H x W`` plane of ``input`` to
    ``min_val`` and its maximum to ``max_val``, using that plane's own range:
    ``(max_val - min_val) * (x - x.min()) / (x.max() - x.min() + eps) + min_val``. A constant plane maps to ``min_val``.

    Convention:
        - Input is ``(*, C, H, W)``. Empty leading dimensions are preserved; channel and spatial dimensions must be
          nonzero.
        - Integer and boolean inputs are promoted to torch's default floating dtype first, as in :func:`normalize`;
          floating inputs keep their dtype.

    Args:
        input: image with shape ``(*, C, H, W)``.
        min_val: value of each plane's minimum, a ``float``.
        max_val: value of each plane's maximum, a ``float``.
        eps: value added to each plane's range, so the maximum lands slightly below ``max_val``.

    Returns:
        Rescaled image with the shape of ``input``.

    See Also:
        - :func:`normalize`: subtracts a given mean and divides by a given standard deviation instead.

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

    if not input.is_floating_point() and not input.is_complex():
        # Promote as `_promote_integer_data` does; `* 1.0` also follows the default dtype under TorchScript.
        input = input * 1.0
    shape = input.shape
    x_reshaped = input.flatten(start_dim=-2)
    x_min = x_reshaped.min(-1, keepdim=True)[0]
    x_max = x_reshaped.max(-1, keepdim=True)[0]

    x_out = (max_val - min_val) * (x_reshaped - x_min) / (x_max - x_min + eps) + min_val
    return x_out.reshape(shape)
