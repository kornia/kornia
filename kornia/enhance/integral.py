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

from typing import Optional, Tuple

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


def integral_tensor(input: torch.Tensor, dim: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Sum a tensor cumulatively along chosen axes.

    ``out = integral_tensor(input, dim)`` takes the inclusive cumulative sum of ``input`` along each axis in ``dim``.
    For ``dim=(-2, -1)``, ``out[..., i, j]`` is the sum of ``input[..., :i + 1, :j + 1]``.

    Convention:
        - ``dim`` is a non-empty tuple of unique axes; an ``int`` raises ``TypeError``.
        - There is no zero row or column in front, so ``cv2.integral(x)[1:, 1:]`` equals the result for the two
          spatial axes.

    Args:
        input: tensor with shape ``(*, D)``, one axis or more.
        dim: axes to sum along. The default is the last axis.

    Returns:
        Cumulative sums with the shape of ``input``.

    See Also:
        - :func:`integral_image`: sums the two spatial axes.
        - :class:`IntegralTensor`: the module form.

    Examples:
        >>> input = torch.ones(3, 5)
        >>> output = integral_tensor(input, (-2, -1))
        >>> output
        tensor([[ 1.,  2.,  3.,  4.,  5.],
                [ 2.,  4.,  6.,  8., 10.],
                [ 3.,  6.,  9., 12., 15.]])

    """
    KORNIA_CHECK_SHAPE(input, ["*", "D"])

    if dim is None:
        dim = (-1,)

    KORNIA_CHECK(len(dim) > 0, "dim must be a non-empty tuple.")
    KORNIA_CHECK(len(dim) <= len(input.shape), "dim must be a tuple of length <= input.shape.")

    normalized_dim = tuple(d if d >= 0 else d + input.ndim for d in dim)
    KORNIA_CHECK(
        all(0 <= d < input.ndim for d in normalized_dim),
        "dim must contain valid dimensions for the input.",
    )
    KORNIA_CHECK(
        len(set(normalized_dim)) == len(normalized_dim),
        "dim must contain unique dimensions.",
    )

    output = input
    for i in normalized_dim:
        output = output.cumsum(i)
    return output


def integral_image(image: torch.Tensor) -> torch.Tensor:
    """Sum an image cumulatively along its rows and columns.

    ``out = integral_image(image)`` returns the integral image of ``image``: ``out[..., i, j]`` is the sum of
    ``image[..., :i + 1, :j + 1]``.

    Convention:
        There is no zero row or column in front, unlike ``cv2.integral``, whose result is larger by one row and
        column.

    Args:
        image: image with shape ``(*, H, W)``.

    Returns:
        Integral image with shape ``(*, H, W)``.

    See Also:
        - :func:`integral_tensor`: sums along any axes.
        - :class:`IntegralImage`: the module form.

    Examples:
        >>> input = torch.ones(1, 5, 5)
        >>> output = integral_image(input)
        >>> output
        tensor([[[ 1.,  2.,  3.,  4.,  5.],
                 [ 2.,  4.,  6.,  8., 10.],
                 [ 3.,  6.,  9., 12., 15.],
                 [ 4.,  8., 12., 16., 20.],
                 [ 5., 10., 15., 20., 25.]]])

    """
    KORNIA_CHECK_SHAPE(image, ["*", "H", "W"])

    return integral_tensor(image, (-2, -1))


class IntegralTensor(nn.Module):
    """Sum a tensor cumulatively along chosen axes (module form).

    ``op = IntegralTensor(dim)`` stores ``dim`` and ``out = op(input)`` is :func:`integral_tensor` with it.

    Convention:
        - See the convention block of :func:`integral_tensor`.

    Args:
        dim: axes to sum along. The default is the last axis.

    Shape:
        - Input: ``(*, D)``
        - Output: same shape as the input.

    See Also:
        - :func:`integral_tensor`: the one-call form.
        - :class:`IntegralImage`: sums the two spatial axes.

    Examples:
        >>> input = torch.ones(3, 5)
        >>> dim = (-2, -1)
        >>> output = IntegralTensor(dim)(input)
        >>> output
        tensor([[ 1.,  2.,  3.,  4.,  5.],
                [ 2.,  4.,  6.,  8., 10.],
                [ 3.,  6.,  9., 12., 15.]])

    """

    def __init__(self, dim: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute cumulative sums of the input tensor along configured axes.

        Args:
            input: Tensor to integrate. The operation supports arbitrary rank,
                and dimensions are accumulated according to ``self.dim``.

        Returns:
            A tensor with the same shape as ``input`` containing cumulative
            sums. If ``self.dim`` is ``None``, integration is applied to the
            last dimension by :func:`integral_tensor`.
        """
        return integral_tensor(input, self.dim)


class IntegralImage(nn.Module):
    """Sum an image cumulatively along its rows and columns (module form).

    ``out = IntegralImage()(image)`` is :func:`integral_image`.

    Convention:
        - See the convention block of :func:`integral_image`.

    Shape:
        - Input: ``(*, H, W)``
        - Output: same shape as the input.

    See Also:
        - :func:`integral_image`: the one-call form.
        - :class:`IntegralTensor`: sums along any axes.

    Examples:
        >>> input = torch.ones(1, 5, 5)
        >>> output = IntegralImage()(input)
        >>> output
        tensor([[[ 1.,  2.,  3.,  4.,  5.],
                 [ 2.,  4.,  6.,  8., 10.],
                 [ 3.,  6.,  9., 12., 15.],
                 [ 4.,  8., 12., 16., 20.],
                 [ 5., 10., 15., 20., 25.]]])

    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute a 2D integral image by summing over height and width.

        Args:
            input: Tensor whose last two dimensions represent spatial axes,
                typically shaped ``(*, H, W)``.

        Returns:
            A tensor of the same shape as ``input`` where each location stores
            the cumulative sum over the top-left rectangle ending at that
            location.
        """
        return integral_image(input)
