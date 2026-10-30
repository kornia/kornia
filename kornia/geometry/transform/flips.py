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

"""Mirror images left to right or top to bottom, or turn them by 180 degrees, by reversing tensor axes.

The flips reverse the last axes of a tensor of any batch shape: :func:`hflip` the last one (the width ``W``),
:func:`vflip` the second to last (the height ``H``) and :func:`rot180` both. They copy elements without interpolation,
so they are exact for every dtype and differentiable, and their module forms take no arguments.

Which function to use:

- Mirror left to right: :func:`hflip` or :class:`Hflip`; top to bottom: :func:`vflip` or :class:`Vflip`.
- Turn by a half turn: :func:`rot180` or :class:`Rot180`.
- To rotate by any angle, use :func:`~kornia.geometry.transform.rotate`; for random flips in an augmentation
  pipeline, :class:`~kornia.augmentation.RandomHorizontalFlip` or :class:`~kornia.augmentation.RandomVerticalFlip`.
"""

import torch
from torch import nn

__all__ = ["Hflip", "Rot180", "Vflip", "hflip", "rot180", "vflip"]


class Vflip(nn.Module):
    r"""Flip images top to bottom by reversing the height axis (module form).

    Build ``flipper = Vflip()``, which takes no arguments, then ``out = flipper(input)`` reverses the rows of
    ``(*, H, W)`` tensors as :func:`vflip` does. To mirror left to right, use :class:`Hflip`; for a half turn,
    :class:`Rot180`.

    Convention:
        - See the convention block of :func:`~kornia.geometry.transform.vflip`.

    Returns:
        The vertically flipped image torch.Tensor, with the shape of the input.

    See Also:
        - :func:`vflip`: the one-call form.
        - :class:`Hflip`: mirrors left to right instead.
        - :class:`~kornia.augmentation.RandomVerticalFlip`: flips with a probability, in an augmentation pipeline.

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Flip the image content along the vertical image axis.

        Args:
            input: Tensor containing one image or a batch of images, typically
                shaped ``(*, C, H, W)``, where :math:`H` is height and
                :math:`W` is width.

        Returns:
            Tensor with the same shape as ``input`` whose rows are reversed
            from top to bottom.
        """
        return vflip(input)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Hflip(nn.Module):
    r"""Mirror images left to right by reversing the width axis (module form).

    Build ``flipper = Hflip()``, which takes no arguments, then ``out = flipper(input)`` reverses the columns of
    ``(*, W)`` tensors as :func:`hflip` does. To flip top to bottom, use :class:`Vflip`; for a half turn,
    :class:`Rot180`.

    Convention:
        - See the convention block of :func:`~kornia.geometry.transform.hflip`.

    Returns:
        The horizontally flipped image torch.Tensor, with the shape of the input.

    See Also:
        - :func:`hflip`: the one-call form.
        - :class:`Vflip`: flips top to bottom instead.
        - :class:`~kornia.augmentation.RandomHorizontalFlip`: flips with a probability, in an augmentation pipeline.

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Flip the image content along the horizontal image axis.

        Args:
            input: Tensor containing one image or a batch of images, typically
                shaped ``(*, C, H, W)``, where :math:`H` is height and
                :math:`W` is width.

        Returns:
            Tensor with the same shape as ``input`` whose columns are reversed
            from left to right.
        """
        return hflip(input)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Rot180(nn.Module):
    r"""Turn images by 180 degrees by reversing the height and width axes (module form).

    Build ``rotator = Rot180()``, which takes no arguments, then ``out = rotator(input)`` reverses both the rows and the
    columns of ``(*, H, W)`` tensors as :func:`rot180` does, without interpolation. To rotate by a stored angle, use
    :class:`Rotate`.

    Convention:
        - See the convention block of :func:`~kornia.geometry.transform.rot180`.

    Returns:
        The rotated image torch.Tensor, with the shape of the input.

    See Also:
        - :func:`rot180`: the one-call form.
        - :class:`Rotate`: rotates by any stored angle, with interpolation.

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Rotate image content by 180 degrees without changing tensor shape.

        Args:
            input: Tensor containing one image or a batch of images, typically
                shaped ``(*, C, H, W)``, where :math:`H` is height and
                :math:`W` is width.

        Returns:
            Tensor with the same shape as ``input`` and both spatial axes
            reversed.
        """
        return rot180(input)

    def __repr__(self) -> str:
        return self.__class__.__name__


def rot180(input: torch.Tensor) -> torch.Tensor:
    r"""Turn images by 180 degrees by reversing the height and width axes.

    ``out = rot180(input)`` reverses the last two axes of ``(*, H, W)`` tensors and returns the same shape: the image
    turned by a half turn, exactly, without the interpolation of :func:`rotate`. For any other angle, or a turn about
    another centre, use :func:`rotate`; for the module form, :class:`Rot180`.

    .. image:: _static/img/rot180.png

    Convention:
        - flips both the height and width axes (equivalent to :func:`vflip` followed by
          :func:`hflip`)
        - requires rank ≥ 2: unbatched ``(H, W)``/``(C, H, W)`` or batched
          ``(*, C, H, W)`` all work

    Args:
        input: input torch.Tensor.

    Returns:
        The rotated image torch.Tensor, with the shape of the input.

    See Also:
        - :class:`Rot180`: the module form.
        - :func:`rotate`: rotates by any angle about the centre or a given point, with interpolation; at 180 degrees
          about the default centre it matches ``rot180`` up to float rounding.
        - :func:`hflip`: reverses the width axis only.
        - :func:`vflip`: reverses the height axis only.

    """
    return torch.flip(input, [-2, -1])


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Mirror images left to right by reversing the width axis.

    ``out = hflip(input)`` reverses the last axis of ``(*, W)`` tensors, the width of ``(*, H, W)`` images, and returns
    the same shape. To flip top to bottom, use :func:`vflip`; for a half turn, :func:`rot180`; for random flips in an
    augmentation pipeline, :class:`~kornia.augmentation.RandomHorizontalFlip`.

    .. image:: _static/img/hflip.png

    Convention:
        - flips the last dimension (dim ``-1``); accepts any rank :math:`\geq 1`
          (rank 0 is outside the contract)

    Args:
        input: input torch.Tensor.

    Returns:
        The horizontally flipped image torch.Tensor, with the shape of the input.

    See Also:
        - :class:`Hflip`: the module form.
        - :func:`vflip`: reverses the height axis instead.
        - :func:`rot180`: reverses both axes.
        - :class:`~kornia.augmentation.RandomHorizontalFlip`: flips with a probability, in an augmentation pipeline.

    """
    return input.flip(-1).contiguous()


def vflip(input: torch.Tensor) -> torch.Tensor:
    r"""Flip images top to bottom by reversing the height axis.

    ``out = vflip(input)`` reverses the second-to-last axis of ``(*, H, W)`` tensors and returns the same shape. To
    mirror left to right, use :func:`hflip`; for a half turn, :func:`rot180`; for random flips in an augmentation
    pipeline, :class:`~kornia.augmentation.RandomVerticalFlip`.

    .. image:: _static/img/vflip.png

    Convention:
        - flips along the height axis (the second-to-last dimension)
        - requires rank ≥ 2 (operates on dim ``-2``): unbatched ``(H, W)``/
          ``(C, H, W)`` or batched ``(*, C, H, W)`` all work

    Args:
        input: input torch.Tensor.

    Returns:
        The vertically flipped image torch.Tensor, with the shape of the input.

    See Also:
        - :class:`Vflip`: the module form.
        - :func:`hflip`: reverses the width axis instead.
        - :func:`rot180`: reverses both axes.
        - :class:`~kornia.augmentation.RandomVerticalFlip`: flips with a probability, in an augmentation pipeline.

    """
    return input.flip(-2).contiguous()
