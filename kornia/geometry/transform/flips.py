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

import torch
from torch import nn

__all__ = ["Hflip", "Rot180", "Vflip", "hflip", "rot180", "vflip"]


class Vflip(nn.Module):
    r"""Vertically flip a torch.tensor image or a batch of torch.tensor images.

    Input must be a torch.tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input torch.tensor.

    Returns:
        The vertically flipped image torch.tensor.

    Examples:
        >>> vflip = Vflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> vflip(input)
        torch.tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])

    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return vflip(input)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Hflip(nn.Module):
    r"""Horizontally flip a torch.tensor image or a batch of torch.tensor images.

    Input must be a torch.tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input torch.tensor.

    Returns:
        The horizontally flipped image torch.tensor.

    Examples:
        >>> hflip = Hflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> hflip(input)
        torch.tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]])

    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return hflip(input)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Rot180(nn.Module):
    r"""Rotate a torch.tensor image or a batch of torch.tensor images 180 degrees.

    Input must be a torch.tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input torch.tensor.

    Examples:
        >>> rot180 = Rot180()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> rot180(input)
        torch.tensor([[[[1., 1., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])

    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rot180(input)

    def __repr__(self) -> str:
        return self.__class__.__name__


def rot180(input: torch.Tensor) -> torch.Tensor:
    r"""Rotate a torch.tensor image or a batch of torch.tensor images 180 degrees.

    .. image:: _static/img/rot180.png

    Input must be a torch.tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input torch.tensor.

    Returns:
        The rotated image torch.tensor.

    """
    return torch.flip(input, [-2, -1])


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Horizontally flip a torch.tensor image or a batch of torch.tensor images.

    .. image:: _static/img/hflip.png

    Input must be a torch.tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input torch.tensor.

    Returns:
        The horizontally flipped image torch.tensor.

    """
    return input.flip(-1).contiguous()


def vflip(input: torch.Tensor) -> torch.Tensor:
    r"""Vertically flip a torch.tensor image or a batch of torch.tensor images.

    .. image:: _static/img/vflip.png

    Input must be a torch.tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input torch.tensor.

    Returns:
        The vertically flipped image torch.tensor.

    """
    return input.flip(-2).contiguous()
