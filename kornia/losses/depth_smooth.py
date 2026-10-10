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

from __future__ import annotations

import torch
from torch import nn

# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def inverse_depth_smoothness_loss(idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \operatorname{mean}_{nij} \left| \partial_x d_{nij} \right|
        e^{-\frac{1}{C} \sum_c \left| \partial_x I_{ncij} \right|}
        + \operatorname{mean}_{nij} \left| \partial_y d_{nij} \right|
        e^{-\frac{1}{C} \sum_c \left| \partial_y I_{ncij} \right|}

    with forward differences :math:`\partial_x` and :math:`\partial_y`, each mean taken over its own differences.

    Convention:
        - ``idepth`` is penalised and ``image`` only sets the weights: a constant inverse depth gives 0 for any image,
          and a constant image gives the plain mean of :math:`|\partial_x d|` plus that of :math:`|\partial_y d|`.
        - The weights use the raw image values, so ``image`` is expected in a unit range: in ``[0, 255]``, a step of
          a few levels already drives its weight to about 0.
        - ``idepth`` is not normalised, so the loss is linear in its scale;
          :ref:`Losses and metrics <losses-metrics-conventions>` shows how to port a normalised smoothness term.
        - Only the rank, :math:`(H, W)`, device and dtype are validated: the channel counts are not, a batch of one
          broadcasts against the other argument, and swapped arguments return a value.

    Args:
        idepth: tensor with the inverse depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.

    Return:
        a scalar with the computed loss.

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> loss = inverse_depth_smoothness_loss(idepth, image)

    """
    if not isinstance(idepth, torch.Tensor):
        raise TypeError(f"Input idepth type is not a torch.Tensor. Got {type(idepth)}")

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}")

    if not len(idepth.shape) == 4:
        raise ValueError(f"Invalid idepth shape, we expect BxCxHxW. Got: {idepth.shape}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if not idepth.shape[-2:] == image.shape[-2:]:
        raise ValueError(f"idepth and image shapes must be the same. Got: {idepth.shape} and {image.shape}")

    if not idepth.device == image.device:
        raise ValueError(f"idepth and image must be in the same device. Got: {idepth.device} and {image.device}")

    if not idepth.dtype == image.dtype:
        raise ValueError(f"idepth and image must be in the same dtype. Got: {idepth.dtype} and {image.dtype}")

    # compute the gradients
    idepth_dx: torch.Tensor = _gradient_x(idepth)
    idepth_dy: torch.Tensor = _gradient_y(idepth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)

    # compute image weights
    weights_x: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # apply image weights to depth
    smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
    smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


class InverseDepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \operatorname{mean}_{nij} \left| \partial_x d_{nij} \right|
        e^{-\frac{1}{C} \sum_c \left| \partial_x I_{ncij} \right|}
        + \operatorname{mean}_{nij} \left| \partial_y d_{nij} \right|
        e^{-\frac{1}{C} \sum_c \left| \partial_y I_{ncij} \right|}

    with forward differences :math:`\partial_x` and :math:`\partial_y`, each mean taken over its own differences.

    Convention:
        See the Convention block of :func:`~kornia.losses.inverse_depth_smoothness_loss`.

    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = InverseDepthSmoothnessLoss()
        >>> loss = smooth(idepth, image)

    """

    def forward(self, idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Compute image-aware smoothness regularization for inverse depth.

        Args:
            idepth: Inverse-depth tensor with shape :math:`(B, 1, H, W)`.
            image: RGB image tensor with shape :math:`(B, 3, H, W)` used to
                reduce the smoothness penalty across visible image edges.

        Returns:
            Scalar tensor containing the edge-aware smoothness loss. Smoothness
            is encouraged in flat image regions and relaxed near strong image
            gradients.
        """
        return inverse_depth_smoothness_loss(idepth, image)
