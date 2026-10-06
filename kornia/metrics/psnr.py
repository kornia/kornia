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
from torch.nn.functional import mse_loss as mse


def psnr(image: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Create a function that calculates the PSNR between 2 images.

    PSNR is the Peak Signal to Noise Ratio. For one m x n image, the PSNR is:

    .. math::

        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::

        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Convention:
        - One MSE is pooled over every element, batch and channels included, and the result is a single value: for a
          batch it is the PSNR of the pooled MSE, not the mean of per-image PSNRs, so compute per-image values one
          image at a time. :ref:`Losses and metrics <losses-metrics-conventions>` maps both onto scikit-image and
          torchmetrics.
        - ``max_val`` is :math:`\text{MAX}_I`, the data range of the images (``1.0`` for ``[0, 1]``, ``255.0`` for
          ``[0, 255]``), not the maximum of the tensor; pixel values are not rescaled.
        - ``image`` and ``target`` must have the same shape, without broadcasting; the value is symmetric in them.
          Identical inputs give an MSE of 0 and ``inf``, while a batch with one identical pair stays finite.
        - Known defect: integer images are not supported: they reach torch's ``mse_loss`` unconverted, where
          :func:`~kornia.metrics.ssim` computes them in float32, and the result is undefined
          (`#5536 <https://github.com/kornia/kornia/issues/5536>`_).

    Args:
        image: the input image with arbitrary shape :math:`(*)`.
        target: the labels image with arbitrary shape :math:`(*)`.
        max_val: the data range of the images, :math:`\text{MAX}_I`.

    Return:
        the PSNR as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(20.0000)

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(image)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if image.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {image.shape} and {target.shape}")

    return 10.0 * torch.log10(max_val**2 / mse(image, target, reduction="mean"))
