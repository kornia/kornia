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
        - Integer and bool images are converted to float32 before computing the MSE; pixel values are not rescaled.
        - Float16 and bfloat16 inputs are evaluated in float32; the result retains their promoted input dtype,
          also promoted with ``max_val.dtype`` when the data range is a tensor. A float32 tensor data range therefore
          gives a float32 result for half-precision images.

    Args:
        image: the input image with arbitrary shape :math:`(*)`.
        target: the labels image with arbitrary shape :math:`(*)`.
        max_val: the data range of the images, :math:`\text{MAX}_I`.

    Return:
        the PSNR as a scalar.

    .. note::
        Integer and bool images are converted to float32, as in :func:`~kornia.metrics.ssim`, and the two images are
        compared in their promoted dtype. An integer image paired with a float16 or bfloat16 image therefore gives a
        float32 result. Pixel values are not rescaled.

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

    # mse has no integer kernel; compute integer images in float32 as ssim does.
    if not image.is_floating_point() and not image.is_complex():
        image = image.to(torch.float32)
    if not target.is_floating_point() and not target.is_complex():
        target = target.to(torch.float32)
    # Give mse_loss one dtype: on two dtypes it aborts the process on MPS, and its backward raises on torch 2.5.1.
    dtype = torch.promote_types(image.dtype, target.dtype)
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    image, target = image.to(compute_dtype), target.to(compute_dtype)
    result_dtype = dtype
    if dtype in (torch.float16, torch.bfloat16) and isinstance(max_val, torch.Tensor):
        result_dtype = torch.promote_types(dtype, max_val.dtype)
        # Keep the tensor data range's square and division out of half precision too, without narrowing float64.
        max_val = max_val.to(torch.promote_types(compute_dtype, max_val.dtype))

    # The legacy (TorchScript) ONNX exporter writes ``log10`` as a division by a ``(1,)`` constant, which turns a 0-d
    # result into a one-element vector. Reshaping to the shape of the ratio keeps the eager shape there (``()`` for a
    # scalar ``max_val``, the shape of a tensor one) and is a no-op view eagerly.
    ratio = max_val**2 / mse(image, target, reduction="mean")
    result = 10.0 * torch.log10(ratio)
    if dtype in (torch.float16, torch.bfloat16):
        result = result.to(result_dtype)
    return result.reshape(ratio.shape)
