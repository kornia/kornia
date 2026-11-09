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

import math
from typing import Optional, Tuple

import torch

from kornia.core.check import _check_async
from kornia.core.exceptions import ShapeError


def marginal_pdf(
    values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, epsilon: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the marginal probability distribution function of the input based on the number of histogram bins.

    Args:
        values: shape [BxNx1].
        bins: shape [NUM_BINS].
        sigma: shape [1], gaussian smoothing factor.
        epsilon: scalar, for numerical stability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
          - torch.Tensor: shape [BxN].
          - torch.Tensor: shape [BxNxNUM_BINS].

    """
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"Input values type is not a torch.Tensor. Got {type(values)}")

    if not isinstance(bins, torch.Tensor):
        raise TypeError(f"Input bins type is not a torch.Tensor. Got {type(bins)}")

    if not isinstance(sigma, torch.Tensor):
        raise TypeError(f"Input sigma type is not a torch.Tensor. Got {type(sigma)}")

    if not values.dim() == 3:
        raise ValueError(f"Input values must be a of the shape BxNx1. Got {values.shape}")

    if not bins.dim() == 1:
        raise ValueError(f"Input bins must be a of the shape NUM_BINS. Got {bins.shape}")

    if not sigma.dim() == 0:
        raise ValueError(f"Input sigma must be a of the shape 1. Got {sigma.shape}")

    # Asynchronous like the other tensor-parameter checks in kornia.enhance: no host sync, and a compiled graph
    # keeps the check instead of skipping it.
    _check_async(torch.isfinite(sigma) & (sigma > 0), "Bandwidth must be finite and greater than zero.")

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return pdf, kernel_values


def joint_pdf(kernel_values1: torch.Tensor, kernel_values2: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Calculate the joint probability distribution function of the input tensors based on the number of histogram bins.

    Args:
        kernel_values1: shape [BxNxNUM_BINS].
        kernel_values2: shape [BxNxNUM_BINS].
        epsilon: scalar, for numerical stability.

    Returns:
        shape [BxNUM_BINSxNUM_BINS].

    """
    if not isinstance(kernel_values1, torch.Tensor):
        raise TypeError(f"Input kernel_values1 type is not a torch.Tensor. Got {type(kernel_values1)}")

    if not isinstance(kernel_values2, torch.Tensor):
        raise TypeError(f"Input kernel_values2 type is not a torch.Tensor. Got {type(kernel_values2)}")

    if not kernel_values1.dim() == 3:
        raise ValueError(f"Input kernel_values1 must be a of the shape BxN. Got {kernel_values1.shape}")

    if not kernel_values2.dim() == 3:
        raise ValueError(f"Input kernel_values2 must be a of the shape BxN. Got {kernel_values2.shape}")

    if kernel_values1.shape != kernel_values2.shape:
        raise ValueError(
            "Inputs kernel_values1 and kernel_values2 must have the same shape."
            f" Got {kernel_values1.shape} and {kernel_values2.shape}"
        )

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + epsilon
    return joint_kernel_values / normalization


def histogram(x: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Estimate the normalized histogram of each sample row with Gaussian kernels.

    ``hist = histogram(x, bins, bandwidth)`` spreads every value of a row of ``x`` over the bin centers ``bins`` with a
    Gaussian kernel and normalizes each row by its weight sum plus ``epsilon``.

    Convention:
        - Rows are independent samples, and ``bins`` is the one center grid they share.
        - A ``bandwidth`` that is not finite and positive raises ``RuntimeError``; on CUDA, and on MPS from torch 2.13,
          it raises by the next synchronization.
        - For float16 and bfloat16 samples and bins, the kernels are evaluated in float32 and the result is cast back to
          that dtype.

    Args:
        x: samples with shape ``(B, D)``.
        bins: bin centers with shape ``(N_bins,)``, in the units of ``x``.
        bandwidth: 0-d tensor, the standard deviation of the Gaussian kernel in the units of ``x``. A Python float or a
            ``(1,)`` tensor raises.
        epsilon: value added to the normalization sum.

    Returns:
        Histogram with shape ``(B, N_bins)``. Each row sums approximately to 1 when its weight sum dominates
        ``epsilon``; it is zero when all kernel weights underflow to zero.

    See Also:
        - :func:`histogram2d`: the joint histogram of two sample rows.
        - :func:`image_histogram2d`: counts the pixels of each image channel, with a choice of kernel.

    Examples:
        >>> x = torch.rand(1, 10)
        >>> bins = torch.linspace(0, 1, 128)
        >>> hist = histogram(x, bins, bandwidth=torch.tensor(0.01))
        >>> hist.shape
        torch.Size([1, 128])

    """
    # In half precision the gradient reaches the samples through per-bin terms below the normal range: evaluate the
    # kernels in float32. The 0-d bandwidth does not promote the kernels, so it does not pick the dtype either.
    kernel_dtype = torch.promote_types(x.dtype, bins.dtype)
    if kernel_dtype in (torch.float16, torch.bfloat16):
        pdf, _ = marginal_pdf(x.float().unsqueeze(2), bins.float(), bandwidth.float(), epsilon)
        return pdf.to(kernel_dtype)

    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)

    return pdf


def histogram2d(
    x1: torch.Tensor, x2: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor, epsilon: float = 1e-10
) -> torch.Tensor:
    """Estimate the normalized joint histogram of two sample rows with Gaussian kernels.

    ``hist = histogram2d(x1, x2, bins, bandwidth)`` pairs the values ``x1[b, n]`` and ``x2[b, n]`` and returns their
    joint weights over the bin centers: ``hist[b, i, j]`` is the weight of ``x1`` in bin ``i`` and ``x2`` in bin ``j``,
    normalized by their sum plus ``epsilon`` per batch element. The sum approaches 1 when the joint weight sum
    dominates ``epsilon``; all-zero kernel weights give an all-zero result.

    Convention:
        - ``x1`` and ``x2`` must have the same shape.
        - For float16 and bfloat16 samples and bins, the kernels, joint counts and normalization are evaluated in
          float32 and the result is cast back to that dtype.

    Args:
        x1: first samples with shape ``(B, D)``.
        x2: second samples with shape ``(B, D)``.
        bins: bin centers with shape ``(N_bins,)``, in the units of the samples.
        bandwidth: 0-d tensor, the standard deviation of the Gaussian kernel in the units of the samples.
        epsilon: value added to the normalization sum.

    Returns:
        Joint histogram with shape ``(B, N_bins, N_bins)``; the first bin axis belongs to ``x1``.

    See Also:
        - :func:`histogram`: the histogram of one sample row.
        - :func:`image_histogram2d`: one histogram per image channel, not a joint one.

    Examples:
        >>> x1 = torch.rand(2, 32)
        >>> x2 = torch.rand(2, 32)
        >>> bins = torch.linspace(0, 1, 128)
        >>> hist = histogram2d(x1, x2, bins, bandwidth=torch.tensor(0.01))
        >>> hist.shape
        torch.Size([2, 128, 128])

    """
    # In float16 the joint counts and their normalization overflow on ordinary images, and the gradient reaches the
    # samples through per-bin terms below the float16 normal range: evaluate the kernels in float32 as well.
    kernel_dtype = torch.promote_types(x1.dtype, bins.dtype)
    if kernel_dtype in (torch.float16, torch.bfloat16):
        x1, x2, bins, bandwidth = x1.float(), x2.float(), bins.float(), bandwidth.float()
    _, kernel_values1 = marginal_pdf(x1.unsqueeze(2), bins, bandwidth, epsilon)
    _, kernel_values2 = marginal_pdf(x2.unsqueeze(2), bins, bandwidth, epsilon)

    return joint_pdf(kernel_values1, kernel_values2, epsilon).to(kernel_dtype)


def _check_image_rank(image: torch.Tensor) -> None:
    """Reject the ranks that image_histogram2d does not document instead of failing inside the computation."""
    if image.dim() < 2 or image.dim() > 4:
        raise ShapeError(f"Input image must have shape (H, W), (C, H, W) or (B, C, H, W). Got {image.shape}.")


def _restore_float_dtype(hist: torch.Tensor, image: torch.Tensor, auto_centers: bool) -> torch.Tensor:
    """Hand back the image's dtype when it can represent the histogram counts.

    Only for auto-built centers on a floating-point image: an integer image or explicit ``centers`` keep the promoted
    dtype (at least float32), and half-precision counts stay float32 because ordinary image counts exceed their range.
    """
    if auto_centers and image.is_floating_point() and image.dtype not in (torch.float16, torch.bfloat16):
        return hist.to(image.dtype)
    return hist


def _is_finite(value: float) -> bool:
    """Same as ``math.isfinite``, but traceable when Dynamo turns a changed float argument into a symbolic float."""
    return -math.inf < value < math.inf


def _validate_histogram_bandwidth(
    min: float, max: float, n_bins: int, bandwidth: Optional[float], auto_centers: bool
) -> float:
    """Validate the range needed for automatic values and resolve the bandwidth sentinel."""
    auto_bandwidth = bandwidth is None or bandwidth == -1.0
    if auto_centers or auto_bandwidth:
        if n_bins <= 0:
            raise ValueError(f"n_bins must be greater than zero for automatically generated values. Got {n_bins}.")
        if not _is_finite(min) or not _is_finite(max) or max <= min:
            raise ValueError(
                f"The automatically generated histogram range must be finite and non-empty. Got [{min}, {max}]."
            )

    if bandwidth is None or bandwidth == -1.0:
        bandwidth = (max - min) / n_bins

    if not _is_finite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(f"Bandwidth must be finite and greater than zero. Got {bandwidth}.")
    return bandwidth


def image_histogram2d(
    image: torch.Tensor,
    min: float = 0.0,
    max: float = 255.0,
    n_bins: int = 256,
    bandwidth: Optional[float] = None,
    centers: Optional[torch.Tensor] = None,
    return_pdf: bool = False,
    kernel: str = "triangular",
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Estimate the intensity histogram of each channel of an image.

    ``hist, pdf = image_histogram2d(image, min, max, n_bins)`` weights every pixel over ``n_bins`` bin centers with a
    kernel and returns the summed weights per channel as ``hist``, and their normalization as ``pdf`` if
    ``return_pdf=True``. Despite its name it is a one-dimensional histogram per channel; for the joint histogram of two
    signals use :func:`histogram2d`.

    Convention:
        - The spatial axes are the final two. An image ``(H, W)``, ``(C, H, W)`` or ``(B, C, H, W)`` gives ``hist`` with
          the same leading axes followed by the bins; another rank raises.
        - A pixel outside the range still adds the kernel's weight to the centers within reach; it is not clipped
          into an end bin.
        - Float16 and bfloat16 images are accumulated and returned in float32, or in the dtype of wider explicit
          ``centers``: ordinary bin counts exceed float16's range (65504) and bfloat16's precision.

    Args:
        image: image with shape ``(H, W)``, ``(C, H, W)`` or ``(B, C, H, W)``.
        min: lower end of the range, a ``float``. It sets the automatic centers and bandwidth.
        max: upper end of the range, a ``float``. The default ``255.0`` puts a :math:`[0, 1]` image in two bins; pass
            ``1.0`` for that range. Ignored when ``bandwidth`` is given.
        n_bins: number of automatic bins.
        bandwidth: kernel width, a positive finite ``float``. The default, also ``-1``, is ``(max - min) / n_bins``.
        centers: bin centers with shape ``(K,)``. The default, also an empty tensor, is ``n_bins`` centers at
            ``min + (i + 0.5) * bandwidth``, so an explicit ``bandwidth`` moves them off ``[min, max]``
            (`#6664 <https://github.com/kornia/kornia/issues/6664>`_). Pass ``bandwidth`` with it: the default
            bandwidth still comes from ``min``, ``max`` and ``n_bins``
            (`#6662 <https://github.com/kornia/kornia/issues/6662>`_).
        return_pdf: whether to also return the normalized histogram.
        kernel: weight of a pixel at distance ``d`` from a center, with ``u = d / bandwidth``: ``'triangular'``
            (:math:`1 - u` for :math:`u \leq 1`), ``'gaussian'`` (:math:`e^{-u^2 / 2}`), ``'uniform'`` (1 for
            :math:`u \leq 1`) or ``'epanechnikov'`` (:math:`1 - u^2` for :math:`u \leq 1`). Gradients reach ``image``
            except with ``'uniform'``.
        eps: value added to the normalization sum.

    Returns:
        Tuple ``(hist, pdf)``. ``hist`` has shape ``(K,)``, ``(C, K)`` or ``(B, C, K)``, where the bin axis ``K`` has
        ``n_bins`` entries, or one per center. ``pdf`` has the same shape and divides the histogram by its bin sum
        plus ``eps``. Its sum approaches 1 when the bin sum dominates ``eps``; it is zeros when no pixels contribute
        or ``return_pdf`` is ``False``.

    See Also:
        - :func:`histogram`: the Gaussian histogram of sample rows, normalized.
        - :func:`histogram2d`: the joint histogram of two sample rows.

    Example:
        >>> image = torch.rand(2, 3, 8, 8)
        >>> hist, pdf = image_histogram2d(image, 0.0, 1.0, n_bins=16, return_pdf=True)
        >>> hist.shape
        torch.Size([2, 3, 16])

    """
    if image is not None and not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}.")

    _check_image_rank(image)

    if centers is not None and not isinstance(centers, torch.Tensor):
        raise TypeError(f"Bins' centers type is not a torch.Tensor. Got {type(centers)}.")

    if centers is not None and len(centers.shape) > 0 and centers.dim() != 1:
        raise ValueError(f"Bins' centers must be a torch.Tensor of the shape (n_bins,). Got {centers.shape}.")

    if not isinstance(min, float):
        raise TypeError(f"Type of lower end of the range is not a float. Got {type(min)}.")

    if not isinstance(max, float):
        raise TypeError(f"Type of upper end of the range is not a float. Got {type(min)}.")

    if not isinstance(n_bins, int):
        raise TypeError(f"Type of number of bins is not an int. Got {type(n_bins)}.")

    if bandwidth is not None and not isinstance(bandwidth, float):
        raise TypeError(f"Bandwidth type is not a float. Got {type(bandwidth)}.")

    if not isinstance(return_pdf, bool):
        raise TypeError(f"Return_pdf type is not a bool. Got {type(return_pdf)}.")

    if centers is not None and centers.numel() == 0:
        centers = None
    auto_centers = centers is None
    bandwidth = _validate_histogram_bandwidth(min, max, n_bins, bandwidth, auto_centers)

    # Preserve the supplied center precision while promoting half inputs enough to represent image-sized counts.
    center_dtype = centers.dtype if centers is not None else torch.float32
    compute_dtype = torch.promote_types(torch.float32, torch.promote_types(image.dtype, center_dtype))
    if centers is None:
        centers = min + bandwidth * (torch.arange(n_bins, device=image.device, dtype=compute_dtype) + 0.5)
    centers = centers.to(compute_dtype).reshape(-1, 1, 1, 1, 1)

    u = torch.abs(image.to(compute_dtype).unsqueeze(0) - centers) / bandwidth

    if kernel == "gaussian":
        kernel_values = torch.exp(-0.5 * u**2)
    elif kernel in ("triangular", "uniform", "epanechnikov"):
        # compute the mask and cast to floating point
        mask = (u <= 1).to(u.dtype)
        if kernel == "triangular":
            kernel_values = (1.0 - u) * mask
        elif kernel == "uniform":
            kernel_values = mask
        else:  # kernel == "epanechnikov"
            kernel_values = (1.0 - u**2) * mask
    else:
        raise ValueError(f"Kernel must be 'triangular', 'gaussian', 'uniform' or 'epanechnikov'. Got {kernel}.")

    hist = torch.sum(kernel_values, dim=(-2, -1)).permute(1, 2, 0)
    hist = _restore_float_dtype(hist, image, auto_centers)

    if return_pdf:
        normalization = torch.sum(hist, dim=-1, keepdim=True) + eps
        pdf = hist / normalization
        if image.dim() == 2:
            hist = hist[0, 0]
            pdf = pdf[0, 0]
        elif image.dim() == 3:
            hist = hist.squeeze(0)
            pdf = pdf.squeeze(0)
        return hist, pdf

    if image.dim() == 2:
        hist = hist[0, 0]
    elif image.dim() == 3:
        hist = hist.squeeze(0)

    return hist, torch.zeros_like(hist)
