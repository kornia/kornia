from typing import Optional, Tuple

import torch

from kornia.core import Tensor


def marginal_pdf(values: Tensor, bins: Tensor, sigma: Tensor, epsilon: float = 1e-10) -> Tuple[Tensor, Tensor]:
    """Calculate the marginal probability distribution function of the input tensor based on the number of
    histogram bins.

    Args:
        values: shape [BxNx1].
        bins: shape [NUM_BINS].
        sigma: shape [1], gaussian smoothing factor.
        epsilon: scalar, for numerical stability.

    Returns:
        Tuple[Tensor, Tensor]:
          - Tensor: shape [BxN].
          - Tensor: shape [BxNxNUM_BINS].
    """

    if not isinstance(values, Tensor):
        raise TypeError(f"Input values type is not a Tensor. Got {type(values)}")

    if not isinstance(bins, Tensor):
        raise TypeError(f"Input bins type is not a Tensor. Got {type(bins)}")

    if not isinstance(sigma, Tensor):
        raise TypeError(f"Input sigma type is not a Tensor. Got {type(sigma)}")

    if not values.dim() == 3:
        raise ValueError(f"Input values must be a of the shape BxNx1. Got {values.shape}")

    if not bins.dim() == 1:
        raise ValueError(f"Input bins must be a of the shape NUM_BINS. Got {bins.shape}")

    if not sigma.dim() == 0:
        raise ValueError(f"Input sigma must be a of the shape 1. Got {sigma.shape}")

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return pdf, kernel_values


def joint_pdf(kernel_values1: Tensor, kernel_values2: Tensor, epsilon: float = 1e-10) -> Tensor:
    """Calculate the joint probability distribution function of the input tensors based on the number of histogram
    bins.

    Args:
        kernel_values1: shape [BxNxNUM_BINS].
        kernel_values2: shape [BxNxNUM_BINS].
        epsilon: scalar, for numerical stability.

    Returns:
        shape [BxNUM_BINSxNUM_BINS].
    """
    if not isinstance(kernel_values1, Tensor):
        raise TypeError(f"Input kernel_values1 type is not a Tensor. Got {type(kernel_values1)}")

    if not isinstance(kernel_values2, Tensor):
        raise TypeError(f"Input kernel_values2 type is not a Tensor. Got {type(kernel_values2)}")

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
    pdf = joint_kernel_values / normalization

    return pdf


def histogram(x: Tensor, bins: Tensor, bandwidth: Tensor, epsilon: float = 1e-10) -> Tensor:
    """Estimate the histogram of the input tensor.

    The calculation uses kernel density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x: Input tensor to compute the histogram with shape :math:`(B, D)`.
        bins: The number of bins to use the histogram :math:`(N_{bins})`.
        bandwidth: Gaussian smoothing factor with shape shape [1].
        epsilon: A scalar, for numerical stability.

    Returns:
        Computed histogram of shape :math:`(B, N_{bins})`.

    Examples:
        >>> x = torch.rand(1, 10)
        >>> bins = torch.torch.linspace(0, 255, 128)
        >>> hist = histogram(x, bins, bandwidth=torch.tensor(0.9))
        >>> hist.shape
        torch.Size([1, 128])
    """
    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)

    return pdf


def histogram2d(x1: Tensor, x2: Tensor, bins: Tensor, bandwidth: Tensor, epsilon: float = 1e-10) -> Tensor:
    """Estimate the 2d histogram of the input tensor.

    The calculation uses kernel density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x1: Input tensor to compute the histogram with shape :math:`(B, D1)`.
        x2: Input tensor to compute the histogram with shape :math:`(B, D2)`.
        bins: The number of bins to use the histogram :math:`(N_{bins})`.
        bandwidth: Gaussian smoothing factor with shape shape [1].
        epsilon: A scalar, for numerical stability. Default: 1e-10.

    Returns:
        Computed histogram of shape :math:`(B, N_{bins}), N_{bins})`.

    Examples:
        >>> x1 = torch.rand(2, 32)
        >>> x2 = torch.rand(2, 32)
        >>> bins = torch.torch.linspace(0, 255, 128)
        >>> hist = histogram2d(x1, x2, bins, bandwidth=torch.tensor(0.9))
        >>> hist.shape
        torch.Size([2, 128, 128])
    """

    _, kernel_values1 = marginal_pdf(x1.unsqueeze(2), bins, bandwidth, epsilon)
    _, kernel_values2 = marginal_pdf(x2.unsqueeze(2), bins, bandwidth, epsilon)

    pdf = joint_pdf(kernel_values1, kernel_values2)

    return pdf


def image_histogram2d(
    image: Tensor,
    min: float = 0.0,
    max: float = 255.0,
    n_bins: int = 256,
    bandwidth: Optional[float] = None,
    centers: Optional[Tensor] = None,
    return_pdf: bool = False,
    kernel: str = "triangular",
    eps: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """Estimate the histogram of the input image(s).

    The calculation uses triangular kernel density estimation.

    Args:
        image: Input tensor to compute the histogram with shape
          :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive). Ignored when
          :attr:`centers` is specified.
        n_bins: The number of histogram bins. Ignored when
          :attr:`centers` is specified.
        bandwidth: Smoothing factor. If not specified or equal to -1,
          :math:`(bandwidth = (max - min) / n_bins)`.
        centers: Centers of the bins with shape :math:`(n_bins,)`.
          If not specified or empty, it is calculated as centers of
          equal width bins of [min, max] range.
        return_pdf: If True, also return probability densities for
          each bin.
        kernel: kernel to perform kernel density estimation
          ``(`triangular`, `gaussian`, `uniform`, `epanechnikov`)``.

    Returns:
        Computed histogram of shape :math:`(bins)`, :math:`(C, bins)`,
          :math:`(B, C, bins)`.
        Computed probability densities of shape :math:`(bins)`, :math:`(C, bins)`,
          :math:`(B, C, bins)`, if return_pdf is ``True``. Tensor of zeros with shape
          of the histogram otherwise.
    """
    if image is not None and not isinstance(image, Tensor):
        raise TypeError(f"Input image type is not a Tensor. Got {type(image)}.")

    if centers is not None and not isinstance(centers, Tensor):
        raise TypeError(f"Bins' centers type is not a Tensor. Got {type(centers)}.")

    if centers is not None and len(centers.shape) > 0 and centers.dim() != 1:
        raise ValueError(f"Bins' centers must be a Tensor of the shape (n_bins,). Got {centers.shape}.")

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

    if bandwidth is None:
        bandwidth = (max - min) / n_bins

    if centers is None:
        centers = min + bandwidth * (torch.arange(n_bins, device=image.device, dtype=image.dtype) + 0.5)
    centers = centers.reshape(-1, 1, 1, 1, 1)

    u = torch.abs(image.unsqueeze(0) - centers) / bandwidth

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

    if return_pdf:
        normalization = torch.sum(hist, dim=-1, keepdim=True) + eps
        pdf = hist / normalization
        if image.dim() == 2:
            hist = hist.squeeze()
            pdf = pdf.squeeze()
        elif image.dim() == 3:
            hist = hist.squeeze(0)
            pdf = pdf.squeeze(0)
        return hist, pdf

    if image.dim() == 2:
        hist = hist.squeeze()
    elif image.dim() == 3:
        hist = hist.squeeze(0)

    return hist, torch.zeros_like(hist)
