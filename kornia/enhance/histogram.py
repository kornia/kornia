import math
from typing import Optional, Tuple

import torch

__all__ = ["histogram", "histogram2d", "image_histogram2d"]


def marginal_pdf(
    values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, epsilon: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the marginal probability distribution function of the input tensor based on the number of
    histogram bins.

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
        raise ValueError("Input values must be a of the shape BxNx1." " Got {}".format(values.shape))

    if not bins.dim() == 1:
        raise ValueError("Input bins must be a of the shape NUM_BINS" " Got {}".format(bins.shape))

    if not sigma.dim() == 0:
        raise ValueError("Input sigma must be a of the shape 1" " Got {}".format(sigma.shape))

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return pdf, kernel_values


def joint_pdf(kernel_values1: torch.Tensor, kernel_values2: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Calculate the joint probability distribution function of the input tensors based on the number of histogram
    bins.

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
        raise ValueError("Input kernel_values1 must be a of the shape BxN." " Got {}".format(kernel_values1.shape))

    if not kernel_values2.dim() == 3:
        raise ValueError("Input kernel_values2 must be a of the shape BxN." " Got {}".format(kernel_values2.shape))

    if kernel_values1.shape != kernel_values2.shape:
        raise ValueError(
            "Inputs kernel_values1 and kernel_values2 must have the same shape."
            " Got {} and {}".format(kernel_values1.shape, kernel_values2.shape)
        )

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + epsilon
    pdf = joint_kernel_values / normalization

    return pdf


def histogram(x: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
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


def histogram2d(
    x1: torch.Tensor, x2: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor, epsilon: float = 1e-10
) -> torch.Tensor:
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
    if image is not None and not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}.")

    if centers is not None and not isinstance(centers, torch.Tensor):
        raise TypeError(f"Bins' centers type is not a torch.Tensor. Got {type(centers)}.")

    if centers is not None and len(centers.shape) > 0 and centers.dim() != 1:
        raise ValueError(f"Bins' centers must be a torch.Tensor of the shape (n_bins,). Got {centers.shape}.")

    if not isinstance(min, float):
        raise TypeError(f'Type of lower end of the range is not a float. Got {type(min)}.')

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
        centers = min + bandwidth * (torch.arange(n_bins, device=image.device, dtype=image.dtype).float() + 0.5)
    centers = centers.reshape(-1, 1, 1, 1, 1)
    u = torch.abs(image.unsqueeze(0) - centers) / bandwidth
    if kernel == "triangular":
        mask = (u <= 1).to(u.dtype)
        kernel_values = (1 - u) * mask
    elif kernel == "gaussian":
        kernel_values = torch.exp(-0.5 * u ** 2)
    elif kernel == "uniform":
        mask = (u <= 1).to(u.dtype)
        kernel_values = torch.ones_like(u, dtype=u.dtype, device=u.device) * mask
    elif kernel == "epanechnikov":
        mask = (u <= 1).to(u.dtype)
        kernel_values = (1 - u ** 2) * mask
    else:
        raise ValueError(f"Kernel must be 'triangular', 'gaussian', " f"'uniform' or 'epanechnikov'. Got {kernel}.")

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
    return hist, torch.zeros_like(hist, dtype=hist.dtype, device=hist.device)
