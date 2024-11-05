from typing import List

from kornia.core import Module, Tensor, pad
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters import filter3d, get_gaussian_kernel3d
from kornia.filters.filter import _compute_padding


def _crop(img: Tensor, cropping_shape: List[int]) -> Tensor:
    """Crop out the part of "valid" convolution area."""
    return pad(
        img,
        (
            -cropping_shape[4],
            -cropping_shape[5],
            -cropping_shape[2],
            -cropping_shape[3],
            -cropping_shape[0],
            -cropping_shape[1],
        ),
    )


def ssim3d(
    img1: Tensor, img2: Tensor, window_size: int, max_val: float = 1.0, eps: float = 1e-12, padding: str = "same"
) -> Tensor:
    r"""Function that computes the Structural Similarity (SSIM) index map between two images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        img1: the first input image with shape :math:`(B, C, D, H, W)`.
        img2: the second input image with shape :math:`(B, C, D, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
       The ssim index map with shape :math:`(B, C, D, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5, 5)
        >>> ssim_map = ssim3d(input1, input2, 5)  # 1x4x5x5x5
    """
    KORNIA_CHECK_IS_TENSOR(img1)
    KORNIA_CHECK_IS_TENSOR(img2)
    KORNIA_CHECK_SHAPE(img1, ["B", "C", "D", "H", "W"])
    KORNIA_CHECK_SHAPE(img2, ["B", "C", "D", "H", "W"])
    KORNIA_CHECK(img1.shape == img2.shape, f"img1 and img2 shapes must be the same. Got: {img1.shape} and {img2.shape}")

    if not isinstance(max_val, float):
        raise TypeError(f"Input max_val type is not a float. Got {type(max_val)}")

    # prepare kernel
    kernel: Tensor = get_gaussian_kernel3d((window_size, window_size, window_size), (1.5, 1.5, 1.5))

    # compute coefficients
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2

    # compute local mean per channel
    mu1: Tensor = filter3d(img1, kernel)
    mu2: Tensor = filter3d(img2, kernel)

    cropping_shape: List[int] = []
    if padding == "valid":
        depth, height, width = kernel.shape[-3:]
        cropping_shape = _compute_padding([depth, height, width])
        mu1 = _crop(mu1, cropping_shape)
        mu2 = _crop(mu2, cropping_shape)
    elif padding == "same":
        pass

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    mu_img1_sq = filter3d(img1**2, kernel)
    mu_img2_sq = filter3d(img2**2, kernel)
    mu_img1_img2 = filter3d(img1 * img2, kernel)

    if padding == "valid":
        mu_img1_sq = _crop(mu_img1_sq, cropping_shape)
        mu_img2_sq = _crop(mu_img2_sq, cropping_shape)
        mu_img1_img2 = _crop(mu_img1_img2, cropping_shape)
    elif padding == "same":
        pass

    # compute local sigma per channel
    sigma1_sq = mu_img1_sq - mu1_sq
    sigma2_sq = mu_img2_sq - mu2_sq
    sigma12 = mu_img1_img2 - mu1_mu2

    # compute the similarity index map
    num: Tensor = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den: Tensor = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return num / (den + eps)


class SSIM3D(Module):
    r"""Create a module that computes the Structural Similarity (SSIM) index between two 3D images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Shape:
        - Input: :math:`(B, C, D, H, W)`.
        - Target :math:`(B, C, D, H, W)`.
        - Output: :math:`(B, C, D, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5, 5)
        >>> ssim = SSIM3D(5)
        >>> ssim_map = ssim(input1, input2)  # 1x4x5x5x5
    """

    def __init__(self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, padding: str = "same") -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps = eps
        self.padding = padding

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        return ssim3d(img1, img2, self.window_size, self.max_val, self.eps, self.padding)
