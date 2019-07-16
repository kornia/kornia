import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.filters import gaussian_blur2d
from typing import Tuple, List

__all__ = [
    "PyrDown",
    "PyrUp",
    "ScalePyramid",
    "pyrdown",
    "pyrup",
]


def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]]) / 256.


class PyrDown(nn.Module):
    r"""Blurs a tensor and downsamples it.

    Args:
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Return:
        torch.Tensor: the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = kornia.transform.PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self, border_type: str = 'reflect') -> None:
        super(PyrDown, self).__init__()
        self.border_type: str = border_type
        self.kernel: torch.Tensor = _get_pyramid_gaussian_kernel()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # blur image
        x_blur: torch.Tensor = kornia.filter2D(
            input, self.kernel, self.border_type)

        # reject even rows and columns.
        out: torch.Tensor = F.avg_pool2d(x_blur, 2, 2)
        return out


class PyrUp(nn.Module):
    r"""Upsamples a tensor and then blurs it.

    Args:
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Return:
        torch.Tensor: the upsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H * 2, W * 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = kornia.transform.PyrUp()(input)  # 1x2x8x8
    """

    def __init__(self, border_type: str = 'reflect'):
        super(PyrUp, self).__init__()
        self.border_type: str = border_type
        self.kernel: torch.Tensor = _get_pyramid_gaussian_kernel()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # upsample tensor
        b, c, height, width = input.shape
        x_up: torch.Tensor = F.interpolate(input, size=(height * 2, width * 2),
                                           mode='bilinear', align_corners=True)

        # blurs upsampled tensor
        x_blur: torch.Tensor = kornia.filter2D(
            x_up, self.kernel, self.border_type)
        return x_blur


class ScalePyramid(nn.Module):
    r"""Creates an scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur and
    downscaled.
    Arguments:
        n_levels (int): number of the levels in octave.
        init_sigma (float): initial blur level.
        min_size (int): the minimum size of the octave in pixels. Default is 5
    Returns:
        Tuple(List(Tensors), List(Tensors), List(Tensors)):
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, NL, C, H, W), (B, NL, C, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples::
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = kornia.ScalePyramid(3, 15)(input)
    """

    def __init__(self,
                 n_levels: int = 3,
                 init_sigma: float = 1.6,
                 min_size: int = 5):
        super(ScalePyramid, self).__init__()
        self.n_levels = n_levels
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1. / float(self.n_levels))
        return

    def get_kernel_size(self, sigma: float):
        ksize = int(2.0 * 3.0 * sigma + 1.0)
        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def forward(self, x: torch.Tensor) -> Tuple[  # type: ignore
            List, List, List]:
        bs, ch, h, w = x.size()
        pixel_distance = 1.0
        cur_sigma = 0.5
        if self.init_sigma > cur_sigma:
            sigma = math.sqrt(self.init_sigma**2 - cur_sigma**2)
            cur_sigma = self.init_sigma
            ksize = self.get_kernel_size(sigma)
            cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
        else:
            cur_level = x
        sigmas = [cur_sigma * torch.ones(bs, self.n_levels).to(x.device)]
        pixel_dists = [pixel_distance * torch.ones(
                       bs,
                       self.n_levels).to(
                       x.device)]
        pyr = [[cur_level.unsqueeze(1)]]
        oct_idx = 0
        while True:
            cur_level = pyr[-1][0].squeeze(1)
            for level_idx in range(1, self.n_levels):
                sigma = cur_sigma * math.sqrt(self.sigma_step**2 - 1.0)
                cur_level = gaussian_blur2d(
                    cur_level, (ksize, ksize), (sigma, sigma))
                cur_sigma *= self.sigma_step
                pyr[-1].append(cur_level.unsqueeze(1))
                sigmas[-1][:, level_idx] = cur_sigma
                pixel_dists[-1][:, level_idx] = pixel_distance
            nextOctaveFirstLevel = F.avg_pool2d(
                cur_level, kernel_size=2, stride=2, padding=0)
            pixel_distance *= 2.0
            cur_sigma = self.init_sigma
            if (min(nextOctaveFirstLevel.size(2),
                    nextOctaveFirstLevel.size(3)) <= self.min_size):
                break
            pyr.append([nextOctaveFirstLevel.unsqueeze(1)])
            sigmas.append(cur_sigma * torch.ones(
                          bs,
                          self.n_levels).to(
                          x.device))
            pixel_dists.append(
                pixel_distance * torch.ones(
                    bs,
                    self.n_levels).to(
                    x.device))
            oct_idx += 1
        for i in range(len(pyr)):
            pyr[i] = torch.cat(pyr[i], dim=1)  # type: ignore
        return pyr, sigmas, pixel_dists


# functiona api


def pyrdown(
        input: torch.Tensor,
        border_type: str = 'reflect') -> torch.Tensor:
    r"""Blurs a tensor and downsamples it.

    See :class:`~kornia.transform.PyrDown` for details.
    """
    return PyrDown(border_type)(input)


def pyrup(input: torch.Tensor, border_type: str = 'reflect') -> torch.Tensor:
    r"""Upsamples a tensor and then blurs it.

    See :class:`~kornia.transform.PyrUp` for details.
    """
    return PyrUp(border_type)(input)
