from typing import Tuple, List

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia

__all__ = [
    "PyrDown",
    "PyrUp",
    "ScalePyramid",
    "pyrdown",
    "pyrup",
    "build_pyramid",
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


class PyrDown(kornia.nn.geometry.PyrDown):
    r"""Blurs a tensor and downsamples it.

    Args:
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Return:
        torch.Tensor: the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = kornia.transform.PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self, border_type: str = 'reflect', align_corners: bool = False) -> None:
        super(PyrDown, self).__init__()
        kornia.deprecation_warning("kornia.geometry.PyrDown", "kornia.nn.geometry.PyrDown")


class PyrUp(kornia.nn.geometry.PyrUp):
    r"""Upsamples a tensor and then blurs it.

    Args:
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Return:
        torch.Tensor: the upsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H * 2, W * 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = kornia.transform.PyrUp()(input)  # 1x2x8x8
    """

    def __init__(self, border_type: str = 'reflect', align_corners: bool = False):
        super(PyrUp, self).__init__()
        kornia.deprecation_warning("kornia.geometry.PyrUp", "kornia.nn.geometry.PyrUp")


class ScalePyramid(kornia.nn.geometry.ScalePyramid):
    r"""Creates an scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur and
    downscaled.
    Arguments:
        n_levels (int): number of the levels in octave.
        init_sigma (float): initial blur level.
        min_size (int): the minimum size of the octave in pixels. Default is 5
        double_image (bool): add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this. Default is False
    Returns:
        Tuple(List(Tensors), List(Tensors), List(Tensors)):
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, C, NL, H, W), (B, C, NL, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples::
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = kornia.ScalePyramid(3, 15)(input)
    """

    def __init__(self,
                 n_levels: int = 3,
                 init_sigma: float = 1.6,
                 min_size: int = 15,
                 double_image: bool = False):
        super(ScalePyramid, self).__init__()
        kornia.deprecation_warning("kornia.geometry.ScalePyramid", "kornia.nn.geometry.ScalePyramid")


# functional api
def pyrdown(
        input: torch.Tensor,
        border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Blurs a tensor and downsamples it.

    See :class:`~kornia.transform.PyrDown` for details.
    """
    return kornia.nn.PyrDown(border_type, align_corners)(input)


def pyrup(input: torch.Tensor, border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Upsamples a tensor and then blurs it.

    See :class:`~kornia.transform.PyrUp` for details.
    """
    return kornia.nn.PyrUp(border_type, align_corners)(input)


def build_pyramid(
        input: torch.Tensor,
        max_level: int,
        border_type: str = 'reflect', align_corners: bool = False) -> List[torch.Tensor]:
    r"""Constructs the Gaussian pyramid for an image.

    The function constructs a vector of images and builds the Gaussian pyramid
    by recursively applying pyrDown to the previously built pyramid layers.

    Args:
        input (torch.Tensor): the tensor to be used to constructuct the pyramid.
        max_level (int): 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, NL, C, H, W), (B, NL, C, H/2, W/2), ...]`
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if not isinstance(max_level, int) or max_level < 0:
        raise ValueError(f"Invalid max_level, it must be a positive integer. Got: {max_level}")

    # create empty list and append the original image
    pyramid: List[torch.Tensor] = []
    pyramid.append(input)

    # iterate and downsample

    for _ in range(max_level - 1):
        img_curr: torch.Tensor = pyramid[-1]
        img_down: torch.Tensor = pyrdown(img_curr, border_type, align_corners)
        pyramid.append(img_down)

    return pyramid
