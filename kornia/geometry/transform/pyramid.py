import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia

__all__ = [
    "PyrDown",
    "PyrUp",
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
        out: torch.Tensor = F.avg_pool2d(x_blur, 2,2)
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
