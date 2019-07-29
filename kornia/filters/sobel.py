import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])


class SpatialGradient(nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self) -> None:
        super(SpatialGradient, self).__init__()
        self.kernel: torch.Tensor = self.get_sobel_kernel()

    @staticmethod
    def get_sobel_kernel() -> torch.Tensor:
        kernel_x: torch.Tensor = _get_sobel_kernel_3x3()
        kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
        return torch.stack([kernel_x, kernel_y])

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with sobel kernel
        kernel_flip: torch.Tensor = kernel.flip(-3)
        # Pad with "replicate for spatial dims, but with zeros for channel dim
        padded_inp = F.pad(F.pad(input, [1, 1, 1, 1], 'replicate')[:, :, None],
                           [0, 0, 0, 0, 1, 1], 'constant')
        return F.conv3d(padded_inp, kernel_flip, padding=0, groups=c)


class Sobel(nn.Module):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Return:
        torch.Tensor: the sobel edge gradient maginitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.Sobel()(input)  # 1x3x4x4
    """

    def __init__(self) -> None:
        super(Sobel, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # comput the x/y gradients
        edges: torch.Tensor = spatial_gradient(input)

        # unpack the edges
        gx: torch.Tensor = edges[:, :, 0]
        gy: torch.Tensor = edges[:, :, 1]

        # compute gradient maginitude
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy)
        return magnitude


# functiona api


def spatial_gradient(input: torch.Tensor) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    See :class:`~kornia.filters.SpatialGradient` for details.
    """
    return SpatialGradient()(input)


def sobel(input: torch.Tensor) -> torch.Tensor:
    r"""Computes the Sobel operator and returns the magnitude per channel.

    See :class:`~kornia.filters.Sobel` for details.
    """
    return Sobel()(input)
