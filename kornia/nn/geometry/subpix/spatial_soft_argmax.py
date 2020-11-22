from typing import Tuple, Union

import torch
import torch.nn as nn

import kornia

__all__ = [
    "ConvSoftArgmax2d",
    "ConvSoftArgmax3d",
    "SpatialSoftArgmax2d",
    "ConvQuadInterp3d",
]


class ConvSoftArgmax2d(nn.Module):
    r"""Module that calculates soft argmax 2d per window.

    See :func:`~kornia.geometry.subpix.conv_soft_argmax2d` for details.
    """

    def __init__(
        self, kernel_size: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1), temperature: Union[torch.Tensor, float] = torch.tensor(1.0),
        normalized_coordinates: bool = True, eps: float = 1e-8, output_value: bool = False
    ) -> None:
        super(ConvSoftArgmax2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'(kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}, '
            f'temperature={self.temperature}, '
            f'normalized_coordinates={self.normalized_coordinates}, '
            f'eps={self.eps}, '
            f'output_value={self.output_value})'
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        return kornia.geometry.conv_soft_argmax2d(
            x, self.kernel_size, self.stride, self.padding, self.temperature, self.normalized_coordinates,
            self.eps, self.output_value
        )


class ConvSoftArgmax3d(nn.Module):
    r"""Module that calculates soft argmax 3d per window.

    See :func:`~kornia.geometry.subpix.conv_soft_argmax3d` for details.
    """

    def __init__(
        self, kernel_size: Tuple[int, int, int] = (3, 3, 3), stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1), temperature: Union[torch.Tensor, float] = torch.tensor(1.0),
        normalized_coordinates: bool = False, eps: float = 1e-8, output_value: bool = True,
        strict_maxima_bonus: float = 0.0
    ) -> None:
        super(ConvSoftArgmax3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value
        self.strict_maxima_bonus = strict_maxima_bonus
        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'(kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}, '
            f'temperature={self.temperature}, '
            f'normalized_coordinates={self.normalized_coordinates}, '
            f'eps={self.eps}, '
            f'strict_maxima_bonus={self.strict_maxima_bonus}, '
            f'output_value={self.output_value})')

    def forward(self, x: torch.Tensor):  # type: ignore
        return kornia.geometry.conv_soft_argmax3d(
            x, self.kernel_size, self.stride, self.padding, self.temperature, self.normalized_coordinates,
            self.eps, self.output_value, self.strict_maxima_bonus
        )


class SpatialSoftArgmax2d(nn.Module):
    r"""Module that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.geometry.subpix.spatial_soft_argmax2d` for details.
    """

    def __init__(
        self, temperature: torch.Tensor = torch.tensor(1.0), normalized_coordinates: bool = True, eps: float = 1e-8
    ) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.temperature: torch.Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates
        self.eps: float = eps

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'(temperature={self.temperature}, '
            f'normalized_coordinates={self.normalized_coordinates}, '
            f'eps={self.eps})'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.geometry.spatial_soft_argmax2d(input, self.temperature, self.normalized_coordinates, self.eps)


class ConvQuadInterp3d(nn.Module):
    r"""Module that calculates soft argmax 3d per window
    See :func:`~kornia.geometry.subpix.conv_quad_interp3d` for details.
    """

    def __init__(self, strict_maxima_bonus: float = 10.0, eps: float = 1e-7) -> None:
        super(ConvQuadInterp3d, self).__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(strict_maxima_bonus={self.strict_maxima_bonus})'

    def forward(self, x: torch.Tensor):  # type: ignore
        return kornia.geometry.conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps)
