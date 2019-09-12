import torch
import torch.nn as nn

from kornia.contrib.dsnt import (spatial_softmax_2d,
                                 spatial_softargmax_2d)


def spatial_soft_argmax2d(
        input: torch.Tensor,
        temperature: torch.Tensor = torch.tensor(1.0),
        normalized_coordinates: bool = True,
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
        eps (float): small value to avoid zero division. Default is 1e-8.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 10., 0.],
            [0., 0., 0.]]]])
        >>> coords = kornia.spatial_soft_argmax2d(input, False)
        tensor([[[1.0000, 1.0000]]])
    """
    input_soft: torch.Tensor = spatial_softmax_2d(input, temperature)
    output: torch.Tensor = spatial_softargmax_2d(input_soft,
                                                 normalized_coordinates)
    return output


class SpatialSoftArgmax2d(nn.Module):
    r"""Function that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :class:`~kornia.contrib.spatial_soft_argmax2d` for details.
    """

    def __init__(self,
                 temperature: torch.Tensor = torch.tensor(1.0),
                 normalized_coordinates: bool = True,
                 eps: float = 1e-8) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.temperature: torch.Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates
        self.eps: float = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return spatial_soft_argmax2d(input, self.temperature,
                                     self.normalized_coordinates, self.eps)
