import torch
import torch.nn as nn

from kornia.utils import create_meshgrid


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
        normalized_coordinates (bool): wether to return the
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
    if not torch.is_tensor(input):
        raise TypeError("Input input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))
    # unpack shapes and create view from input tensor
    batch_size, channels, height, width = input.shape
    x: torch.Tensor = temperature * input.view(batch_size, channels, -1)

    # compute softmax with max substraction trick
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    exp_x_sum = torch.tensor(
        1.0) / (exp_x.sum(dim=-1, keepdim=True) + eps)

    # create coordinates grid
    grid: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates)
    grid = grid.to(input.device).to(input.dtype)

    pos_x: torch.Tensor = grid[..., 0].reshape(-1)
    pos_y: torch.Tensor = grid[..., 1].reshape(-1)

    # compute the expected coordinates
    expected_y: torch.Tensor = torch.sum(
        (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
    expected_x: torch.Tensor = torch.sum(
        (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)

    output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
    return output.view(batch_size, channels, 2)  # BxNx2


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
