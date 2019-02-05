from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def create_meshgrid(x):
    pos_y, pos_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, x.size(2), device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, x.size(3), device=x.device, dtype=x.dtype))
    return pos_y, pos_x


class SpatialSoftArgmax2d(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum(
            (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x: torch.Tensor = torch.sum(
            (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2


######################
# functional interface
######################


def spatial_soft_argmax2d(input: torch.Tensor) -> torch.Tensor:
    r"""Function that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :class:`torchgeometry.contrib.SpatialSoftArgmax2d` for details.
    """
    return SpatialSoftArgmax2d()(input)
