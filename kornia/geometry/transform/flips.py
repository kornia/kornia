import torch
import torch.nn as nn
from typing import Union, Tuple, List


class Flip(nn.Module):
    r"""Reverse the order of a tensor along a given axis

    Args:
        input (torch.Tensor): input tensor
        dims (list or tuple): axis to perform the flip on

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> vflip = kornia.flip(input, -2)
        tensor([[[0, 1, 1],
                 [0, 0, 0],
                 [0, 0, 0]]])
        >>> rot180 = kornia.flip(input, [-2, -1])
        tensor([[[1, 1, 0],
                 [0, 0, 0],
                 [0, 0, 0]]])

    """

    def __init__(self, dims: Union[Tuple[int, ...], List[int], int]) -> None:

        super(Flip, self).__init__()

        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return flip(input, self.dims)

    def __repr__(self):
        repr = "(axis={0})".format(self.dims)
        return self.__class__.__name__ + repr


def flip(input: torch.Tensor, dims: Union[Tuple[int, ...], List[int], int]) -> torch.Tensor:
    r"""Reverse the order of a tensor along a given axis

    Args:
        input (torch.Tensor): input tensor
        dims (list or tuple): axis to perform the flip on

        return torch.flip(input, dims])
    """

    # Allow ints
    if isinstance(dims, int):
        dims = [dims]

    return torch.flip(input, dims)


# def rot180(input: torch.Tensor) -> torch.Tensor:
# return torch.flip(input, [-2, -1])
