from typing import Tuple, Union
import torch
import torch.nn as nn

from kornia.geometry.transform.flips import hflip


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class RandomHorizontalFlip(nn.Module):

    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transformation (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.
    """

    def __init__(self, p: float = 0.5, return_transformation: bool = False) -> None:
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        self.return_transformation = return_transformation

    def __repr__(self) -> str:
        repr = f"(p={self.p})"
        return self.__class__.__name__ + repr

    def forward(self, input: torch.Tensor) -> UnionType:  # type: ignore
        return random_hflip(input, self.p, self.return_transformation)


def random_hflip(input: torch.Tensor, p: float = 0.5, return_transformation: bool = False) -> UnionType:
    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transformation (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The horizontally flipped input
        torch.Tensor: The applied transformation matrix if return_transformation flag is set to ``True``
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transformation, bool):
        raise TypeError(f"The return_transformation flag must be a bool. Got {type(return_transformation)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))
    probs: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(0, 1)

    to_flip: torch.Tensor = probs < p
    flipped: torch.Tensor = input.clone()

    trans_mat: torch.Tensor = torch.eye(3, device=device, dtype=dtype).expand(input.shape[0], -1, -1)

    flipped[to_flip] = hflip(input[to_flip])
    flipped.squeeze_()

    if return_transformation:

        w: int = input.shape[-2]
        flip_mat: torch.Tensor = torch.tensor([[-1, 0, w],
                                               [0, 1, 0],
                                               [0, 0, 0]])

        trans_mat[to_flip] = flip_mat.to(device).to(dtype)

        return flipped, trans_mat

    return flipped
