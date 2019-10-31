import torch
import torch.nn as nn
from typing import Tuple
from kornia.geometry.transform.flips import hflip


class RandomHorizontalFlip(nn.Module):

    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p: float = p

    def __repr__(self):
        repr = "(p={0})".format(self.p)
        return self.__class__.__name__ + repr

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return random_hflip(input, self.p)


def random_hflip(input, p=0.5) -> Tuple[torch.Tensor, torch.Tensor]:

    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    Returns:
        torch.Tensor: The horizontally flipped input
        torch.Tensor: The applied transformation matrix
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number.")

    device: torch.device = input.device

    if len(input.shape) > 3:
        input = input.view((-1, input.shape[-2], input.shape[-1]))
        probs: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(0, 1)
    else:
        input = input.unsqueeze(0)
        probs = torch.empty(1, device=device).uniform_(0, 1)

    to_flip: torch.Tensor = probs < p

    if input[to_flip].nelement() != 0:

        flipped: torch.Tensor = hflip(input[to_flip])
        trans_mat: torch.Tensor = torch.zeros((3, 3))
        trans_mat[0][0] = -1
        trans_mat[1][1] = 1
        trans_mat[0][2] = input.shape[-2]

        trans_mat = trans_mat.unsqueeze(0).repeat(input.shape[0], 1, 1)
        trans_mat[~to_flip] = torch.eye(3)  # no transform on these inputs

        return flipped, trans_mat

    return input, torch.eye(3).unsqueeze(0).repeat(input.shape[0], 1, 1)
