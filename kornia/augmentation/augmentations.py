from typing import Tuple, Union, cast
import torch
import torch.nn as nn

from kornia.geometry.transform.flips import hflip

UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class RandomHorizontalFlip(nn.Module):

    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> seq = nn.Sequential(kornia.augmentation.RandomHorizontalFlip(p=1.0, return_transform=True),
                                kornia.augmentation.RandomHorizontalFlip(p=1.0, return_transform=True)
                               )
        >>> seq(input)
        (tensor([[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 1., 1.]]),
        tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]))

    """

    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        self.return_transform = return_transform

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward(self, input: UnionType) -> UnionType:  # type: ignore

        if isinstance(input, tuple):

            inp: torch.Tensor = input[0]
            prev_trans: torch.Tensor = input[1]

            if self.return_transform:

                out = random_hflip(inp, p=self.p, return_transform=True)
                img: torch.Tensor = out[0]
                trans_mat: torch.Tensor = out[1]

                return img, prev_trans @ trans_mat

            # https://mypy.readthedocs.io/en/latest/casts.html cast the return type to please mypy gods
            img = cast(torch.Tensor, random_hflip(inp, p=self.p, return_transform=False))

            # Transform image but pass along the previous transformation
            return img, prev_trans

        return random_hflip(input, p=self.p, return_transform=self.return_transform)


def random_hflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The horizontally flipped input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
                      is set to ``True``
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

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

    if return_transform:

        w: int = input.shape[-2]
        flip_mat: torch.Tensor = torch.tensor([[-1, 0, w],
                                               [0, 1, 0],
                                               [0, 0, 1]])

        trans_mat[to_flip] = flip_mat.to(device).to(dtype)

        return flipped, trans_mat

    return flipped
