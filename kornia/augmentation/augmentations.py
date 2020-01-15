from typing import Callable, Tuple, Union, cast

import torch
import torch.nn as nn

from kornia.augmentation.functional import *

UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


class AugmentationBase(nn.Module):
    def __init__(self, return_transform: bool = False):
        super(AugmentationBase, self).__init__()
        self.return_transform = return_transform

    def __repr__(self) -> str:
        repr = f"(return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward(self, input: UnionType) -> UnionType:  # type: ignore
        raise NotImplementedError("forward method is not implemented for this class.")

    @staticmethod
    def get_params():
        raise NotImplementedError("forward method is not implemented for this class.")


class RandomFlip(AugmentationBase):
    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomFlip, self).__init__(return_transform)
        self.p = p
        self.return_transform = return_transform
        self.flip_func = None

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    @staticmethod
    def get_params(batch_size: int, p: float = 0.5):
        return get_random_p_params(batch_size, p)

    def forward_flip(self, input: UnionType, flip_func: Callable, params: dict = None) -> UnionType:  # type: ignore

        if params is None:
            if isinstance(input, tuple):
                batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
            else:
                batch_size = input.shape[0] if len(input.shape) == 4 else 1
            params = RandomFlip.get_params(batch_size, self.p)

        if isinstance(input, tuple):

            inp: torch.Tensor = input[0]
            prev_trans: torch.Tensor = input[1]

            if self.return_transform:

                out = flip_func(inp, params, return_transform=True)
                img: torch.Tensor = out[0]
                trans_mat: torch.Tensor = out[1]

                return img, prev_trans @ trans_mat

            # https://mypy.readthedocs.io/en/latest/casts.html cast the return type to please mypy gods
            img = cast(torch.Tensor, flip_func(inp, params, return_transform=False))

            # Transform image but pass along the previous transformation
            return img, prev_trans

        return flip_func(input, params, return_transform=self.return_transform)

    def forward(self, input: UnionType) -> UnionType:  # type: ignore
        raise NotImplementedError("forward method is not implemented for this class.")


class RandomHorizontalFlip(RandomFlip):

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
        super(RandomHorizontalFlip, self).__init__(p=p, return_transform=return_transform)

    def forward(self, input: UnionType, params: dict = None) -> UnionType:  # type: ignore
        return self.forward_flip(input, apply_hflip, params)


class RandomVerticalFlip(RandomFlip):

    r"""Vertically flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Vertically flip the tensors and concatenate the corresponding transformation matrix to the
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
        >>> seq = nn.Sequential(kornia.augmentation.RandomVerticalFlip(p=1.0, return_transform=True))
        >>> seq(input)
        (tensor([[0., 1., 1.],
                 [0., 0., 0.],
                 [0., 0., 0.]]),
        tensor([[[1., 0., 0.],
                 [0., -1., 3.],
                 [0., 0., 1.]]]))

    """

    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomVerticalFlip, self).__init__(p=p, return_transform=return_transform)

    def forward(self, input: UnionType, params: dict = None) -> UnionType:  # type: ignore
        return self.forward_flip(input, apply_vflip, params)


class ColorJitter(AugmentationBase):

    r"""Change the brightness, contrast, saturation and hue randomly given tensor image or a batch of tensor images.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0
    """

    def __init__(self, brightness: FloatUnionType = 0., contrast: FloatUnionType = 0.,
                 saturation: FloatUnionType = 0., hue: FloatUnionType = 0., return_transform: bool = False) -> None:
        super(ColorJitter, self).__init__()
        self.brightness: FloatUnionType = brightness
        self.contrast: FloatUnionType = contrast
        self.saturation: FloatUnionType = saturation
        self.hue: FloatUnionType = hue
        self.return_transform: bool = return_transform

    def __repr__(self) -> str:
        repr = f"(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation},\
            hue={self.hue}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    @staticmethod
    def get_params(batch_size: int, brightness: FloatUnionType = 0., contrast: FloatUnionType = 0.,
                   saturation: FloatUnionType = 0., hue: FloatUnionType = 0.) -> dict:
        return get_color_jitter_params(batch_size, brightness, contrast, saturation, hue)

    def forward(self, input: UnionType, params: dict = None) -> UnionType:  # type: ignore
        if params is None:
            if isinstance(input, tuple):
                batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
            else:
                batch_size = input.shape[0] if len(input.shape) == 4 else 1
            params = ColorJitter.get_params(batch_size, self.brightness, self.contrast, self.saturation, self.hue)

        if isinstance(input, tuple):

            jittered: torch.Tensor = cast(
                torch.Tensor,
                apply_color_jitter(
                    input[0],
                    params=params,
                    return_transform=False))

            return jittered, input[1]

        return apply_color_jitter(input, params, self.return_transform)


class RandomGrayscale(AugmentationBase):
    r"""Random Grayscale transformation according to a probability p value

    Args:
        p (float): probability of the image to be transformed to grayscale. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomGrayscale, self).__init__(return_transform)
        self.p = p

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    @staticmethod
    def get_params(batch_size: int, p: float = .5) -> dict:
        return get_random_p_params(batch_size, p)

    def forward(self, input: UnionType, params: dict = None) -> UnionType:  # type: ignore
        if params is None:
            if isinstance(input, tuple):
                batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
            else:
                batch_size = input.shape[0] if len(input.shape) == 4 else 1
            params = RandomGrayscale.get_params(batch_size, self.p)

        if isinstance(input, tuple):

            output: torch.Tensor = cast(
                torch.Tensor,
                apply_grayscale(
                    input[0],
                    params=params,
                    return_transform=False))

            return output, input[1]

        return apply_grayscale(input, params=params, return_transform=self.return_transform)
