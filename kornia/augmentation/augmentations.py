from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn

import kornia.augmentation.functional as F
import kornia.augmentation.param_gen as pg

UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


class AugmentationBase(nn.Module):
    def __init__(self, apply_fcn: Callable, return_transform: bool = False) -> None:
        super(AugmentationBase, self).__init__()
        self.return_transform = return_transform
        self._apply_fcn: Callable = apply_fcn

    def infer_batch_size(self, input):
        if isinstance(input, tuple):
            batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
        else:
            batch_size = input.shape[0] if len(input.shape) == 4 else 1
        return batch_size

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if isinstance(input, tuple):

            inp: torch.Tensor = input[0]
            prev_trans: torch.Tensor = input[1]

            if self.return_transform:

                out = self._apply_fcn(inp, params, return_transform=True)
                img: torch.Tensor = out[0]
                trans_mat: torch.Tensor = out[1]

                return img, prev_trans @ trans_mat

            # https://mypy.readthedocs.io/en/latest/casts.html cast the return type to please mypy gods
            img = cast(torch.Tensor, self._apply_fcn(inp, params, return_transform=False))

            # Transform image but pass along the previous transformation
            return img, prev_trans

        return self._apply_fcn(input, params, return_transform=self.return_transform)


class RandomHorizontalFlip(AugmentationBase):

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
        super(RandomHorizontalFlip, self).__init__(F.apply_hflip, return_transform)
        self.p: float = p

    @staticmethod
    def get_params(batch_size: int, p: float) -> Dict[str, torch.Tensor]:
        return pg._random_prob_gen(batch_size, p)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            batch_size = self.infer_batch_size(input)
            params: Dict[str, torch.Tensor] = self.get_params(batch_size, self.p)
        return super().forward(input, params)


class RandomVerticalFlip(AugmentationBase):

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
        super(RandomVerticalFlip, self).__init__(F.apply_vflip, return_transform)
        self.p: float = p

    @staticmethod
    def get_params(batch_size: int, p: float) -> Dict[str, torch.Tensor]:
        return pg._random_prob_gen(batch_size, p)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            batch_size = self.infer_batch_size(input)
            params: Dict[str, torch.Tensor] = self.get_params(batch_size, self.p)
        return super().forward(input, params)


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
        super(ColorJitter, self).__init__(F.apply_color_jitter, return_transform)
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
                   saturation: FloatUnionType = 0., hue: FloatUnionType = 0.) -> Dict[str, torch.Tensor]:
        return pg._random_color_jitter_gen(batch_size, brightness, contrast, saturation, hue)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            batch_size = self.infer_batch_size(input)
            params = ColorJitter.get_params(batch_size, self.brightness, self.contrast, self.saturation, self.hue)
        return super().forward(input, params)


class RandomGrayscale(AugmentationBase):
    r"""Random Grayscale transformation according to a probability p value

    Args:
        p (float): probability of the image to be transformed to grayscale. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomGrayscale, self).__init__(F.apply_grayscale, return_transform)
        self.p = p

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    @staticmethod
    def get_params(batch_size: int, p: float = .5) -> Dict[str, torch.Tensor]:
        return pg._random_prob_gen(batch_size, p)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            batch_size = self.infer_batch_size(input)
            params = RandomGrayscale.get_params(batch_size, self.p)
        return super().forward(input, params)
