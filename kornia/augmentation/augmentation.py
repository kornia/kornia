from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn

from . import functional as F
from . import param_gen as pg


TupleFloat = Tuple[float, float]
UnionFloat = Union[float, TupleFloat]
UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


class AugmentationBase(nn.Module):
    def __init__(self, apply_fcn: Callable, return_transform: bool = False) -> None:
        super(AugmentationBase, self).__init__()
        self.return_transform = return_transform
        self._apply_fcn: Callable = apply_fcn

    def infer_batch_size(self, input) -> int:
        if isinstance(input, tuple):
            batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
        else:
            batch_size = input.shape[0] if len(input.shape) == 4 else 1
        return batch_size

    def infer_image_shape(self, input: UnionType) -> Tuple[int, int]:
        if isinstance(input, tuple):
            data, _ = cast(Tuple, input)
        else:
            data = cast(torch.Tensor, input)
        return data.shape[-2:]

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
            params = self.get_params(batch_size, self.p)
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
            params = self.get_params(batch_size, self.p)
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


class RandomRectangleErasing(nn.Module):
    r"""
    Erases a random selected rectangle for each image in the batch, putting the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [erase_scale_range[0], erase_scale_range[1]) and an aspect ratio sampled
    between [aspect_ratio_range[0], aspect_ratio_range[1])

    Args:
        erase_scale_range (Tuple[float, float]): range of proportion of erased area against input image.
        aspect_ratio_range (Tuple[float, float]): range of aspect ratio of erased area.

    Examples:
        >>> inputs = torch.ones(1, 1, 3, 3)
        >>> rec_er = kornia.augmentation.RandomRectangleErasing((.4, .8), (.3, 1/.3))
        >>> rec_er(inputs)
        tensor([[[[1., 0., 0.],
                  [1., 0., 0.],
                  [1., 0., 0.]]]])
    """

    def __init__(
            self, erase_scale_range: Tuple[float, float], aspect_ratio_range: Tuple[float, float]
    ) -> None:
        super(RandomRectangleErasing, self).__init__()
        self.erase_scale_range: Tuple[float, float] = erase_scale_range
        self.aspect_ratio_range: Tuple[float, float] = aspect_ratio_range

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore
        return F.random_rectangle_erase(
            images,
            self.erase_scale_range,
            self.aspect_ratio_range
        )


class RandomPerspective(AugmentationBase):
    r"""Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.
        input tensor.
    """

    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomPerspective, self).__init__(F.apply_perspective, return_transform)
        self.p: float = p
        self.distortion_scale: float = distortion_scale
        self.return_transform: bool = return_transform

    def __repr__(self) -> str:
        repr = f"(distortion_scale={self.distortion_scale}, p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    @staticmethod
    def get_params(batch_size: int, height: int, width: int, p: float,
                   distortion_scale: float) -> Dict[str, torch.Tensor]:
        return pg._random_perspective_gen(batch_size, height, width, p, distortion_scale)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            height, width = self.infer_image_shape(input)
            batch_size: int = self.infer_batch_size(input)
            params = self.get_params(batch_size, height, width, self.p, self.distortion_scale)
        return super().forward(input, params)


class RandomAffine(AugmentationBase):
    r"""Random affine transformation of the image keeping center invariant.
        Args:
            degrees (float or tuple): Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to deactivate rotations.
            translate (tuple, optional): tuple of maximum absolute fraction for horizontal
                and vertical translations. For example translate=(a, b), then horizontal shift
                is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
                randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            shear (sequence or float, optional): Range of degrees to select from.
                If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
                will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
                range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
                a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
                Will not apply shear by default
            return_transform (bool): if ``True`` return the matrix describing the transformation
                applied to each. Default: False.
            mode (str): interpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'.
            padding_mode (str): padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'.
    """

    def __init__(self,
                 degrees: UnionFloat,
                 translate: Optional[TupleFloat] = None,
                 scale: Optional[TupleFloat] = None,
                 shear: Optional[UnionFloat] = None,
                 return_transform: bool = False) -> None:
        super(RandomAffine, self).__init__(F.apply_affine, return_transform)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.return_transform = return_transform

    @staticmethod
    def get_params(batch_size: int,
                   height: int,
                   width: int,
                   degrees: UnionFloat,
                   translate: Optional[TupleFloat] = None,
                   scale: Optional[TupleFloat] = None,
                   shear: Optional[UnionFloat] = None) -> Dict[str, torch.Tensor]:
        return pg._random_affine_gen(batch_size, height, width, degrees, translate, scale, shear)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            height, width = self.infer_image_shape(input)
            batch_size: int = self.infer_batch_size(input)
            params = self.get_params(batch_size, height, width, self.degrees, self.translate, self.scale, self.shear)
        return super().forward(input, params)
