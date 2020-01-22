from typing import Callable, Tuple, Union, List, Optional, Dict, cast
import random
import math

import torch
import torch.nn as nn
from torch.nn.functional import pad

from . import functional as F
from . import param_gen as pg


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]
Int2DBoarderUnionType = Union[int, Tuple[int, int], Tuple[int, int, int, int]]


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


class RandomCrop(AugmentationBase):
    r"""Random Crop on given size.

    Args:
        size (tuple): Desired output size of the crop, like (h, w).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    def __init__(self, size: Tuple[int, int], padding: Optional[Int2DBoarderUnionType] = None,
                 pad_if_needed: Optional[bool]=False, fill=0, padding_mode='constant',
                 return_transform: bool = False) -> None:
        super(RandomCrop, self).__init__(F.apply_crop, return_transform)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __repr__(self) -> str:
        repr = f"(crop_size={self.size}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    @staticmethod
    def get_params(batch_size: int, input_size: Tuple[int, int], size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        return pg._random_crop_gen(batch_size, input_size, size)

    def precrop_padding(self, input: torch.Tensor):

        if self.padding is not None:
            if isinstance(self.padding, int):
                padding = [self.padding, self.padding, self.padding, self.padding]
            elif isinstance(self.padding, tuple) and len(self.padding) == 2:
                padding = [self.padding[1], self.padding[1], self.padding[0], self.padding[0]]
            elif isinstance(self.padding, tuple) and len(self.padding) == 4:
                padding = [self.padding[3], self.padding[2], self.padding[1], self.padding[0]]  # type:ignore
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-2] < self.size[1]:
            padding = [self.size[1] - input.shape[-2], self.size[1] - input.shape[-2], 0, 0]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-1] < self.size[0]:
            padding = [0, 0, self.size[0] - input.shape[-1], self.size[0] - input.shape[-1]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        return input

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if isinstance(input, tuple):
            batch_shape = input[0].shape
            input = (self.precrop_padding(input[0]), self.precrop_padding(input[1]))
        else:
            batch_shape = input.shape
            input = self.precrop_padding(input)

        if params is None:
            batch_size = self.infer_batch_size(input)
            params = RandomCrop.get_params(batch_size, batch_shape[-2:], self.size)  # type: ignore
        return super().forward(input, params)


class RandomResizedCrop(AugmentationBase):
    r"""Random Crop on given size and resizing the cropped patch to another.

    Args:
        size (Tuple[int, int]): expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    def __init__(self, size: Tuple[int, int], scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=None, return_transform: bool = False) -> None:
        super(RandomResizedCrop, self).__init__(F.apply_crop, return_transform)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        if interpolation is not None:
            raise ValueError("Interpolation has not been implemented. Please set to None")

    def __repr__(self) -> str:
        repr = f"(size={self.size}, resize_to={self.resize_to}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    @staticmethod
    def get_params(batch_size: int, input_size: Tuple[int, int], size: Tuple[int, int],
                   scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)) -> Dict[str, torch.Tensor]:
        target_size = RandomResizedCrop.get_aspected_size(size, scale, ratio)
        # TODO: scale and aspect ratio were fixed for one batch for now. Need to be separated.
        return pg._random_crop_gen(batch_size, input_size, target_size, resize_to=size)

    @staticmethod
    def get_aspected_size(size, scale, ratio):
        for attempt in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w < size[0] and 0 < h < size[1]:
                return (w, h)

        # Fallback to center crop
        in_ratio = float(size[0]) / float(size[1])
        if (in_ratio < min(ratio)):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = size[0]
            h = size[1]
        return (w, h)

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            batch_size = self.infer_batch_size(input)
            if isinstance(input, tuple):
                batch_shape = input[0].shape
            else:
                batch_shape = input.shape
            params = RandomResizedCrop.get_params(
                batch_size, batch_shape[-2:], self.size, self.resize_to)  # type: ignore
        return super().forward(input, params)
