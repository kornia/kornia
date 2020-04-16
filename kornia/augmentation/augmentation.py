from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn
from torch.nn.functional import pad

from kornia.constants import Resample
from . import functional as F
from . import random_generator as rg
from .utils import (
    _adapted_uniform,
    _infer_batch_shape
)
from .types import (
    TupleFloat,
    UnionFloat,
    UnionType,
    FloatUnionType,
    BoarderUnionType
)


class AugmentationBase(nn.Module):
    def __init__(self, apply_fcn: Callable, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(AugmentationBase, self).__init__()
        self.return_transform = return_transform
        self._apply_fcn: Callable = apply_fcn
        self.same_on_batch = same_on_batch
        self._params: Dict[str, torch.Tensor] = {}

    def infer_batch_shape(self, input: UnionType) -> torch.Size:
        return _infer_batch_shape(input)

    def get_params(self, input_shape: torch.Size) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            self._params = self.get_params(batch_shape)
        else:
            self._params = params

        if isinstance(input, tuple):

            inp: torch.Tensor = input[0]
            prev_trans: torch.Tensor = input[1]

            if self.return_transform:

                out = self._apply_fcn(inp, self._params, return_transform=True)
                img: torch.Tensor = out[0]
                trans_mat: torch.Tensor = out[1]

                return img, prev_trans @ trans_mat

            # https://mypy.readthedocs.io/en/latest/casts.html cast the return type to please mypy gods
            img = cast(torch.Tensor, self._apply_fcn(inp, self._params, return_transform=False))

            # Transform image but pass along the previous transformation
            return img, prev_trans

        return self._apply_fcn(input, self._params, return_transform=self.return_transform)


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
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> input = torch.tensor([[[[0., 0., 0.],
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

    def __init__(self, p: float = 0.5, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(RandomHorizontalFlip, self).__init__(F.apply_hflip, return_transform, same_on_batch)
        self.p: float = p

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)


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
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> input = torch.tensor([[[[0., 0., 0.],
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

    def __init__(self, p: float = 0.5, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(RandomVerticalFlip, self).__init__(F.apply_vflip, return_transform, same_on_batch)
        self.p: float = p

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)


class ColorJitter(AugmentationBase):

    r"""Change the brightness, contrast, saturation and hue randomly given tensor image or a batch of tensor images.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def __init__(
        self, brightness: FloatUnionType = 0., contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
        hue: FloatUnionType = 0., return_transform: bool = False, same_on_batch: bool = False
    ) -> None:
        super(ColorJitter, self).__init__(F.apply_color_jitter, return_transform, same_on_batch)
        self.brightness: FloatUnionType = brightness
        self.contrast: FloatUnionType = contrast
        self.saturation: FloatUnionType = saturation
        self.hue: FloatUnionType = hue

    def __repr__(self) -> str:
        repr = f"(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation},\
            hue={self.hue}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_color_jitter_generator(
            batch_shape[0], self.brightness, self.contrast, self.saturation, self.hue, self.same_on_batch)


class RandomGrayscale(AugmentationBase):
    r"""Random Grayscale transformation according to a probability p value

    Args:
        p (float): probability of the image to be transformed to grayscale. Default value is 0.1
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def __init__(self, p: float = 0.1, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(RandomGrayscale, self).__init__(F.apply_grayscale, return_transform, same_on_batch)
        self.p = p

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)


class RandomErasing(AugmentationBase):
    r"""
    Erases a random selected rectangle for each image in the batch, putting the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [scale[0], scale[1]) and an aspect ratio sampled
    between [ratio[0], ratio[1])

    Args:
        p (float): probability that the random erasing operation will be performed.
        scale (Tuple[float, float]): range of proportion of erased area against input image.
        ratio (Tuple[float, float]): range of aspect ratio of erased area.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> inputs = torch.ones(1, 1, 3, 3)
        >>> rec_er = kornia.augmentation.RandomErasing((.4, .8), (.3, 1/.3))
        >>> rec_er(inputs)
        tensor([[[[1., 0., 0.],
                  [1., 0., 0.],
                  [1., 0., 0.]]]])
    """
    # Note: Extra params, inplace=False in Torchvision.
    def __init__(
            self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33), ratio: Tuple[float, float] = (0.3, 3.3),
            value: float = 0., return_transform: bool = False, same_on_batch: bool = False
    ) -> None:
        super(RandomErasing, self).__init__(F.apply_erase_rectangles, return_transform, same_on_batch)
        self.p = p
        self.scale: Tuple[float, float] = scale
        self.ratio: Tuple[float, float] = ratio
        self.value: float = value

    def __repr__(self) -> str:
        repr = f"(scale={self.scale}, ratio={self.ratio}, value={self.value}, "
        f"return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_rectangles_params_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], p=self.p, scale=self.scale, ratio=self.ratio,
            value=self.value, same_on_batch=self.same_on_batch)


class RandomPerspective(AugmentationBase):
    r"""Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation
                                 applied to each. Default: False.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def __init__(
        self, distortion_scale: float = 0.5, p: float = 0.5,
        interpolation: Union[str, int, Resample] = Resample.BILINEAR,
        return_transform: bool = False, same_on_batch: bool = False
    ) -> None:
        super(RandomPerspective, self).__init__(F.apply_perspective, return_transform, same_on_batch)
        self.p: float = p
        self.distortion_scale: float = distortion_scale
        self.interpolation: Resample = Resample.get(interpolation)

    def __repr__(self) -> str:
        repr = f"(distortion_scale={self.distortion_scale}, p={self.p}, interpolation={self.interpolation.name}, "
        f"return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_perspective_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], self.p, self.distortion_scale,
            self.interpolation, self.same_on_batch)


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
            resample (int, str or kornia.Resample): Default: Resample.BILINEAR
            return_transform (bool): if ``True`` return the matrix describing the transformation
                applied to each. Default: False.
            same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
        >>> input = torch.rand(2, 3, 224, 224)
        >>> my_fcn = kornia.augmentation.RandomAffine((-15., 20.), return_transform=True)
        >>> out, transform = my_fcn(input)  # 2x3x224x224 / 2x3x3
    """

    def __init__(
        self, degrees: UnionFloat, translate: Optional[TupleFloat] = None, scale: Optional[TupleFloat] = None,
        shear: Optional[UnionFloat] = None, resample: Union[str, int, Resample] = Resample.BILINEAR,
        return_transform: bool = False, same_on_batch: bool = False
    ) -> None:
        super(RandomAffine, self).__init__(F.apply_affine, return_transform, same_on_batch)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample: Resample = Resample.get(resample)

    def __repr__(self) -> str:
        repr = f"(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}, "
        f"resample={self.resample.name}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch}"
        return self.__class__.__name__ + repr

    # def get_params(self, batch_size: int, height: int, width: int) -> Dict[str, torch.Tensor]:
    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_affine_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], self.degrees, self.translate, self.scale, self.shear,
            self.resample, self.same_on_batch)


class CenterCrop(AugmentationBase):
    r"""Crops the given torch.Tensor at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        return_transform (bool): if ``True`` return the matrix describing the transformation
            applied to each. Default: False.
    """

    def __init__(self, size: Union[int, Tuple[int, int]], return_transform: bool = False) -> None:
        # same_on_batch is always True for CenterCrop
        super(CenterCrop, self).__init__(F.apply_crop, return_transform, True)
        self.size = size

    def __repr__(self) -> str:
        repr = f"(size={self.size}, return_transform={self.return_transform}"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        if isinstance(self.size, tuple):
            size_param = (self.size[0], self.size[1])
        elif isinstance(self.size, int):
            size_param = (self.size, self.size)
        else:
            raise Exception(f"Invalid size type. Expected (int, tuple(int, int). "
                            f"Got: {type(self.size)}.")
        return rg.center_crop_params_generator(batch_shape[0], batch_shape[-2], batch_shape[-1], size_param)


class RandomRotation(AugmentationBase):

    r"""Rotate a tensor image or a batch of tensor images a random amount of degrees.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will rotate the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        degrees (sequence or float or tensor): range of degrees to select from. If degrees is a number the
        range of degrees to select from will be (-degrees, +degrees)
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Examples:
    >>> input = torch.tensor([[[[10., 0., 0.],
                                [0., 4.5, 4.],
                                [0., 1., 1.]]]])
    >>> seq = nn.Sequential(kornia.augmentation.RandomRotation(degrees=90.0, return_transform=True))
    >>> seq(input)
    (tensor([[[0.0000e+00, 8.8409e-02, 9.8243e+00],
              [9.9131e-01, 4.5000e+00, 1.7524e-04],
              [9.9121e-01, 3.9735e+00, 3.5140e-02]]]),
    tensor([[[ 0.0088, -1.0000,  1.9911],
             [ 1.0000,  0.0088, -0.0088],
             [ 0.0000,  0.0000,  1.0000]]]))
    """
    # Note: Extra params, center=None, fill=0 in TorchVision
    def __init__(
        self, degrees: FloatUnionType, interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False
    ) -> None:
        super(RandomRotation, self).__init__(F.apply_rotation, return_transform, same_on_batch)
        self.degrees = degrees
        self.interpolation: Resample = Resample.get(interpolation)

    def __repr__(self) -> str:
        repr = f"(degrees={self.degrees}, interpolation={self.interpolation.name}, "
        f"return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_rotation_generator(batch_shape[0], self.degrees, self.interpolation, self.same_on_batch)


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
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def __init__(
        self, size: Tuple[int, int], padding: Optional[BoarderUnionType] = None, pad_if_needed: Optional[bool] = False,
        fill: int = 0, padding_mode: str = 'constant', return_transform: bool = False, same_on_batch: bool = False
    ) -> None:
        super(RandomCrop, self).__init__(F.apply_crop, return_transform, same_on_batch)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __repr__(self) -> str:
        repr = f"(crop_size={self.size}, padding={self.padding}, fill={self.fill}, "
        f"pad_if_needed={self.pad_if_needed}, padding_mode=${self.padding_mode}, "
        f"return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), self.size,
                                        same_on_batch=self.same_on_batch)

    def precrop_padding(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding is not None:
            if isinstance(self.padding, int):
                padding = [self.padding, self.padding, self.padding, self.padding]
            elif isinstance(self.padding, tuple) and len(self.padding) == 2:
                padding = [self.padding[1], self.padding[1], self.padding[0], self.padding[0]]
            elif isinstance(self.padding, tuple) and len(self.padding) == 4:
                padding = [self.padding[3], self.padding[2], self.padding[1], self.padding[0]]  # type:ignore
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-2] < self.size[0]:
            padding = [0, 0, (self.size[0] - input.shape[-2]), self.size[0] - input.shape[-2]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-1] < self.size[1]:
            padding = [self.size[1] - input.shape[-1], self.size[1] - input.shape[-1], 0, 0]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        return input
    
    def auto_padding(self, input: UnionType) -> UnionType:
        if isinstance(input, tuple):
            input = (self.precrop_padding(input[0]), self.precrop_padding(input[1]))
        else:
            input = self.precrop_padding(input)
        return input

    def forward(self, input: UnionType, params: Optional[Dict[str, torch.Tensor]] = None) -> UnionType:  # type: ignore
        input = self.auto_padding(input)
        return super().forward(input, params)


class RandomResizedCrop(AugmentationBase):
    r"""Random Crop on given size and resizing the cropped patch to another.
    Args:
        size (Tuple[int, int]): expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def __init__(
        self, size: Tuple[int, int], scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (1.75, 4. / 3.), 
        interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False
    ) -> None:
        super(RandomResizedCrop, self).__init__(F.apply_crop, return_transform, same_on_batch)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation: Resample = Resample.get(interpolation)

    def __repr__(self) -> str:
        repr = f"(size={self.size}, resize_to={self.scale}, resize_to={self.ratio}, "
        f"interpolation={self.interpolation.name}, return_transform={self.return_transform}, "
        f"same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def get_params(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        target_size = rg.random_crop_size_generator(self.size, self.scale, self.ratio)
        _target_size = (int(target_size[0].data.item()), int(target_size[1].data.item()))
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), _target_size,
                                        resize_to=self.size, same_on_batch=self.same_on_batch)
