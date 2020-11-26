from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn
from torch.nn.functional import pad

from kornia.constants import Resample, BorderType, SamplePadding
from kornia.augmentation import AugmentationBase2D
from . import functional as F
from . import random_generator as rg
from .utils import (
    _range_bound,
    _singular_range_check
)


class AugmentationBase(AugmentationBase2D):
    __doc__ = AugmentationBase2D.__doc__

    def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5,
                 keepdim: bool = False) -> None:
        super(AugmentationBase2D, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                                 keepdim=keepdim)
        raise DeprecationWarning(
            "`AugmentationBase` is deprecated. Please use `kornia.augmentation.AugmentationBase2D instead.`")


class RandomHorizontalFlip(AugmentationBase2D):

    r"""Applies a random horizontal flip to a tensor image or a batch of tensor images with a given probability.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> input = torch.tensor([[[[0., 0., 0.],
        ...                         [0., 0., 0.],
        ...                         [0., 1., 1.]]]])
        >>> seq = nn.Sequential(RandomHorizontalFlip(p=1.0, return_transform=True),
        ...                     RandomHorizontalFlip(p=1.0, return_transform=True))
        >>> seq(input)
        (tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 1., 1.]]]]), tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]))
    """

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_hflip_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_hflip(input)


class RandomVerticalFlip(AugmentationBase2D):

    r"""Applies a random vertical flip to a tensor image or a batch of tensor images with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> input = torch.tensor([[[[0., 0., 0.],
        ...                         [0., 0., 0.],
        ...                         [0., 1., 1.]]]])
        >>> seq = RandomVerticalFlip(p=1.0, return_transform=True)
        >>> seq(input)
        (tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]]), tensor([[[ 1.,  0.,  0.],
                 [ 0., -1.,  2.],
                 [ 0.,  0.,  1.]]]))

    """

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_vflip_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_vflip(input)


class ColorJitter(AugmentationBase2D):

    r"""Applies a random transformation to the brightness, contrast, saturation and hue of a tensor image.

    Args:
        p (float): probability of applying the transformation. Default value is 1.
        brightness (float or tuple): Default value is 0.
        contrast (float or tuple): Default value is 0.
        saturation (float or tuple): Default value is 0.
        hue (float or tuple): Default value is 0.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> aug(inputs)
        tensor([[[[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]]]])
    """

    def __init__(
        self, brightness: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
        contrast: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
        saturation: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
        hue: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
        return_transform: bool = False, same_on_batch: bool = False, p: float = 1.,
        keepdim: bool = False
    ) -> None:
        super(ColorJitter, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                          keepdim=keepdim)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self) -> str:
        repr = f"brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        brightness: torch.Tensor = _range_bound(
            self.brightness, 'brightness', center=1., bounds=(0, 2), device=self.device, dtype=self.dtype)
        contrast: torch.Tensor = _range_bound(
            self.contrast, 'contrast', center=1., device=self.device, dtype=self.dtype)
        saturation: torch.Tensor = _range_bound(
            self.saturation, 'saturation', center=1., device=self.device, dtype=self.dtype)
        hue: torch.Tensor = _range_bound(
            self.hue, 'hue', bounds=(-0.5, 0.5), device=self.device, dtype=self.dtype)
        return rg.random_color_jitter_generator(
            batch_shape[0], brightness, contrast, saturation, hue, self.same_on_batch,
            self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_color_jitter(input, params)


class RandomGrayscale(AugmentationBase2D):
    r"""Applies random transformation to Grayscale according to a probability p value.

    Args:
        p (float): probability of the image to be transformed to grayscale. Default value is 0.1.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn((1, 3, 3, 3))
        >>> rec_er = RandomGrayscale(p=1.0)
        >>> rec_er(inputs)
        tensor([[[[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]]]])
    """

    def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.1,
                 keepdim: bool = False) -> None:
        super(RandomGrayscale, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                              keepdim=keepdim)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_grayscale(input)


class RandomErasing(AugmentationBase2D):
    r"""Erases a random rectangle of a tensor image according to a probability p value.

    The operator removes image parts and fills them with zero values at a selected rectangle
    for each of the images in the batch.

    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [scale[0], scale[1]) and an aspect ratio sampled
    between [ratio[0], ratio[1])

    Args:
        p (float): probability that the random erasing operation will be performed. Default value is 0.5.
        scale (Tuple[float, float]): range of proportion of erased area against input image.
        ratio (Tuple[float, float]): range of aspect ratio of erased area.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 1, 3, 3)
        >>> rec_er = RandomErasing((.4, .8), (.3, 1/.3), p=0.5)
        >>> rec_er(inputs)
        tensor([[[[1., 0., 0.],
                  [1., 0., 0.],
                  [1., 0., 0.]]]])
    """

    # Note: Extra params, inplace=False in Torchvision.
    def __init__(
            self, scale: Union[torch.Tensor, Tuple[float, float]] = (0.02, 0.33),
            ratio: Union[torch.Tensor, Tuple[float, float]] = (0.3, 3.3),
            value: float = 0., return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5,
            keepdim: bool = False
    ) -> None:
        super(RandomErasing, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                            keepdim=keepdim)
        self.scale = cast(torch.Tensor, scale) if isinstance(scale, torch.Tensor) else torch.tensor(scale)
        self.ratio = cast(torch.Tensor, ratio) if isinstance(ratio, torch.Tensor) else torch.tensor(ratio)
        self.value: float = value

    def __repr__(self) -> str:
        repr = f"scale={self.scale}, ratio={self.ratio}, value={self.value}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_rectangles_params_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], scale=self.scale, ratio=self.ratio,
            value=self.value, same_on_batch=self.same_on_batch, device=self.device, dtype=self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_erase_rectangles(input, params)


class RandomPerspective(AugmentationBase2D):
    r"""Applies a random perspective transformation to an image tensor with a given probability.

    Args:
        p (float): probability of the image being perspectively transformed. Default value is 0.5.
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR.
        return_transform (bool): if ``True`` return the matrix describing the transformation
                                 applied to each. Default: False.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        align_corners(bool): interpolation flag. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs= torch.tensor([[[[1., 0., 0.],
        ...                         [0., 1., 0.],
        ...                         [0., 0., 1.]]]])
        >>> aug = RandomPerspective(0.5, p=0.5)
        >>> aug(inputs)
        tensor([[[[0.0000, 0.2289, 0.0000],
                  [0.0000, 0.4800, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """

    def __init__(
        self, distortion_scale: Union[torch.Tensor, float] = 0.5,
        interpolation: Optional[Union[str, int, Resample]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False,
        align_corners: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomPerspective, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                                keepdim=keepdim)
        self.distortion_scale = cast(torch.Tensor, distortion_scale) \
            if isinstance(distortion_scale, torch.Tensor) else torch.tensor(distortion_scale)
        self.resample: Resample
        if interpolation is not None:
            import warnings
            warnings.warn("interpolation is deprecated. Please use resample instead.", category=DeprecationWarning)
            self.resample = Resample.get(interpolation)
        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = (f"distortion_scale={self.distortion_scale}, interpolation={self.resample.name}, "
                f"align_corners={self.align_corners}")
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_perspective_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], self.distortion_scale, self.same_on_batch,
            self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_perspective_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_perspective(input, params, self.flags)


class RandomAffine(AugmentationBase2D):
    r"""Applies a random 2D affine transformation to a tensor image.

    The transformation is computed so that the image center is kept invariant.

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        degrees (float or tuple): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If (a, b, c, d), the scale is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d.
            Will keep original scale by default.
        shear (sequence or float, optional): Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be apllied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be apllied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
        resample (int, str or kornia.Resample): resample mode from "nearest" (0) or "bilinear" (1).
            Default: Resample.BILINEAR.
        padding_mode (int, str or kornia.SamplePadding): padding mode from "zeros" (0), "border" (1)
            or "refection" (2). Default: SamplePadding.ZEROS.
        return_transform (bool): if ``True`` return the matrix describing the transformation
            applied to each. Default: False.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        align_corners(bool): interpolation flag. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomAffine((-15., 20.), return_transform=True, p=1.)
        >>> aug(input)
        (tensor([[[[0.3961, 0.7310, 0.1574],
                  [0.1781, 0.3074, 0.5648],
                  [0.4804, 0.8379, 0.4234]]]]), tensor([[[ 0.9923, -0.1241,  0.1319],
                 [ 0.1241,  0.9923, -0.1164],
                 [ 0.0000,  0.0000,  1.0000]]]))
    """

    def __init__(
        self, degrees: Union[torch.Tensor, float, Tuple[float, float]],
        translate: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        scale: Optional[Union[torch.Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[torch.Tensor, float, Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomAffine, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                           keepdim=keepdim)
        degrees = cast(torch.Tensor, degrees) if isinstance(degrees, torch.Tensor) else torch.tensor(degrees)
        self.degrees = _range_bound(degrees, 'degrees', 0, (-360, 360))
        self.translate: Optional[torch.Tensor] = None
        if translate is not None:
            self.translate = _range_bound(translate, 'translate', bounds=(0, 1), check='singular')
        self.scale: Optional[torch.Tensor] = None
        if scale is not None:
            scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)
            if len(scale) == 2:
                self.scale = _range_bound(scale, 'scale', bounds=(0, float('inf')), check='singular')
            elif len(scale) == 4:
                self.scale = torch.cat([
                    _range_bound(scale[:2], 'scale_x', bounds=(0, float('inf')), check='singular'),
                    _range_bound(scale[2:], 'scale_y', bounds=(0, float('inf')), check='singular')
                ])
            else:
                raise ValueError("'scale' expected to be either 2 or 4 elements. Got {scale}")
        self.shear: Optional[torch.Tensor] = None
        if shear is not None:
            shear = shear if isinstance(shear, torch.Tensor) else torch.tensor(shear)
            self.shear = torch.stack([
                _range_bound(shear if shear.dim() == 0 else shear[:2], 'shear-x', 0, (-360, 360)),
                torch.tensor([0, 0], device=shear.device, dtype=shear.dtype) if shear.dim() == 0 or len(shear) == 2
                else _range_bound(shear[2:], 'shear-y', 0, (-360, 360))
            ])
        self.resample: Resample = Resample.get(resample)
        self.padding_mode: SamplePadding = SamplePadding.get(padding_mode)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            resample=torch.tensor(self.resample.value),
            padding_mode=torch.tensor(self.padding_mode.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = (f"degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}, "
                f"resample={self.resample.name}")
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_affine_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], self.degrees, self.translate, self.scale, self.shear,
            self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_affine_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_affine(input, params, self.flags)


class CenterCrop(AugmentationBase2D):
    r"""Crops a given image tensor at the center.

    Args:
        p (float): probability of applying the transformation for the whole batch. Default value is 1.
        size (Tuple[int, int] or int): Desired output size (out_h, out_w) of the crop.
            If integer,  out_h = out_w = size.
            If Tuple[int, int], out_h = size[0], out_w = size[1].
        return_transform (bool): if ``True`` return the matrix describing the transformation
            applied to each. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 4, 4)
        >>> inputs
        tensor([[[[-1.1258, -1.1524, -0.2506, -0.4339],
                  [ 0.8487,  0.6920, -0.3160, -2.1152],
                  [ 0.3223, -1.2633,  0.3500,  0.3081],
                  [ 0.1198,  1.2377,  1.1168, -0.2473]]]])
        >>> aug = CenterCrop(2, p=1.)
        >>> aug(inputs)
        tensor([[[[ 0.6920, -0.3160],
                  [-1.2633,  0.3500]]]])
    """

    def __init__(self, size: Union[int, Tuple[int, int]], align_corners: bool = True,
                 resample: Union[str, int, Resample] = Resample.BILINEAR.name,
                 return_transform: bool = False, p: float = 1., keepdim: bool = False) -> None:
        # same_on_batch is always True for CenterCrop
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super(CenterCrop, self).__init__(p=1., return_transform=return_transform, same_on_batch=True, p_batch=p,
                                         keepdim=keepdim)
        self.size = size
        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = f"size={self.size}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        if isinstance(self.size, tuple):
            size_param = (self.size[0], self.size[1])
        elif isinstance(self.size, int):
            size_param = (self.size, self.size)
        else:
            raise Exception(f"Invalid size type. Expected (int, tuple(int, int). "
                            f"Got: {type(self.size)}.")
        return rg.center_crop_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], size_param, self.device)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_crop_transformation(input, params, self.flags)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_crop(input, params, self.flags)


class RandomRotation(AugmentationBase2D):
    r"""Applies a random rotation to a tensor image or a batch of tensor images given an amount of degrees.

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        degrees (sequence or float or tensor): range of degrees to select from. If degrees is a number the
          range of degrees to select from will be (-degrees, +degrees).
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        align_corners(bool): interpolation flag. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.tensor([[1., 0., 0., 2.],
        ...                       [0., 0., 0., 0.],
        ...                       [0., 1., 2., 0.],
        ...                       [0., 0., 1., 2.]])
        >>> seq = RandomRotation(degrees=45.0, return_transform=True, p=1.)
        >>> seq(input)
        (tensor([[[[0.9824, 0.0088, 0.0000, 1.9649],
                  [0.0000, 0.0029, 0.0000, 0.0176],
                  [0.0029, 1.0000, 1.9883, 0.0000],
                  [0.0000, 0.0088, 1.0117, 1.9649]]]]), tensor([[[ 1.0000, -0.0059,  0.0088],
                 [ 0.0059,  1.0000, -0.0088],
                 [ 0.0000,  0.0000,  1.0000]]]))
    """
    # Note: Extra params, center=None, fill=0 in TorchVision

    def __init__(
        self, degrees: Union[torch.Tensor, float, Tuple[float, float], List[float]],
        interpolation: Optional[Union[str, int, Resample]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = True, p: float = 0.5,
        keepdim: bool = False
    ) -> None:
        super(RandomRotation, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                             keepdim=keepdim)
        degrees = cast(torch.Tensor, degrees) if isinstance(degrees, torch.Tensor) else torch.tensor(degrees)
        self.degrees = _range_bound(degrees, 'degrees', 0, (-360, 360))
        self.resample: Resample
        if interpolation is not None:
            import warnings
            warnings.warn("interpolation is deprecated. Please use resample instead.", category=DeprecationWarning)
            self.resample = Resample.get(interpolation)
        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = f"degrees={self.degrees}, interpolation={self.resample.name}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_rotation_generator(batch_shape[0], self.degrees, self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_rotate_tranformation(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_rotation(input, params, self.flags)


class RandomCrop(AugmentationBase2D):
    r"""Crops random patches of a tensor image on a given size.

    Args:
        p (float): probability of applying the transformation for the whole batch. Default value is 1.0.
        size (Tuple[int, int]): Desired output size (out_h, out_w) of the crop.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
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
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners(bool): interpolation flag. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 3, 3)
        >>> aug = RandomCrop((2, 2), p=1.)
        >>> aug(inputs)
        tensor([[[[-0.6562, -1.0009],
                  [ 0.2223, -0.5507]]]])
    """

    def __init__(
        self, size: Tuple[int, int], padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None,
        pad_if_needed: Optional[bool] = False, fill: int = 0, padding_mode: str = 'constant',
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = False, p: float = 1.0,
        keepdim: bool = False
    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super(RandomCrop, self).__init__(
            p=1., return_transform=return_transform, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = (f"crop_size={self.size}, padding={self.padding}, fill={self.fill}, pad_if_needed={self.pad_if_needed}, "
                f"padding_mode={self.padding_mode}, resample={self.resample.name}")
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), self.size,
                                        same_on_batch=self.same_on_batch, device=self.device, dtype=self.dtype)

    def precrop_padding(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding is not None:
            if isinstance(self.padding, int):
                self.padding = cast(int, self.padding)
                padding = [self.padding, self.padding, self.padding, self.padding]
            elif isinstance(self.padding, tuple) and len(self.padding) == 2:
                self.padding = cast(Tuple[int, int], self.padding)
                padding = [self.padding[1], self.padding[1], self.padding[0], self.padding[0]]
            elif isinstance(self.padding, tuple) and len(self.padding) == 4:
                self.padding = cast(Tuple[int, int, int, int], self.padding)
                padding = [self.padding[3], self.padding[2], self.padding[1], self.padding[0]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-2] < self.size[0]:
            padding = [0, 0, (self.size[0] - input.shape[-2]), self.size[0] - input.shape[-2]]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        if self.pad_if_needed and input.shape[-1] < self.size[1]:
            padding = [self.size[1] - input.shape[-1], self.size[1] - input.shape[-1], 0, 0]
            input = pad(input, padding, value=self.fill, mode=self.padding_mode)

        return input

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_crop_transformation(input, params, self.flags)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_crop(input, params, self.flags)

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                params: Optional[Dict[str, torch.Tensor]] = None, return_transform: Optional[bool] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if type(input) == tuple:
            input = (self.precrop_padding(input[0]), input[1])
        else:
            input = cast(torch.Tensor, input)
            input = self.precrop_padding(input)
        return super().forward(input, params, return_transform)


class RandomResizedCrop(AugmentationBase2D):
    r"""Crops random patches in an image tensor and resizes to a given size.

    Args:
        size (Tuple[int, int]): Desired output size (out_h, out_w) of each edge.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        scale: range of size of the origin size cropped.
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        align_corners(bool): interpolation flag. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Example:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.tensor([[[0., 1., 2.],
        ...                         [3., 4., 5.],
        ...                         [6., 7., 8.]]])
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1.)
        >>> aug(inputs)
        tensor([[[[1.2500, 1.7500, 1.5000],
                  [4.2500, 4.7500, 3.7500],
                  [7.2500, 7.7500, 6.0000]]]])
    """

    def __init__(
        self, size: Tuple[int, int], scale: Union[torch.Tensor, Tuple[float, float]] = (0.08, 1.0),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (3. / 4., 4. / 3.),
        interpolation: Optional[Union[str, int, Resample]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False,
        align_corners: bool = False, p: float = 1., keepdim: bool = False
    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens all the time.
        super(RandomResizedCrop, self).__init__(
            p=1., return_transform=return_transform, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self.size = size
        self.scale = cast(torch.Tensor, scale) if isinstance(scale, torch.Tensor) else torch.tensor(scale)
        self.ratio = cast(torch.Tensor, ratio) if isinstance(ratio, torch.Tensor) else torch.tensor(ratio)
        self.resample: Resample
        if interpolation is not None:
            import warnings
            warnings.warn("interpolation is deprecated. Please use resample instead.", category=DeprecationWarning)
            self.resample = Resample.get(interpolation)
        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = f"size={self.size}, scale={self.scale}, ratio={self.ratio}, interpolation={self.resample.name}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        target_size: torch.Tensor = rg.random_crop_size_generator(
            batch_shape[0], self.size, self.scale, self.ratio, self.same_on_batch, self.device, self.dtype)['size']
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), target_size,
                                        resize_to=self.size, same_on_batch=self.same_on_batch,
                                        device=self.device, dtype=self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_crop_transformation(input, params, self.flags)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_crop(input, params, self.flags)


class RandomMotionBlur(AugmentationBase2D):
    r"""Perform motion blur on 2D images (4D tensor).

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        kernel_size (int or Tuple[int, int]): motion kernel size (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate the value from the range batch-wisely.
        angle (float or Tuple[float, float]): angle of the motion blur in degrees (anti-clockwise rotation).
            If float, it will generate the value from (-angle, angle).
        direction (float or Tuple[float, float]): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.
        border_type (int, str or kornia.BorderType): the padding mode to be applied before convolving.
            CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3. Default: BorderType.CONSTANT.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.ones(1, 1, 5, 5)
        >>> motion_blur = RandomMotionBlur(3, 35., 0.5, p=1.)
        >>> motion_blur(input)
        tensor([[[[0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561]]]])
    """

    def __init__(
            self, kernel_size: Union[int, Tuple[int, int]],
            angle: Union[torch.Tensor, float, Tuple[float, float]],
            direction: Union[torch.Tensor, float, Tuple[float, float]],
            border_type: Union[int, str, BorderType] = BorderType.CONSTANT.name,
            return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomMotionBlur, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                               keepdim=keepdim)
        self.kernel_size: Union[int, Tuple[int, int]] = kernel_size

        angle = cast(torch.Tensor, angle) if isinstance(angle, torch.Tensor) else torch.tensor(angle)
        self.angle = _range_bound(angle, 'angle', center=0., bounds=(-360, 360))

        direction = \
            cast(torch.Tensor, direction) if isinstance(direction, torch.Tensor) else torch.tensor(direction)
        self.direction = _range_bound(direction, 'direction', center=0., bounds=(-1, 1))
        self.border_type = BorderType.get(border_type)
        self.flags: Dict[str, torch.Tensor] = {
            "border_type": torch.tensor(self.border_type.value)
        }

    def __repr__(self) -> str:
        repr = f"kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}, " +\
            f"border_type='{self.border_type.name.lower()}'"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_motion_blur_generator(
            batch_shape[0], self.kernel_size, self.angle, self.direction, self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_motion_blur(input, params, self.flags)


class RandomSolarize(AugmentationBase2D):
    r"""Solarize given tensor image or a batch of tensor images randomly.

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        thresholds (float or tuple): Default value is 0.1.
            If float x, threshold will be generated from (0.5 - x, 0.5 + x).
            If tuple (x, y), threshold will be generated from (x, y).
        additions (float or tuple): Default value is 0.1.
            If float x, addition will be generated from (-x, x).
            If tuple (x, y), addition will be generated from (x, y).
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> solarize = RandomSolarize(0.1, 0.1, p=1.)
        >>> solarize(input)
        tensor([[[[0.4132, 0.1412, 0.1790, 0.2226, 0.3980],
                  [0.2754, 0.4194, 0.0130, 0.4538, 0.2771],
                  [0.4394, 0.4923, 0.1129, 0.2594, 0.3844],
                  [0.3909, 0.2118, 0.1094, 0.2516, 0.3728],
                  [0.2278, 0.0000, 0.4876, 0.0353, 0.5100]]]])
    """

    def __init__(
        self, thresholds: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.1,
        additions: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.1,
        same_on_batch: bool = False, return_transform: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomSolarize, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch)

        thresholds = \
            cast(torch.Tensor, thresholds) if isinstance(thresholds, torch.Tensor) else torch.tensor(thresholds)
        self.thresholds = _range_bound(thresholds, 'thresholds', center=0.5, bounds=(0., 1.))

        additions = \
            cast(torch.Tensor, additions) if isinstance(additions, torch.Tensor) else torch.tensor(additions)
        self.additions = _range_bound(additions, 'additions', bounds=(-0.5, 0.5))

    def __repr__(self) -> str:
        repr = f"thresholds={self.thresholds}, additions={self.additions}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_solarize_generator(batch_shape[0], self.thresholds, self.additions, self.same_on_batch,
                                            self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_solarize(input, params)


class RandomPosterize(AugmentationBase2D):
    r"""Posterize given tensor image or a batch of tensor images randomly.

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        bits (int or tuple): Integer that ranged from (0, 8], in which 0 gives black image and 8 gives the original.
            If int x, bits will be generated from (x, 8).
            If tuple (x, y), bits will be generated from (x, y).
            Default value is 3.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> posterize = RandomPosterize(3, p=1.)
        >>> posterize(input)
        tensor([[[[0.4706, 0.7529, 0.0627, 0.1255, 0.2824],
                  [0.6275, 0.4706, 0.8784, 0.4392, 0.6275],
                  [0.3451, 0.3765, 0.0000, 0.1569, 0.2824],
                  [0.5020, 0.6902, 0.7843, 0.1569, 0.2510],
                  [0.6588, 0.9098, 0.3765, 0.8471, 0.4078]]]])
    """

    def __init__(
        self, bits: Union[int, Tuple[int, int], torch.Tensor] = 3,
        same_on_batch: bool = False, return_transform: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomPosterize, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                              keepdim=keepdim)
        bits = cast(torch.Tensor, bits) if isinstance(bits, torch.Tensor) else torch.tensor(bits)
        if len(bits.size()) == 0:
            self.bits = torch.tensor([bits, torch.tensor(8)], dtype=torch.float32)
        elif len(bits.size()) == 1 and bits.size(0) == 2:
            self.bits = torch.tensor([bits[0], bits[1]], dtype=torch.float32)
        else:
            raise ValueError(f"'bits' shall be either a scalar or a length 2 tensor. Got {bits}.")

    def __repr__(self) -> str:
        repr = f"(bits={self.bits}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_posterize_generator(batch_shape[0], self.bits, self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_posterize(input, params)


class RandomSharpness(AugmentationBase2D):
    r"""Sharpen given tensor image or a batch of tensor images randomly.

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        sharpness (float or tuple): factor of sharpness strength. Must be above 0. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> sharpness = RandomSharpness(1., p=1.)
        >>> sharpness(input)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4810, 0.7367, 0.4177, 0.6323],
                  [0.3489, 0.4428, 0.1562, 0.2443, 0.2939],
                  [0.5185, 0.6462, 0.7050, 0.2288, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])
    """

    def __init__(
        self, sharpness: Union[torch.Tensor, float, Tuple[float, float], torch.Tensor] = 0.5,
        same_on_batch: bool = False, return_transform: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomSharpness, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                              keepdim=keepdim)
        sharpness = cast(torch.Tensor, sharpness) if isinstance(sharpness, torch.Tensor) else torch.tensor(sharpness)
        if sharpness.dim() == 0:
            self.sharpness = torch.tensor([0, sharpness], dtype=torch.float32)
        elif sharpness.dim() == 1 and sharpness.size(0) == 2:
            self.sharpness = torch.tensor([sharpness[0], sharpness[1]], dtype=torch.float32)
        else:
            raise ValueError(f"'sharpness' must be a scalar or a length 2 tensor. Got {sharpness}.")

    def __repr__(self) -> str:
        repr = f"sharpness={self.sharpness}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_sharpness_generator(batch_shape[0], self.sharpness, self.same_on_batch,
                                             self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_sharpness(input, params)


class RandomEqualize(AugmentationBase2D):
    r"""Equalize given tensor image or a batch of tensor images randomly.

    Args:
        p (float): Probability to equalize an image. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> equalize = RandomEqualize(p=1.)
        >>> equalize(input)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])
    """

    def __init__(
        self, same_on_batch: bool = False, return_transform: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomEqualize, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                             keepdim=keepdim)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_equalize(input, params)
