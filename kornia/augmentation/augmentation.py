from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn.functional import pad

from kornia.constants import Resample, BorderType, SamplePadding, pi
from kornia.augmentation import AugmentationBase2D
from kornia.filters import gaussian_blur2d, motion_blur
from kornia.geometry import (
    affine,
    bbox_generator,
    bbox_to_mask,
    crop_by_boxes,
    crop_by_transform_mat,
    deg2rad,
    get_perspective_transform,
    get_affine_matrix2d,
    hflip,
    vflip,
    rotate,
    warp_affine,
    warp_perspective,
    resize,
)
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_tensor_center
from kornia.color import rgb_to_grayscale
from kornia.enhance import (
    equalize,
    posterize,
    solarize,
    sharpness,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    adjust_gamma,
    Invert,
)
from kornia.utils import _extract_device_dtype
from kornia.enhance.normalize import normalize, denormalize
from kornia.enhance import Invert

from . import random_generator as rg
from .utils import (
    _range_bound,
    _transform_input
)


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

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        w: int = input.shape[-1]
        flip_mat: torch.Tensor = torch.tensor([[-1, 0, w - 1],
                                               [0, 1, 0],
                                               [0, 0, 1]], device=input.device, dtype=input.dtype)

        return flip_mat.repeat(input.size(0), 1, 1)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return hflip(input)


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

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        h: int = input.shape[-2]
        flip_mat: torch.Tensor = torch.tensor([[1, 0, 0],
                                               [0, -1, h - 1],
                                               [0, 0, 1]], device=input.device, dtype=input.dtype)

        return flip_mat.repeat(input.size(0), 1, 1)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return vflip(input)


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
        self._device, self._dtype = _extract_device_dtype([brightness, contrast, hue, saturation])
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self) -> str:
        repr = f"brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        brightness: torch.Tensor = _range_bound(
            self.brightness, 'brightness', center=1., bounds=(0, 2), device=self._device, dtype=self._dtype)
        contrast: torch.Tensor = _range_bound(
            self.contrast, 'contrast', center=1., device=self._device, dtype=self._dtype)
        saturation: torch.Tensor = _range_bound(
            self.saturation, 'saturation', center=1., device=self._device, dtype=self._dtype)
        hue: torch.Tensor = _range_bound(
            self.hue, 'hue', bounds=(-0.5, 0.5), device=self._device, dtype=self._dtype)
        return rg.random_color_jitter_generator(
            batch_shape[0], brightness, contrast, saturation, hue, self.same_on_batch,
            self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        transforms = [
            lambda img: adjust_brightness(img, params['brightness_factor'] - 1),
            lambda img: adjust_contrast(img, params['contrast_factor']),
            lambda img: adjust_saturation(img, params['saturation_factor']),
            lambda img: adjust_hue(img, params['hue_factor'] * 2 * pi)
        ]

        jittered = input
        for idx in params['order'].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered


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

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Make sure it returns (*, 3, H, W)
        grayscale = torch.ones_like(input)
        grayscale[:] = rgb_to_grayscale(input)
        return grayscale


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
        self._device, self._dtype = _extract_device_dtype([scale, ratio])
        self.scale = scale
        self.ratio = ratio
        self.value: float = float(value)

    def __repr__(self) -> str:
        repr = f"scale={self.scale}, ratio={self.ratio}, value={self.value}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        scale = torch.as_tensor(self.scale, device=self._device, dtype=self._dtype)
        ratio = torch.as_tensor(self.ratio, device=self._device, dtype=self._dtype)
        return rg.random_rectangles_params_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], scale=scale, ratio=ratio,
            value=self.value, same_on_batch=self.same_on_batch, device=self.device, dtype=self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, c, h, w = input.size()
        values = params['values'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, *input.shape[1:]).to(input)

        bboxes = bbox_generator(params['xs'], params['ys'], params['widths'], params['heights'])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        transformed = torch.where(mask == 1., values, input)
        return transformed


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
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False,
        align_corners: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomPerspective, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                                keepdim=keepdim)
        self._device, self._dtype = _extract_device_dtype([distortion_scale])
        self.distortion_scale = distortion_scale
        self.resample: Resample = Resample.get(resample)
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
        distortion_scale = torch.as_tensor(self.distortion_scale, device=self._device, dtype=self._dtype)
        return rg.random_perspective_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], distortion_scale, self.same_on_batch,
            self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return get_perspective_transform(
            params['start_points'].to(input), params['end_points'].to(input))

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return warp_perspective(
            input, transform, (height, width),
            mode=self.resample.name.lower(), align_corners=self.align_corners)


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
        self._device, self._dtype = _extract_device_dtype([degrees, translate, scale, shear])
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
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
        degrees = _range_bound(
            self.degrees, 'degrees', 0, (-360, 360), device=self._device, dtype=self._dtype)
        translate: Optional[torch.Tensor] = None
        scale: Optional[torch.Tensor] = None
        shear: Optional[torch.Tensor] = None

        if self.translate is not None:
            translate = _range_bound(
                self.translate, 'translate', bounds=(0, 1), check='singular', device=self._device, dtype=self._dtype)
        if self.scale is not None:
            scale = torch.as_tensor(self.scale, device=self._device, dtype=self._dtype)
            if len(scale) == 2:
                scale = _range_bound(
                    scale, 'scale', bounds=(0, float('inf')), check='singular', device=self._device, dtype=self._dtype)
            elif len(scale) == 4:
                scale = torch.cat([
                    _range_bound(scale[:2], 'scale_x', bounds=(0, float('inf')), check='singular',
                                 device=self._device, dtype=self._dtype),
                    _range_bound(scale[2:], 'scale_y', bounds=(0, float('inf')), check='singular',
                                 device=self._device, dtype=self._dtype)
                ])
            else:
                raise ValueError(f"'scale' expected to be either 2 or 4 elements. Got {scale}")
        if self.shear is not None:
            shear = torch.as_tensor(self.shear, device=self._device, dtype=self._dtype)
            shear = torch.stack([
                _range_bound(shear if shear.dim() == 0 else shear[:2], 'shear-x', 0, (-360, 360),
                             device=self._device, dtype=self._dtype),
                torch.tensor([0, 0], device=self._device, dtype=self._dtype) if shear.dim() == 0 or len(shear) == 2
                else _range_bound(shear[2:], 'shear-y', 0, (-360, 360), device=self._device, dtype=self._dtype)
            ])
        return rg.random_affine_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], degrees, translate, scale, shear,
            self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return get_affine_matrix2d(
            params['translations'], params['center'], params['scale'], params['angle'],
            deg2rad(params['sx']), deg2rad(params['sy'])
        ).type_as(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return warp_affine(
            input, transform[:, :2, :], (height, width), self.resample.name.lower(),
            align_corners=self.align_corners, padding_mode=self.padding_mode.name.lower())


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
        cropping_mode (str): The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
                             on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                             to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                             differentiability. Default: `slice`.

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

    def __init__(
            self,
            size: Union[int, Tuple[int, int]],
            align_corners: bool = True,
            resample: Union[str, int, Resample] = Resample.BILINEAR.name,
            return_transform: bool = False,
            p: float = 1.,
            keepdim: bool = False,
            cropping_mode: str = 'slice',
    ) -> None:
        # same_on_batch is always True for CenterCrop
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super(CenterCrop, self).__init__(p=1., return_transform=return_transform, same_on_batch=True, p_batch=p,
                                         keepdim=keepdim)
        if isinstance(size, tuple):
            self.size = (size[0], size[1])
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise Exception(f"Invalid size type. Expected (int, tuple(int, int). "
                            f"Got: {type(size)}.")
        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )
        self.cropping_mode = cropping_mode

    def __repr__(self) -> str:
        repr = f"size={self.size}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.center_crop_generator(
            batch_shape[0], batch_shape[-2], batch_shape[-1], self.size, self.device)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        transform: torch.Tensor = get_perspective_transform(
            params['src'].to(input), params['dst'].to(input))
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.cropping_mode == 'resample':  # uses bilinear interpolation to crop
            transform = cast(torch.Tensor, transform)
            return crop_by_transform_mat(
                input, transform[:, :2, :], self.size, self.resample.name.lower(), 'zeros', self.align_corners)
        elif self.cropping_mode == 'slice':   # uses advanced slicing to crop
            # TODO: implement as separated function `crop_and_resize_iterative`
            B, C, _, _ = input.shape
            H, W = self.size
            out = torch.empty(B, C, H, W, device=input.device, dtype=input.dtype)
            for i in range(B):
                x1 = int(params['src'][i, 0, 0])
                x2 = int(params['src'][i, 1, 0]) + 1
                y1 = int(params['src'][i, 0, 1])
                y2 = int(params['src'][i, 3, 1]) + 1
                out[i] = input[i:i + 1, :, y1:y2, x1:x2]
            return out
        else:
            raise NotImplementedError(f"Not supported type: {self.cropping_mode}.")


class RandomRotation(AugmentationBase2D):
    r"""Applies a random rotation to a tensor image or a batch of tensor images given an amount of degrees.

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        degrees (sequence or float or tensor): range of degrees to select from. If degrees is a number the
          range of degrees to select from will be (-degrees, +degrees).
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
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = True, p: float = 0.5,
        keepdim: bool = False
    ) -> None:
        super(RandomRotation, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                             keepdim=keepdim)
        self._device, self._dtype = _extract_device_dtype([degrees])
        self.degrees = degrees
        self.resample: Resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = f"degrees={self.degrees}, interpolation={self.resample.name}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        degrees = _range_bound(self.degrees, 'degrees', 0, (-360, 360), device=self._device, dtype=self._dtype)
        return rg.random_rotation_generator(batch_shape[0], degrees, self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: Update to use `get_rotation_matrix2d`
        angles: torch.Tensor = params["degrees"].to(input)

        center: torch.Tensor = _compute_tensor_center(input)
        rotation_mat: torch.Tensor = _compute_rotation_matrix(angles, center.expand(angles.shape[0], -1))

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]

        return trans_mat

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        transform = cast(torch.Tensor, transform)
        return affine(input, transform[..., :2, :3], self.resample.name.lower(), 'zeros', self.align_corners)


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
        align_corners(bool): interpolation flag. Default: True.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
        cropping_mode (str): The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
                             on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                             to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                             differentiability. Default: `slice`.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> inputs = torch.arange(1*1*3*3.).view(1, 1, 3, 3)
        >>> aug = RandomCrop((2, 2), p=1.)
        >>> aug(inputs)
        tensor([[[[3., 4.],
                  [6., 7.]]]])
    """

    def __init__(
        self,
        size: Tuple[int, int],
        padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None,
        pad_if_needed: Optional[bool] = False,
        fill: int = 0,
        padding_mode: str = 'constant',
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = 'slice',
    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super(RandomCrop, self).__init__(
            p=1., return_transform=return_transform, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.resample: Resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )
        self.cropping_mode = cropping_mode

    def __repr__(self) -> str:
        repr = (f"crop_size={self.size}, padding={self.padding}, fill={self.fill}, pad_if_needed={self.pad_if_needed}, "
                f"padding_mode={self.padding_mode}, resample={self.resample.name}")
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), self.size,
                                        same_on_batch=self.same_on_batch, device=self.device, dtype=self.dtype)

    def precrop_padding(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 4, f"Expected BCHW. Got {input.shape}"
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
        transform: torch.Tensor = get_perspective_transform(
            params['src'].to(input), params['dst'].to(input))
        return transform

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.cropping_mode == 'resample':  # uses bilinear interpolation to crop
            transform = cast(torch.Tensor, transform)
            return crop_by_transform_mat(
                input, transform, self.size, mode=self.resample.name.lower(),
                padding_mode='zeros', align_corners=self.align_corners)
        elif self.cropping_mode == 'slice':   # uses advanced slicing to crop
            B, C, _, _ = input.shape
            out = torch.empty(B, C, *self.size, device=input.device, dtype=input.dtype)
            for i in range(B):
                x1 = int(params['src'][i, 0, 0])
                x2 = int(params['src'][i, 1, 0]) + 1
                y1 = int(params['src'][i, 0, 1])
                y2 = int(params['src'][i, 3, 1]) + 1
                out[i] = input[i:i + 1, :, y1:y2, x1:x2]
            return out
        else:
            raise NotImplementedError(f"Not supported type: {self.flags['mode']}.")

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                params: Optional[Dict[str, torch.Tensor]] = None, return_transform: Optional[bool] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(input, (tuple, list)):
            input_temp = _transform_input(input[0])
            _input = (self.precrop_padding(input_temp), input[1])
        else:
            input = cast(torch.Tensor, input)  # TODO: weird that cast is not working under this context.
            input = _transform_input(input)
            _input = self.precrop_padding(input)  # type: ignore
        out = super().forward(_input, params, return_transform)
        if not self._params['batch_prob'].all():
            # undo the pre-crop if nothing happened.
            if isinstance(out, tuple) and isinstance(input, tuple):
                return input[0], out[1]
            elif isinstance(out, tuple) and not isinstance(input, tuple):
                return input, out[1]
            else:
                return input
        return out


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
        cropping_mode (str): The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
                             on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                             to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                             differentiability. Default: `slice`.

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
        tensor([[[[1.0000, 1.5000, 2.0000],
                  [4.0000, 4.5000, 5.0000],
                  [7.0000, 7.5000, 8.0000]]]])
    """

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.08, 1.0),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (3. / 4., 4. / 3.),
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.,
        keepdim: bool = False,
        cropping_mode: str = 'slice',
    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens all the time.
        super(RandomResizedCrop, self).__init__(
            p=1., return_transform=return_transform, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self._device, self._dtype = _extract_device_dtype([scale, ratio])
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.resample: Resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners),
        )
        self.cropping_mode = cropping_mode

    def __repr__(self) -> str:
        repr = f"size={self.size}, scale={self.scale}, ratio={self.ratio}, interpolation={self.resample.name}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        scale = torch.as_tensor(self.scale, device=self._device, dtype=self._dtype)
        ratio = torch.as_tensor(self.ratio, device=self._device, dtype=self._dtype)
        target_size: torch.Tensor = rg.random_crop_size_generator(
            batch_shape[0], (batch_shape[-2], batch_shape[-1]), scale, ratio, self.same_on_batch,
            self.device, self.dtype)['size']
        return rg.random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), target_size,
                                        resize_to=self.size, same_on_batch=self.same_on_batch,
                                        device=self.device, dtype=self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        transform: torch.Tensor = get_perspective_transform(
            params['src'].to(input), params['dst'].to(input))
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.cropping_mode == 'resample':  # uses bilinear interpolation to crop
            transform = cast(torch.Tensor, transform)
            return crop_by_transform_mat(
                input, transform, self.size, mode=self.resample.name.lower(),
                padding_mode='zeros', align_corners=self.align_corners)
        elif self.cropping_mode == 'slice':   # uses advanced slicing to crop
            B, C, _, _ = input.shape
            out = torch.empty(B, C, *self.size, device=input.device, dtype=input.dtype)
            for i in range(B):
                x1 = int(params['src'][i, 0, 0])
                x2 = int(params['src'][i, 1, 0]) + 1
                y1 = int(params['src'][i, 0, 1])
                y2 = int(params['src'][i, 3, 1]) + 1
                out[i] = resize(input[i:i + 1, :, y1:y2, x1:x2],
                                self.size,
                                interpolation=(self.resample.name).lower(),
                                align_corners=self.align_corners)
            return out
        else:
            raise NotImplementedError(f"Not supported type: {self.cropping_mode}.")


class Normalize(AugmentationBase2D):
    r"""Normalize tensor images with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean (torch.Tensor): Mean for each channel.
        std (torch.Tensor): Standard deviations for each channel.

    Return:
        torch.Tensor: Normalised tensor with same size as input :math:`(*, C, H, W)`.

    Examples:

        >>> norm = Normalize(mean=torch.zeros(1, 4), std=torch.ones(1, 4))
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor,
        return_transform: bool = False, p: float = 1., keepdim: bool = False
    ) -> None:
        super(Normalize, self).__init__(p=p, return_transform=return_transform, same_on_batch=True,
                                        keepdim=keepdim)
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        repr = f"mean={self.mean}, std={self.std}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return normalize(input, self.mean, self.std)


class Denormalize(AugmentationBase2D):
    r"""Denormalize tensor images with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * mean[channel]) + std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean (torch.Tensor): Mean for each channel.
        std (torch.Tensor): Standard deviations for each channel.

    Return:
        torch.Tensor: Denormalised tensor with same size as input :math:`(*, C, H, W)`.

    Examples:

        >>> norm = Denormalize(mean=torch.zeros(1, 4), std=torch.ones(1, 4))
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor,
        return_transform: bool = False, p: float = 1., keepdim: bool = False
    ) -> None:
        super(Denormalize, self).__init__(p=p, return_transform=return_transform, same_on_batch=True,
                                          keepdim=keepdim)
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        repr = f"mean={self.mean}, std={self.std}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return denormalize(input, self.mean, self.std)


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
        resample (int, str or kornia.Resample): Default: Resample.NEAREST.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

        Please set ``resample`` to ``'bilinear'`` if more meaningful gradients wanted.

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
            resample: Union[str, int, Resample] = Resample.NEAREST.name,
            return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super(RandomMotionBlur, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch,
                                               keepdim=keepdim)
        self.kernel_size: Union[int, Tuple[int, int]] = kernel_size
        self._device, self._dtype = _extract_device_dtype([angle, direction])

        self.angle = angle
        self.direction = direction
        self.border_type = BorderType.get(border_type)
        self.resample = Resample.get(resample)
        self.flags: Dict[str, torch.Tensor] = {
            "border_type": torch.tensor(self.border_type.value),
            "interpolation": torch.tensor(self.resample.value)
        }

    def __repr__(self) -> str:
        repr = f"kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}, " +\
            f"border_type='{self.border_type.name.lower()}'"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        angle = _range_bound(
            self.angle, 'angle', center=0., bounds=(-360, 360), device=self._device, dtype=self._dtype)
        direction = _range_bound(
            self.direction, 'direction', center=0., bounds=(-1, 1), device=self._device, dtype=self._dtype)
        return rg.random_motion_blur_generator(
            batch_shape[0], self.kernel_size, angle, direction, self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kernel_size: int = cast(int, params['ksize_factor'].unique().item())
        angle = params['angle_factor']
        direction = params['direction_factor']
        return motion_blur(
            input, kernel_size, angle, direction, border_type=self.border_type.name.lower(),
            mode=self.resample.name.lower())


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
        self._device, self._dtype = _extract_device_dtype([thresholds, additions])
        self.thresholds = thresholds
        self.additions = additions

    def __repr__(self) -> str:
        repr = f"thresholds={self.thresholds}, additions={self.additions}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        thresholds = _range_bound(
            self.thresholds, 'thresholds', center=0.5, bounds=(0., 1.), device=self._device, dtype=self._dtype)
        additions = _range_bound(
            self.additions, 'additions', bounds=(-0.5, 0.5), device=self._device, dtype=self._dtype)
        return rg.random_solarize_generator(batch_shape[0], thresholds, additions, self.same_on_batch,
                                            self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        thresholds = params['thresholds_factor']
        additions: Optional[torch.Tensor]
        if 'additions_factor' in params:
            additions = params['additions_factor']
        else:
            additions = None
        return solarize(input, thresholds, additions)


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
        self._device, self._dtype = _extract_device_dtype([bits])
        self.bits = bits

    def __repr__(self) -> str:
        repr = f"(bits={self.bits}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        bits = torch.as_tensor(self.bits, device=self._device, dtype=self._dtype)
        if len(bits.size()) == 0:
            bits = bits.repeat(2)
            bits[1] = 8
        elif not (len(bits.size()) == 1 and bits.size(0) == 2):
            raise ValueError(f"'bits' shall be either a scalar or a length 2 tensor. Got {bits}.")
        return rg.random_posterize_generator(batch_shape[0], bits, self.same_on_batch, self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return posterize(input, params['bits_factor'])


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
        self._device, self._dtype = _extract_device_dtype([sharpness])
        self.sharpness = sharpness

    def __repr__(self) -> str:
        repr = f"sharpness={self.sharpness}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        sharpness = torch.as_tensor(self.sharpness, device=self._device, dtype=self._dtype)
        if sharpness.dim() == 0:
            sharpness = sharpness.repeat(2)
            sharpness[0] = 0.
        elif not (sharpness.dim() == 1 and sharpness.size(0) == 2):
            raise ValueError(f"'sharpness' must be a scalar or a length 2 tensor. Got {sharpness}.")
        return rg.random_sharpness_generator(batch_shape[0], sharpness, self.same_on_batch,
                                             self.device, self.dtype)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        factor = params['sharpness_factor']
        return sharpness(input, factor)


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

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return equalize(input)


# TODO: Rename to RandomGaussianBlur?
class GaussianBlur(AugmentationBase2D):

    r"""Apply gaussian blur given tensor image or a batch of tensor images randomly.

    Args:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability of applying the transformation. Default value is 0.5.

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
        >>> blur = GaussianBlur((3, 3), (0.1, 2.0), p=1.)
        >>> blur(input)
        tensor([[[[0.6699, 0.4645, 0.3193, 0.1741, 0.1955],
                  [0.5422, 0.6657, 0.6261, 0.6527, 0.5195],
                  [0.3826, 0.2638, 0.1902, 0.1620, 0.2141],
                  [0.6329, 0.6732, 0.5634, 0.4037, 0.2049],
                  [0.8307, 0.6753, 0.7147, 0.5768, 0.7097]]]])
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect',
                 return_transform: bool = False,
                 same_on_batch: bool = False,
                 p: float = 0.5) -> None:
        super(GaussianBlur, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type: BorderType = BorderType.get(border_type)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type.name.lower())


class RandomInvert(AugmentationBase2D):

    r"""Invert the tensor images values randomly.

    Args:
        max_val (torch.Tensor): The expected maximum value in the input tensor. The shape has to
          according to the input tensor shape, or at least has to work with broadcasting. Default: 1.0.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability of applying the transformation. Default value is 0.5.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.rand(1, 1, 5, 5)
        >>> inv = RandomInvert()
        >>> inv(img)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])
    """

    def __init__(self,
                 max_val: torch.Tensor = torch.tensor(1.),
                 return_transform: bool = False,
                 same_on_batch: bool = False,
                 p: float = 0.5) -> None:
        super(RandomInvert, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        self.transform = Invert(max_val)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.transform(input)


class RandomChannelShuffle(AugmentationBase2D):
    r"""Shuffles the channels of a batch of multi-dimensional images.

    Args:
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability of applying the transformation. Default value is 0.5.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.arange(1*2*2*2.).view(1,2,2,2)
        >>> RandomChannelShuffle()(img)
        tensor([[[[4., 5.],
                  [6., 7.]],
        <BLANKLINE>
                 [[0., 1.],
                  [2., 3.]]]])
    """

    def __init__(self,
                 return_transform: bool = False,
                 same_on_batch: bool = False,
                 p: float = 0.5) -> None:
        super(RandomChannelShuffle, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, shape: torch.Size) -> Dict[str, torch.Tensor]:
        B, C, _, _ = shape
        channels = torch.rand(B, C).argsort(dim=1)
        return dict(channels=channels)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = torch.empty_like(input)
        for i in range(out.shape[0]):
            out[i] = input[i, params['channels'][i]]
        return out


class RandomGaussianNoise(AugmentationBase2D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    Args:
        mean (float): The mean of the gaussian distribution. Default: 0.
        std (float): The standard deviation of the gaussian distribution. Default: 1.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability of applying the transformation. Default value is 0.5.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 2, 2)
        >>> RandomGaussianNoise(mean=0., std=1., p=1.)(img)
        tensor([[[[ 2.5410,  0.7066],
                  [-1.1788,  1.5684]]]])
    """

    def __init__(self,
                 mean: float = 0.,
                 std: float = 1.,
                 return_transform: bool = False,
                 same_on_batch: bool = False,
                 p: float = 0.5) -> None:
        super(RandomGaussianNoise, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, shape: torch.Size) -> Dict[str, torch.Tensor]:
        noise = torch.randn(shape)
        return dict(noise=noise)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return input + params['noise'].to(input.device) * self.std + self.mean
