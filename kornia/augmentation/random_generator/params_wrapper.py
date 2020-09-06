from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample, BorderType, SamplePadding
from kornia.augmentation.utils import (
    _adapted_uniform,
    _adapted_beta,
    _joint_range_check,
)
from .types import AugParamDict
from .random_generator import (
    prob_params_generator,
    affine_params_generator,
    color_jitter_params_generator,
    crop_params_generator,
    crop_size_params_generator,
    cutmix_params_generator,
    mixup_params_generator,
    motion_blur_params_generator,
    perspective_params_generator,
    posterize_params_generator,
    rectangles_params_generator,
    rotation_params_generator,
    sharpness_params_generator,
    solarize_params_generator,
    center_crop_params_generator
)


def random_prob_generator(
        batch_size: int, p: float = 0.5, same_on_batch: bool = False) -> torch.Tensor:
    r"""Generate random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability to generate an 1-d binary mask. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        torch.Tensor: parameters to be passed for transformation.
    """
    return prob_params_generator(batch_size, p, same_on_batch)['batch_prob']


def random_color_jitter_generator(
    batch_size: int,
    brightness: Optional[torch.Tensor] = None,
    contrast: Optional[torch.Tensor] = None,
    saturation: Optional[torch.Tensor] = None,
    hue: Optional[torch.Tensor] = None,
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Generate random color jiter parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        brightness (torch.Tensor, optional): Default value is 0.
        contrast (torch.Tensor, optional): Default value is 0.
        saturation (torch.Tensor, optional): Default value is 0.
        hue (torch.Tensor, optional): Default value is 0.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = color_jitter_params_generator(
        batch_size, brightness, contrast, saturation, hue, same_on_batch)
    flags = dict(order=torch.randperm(4))

    return AugParamDict(dict(params=params, flags=flags))


def random_perspective_generator(
    batch_size: int,
    height: int,
    width: int,
    distortion_scale: torch.Tensor,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False,
) -> AugParamDict:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        distortion_scale (torch.Tensor): it controls the degree of distortion and ranges from 0 to 1.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = perspective_params_generator(
        batch_size, height, width, distortion_scale=distortion_scale, same_on_batch=same_on_batch)
    flags = dict(
        interpolation=torch.tensor(Resample.get(interpolation).value),
        align_corners=torch.tensor(align_corners)
    )
    return AugParamDict(dict(params=params, flags=flags))


def random_affine_generator(
    batch_size: int,
    height: int,
    width: int,
    degrees: torch.Tensor,
    translate: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shear: Optional[torch.Tensor] = None,
    resample: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False,
    padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
) -> AugParamDict:
    r"""Get parameters for ``affine`` for a random affine transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
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
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners(bool): interpolation flag. Default: False.See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
        padding_mode (int, str or kornia.SamplePadding): Default: SamplePadding.ZEROS

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = affine_params_generator(
        batch_size, height, width, degrees=degrees, translate=translate, scale=scale, shear=shear,
        same_on_batch=same_on_batch
    )
    flags = dict(
        resample=torch.tensor(Resample.get(resample).value),
        padding_mode=torch.tensor(SamplePadding.get(padding_mode).value),
        align_corners=torch.tensor(align_corners)
    )
    return AugParamDict(dict(params=params, flags=flags))


def random_rotation_generator(
    batch_size: int,
    degrees: torch.Tensor,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> AugParamDict:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): range of degrees with shape (2) to select from.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners (bool): interpolation flag. Default: False.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = rotation_params_generator(batch_size, degrees=degrees, same_on_batch=same_on_batch)
    flags = dict(
        interpolation=torch.tensor(Resample.get(interpolation).value),
        align_corners=torch.tensor(align_corners)
    )

    return AugParamDict(dict(params=params, flags=flags))


def random_crop_generator(
    batch_size: int,
    input_size: Tuple[int, int],
    size: Union[Tuple[int, int], torch.Tensor],
    resize_to: Optional[Tuple[int, int]] = None,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> AugParamDict:
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        batch_size (int): the tensor batch size.
        input_size (tuple): Input image shape, like (h, w).
        size (tuple): Desired size of the crop operation, like (h, w).
        resize_to (tuple): Desired output size of the crop, like (h, w). If None, no resize will be performed.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners (bool): interpolation flag. Default: False.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = crop_params_generator(
        batch_size, input_size, size, resize_to=resize_to, same_on_batch=same_on_batch)
    flags = dict(
        interpolation=torch.tensor(Resample.get(interpolation).value),
        align_corners=torch.tensor(align_corners)
    )

    return AugParamDict(dict(params=params, flags=flags))


def random_crop_size_generator(
    size: Tuple[int, int],
    scale: torch.Tensor,
    ratio: torch.Tensor,
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Get cropping heights and widths for ```crop``` transformation for resized crop transform.

    Args:
        size (Tuple[int, int]): expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = crop_size_params_generator(size, scale, ratio, same_on_batch)
    return AugParamDict(dict(params=params))


def random_rectangles_params_generator(
    batch_size: int,
    height: int,
    width: int,
    scale: torch.Tensor,
    ratio: torch.Tensor,
    value: float = 0.,
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Get parameters for ```erasing``` transformation for erasing transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        scale (torch.Tensor): range of size of the origin size cropped. Shape (2).
        ratio (torch.Tensor): range of aspect ratio of the origin aspect ratio cropped. Shape (2).
        value (float): value to be filled in the erased area.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = rectangles_params_generator(
        batch_size, height, width, scale=scale, ratio=ratio, value=value, same_on_batch=same_on_batch)
    return AugParamDict(dict(params=params))


def center_crop_generator(
    batch_size: int,
    height: int,
    width: int,
    size: Tuple[int, int],
    align_corners: bool = False
) -> AugParamDict:
    r"""Get parameters for ```center_crop``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (h, w).

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = center_crop_params_generator(
        batch_size, height, width, size=size)
    flags = dict(
        interpolation=torch.tensor(Resample.BILINEAR.value),
        align_corners=torch.tensor(align_corners)
    )
    return AugParamDict(dict(params=params, flags=flags))


def random_motion_blur_generator(
    batch_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    angle: torch.Tensor,
    direction: torch.Tensor,
    border_type: Union[int, str, BorderType] = BorderType.CONSTANT.name,
    same_on_batch: bool = True
) -> AugParamDict:
    r"""Get parameters for motion blur.

    Args:
        batch_size (int): the tensor batch size.
        kernel_size (int or (int, int)): motion kernel width and height (odd and positive).
        angle (torch.Tensor): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (torch.Tensor): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with
            angle provided via angle), while higher values towards 1.0 will point the motion
            blur forward. A value of 0.0 leads to a uniformly (but still angled) motion blur.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = motion_blur_params_generator(
        batch_size, kernel_size, angle, direction=direction, same_on_batch=same_on_batch)
    flags = dict(border_type=torch.tensor(BorderType.get(border_type).value))

    return AugParamDict(dict(params=params, flags=flags))


def random_solarize_generator(
    batch_size: int,
    thresholds: torch.Tensor = torch.tensor([0.4, 0.6]),
    additions: torch.Tensor = torch.tensor([-0.1, 0.1]),
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Generate random solarize parameters for a batch of images.

    For each pixel in the image less than threshold, we add 'addition' amount to it and then clip the pixel value
    to be between 0 and 1.0

    Args:
        batch_size (int): the number of images.
        thresholds (torch.Tensor): Pixels less than threshold will selected. Otherwise, subtract 1.0 from the pixel.
            Default value will be sampled from [0.4, 0.6].
        additions (torch.Tensor): The value is between -0.5 and 0.5. Default value will be sampled from [-0.1, 0.1]
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
    """
    params = solarize_params_generator(
        batch_size, thresholds=thresholds, additions=additions, same_on_batch=same_on_batch)

    return AugParamDict(dict(params=params))


def random_posterize_generator(
    batch_size: int,
    bits: torch.Tensor = torch.tensor(3),
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Generate random posterize parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        bits (int or tuple): Default value is 3. Integer that ranged from 0 ~ 8.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
    """
    params = posterize_params_generator(batch_size, bits=bits, same_on_batch=same_on_batch)

    return AugParamDict(dict(params=params))


def random_sharpness_generator(
    batch_size: int,
    sharpness: torch.Tensor = torch.tensor([0, 1.]),
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Generate random sharpness parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        sharpness (torch.Tensor): Must be above 0. Default value is sampled from (0, 1).
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
    """
    params = sharpness_params_generator(batch_size, sharpness=sharpness, same_on_batch=same_on_batch)

    return AugParamDict(dict(params=params))


def random_mixup_generator(
    batch_size: int,
    p: float = 0.5,
    lambda_val: Optional[torch.Tensor] = None,
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        p (flot): probability of applying mixup.
        lambda_val (torch.Tensor, optional): min-max strength for mixup images, ranged from [0., 1.].
            If None, it will be set to tensor([0., 1.]), which means no restrictions.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_mixup_generator(5, 0.7)
        {'mixup_pairs': tensor([4, 0, 3, 1, 2]), 'mixup_lambdas': tensor([0.6323, 0.0000, 0.4017, 0.0223, 0.1689])}
    """
    params = mixup_params_generator(batch_size, p=p, lambda_val=lambda_val, same_on_batch=same_on_batch)

    return AugParamDict(dict(params=params))


def random_cutmix_generator(
    batch_size: int,
    width: int,
    height: int,
    p: float = 0.5,
    num_mix: int = 1,
    beta: Optional[torch.Tensor] = None,
    cut_size: Optional[torch.Tensor] = None,
    same_on_batch: bool = False
) -> AugParamDict:
    r"""Generate cutmix indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        width (int): image width.
        height (int): image height.
        p (float): probability of applying cutmix.
        num_mix (int): number of images to mix with. Default is 1.
        beta (float or torch.Tensor, optional): hyperparameter for generating cut size from beta distribution.
            If None, it will be set to 1.
        cut_size ((float, float) or torch.Tensor, optional): controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_cutmix_generator(3, 224, 224, p=0.5, num_mix=2)
        {'mix_pairs': tensor([[2, 0, 1],
                [1, 2, 0]]), 'crop_src': tensor([[[[ 36,  25],
                  [209,  25],
                  [209, 198],
                  [ 36, 198]],
        <BLANKLINE>
                 [[157, 137],
                  [156, 137],
                  [156, 136],
                  [157, 136]],
        <BLANKLINE>
                 [[  3,  12],
                  [210,  12],
                  [210, 219],
                  [  3, 219]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[ 83, 126],
                  [177, 126],
                  [177, 220],
                  [ 83, 220]],
        <BLANKLINE>
                 [[ 55,   8],
                  [206,   8],
                  [206, 159],
                  [ 55, 159]],
        <BLANKLINE>
                 [[ 97,  70],
                  [ 96,  70],
                  [ 96,  69],
                  [ 97,  69]]]])}

    """
    params = cutmix_params_generator(
        batch_size, width, height, p=p, num_mix=num_mix, beta=beta, cut_size=cut_size, same_on_batch=same_on_batch)

    return AugParamDict(dict(params=params))
