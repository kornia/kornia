from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample, BorderType
from kornia.augmentation.utils import (
    _adapted_uniform,
    _check_and_bound
)

from .types import (
    TupleFloat,
    UnionFloat,
    UnionType,
    FloatUnionType
)


def random_color_jitter_generator(
    batch_size: int,
    brightness: FloatUnionType = 0.,
    contrast: FloatUnionType = 0.,
    saturation: FloatUnionType = 0.,
    hue: FloatUnionType = 0.,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generator random color jiter parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """

    brightness_bound: torch.Tensor = _check_and_bound(
        brightness, 'brightness', center=1., bounds=(0, 2))

    contrast_bound: torch.Tensor = _check_and_bound(
        contrast, 'contrast', center=1.)

    saturation_bound: torch.Tensor = _check_and_bound(
        saturation, 'saturation', center=1.)

    hue_bound: torch.Tensor = _check_and_bound(hue, 'hue', bounds=(-0.5, 0.5))

    brightness_factor = _adapted_uniform(
        (batch_size,), brightness_bound[0], brightness_bound[1], same_on_batch)

    contrast_factor = _adapted_uniform(
        (batch_size,), contrast_bound[0], contrast_bound[1], same_on_batch)

    hue_factor = _adapted_uniform(
        (batch_size,), hue_bound[0], hue_bound[1], same_on_batch)

    saturation_factor = _adapted_uniform(
        (batch_size,), saturation_bound[0], saturation_bound[1], same_on_batch)

    return dict(brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,
                hue_factor=hue_factor,
                saturation_factor=saturation_factor,
                order=torch.randperm(4))


def random_prob_generator(
        batch_size: int, p: float = 0.5, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
    r"""Generator random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability of the image being flipped or grayscaled. Default value is 0.5
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    probs: torch.Tensor = _adapted_uniform((batch_size,), 0, 1, same_on_batch)

    batch_prob: torch.Tensor = (probs < p)

    return dict(batch_prob=batch_prob)


def _get_perspective_params(
    batch_size: int,
    width: int,
    height: int,
    distortion_scale: float,
    same_on_batch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    start_points: torch.Tensor = torch.tensor([[
        [0., 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]]).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx: float = distortion_scale * width / 2
    fy: float = distortion_scale * height / 2

    factor = torch.tensor([fx, fy]).view(-1, 1, 2)

    rand_val: torch.Tensor = _adapted_uniform((batch_size, 4, 2), 0, 1, same_on_batch)
    pts_norm = torch.tensor([[
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1]
    ]])
    end_points = start_points + factor * rand_val * pts_norm

    return start_points, end_points


def random_perspective_generator(
    batch_size: int,
    height: int,
    width: int,
    p: float,
    distortion_scale: float,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        p (float): probability of the image being applied perspective.
        distortion_scale (float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail


    Returns:
        params (Dict[str, torch.Tensor])
    """
    params: Dict[str, torch.Tensor] = random_prob_generator(batch_size, p)
    start_points, end_points = (
        _get_perspective_params(batch_size, width, height, distortion_scale, same_on_batch)
    )
    params['start_points'] = start_points
    params['end_points'] = end_points
    params['interpolation'] = torch.tensor(Resample.get(interpolation).value)
    params['align_corners'] = torch.tensor(align_corners)
    return params


def random_affine_generator(
    batch_size: int,
    height: int,
    width: int,
    degrees: UnionFloat,
    translate: Optional[TupleFloat] = None,
    scale: Optional[TupleFloat] = None,
    shear: Optional[UnionFloat] = None,
    resample: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
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

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    # check angle ranges
    degrees_tmp: TupleFloat
    if isinstance(degrees, (float, int,)):
        if degrees < 0.:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees_tmp = (-degrees, degrees)
    else:
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
            "degrees should be a list or tuple and it must be of length 2."
        degrees_tmp = degrees

    # check translation range
    if translate is not None:
        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
        for t in translate:
            if not (0.0 <= t <= 1.0):
                raise ValueError("translation values should be between 0 and 1")

    # check scale range
    if scale is not None:
        assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
            "scale should be a list or tuple and it must be of length 2."
        for s in scale:
            if s <= 0:
                raise ValueError("scale values should be positive")

    # check shear range
    shear_tmp: Optional[TupleFloat]
    if shear is not None:
        if isinstance(shear, float):
            if shear < 0:
                raise ValueError("If shear is a single number, it must be positive.")
            shear_tmp = (-shear, shear)
        else:
            assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                "shear should be a list or tuple and it must be of length 2."
            shear_tmp = shear
    else:
        shear_tmp = shear

    return _get_random_affine_params(
        batch_size, height, width, degrees_tmp, translate, scale, shear_tmp, resample, same_on_batch, align_corners)


def random_rotation_generator(
    batch_size: int,
    degrees: FloatUnionType,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (sequence or float or tensor): range of degrees to select from. If degrees is a number the
        range of degrees to select from will be (-degrees, +degrees)
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners (bool): interpolation flag. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if not torch.is_tensor(degrees):
        if isinstance(degrees, (float, int)):
            if degrees < 0:
                raise ValueError(f"If Degrees is only one number it must be a positive number. Got{degrees}")
            degrees = torch.tensor([-degrees, degrees]).to(torch.float32)

        elif isinstance(degrees, (tuple, list)):
            degrees = torch.tensor(degrees).to(torch.float32)

        else:
            raise TypeError(f"Degrees should be a float number a sequence or a tensor. Got {type(degrees)}")

    # https://mypy.readthedocs.io/en/latest/casts.html cast to please mypy gods
    degrees = cast(torch.Tensor, degrees)

    if degrees.numel() != 2:
        raise ValueError("If degrees is a sequence it must be of length 2")

    degrees = _adapted_uniform((batch_size,), degrees[0], degrees[1], same_on_batch)

    return dict(degrees=degrees,
                interpolation=torch.tensor(Resample.get(interpolation).value),
                align_corners=torch.tensor(align_corners))


def _get_random_affine_params(
    batch_size: int,
    height: int,
    width: int,
    degrees: TupleFloat,
    translate: Optional[TupleFloat],
    scales: Optional[TupleFloat],
    shears: Optional[TupleFloat],
    resample: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```affine``` transformation random affine transform.
    The returned matrix is Bx3x3.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    angle = _adapted_uniform((batch_size,), degrees[0], degrees[1], same_on_batch)

    # compute tensor ranges
    if scales is not None:
        scale = _adapted_uniform((batch_size,), scales[0], scales[1], same_on_batch)
    else:
        scale = torch.ones(batch_size)

    if translate is not None:
        max_dx: float = translate[0] * width
        max_dy: float = translate[1] * height
        translations = torch.stack([
            _adapted_uniform((batch_size,), -max_dx, max_dx, same_on_batch),
            _adapted_uniform((batch_size,), -max_dy, max_dy, same_on_batch)
        ], dim=-1)
    else:
        translations = torch.zeros(batch_size, 2)

    center: torch.Tensor = torch.tensor(
        [width, height], dtype=torch.float32).view(1, 2) / 2. - 0.5
    center = center.expand(batch_size, -1)

    if shears is not None:
        sx = _adapted_uniform((batch_size,), shears[0], shears[1], same_on_batch)
        sy = _adapted_uniform((batch_size,), shears[0], shears[1], same_on_batch)
    else:
        sx = sy = torch.tensor([0] * batch_size)

    return dict(translations=translations,
                center=center,
                scale=scale,
                angle=angle,
                sx=sx,
                sy=sy,
                resample=torch.tensor(Resample.get(resample).value),
                align_corners=torch.tensor(align_corners))


def random_crop_generator(
    batch_size: int,
    input_size: Tuple[int, int],
    size: Tuple[int, int],
    resize_to: Optional[Tuple[int, int]] = None,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
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
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
     """
    x_diff = input_size[1] - size[1]
    y_diff = input_size[0] - size[0]

    if x_diff < 0 or y_diff < 0:
        raise ValueError("input_size %s cannot be smaller than crop size %s in any dimension."
                         % (str(input_size), str(size)))

    x_start = _adapted_uniform((batch_size,), 0, x_diff + 1, same_on_batch).long()
    y_start = _adapted_uniform((batch_size,), 0, y_diff + 1, same_on_batch).long()

    crop = torch.tensor([[
        [0, 0],
        [size[1] - 1, 0],
        [size[1] - 1, size[0] - 1],
        [0, size[0] - 1],
    ]]).repeat(batch_size, 1, 1)

    crop_src = crop.clone()
    crop_src[:, :, 0] += x_start.unsqueeze(dim=0).reshape(batch_size, 1)
    crop_src[:, :, 1] += y_start.unsqueeze(dim=0).reshape(batch_size, 1)

    if resize_to is None:
        crop_dst = crop
    else:
        crop_dst = torch.tensor([[
            [0, 0],
            [resize_to[1] - 1, 0],
            [resize_to[1] - 1, resize_to[0] - 1],
            [0, resize_to[0] - 1],
        ]]).repeat(batch_size, 1, 1)

    return dict(src=crop_src,
                dst=crop_dst,
                interpolation=torch.tensor(Resample.get(interpolation).value),
                align_corners=torch.tensor(align_corners))


def random_crop_size_generator(
    size: Tuple[int, int],
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    same_on_batch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get cropping heights and widths for ```crop``` transformation for resized crop transform.

    Args:
        size (Tuple[int, int]): expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    area = _adapted_uniform(
        (10,), scale[0] * size[0] * size[1], scale[1] * size[0] * size[1], same_on_batch)
    log_ratio = _adapted_uniform(
        (10,), math.log(ratio[0]), math.log(ratio[1]), same_on_batch)
    aspect_ratio = torch.exp(log_ratio)

    w = torch.sqrt(area * aspect_ratio).int()
    h = torch.sqrt(area / aspect_ratio).int()

    # Element-wise w, h condition
    cond = ((0 < h) * (h < size[1]) * (0 < w) * (w < size[0])).int()
    if torch.sum(cond) > 0:
        return (h[torch.argmax(cond)], w[torch.argmax(cond)])

    # Fallback to center crop
    in_ratio = float(size[0]) / float(size[1])
    if (in_ratio < min(ratio)):
        w = torch.tensor(size[0])
        h = torch.round(w / min(ratio))
    elif (in_ratio > max(ratio)):
        h = torch.tensor(size[1])
        w = torch.round(h * max(ratio))
    else:  # whole image
        w = torch.tensor(size[0])
        h = torch.tensor(size[1])
    return (h, w)


def random_rectangles_params_generator(
    batch_size: int,
    height: int,
    width: int,
    p: float,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    value: float = 0.,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```erasing``` transformation for erasing transform

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        p (float): probability of applying random earaing.
        scale ([int, int]): range of size of the origin size cropped
        ratio ([int, int]): range of aspect ratio of the origin aspect ratio cropped
        value (float): value to be filled in the erased area.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if not (isinstance(scale[0], float) and isinstance(scale[1], float) and scale[0] > 0. and scale[1] > 0.):
        raise TypeError(
            f"'erase_scale_range' must be a Tuple[float, float] with positive values"
        )
    if not (isinstance(ratio[0], float) and isinstance(ratio[1], float) and ratio[0] > 0. and ratio[1] > 0.):
        raise TypeError(
            f"'ratio' must be a Tuple[float, float] with positive values"
        )

    batch_prob = random_prob_generator(batch_size, p, same_on_batch)['batch_prob']
    zeros = torch.zeros((batch_size,))
    images_area = height * width
    target_areas = _adapted_uniform(
        (batch_size,), scale[0], scale[1], same_on_batch) * images_area
    if ratio[0] < 1. and ratio[1] > 1.:
        aspect_ratios1 = _adapted_uniform((batch_size,), ratio[0], 1, same_on_batch)
        aspect_ratios2 = _adapted_uniform((batch_size,), 1, ratio[1], same_on_batch)
        rand_idxs = torch.round(torch.rand((batch_size,))).bool()
        aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
    else:
        aspect_ratios = _adapted_uniform((batch_size,), ratio[0], ratio[1], same_on_batch)

    # based on target areas and aspect ratios, rectangle params are computed
    heights = torch.min(
        torch.max(torch.round((target_areas * aspect_ratios) ** (1 / 2)),
                  torch.tensor(1.)),
        torch.tensor(float(height))
    ).int()

    widths = torch.min(
        torch.max(torch.round((target_areas / aspect_ratios) ** (1 / 2)),
                  torch.tensor(1.)),
        torch.tensor(float(width))
    ).int()

    xs = (_adapted_uniform((batch_size,), 0, 1, same_on_batch) * (torch.tensor(width) - widths + 1).float()).int()
    ys = (_adapted_uniform((batch_size,), 0, 1, same_on_batch) * (torch.tensor(height) - heights + 1).float()).int()

    params: Dict[str, torch.Tensor] = {}
    params["widths"] = torch.where(batch_prob, widths, zeros.to(widths.dtype))
    params["heights"] = torch.where(batch_prob, heights, zeros.to(widths.dtype))
    params["xs"] = xs
    params["ys"] = ys
    params["values"] = torch.tensor([value] * batch_size)
    return params


def center_crop_params_generator(
    batch_size: int,
    height: int,
    width: int,
    size: Tuple[int, int],
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```center_crop``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (h, w).

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """

    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))

    # unpack input sizes
    dst_h, dst_w = size
    src_h, src_w = height, width

    # compute start/end offsets
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = src_w_half - dst_w_half
    start_y = src_h_half - dst_h_half

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1

    # [y, x] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: torch.Tensor = torch.tensor([[
        [start_x, start_y],
        [end_x, start_y],
        [end_x, end_y],
        [start_x, end_y],
    ]])

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ]]).expand(points_src.shape[0], -1, -1)
    return dict(src=points_src,
                dst=points_dst,
                interpolation=torch.tensor(Resample.BILINEAR.value),
                align_corners=torch.tensor(align_corners))


def random_motion_blur_generator(
    batch_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    angle: UnionFloat,
    direction: UnionFloat,
    border_type: Union[int, str, BorderType] = BorderType.CONSTANT.name,
    same_on_batch: bool = True
) -> Dict[str, torch.Tensor]:

    angle_bound: torch.Tensor = _check_and_bound(angle, 'angle', center=0.)
    direction_bound: torch.Tensor = _check_and_bound(direction, 'direction', center=0., bounds=(-1, 1))

    if isinstance(kernel_size, int):
        ksize_factor = torch.tensor([kernel_size] * batch_size)
    elif isinstance(kernel_size, tuple):
        ksize_x, ksize_y = kernel_size
        ksize_factor = _adapted_uniform(
            (batch_size,), ksize_x // 2, ksize_y // 2, same_on_batch).int() * 2 + 1
    else:
        raise TypeError(f"Unsupported type: {type(kernel_size)}")

    angle_factor = _adapted_uniform(
        (batch_size,), angle_bound[0].float(), angle_bound[1].float(), same_on_batch)

    direction_factor = _adapted_uniform(
        (batch_size,), direction_bound[0].float(), direction_bound[1].float(), same_on_batch)

    return dict(ksize_factor=ksize_factor,
                angle_factor=angle_factor,
                direction_factor=direction_factor,
                border_type=torch.tensor(BorderType.get(border_type).value))


def random_solarize_generator(
    batch_size: int,
    thresholds: FloatUnionType = 0.1,
    additions: FloatUnionType = 0.1,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generator random solarize parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        thresholds (float or tuple): Default value is 0
        additions (float or tuple): Default value is 0
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """

    thresholds_bound: torch.Tensor = _check_and_bound(
        thresholds, 'thresholds', center=0.5, bounds=(0., 1.))
    additions_bound: torch.Tensor = _check_and_bound(additions, 'additions', bounds=(-0.5, 0.5))

    thresholds_factor = _adapted_uniform(
        (batch_size,), thresholds_bound[0].float(), thresholds_bound[1].float(), same_on_batch)

    additions_factor = _adapted_uniform(
        (batch_size,), additions_bound[0].float(), additions_bound[1].float(), same_on_batch)

    return dict(
        thresholds_factor=thresholds_factor,
        additions_factor=additions_factor
    )


def random_posterize_generator(
    batch_size: int,
    bits: Union[int, Tuple[int, int], torch.Tensor] = 3,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generator random posterize parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        bits (int or tuple): Default value is 0. Integer that ranged from 0 ~ 8.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if not isinstance(bits, torch.Tensor):
        bits = torch.tensor(bits)

    if len(bits.size()) == 0:
        lower = bits
        upper = torch.tensor(8)
    elif len(bits.size()) == 1 and bits.size(0) == 2:
        lower = bits[0]
        upper = bits[1]
    else:
        raise ValueError(f"Expect float or tuple. Got {bits}.")

    bits_factor = _adapted_uniform((batch_size,), lower.float(), upper.float(), same_on_batch).int()

    return dict(
        bits_factor=bits_factor
    )


def random_sharpness_generator(
    batch_size: int,
    sharpness: Union[float, Tuple[float, float], torch.Tensor] = 1.,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generator random sharpness parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        sharpness (float or tuple): Default value is 0. Range from 0 ~ 8.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if not isinstance(sharpness, torch.Tensor):
        sharpness = torch.tensor(sharpness)

    if len(sharpness.size()) == 0:
        lower = torch.tensor(0)
        upper = sharpness
    elif len(sharpness.size()) == 1 and sharpness.size(0) == 2:
        lower = sharpness[0]
        upper = sharpness[1]
    else:
        raise ValueError(f"Expect float or tuple. Got {sharpness}.")

    sharpness_factor = _adapted_uniform((batch_size,), lower.float(), upper.float(), same_on_batch)

    return dict(
        sharpness_factor=sharpness_factor
    )
