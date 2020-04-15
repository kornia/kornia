from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample
from kornia.augmentation.utils import _adapted_uniform

from .types import (
    TupleFloat,
    UnionFloat,
    UnionType,
    FloatUnionType
)


def random_color_jitter_generator(batch_size: int, brightness: FloatUnionType = 0.,
                                  contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                                  hue: FloatUnionType = 0., same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
    r"""Generator random color jiter parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0

    Returns:
        dict: generated parameter dictionary.

    See :class:`~kornia.augmentation.ColorJitter` for details.
    """

    def _check_and_bound(factor: FloatUnionType, name: str, center: float = 0.,
                         bounds: Tuple[float, float] = (0, float('inf'))) -> torch.Tensor:
        r"""Check inputs and compute the corresponding factor bounds
        """

        if isinstance(factor, float):

            if factor < 0:
                raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor}")

            factor_bound = torch.tensor([center - factor, center + factor], dtype=torch.float32)
            factor_bound = torch.clamp(factor_bound, bounds[0], bounds[1])

        elif (isinstance(factor, torch.Tensor) and factor.dim() == 0):

            if factor < 0:
                raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor}")

            factor_bound = torch.tensor(
                [torch.tensor(center) - factor, torch.tensor(center) + factor], dtype=torch.float32)
            factor_bound = torch.clamp(factor_bound, bounds[0], bounds[1])

        elif isinstance(factor, (tuple, list)) and len(factor) == 2:

            if not bounds[0] <= factor[0] <= factor[1] <= bounds[1]:
                raise ValueError(f"{name}[0] should be smaller than {name}[1] got {factor}")

            factor_bound = torch.tensor(factor, dtype=torch.float32)

        elif isinstance(factor, torch.Tensor) and factor.shape[0] == 2 and factor.dim() == 1:

            if not bounds[0] <= factor[0] <= factor[1] <= bounds[1]:
                raise ValueError(f"{name}[0] should be smaller than {name}[1] got {factor}")

            factor_bound = factor

        else:

            raise TypeError(
                f"The {name} should be a float number or a tuple with length 2 whose values move between {bounds}.")

        return factor_bound

    brightness_bound: torch.Tensor = _check_and_bound(
        brightness, 'brightness', center=1., bounds=(0, 2))
    contrast_bound: torch.Tensor = _check_and_bound(contrast, 'contrast', center=1.)
    saturation_bound: torch.Tensor = _check_and_bound(saturation, 'saturation', center=1.)
    hue_bound: torch.Tensor = _check_and_bound(hue, 'hue', bounds=(-0.5, 0.5))

    brightness_factor = _adapted_uniform((batch_size,), brightness_bound[0], brightness_bound[1], same_on_batch)

    contrast_factor = _adapted_uniform((batch_size,), contrast_bound[0], contrast_bound[1], same_on_batch)

    hue_factor = _adapted_uniform((batch_size,), hue_bound[0], hue_bound[1], same_on_batch)

    saturation_factor = _adapted_uniform((batch_size,), saturation_bound[0], saturation_bound[1], same_on_batch)

    return {
        "brightness_factor": brightness_factor,
        "contrast_factor": contrast_factor,
        "hue_factor": hue_factor,
        "saturation_factor": saturation_factor,
        "order": torch.randperm(4)
    }


def random_prob_generator(batch_size: int, p: float = 0.5, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
    r"""Generator random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability of the image being flipped or grayscaled. Default value is 0.5

    Returns:
        dict: generated parameter dictionary.

    See :class:`~kornia.augmentation.RandomGrayscale` for details.
    See :class:`~kornia.augmentation.RandomHorizontalFlip` for details.
    See :class:`~kornia.augmentation.RandomVerticalFlip` for details.
    """

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    probs: torch.Tensor = _adapted_uniform((batch_size,), 0, 1, same_on_batch)

    batch_prob: torch.Tensor = probs < p

    return {'batch_prob': batch_prob}


def _get_perspective_params(batch_size: int, width: int, height: int, distortion_scale: float,
                            same_on_batch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        width (int): width of the image.
        height (int) : height of the image.
        distortion_scale (float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    Returns:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        The points are in -x order.
    """
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
    batch_size: int, height: int, width: int, p: float, distortion_scale: float, same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = random_prob_generator(batch_size, p)
    start_points, end_points = (
        _get_perspective_params(batch_size, width, height, distortion_scale, same_on_batch)
    )
    params['start_points'] = start_points
    params['end_points'] = end_points
    return params


def random_affine_generator(
        batch_size: int,
        height: int,
        width: int,
        degrees: UnionFloat,
        translate: Optional[TupleFloat] = None,
        scale: Optional[TupleFloat] = None,
        shear: Optional[UnionFloat] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR,
        same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
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
        batch_size, height, width, degrees_tmp, translate, scale, shear_tmp, resample, same_on_batch)


def random_rotation_generator(batch_size: int, degrees: FloatUnionType,
                              interpolation: Union[str, int, Resample] = Resample.BILINEAR,
                              same_on_batch: bool = False) -> Dict[str, torch.Tensor]:

    if not torch.is_tensor(degrees):
        if isinstance(degrees, (float, int)):
            if degrees < 0:
                raise ValueError(f"If Degrees is only one number it must be a positive number. Got{degrees}")
            degrees = torch.tensor([-degrees, degrees])

        elif isinstance(degrees, (tuple, list)):
            degrees = torch.tensor(degrees)

        else:
            raise TypeError(f"Degrees should be a float number a sequence or a tensor. Got {type(degrees)}")

    # https://mypy.readthedocs.io/en/latest/casts.html cast to please mypy gods
    degrees = cast(torch.Tensor, degrees)

    if degrees.numel() != 2:
        raise ValueError("If degrees is a sequence it must be of length 2")

    degrees = _adapted_uniform((batch_size,), degrees[0], degrees[1], same_on_batch)

    return dict(
        degrees=degrees,
        interpolation=torch.tensor(Resample.get(interpolation).value)
    )


def _get_random_affine_params(
    batch_size: int, height: int, width: int,
    degrees: TupleFloat, translate: Optional[TupleFloat],
    scales: Optional[TupleFloat], shears: Optional[TupleFloat],
    resample: Union[str, int, Resample] = Resample.BILINEAR, same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for affine transformation random generation.
    The returned matrix is Bx3x3.

    Returns:
        torch.Tensor: params to be passed to the affine transformation.
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
    return dict(
        translations=translations,
        center=center,
        scale=scale,
        angle=angle,
        sx=sx,
        sy=sy,
        resample=torch.tensor(Resample.get(resample).value)
    )


def random_crop_generator(
    batch_size: int, input_size: Tuple[int, int], size: Tuple[int, int], resize_to: Optional[Tuple[int, int]] = None,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR, same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
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

    return {'src': crop_src, 'dst': crop_dst, 'interpolation': torch.tensor(Resample.get(interpolation).value)}


def random_crop_size_generator(size: Tuple[int, int], scale: Tuple[float, float], ratio: Tuple[float, float],
                               same_on_batch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    area = _adapted_uniform((10,), scale[0] * size[0] * size[1], scale[1] * size[0] * size[1], same_on_batch)
    log_ratio = _adapted_uniform((10,), math.log(ratio[0]), math.log(ratio[1]), same_on_batch)
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


def random_rectangles_params_generator(batch_size: int, height: int, width: int, p: float,
                                       scale: Tuple[float, float], ratio: Tuple[float, float],
                                       value: float = 0., same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
    batch_prob = random_prob_generator(batch_size, p, same_on_batch)['batch_prob']
    zeros = torch.zeros((batch_size,))
    images_area = height * width
    target_areas = _adapted_uniform((batch_size,), scale[0], scale[1], same_on_batch) * images_area
    if ratio[0] < 1. and ratio[1] > 1.:
        aspect_ratios1 = _adapted_uniform((batch_size,), ratio[0], 1, same_on_batch)
        aspect_ratios2 = _adapted_uniform((batch_size,), 1, ratio[1], same_on_batch)
        rand_idxs = torch.round(torch.rand((batch_size,))).bool()
        aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
    else:
        aspect_ratios = _adapted_uniform((batch_size,), ratio[0], ratio[1], same_on_batch)
    # based on target areas and aspect ratios, rectangle params are computed
    heights = torch.min(
        torch.max(torch.round((target_areas * aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
        torch.tensor(float(height))
    ).int()
    widths = torch.min(
        torch.max(torch.round((target_areas / aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
        torch.tensor(float(width))
    ).int()
    xs = (torch.rand((batch_size,)) * (torch.tensor(width) - widths + 1).float()).int()
    ys = (torch.rand((batch_size,)) * (torch.tensor(height) - heights + 1).float()).int()

    params: Dict[str, torch.Tensor] = {}
    params["widths"] = torch.where(batch_prob, widths, zeros.to(widths.dtype))
    params["heights"] = torch.where(batch_prob, heights, zeros.to(widths.dtype))
    params["xs"] = xs
    params["ys"] = ys
    params["values"] = torch.tensor([value] * batch_size)

    return params
