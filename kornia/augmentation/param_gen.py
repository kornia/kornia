from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.geometry import pi
from kornia.geometry.transform import get_rotation_matrix2d


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]

TupleInt = Tuple[int, int]
TupleFloat = Tuple[float, float]
UnionFloat = Union[float, TupleFloat]


def _random_color_jitter_gen(batch_size: int, brightness: FloatUnionType = 0.,
                             contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                             hue: FloatUnionType = 0., random_generator: Optional[torch.Generator] = None
                             ) -> Dict[str, torch.Tensor]:
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
        brightness, 'brightness', bounds=(
            float('-inf'), float('inf')))
    contrast_bound: torch.Tensor = _check_and_bound(contrast, 'contrast', center=1.)
    saturation_bound: torch.Tensor = _check_and_bound(saturation, 'saturation', center=1.)
    hue_bound: torch.Tensor = _check_and_bound(hue, 'hue', bounds=(-pi.item(), pi.item()))

    brightness_factor = torch.FloatTensor(batch_size).uniform_(  # type: ignore
        brightness_bound[0].item(), brightness_bound[1].item(), generator=random_generator)

    contrast_factor = torch.FloatTensor(batch_size).uniform_(  # type: ignore
        contrast_bound[0].item(), contrast_bound[1].item(), generator=random_generator)

    hue_factor = torch.FloatTensor(batch_size).uniform_(  # type: ignore
        hue_bound[0].item(), hue_bound[1].item(), generator=random_generator)

    saturation_factor = torch.FloatTensor(batch_size).uniform_(  # type: ignore
        saturation_bound[0].item(), saturation_bound[1].item(), generator=random_generator)

    return {
        "brightness_factor": brightness_factor,
        "contrast_factor": contrast_factor,
        "hue_factor": hue_factor,
        "saturation_factor": saturation_factor,
    }


def _random_prob_gen(batch_size: int, p: float = 0.5, random_generator: Optional[torch.Generator] = None
                     ) -> Dict[str, torch.Tensor]:
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

    probs: torch.Tensor = torch.FloatTensor(batch_size).uniform_(generator=random_generator)  # type: ignore

    batch_prob: torch.Tensor = probs < p

    return {'batch_prob': batch_prob}


def _random_erasing_gen(batch_size: int, height: int, width: int, erase_scale_range: Tuple[float, float],
                        aspect_ratio_range: Tuple[float, float], random_generator: Optional[torch.Generator] = None
                        ) -> Dict[str, torch.Tensor]:
    r""" The rectangle will have an area equal to the original image area multiplied by a value uniformly
         sampled between the range [erase_scale_range[0], erase_scale_range[1]) and an aspect ratio sampled
         between [aspect_ratio_range[0], aspect_ratio_range[1])
    """
    images_area = height * width
    target_areas = torch.FloatTensor(batch_size).uniform_(  # type: ignore
        erase_scale_range[0], erase_scale_range[1], generator=random_generator) * images_area
    if aspect_ratio_range[0] < 1. and aspect_ratio_range[1] > 1.:
        aspect_ratios1 = torch.FloatTensor(batch_size).uniform_(  # type: ignore
            aspect_ratio_range[0], 1, generator=random_generator)
        aspect_ratios2 = torch.FloatTensor(batch_size).uniform_(  # type: ignore
            1, aspect_ratio_range[1], generator=random_generator)
        rand_idxs = torch.round(torch.FloatTensor(batch_size).uniform_(  # type: ignore
            0, 1, generator=random_generator)).bool()
        aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
    else:
        aspect_ratios = torch.FloatTensor(batch_size).uniform_(  # type: ignore
            aspect_ratio_range[0], aspect_ratio_range[1], generator=random_generator)
    # based on target areas and aspect ratios, rectangle params are computed
    heights = torch.min(
        torch.max(torch.round((target_areas * aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
        torch.tensor(float(height))
    ).int()
    widths = torch.min(
        torch.max(torch.round((target_areas / aspect_ratios) ** (1 / 2)), torch.tensor(1.)),
        torch.tensor(float(width))
    ).int()
    xs = (torch.FloatTensor(batch_size).uniform_(  # type: ignore
        0, 1, generator=random_generator) * (torch.tensor(width) - widths + 1).float()).int()
    ys = (torch.FloatTensor(batch_size).uniform_(  # type: ignore
        0, 1, generator=random_generator) * (torch.tensor(height) - heights + 1).float()).int()
    return {
        'widths': widths,
        'heights': heights,
        'xs': xs,
        'ys': ys
    }


def _get_perspective_params(batch_size: int, width: int, height: int, distortion_scale: float,
                            random_generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        [0, width - 1],
        [height - 1, 0],
        [height - 1, width - 1],
    ]]).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx: float = distortion_scale * width / 2
    fy: float = distortion_scale * height / 2

    factor = torch.tensor([fy, fx]).view(-1, 1, 2)

    rand_val: torch.Tensor = torch.FloatTensor(batch_size, 4, 2).uniform_(  # type: ignore
        0, 1, generator=random_generator)
    offset = 2 * factor * rand_val - 1

    end_points = start_points + offset

    return start_points, end_points


def _random_perspective_gen(batch_size: int, height: int, width: int, p: float, distortion_scale: float,
                            random_generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = _random_prob_gen(batch_size, p, random_generator=random_generator)
    start_points, end_points = (
        _get_perspective_params(batch_size, width, height, distortion_scale, random_generator=random_generator)
    )
    params['start_points'] = start_points
    params['end_points'] = end_points
    return params


def _random_affine_gen(
        batch_size: int,
        height: int,
        width: int,
        degrees: UnionFloat,
        translate: Optional[TupleFloat] = None,
        scale: Optional[TupleFloat] = None,
        shear: Optional[UnionFloat] = None,
        random_generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    # check angle ranges
    degrees_tmp: TupleFloat
    if isinstance(degrees, float):
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

    transform: torch.Tensor = _get_random_affine_params(
        batch_size, height, width, degrees_tmp, translate, scale, shear_tmp)
    return dict(transform=transform)


def _random_rotation_gen(batch_size: int, degrees: FloatUnionType, random_generator: Optional[torch.Generator] = None
                         ) -> Dict[str, torch.Tensor]:

    if not torch.is_tensor(degrees):
        if isinstance(degrees, float):
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

    params: Dict[str, torch.Tensor] = {}
    params["degrees"] = torch.FloatTensor(batch_size).uniform_(  # type: ignore
        degrees[0].item(), degrees[1].item(), generator=random_generator)

    return params


def _get_random_affine_params(
    batch_size: int, height: int, width: int, degrees: TupleFloat, translate: Optional[TupleFloat],
    scales: Optional[TupleFloat], shears: Optional[TupleFloat], random_generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    r"""Get parameters for affine transformation. The returned matrix is Bx3x3.

    Returns:
        torch.Tensor: params to be passed to the affine transformation.
    """
    angle = torch.FloatTensor(batch_size).uniform_(degrees[0], degrees[1], generator=random_generator)  # type: ignore

    # compute tensor ranges
    if scales is not None:
        scale = torch.FloatTensor(batch_size).uniform_(scales[0], scales[1], generator=random_generator)  # type: ignore
    else:
        scale = torch.ones(batch_size)

    if shears is not None:
        shear = torch.FloatTensor(batch_size).uniform_(shears[0], shears[1], generator=random_generator)  # type: ignore
    else:
        shear = torch.zeros(batch_size)

    if translate is not None:
        max_dx: float = translate[0] * width
        max_dy: float = translate[1] * height
        translations = torch.stack([
            torch.FloatTensor(batch_size).uniform_(-max_dx, max_dx, generator=random_generator),  # type: ignore
            torch.FloatTensor(batch_size).uniform_(-max_dy, max_dy, generator=random_generator)   # type: ignore
        ], dim=-1)
    else:
        translations = torch.zeros(batch_size, 2)

    center: torch.Tensor = torch.tensor(
        [width, height], dtype=torch.float32).view(1, 2) / 2
    center = center.expand(batch_size, -1)

    # concatenate transforms
    transform: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    transform[..., 2] += translations  # tx/ty
    transform[..., 0, 1] += shear
    transform[..., 1, 0] += shear

    # pad transform to get Bx3x3
    transform_h = torch.nn.functional.pad(transform, [0, 0, 0, 1], value=0.)
    transform_h[..., -1, -1] += 1.0
    return transform_h


def _center_crop_gen(size: Union[int, Tuple[int, int]], random_generator: Optional[torch.Generator] = None
                     ) -> Dict[str, torch.Tensor]:
    if isinstance(size, tuple):
        size_param = torch.tensor([size[0], size[1]])
    elif isinstance(size, int):
        size_param = torch.tensor([size, size])
    else:
        raise Exception(f"Invalid size type. Expected (int, tuple(int, int). "
                        f"Got: {type(size)}.")
    return dict(size=size_param)


def _random_crop_gen(batch_size: int, input_size: Tuple[int, int], size: Tuple[int, int],
                     resize_to: Optional[Tuple[int, int]] = None, random_generator: Optional[torch.Generator] = None
                     ) -> Dict[str, torch.Tensor]:
    x_diff = input_size[1] - size[1]
    y_diff = input_size[0] - size[0]

    if x_diff < 0 or y_diff < 0:
        raise ValueError("input_size %s cannot be smaller than crop size %s in any dimension."
                         % (str(input_size), str(size)))

    x_start = torch.FloatTensor(batch_size).uniform_(0, x_diff + 1, generator=random_generator).long()  # type: ignore
    y_start = torch.FloatTensor(batch_size).uniform_(0, y_diff + 1, generator=random_generator).long()  # type: ignore

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

    return {'src': crop_src, 'dst': crop_dst}


def _random_crop_size_gen(size: Tuple[int, int], scale: Tuple[float, float],
                          ratio: Tuple[float, float], random_generator: Optional[torch.Generator] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    area = torch.FloatTensor(10).uniform_(  # type: ignore
        scale[0] * size[0] * size[1], scale[1] * size[0] * size[1], generator=random_generator)
    log_ratio = torch.FloatTensor(10).uniform_(  # type: ignore
        math.log(ratio[0]), math.log(ratio[1]), generator=random_generator)
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


def _random_resized_crop_gen(batch_size: int, input_size: Tuple[int, int], size: Tuple[int, int],
                             scale: Tuple[float, float], ratio: Tuple[float, float],
                             random_generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    target_size = _random_crop_size_gen(size, scale, ratio, random_generator=random_generator)
    return _random_crop_gen(batch_size=batch_size, input_size=input_size,
                            size=(int(target_size[0].data.item()), int(target_size[1].data.item())), resize_to=size,
                            random_generator=random_generator)
