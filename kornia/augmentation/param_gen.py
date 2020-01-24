from typing import Tuple, List, Union, Dict, Optional

import torch
from torch.distributions import Uniform

from kornia.geometry import pi
from kornia.geometry.transform import get_rotation_matrix2d


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]

TupleInt = Tuple[int, int]
TupleFloat = Tuple[float, float]
UnionFloat = Union[float, TupleFloat]


def _random_color_jitter_gen(batch_size: int, brightness: FloatUnionType = 0.,
                             contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                             hue: FloatUnionType = 0.) -> Dict[str, torch.Tensor]:
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

    brightness_distribution = Uniform(brightness_bound[0], brightness_bound[1])
    brightness_factor = brightness_distribution.rsample([batch_size])

    contrast_distribution = Uniform(contrast_bound[0], contrast_bound[1])
    contrast_factor = contrast_distribution.rsample([batch_size])

    hue_distribution = Uniform(hue_bound[0], hue_bound[1])
    hue_factor = hue_distribution.rsample([batch_size])

    saturation_distribution = Uniform(saturation_bound[0], saturation_bound[1])
    saturation_factor = saturation_distribution.rsample([batch_size])

    return {
        "brightness_factor": brightness_factor,
        "contrast_factor": contrast_factor,
        "hue_factor": hue_factor,
        "saturation_factor": saturation_factor,
    }


def _random_prob_gen(batch_size: int, p: float = 0.5) -> Dict[str, torch.Tensor]:
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

    probs: torch.Tensor = Uniform(0, 1).rsample((batch_size,))

    batch_prob: torch.Tensor = probs < p

    return {'batch_prob': batch_prob}


def _get_perspective_params(batch_size: int, width: int, height: int, distortion_scale: float
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    rand_val: torch.Tensor = Uniform(0, 1).rsample((batch_size, 4, 2))
    offset = 2 * factor - 1

    end_points = start_points + offset

    return start_points, end_points


def _random_perspective_gen(
    batch_size: int, height: int, width: int, p: float, distortion_scale: float
) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = _random_prob_gen(batch_size, p)
    start_points, end_points = (
        _get_perspective_params(batch_size, width, height, distortion_scale)
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
        shear: Optional[UnionFloat] = None) -> Dict[str, torch.Tensor]:
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


def _get_random_affine_params(
    batch_size: int, height: int, width: int,
    degrees: TupleFloat, translate: Optional[TupleFloat],
    scales: Optional[TupleFloat], shears: Optional[TupleFloat],
) -> torch.Tensor:
    r"""Get parameters for affine transformation. The returned matrix is Bx3x3.

    Returns:
        torch.Tensor: params to be passed to the affine transformation.
    """
    angle = Uniform(degrees[0], degrees[1]).rsample((batch_size,))

    # compute tensor ranges
    if scales is not None:
        scale = Uniform(scales[0], scales[1]).rsample((batch_size,))
    else:
        scale = torch.ones(batch_size)

    if shears is not None:
        shear = Uniform(shears[0], shears[1]).rsample((batch_size,))
    else:
        shear = torch.zeros(batch_size)

    if translate is not None:
        max_dx: float = translate[0] * width
        max_dy: float = translate[1] * height
        translations = torch.stack([
            Uniform(-max_dx, max_dx).rsample((batch_size,)),
            Uniform(-max_dy, max_dy).rsample((batch_size,)),
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
