from typing import Tuple, List, Union, Dict

import torch
from torch.distributions import Uniform

from kornia.geometry import pi


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


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
