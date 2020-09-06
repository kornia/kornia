from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample, BorderType
from kornia.augmentation.utils import (
    _adapted_uniform,
    _tuple_range_reader,
)
from .types import AugParamDict
from .random_generator3d import (
    rotation_params_generator3d,
    affine_params_generator3d
)


def random_rotation_generator3d(
    batch_size: int,
    degrees: torch.Tensor,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> AugParamDict:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): Ranges of degrees (3, 2) for yaw, pitch and roll.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners (bool): interpolation flag. Default: False.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = rotation_params_generator3d(batch_size, degrees=degrees, same_on_batch=same_on_batch)
    flags = dict(
        interpolation=torch.tensor(Resample.get(interpolation).value),
        align_corners=torch.tensor(align_corners)
    )

    return AugParamDict(dict(params=params, flags=flags))


def random_affine_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    degrees: torch.Tensor,
    translate: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shears: Optional[torch.Tensor] = None,
    resample: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> AugParamDict:
    r"""Get parameters for ```3d affine``` transformation random affine transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : height of the image.
        height (int) : height of the image.
        width (int): width of the image.
        degrees (torch.Tensor): Ranges of degrees with shape (3, 2) for yaw, pitch and roll.
        translate (torch.Tensor, optional):  maximum absolute fraction with shape (3,) for horizontal, vertical
            and depthical translations. Will not translate by default.
        scale (torch.Tensor, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float, optional): Range of degrees to select from.
            If shear is a number, a shear to the 6 facets in the range (-shear, +shear) will be apllied.
            If shear is a tuple of 2 values, a shear to the 6 facets in the range (shear[0], shear[1]) will be applied.
            If shear is a tuple of 6 values, a shear to the i-th facet in the range (-shear[i], shear[i])
            will be applied.
            If shear is a tuple of 6 tuples, a shear to the i-th facet in the range (-shear[i, 0], shear[i, 1])
            will be applied.
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners(bool): interpolation flag. Default: False.See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: parameters to be passed for transformation.
            - params['params']: element-wise parameters generated.
            - params['flags']: static flags for the transformation.
    """
    params = rotation_params_generator3d(batch_size, degrees=degrees, same_on_batch=same_on_batch)
    flags = dict(
        resample=torch.tensor(Resample.get(resample).value),
        align_corners=torch.tensor(align_corners)
    )

    return AugParamDict(dict(params=params, flags=flags))
