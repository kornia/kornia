from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample, BorderType
from .utils import (
    _adapted_uniform,
    _check_and_bound
)


def random_rotation_generator3d(
    batch_size: int,
    degrees: Union[torch.Tensor, float, Tuple[float, float, float],
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (float or tuple or list): Range of degrees to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
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
            yaw = torch.tensor([-degrees, degrees]).to(torch.float32)
            pitch = torch.tensor([-degrees, degrees]).to(torch.float32)
            roll = torch.tensor([-degrees, degrees]).to(torch.float32)

        elif isinstance(degrees, (tuple)) and isinstance(degrees[0], (float, int)) and len(degrees) == 2:
            yaw = torch.tensor(degrees).to(torch.float32)
            pitch = torch.tensor(degrees).to(torch.float32)
            roll = torch.tensor(degrees).to(torch.float32)

        elif isinstance(degrees, (tuple)) and len(degrees) == 3 \
            and isinstance(degrees[0], (tuple)) and isinstance(degrees[1], (tuple)) and isinstance(degrees[2], (tuple)):
            yaw = torch.tensor(degrees[0]).to(torch.float32)
            pitch = torch.tensor(degrees[1]).to(torch.float32)
            roll = torch.tensor(degrees[2]).to(torch.float32)

        else:
            raise TypeError(f"Degrees should be a float number a sequence or a tensor. Got {type(degrees)}")
    else:
        # https://mypy.readthedocs.io/en/latest/casts.html cast to please mypy gods
        degrees = cast(torch.Tensor, degrees)
        if len(degrees) != 3 and len(degrees[0]) != len(degrees[1]) != len(degrees[2]) != 2:
            raise ValueError(
                f"Degrees must be a 3x2 tensor for the degree range of yaw, pitch, roll. Got {degrees.shape}")
        yaw = degrees[0]
        pitch = degrees[1]
        roll = degrees[2]

    yaw = _adapted_uniform((batch_size,), yaw[0], yaw[1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), pitch[0], pitch[1], same_on_batch)
    roll = _adapted_uniform((batch_size,), roll[0], roll[1], same_on_batch)

    return dict(yaw=yaw,
                pitch=pitch,
                roll=roll,
                interpolation=torch.tensor(Resample.get(interpolation).value),
                align_corners=torch.tensor(align_corners))
