from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample, BorderType
from .utils import (
    _adapted_uniform,
    _check_and_bound,
    _tuple_range_reader,
)


def random_rotation_generator3d(
    batch_size: int,
    degrees: Union[torch.Tensor, float, Tuple[float, float], Tuple[float, float, float],
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
    _degrees = _tuple_range_reader(degrees, 3)

    yaw = _adapted_uniform((batch_size,), _degrees[0][0], _degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), _degrees[1][0], _degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), _degrees[2][0], _degrees[2][1], same_on_batch)

    return dict(yaw=yaw,
                pitch=pitch,
                roll=roll,
                interpolation=torch.tensor(Resample.get(interpolation).value),
                align_corners=torch.tensor(align_corners))


def random_affine_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    degrees: Union[torch.Tensor, float, Tuple[float, float], Tuple[float, float, float],
                   Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    translate: Optional[Tuple[float, float, float]] = None,
    scale: Optional[Tuple[float, float]] = None,
    shears: Union[torch.Tensor, float, Tuple[float, float], Tuple[float, float, float, float, float, float],
                  Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float],
                  Tuple[float, float], Tuple[float, float]]] = None,
    resample: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``affine`` for a random affine transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : height of the image.
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
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    degrees_tmp = _tuple_range_reader(degrees, 3)
    if shears is not None:
        shears_tmp = _tuple_range_reader(shears, 6)
    else:
        shears_tmp = None

    # check translation range
    if translate is not None:
        assert isinstance(translate, (tuple, list)) and len(translate) == 3, \
            "translate should be a list or tuple and it must be of length 3."
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

    return _get_random_affine_params(
        batch_size, depth, height, width, degrees_tmp, translate, scale, shears_tmp,
        resample, same_on_batch, align_corners)


def _get_random_affine_params(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    degrees: torch.Tensor,
    translate: Optional[Tuple[float, float, float]],
    scales: Optional[Tuple[float, float]],
    shears: torch.Tensor = None,
    resample: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```3d affine``` transformation random affine transform.
    The returned matrix is Bx4x4.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)
    angles = torch.cat([yaw, pitch, roll], dim=-1).view((batch_size, -1))

    # compute tensor ranges
    if scales is not None:
        scale = _adapted_uniform((batch_size,), scales[0], scales[1], same_on_batch)
    else:
        scale = torch.ones(batch_size)

    if translate is not None:
        max_dx: float = translate[0] * depth
        max_dy: float = translate[1] * width
        max_dz: float = translate[2] * height
        translations = torch.stack([
            _adapted_uniform((batch_size,), -max_dx, max_dx, same_on_batch),
            _adapted_uniform((batch_size,), -max_dy, max_dy, same_on_batch),
            _adapted_uniform((batch_size,), -max_dz, max_dz, same_on_batch)
        ], dim=-1)
    else:
        translations = torch.zeros(batch_size, 3)

    center: torch.Tensor = torch.tensor(
        [depth, width, height], dtype=torch.float32).view(1, 3) / 2. - 0.5
    center = center.expand(batch_size, -1)

    if shears is not None:
        sxy = _adapted_uniform((batch_size,), shears[0, 0], shears[0, 1], same_on_batch)
        sxz = _adapted_uniform((batch_size,), shears[1, 0], shears[1, 1], same_on_batch)
        syx = _adapted_uniform((batch_size,), shears[2, 0], shears[2, 1], same_on_batch)
        syz = _adapted_uniform((batch_size,), shears[3, 0], shears[3, 1], same_on_batch)
        szx = _adapted_uniform((batch_size,), shears[4, 0], shears[4, 1], same_on_batch)
        szy = _adapted_uniform((batch_size,), shears[5, 0], shears[5, 1], same_on_batch)
    else:
        sxy = sxz = syx = syz = szx = szy = torch.tensor([0] * batch_size)

    return dict(translations=translations,
                center=center,
                scale=scale,
                angles=angles,
                sxy=sxy,
                sxz=sxz,
                syx=syx,
                syz=syz,
                szx=szx,
                szy=szy,
                resample=torch.tensor(Resample.get(resample).value),
                align_corners=torch.tensor(align_corners))
