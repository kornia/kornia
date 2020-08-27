from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample, BorderType
from .utils import (
    _adapted_uniform,
    _tuple_range_reader,
)


def random_rotation_generator3d(
    batch_size: int,
    degrees: torch.Tensor,
    interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): Ranges of degrees (3, 2) for yaw, pitch and roll.
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners (bool): interpolation flag. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    assert degrees.shape == torch.Size([3, 2]), f"'degrees' must be the shape of (3, 2). Got {degrees.shape}."
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)

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
    degrees: torch.Tensor,
    translate: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shears: Optional[torch.Tensor] = None,
    resample: Union[str, int, Resample] = Resample.BILINEAR.name,
    same_on_batch: bool = False,
    align_corners: bool = False
) -> Dict[str, torch.Tensor]:
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
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    assert degrees.shape == torch.Size([3, 2]), f"'degrees' must be the shape of (3, 2). Got {degrees.shape}."
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)
    angles = torch.cat([yaw, pitch, roll], dim=-1).view((batch_size, -1))

    # compute tensor ranges
    if scale is not None:
        assert scale.shape == torch.Size([3, 2]), f"'scale' must be the shape of (3, 2). Got {scale.shape}."
        scale = torch.cat([
            _adapted_uniform((batch_size,), scale[0, 0], scale[0, 1], same_on_batch),
            _adapted_uniform((batch_size,), scale[1, 0], scale[1, 1], same_on_batch),
            _adapted_uniform((batch_size,), scale[2, 0], scale[2, 1], same_on_batch),
        ], dim=1)
    else:
        scale = torch.ones(batch_size).repeat(1, 3)

    if translate is not None:
        assert translate.shape == torch.Size([3]), f"'translate' must be the shape of (2). Got {translate.shape}."
        max_dx: torch.Tensor = translate[0] * depth
        max_dy: torch.Tensor = translate[1] * width
        max_dz: torch.Tensor = translate[2] * height
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
        assert shears.shape == torch.Size([6, 2]), f"'shears' must be the shape of (6, 2). Got {shears.shape}."
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
