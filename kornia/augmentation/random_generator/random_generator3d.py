from typing import Tuple, List, Union, Dict, Optional, cast
import random
import math

import torch

from kornia.constants import Resample, BorderType
from kornia.geometry import bbox_generator3d
from ..utils import (
    _adapted_uniform,
    _tuple_range_reader,
)


def random_rotation_generator3d(
    batch_size: int,
    degrees: torch.Tensor,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): Ranges of degrees (3, 2) for yaw, pitch and roll.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    assert degrees.shape == torch.Size([3, 2]), f"'degrees' must be the shape of (3, 2). Got {degrees.shape}."
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)

    return dict(yaw=yaw,
                pitch=pitch,
                roll=roll)


def random_affine_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    degrees: torch.Tensor,
    translate: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shears: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```3d affine``` transformation random affine transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : depth of the image.
        height (int) : height of the image.
        width (int): width of the image.
        degrees (torch.Tensor): Ranges of degrees with shape (3, 2) for yaw, pitch and roll.
        translate (torch.Tensor, optional):  maximum absolute fraction with shape (3,) for horizontal, vertical
            and depthical translations (dx,dy,dz). Will not translate by default.
        scale (torch.Tensor, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float, optional): Range of degrees to select from.
            If shear is a number, a shear to the 6 facets in the range (-shear, +shear) will be apllied.
            If shear is a tuple of 2 values, a shear to the 6 facets in the range (shear[0], shear[1]) will be applied.
            If shear is a tuple of 6 values, a shear to the i-th facet in the range (-shear[i], shear[i])
            will be applied.
            If shear is a tuple of 6 tuples, a shear to the i-th facet in the range (-shear[i, 0], shear[i, 1])
            will be applied.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    assert degrees.shape == torch.Size([3, 2]), f"'degrees' must be the shape of (3, 2). Got {degrees.shape}."
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)
    angles = torch.stack([yaw, pitch, roll], dim=1)

    # compute tensor ranges
    if scale is not None:
        assert scale.shape == torch.Size([3, 2]), f"'scale' must be the shape of (3, 2). Got {scale.shape}."
        scale = torch.stack([
            _adapted_uniform((batch_size,), scale[0, 0], scale[0, 1], same_on_batch),
            _adapted_uniform((batch_size,), scale[1, 0], scale[1, 1], same_on_batch),
            _adapted_uniform((batch_size,), scale[2, 0], scale[2, 1], same_on_batch),
        ], dim=1)
    else:
        scale = torch.ones(batch_size).repeat(1, 3)

    if translate is not None:
        assert translate.shape == torch.Size([3]), f"'translate' must be the shape of (2). Got {translate.shape}."
        max_dx: torch.Tensor = translate[0] * width
        max_dy: torch.Tensor = translate[1] * height
        max_dz: torch.Tensor = translate[2] * depth
        # translations should be in x,y,z
        translations = torch.stack([
            _adapted_uniform((batch_size,), -max_dx, max_dx, same_on_batch),
            _adapted_uniform((batch_size,), -max_dy, max_dy, same_on_batch),
            _adapted_uniform((batch_size,), -max_dz, max_dz, same_on_batch)
        ], dim=1)
    else:
        translations = torch.zeros(batch_size, 3)

    # center should be in x,y,z
    center: torch.Tensor = torch.tensor(
        [width, height, depth], dtype=torch.float32).view(1, 3) / 2. - 0.5
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
                szy=szy)


def random_motion_blur_generator3d(
    batch_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    angle: torch.Tensor,
    direction: torch.Tensor,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for motion blur.

    Args:
        batch_size (int): the tensor batch size.
        kernel_size (int or (int, int)): motion kernel size (odd and positive) or range.
        angle (torch.Tensor): yaw, pitch and roll range of the motion blur in degrees :math:`(3, 2)`.
        direction (torch.Tensor): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with
            angle provided via angle), while higher values towards 1.0 will point the motion
            blur forward. A value of 0.0 leads to a uniformly (but still angled) motion blur.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if isinstance(kernel_size, int):
        ksize_factor = torch.tensor([kernel_size] * batch_size)
    elif isinstance(kernel_size, tuple):
        # kernel_size is fixed across the batch
        ksize_factor = _adapted_uniform(
            (batch_size,), kernel_size[0] // 2, kernel_size[1] // 2, same_on_batch=True).int() * 2 + 1
    else:
        raise TypeError(f"Unsupported type: {type(kernel_size)}")

    assert angle.shape == torch.Size([3, 2]), f"'angle' must be the shape of (3, 2). Got {angle.shape}."
    yaw = _adapted_uniform((batch_size,), angle[0][0], angle[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), angle[1][0], angle[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), angle[2][0], angle[2][1], same_on_batch)
    angle_factor = torch.stack([yaw, pitch, roll], dim=1)

    direction_factor = _adapted_uniform(
        (batch_size,), direction[0], direction[1], same_on_batch)

    return dict(ksize_factor=ksize_factor,
                angle_factor=angle_factor,
                direction_factor=direction_factor)


def center_crop_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    size: Tuple[int, int, int]
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```center_crop3d``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : depth of the image.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (d, h, w).

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if not isinstance(size, (tuple, list,)) and len(size) == 3:
        raise ValueError("Input size must be a tuple/list of length 3. Got {}"
                         .format(size))

    # unpack input sizes
    dst_d, dst_h, dst_w = size
    src_d, src_h, src_w = (depth, height, width)

    # compute start/end offsets
    dst_d_half = dst_d / 2
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_d_half = src_d / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = src_w_half - dst_w_half
    start_y = src_h_half - dst_h_half
    start_z = src_d_half - dst_d_half

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1
    end_z = start_z + dst_d - 1
    # [x, y, z] origin
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_src: torch.Tensor = torch.tensor([[
        [start_x, start_y, start_z],
        [end_x, start_y, start_z],
        [end_x, end_y, start_z],
        [start_x, end_y, start_z],
        [start_x, start_y, end_z],
        [end_x, start_y, end_z],
        [end_x, end_y, end_z],
        [start_x, end_y, end_z],
    ]])

    # [x, y, z] destination
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0, 0],
        [dst_w - 1, 0, 0],
        [dst_w - 1, dst_h - 1, 0],
        [0, dst_h - 1, 0],
        [0, 0, dst_d - 1],
        [dst_w - 1, 0, dst_d - 1],
        [dst_w - 1, dst_h - 1, dst_d - 1],
        [0, dst_h - 1, dst_d - 1],
    ]]).expand(points_src.shape[0], -1, -1)
    return dict(src=points_src,
                dst=points_dst)


def random_crop_generator3d(
    batch_size: int,
    input_size: Tuple[int, int, int],
    size: Union[Tuple[int, int, int], torch.Tensor],
    resize_to: Optional[Tuple[int, int, int]] = None,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        batch_size (int): the tensor batch size.
        input_size (tuple): Input image shape, like (d, h, w).
        size (tuple): Desired size of the crop operation, like (d, h, w).
            If tensor, it must be (B, 3).
        resize_to (tuple): Desired output size of the crop, like (d, h, w). If None, no resize will be performed.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size).repeat(batch_size, 1)
    assert size.shape == torch.Size([batch_size, 3]), \
        f"If `size` is a tensor, it must be shaped as (B, 3). Got {size.shape}."

    x_diff = input_size[2] - size[:, 2] + 1
    y_diff = input_size[1] - size[:, 1] + 1
    z_diff = input_size[0] - size[:, 0] + 1

    if (x_diff < 0).any() or (y_diff < 0).any() or (z_diff < 0).any():
        raise ValueError("input_size %s cannot be smaller than crop size %s in any dimension."
                         % (str(input_size), str(size)))

    if same_on_batch:
        # If same_on_batch, select the first then repeat.
        x_start = _adapted_uniform((batch_size,), 0, x_diff[0], same_on_batch).long()
        y_start = _adapted_uniform((batch_size,), 0, y_diff[0], same_on_batch).long()
        z_start = _adapted_uniform((batch_size,), 0, z_diff[0], same_on_batch).long()
    else:
        x_start = _adapted_uniform((1,), 0, x_diff, same_on_batch).long()
        y_start = _adapted_uniform((1,), 0, y_diff, same_on_batch).long()
        z_start = _adapted_uniform((1,), 0, z_diff, same_on_batch).long()

    crop_src = bbox_generator3d(x_start.view(-1), y_start.view(-1), z_start.view(-1),
                                size[:, 2] - 1, size[:, 1] - 1, size[:, 0] - 1)

    if resize_to is None:
        crop_dst = bbox_generator3d(
            torch.tensor([0] * batch_size), torch.tensor([0] * batch_size), torch.tensor([0] * batch_size),
            size[:, 2] - 1, size[:, 1] - 1, size[:, 0] - 1)
    else:
        crop_dst = torch.tensor([[
            [0, 0, 0],
            [resize_to[-1] - 1, 0, 0],
            [resize_to[-1] - 1, resize_to[-2] - 1, 0],
            [0, resize_to[-2] - 1, 0],
            [0, 0, resize_to[-3] - 1],
            [resize_to[-1] - 1, 0, resize_to[-3] - 1],
            [resize_to[-1] - 1, resize_to[-2] - 1, resize_to[-3] - 1],
            [0, resize_to[-2] - 1, resize_to[-3] - 1],
        ]]).repeat(batch_size, 1, 1)

    return dict(src=crop_src,
                dst=crop_dst)


def random_perspective_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    distortion_scale: torch.Tensor,
    same_on_batch: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : depth of the image.
        height (int) : height of the image.
        width (int): width of the image.
        distortion_scale (torch.Tensor): it controls the degree of distortion and ranges from 0 to 1.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params (Dict[str, torch.Tensor])
    """
    assert distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1, \
        f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}"

    start_points: torch.Tensor = torch.tensor([[
        [0., 0, 0],
        [width - 1, 0, 0],
        [width - 1, height - 1, 0],
        [0, height - 1, 0],
        [0., 0, depth - 1],
        [width - 1, 0, depth - 1],
        [width - 1, height - 1, depth - 1],
        [0, height - 1, depth - 1],
    ]]).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx = distortion_scale * width / 2
    fy = distortion_scale * height / 2
    fz = distortion_scale * depth / 2

    factor = torch.stack([fx, fy, fz], dim=0).view(-1, 1, 3)

    # TODO: This line somehow breaks the gradcheck
    rand_val: torch.Tensor = _adapted_uniform(start_points.shape, 0, 1, same_on_batch)

    pts_norm = torch.tensor([[
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
    ]])
    end_points = start_points + factor * rand_val * pts_norm

    return dict(start_points=start_points,
                end_points=end_points)
