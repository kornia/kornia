from typing import Tuple, List, Union, Dict, cast, Optional

import torch

import kornia as K
from kornia.constants import Resample, BorderType, pi
from kornia.geometry.transform.affwarp import (
    _compute_rotation_matrix3d, _compute_tensor_center3d
)
from kornia.geometry.transform.projwarp import warp_affine3d
from kornia.geometry import (
    crop_by_boxes3d,
    warp_perspective3d,
    get_perspective_transform3d,
    rotate3d,
    get_affine_matrix3d,
    deg2rad
)
from kornia.enhance import (
    equalize3d
)

from .. import random_generator as rg
from ..utils import (
    _validate_input3d
)
from kornia.filters import motion_blur3d

from .__temp__ import __deprecation_warning, _deprecation_wrapper


@_deprecation_wrapper
@_validate_input3d
def apply_hflip3d(input: torch.Tensor) -> torch.Tensor:
    r"""Apply horizontal flip on a 3D tensor volume or a batch of tensors volumes with given random parameters.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.

    Returns:
        torch.Tensor: Horizontal flipped input with shape :math:`(*, C, D, H, W)`.
    """

    return torch.flip(input, [-1])


@_deprecation_wrapper
@_validate_input3d
def compute_hflip_transformation3d(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the horizontal flip transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.

    Returns:
        torch.Tensor: Horizontal flip transformation matrix :math: `(*, 4, 4)`.
    """

    w: int = input.shape[-1]
    flip_mat: torch.Tensor = torch.tensor([[-1, 0, 0, w - 1],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

    return flip_mat.repeat(input.size(0), 1, 1).to(input)


@_deprecation_wrapper
@_validate_input3d
def apply_vflip3d(input: torch.Tensor) -> torch.Tensor:
    r"""Apply vertical flip on a 3D tensor volume or a batch of tensors volumes with given random parameters.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.

    Returns:
        torch.Tensor: Vertical flipped input with shape :math:`(*, C, D, H, W)`.
    """

    return torch.flip(input, [-2])


@_deprecation_wrapper
@_validate_input3d
def compute_vflip_transformation3d(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the veritical flip transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.

    Returns:
        torch.Tensor: The vertical flip transformation matrix :math: `(*, 4, 4)`.
    """

    h: int = input.shape[-2]
    flip_mat: torch.Tensor = torch.tensor([[1, 0, 0, 0],
                                           [0, -1, 0, h - 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

    return flip_mat.repeat(input.size(0), 1, 1).to(input)


@_deprecation_wrapper
@_validate_input3d
def apply_dflip3d(input: torch.Tensor) -> torch.Tensor:
    r"""Apply depthical flip on a 3D tensor volume or a batch of tensors volumes with given random parameters.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.

    Returns:
        torch.Tensor: Depthical flipped input with shape :math:`(*, C, D, H, W)`.
    """

    return torch.flip(input, [-3])


@_deprecation_wrapper
@_validate_input3d
def compute_intensity_transformation3d(input: torch.Tensor):
    r"""Compute the identity matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.

    Returns:
        torch.Tensor: Identity matrix :math: `(*, 4, 4)`.
    """
    identity: torch.Tensor = torch.eye(4, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
    return identity


@_deprecation_wrapper
@_validate_input3d
def compute_dflip_transformation3d(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the depthical flip transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.

    Returns:
        torch.Tensor: Depthical flip transformation matrix :math: `(*, 4, 4)`.
    """

    d: int = input.shape[-3]
    flip_mat: torch.Tensor = torch.tensor([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, -1, d - 1],
                                           [0, 0, 0, 1]])

    return flip_mat.repeat(input.size(0), 1, 1).to(input)


@_deprecation_wrapper
@_validate_input3d
def apply_affine3d(input: torch.Tensor, params: Dict[str, torch.Tensor],
                   flags: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Random affine transformation of the image keeping center invariant.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['angles']: Degrees of rotation with the shape of :math: `(*, 3)` for yaw, pitch, roll.
            - params['translations']: Horizontal, vertical and depthical translations (dx,dy,dz).
            - params['center']: Rotation center (x,y,z).
            - params['scale']: Isotropic scaling params.
            - params['sxy']: Shear param toward x-y-axis.
            - params['sxz']: Shear param toward x-z-axis.
            - params['syx']: Shear param toward y-x-axis.
            - params['syz']: Shear param toward y-z-axis.
            - params['szx']: Shear param toward z-x-axis.
            - params['szy']: Shear param toward z-y-axis.
        flags (Dict[str, torch.Tensor]):
            - params['resample']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: Affine transfromed input with shape :math:`(*, C, D, H, W)`.
    """

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-4:])

    depth, height, width = x_data.shape[-3:]

    # concatenate transforms
    transform: torch.Tensor = compute_affine_transformation3d(input, params)

    resample_name: str = Resample(flags['resample'].item()).name.lower()
    align_corners: bool = cast(bool, flags['align_corners'].item())

    out_data: torch.Tensor = warp_affine3d(x_data, transform[:, :3, :],
                                           (depth, height, width), resample_name,
                                           align_corners=align_corners)
    return out_data.view_as(input)


@_deprecation_wrapper
@_validate_input3d
def compute_affine_transformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the affine transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['angles']: Degrees of rotation with the shape of :math: `(*, 3)` for yaw, pitch, roll.
            - params['translations']: Horizontal, vertical and depthical translations (dx,dy,dz).
            - params['center']: Rotation center (x,y,z).
            - params['scale']: Isotropic scaling params.
            - params['sxy']: Shear param toward x-y-axis.
            - params['sxz']: Shear param toward x-z-axis.
            - params['syx']: Shear param toward y-x-axis.
            - params['syz']: Shear param toward y-z-axis.
            - params['szx']: Shear param toward z-x-axis.
            - params['szy']: Shear param toward z-y-axis.

    Returns:
        torch.Tensor: The affine transformation matrix :math: `(*, 4, 4)`.
    """
    transform = get_affine_matrix3d(
        params['translations'], params['center'], params['scale'], params['angles'],
        deg2rad(params['sxy']), deg2rad(params['sxz']), deg2rad(params['syx']),
        deg2rad(params['syz']), deg2rad(params['szx']), deg2rad(params['szy'])
    ).to(input)
    return transform


@_deprecation_wrapper
@_validate_input3d
def apply_rotation3d(input: torch.Tensor, params: Dict[str, torch.Tensor],
                     flags: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Rotate a tensor image or a batch of tensor images a random amount of degrees.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['degrees']: degree to be applied.
        flags (Dict[str, torch.Tensor]):
            - params['resample']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: The cropped input.
    """
    yaw: torch.Tensor = params["yaw"].to(input)
    pitch: torch.Tensor = params["pitch"].to(input)
    roll: torch.Tensor = params["roll"].to(input)

    resample_mode: str = Resample(flags['resample'].item()).name.lower()
    align_corners: bool = cast(bool, flags['align_corners'].item())

    transformed: torch.Tensor = rotate3d(input, yaw, pitch, roll, mode=resample_mode, align_corners=align_corners)

    return transformed


@_deprecation_wrapper
@_validate_input3d
def compute_rotate_tranformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]):
    r"""Compute the rotation transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['yaw']: degree to be applied.
            - params['pitch']: degree to be applied.
            - params['roll']: degree to be applied.

    Returns:
        torch.Tensor: The rotation transformation matrix :math: `(*, 4, 4)`.
    """
    yaw: torch.Tensor = params["yaw"].to(input)
    pitch: torch.Tensor = params["pitch"].to(input)
    roll: torch.Tensor = params["roll"].to(input)

    center: torch.Tensor = _compute_tensor_center3d(input)
    rotation_mat: torch.Tensor = _compute_rotation_matrix3d(yaw, pitch, roll, center.expand(yaw.shape[0], -1))

    # rotation_mat is B x 3 x 4 and we need a B x 4 x 4 matrix
    trans_mat: torch.Tensor = torch.eye(4, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
    trans_mat[:, 0] = rotation_mat[:, 0]
    trans_mat[:, 1] = rotation_mat[:, 1]
    trans_mat[:, 2] = rotation_mat[:, 2]

    return trans_mat


@_deprecation_wrapper
@_validate_input3d
def apply_motion_blur3d(input: torch.Tensor, params: Dict[str, torch.Tensor],
                        flags: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Perform motion blur on an image.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['ksize_factor']: motion kernel width and height (odd and positive).
            - params['angle_factor']: yaw, pitch and roll range of the motion blur in degrees :math:`(B, 3)`.
            - params['direction_factor']: forward/backward direction of the motion blur.
              Lower values towards -1.0 will point the motion blur towards the back (with
              angle provided via angle), while higher values towards 1.0 will point the motion
              blur forward. A value of 0.0 leads to a uniformly (but still angled) motion blur.
        flags (Dict[str, torch.Tensor]):
            - flags['border_type']: the padding mode to be applied before convolving.
              CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3. Default: BorderType.CONSTANT.

    Returns:
        torch.Tensor: adjusted image tensor with shape :math:`(*, C, D, H, W)`.
    """

    kernel_size: int = cast(int, params['ksize_factor'].unique().item())
    angle = params['angle_factor']
    direction = params['direction_factor']
    border_type: str = cast(str, BorderType(flags['border_type'].item()).name.lower())
    mode: str = cast(str, Resample(flags['interpolation'].item()).name.lower())

    return motion_blur3d(input, kernel_size, angle, direction, border_type, mode)


@_deprecation_wrapper
@_validate_input3d
def apply_crop3d(input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Apply cropping by src bounding box and dst bounding box.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['src']: The applied cropping src matrix :math: `(*, 8, 3)`.
            - params['dst']: The applied cropping dst matrix :math: `(*, 8, 3)`.
        flags (Dict[str, torch.Tensor]):
            - params['interpolation']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: The cropped input.

    Note:
        BBox order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
        back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
    """

    resample_mode: str = Resample.get(flags['interpolation'].item()).name.lower()  # type: ignore
    align_corners: bool = cast(bool, flags['align_corners'].item())

    return crop_by_boxes3d(
        input, params['src'], params['dst'], resample_mode, align_corners=align_corners)


@_deprecation_wrapper
@_validate_input3d
def compute_crop_transformation3d(
    input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]
) -> torch.Tensor:
    r"""Compute the cropping transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['src']: The applied cropping src matrix :math: `(*, 8, 3)`.
            - params['dst']: The applied cropping dst matrix :math: `(*, 8, 3)`.

    Returns:
        torch.Tensor: The cropping transformation matrix :math: `(*, 4, 4)`.
    """
    transform: torch.Tensor = get_perspective_transform3d(params['src'].to(input.dtype), params['dst'].to(input.dtype))
    transform = transform.expand(input.shape[0], -1, -1).to(input)
    return transform


@_deprecation_wrapper
@_validate_input3d
def apply_perspective3d(
    input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]
) -> torch.Tensor:
    r"""Perform perspective transform of the given torch.Tensor or batch of tensors.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['start_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the original image with shape Bx8x3.
            - params['end_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the transformed image with shape Bx8x3.
        flags (Dict[str, torch.Tensor]):
            - params['interpolation']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: Perspectively transformed tensor with shape :math:`(*, C, D, H, W)`.
    """

    _, _, depth, height, width = input.shape

    # compute the homography between the input points
    transform: torch.Tensor = compute_perspective_transformation3d(input, params)

    out_data: torch.Tensor = input.clone()

    # apply the computed transform
    depth, height, width = input.shape[-3:]
    resample_name: str = Resample(flags['interpolation'].item()).name.lower()
    align_corners: bool = cast(bool, flags['align_corners'].item())

    out_data = warp_perspective3d(
        input, transform, (depth, height, width),
        flags=resample_name, align_corners=align_corners)

    return out_data.view_as(input)


@_deprecation_wrapper
@_validate_input3d
def compute_perspective_transformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the perspective transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['start_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the orignal image with shape Bx8x3.
            - params['end_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the transformed image with shape Bx8x3.

    Returns:
        torch.Tensor: The perspective transformation matrix :math: `(*, 4, 4)`
    """
    perspective_transform: torch.Tensor = get_perspective_transform3d(
        params['start_points'], params['end_points']).to(input)

    transform: torch.Tensor = K.eye_like(4, input)

    transform = perspective_transform

    return transform


@_deprecation_wrapper
@_validate_input3d
def apply_equalize3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Equalize a tensor volume or a batch of tensors volumes with given random parameters.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]): shall be empty.

    Returns:
        torch.Tensor: Equalized input with shape :math:`(*, C, D, H, W)`.
    """

    return equalize3d(input)
