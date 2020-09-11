from typing import Tuple, List, Union, Dict, cast, Optional

import torch

from . import random_generator as rg
from .utils import _transform_input3d, _validate_input_dtype
from kornia.constants import Resample, BorderType, pi
from kornia.geometry.transform.affwarp import (
    _compute_rotation_matrix3d, _compute_tensor_center3d
)
from kornia.geometry.transform.projwarp import warp_projective
from kornia.geometry import (
    rotate3d,
    get_affine_matrix3d,
    deg2rad
)


def random_hflip3d(input: torch.Tensor, p: float = 0.5, return_transform: bool = False
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random_generator.random_prob_generator` for details.
    See :func:`~kornia.augmentation.functional.apply_hflip3d` for details.
    """
    input = _transform_input3d(input)
    batch_size, _, d, h, w = input.size()
    params = rg.random_prob_generator(batch_size, p=p)
    output = apply_hflip3d(input, params)
    if return_transform:
        raise NotImplementedError
    return output


def random_vflip3d(input: torch.Tensor, p: float = 0.5, return_transform: bool = False
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random_generator.random_prob_generator` for details.
    See :func:`~kornia.augmentation.functional3d.apply_vflip3d` for details.
    """
    input = _transform_input3d(input)
    batch_size, _, d, h, w = input.size()
    params = rg.random_prob_generator(batch_size, p=p)
    output = apply_vflip3d(input, params)
    if return_transform:
        raise NotImplementedError
    return output


def random_dflip3d(input: torch.Tensor, p: float = 0.5, return_transform: bool = False
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random_generator.random_prob_generator` for details.
    See :func:`~kornia.augmentation.functional3d.apply_dflip3d` for details.
    """
    input = _transform_input3d(input)
    batch_size, _, d, h, w = input.size()
    params = rg.random_prob_generator(batch_size, p=p)
    output = apply_dflip3d(input, params)
    if return_transform:
        raise NotImplementedError
    return output


def apply_hflip3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Apply horizontal flip on a 3D tensor volume or a batch of tensors volumes with given random parameters.
    Input should be a tensor of shape :math:`(D, H, W)`, :math:`(C, D, H, W)` or :math:`(*, C, D, H, W)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(D, H, W)`, :math:`(C, D, H, W)`,
            :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['batch_prob']: A boolean tensor that indicating whether if to transform an image in a batch.
                Example: With input batchsize of 4, only the first two tensors will be transformed if
                batch_prob is [True, True, False, False].

    Returns:
        torch.Tensor: The horizontal flipped input
    """
    # TODO: params validation

    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

    flipped: torch.Tensor = input.clone()

    to_flip = params['batch_prob'].to(input.device)
    flipped[to_flip] = torch.flip(input[to_flip], [-1])

    return flipped


def compute_hflip_transformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the applied transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(D, H, W)`, :math:`(C, D, H, W)`,
            :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['batch_prob']: A boolean tensor that indicating whether if to transform an image in a batch.
                Example: With input batchsize of 4, only the first two tensors will be transformed if
                batch_prob is [True, True, False, False].

    Returns:
        torch.Tensor: The applied transformation matrix :math: `(*, 4, 4)`
    """
    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
    to_flip = params['batch_prob'].to(input.device)
    trans_mat: torch.Tensor = torch.eye(4, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
    w: int = input.shape[-1]
    flip_mat: torch.Tensor = torch.tensor([[-1, 0, 0, w - 1],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
    trans_mat[to_flip] = flip_mat.type_as(input)

    return trans_mat


def apply_vflip3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Apply vertical flip on a 3D tensor volume or a batch of tensors volumes with given random parameters.
    Input should be a tensor of shape :math:`(D, H, W)`, :math:`(C, D, H, W)` or :math:`(*, C, D, H, W)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(D, H, W)`, :math:`(C, D, H, W)`,
            :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['batch_prob']: A boolean tensor that indicating whether if to transform an image in a batch.
                Example: With input batchsize of 4, only the first two tensors will be transformed if
                batch_prob is [True, True, False, False].

    Returns:
        torch.Tensor: The vertical flipped input
    """
    # TODO: params validation

    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

    flipped: torch.Tensor = input.clone()
    to_flip = params['batch_prob'].to(input.device)
    flipped[to_flip] = torch.flip(input[to_flip], [-2])

    return flipped


def compute_vflip_transformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the applied transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(D, H, W)`, :math:`(C, D, H, W)`,
            :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['batch_prob']: A boolean tensor that indicating whether if to transform an image in a batch.
                Example: With input batchsize of 4, only the first two tensors will be transformed if
                batch_prob is [True, True, False, False].

    Returns:
        torch.Tensor: The applied transformation matrix :math: `(*, 4, 4)`
    """
    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
    to_flip = params['batch_prob'].to(input.device)
    trans_mat: torch.Tensor = torch.eye(4, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)

    h: int = input.shape[-2]
    flip_mat: torch.Tensor = torch.tensor([[1, 0, 0, 0],
                                           [0, -1, 0, h - 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

    trans_mat[to_flip] = flip_mat.type_as(input)

    return trans_mat


def apply_dflip3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Apply depthical flip on a 3D tensor volume or a batch of tensors volumes with given random parameters.
    Input should be a tensor of shape :math:`(D, H, W)`, :math:`(C, D, H, W)` or :math:`(*, C, D, H, W)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(D, H, W)`, :math:`(C, D, H, W)`,
            :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['batch_prob']: A boolean tensor that indicating whether if to transform an image in a batch.
                Example: With input batchsize of 4, only the first two tensors will be transformed if
                batch_prob is [True, True, False, False].

    Returns:
        torch.Tensor: The depthical flipped input.
    """
    # TODO: params validation

    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

    flipped: torch.Tensor = input.clone()
    to_flip = params['batch_prob'].to(input.device)
    flipped[to_flip] = torch.flip(input[to_flip], [-3])

    return flipped


def compute_dflip_transformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the applied transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(D, H, W)`, :math:`(C, D, H, W)`,
            :math:`(*, C, D, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['batch_prob']: A boolean tensor that indicating whether if to transform an image in a batch.
                Example: With input batchsize of 4, only the first two tensors will be transformed if
                batch_prob is [True, True, False, False].

    Returns:
        torch.Tensor: The applied transformation matrix :math: `(*, 4, 4)`
    """
    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
    to_flip = params['batch_prob'].to(input.device)
    trans_mat: torch.Tensor = torch.eye(4, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)

    d: int = input.shape[-3]
    flip_mat: torch.Tensor = torch.tensor([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, -1, d - 1],
                                           [0, 0, 0, 1]])

    trans_mat[to_flip] = flip_mat.type_as(input)

    return trans_mat


def apply_affine3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Random affine transformation of the image keeping center invariant.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape (D, H, W), (C, D, H, W), (B, C, D, H, W).
        params (Dict[str, torch.Tensor]):
            - params['angles']: Degrees of rotation with the shape of :math: `(*, 3)` for yaw, pitch, roll.
            - params['translations']: Depthical, Horizontal and vertical translations.
            - params['center']: Rotation center.
            - params['scale']: Isotropic scaling params.
            - params['sxy']: Shear param toward x-y-axis.
            - params['sxz']: Shear param toward x-z-axis.
            - params['syx']: Shear param toward y-x-axis.
            - params['syz']: Shear param toward y-z-axis.
            - params['szx']: Shear param toward z-x-axis.
            - params['szy']: Shear param toward z-y-axis.
            - params['resample']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: The transfromed input
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-4:])

    depth, height, width = x_data.shape[-3:]

    # concatenate transforms
    transform: torch.Tensor = compute_affine_transformation3d(input, params)

    resample_name: str = Resample(params['resample'].item()).name.lower()
    align_corners: bool = cast(bool, params['align_corners'].item())

    out_data: torch.Tensor = warp_projective(x_data, transform[:, :3, :],
                                             (depth, height, width), resample_name,
                                             align_corners=align_corners)
    return out_data.view_as(input)


def compute_affine_transformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the applied transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape (D, H, W), (C, D, H, W), (B, C, D, H, W).
        params (Dict[str, torch.Tensor]):
            - params['angles']: Degrees of rotation with the shape of :math: `(*, 3)` for yaw, pitch, roll.
            - params['translations']: Depthical, Horizontal and vertical translations.
            - params['center']: Rotation center.
            - params['scale']: Isotropic scaling params.
            - params['sxy']: Shear param toward x-y-axis.
            - params['sxz']: Shear param toward x-z-axis.
            - params['syx']: Shear param toward y-x-axis.
            - params['syz']: Shear param toward y-z-axis.
            - params['szx']: Shear param toward z-x-axis.
            - params['szy']: Shear param toward z-y-axis.
            - params['resample']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: The applied transformation matrix :math: `(*, 4, 4)`
    """
    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
    transform = get_affine_matrix3d(
        params['translations'], params['center'], params['scale'], params['angles'],
        deg2rad(params['sxy']), deg2rad(params['sxz']), deg2rad(params['syx']),
        deg2rad(params['syz']), deg2rad(params['szx']), deg2rad(params['szy'])
    ).type_as(input)
    return transform


def apply_rotation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Rotate a tensor image or a batch of tensor images a random amount of degrees.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape (H, W), (C, H, W), (B, C, H, W).
        params (Dict[str, torch.Tensor]):
            - params['degrees']: degree to be applied.

    Returns:
        torch.Tensor: The cropped input
    """
    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
    yaw: torch.Tensor = params["yaw"].type_as(input)
    pitch: torch.Tensor = params["pitch"].type_as(input)
    roll: torch.Tensor = params["roll"].type_as(input)

    resample_mode: str = Resample(params['interpolation'].item()).name.lower()
    align_corners: bool = cast(bool, params['align_corners'].item())

    transformed: torch.Tensor = rotate3d(input, yaw, pitch, roll, mode=resample_mode, align_corners=align_corners)

    return transformed


def compute_rotate_tranformation3d(input: torch.Tensor, params: Dict[str, torch.Tensor]):
    r"""Compute the applied transformation matrix :math: `(*, 4, 4)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape (D, H, W), (C, D, H, W), (B, C, D, H, W).
        params (Dict[str, torch.Tensor]):
            - params['yaw']: degree to be applied.
            - params['pitch']: degree to be applied.
            - params['roll']: degree to be applied.

    Returns:
        torch.Tensor: The applied transformation matrix :math: `(*, 4, 4)`
    """
    input = _transform_input3d(input)
    _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
    yaw: torch.Tensor = params["yaw"].type_as(input)
    pitch: torch.Tensor = params["pitch"].type_as(input)
    roll: torch.Tensor = params["roll"].type_as(input)

    center: torch.Tensor = _compute_tensor_center3d(input)
    rotation_mat: torch.Tensor = _compute_rotation_matrix3d(yaw, pitch, roll, center.expand(yaw.shape[0], -1))

    # rotation_mat is B x 3 x 4 and we need a B x 4 x 4 matrix
    trans_mat: torch.Tensor = torch.eye(4, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
    trans_mat[:, 0] = rotation_mat[:, 0]
    trans_mat[:, 1] = rotation_mat[:, 1]
    trans_mat[:, 2] = rotation_mat[:, 2]

    return trans_mat
