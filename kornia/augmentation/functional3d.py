from typing import Tuple, List, Union, Dict, cast, Optional

import torch
import torch.nn as nn

from kornia.constants import Resample, BorderType, pi
from kornia.geometry import (
    get_perspective_transform,
    get_rotation_matrix2d,
    get_affine_matrix2d,
    warp_perspective,
    rotate,
    crop_by_boxes,
    warp_affine,
    hflip,
    vflip,
    deg2rad
)
from kornia.color import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    adjust_gamma,
    rgb_to_grayscale
)
from kornia.filters import motion_blur
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_tensor_center

from . import random_generator as rg
from .utils import _transform_input3d, _validate_input_dtype
from .types import (
    TupleFloat,
    UnionFloat,
    UnionType,
    FloatUnionType
)


def random_hflip3d(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
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


def random_vflip3d(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
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


def random_dflip3d(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
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
    flipped[to_flip] = hflip(input[to_flip])

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
    flip_mat: torch.Tensor = torch.tensor([[-1, 0, 0, w],
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
    flipped[to_flip] = vflip(input[to_flip])

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
                                           [0, -1, 0, h],
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
                                           [0, 0, -1, d],
                                           [0, 0, 0, 1]])

    trans_mat[to_flip] = flip_mat.type_as(input)

    return trans_mat
