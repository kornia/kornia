from typing import Tuple, List, Union, Dict, cast, Optional

import torch
import torch.nn as nn

from kornia.geometry.transform.flips import hflip, vflip
from kornia.geometry.transform import get_perspective_transform, warp_perspective
from kornia.color.adjust import AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue
from kornia.color.gray import rgb_to_grayscale

from . import param_gen as pg
from .erasing import erase_rectangles, get_random_rectangles_params


TupleFloat = Tuple[float, float]
UnionFloat = Union[float, TupleFloat]
UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


def random_hflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.param_gen._random_prob_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_hflip` for details.
    """
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg._random_prob_gen(batch_size, p=p)
    return apply_hflip(input, params, return_transform)


def random_vflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.param_gen._random_prob_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_vflip` for details.
    """
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg._random_prob_gen(batch_size, p=p)
    return apply_vflip(input, params, return_transform)


def color_jitter(input: torch.Tensor, brightness: FloatUnionType = 0.,
                 contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                 hue: FloatUnionType = 0., return_transform: bool = False) -> UnionType:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.param_gen._random_color_jitter_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_color_jitter` for details.
    """
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg._random_color_jitter_gen(batch_size, brightness, contrast, saturation, hue)
    return apply_color_jitter(input, params, return_transform)


def random_grayscale(input: torch.Tensor, p: float = 0.5, return_transform: bool = False):
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.param_gen._random_prob_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_grayscale` for details.
    """
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg._random_prob_gen(batch_size, p=p)
    return apply_grayscale(input, params, return_transform)


def random_perspective(input: torch.Tensor,
                       distortion_scale: float = 0.5,
                       p: float = 0.5,
                       return_transform: bool = False) -> UnionType:
    r"""Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape (*, C, H, W).
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.
        input tensor.
    """
    batch_size, _, height, width = input.shape
    params: Dict[str, torch.Tensor] = pg._random_perspective_gen(
        batch_size, height, width, p, distortion_scale)
    return apply_perspective(input, params, return_transform)


def random_affine(input: torch.Tensor,
                  degrees: UnionFloat,
                  translate: Optional[TupleFloat] = None,
                  scale: Optional[TupleFloat] = None,
                  shear: Optional[UnionFloat] = None,
                  return_transform: bool = False) -> UnionType:
    r"""Random affine transformation of the image keeping center invariant
        Args:
            input (torch.Tensor): Tensor to be transformed with shape (*, C, H, W).
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
                If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
                will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
                range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
                a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
                Will not apply shear by default
            return_transform (bool): if ``True`` return the matrix describing the transformation
                applied to each. Default: False.
            mode (str): interpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'.
            padding_mode (str): padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'.
    """
    batch_size, _, height, width = input.shape
    params: Dict[str, torch.Tensor] = pg._random_affine_gen(
        batch_size, height, width, degrees, translate, scale, shear)
    return apply_affine(input, params, return_transform)


def apply_hflip(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    r"""Apply Horizontally flip on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (H, W), (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        params (dict): A dict that must have {'batch_prob': torch.Tensor}. Can be generated from
        kornia.augmentation.param_gen._random_prob_gen.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The horizontally flipped input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) == 2:
        input = input.unsqueeze(0)

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) != 4:
        raise ValueError(f"Input size must have a shape of (*, C, H, W). Got {input.shape}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    flipped: torch.Tensor = input.clone()

    to_flip = params['batch_prob'].to(device)
    flipped[to_flip] = hflip(input[to_flip])
    flipped.squeeze_()

    if return_transform:

        trans_mat: torch.Tensor = torch.eye(3, device=device, dtype=dtype).repeat(input.shape[0], 1, 1)

        w: int = input.shape[-1]
        flip_mat: torch.Tensor = torch.tensor([[-1, 0, w],
                                               [0, 1, 0],
                                               [0, 0, 1]])

        trans_mat[to_flip] = flip_mat.to(device).to(dtype)

        return flipped, trans_mat

    return flipped


def apply_vflip(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    r"""Apply vertically flip on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (H, W), (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        params (dict): A dict that must have {'batch_prob': torch.Tensor}. Can be generated from
        kornia.augmentation.param_gen._random_prob_gen.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The vertically flipped input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """
    # TODO: params validation

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) == 2:
        input = input.unsqueeze(0)

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) != 4:
        raise ValueError(f"Input size must have a shape of (*, C, H, W). Got {input.shape}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    flipped: torch.Tensor = input.clone()
    to_flip = params['batch_prob'].to(device)
    flipped[to_flip] = vflip(input[to_flip])

    if return_transform:

        trans_mat: torch.Tensor = torch.eye(3, device=device, dtype=dtype).repeat(input.shape[0], 1, 1)

        h: int = input.shape[-2]
        flip_mat: torch.Tensor = torch.tensor([[1, 0, 0],
                                               [0, -1, h],
                                               [0, 0, 1]])

        trans_mat[to_flip] = flip_mat.to(device).to(dtype)

        return flipped, trans_mat

    return flipped


def apply_color_jitter(input: torch.Tensor, params: Dict[str, torch.Tensor],
                       return_transform: bool = False) -> UnionType:
    r"""Apply Color Jitter on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (H, W), (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        params (dict): A dict that must have {
            'brightness_factor': torch.Tensor,
            'contrast_factor': torch.Tensor,
            'hue_factor': torch.Tensor,
            'saturation_factor': torch.Tensor,
            }. Can be generated from kornia.augmentation.param_gen._random_color_jitter_gen
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The color jitterred input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """
    # TODO: params validation

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) == 2:
        input = input.unsqueeze(0)

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) != 4:
        raise ValueError(f"Input size must have a shape of (*, C, H, W). Got {input.shape}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    transforms = nn.ModuleList([AdjustBrightness(params['brightness_factor'].to(device)),
                                AdjustContrast(params['contrast_factor'].to(device)),
                                AdjustSaturation(params['saturation_factor'].to(device)),
                                AdjustHue(params['hue_factor'].to(device))])

    jittered = input

    for idx in torch.randperm(4).tolist():
        t = transforms[idx]
        jittered = t(jittered)

    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=device, dtype=dtype).repeat(input.shape[0], 1, 1)

        return jittered, identity

    return jittered


def apply_grayscale(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    r"""Apply Gray Scale on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (3, H, W) or a batch of tensors :math:`(*, 3, H, W)`.

    Args:
        params (dict): A dict that must have {'batch_prob': torch.Tensor}. Can be generated from
        kornia.augmentation.param_gen._random_prob_gen
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The grayscaled input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """
    # TODO: params validation

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) == 3 and input.shape[-3] == 3:
        input = input.unsqueeze(0)

    if len(input.shape) != 4 or input.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    grayscale: torch.Tensor = input.clone()

    to_gray = params['batch_prob'].to(device)

    grayscale[to_gray] = rgb_to_grayscale(input[to_gray])
    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=device, dtype=dtype).repeat(input.shape[0], 1, 1)

        return grayscale, identity

    return grayscale


def random_rectangle_erase(
        images: torch.Tensor,
        erase_scale_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float]
) -> torch.Tensor:
    r"""
    Function that erases a random selected rectangle for each image in the batch, putting
    the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [erase_scale_range[0], erase_scale_range[1]) and an aspect ratio sampled
    between [aspect_ratio_range[0], aspect_ratio_range[1])

    Args:
        images (torch.Tensor): input images.
        erase_scale_range (Tuple[float, float]): range of proportion of erased area against input image.
        aspect_ratio_range (Tuple[float, float]): range of aspect ratio of erased area.
    """
    if not (isinstance(erase_scale_range[0], float) and
            isinstance(erase_scale_range[1], float) and
            erase_scale_range[0] > 0. and erase_scale_range[1] > 0.):
        raise TypeError(
            f"'erase_scale_range' must be a Tuple[float, float] with positive values"
        )
    if not (isinstance(aspect_ratio_range[0], float) and
            isinstance(aspect_ratio_range[1], float) and
            aspect_ratio_range[0] > 0. and aspect_ratio_range[1] > 0.):
        raise TypeError(
            f"'aspect_ratio_range' must be a Tuple[float, float] with positive values"
        )

    images_size = images.size()
    b, _, h, w = images_size
    rect_params = get_random_rectangles_params(
        (b, ), h, w, erase_scale_range, aspect_ratio_range
    )
    images = erase_rectangles(images, rect_params)
    return images


def apply_perspective(input: torch.Tensor,
                      params: Dict[str, torch.Tensor],
                      return_transform: bool = False) -> UnionType:
    r"""Perform perspective transform of the given torch.Tensor or batch of tensors.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape BxCxHxW.
        start_points (torch.Tensor): Tensor containing [top-left, top-right, bottom-right,
        bottom-left] of the orignal image with shape Bx4x2.
        end_points (torch.Tensor): Tensor containing [top-left, top-right, bottom-right,
        bottom-left] of the transformed image with shape Bx4x2.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.

    Returns:
        torch.Tensor: Perspectively transformed tensor.
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-3:])

    batch_size, _, height, width = x_data.shape

    # compute the homography between the input points
    transform: torch.Tensor = get_perspective_transform(
        params['start_points'], params['end_points']).to(device, dtype)

    out_data: torch.Tensor = x_data.clone()

    # process valid samples
    mask = params['batch_prob'].to(device)

    # apply the computed transform
    height, width = x_data.shape[-2:]
    out_data[mask] = warp_perspective(x_data[mask], transform[mask], (height, width))

    if return_transform:
        return out_data.view_as(input), transform

    return out_data.view_as(input)


def apply_affine(input: torch.Tensor,
                 params: Dict[str, torch.Tensor],
                 return_transform: bool = False) -> UnionType:
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-3:])

    height, width = x_data.shape[-2:]
    transform: torch.Tensor = params['transform'].to(device, dtype)

    out_data: torch.Tensor = warp_perspective(x_data, transform, (height, width))

    if return_transform:
        return out_data.view_as(input), transform

    return out_data.view_as(input)
