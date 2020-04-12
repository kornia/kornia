from typing import Tuple, List, Union, Dict, cast, Optional

import torch
import torch.nn as nn

from kornia.constants import Resample
from kornia.geometry.transform.flips import hflip, vflip
from kornia.geometry.transform import (
    get_perspective_transform,
    get_rotation_matrix2d,
    warp_perspective,
    center_crop,
    rotate,
    crop_by_boxes,
    warp_affine
)
from kornia.geometry.conversions import convert_affinematrix_to_homography, deg2rad
from kornia.color.adjust import (
    adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, adjust_gamma)
from kornia.color.adjust import AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue
from kornia.color.gray import rgb_to_grayscale
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_tensor_center
from kornia.geometry import pi

from . import random as pg
from .utils import _transform_input, _validate_input_shape
from .types import (
    TupleFloat,
    UnionFloat,
    UnionType,
    FloatUnionType
)


def random_hflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random.random_prob_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_hflip` for details.
    """

    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg.random_prob_gen(batch_size, p=p)
    return apply_hflip(input, params, return_transform)


def random_vflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random.random_prob_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_vflip` for details.
    """

    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg.random_prob_gen(batch_size, p=p)
    return apply_vflip(input, params, return_transform)


def color_jitter(input: torch.Tensor, brightness: FloatUnionType = 0.,
                 contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                 hue: FloatUnionType = 0., return_transform: bool = False) -> UnionType:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random.random_color_jitter_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_color_jitter` for details.
    """

    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg.random_color_jitter_gen(batch_size, brightness, contrast, saturation, hue)
    return apply_color_jitter(input, params, return_transform)


def random_grayscale(input: torch.Tensor, p: float = 0.5, return_transform: bool = False):
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random.random_prob_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_grayscale` for details.
    """

    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = pg.random_prob_gen(batch_size, p=p)
    return apply_grayscale(input, params, return_transform)


def random_perspective(input: torch.Tensor,
                       distortion_scale: float = 0.5,
                       p: float = 0.5,
                       return_transform: bool = False) -> UnionType:
    r"""Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    See :func:`~kornia.augmentation.random.random_perspective_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_perspective` for details.
    """

    batch_size, _, height, width = input.shape
    params: Dict[str, torch.Tensor] = pg.random_perspective_gen(
        batch_size, height, width, p, distortion_scale)
    return apply_perspective(input, params, return_transform)


def random_affine(input: torch.Tensor,
                  degrees: UnionFloat,
                  translate: Optional[TupleFloat] = None,
                  scale: Optional[TupleFloat] = None,
                  shear: Optional[UnionFloat] = None,
                  resample: Resample = Resample.BILINEAR,
                  return_transform: bool = False) -> UnionType:
    r"""Random affine transformation of the image keeping center invariant

    See :func:`~kornia.augmentation.random.random_affine_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_affine` for details.
    """

    batch_size, _, height, width = input.shape
    params: Dict[str, torch.Tensor] = pg.random_affine_gen(
        batch_size, height, width, degrees, translate, scale, shear, resample)
    return apply_affine(input, params, return_transform)


def random_rectangle_erase(
        input: torch.Tensor,
        erase_scale_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float],
        return_transform: bool = False
) -> UnionType:
    r"""
    Function that erases a random selected rectangle for each image in the batch, putting
    the value to zero.
    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [erase_scale_range[0], erase_scale_range[1]) and an aspect ratio sampled
    between [aspect_ratio_range[0], aspect_ratio_range[1])

    Args:
        input (torch.Tensor): input images.
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

    images_size = input.size()
    b, _, h, w = images_size
    rect_params = pg.random_rectangles_params_gen(
        b, h, w, erase_scale_range, aspect_ratio_range
    )
    return apply_erase_rectangles(input, rect_params, return_transform=return_transform)


def random_rotation(input: torch.Tensor, degrees: FloatUnionType, return_transform: bool = False) -> UnionType:
    r"""Generate params and apply operation on input tensor.

    See :func:`~kornia.augmentation.random.random_rotation_gen` for details.
    See :func:`~kornia.augmentation.functional.apply_rotation` for details.
    """
    input_tmp: torch.Tensor = input.unsqueeze(0)
    input_tmp = input_tmp.view(-1, *input_tmp.shape[-3:])
    batch_size = input_tmp.shape[0]

    params = pg.random_rotation_gen(batch_size, degrees=degrees)

    return apply_rotation(input, params, return_transform)


def apply_hflip(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    r"""Apply Horizontally flip on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (H, W), (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        params (dict): A dict that must have {'batch_prob': torch.Tensor}. Can be generated from
        kornia.augmentation.random.random_prob_gen.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The horizontally flipped input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """

    input = _transform_input(input)

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    flipped: torch.Tensor = input.clone()

    to_flip = params['batch_prob'].to(input.device)
    flipped[to_flip] = hflip(input[to_flip])

    if return_transform:

        trans_mat: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)

        w: int = input.shape[-1]
        flip_mat: torch.Tensor = torch.tensor([[-1, 0, w],
                                               [0, 1, 0],
                                               [0, 0, 1]])

        trans_mat[to_flip] = flip_mat.type_as(input)

        return flipped, trans_mat

    return flipped


def apply_vflip(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    r"""Apply vertically flip on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (H, W), (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        params (dict): A dict that must have {'batch_prob': torch.Tensor}. Can be generated from
        kornia.augmentation.random.random_prob_gen.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The vertically flipped input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """
    # TODO: params validation

    input = _transform_input(input)

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    flipped: torch.Tensor = input.clone()
    to_flip = params['batch_prob'].to(input.device)
    flipped[to_flip] = vflip(input[to_flip])

    if return_transform:

        trans_mat: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)

        h: int = input.shape[-2]
        flip_mat: torch.Tensor = torch.tensor([[1, 0, 0],
                                               [0, -1, h],
                                               [0, 0, 1]])

        trans_mat[to_flip] = flip_mat.type_as(input)

        return flipped, trans_mat

    return flipped


def apply_color_jitter(input: torch.Tensor,
                       params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    r"""Apply Color Jitter on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (H, W), (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        params (dict): A dict that must have {
            'brightness_factor': torch.Tensor,
            'contrast_factor': torch.Tensor,
            'hue_factor': torch.Tensor,
            'saturation_factor': torch.Tensor,
            'order': torch.Tensor (can be generated by torch.perm(4) by default)
            }. Can be generated from kornia.augmentation.random.random_color_jitter_gen
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The color jitterred input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """
    # TODO: params validation

    input = _transform_input(input)

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    transforms = [
        lambda img: apply_adjust_brightness(img, params),
        lambda img: apply_adjust_contrast(img, params),
        lambda img: apply_adjust_saturation(img, params),
        lambda img: apply_adjust_hue(img, params)
    ]

    jittered = input
    for idx in params['order'].tolist():
        t = transforms[idx]
        jittered = t(jittered)

    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)

        return jittered, identity

    return jittered


def apply_grayscale(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    r"""Apply Gray Scale on a tensor image or a batch of tensor images with given random parameters.
    Input should be a tensor of shape (3, H, W) or a batch of tensors :math:`(*, 3, H, W)`.

    Args:
        params (dict): A dict that must have {'batch_prob': torch.Tensor}. Can be generated from
        kornia.augmentation.random.random_prob_gen
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The grayscaled input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
        is set to ``True``
    """
    # TODO: params validation

    input = _transform_input(input)

    if _validate_input_shape(input, 2, 3):
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    grayscale: torch.Tensor = input.clone()

    to_gray = params['batch_prob'].to(input.device)

    grayscale[to_gray] = rgb_to_grayscale(input[to_gray])
    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)

        return grayscale, identity

    return grayscale


def apply_perspective(input: torch.Tensor, params: Dict[str, torch.Tensor],
                      return_transform: bool = False) -> UnionType:
    r"""Perform perspective transform of the given torch.Tensor or batch of tensors.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape (H, W), (C, H, W), (*, C, H, W).
        start_points (torch.Tensor): Tensor containing [top-left, top-right, bottom-right,
        bottom-left] of the orignal image with shape Bx4x2.
        end_points (torch.Tensor): Tensor containing [top-left, top-right, bottom-right,
        bottom-left] of the transformed image with shape Bx4x2.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.

    Returns:
        torch.Tensor: Perspectively transformed tensor.
    """

    input = _transform_input(input)

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-3:])

    _, _, height, width = x_data.shape

    # compute the homography between the input points
    transform: torch.Tensor = get_perspective_transform(
        params['start_points'], params['end_points']).type_as(input)

    out_data: torch.Tensor = x_data.clone()

    # process valid samples
    mask = params['batch_prob'].to(input.device)

    # TODO: look for a workaround for this hack. In CUDA it fails when no elements found.

    if bool(mask.sum() > 0):
        # apply the computed transform
        height, width = x_data.shape[-2:]
        out_data[mask] = warp_perspective(x_data[mask], transform[mask], (height, width))

    if return_transform:
        return out_data.view_as(input), transform

    return out_data.view_as(input)


def apply_affine(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    r"""Random affine transformation of the image keeping center invariant
        Args:
            input (torch.Tensor): Tensor to be transformed with shape (H, W), (C, H, W), (*, C, H, W).
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
                will be applied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
                range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
                a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
                Will not apply shear by default
            resample (int): Can be retrieved from Resample. 0 is NEAREST, 1 is BILINEAR.
            return_transform (bool): if ``True`` return the matrix describing the transformation
                applied to each. Default: False.
            mode (str): interpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'.
            padding_mode (str): padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'.
    """

    input = _transform_input(input)

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-3:])

    height, width = x_data.shape[-2:]

    # concatenate transforms
    transform = _compose_affine_matrix_3x3(
        params['translations'], params['center'], params['scale'], params['angle'],
        deg2rad(params['sx']), deg2rad(params['sy'])
    ).type_as(input)

    resample_name = Resample(params['resample'].item()).name.lower()

    out_data: torch.Tensor = warp_affine(x_data, transform[:, :2, :], (height, width), resample_name)

    if return_transform:
        return out_data.view_as(input), transform

    return out_data.view_as(input)


def apply_center_crop(input: torch.Tensor, params: Dict[str, torch.Tensor],
                      return_transform: bool = False) -> UnionType:
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    size1: int = int(params['size'][0].item())
    size2: int = int(params['size'][1].item())
    return center_crop(input, (size1, size2), return_transform)


def apply_rotation(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False):
    r"""Rotate a tensor image or a batch of tensor images a random amount of degrees.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        params (dict): A dict that must have {'degrees': torch.Tensor}. Can be generated from
                       kornia.augmentation.random.random_rotation_gen
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    input = _transform_input(input)
    angles: torch.Tensor = params["degrees"].type_as(input)

    transformed: torch.Tensor = rotate(
        input, angles, mode=Resample(params['interpolation'].item()).name.lower())

    if return_transform:

        center: torch.Tensor = _compute_tensor_center(input)
        rotation_mat: torch.Tensor = _compute_rotation_matrix(angles, center.expand(angles.shape[0], -1))

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]

        return transformed, trans_mat

    return transformed


def apply_crop(input: torch.Tensor, params: Dict[str, torch.Tensor], return_transform: bool = False) -> UnionType:
    """
    Args:
        params (dict): A dict that must have {'src': torch.Tensor, 'dst': torch.Tensor}. Can be generated from
        kornia.augmentation.random.random_crop_gen
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.
    Returns:
        torch.Tensor: The grayscaled input
        torch.Tensor: The applied cropping matrix :math: `(*, 4, 2)` if return_transform flag
        is set to ``True``
    """
    input = _transform_input(input)

    return crop_by_boxes(
        input,
        params['src'].type_as(input),
        params['dst'].type_as(input),
        return_transform=return_transform
    )


def apply_erase_rectangles(input: torch.Tensor, params: Dict[str, torch.Tensor],
                           return_transform: bool = False) -> UnionType:
    r"""
    Generate a {0, 1} mask with drawed rectangle having parameters defined by params
    and size by input.size()

    Args:
        input (torch.Tensor): input image.
        params Dict[str, torch.Tensor]:
            params['widths'] must be widths tensor
            params['heights'] must be heights tensor
            params['xs'] must be x positions tensor
            params['ys'] must be y positions tensor
            params['values'] is the value to fill in
    """
    if not (params['widths'].size() == params['heights'].size() == params['xs'].size() == params['ys'].size()):
        raise TypeError(
            f"''rectangle params components must have same shape"
        )

    mask = torch.zeros(input.size()).type_as(input)
    values = torch.zeros(input.size()).type_as(input)

    widths = params['widths']
    heights = params['heights']
    xs = params['xs']
    ys = params['ys']
    vs = params['values']
    for i_elem in range(input.size()[0]):
        h = widths[i_elem].item()
        w = heights[i_elem].item()
        y = ys[i_elem].item()
        x = xs[i_elem].item()
        v = vs[i_elem].item()
        mask[i_elem, :, int(y):int(y + w), int(x):int(x + h)] = 1.
        values[i_elem, :, int(y):int(y + w), int(x):int(x + h)] = v
    transformed = torch.where(mask == 1., values, input)

    if return_transform:
        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        return transformed, identity

    return transformed


def apply_adjust_brightness(input: torch.Tensor, params: Dict[str, torch.Tensor],
                            return_transform: bool = False) -> UnionType:
    """ Wrapper for adjust_brightness for Torchvision-like param settings.

    Args:
        input (torch.Tensor): Image/Input to be adjusted in the shape of (*, N).
        brightness_factor (Union[float, torch.Tensor]): Brightness adjust factor per element
          in the batch. 0 gives a black image, 1 does not modify the input image and 2 gives a
          white image, while any other number modify the brightness.

    Returns:
        torch.Tensor: Adjusted image.
    """
    input = _transform_input(input)
    transformed = adjust_brightness(input, params['brightness_factor'].type_as(input) - 1)

    if return_transform:
        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        return transformed, identity

    return transformed


def apply_adjust_contrast(input: torch.Tensor, params: Dict[str, torch.Tensor],
                          return_transform: bool = False):
    """Wrapper for adjust_contrast for Torchvision-like param settings.
    Args:
        input (torch.Tensor): Image to be adjusted in the shape of (*, N).
        params['contrast_factor'] (Union[float, torch.Tensor]):
          Contrast adjust factor per element in the batch.
          0 generates a compleatly black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image.
    """
    input = _transform_input(input)
    transformed = adjust_contrast(input, params['contrast_factor'].type_as(input))

    if return_transform:
        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        return transformed, identity

    return transformed


def apply_adjust_saturation(input: torch.Tensor, params: Dict[str, torch.Tensor],
                            return_transform: bool = False):
    """Wrapper for adjust_saturation for Torchvision-like param settings.

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (*, N).
        saturation_factor (float):  How much to adjust the saturation. 0 will give a black
        and white image, 1 will give the original image while 2 will enhance the saturation
        by a factor of 2.

    Returns:
        torch.Tensor: Adjusted image.
    """
    input = _transform_input(input)
    transformed = adjust_saturation(input, params['saturation_factor'].type_as(input))

    if return_transform:
        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        return transformed, identity

    return transformed


def apply_adjust_hue(input: torch.Tensor, params: Dict[str, torch.Tensor],
                     return_transform: bool = False):
    """Wrapper for adjust_hue for Torchvision-like param settings.

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (*, N).
        hue_factor (float): How much to shift the hue channel. Should be in [-0.5, 0.5]. 0.5
          and -0.5 give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Returns:
        torch.Tensor: Adjusted image.
    """
    input = _transform_input(input)
    transformed = adjust_hue(input, params['hue_factor'].type_as(input) * 2 * pi)

    if return_transform:
        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        return transformed, identity

    return transformed


def apply_adjust_gamma(input: torch.Tensor, params: Dict[str, torch.Tensor],
                       return_transform: bool = False):
    r"""Perform gamma correction on an image.

    The input image is expected to be in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (\*, N).
        gamma (float): Non negative real number, same as γ\gammaγ in the equation.
          gamma larger than 1 make the shadows darker, while gamma smaller than 1 make
          dark regions lighter.
        gain (float, optional): The constant multiplier. Default 1.

    Returns:
        torch.Tensor: Adjusted image.
    """
    input = _transform_input(input)
    transformed = adjust_gamma(input, params['gamma_factor'].type_as(input))

    if return_transform:
        identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        return transformed, identity

    return transformed


def _compose_affine_matrix_3x3(translations: torch.Tensor,
                               center: torch.Tensor,
                               scale: torch.Tensor,
                               angle: torch.Tensor,
                               sx: Optional[torch.Tensor] = None,
                               sy: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Composes affine matrix Bx3x3 from the components
    Returns:
        torch.Tensor: params to be passed to the affine transformation.
    """
    transform: torch.Tensor = get_rotation_matrix2d(center, -angle, scale)
    transform[..., 2] += translations  # tx/ty
    # pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography(transform)

    if sx is not None:
        x, y = torch.split(center, 1, dim=-1)
        x = x.view(-1)
        y = y.view(-1)
        sx_tan = torch.tan(sx)  # type: ignore
        sy_tan = torch.tan(sy)  # type: ignore
        zeros = torch.zeros_like(sx)  # type: ignore
        ones = torch.ones_like(sx)  # type: ignore
        shear_mat = torch.stack([ones,   -sx_tan,                 sx_tan * x,  # type: ignore   # noqa: E241
                                 -sy_tan, ones + sx_tan * sy_tan, sy_tan * (-sx_tan * x + y)],  # noqa: E241
                                dim=-1).view(-1, 2, 3)
        shear_mat = convert_affinematrix_to_homography(shear_mat)
        transform_h = transform_h @ shear_mat
    return transform_h
