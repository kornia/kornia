from typing import Tuple, List, Union, Dict, cast, Optional

import kornia as K
import torch

from kornia.constants import Resample, BorderType, SamplePadding, pi
from kornia.geometry import (
    get_perspective_transform,
    get_affine_matrix2d,
    warp_perspective,
    rotate,
    crop_by_boxes,
    bbox_generator,
    warp_affine,
    hflip,
    vflip,
    deg2rad,
    bbox_to_mask,
    infer_box_shape
)
from kornia.color import rgb_to_grayscale
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    adjust_gamma,
    solarize,
    equalize,
    posterize,
    sharpness
)
from kornia.filters import motion_blur
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_tensor_center

from .. import random_generator as rg
from ..utils import (
    _validate_input_shape,
    _validate_input_dtype,
    _range_bound,
    _shape_validation,
    _validate_input
)

from .__temp__ import __deprecation_warning, _deprecation_wrapper


@_deprecation_wrapper
@_validate_input
def apply_hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Apply Horizontally flip on a tensor image or a batch of tensor images with given random parameters.

    Input should be a tensor of shape (H, W), (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.

    Returns:
        torch.Tensor: The horizontally flipped input with shape :math:`(*, C, H, W)`.
    """

    return hflip(input)


@_deprecation_wrapper
@_validate_input
def compute_hflip_transformation(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the horizontal flip transformation matrix :math: `(*, 3, 3)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.

    Returns:
        torch.Tensor: The horizontal flip transformation matrix :math: `(*, 3, 3)`
    """

    w: int = input.shape[-1]
    flip_mat: torch.Tensor = torch.tensor([[-1, 0, w - 1],
                                           [0, 1, 0],
                                           [0, 0, 1]], device=input.device, dtype=input.dtype)

    return flip_mat.repeat(input.size(0), 1, 1)


@_deprecation_wrapper
@_validate_input
def apply_vflip(input: torch.Tensor) -> torch.Tensor:
    r"""Apply vertically flip on a tensor image or a batch of tensor images with given random parameters.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.

    Returns:
        torch.Tensor: Vertically flipped input with shape :math:`(B, C, H, W)`.
    """

    return vflip(input)


@_deprecation_wrapper
@_validate_input
def compute_vflip_transformation(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the vertical flip transformation matrix :math: `(*, 3, 3)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.

    Returns:
        torch.Tensor: The vertical flip transformation matrix :math: `(*, 3, 3)`
    """

    h: int = input.shape[-2]
    flip_mat: torch.Tensor = torch.tensor([[1, 0, 0],
                                           [0, -1, h - 1],
                                           [0, 0, 1]], device=input.device, dtype=input.dtype)

    return flip_mat.repeat(input.size(0), 1, 1)


@_deprecation_wrapper
@_validate_input
def apply_color_jitter(
    input: torch.Tensor, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    r"""Apply Color Jitter on a tensor image or a batch of tensor images with given random parameters.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['brightness_factor']: The brightness factor.
            - params['contrast_factor']: The contrast factor.
            - params['hue_factor']: The hue factor.
            - params['saturation_factor']: The saturation factor.
            - params['order']: The order of applying color transforms.
              0 is brightness, 1 is contrast, 2 is saturation, 4 is hue.

    Returns:
        torch.Tensor: Color jitterred input with shape :math:`(B, C, H, W)`.
    """

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

    return jittered


@_deprecation_wrapper
@_validate_input
def compute_intensity_transformation(input: torch.Tensor):
    r"""Compute the identity matrix :math: `(*, 3, 3)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.

    Returns:
        torch.Tensor: The applied matrix :math: `(*, 3, 3)`.
    """
    identity: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
    return identity


@_deprecation_wrapper
@_validate_input
def apply_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Apply Gray Scale on a tensor image or a batch of tensor images with given random parameters.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.

    Returns:
        torch.Tensor: The grayscaled input with shape :math:`(B, C, H, W)`.
    """
    if not _validate_input_shape(input, 1, 3):
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    grayscale: torch.Tensor = input.clone()

    # Make sure it returns (*, 3, H, W)
    grayscale[:] = rgb_to_grayscale(input)

    return grayscale


@_deprecation_wrapper
@_validate_input
def apply_perspective(
    input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]
) -> torch.Tensor:
    r"""Perform perspective transform of the given torch.Tensor or batch of tensors.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['start_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the original image with shape Bx4x2.
            - params['end_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the transformed image with shape Bx4x2.
        flags (Dict[str, torch.Tensor]):
            - params['interpolation']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: Perspectively transformed tensor.
    """
    _, _, height, width = input.shape

    # compute the homography between the input points
    transform: torch.Tensor = compute_perspective_transformation(input, params)

    out_data: torch.Tensor = input.clone()

    # apply the computed transform
    height, width = input.shape[-2:]
    resample_name: str = Resample(flags['interpolation'].item()).name.lower()
    align_corners: bool = cast(bool, flags['align_corners'].item())

    out_data = warp_perspective(
        input, transform, (height, width),
        mode=resample_name, align_corners=align_corners)

    return out_data.view_as(input)


@_deprecation_wrapper
@_validate_input
def compute_perspective_transformation(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the perspective transformation matrix :math: `(*, 3, 3)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['start_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the orignal image with shape Bx4x2.
            - params['end_points']: Tensor containing [top-left, top-right, bottom-right,
              bottom-left] of the transformed image with shape Bx4x2.

    Returns:
        torch.Tensor: The perspective transformation matrix :math: `(*, 3, 3)`
    """
    perspective_transform: torch.Tensor = get_perspective_transform(
        params['start_points'], params['end_points']).type_as(input)

    transform: torch.Tensor = K.eye_like(3, input)

    transform = perspective_transform

    return transform


@_deprecation_wrapper
@_validate_input
def apply_affine(input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Random affine transformation of the image keeping center invariant.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['angle']: Degrees of rotation.
            - params['translations']: Horizontal and vertical translations.
            - params['center']: Rotation center.
            - params['scale']: Scaling params.
            - params['sx']: Shear param toward x-axis.
            - params['sy']: Shear param toward y-axis.
        flags (Dict[str, torch.Tensor]):
            - params['resample']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['padding_mode']: Integer tensor, see SamplePadding enum.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: The transfromed input with shape :math:`(B, C, H, W)`.
    """
    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-3:])

    height, width = x_data.shape[-2:]

    # concatenate transforms
    transform: torch.Tensor = compute_affine_transformation(input, params)

    resample_name: str = Resample(flags['resample'].item()).name.lower()
    padding_mode: str = SamplePadding(flags['padding_mode'].item()).name.lower()
    align_corners: bool = cast(bool, flags['align_corners'].item())

    out_data: torch.Tensor = warp_affine(x_data, transform[:, :2, :],
                                         (height, width), resample_name,
                                         align_corners=align_corners,
                                         padding_mode=padding_mode)
    return out_data.view_as(input)


@_deprecation_wrapper
@_validate_input
def compute_affine_transformation(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the affine transformation matrix :math: `(*, 3, 3)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['angle']: Degrees of rotation.
            - params['translations']: Horizontal and vertical translations.
            - params['center']: Rotation center.
            - params['scale']: Scaling params.
            - params['sx']: Shear param toward x-axis.
            - params['sy']: Shear param toward y-axis.

    Returns:
        torch.Tensor: The affine transformation matrix :math: `(*, 3, 3)`.
    """
    transform = get_affine_matrix2d(
        params['translations'], params['center'], params['scale'], params['angle'],
        deg2rad(params['sx']), deg2rad(params['sy'])
    ).type_as(input)
    return transform


@_deprecation_wrapper
@_validate_input
def apply_rotation(
    input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]
) -> torch.Tensor:
    r"""Rotate a tensor image or a batch of tensor images a random amount of degrees.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['degrees']: degree to be applied.
        flags (Dict[str, torch.Tensor]):
            - params['interpolation']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: The cropped input.
    """
    angles: torch.Tensor = params["degrees"].type_as(input)

    resample_mode: str = Resample(flags['interpolation'].item()).name.lower()
    align_corners: bool = cast(bool, flags['align_corners'].item())

    transformed: torch.Tensor = rotate(input, angles, mode=resample_mode, align_corners=align_corners)

    return transformed


@_deprecation_wrapper
@_validate_input
def compute_rotate_tranformation(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Compute the rotation transformation matrix :math: `(*, 3, 3)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['degrees']: degree to be applied.

    Returns:
        torch.Tensor: The rotation transformation matrix :math: `(*, 3, 3)`
    """
    angles: torch.Tensor = params["degrees"].type_as(input)

    # TODO: This part should be inferred from rotate directly
    center: torch.Tensor = _compute_tensor_center(input)
    rotation_mat: torch.Tensor = _compute_rotation_matrix(angles, center.expand(angles.shape[0], -1))

    # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
    trans_mat: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
    trans_mat[:, 0] = rotation_mat[:, 0]
    trans_mat[:, 1] = rotation_mat[:, 1]

    return trans_mat


@_deprecation_wrapper
@_validate_input
def apply_crop(input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Apply cropping by src bounding box and dst bounding box.

    Order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['src']: The applied cropping src matrix :math: `(*, 4, 2)`.
            - params['dst']: The applied cropping dst matrix :math: `(*, 4, 2)`.
        flags (Dict[str, torch.Tensor]):
            - params['interpolation']: Integer tensor. NEAREST = 0, BILINEAR = 1.
            - params['align_corners']: Boolean tensor.

    Returns:
        torch.Tensor: The cropped input.
    """
    resample_mode: str = Resample.get(flags['interpolation'].item()).name.lower()  # type: ignore
    align_corners: bool = cast(bool, flags['align_corners'].item())

    return crop_by_boxes(
        input, params['src'], params['dst'], resample_mode, align_corners=align_corners)


@_deprecation_wrapper
@_validate_input
def compute_crop_transformation(input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, torch.Tensor]):
    r"""Compute the applied transformation matrix :math: `(*, 3, 3)`.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['src']: The applied cropping src matrix :math: `(*, 4, 2)`.
            - params['dst']: The applied cropping dst matrix :math: `(*, 4, 2)`.

    Returns:
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)`
    """
    transform: torch.Tensor = get_perspective_transform(params['src'].to(input.dtype), params['dst'].to(input.dtype))
    transform = transform.expand(input.shape[0], -1, -1).type_as(input)
    return transform


@_deprecation_wrapper
@_validate_input
def apply_erase_rectangles(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Apply rectangle erase by params.

    Generate a {0, 1} mask with drawed rectangle having parameters defined by params and size by input.size().

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['widths']: widths tensor
            - params['heights']: heights tensor
            - params['xs']: x positions tensor
            - params['ys']: y positions tensor
            - params['values']: the value to fill in

    Returns:
        torch.Tensor: Erased image with shape :math:`(B, C, H, W)`.
    """
    if not (params['widths'].size() == params['heights'].size() == params['xs'].size() == params['ys'].size()):
        raise TypeError(
            "rectangle params components must have same shape. "
            f"Got ({params['widths'].size()}, {params['heights'].size()}) "
            f"and ({params['xs'].size()}, {params['ys'].size()})"
        )

    values = params['values'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, *input.shape[1:]).to(input)

    _, c, h, w = input.size()

    bboxes = bbox_generator(params['xs'], params['ys'], params['widths'], params['heights'])
    mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
    mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
    transformed = torch.where(mask == 1., values, input)
    return transformed


@_deprecation_wrapper
@_validate_input
def apply_adjust_brightness(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply brightness adjustment.

    Wrapper for adjust_brightness for Torchvision-like param settings.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['brightness_factor']: Brightness adjust factor per element
              in the batch. 0 gives a black image, 1 does not modify the input image and 2 gives a
              white image, while any other number modify the brightness.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    transformed = adjust_brightness(input, params['brightness_factor'].to(input.dtype) - 1)

    return transformed


@_deprecation_wrapper
@_validate_input
def apply_adjust_contrast(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply contrast adjustment.

    Wrapper for adjust_contrast for Torchvision-like param settings.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['contrast_factor']: Contrast adjust factor per element in the batch.
              0 generates a compleatly black image, 1 does not modify the input image while any other
              non-negative number modify the brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    transformed = adjust_contrast(input, params['contrast_factor'].to(input.dtype))

    return transformed


@_deprecation_wrapper
@_validate_input
def apply_adjust_saturation(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply saturation adjustment.

    Wrapper for adjust_saturation for Torchvision-like param settings.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['saturation_factor']:  How much to adjust the saturation. 0 will give a black
              and white image, 1 will give the original image while 2 will enhance the saturation
              by a factor of 2.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    transformed = adjust_saturation(input, params['saturation_factor'].to(input.dtype))

    return transformed


@_deprecation_wrapper
@_validate_input
def apply_adjust_hue(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply hue adjustment.

    Wrapper for adjust_hue for Torchvision-like param settings.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['hue_factor']: How much to shift the hue channel. Should be in [-0.5, 0.5].
              0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative
              direction respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will give an
              image with complementary colors while 0 gives the original image.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    transformed = adjust_hue(input, params['hue_factor'].to(input.dtype) * 2 * pi)

    return transformed


@_deprecation_wrapper
@_validate_input
def apply_adjust_gamma(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Perform gamma correction on an image.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['gamma_factor']: Non negative real number, same as γ\gammaγ in the equation.
              gamma larger than 1 make the shadows darker, while gamma smaller than 1 make
              dark regions lighter.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    transformed = adjust_gamma(input, params['gamma_factor'].to(input.dtype))

    return transformed


@_deprecation_wrapper
@_validate_input
def apply_motion_blur(input: torch.Tensor, params: Dict[str, torch.Tensor],
                      flags: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Perform motion blur on an image.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['ksize_factor']: motion kernel width and height (odd and positive).
            - params['angle_factor']: angle of the motion blur in degrees (anti-clockwise rotation).
            - params['direction_factor']: forward/backward direction of the motion blur.
              Lower values towards -1.0 will point the motion blur towards the back (with
              angle provided via angle), while higher values towards 1.0 will point the motion
              blur forward. A value of 0.0 leads to a uniformly (but still angled) motion blur.
        flags (Dict[str, torch.Tensor]):
            - flags['border_type']: the padding mode to be applied before convolving.
              CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3. Default: BorderType.CONSTANT.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.

    """
    kernel_size: int = cast(int, params['ksize_factor'].unique().item())
    angle = params['angle_factor']
    direction = params['direction_factor']
    border_type: str = cast(str, BorderType(flags['border_type'].item()).name.lower())
    mode: str = cast(str, Resample(flags['interpolation'].item()).name.lower())

    return motion_blur(input, kernel_size, angle, direction, border_type, mode)


@_deprecation_wrapper
@_validate_input
def apply_solarize(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Solarize an image.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['thresholds_factor']: thresholds ranged from 0 ~ 1.
            - params['additions_factor']: additions to add on before solarizing.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    thresholds = params['thresholds_factor']
    additions: Optional[torch.Tensor]
    if 'additions_factor' in params:
        additions = params['additions_factor']
    else:
        additions = None
    return solarize(input, thresholds, additions)


@_deprecation_wrapper
@_validate_input
def apply_posterize(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Posterize an image.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['bits_factor']: uint8 bits number ranged from 0 to 8.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    bits = params['bits_factor']

    return posterize(input, bits)


@_deprecation_wrapper
@_validate_input
def apply_sharpness(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Sharpen an image.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        params (Dict[str, torch.Tensor]):
            - params['sharpness_factor']: Sharpness strength. Must be above 0.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    factor = params['sharpness_factor']

    return sharpness(input, factor)


@_deprecation_wrapper
@_validate_input
def apply_equalize(input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""Equalize an image.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.

    Returns:
        torch.Tensor: Adjusted image with shape :math:`(B, C, H, W)`.
    """
    return equalize(input)


@_deprecation_wrapper
@_validate_input
def apply_mixup(input: torch.Tensor, labels: torch.Tensor,
                params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply mixup to images in a batch.

    MixUp augmentation strategy: overlap images with different alpha values.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        labels (torch.Tensor): Label tensor with shape :math:`(B,)`.
        params (Dict[str, torch.Tensor]):
            - params['mixup_pairs']: Mixup indexes.
            - params['mixup_lambdas']: Lambda for the mixup strength.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - Adjusted image, shape of :math:`(B, C, H, W)`.
        - Raw labels, corresponding labels and lambdas for each mix, shape of :math:`(B, 3)`.

    Examples:
        >>> input = torch.stack([torch.eye(5).unsqueeze(dim=0), torch.ones(5, 5).unsqueeze(dim=0)])
        >>> labels = torch.tensor([0, 1])
        >>> params = dict(mixup_pairs=torch.tensor([1, 0]), mixup_lambdas=torch.tensor([0.5, 0.9]))
        >>> out_img, out_label = apply_mixup(input, labels, params)
        >>> out_img
        tensor([[[[1.0000, 0.5000, 0.5000, 0.5000, 0.5000],
                  [0.5000, 1.0000, 0.5000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 1.0000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 0.5000, 1.0000, 0.5000],
                  [0.5000, 0.5000, 0.5000, 0.5000, 1.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[1.0000, 0.1000, 0.1000, 0.1000, 0.1000],
                  [0.1000, 1.0000, 0.1000, 0.1000, 0.1000],
                  [0.1000, 0.1000, 1.0000, 0.1000, 0.1000],
                  [0.1000, 0.1000, 0.1000, 1.0000, 0.1000],
                  [0.1000, 0.1000, 0.1000, 0.1000, 1.0000]]]])
        >>> out_label
        tensor([[0.0000, 1.0000, 0.5000],
                [1.0000, 0.0000, 0.9000]])
    """
    input_permute = input.index_select(dim=0, index=params['mixup_pairs'].to(input.device))
    labels_permute = labels.index_select(dim=0, index=params['mixup_pairs'].to(labels.device))

    lam = params['mixup_lambdas'].view(-1, 1, 1, 1).expand_as(input).to(labels.device)
    inputs = input * (1 - lam) + input_permute * lam
    labels = torch.stack([
        labels.to(input.dtype), labels_permute.to(input.dtype), params['mixup_lambdas'].to(labels.device, input.dtype)
    ], dim=-1).to(labels.device)
    return inputs, labels


@_deprecation_wrapper
@_validate_input
def apply_cutmix(input: torch.Tensor, labels: torch.Tensor,
                 params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply cutmix to images in a batch.

    CutMix augmentation strategy: patches are cut and pasted among training images where the ground
    truth labels are also mixed proportionally to the area of the patches.

    Args:
        input (torch.Tensor): Tensor to be transformed with shape :math:`(*, C, H, W)`.
        labels (torch.Tensor): Label tensor with shape :math:`(B,)`.
        params (Dict[str, torch.Tensor]):
            - params['mix_pairs']: Mixup indexes with shape :math:`(num_mixes, B)`.
            - params['crop_src']: Lambda for the mixup strength :math:`(num_mixes, B, 4, 2)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - Adjusted image, shape of :math:`(B, C, H, W)`.
        - Corresponding labels and lambdas for each mix, shape of :math:`(num_mixes, B, 2)`.

    Examples:
        >>> input = torch.stack([torch.zeros(1, 5, 5), torch.ones(1, 5, 5)], dim=0)
        >>> labels = torch.tensor([0, 1])
        >>> params = {'mix_pairs': torch.tensor([[1, 0]]), 'crop_src': torch.tensor([[[
        ...        [1., 1.],
        ...        [2., 1.],
        ...        [2., 2.],
        ...        [1., 2.]],
        ...       [[1., 1.],
        ...        [3., 1.],
        ...        [3., 2.],
        ...        [1., 2.]]]])}
        >>> apply_cutmix(input, labels, params)
        (tensor([[[[0., 0., 0., 0., 0.],
                  [0., 1., 1., 0., 0.],
                  [0., 1., 1., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[1., 1., 1., 1., 1.],
                  [1., 0., 0., 0., 1.],
                  [1., 0., 0., 0., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.]]]]), tensor([[[0.0000, 1.0000, 0.1600],
                 [1.0000, 0.0000, 0.2400]]]))
    """
    height, width = input.size(2), input.size(3)
    num_mixes = params['mix_pairs'].size(0)
    batch_size = params['mix_pairs'].size(1)

    _shape_validation(params['mix_pairs'], [num_mixes, batch_size], 'mix_pairs')
    _shape_validation(params['crop_src'], [num_mixes, batch_size, 4, 2], 'crop_src')

    out_inputs = input.clone()
    out_labels = []
    for pair, crop in zip(params['mix_pairs'], params['crop_src']):
        input_permute = input.index_select(dim=0, index=pair.to(input.device))
        labels_permute = labels.index_select(dim=0, index=pair.to(labels.device))
        w, h = infer_box_shape(crop)
        lam = w.to(input.dtype) * h.to(input.dtype) / (width * height)  # width_beta * height_beta
        # compute mask to match input shape
        mask = bbox_to_mask(crop, width, height).bool().unsqueeze(dim=1).repeat(1, input.size(1), 1, 1)
        out_inputs[mask] = input_permute[mask]
        out_labels.append(torch.stack([
            labels.to(input.dtype), labels_permute.to(input.dtype), lam.to(labels.device)], dim=1))

    return out_inputs, torch.stack(out_labels, dim=0)
