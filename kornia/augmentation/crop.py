from typing import Union, Tuple, cast, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

TupleInt = Tuple[int, int]
TupleFloat = Tuple[float, float]
UnionFloat = Union[float, TupleFloat]


def _get_affine_params(degrees: TupleFloat, translate: Optional[TupleFloat],
                       scales: Optional[TupleFloat], shears: Optional[TupleFloat],
                       img_size: TupleInt, batch_size: int) -> torch.Tensor:
    r"""Get parameters for affine transformation. The returned matrix is Bx3x3.

    Returns:
        torch.Tensor: params to be passed to the affine transformation.
    """
    angle = torch.empty(batch_size).uniform_(degrees[0], degrees[1])

    # compute tensor ranges
    if scales is not None:
        scale = torch.empty(batch_size).uniform_(scales[0], scales[1])
    else:
        scale = torch.ones(batch_size)

    if shears is not None:
        shear = torch.empty(batch_size).uniform_(shears[0], shears[1])
    else:
        shear = torch.zeros(batch_size)

    height, width = img_size
    if translate is not None:
        max_dx: float = translate[0] * width
        max_dy: float = translate[1] * height
        translations = torch.cat([
            torch.empty(batch_size).uniform_(-max_dx, max_dx),
            torch.empty(batch_size).uniform_(-max_dy, max_dy),
        ], dim=-1)
    else:
        translations = torch.zeros(batch_size, 2)

    center: torch.Tensor = torch.tensor(
        [width, height], dtype=torch.float32).view(1, 2) / 2

    # concatenate transforms
    transform: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
    transform[..., 2] += translations  # tx/ty
    transform[..., 0, 1] += shear
    transform[..., 1, 0] += shear

    # pad transform to get Bx3x3
    transform_h = F.pad(transform, [0, 0, 0, 1], value=0.)
    transform_h[..., -1, -1] += 1.0
    return transform_h


def _get_perspective_params(batch_size: int, width: int, height: int,
                            distortion_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        width (int): width of the image.
        height (int) : height of the image.
        distortion_scale (float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    Returns:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        The points are in -x order.
    """
    start_points: torch.Tensor = torch.tensor([[
        [0., 0],
        [0, width - 1],
        [height - 1, 0],
        [height - 1, width - 1],
    ]]).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx: float = distortion_scale * width / 2
    fy: float = distortion_scale * height / 2

    factor = torch.tensor([fy, fx]).view(-1, 1, 2)
    offset = factor * torch.rand(batch_size, 4, 2) * 2 - 1

    end_points = start_points + offset

    return start_points, end_points


def perspective(input: torch.Tensor,
                start_points: torch.Tensor,
                end_points: torch.Tensor,
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
        torch.Tensor:  Perspectively transformed tensor.
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not torch.is_tensor(start_points):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(start_points)}")

    if not torch.is_tensor(end_points):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(end_points)}")

    # compute the homography between the input points
    transform: torch.Tensor = kornia.get_perspective_transform(start_points, end_points)

    # apply the computed transform
    height, width = input.shape[-2:]
    img_warped: torch.Tensor = kornia.warp_perspective(input, transform, (height, width))

    if return_transform:
        return img_warped, transform

    return img_warped


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
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) > 4:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    # arrange input data
    x_data: torch.Tensor = input.view(-1, *input.shape[-3:])

    batch_size, _, height, width = x_data.shape

    # sample probabilities
    probs: torch.Tensor = torch.rand(batch_size, device=x_data.device)
    mask: torch.Tensor = probs < p

    # compute points
    start_points, end_points = (
        _get_perspective_params(batch_size, width, height, distortion_scale)
    )
    start_points = start_points.to(x_data.device, x_data.dtype)
    end_points = end_points.to(x_data.device, x_data.dtype)

    # compute and apply transform
    out_perspective: UnionType = perspective(
        x_data, start_points, end_points, return_transform
    )

    out_data: torch.Tensor = x_data.clone()

    if return_transform:
        x_warped, transform = out_perspective
        out_data[mask] = x_warped[mask]
        return out_data, transform
    else:
        x_warped = cast(torch.Tensor, out_perspective)

    out_data[mask] = x_warped[mask]

    return out_data


def random_affine(input: torch.Tensor,
                  degrees: UnionFloat,
                  translate: Optional[TupleFloat] = None,
                  scale: Optional[TupleFloat] = None,
                  shear: Optional[UnionFloat] = None,
                  return_transform: bool = False,
                  mode: str = 'bilinear',
                  padding_mode: str = 'zeros') -> UnionType:
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
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    # check angle ranges
    degrees_tmp: TupleFloat
    if isinstance(degrees, float):
        if degrees < 0.:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees_tmp = (-degrees, degrees)
    else:
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
            "degrees should be a list or tuple and it must be of length 2."
        degrees_tmp = degrees

    # check translation range
    if translate is not None:
        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
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

    # check shear range
    shear_tmp: Optional[TupleFloat]
    if shear is not None:
        if isinstance(shear, float):
            if shear < 0:
                raise ValueError("If shear is a single number, it must be positive.")
            shear_tmp = (-shear, shear)
        else:
            assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                "shear should be a list or tuple and it must be of length 2."
            shear_tmp = shear
    else:
        shear_tmp = shear

    # compute affine parameters and apply transform
    batch_size, _, height, width = input.shape

    transform: torch.Tensor = _get_affine_params(
        degrees_tmp, translate, scale, shear_tmp, (height, width), batch_size)
    transform = transform.to(input.device, input.dtype)

    out_affine: torch.Tensor = kornia.warp_affine(
        input, transform[:, :2], (height, width), mode, padding_mode)

    if return_transform:
        return out_affine, transform

    return out_affine


class RandomPerspective(nn.Module):
    r"""Performs Perspective transformation of the given torch.Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation
        applied to each. Default: False.
        input tensor.
    """

    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomPerspective, self).__init__()
        self.p: float = p
        self.distortion_scale: float = distortion_scale
        self.return_transform: bool = return_transform

    def __repr__(self) -> str:
        repr = f"(distortion_scale={self.distortion_scale}, p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward(self, input: UnionType) -> UnionType:  # type: ignore

        if isinstance(input, tuple):
            raise NotImplementedError("wait for AugmentationBase class")

        return random_perspective(input, self.distortion_scale, self.p, self.return_transform)


class CenterCrop(nn.Module):
    r"""Crops the given torch.Tensor at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size: Union[int, Tuple[int, int]], return_transform: bool = False) -> None:
        super(CenterCrop, self).__init__()
        self.size = size
        self.return_transform = return_transform

    def forward(self, input: UnionType) -> UnionType:  # type: ignore

        if isinstance(input, tuple):
            raise NotImplementedError("wait for AugmentationBase class")

        return kornia.center_crop(input, self.size, self.return_transform)
