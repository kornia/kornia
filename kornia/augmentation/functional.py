from typing import Tuple, List, Union, Dict
import warnings

import torch
import torch.nn as nn
from torch.distributions import Uniform

from kornia.geometry.transform.flips import hflip, vflip
from kornia.color.adjust import AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue
from kornia.color.gray import rgb_to_grayscale
from kornia.geometry import pi


UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


def random_hflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    warnings.warn("random_hflip is going to be deprecated. Please use kornia.augmentation.RandomHorizontalFlip instead.",
            DeprecationWarning, stacklevel=1)
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = _get_random_p_params(batch_size, p=p)
    return apply_hflip(input, params, return_transform)


def random_vflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    warnings.warn("random_vflip is going to be deprecated. Please use kornia.augmentation.RandomVerticalFlip instead.",
            DeprecationWarning, stacklevel=1)
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = _get_random_p_params(batch_size, p=p)
    return apply_vflip(input, params, return_transform)


def color_jitter(input: torch.Tensor, brightness: FloatUnionType = 0.,
                 contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                 hue: FloatUnionType = 0., return_transform: bool = False) -> UnionType:
    warnings.warn("color_jitter is going to be deprecated. Please use kornia.augmentation.ColorJitter instead.",
            DeprecationWarning, stacklevel=1)
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = _get_color_jitter_params(batch_size, brightness, contrast, saturation, hue)
    return apply_color_jitter(input, params, return_transform)


def random_grayscale(input: torch.Tensor, p: float = 0.5, return_transform: bool = False):
    warnings.warn("random_grayscale is going to be deprecated. Please use kornia.augmentation.RandomGrayScale instead.",
            DeprecationWarning, stacklevel=1)
    if isinstance(input, tuple):
        batch_size = input[0].shape[0] if len(input[0].shape) == 4 else 1
    else:
        batch_size = input.shape[0] if len(input.shape) == 4 else 1
    params = _get_random_p_params(batch_size, p=p)
    return apply_grayscale(input, params, return_transform)


def apply_hflip(input: torch.Tensor, params: dict, return_transform: bool = False) -> UnionType:
    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
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

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))

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


def apply_vflip(input: torch.Tensor, params: dict, return_transform: bool = False) -> UnionType:
    r"""Vertically flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
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

    # input = input.unsqueeze(0)
    # input = input.view((-1, (*input.shape[-3:])))

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


def apply_color_jitter(input: torch.Tensor, params: dict, return_transform: bool = False) -> UnionType:
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


def _get_color_jitter_params(batch_size: int, brightness: FloatUnionType = 0.,
                            contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                            hue: FloatUnionType = 0.) -> Dict[str, torch.Tensor]:
    r"""Random color jiter of an image or batch of images.

    See :class:`~kornia.augmentation.ColorJitter` for details.
    """

    def _check_and_bound(factor: FloatUnionType, name: str, center: float = 0.,
                         bounds: Tuple[float, float] = (0, float('inf')),
                         device: torch.device = torch.device('cpu'),
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
        r"""Check inputs and compute the corresponding factor bounds
        """

        if isinstance(factor, float):

            if factor < 0:
                raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor}")

            factor_bound = torch.tensor([center - factor, center + factor])
            factor_bound = torch.clamp(factor_bound, bounds[0], bounds[1])

        elif (isinstance(factor, torch.Tensor) and factor.dim() == 0):

            if factor < 0:
                raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor}")

            factor_bound = torch.tensor([torch.tensor(center) - factor, torch.tensor(center) + factor])
            factor_bound = torch.clamp(factor_bound, bounds[0], bounds[1])

        elif isinstance(factor, (tuple, list)) and len(factor) == 2:

            if not bounds[0] <= factor[0] <= factor[1] <= bounds[1]:
                raise ValueError(f"{name}[0] should be smaller than {name}[1] got {factor}")

            factor_bound = torch.tensor(factor)

        elif isinstance(factor, torch.Tensor) and factor.shape[0] == 2 and factor.dim() == 1:

            if not bounds[0] <= factor[0] <= factor[1] <= bounds[1]:
                raise ValueError(f"{name}[0] should be smaller than {name}[1] got {factor}")

            factor_bound = factor

        else:

            raise TypeError(
                f"The {name} should be a float number or a tuple with length 2 whose values move between {bounds}.")

        return factor_bound.to(device).to(dtype)

    brightness_bound: torch.Tensor = _check_and_bound(
        brightness, 'brightness', bounds=(
            float('-inf'), float('inf')))
    contrast_bound: torch.Tensor = _check_and_bound(contrast, 'contrast', center=1.)
    saturation_bound: torch.Tensor = _check_and_bound(saturation, 'saturation', center=1.)
    hue_bound: torch.Tensor = _check_and_bound(hue, 'hue', bounds=(-pi.item(), pi.item()))

    brightness_distribution = Uniform(brightness_bound[0], brightness_bound[1])
    brightness_factor = brightness_distribution.rsample([batch_size])

    contrast_distribution = Uniform(contrast_bound[0], contrast_bound[1])
    contrast_factor = contrast_distribution.rsample([batch_size])

    hue_distribution = Uniform(hue_bound[0], hue_bound[1])
    hue_factor = hue_distribution.rsample([batch_size])

    saturation_distribution = Uniform(saturation_bound[0], saturation_bound[1])
    saturation_factor = saturation_distribution.rsample([batch_size])

    return {
        "brightness_factor": brightness_factor,
        "contrast_factor": contrast_factor,
        "hue_factor": hue_factor,
        "saturation_factor": saturation_factor,
    }


def apply_grayscale(input: torch.Tensor, params: dict, return_transform: bool = False) -> UnionType:
    """
    params: {'bool_p': torch.Tensor}
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
    # print(input.detach().shape)
    # input = input.view((-1, (*input.shape[-3:])))
    # print(input.detach().shape)

    grayscale: torch.Tensor = input.clone()

    to_gray = params['batch_prob'].to(device)

    grayscale[to_gray] = rgb_to_grayscale(input[to_gray])
    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=device, dtype=dtype).repeat(input.shape[0], 1, 1)

        return grayscale, identity

    return grayscale


def _get_random_p_params(batch_size: int, p: float = .5) -> Dict[str, torch.Tensor]:
    r"""Random grayscale of an image or batch of images.

    See :class:`~kornia.augmentation.RandomGrayscale` for details.
    """

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    probs: torch.Tensor = torch.empty(batch_size, device='cpu').uniform_(0, 1)

    batch_prob: torch.Tensor = probs < p

    return {'batch_prob': batch_prob}
