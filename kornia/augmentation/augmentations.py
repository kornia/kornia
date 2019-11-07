from typing import Tuple, List, Union
import torch
import torch.nn as nn

from kornia.geometry.transform.flips import hflip
from kornia.color.adjust import AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue

UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[float, Tuple[float, float], List[float]]


class RandomHorizontalFlip(nn.Module):

    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transformation (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.
    """

    def __init__(self, p: float = 0.5, return_transformation: bool = False) -> None:
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        self.return_transformation = return_transformation

    def __repr__(self) -> str:
        repr = f"(p={self.p})"
        return self.__class__.__name__ + repr

    def forward(self, input: torch.Tensor) -> UnionType:  # type: ignore
        return random_hflip(input, self.p, self.return_transformation)


class ColorJitter(nn.Module):

    r"""Change the brightness, contrast, saturation and  hue randomly given tensor image or a batch of tensor images.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0
    """

    def __init__(self, brightness: FloatUnionType = 0., contrast: FloatUnionType = 0.,
                 saturation: FloatUnionType = 0., hue: FloatUnionType = 0., return_transform: bool = False) -> None:
        super(ColorJitter, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.return_transform = return_transform

    def __repr__(self) -> str:
        repr = f"(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}), return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward(self, input: UnionType) -> UnionType:  # type: ignore

        if isinstance(input, tuple):

            jittered: torch.Tensor = color_jitter(input[0], self.brightness, self.contrast, self.saturation, self.hue) # type: ignore

            return jittered, input[1]

        return color_jitter(input, self.brightness, self.contrast, self.saturation, self.hue, self.return_transform)


def random_hflip(input: torch.Tensor, p: float = 0.5, return_transformation: bool = False) -> UnionType:
    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transformation (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.

    Returns:
        torch.Tensor: The horizontally flipped input
        torch.Tensor: The applied transformation matrix if return_transformation flag is set to ``True``
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transformation, bool):
        raise TypeError(f"The return_transformation flag must be a bool. Got {type(return_transformation)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))
    probs: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(0, 1)

    to_flip: torch.Tensor = probs < p
    flipped: torch.Tensor = input.clone()

    trans_mat: torch.Tensor = torch.eye(3, device=device, dtype=dtype).expand(input.shape[0], -1, -1)

    flipped[to_flip] = hflip(input[to_flip])
    flipped.squeeze_()

    if return_transformation:

        w: int = input.shape[-2]
        flip_mat: torch.Tensor = torch.tensor([[-1, 0, w],
                                               [0, 1, 0],
                                               [0, 0, 1]])

        trans_mat[to_flip] = flip_mat.to(device).to(dtype)

        return flipped, trans_mat

    return flipped


def color_jitter(input: torch.Tensor, brightness: FloatUnionType = 0., contrast: FloatUnionType = 0.,
                 saturation: FloatUnionType = 0., hue: FloatUnionType = 0., return_transform: bool = False) -> UnionType:
    r"""Random color jiter of an image or batch of images.

    See :class:`~kornia.augmentation.ColorJitter` for details.
    """

    def _input_check(factor: FloatUnionType, name: str, center: float = 0.,
                     bounds: Tuple[float, float] = (0, float('inf'))):
        r"""Check inputs and compute the corresponding factor bounds
        """

        if isinstance(factor, float):
            if factor < 0:
                raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor}")
            factor_bound = [center - factor, center + factor]
            factor_bound[0] = max(factor_bound[0], bounds[0])
            factor_bound[1] = min(factor_bound[1], bounds[1])
        elif isinstance(factor, (tuple, list)) and len(factor) == 2:
            if not bounds[0] <= factor[0] <= factor[1] <= bounds[1]:
                raise ValueError(f"{name}[0] should be smaller than {name}[1] got {factor}")
            factor_bound = list(factor)
        else:
            raise TypeError(
                f"The {name} should be a float number or a tuple with length 2 whose values move between [-Inf, Inf].")
        return factor_bound

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    brightness_bound = _input_check(brightness, 'brightness', bounds=(float('-inf'), float('inf')))
    contrast_bound = _input_check(contrast, 'contrast', center=1.)
    saturation_bound = _input_check(saturation, 'saturation', center=1.)
    hue_bound = _input_check(hue, 'hue', bounds=(-.5, .5))

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))

    brightness_factor: torch.Tensor = torch.empty(
        input.shape[0], device=device).uniform_(
        brightness_bound[0], brightness_bound[1])

    contrast_factor: torch.Tensor = torch.empty(
        input.shape[0], device=device).uniform_(
        contrast_bound[0], contrast_bound[1])

    hue_factor: torch.Tensor = torch.empty(
        input.shape[0], device=device).uniform_(
        hue_bound[0], hue_bound[1])

    saturation_factor: torch.Tensor = torch.empty(
        input.shape[0], device=device).uniform_(
        saturation_bound[0], saturation_bound[1])

    transforms = nn.ModuleList([AdjustBrightness(brightness_factor),
                                AdjustContrast(contrast_factor),
                                AdjustSaturation(saturation_factor),
                                AdjustHue(hue_factor)])

    jittered = input

    for idx in torch.randperm(4).tolist():
        t = transforms[idx]
        jittered = t(jittered)

    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=device, dtype=dtype).expand(input.shape[0], -1, -1)

        return jittered, identity

    return jittered
