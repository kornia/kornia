from typing import Callable, Tuple, List, Union, cast

import torch
import torch.nn as nn
from torch.distributions import Uniform

from kornia.geometry.transform.flips import hflip, vflip
from kornia.color.adjust import AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue
from kornia.color.gray import rgb_to_grayscale
from kornia.geometry import pi

UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
FloatUnionType = Union[torch.Tensor, float, Tuple[float, float], List[float]]


class RandomFlip(nn.Module):
    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomFlip, self).__init__()
        self.p = p
        self.return_transform = return_transform
        self.flip_func = None

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward_flip(self, input: UnionType, flip_func: Callable) -> UnionType:  # type: ignore

        if isinstance(input, tuple):

            inp: torch.Tensor = input[0]
            prev_trans: torch.Tensor = input[1]

            if self.return_transform:

                out = flip_func(inp, p=self.p, return_transform=True)
                img: torch.Tensor = out[0]
                trans_mat: torch.Tensor = out[1]

                return img, prev_trans @ trans_mat

            # https://mypy.readthedocs.io/en/latest/casts.html cast the return type to please mypy gods
            img = cast(torch.Tensor, flip_func(inp, p=self.p, return_transform=False))

            # Transform image but pass along the previous transformation
            return img, prev_trans

        return flip_func(input, p=self.p, return_transform=self.return_transform)

    def forward(self, input: UnionType) -> UnionType:  # type: ignore
        raise NotImplementedError("forward method is not implemented for this class.")


class RandomHorizontalFlip(RandomFlip):

    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> seq = nn.Sequential(kornia.augmentation.RandomHorizontalFlip(p=1.0, return_transform=True),
                                kornia.augmentation.RandomHorizontalFlip(p=1.0, return_transform=True)
                               )
        >>> seq(input)
        (tensor([[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 1., 1.]]),
        tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]))

    """

    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        self.return_transform = return_transform

    def forward(self, input: UnionType) -> UnionType:  # type: ignore
        return self.forward_flip(input, random_hflip)


class RandomVerticalFlip(RandomFlip):

    r"""Vertically flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Vertically flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated

    Examples:
        >>> input = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 1.]]]])
        >>> seq = nn.Sequential(kornia.augmentation.RandomVerticalFlip(p=1.0, return_transform=True))
        >>> seq(input)
        (tensor([[0., 1., 1.],
                 [0., 0., 0.],
                 [0., 0., 0.]]),
        tensor([[[1., 0., 0.],
                 [0., -1., 3.],
                 [0., 0., 1.]]]))

    """

    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomVerticalFlip, self).__init__()
        self.p = p
        self.return_transform = return_transform

    def forward(self, input: UnionType) -> UnionType:  # type: ignore
        return self.forward_flip(input, random_vflip)


class ColorJitter(nn.Module):

    r"""Change the brightness, contrast, saturation and hue randomly given tensor image or a batch of tensor images.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated

    """

    def __init__(self, brightness: FloatUnionType = 0., contrast: FloatUnionType = 0.,
                 saturation: FloatUnionType = 0., hue: FloatUnionType = 0., return_transform: bool = False) -> None:
        super(ColorJitter, self).__init__()
        self.brightness: FloatUnionType = brightness
        self.contrast: FloatUnionType = contrast
        self.saturation: FloatUnionType = saturation
        self.hue: FloatUnionType = hue
        self.return_transform: bool = return_transform

    def __repr__(self) -> str:
        repr = f"(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation},\
            hue={self.hue}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward(self, input: UnionType) -> UnionType:  # type: ignore

        if isinstance(input, tuple):

            jittered: torch.Tensor = cast(
                torch.Tensor,
                color_jitter(
                    input[0],
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                    return_transform=False))

            return jittered, input[1]

        return color_jitter(input, self.brightness, self.contrast, self.saturation, self.hue, self.return_transform)


class RandomGrayscale(nn.Module):
    r"""Random Grayscale transformation according to a probability p value

    Args:
        p (float): probability of the image to be transformed to grayscale. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    def __init__(self, p: float = 0.5, return_transform: bool = False) -> None:
        super(RandomGrayscale, self).__init__()
        self.p = p
        self.return_transform = return_transform

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform})"
        return self.__class__.__name__ + repr

    def forward(self, input: UnionType) -> UnionType:  # type: ignore

        if isinstance(input, tuple):

            output: torch.Tensor = cast(
                torch.Tensor,
                random_grayscale(
                    input[0],
                    p=self.p,
                    return_transform=False))

            return output, input[1]

        return random_grayscale(input, p=self.p, return_transform=self.return_transform)


def random_hflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Horizontally Flip an image or a batch of images

    See :class:`~kornia.augmentation.RandomHorizontalFlip` for details.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))

    probs: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(0, 1)

    to_flip: torch.Tensor = probs < p
    flipped: torch.Tensor = input.clone()

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


def random_vflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Vertically flip an image or a batch of images

    See :class:`~kornia.augmentation.RandomVerticalFlip` for details.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))

    probs: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(0, 1)

    to_flip: torch.Tensor = probs < p
    flipped: torch.Tensor = input.clone()

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


def color_jitter(input: torch.Tensor, brightness: FloatUnionType = 0.,
                 contrast: FloatUnionType = 0., saturation: FloatUnionType = 0.,
                 hue: FloatUnionType = 0., return_transform: bool = False) -> UnionType:
    r"""Random color jitter of an image or batch of images.

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

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    brightness_bound: torch.Tensor = _check_and_bound(
        brightness, 'brightness', bounds=(
            float('-inf'), float('inf')), device=device, dtype=dtype)
    contrast_bound: torch.Tensor = _check_and_bound(contrast, 'contrast', center=1., device=device, dtype=dtype)
    saturation_bound: torch.Tensor = _check_and_bound(saturation, 'saturation', center=1., device=device, dtype=dtype)
    hue_bound: torch.Tensor = _check_and_bound(hue, 'hue', bounds=(-pi.item(), pi.item()), device=device, dtype=dtype)

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))

    brightness_distribution = Uniform(brightness_bound[0], brightness_bound[1])
    brightness_factor = brightness_distribution.rsample([input.shape[0]])

    contrast_distribution = Uniform(contrast_bound[0], contrast_bound[1])
    contrast_factor = contrast_distribution.rsample([input.shape[0]])

    hue_distribution = Uniform(hue_bound[0], hue_bound[1])
    hue_factor = hue_distribution.rsample([input.shape[0]])

    saturation_distribution = Uniform(saturation_bound[0], saturation_bound[1])
    saturation_factor = saturation_distribution.rsample([input.shape[0]])

    transforms = nn.ModuleList([AdjustBrightness(brightness_factor),
                                AdjustContrast(contrast_factor),
                                AdjustSaturation(saturation_factor),
                                AdjustHue(hue_factor)])

    jittered = input

    for idx in torch.randperm(4).tolist():
        t = transforms[idx]
        jittered = t(jittered)

    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=device, dtype=dtype).repeat(input.shape[0], 1, 1)

        return jittered, identity

    return jittered


def random_grayscale(input: torch.Tensor, p: float = .5, return_transform: bool = False) -> UnionType:
    r"""Random grayscale of an image or batch of images.

    See :class:`~kornia.augmentation.RandomGrayscale` for details.
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) > 3 and input.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {input.shape}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))

    probs: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(0, 1)

    to_gray: torch.Tensor = probs < p
    grayscale: torch.Tensor = input.clone()

    grayscale[to_gray] = rgb_to_grayscale(input[to_gray])
    if return_transform:

        identity: torch.Tensor = torch.eye(3, device=device, dtype=dtype).repeat(input.shape[0], 1, 1)

        return grayscale, identity

    return grayscale
