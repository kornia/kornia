from typing import Tuple, List, Union
import torch
import torch.nn as nn

from kornia.geometry.transform.flips import hflip
from kornia.color.adjust import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue

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
                 saturation: FloatUnionType = 0., hue: FloatUnionType = 0.) -> None:
        super(ColorJitter, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self) -> str:
        repr = f"(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})"
        return self.__class__.__name__ + repr

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return color_jitter(input, self.brightness, self.contrast, self.saturation, self.hue)


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
                 saturation: FloatUnionType = 0., hue: FloatUnionType = 0.) -> torch.Tensor:
    r"""Change the brightness, contrast, saturation and  hue randomly given tensor image or a batch of tensor images.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        brightness (float or tuple): Default value is 0
        contrast (float or tuple): Default value is 0
        saturation (float or tuple): Default value is 0
        hue (float or tuple): Default value is 0

    Returns:
        torch.Tensor: The transformed image
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if isinstance(brightness, float):
        if brightness < 0:
            raise ValueError(f"If brightness is a single number number, it must be non negative. Got {brightness}")
        brightness_bound = [0 - brightness, 0 + brightness]
    elif isinstance(brightness, (tuple, list)) and len(brightness) == 2:
        if not brightness[0] <= brightness[1]:
            raise ValueError(f"Brightness[0] should be smaller than brightness[1] got {brightness}")
        brightness_bound = list(brightness)
    else:
        raise TypeError(
            f"The brightness should be a float number or a tuple with length 2 whose valuese move between [-Inf, Inf].")

    if isinstance(contrast, float):
        if contrast < 0:
            raise ValueError(f"If contrast is a single number number, it must be non negative. Got {contrast}")
        contrast_bound = [1 - contrast, 1 + contrast]
        contrast_bound[0] = max(contrast_bound[0], 0.)
    elif isinstance(contrast, (tuple, list)) and len(contrast) == 2:
        if not 0 <= contrast[0] <= contrast[1]:
            raise ValueError("Contrast values should be between [0, Inf]")
        contrast_bound = list(contrast)
    else:
        raise TypeError(f"The contrast should be a float number or a tuple with length 2.")

    if isinstance(saturation, float):
        if saturation < 0:
            raise ValueError(f"If saturation is a single number, it must be non negative. Got {saturation}")
        saturation_bound = [1 - saturation, 1 + saturation]
        saturation_bound[0] = max(saturation_bound[0], 0.)
    elif isinstance(saturation, (tuple, list)) and len(saturation) == 2:
        if not 0 <= saturation[0] <= saturation[1] <= float('inf'):
            raise ValueError("Saturation values should be between [0, Inf]")
        saturation_bound = list(saturation)
    else:
        raise TypeError(f"The saturation should be a float number or a tuple with length 2.")

    if isinstance(hue, float):
        if not 0 <= hue <= 0.5:
            raise ValueError(f"If hue is a single number, it must belong to [0, 0.5]. Got {hue}")
        hue_bound = [0 - hue, 0 + hue]

    elif isinstance(hue, (tuple, list)) and len(hue) == 2:
        if not -0.5 <= hue[0] <= hue[1] <= 0.5:
            raise ValueError("Hue values should be between [-0.5, 0.5]")
        hue_bound = list(hue)
    else:
        raise TypeError(f"The hue should be a float number or a tuple with length 2.")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view((-1, (*input.shape[-3:])))
    
    transforms = []

    brightness_factor: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(brightness_bound[0], brightness_bound[1])
    contrast_factor: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(contrast_bound[0], contrast_bound[1])
    hue_factor: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(hue_bound[0], hue_bound[1])
    saturation_factor: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(saturation_bound[0], saturation_bound[1])

    transforms.append(lambda img: adjust_brightness(img, brightness_factor))
    transforms.append(lambda img: adjust_contrast(img, contrast_factor))
    transforms.append(lambda img: adjust_saturation(img, saturation_factor))
    transforms.append(lambda img: adjust_hue(img, hue_factor))

   # nn.Sequential(  AdjustBrigtness(brightness_factor),
   #                 AdjustContrast(contrast_factor),

    jittered = input

    for idx in torch.randperm(4):
        t = transforms[idx]
        jittered = t(jittered)

    return jittered
