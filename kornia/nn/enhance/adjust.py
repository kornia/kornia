from typing import Union, Optional

import torch
import torch.nn as nn

import kornia


__all__ = [
    "AdjustBrightness",
    "AdjustContrast",
    "AdjustGamma",
    "AdjustHue",
    "AdjustSaturation",
    "Solarize",
    "Equalize",
    "Equalize3D",
    "Posterize",
    "Sharpeness",
]


class Solarize(nn.Module):
    r"""
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(Solarize, self).__init__()
        raise NotImplementedError()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return image


class Equalize(nn.Module):
    r"""
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(Equalize, self).__init__()
        raise NotImplementedError()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return image


class Equalize3D(nn.Module):
    r"""
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(Equalize3D, self).__init__()
        raise NotImplementedError()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return image


class Posterize(nn.Module):
    r"""
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(Posterize, self).__init__()
        raise NotImplementedError()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return image


class Sharpeness(nn.Module):
    r"""
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(Sharpeness, self).__init__()
        raise NotImplementedError()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return image


class AdjustSaturation(nn.Module):
    r"""Adjust color saturation of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (\*, N).
        saturation_factor (float):  How much to adjust the saturation. 0 will give a black
        and white image, 1 will give the original image while 2 will enhance the saturation
        by a factor of 2.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, saturation_factor: Union[float, torch.Tensor]) -> None:
        super(AdjustSaturation, self).__init__()
        self.saturation_factor: Union[float, torch.Tensor] = saturation_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.enhance.adjust_saturation(input, self.saturation_factor)


class AdjustHue(nn.Module):
    r"""Adjust hue of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of (\*, N).
        hue_factor (float): How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, hue_factor: Union[float, torch.Tensor]) -> None:
        super(AdjustHue, self).__init__()
        self.hue_factor: Union[float, torch.Tensor] = hue_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.enhance.adjust_hue(input, self.hue_factor)


class AdjustGamma(nn.Module):
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

    def __init__(self, gamma: Union[float, torch.Tensor], gain: Union[float, torch.Tensor] = 1.) -> None:
        super(AdjustGamma, self).__init__()
        self.gamma: Union[float, torch.Tensor] = gamma
        self.gain: Union[float, torch.Tensor] = gain

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.enhance.adjust_gamma(input, self.gamma, self.gain)


class AdjustContrast(nn.Module):
    r"""Adjust Contrast of an image. This implementation aligns OpenCV, not PIL. Hence,
    the output differs from TorchVision.

    The input image is expected to be in the range of [0, 1].

    Args:
        input (torch.Tensor): Image to be adjusted in the shape of (\*, N).
        contrast_factor (Union[float, torch.Tensor]): Contrast adjust factor per element
          in the batch. 0 generates a compleatly black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, contrast_factor: Union[float, torch.Tensor]) -> None:
        super(AdjustContrast, self).__init__()
        self.contrast_factor: Union[float, torch.Tensor] = contrast_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.enhance.adjust_contrast(input, self.contrast_factor)


class AdjustBrightness(nn.Module):
    r"""Adjust Brightness of an image. This implementation aligns OpenCV, not PIL. Hence,
    the output differs from TorchVision.

    The input image is expected to be in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Input to be adjusted in the shape of (\*, N).
        brightness_factor (Union[float, torch.Tensor]): Brightness adjust factor per element
          in the batch. 0 does not modify the input image while any other number modify the
          brightness.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, brightness_factor: Union[float, torch.Tensor]) -> None:
        super(AdjustBrightness, self).__init__()
        self.brightness_factor: Union[float, torch.Tensor] = brightness_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return kornia.enhance.adjust_brightness(input, self.brightness_factor)
