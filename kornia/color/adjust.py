from typing import Union

import torch
import torch.nn as nn

from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb
from kornia.geometry import pi


def adjust_saturation(input: torch.Tensor, saturation_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust color saturation of an image.

    See :class:`~kornia.color.AdjustSaturation` for details.
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(saturation_factor, (float, torch.Tensor,)):
        raise TypeError(f"The saturation_factor should be a float number or torch.Tensor."
                        f"Got {type(saturation_factor)}")

    if isinstance(saturation_factor, float):
        saturation_factor = torch.tensor([saturation_factor])

    saturation_factor = saturation_factor.to(input.device).to(input.dtype)

    if (saturation_factor < 0).any():
        raise ValueError(f"Saturation factor must be non-negative. Got {saturation_factor}")

    for _ in input.shape[1:]:
        saturation_factor = torch.unsqueeze(saturation_factor, dim=-1)

    # convert the rgb image to hsv
    x_hsv: torch.Tensor = rgb_to_hsv(input)

    # unpack the hsv values
    h, s, v = torch.chunk(x_hsv, chunks=3, dim=-3)

    # transform the hue value and appl module
    s_out: torch.Tensor = torch.clamp(s * saturation_factor, min=0, max=1)

    # pack back back the corrected hue
    x_adjusted: torch.Tensor = torch.cat([h, s_out, v], dim=-3)

    # convert back to rgb
    out: torch.Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_hue(input: torch.Tensor, hue_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust hue of an image.

    See :class:`~kornia.color.AdjustHue` for details.
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(hue_factor, (float, torch.Tensor)):
        raise TypeError(f"The hue_factor should be a float number or torch.Tensor in the range between"
                        f" [-PI, PI]. Got {type(hue_factor)}")

    if isinstance(hue_factor, float):
        hue_factor = torch.tensor([hue_factor])

    hue_factor = hue_factor.to(input.device).to(input.dtype)

    if ((hue_factor < -pi) | (hue_factor > pi)).any():
        raise ValueError(f"Hue-factor must be in the range [-PI, PI]. Got {hue_factor}")

    for _ in input.shape[1:]:
        hue_factor = torch.unsqueeze(hue_factor, dim=-1)

    # convert the rgb image to hsv
    x_hsv: torch.Tensor = rgb_to_hsv(input)

    # unpack the hsv values
    h, s, v = torch.chunk(x_hsv, chunks=3, dim=-3)

    # transform the hue value and appl module
    h_out: torch.Tensor = torch.fmod(h + hue_factor, 2*pi)

    # pack back back the corrected hue
    x_adjusted: torch.Tensor = torch.cat([h_out, s, v], dim=-3)

    # convert back to rgb
    out: torch.Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_gamma(input: torch.Tensor, gamma: Union[float, torch.Tensor],
                 gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
    r"""Perform gamma correction on an image.

    See :class:`~kornia.color.AdjustGamma` for details.
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(gamma, (float, torch.Tensor)):
        raise TypeError(f"The gamma should be a positive float or torch.Tensor. Got {type(gamma)}")

    if not isinstance(gain, (float, torch.Tensor)):
        raise TypeError(f"The gain should be a positive float or torch.Tensor. Got {type(gain)}")

    if isinstance(gamma, float):
        gamma = torch.tensor([gamma])

    if isinstance(gain, float):
        gain = torch.tensor([gain])

    gamma = gamma.to(input.device).to(input.dtype)
    gain = gain.to(input.device).to(input.dtype)

    if (gamma < 0.0).any():
        raise ValueError(f"Gamma must be non-negative. Got {gamma}")

    if (gain < 0.0).any():
        raise ValueError(f"Gain must be non-negative. Got {gain}")

    for _ in input.shape[1:]:
        gamma = torch.unsqueeze(gamma, dim=-1)
        gain = torch.unsqueeze(gain, dim=-1)

    # Apply the gamma correction
    x_adjust: torch.Tensor = gain * torch.pow(input, gamma)

    # Truncate between pixel values
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_contrast(input: torch.Tensor,
                    contrast_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust Contrast of an image.

    See :class:`~kornia.color.AdjustContrast` for details.
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(contrast_factor, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(contrast_factor)}")

    if isinstance(contrast_factor, float):
        contrast_factor = torch.tensor([contrast_factor])

    contrast_factor = contrast_factor.to(input.device).to(input.dtype)

    if (contrast_factor < 0).any():
        raise ValueError(f"Contrast factor must be non-negative. Got {contrast_factor}")

    for _ in input.shape[1:]:
        contrast_factor = torch.unsqueeze(contrast_factor, dim=-1)

    # Apply contrast factor to each channel
    x_adjust: torch.Tensor = input * contrast_factor

    # Truncate between pixel values
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_brightness(input: torch.Tensor,
                      brightness_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust Brightness of an image.

    See :class:`~kornia.color.AdjustBrightness` for details.
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(brightness_factor, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(brightness_factor)}")

    if isinstance(brightness_factor, float):
        brightness_factor = torch.tensor([brightness_factor])

    brightness_factor = brightness_factor.to(input.device).to(input.dtype)

    for _ in input.shape[1:]:
        brightness_factor = torch.unsqueeze(brightness_factor, dim=-1)

    # Apply brightness factor to each channel
    x_adjust: torch.Tensor = input + brightness_factor

    # Truncate between pixel values
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)

    return out


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
        return adjust_saturation(input, self.saturation_factor)


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
        return adjust_hue(input, self.hue_factor)


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
        return adjust_gamma(input, self.gamma, self.gain)


class AdjustContrast(nn.Module):
    r"""Adjust Contrast of an image.

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
        return adjust_contrast(input, self.contrast_factor)


class AdjustBrightness(nn.Module):
    r"""Adjust Brightness of an image.

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
        return adjust_brightness(input, self.brightness_factor)
