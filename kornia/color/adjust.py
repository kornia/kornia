from typing import Union

import torch
import torch.nn as nn


def adjust_gamma(input: torch.Tensor, gamma: float, gain: float = 1.) -> torch.Tensor:
    r"""Perform gamma correction on an image.

    See :class:`~kornia.color.AdjustGamma` for details.
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(gamma, float) and gamma > 0.:
        raise TypeError(f"The gamma should be a positive a float. Got {type(gamma)}")

    if not isinstance(gain, float) and gain >= 1.:
        raise TypeError(f"The gamma should be a positive a float. Got {type(gain)}")

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

    # Apply brightness factor to each channel
    x_adjust: torch.Tensor = input + contrast_factor

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

    if (brightness_factor < 0).any():
        raise ValueError(f"Brightness factor must be non-negative. Got {brightness_factor}")

    for _ in input.shape[1:]:
        brightness_factor = torch.unsqueeze(brightness_factor, dim=-1)

    # Apply brightness factor to each channel
    x_adjust: torch.Tensor = input * brightness_factor

    # Truncate between pixel values
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)

    return out


class AdjustGamma(nn.Module):
    r"""Perform gamma correction on an image.

    The input image is expected to be in the range of [0, 1].

    Args:
        image (torch.Tensor): Image to be adjusted in the shape of (*, N).
        gamma (float): Non negative real number, same as γ\gammaγ in the equation.
          gamma larger than 1 make the shadows darker, while gamma smaller than 1 make
          dark regions lighter.
        gain (float optional): The constant multiplier. Default 1.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, gamma: float, gain: float = 1.) -> None:
        super(AdjustGamma, self).__init__()
        self.gamma: float = gamma
        self.gain: float = gain

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return adjust_gamma(input, self.gamma, self.gain)


class AdjustContrast(nn.Module):
    r"""Adjust Contrast of an image.

    The input image is expected to be in the range of [0, 1].

    Args:
        image (torch.Tensor): Image to be adjusted in the shape of (*, N).
        contrast_factor (Union[float, torch.Tensor]): Contrast adjust factor per element
          in the batch. 0 generates a compleatly black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, contrast_factor: Union[float, torch.Tensor]) -> None:
        super(AdjustContrast, self).__init__()
        self.contrast_factor = contrast_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return adjust_contrast(input, self.contrast_factor)


class AdjustBrightness(nn.Module):
    r"""Adjust Brightness of an image.

    The input image is expected to be in the range of [0, 1].

    Args:
        image (torch.Tensor): Image to be adjusted in the shape of (*, N).
        brightness_factor (Union[float, torch.Tensor]): Brightness adjust factor per element
          in the batch. 0 generates a compleatly black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self, brightness_factor: Union[float, torch.Tensor]) -> None:
        super(AdjustBrightness, self).__init__()
        self.brightness_factor = brightness_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return adjust_brightness(input, self.brightness_factor)
