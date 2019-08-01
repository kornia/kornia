import torch
import torch.nn as nn


class AdjustBrightness(nn.Module):
    r"""Adjust Brightness of an Image

    See :class:`~kornia.color.AdjustBrightness` for details.

    Args:
        image (torch.Tensor): Image to be adjusted.
        brightness_factor (torch.Tensor): Brightness adjust factor per element
          in the batch. 0 generates a compleatly black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image.
    """

    def __init__(self) -> None:
        super(AdjustBrightness, self).__init__()

    def forward(self,  # type: ignore
                image: torch.Tensor,  # type: ignore
                brightness_factor: torch.Tensor  # type: ignore
                ) -> torch.Tensor:  # type: ignore
        return adjust_brightness(image, brightness_factor)


def adjust_brightness(image: torch.Tensor,
                      brightness_factor: torch.Tensor) -> torch.Tensor:
    r"""Adjust Brightness of an Image

    See :class:`~kornia.color.AdjustBrightness` for details.

    Args:
        image (torch.Tensor): Image to be adjusted.
        brightness_factor (torch.Tensor): Brightness adjust factor per element
          in the batch. 0 generates a compleatly black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Returns:
        torch.Tensor: Adjusted image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3:
        raise ValueError("Input size must have a shape of (*, H, W). Got {}"
                         .format(image.shape))

    if (brightness_factor < torch.zeros(1)).any():
        raise ValueError("Brightness factor must be non-negative. Got {}"
                         .format(brightness_factor))

    if torch.is_tensor(brightness_factor):
        for _ in image.shape[1:]:
            brightness_factor = brightness_factor.unsqueeze(-1)

    # Apply brightness factor to each channel
    adjust_image: torch.Tensor = image * brightness_factor

    # Truncate between pixel values
    out: torch.Tensor = torch.clamp(adjust_image, 0.0, 1.0)

    return out
