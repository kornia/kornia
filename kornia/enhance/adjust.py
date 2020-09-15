from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.augmentation.utils import _transform_input3d
from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb
from kornia.utils.image import _to_bchw
from kornia.constants import pi


__all__ = [
    "adjust_brightness",
    "adjust_contrast",
    "adjust_gamma",
    "adjust_hue",
    "adjust_saturation",
    "adjust_hue_raw",
    "adjust_saturation_raw",
    "solarize",
    "equalize",
    "equalize3d",
    "posterize",
    "sharpness",
    "AdjustBrightness",
    "AdjustContrast",
    "AdjustGamma",
    "AdjustHue",
    "AdjustSaturation",
]


def adjust_saturation_raw(input: torch.Tensor, saturation_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust color saturation of an image. Expecting input to be in hsv format already.

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

    # unpack the hsv values
    h, s, v = torch.chunk(input, chunks=3, dim=-3)

    # transform the hue value and appl module
    s_out: torch.Tensor = torch.clamp(s * saturation_factor, min=0, max=1)

    # pack back back the corrected hue
    out: torch.Tensor = torch.cat([h, s_out, v], dim=-3)

    return out


def adjust_saturation(input: torch.Tensor, saturation_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust color saturation of an image.

    See :class:`~kornia.color.AdjustSaturation` for details.
    """

    # convert the rgb image to hsv
    x_hsv: torch.Tensor = rgb_to_hsv(input)

    # perform the conversion
    x_adjusted: torch.Tensor = adjust_saturation_raw(x_hsv, saturation_factor)

    # convert back to rgb
    out: torch.Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_hue_raw(input: torch.Tensor, hue_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust hue of an image. Expecting input to be in hsv format already.

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

    # unpack the hsv values
    h, s, v = torch.chunk(input, chunks=3, dim=-3)

    # transform the hue value and appl module
    divisor: float = 2 * pi.item()
    h_out: torch.Tensor = torch.fmod(h + hue_factor, divisor)

    # pack back back the corrected hue
    out: torch.Tensor = torch.cat([h_out, s, v], dim=-3)

    return out


def adjust_hue(input: torch.Tensor, hue_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust hue of an image.

    See :class:`~kornia.color.AdjustHue` for details.
    """

    # convert the rgb image to hsv
    x_hsv: torch.Tensor = rgb_to_hsv(input)

    # perform the conversion
    x_adjusted: torch.Tensor = adjust_hue_raw(x_hsv, hue_factor)

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


def _solarize(input: torch.Tensor, thresholds: Union[float, torch.Tensor] = 0.5) -> torch.Tensor:
    r""" For each pixel in the image, select the pixel if the value is less than the threshold.
    Otherwise, subtract 1.0 from the pixel.

    Args:
        input (torch.Tensor): image or batched images to solarize.
        thresholds (float or torch.Tensor): solarize thresholds.
            If int or one element tensor, input will be solarized across the whole batch.
            If 1-d tensor, input will be solarized element-wise, len(thresholds) == len(input).

    Returns:
        torch.Tensor: Solarized images.
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(thresholds)}")

    if isinstance(thresholds, torch.Tensor) and len(thresholds.shape) != 0:
        assert input.size(0) == len(thresholds) and len(thresholds.shape) == 1, \
            f"threshholds must be a 1-d vector of shape ({input.size(0)},). Got {thresholds}"
        # TODO: I am not happy about this line, but no easy to do batch-wise operation
        thresholds = thresholds.to(input.device).to(input.dtype)
        thresholds = torch.stack([x.expand(*input.shape[1:]) for x in thresholds])

    return torch.where(input < thresholds, input, 1.0 - input)


def solarize(input: torch.Tensor, thresholds: Union[float, torch.Tensor] = 0.5,
             additions: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
    r"""For each pixel in the image less than threshold, we add 'addition' amount to it and then clip the
    pixel value to be between 0 and 1.0. The value of 'addition' is between -0.5 and 0.5.

    Args:
        input (torch.Tensor): image tensor with shapes like (C, H, W) or (B, C, H, W) to solarize.
        thresholds (float or torch.Tensor): solarize thresholds.
            If int or one element tensor, input will be solarized across the whole batch.
            If 1-d tensor, input will be solarized element-wise, len(thresholds) == len(input).
        additions (optional, float or torch.Tensor): between -0.5 and 0.5. Default None.
            If None, no addition will be performed.
            If int or one element tensor, same addition will be added across the whole batch.
            If 1-d tensor, additions will be added element-wisely, len(additions) == len(input).

    Returns:
        torch.Tensor: Solarized images.
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(thresholds)}")

    if isinstance(thresholds, float):
        thresholds = torch.tensor(thresholds)

    if additions is not None:
        if not isinstance(additions, (float, torch.Tensor,)):
            raise TypeError(f"The factor should be either a float or torch.Tensor. "
                            f"Got {type(additions)}")

        if isinstance(additions, float):
            additions = torch.tensor(additions)

        assert torch.all((additions < 0.5) * (additions > -0.5)), \
            f"The value of 'addition' is between -0.5 and 0.5. Got {additions}."

        if isinstance(additions, torch.Tensor) and len(additions.shape) != 0:
            assert input.size(0) == len(additions) and len(additions.shape) == 1, \
                f"additions must be a 1-d vector of shape ({input.size(0)},). Got {additions}"
            # TODO: I am not happy about this line, but no easy to do batch-wise operation
            additions = additions.to(input.device).to(input.dtype)
            additions = torch.stack([x.expand(*input.shape[1:]) for x in additions])
        input = input + additions
        input = input.clamp(0., 1.)

    return _solarize(input, thresholds)


def posterize(input: torch.Tensor, bits: Union[int, torch.Tensor]) -> torch.Tensor:
    r"""Reduce the number of bits for each color channel. Non-differentiable function, uint8 involved.

    Args:
        input (torch.Tensor): image tensor with shapes like (C, H, W) or (B, C, H, W) to posterize.
        bits (int or torch.Tensor): number of high bits. Must be in range [0, 8].
            If int or one element tensor, input will be posterized by this bits.
            If 1-d tensor, input will be posterized element-wisely, len(bits) == input.shape[1].
            If n-d tensor, input will be posterized element-channel-wisely, bits.shape == input.shape[:len(bits.shape)]

    Returns:
        torch.Tensor: Image with reduced color channels.
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if isinstance(bits, int):
        bits = torch.tensor(bits)

    if not torch.all((bits >= 0) * (bits <= 8)) and bits.dtype == torch.int:
        raise ValueError(f"bits must be integers within range [0, 8]. Got {bits}.")

    # TODO: Make a differentiable version
    # Current version:
    # Ref: https://github.com/open-mmlab/mmcv/pull/132/files#diff-309c9320c7f71bedffe89a70ccff7f3bR19
    # Ref: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L222
    # Potential approach: implementing kornia.LUT with floating points
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py#L472
    def _left_shift(input: torch.Tensor, shift: torch.Tensor):
        return ((input * 255).to(torch.uint8) * (2 ** shift)).to(input.dtype) / 255.

    def _right_shift(input: torch.Tensor, shift: torch.Tensor):
        return (input * 255).to(torch.uint8) / (2 ** shift).to(input.dtype) / 255.

    def _posterize_one(input: torch.Tensor, bits: torch.Tensor):
        # Single bits value condition
        if bits == 0:
            return torch.zeros_like(input)
        if bits == 8:
            return input.clone()
        bits = 8 - bits
        return _left_shift(_right_shift(input, bits), bits)

    if len(bits.shape) == 0 or (len(bits.shape) == 1 and len(bits) == 1):
        return _posterize_one(input, bits)

    res = []
    if len(bits.shape) == 1:
        input = _to_bchw(input)

        assert bits.shape[0] == input.shape[0], \
            f"Batch size must be equal between bits and input. Got {bits.shape[0]}, {input.shape[0]}."

        for i in range(input.shape[0]):
            res.append(_posterize_one(input[i], bits[i]))
        return torch.stack(res, dim=0)

    assert bits.shape == input.shape[:len(bits.shape)], \
        f"Batch and channel must be equal between bits and input. Got {bits.shape}, {input.shape[:len(bits.shape)]}."
    _input = input.view(-1, *input.shape[len(bits.shape):])
    _bits = bits.flatten()
    for i in range(input.shape[0]):
        res.append(_posterize_one(_input[i], _bits[i]))
    return torch.stack(res, dim=0).reshape(*input.shape)


def sharpness(input: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Implements Sharpness function from PIL using torch ops.

    Args:
        input (torch.Tensor): image tensor with shapes like (C, H, W) or (B, C, H, W) to sharpen.
        factor (float or torch.Tensor): factor of sharpness strength. Must be above 0.
            If float or one element tensor, input will be sharpened by the same factor across the whole batch.
            If 1-d tensor, input will be sharpened element-wisely, len(factor) == len(input).

    Returns:
        torch.Tensor: Sharpened image or images.
    """
    input = _to_bchw(input)
    if isinstance(factor, torch.Tensor):
        factor = factor.squeeze()
        if len(factor.size()) != 0:
            assert input.size(0) == factor.size(0), \
                f"Input batch size shall match with factor size if 1d array. Got {input.size(0)} and {factor.size(0)}"
    else:
        factor = float(factor)
    kernel = torch.tensor([
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ], dtype=input.dtype).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # This shall be equivalent to depthwise conv2d:
    # Ref: https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2
    degenerate = torch.nn.functional.conv2d(input, kernel, bias=None, stride=1, groups=input.size(1))
    degenerate = torch.clamp(degenerate, 0., 1.)

    mask = torch.ones_like(degenerate)
    padded_mask = torch.nn.functional.pad(mask, [1, 1, 1, 1])
    padded_degenerate = torch.nn.functional.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, input)

    def _blend_one(input1: torch.Tensor, input2: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(factor, torch.Tensor):
            factor = factor.squeeze()
            assert len(factor.size()) == 0, f"Factor shall be a float or single element tensor. Got {factor}"
        if factor == 0.:
            return input1
        if factor == 1.:
            return input2
        diff = (input2 - input1) * factor
        res = input1 + diff
        if factor > 0. and factor < 1.:
            return res
        return torch.clamp(res, 0, 1)
    if isinstance(factor, (float)) or len(factor.size()) == 0:
        return _blend_one(input, result, factor)
    return torch.stack([_blend_one(input[i], result[i], factor[i]) for i in range(len(factor))])


# Code taken from: https://github.com/pytorch/vision/pull/796
def _scale_channel(im):
    """Scale the data in the channel to implement equalize."""
    im = im * 255

    # Compute the histogram of the image channel.
    histo = torch.histc(im, bins=256, min=0, max=255)
    # For the purposes of computing the step, filter out the nonzeros.
    nonzero_histo = torch.reshape(histo[histo != 0], [-1])
    step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
        # Compute the cumulative sum, shifting by step // 2
        # and then normalization by step.
        lut = (torch.cumsum(histo, 0) + (step // 2)) // step
        # Shift lut, prepending with 0.
        lut = torch.cat([torch.zeros(1, device=lut.device), lut[:-1]])
        # Clip the counts to be in range.  This is done
        # in the C code for image.point.
        return torch.clamp(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    if step == 0:
        result = im
    else:
        # can't index using 2d index. Have to flatten and then reshape
        result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
        result = result.reshape_as(im)

    return result / 255.


def equalize(input: torch.Tensor) -> torch.Tensor:
    r"""Apply equalize on the input tensor.
    Implements Equalize function from PIL using PyTorch ops based on uint8 format:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352

    Args:
        input (torch.Tensor): image tensor with shapes like :math:(C, H, W) or :math:(B, C, H, W) to equalize.

    Returns:
        torch.Tensor: Sharpened image or images.
    """
    input = _to_bchw(input)

    res = []
    for image in input:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_image = torch.stack([_scale_channel(image[i, :, :]) for i in range(len(image))])
        res.append(scaled_image)
    return torch.stack(res)


def equalize3d(input: torch.Tensor) -> torch.Tensor:
    r"""Equalizes the values for a 3D volumetric tensor.

    Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352

    Args:
    input (torch.Tensor): image tensor with shapes like :math:(C, D, H, W) or :math:(B, C, D, H, W) to equalize.

    Returns:
    torch.Tensor: Sharpened image or images with same shape as the input.
    """
    input = _transform_input3d(input)

    res = []
    for volume in input:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_input = torch.stack([_scale_channel(volume[i, :, :, :]) for i in range(len(volume))])
        res.append(scaled_input)

    return torch.stack(res)


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
        return adjust_contrast(input, self.contrast_factor)


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
        return adjust_brightness(input, self.brightness_factor)
