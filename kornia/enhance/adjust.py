from __future__ import annotations

from math import pi
from typing import ClassVar, Optional, Union

import torch

from kornia.color import hsv_to_rgb, rgb_to_grayscale, rgb_to_hsv
from kornia.core import ImageModule as Module
from kornia.core import Parameter, Tensor
from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_COLOR_OR_GRAY,
    KORNIA_CHECK_IS_TENSOR,
)
from kornia.utils.helpers import _torch_histc_cast
from kornia.utils.image import perform_keep_shape_image, perform_keep_shape_video


def adjust_saturation_raw(image: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Adjust color saturation of an image.

    Expecting image to be in hsv format already.
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, Tensor)), "Factor should be float or Tensor.")

    if isinstance(factor, float):
        # TODO: figure out how to create later a tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, Tensor):
        factor = factor.to(image.device, image.dtype)

    # make factor broadcastable
    while len(factor.shape) != len(image.shape):
        factor = factor[..., None]

    # unpack the hsv values
    h, s, v = torch.chunk(image, chunks=3, dim=-3)

    # transform the hue value and appl module
    s_out: Tensor = torch.clamp(s * factor, min=0, max=1)

    # pack back back the corrected hue
    out: Tensor = torch.cat([h, s_out, v], dim=-3)

    return out


def adjust_saturation_with_gray_subtraction(image: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Adjust color saturation of an image by blending the image with its grayscaled version.

    The image is expected to be an RGB image or a gray image in the range of [0, 1].
    If it is an RGB image, returns blending of the image with its grayscaled version.
    If it is a gray image, returns the image.

    .. note::
        this is just a convenience function to have compatibility with Pil

    Args:
        image: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.

    Return:
        Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> adjust_saturation_with_gray_subtraction(x, 2.).shape
        torch.Size([1, 3, 3, 3])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.tensor([1., 2.])
        >>> adjust_saturation_with_gray_subtraction(x, y).shape
        torch.Size([2, 3, 3, 3])
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, Tensor)), "Factor should be float or Tensor.")
    KORNIA_CHECK_IS_COLOR_OR_GRAY(image, "Image should be an RGB or gray image")

    if image.shape[-3] == 1:
        return image

    if isinstance(factor, float):
        # TODO: figure out how to create later a tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, Tensor):
        factor = factor.to(image.device, image.dtype)

    # make factor broadcastable
    while len(factor.shape) != len(image.shape):
        factor = factor[..., None]

    x_other: Tensor = rgb_to_grayscale(image)

    # blend the image with the grayscaled image
    x_adjusted: Tensor = (1 - factor) * x_other + factor * image

    # clamp the output
    out: Tensor = torch.clamp(x_adjusted, 0.0, 1.0)

    return out


def adjust_saturation(image: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Adjust color saturation of an image.

    .. image:: _static/img/adjust_saturation.png

    The image is expected to be an RGB image in the range of [0, 1].

    Args:
        image: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
        saturation_mode: The mode to adjust saturation.

    Return:
        Adjusted image in the shape of :math:`(*, 3, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/image_enhancement.html>`__.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> adjust_saturation(x, 2.).shape
        torch.Size([1, 3, 3, 3])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.tensor([1., 2.])
        >>> adjust_saturation(x, y).shape
        torch.Size([2, 3, 3, 3])
    """
    # convert the rgb image to hsv
    x_hsv: Tensor = rgb_to_hsv(image)

    # perform the conversion
    x_adjusted: Tensor = adjust_saturation_raw(x_hsv, factor)

    # convert back to rgb
    out: Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_hue_raw(image: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Adjust hue of an image.

    Expecting image to be in hsv format already.
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(
        isinstance(factor, (float, Tensor)),
        f"The factor should be a float number or Tensor in the range between [-PI, PI]. Got {type(factor)}",
    )

    if isinstance(factor, float):
        factor = torch.as_tensor(factor)

    factor = factor.to(image.device, image.dtype)

    # make factor broadcastable
    while len(factor.shape) != len(image.shape):
        factor = factor[..., None]

    # unpack the hsv values
    h, s, v = torch.chunk(image, chunks=3, dim=-3)

    # transform the hue value and appl module
    divisor: float = 2 * pi
    h_out: Tensor = torch.fmod(h + factor, divisor)

    # pack back back the corrected hue
    out: Tensor = torch.cat([h_out, s, v], dim=-3)

    return out


def adjust_hue(image: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Adjust hue of an image.

    .. image:: _static/img/adjust_hue.png

    The image is expected to be an RGB image in the range of [0, 1].

    Args:
        image: Image to be adjusted in the shape of :math:`(*, 3, H, W)`.
        factor: How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Return:
        Adjusted image in the shape of :math:`(*, 3, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/image_enhancement.html>`__.

    Example:
        >>> x = torch.ones(1, 3, 2, 2)
        >>> adjust_hue(x, 3.141516).shape
        torch.Size([1, 3, 2, 2])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2) * 3.141516
        >>> adjust_hue(x, y).shape
        torch.Size([2, 3, 3, 3])
    """
    # convert the rgb image to hsv
    x_hsv: Tensor = rgb_to_hsv(image)

    # perform the conversion
    x_adjusted: Tensor = adjust_hue_raw(x_hsv, factor)

    # convert back to rgb
    out: Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_gamma(input: Tensor, gamma: Union[float, Tensor], gain: Union[float, Tensor] = 1.0) -> Tensor:
    r"""Perform gamma correction on an image.

    .. image:: _static/img/adjust_contrast.png

    The input image is expected to be in the range of [0, 1].

    Args:
        input: Image to be adjusted in the shape of :math:`(*, H, W)`.
        gamma: Non negative real number, same as y\gammay in the equation.
            gamma larger than 1 make the shadows darker, while gamma smaller than 1 make
            dark regions lighter.
        gain: The constant multiplier.

    Return:
        Adjusted image in the shape of :math:`(*, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/image_enhancement.html>`__.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_gamma(x, 1.0, 2.0)
        tensor([[[[1., 1.],
                  [1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y1 = torch.ones(2) * 1.0
        >>> y2 = torch.ones(2) * 2.0
        >>> adjust_gamma(x, y1, y2).shape
        torch.Size([2, 5, 3, 3])
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

    if not isinstance(gamma, (float, Tensor)):
        raise TypeError(f"The gamma should be a positive float or Tensor. Got {type(gamma)}")

    if not isinstance(gain, (float, Tensor)):
        raise TypeError(f"The gain should be a positive float or Tensor. Got {type(gain)}")

    if isinstance(gamma, float):
        gamma = Tensor([gamma])

    if isinstance(gain, float):
        gain = Tensor([gain])

    gamma = gamma.to(input.device).to(input.dtype)
    gain = gain.to(input.device).to(input.dtype)

    if (gamma < 0.0).any():
        raise ValueError(f"Gamma must be non-negative. Got {gamma}")

    if (gain < 0.0).any():
        raise ValueError(f"Gain must be non-negative. Got {gain}")

    for _ in range(len(input.shape) - len(gamma.shape)):
        gamma = torch.unsqueeze(gamma, dim=-1)

    for _ in range(len(input.shape) - len(gain.shape)):
        gain = torch.unsqueeze(gain, dim=-1)

    # Apply the gamma correction
    x_adjust: Tensor = gain * torch.pow(input, gamma)

    # Truncate between pixel values
    out: Tensor = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_contrast(image: Tensor, factor: Union[float, Tensor], clip_output: bool = True) -> Tensor:
    r"""Adjust the contrast of an image tensor.

    .. image:: _static/img/adjust_contrast.png

    This implementation follows Szeliski's book convention, where contrast is defined as
    a `multiplicative` operation directly to raw pixel values. Beware that other frameworks
    might use different conventions which can be difficult to reproduce exact results.

    The input image and factor is expected to be in the range of [0, 1].

    .. tip::
        This is not the preferred way to adjust the contrast of an image. Ideally one must
        implement :func:`kornia.enhance.adjust_gamma`. More details in the following link:
        https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_log_gamma.html#sphx-glr-auto-examples-color-exposure-plot-log-gamma-py

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        factor: Contrast adjust factor per element
            in the batch. 0 generates a completely black image, 1 does not modify
            the input image while any other non-negative number modify the
            brightness by this factor.
        clip_output: whether to clip the output image with range of [0, 1].

    Return:
        Adjusted image in the shape of :math:`(*, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/image_enhancement.html>`__.

    Example:
        >>> import torch
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_contrast(x, 0.5)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.tensor([0.65, 0.50])
        >>> adjust_contrast(x, y).shape
        torch.Size([2, 5, 3, 3])
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, Tensor)), "Factor should be float or Tensor.")

    if isinstance(factor, float):
        # TODO: figure out how to create later a tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, Tensor):
        factor = factor.to(image.device, image.dtype)

    # make factor broadcastable
    while len(factor.shape) != len(image.shape):
        factor = factor[..., None]

    KORNIA_CHECK(any(factor >= 0), "Contrast factor must be positive.")

    # Apply contrast factor to each channel
    img_adjust: Tensor = image * factor

    # Truncate between pixel values
    if clip_output:
        img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    return img_adjust


def adjust_contrast_with_mean_subtraction(image: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Adjust the contrast of an image tensor by subtracting the mean over channels.

    .. note::
        this is just a convenience function to have compatibility with Pil. For exact
        definition of image contrast adjustment consider using :func:`kornia.enhance.adjust_gamma`.

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        factor: Contrast adjust factor per element
            in the batch. 0 generates a completely black image, 1 does not modify
            the input image while any other non-negative number modify the
            brightness by this factor.

    Return:
        Adjusted image in the shape of :math:`(*, H, W)`.

    Example:
        >>> import torch
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_contrast_with_mean_subtraction(x, 0.5)
        tensor([[[[1., 1.],
                  [1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.tensor([0.65, 0.50])
        >>> adjust_contrast_with_mean_subtraction(x, y).shape
        torch.Size([2, 5, 3, 3])
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, Tensor)), "Factor should be float or Tensor.")

    if isinstance(factor, float):
        # TODO: figure out how to create later a tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, Tensor):
        factor = factor.to(image.device, image.dtype)

    # make factor broadcastable
    while len(factor.shape) != len(image.shape):
        factor = factor[..., None]

    # KORNIA_CHECK(any(factor >= 0), "Contrast factor must be positive.")

    if image.shape[-3] == 3:
        img_mean = rgb_to_grayscale(image).mean((-2, -1), True)
    else:
        img_mean = image.mean()

    # Apply contrast factor subtracting the mean
    img_adjust: Tensor = image * factor + img_mean * (1 - factor)

    img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    return img_adjust


def adjust_brightness(image: Tensor, factor: Union[float, Tensor], clip_output: bool = True) -> Tensor:
    r"""Adjust the brightness of an image tensor.

    .. image:: _static/img/adjust_brightness.png

    This implementation follows Szeliski's book convention, where brightness is defined as
    an `additive` operation directly to raw pixel and shift its values according the applied
    factor and range of the image values. Beware that other framework might use different
    conventions which can be difficult to reproduce exact results.

    The input image and factor is expected to be in the range of [0, 1].

    .. tip::
        By applying a large factor might prouce clipping or loss of image detail. We recommenda to
        apply small factors to avoid the mentioned issues. Ideally one must implement the adjustment
        of image intensity with other techniques suchs as :func:`kornia.enhance.adjust_gamma`. More
        details in the following link:
        https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_log_gamma.html#sphx-glr-auto-examples-color-exposure-plot-log-gamma-py

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        factor: Brightness adjust factor per element in the batch. It's recommended to
            bound the factor by [0, 1]. 0 does not modify the input image while any other
            number modify the brightness.

    Return:
        Adjusted tensor in the shape of :math:`(*, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/image_enhancement.html>`__.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_brightness(x, 1.)
        tensor([[[[1., 1.],
                  [1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.tensor([0.25, 0.50])
        >>> adjust_brightness(x, y).shape
        torch.Size([2, 5, 3, 3])
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, Tensor)), "Factor should be float or Tensor.")

    # convert factor to a tensor
    if isinstance(factor, float):
        # TODO: figure out how to create later a tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, Tensor):
        factor = factor.to(image.device, image.dtype)

    # make factor broadcastable
    while len(factor.shape) != len(image.shape):
        factor = factor[..., None]

    # shift pixel values
    img_adjust: Tensor = image + factor

    # truncate between pixel values
    if clip_output:
        img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    return img_adjust


def adjust_brightness_accumulative(image: Tensor, factor: Union[float, Tensor], clip_output: bool = True) -> Tensor:
    r"""Adjust the brightness accumulatively of an image tensor.

    This implementation follows PIL convention.

    The input image and factor is expected to be in the range of [0, 1].

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        factor: Brightness adjust factor per element in the batch. It's recommended to
            bound the factor by [0, 1]. 0 does not modify the input image while any other
            number modify the brightness.

    Return:
        Adjusted tensor in the shape of :math:`(*, H, W)`.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_brightness_accumulative(x, 1.)
        tensor([[[[1., 1.],
                  [1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.tensor([0.25, 0.50])
        >>> adjust_brightness_accumulative(x, y).shape
        torch.Size([2, 5, 3, 3])
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, Tensor)), "Factor should be float or Tensor.")

    # convert factor to a tensor
    if isinstance(factor, float):
        # TODO: figure out how to create later a tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, Tensor):
        factor = factor.to(image.device, image.dtype)

    # make factor broadcastable
    while len(factor.shape) != len(image.shape):
        factor = factor[..., None]

    # shift pixel values
    img_adjust: Tensor = image * factor

    # truncate between pixel values
    if clip_output:
        img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    return img_adjust


def adjust_sigmoid(image: Tensor, cutoff: float = 0.5, gain: float = 10, inv: bool = False) -> Tensor:
    """Adjust sigmoid correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
        [1]: Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions",
             http://markfairchild.org/PDFs/PAP07.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        cutoff: The cutoff of sigmoid function.
        gain: The multiplier of sigmoid function.
        inv: If is set to True the function will return the inverse sigmoid correction.

    Returns:
         Adjusted tensor in the shape of :math:`(*, H, W)`.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_sigmoid(x, gain=0)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")

    if inv:
        img_adjust = 1 - 1 / (1 + (gain * (cutoff - image)).exp())
    else:
        img_adjust = 1 / (1 + (gain * (cutoff - image)).exp())
    return img_adjust


def adjust_log(image: Tensor, gain: float = 1, inv: bool = False, clip_output: bool = True) -> Tensor:
    """Adjust log correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
    [1]: http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        gain: The multiplier of logarithmic function.
        inv:  If is set to True the function will return the inverse logarithmic correction.
        clip_output: Whether to clip the output image with range of [0, 1].

    Returns:
        Adjusted tensor in the shape of :math:`(*, H, W)`.

    Example:
        >>> x = torch.zeros(1, 1, 2, 2)
        >>> adjust_log(x, inv=True)
        tensor([[[[0., 0.],
                  [0., 0.]]]])
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")

    if inv:
        img_adjust = (2**image - 1) * gain
    else:
        img_adjust = (1 + image).log2() * gain

    # truncate between pixel values
    if clip_output:
        img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    return img_adjust


def _solarize(input: Tensor, thresholds: Union[float, Tensor] = 0.5) -> Tensor:
    r"""For each pixel in the image, select the pixel if the value is less than the threshold. Otherwise, subtract
    1.0 from the pixel.

    Args:
        input: image or batched images to solarize.
        thresholds: solarize thresholds.
            If int or one element tensor, input will be solarized across the whole batch.
            If 1-d tensor, input will be solarized element-wise, len(thresholds) == len(input).

    Returns:
        Solarized images.
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, Tensor)):
        raise TypeError(f"The factor should be either a float or Tensor. Got {type(thresholds)}")

    if isinstance(thresholds, Tensor) and len(thresholds.shape) != 0:
        if not (input.size(0) == len(thresholds) and len(thresholds.shape) == 1):
            raise AssertionError(f"thresholds must be a 1-d vector of shape ({input.size(0)},). Got {thresholds}")
        # TODO: I am not happy about this line, but no easy to do batch-wise operation
        thresholds = thresholds.to(input.device).to(input.dtype)
        thresholds = torch.stack([x.expand(*input.shape[-3:]) for x in thresholds])

    return torch.where(input < thresholds, input, 1.0 - input)


def solarize(
    input: Tensor,
    thresholds: Union[float, Tensor] = 0.5,
    additions: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    r"""For each pixel in the image less than threshold.

    .. image:: _static/img/solarize.png

    We add 'addition' amount to it and then clip the pixel value to be between 0 and 1.0.
    The value of 'addition' is between -0.5 and 0.5.

    Args:
        input: image tensor with shapes like :math:`(*, C, H, W)` to solarize.
        thresholds: solarize thresholds.
            If int or one element tensor, input will be solarized across the whole batch.
            If 1-d tensor, input will be solarized element-wise, len(thresholds) == len(input).
        additions: between -0.5 and 0.5.
            If None, no addition will be performed.
            If int or one element tensor, same addition will be added across the whole batch.
            If 1-d tensor, additions will be added element-wisely, len(additions) == len(input).

    Returns:
        The solarized images with shape :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = solarize(x, thresholds=0.5, additions=0.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(2, 4, 3, 3)
        >>> thresholds = torch.tensor([0.8, 0.5])
        >>> additions = torch.tensor([-0.25, 0.25])
        >>> solarize(x, thresholds, additions).shape
        torch.Size([2, 4, 3, 3])
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, Tensor)):
        raise TypeError(f"The factor should be either a float or Tensor. Got {type(thresholds)}")

    if isinstance(thresholds, float):
        thresholds = torch.as_tensor(thresholds)

    if additions is not None:
        if not isinstance(additions, (float, Tensor)):
            raise TypeError(f"The factor should be either a float or Tensor. Got {type(additions)}")

        if isinstance(additions, float):
            additions = torch.as_tensor(additions)

        if not torch.all((additions < 0.5) * (additions > -0.5)):
            raise AssertionError(f"The value of 'addition' is between -0.5 and 0.5. Got {additions}.")

        if isinstance(additions, Tensor) and len(additions.shape) != 0:
            if not (input.size(0) == len(additions) and len(additions.shape) == 1):
                raise AssertionError(f"additions must be a 1-d vector of shape ({input.size(0)},). Got {additions}")
            # TODO: I am not happy about this line, but no easy to do batch-wise operation
            additions = additions.to(input.device).to(input.dtype)
            additions = torch.stack([x.expand(*input.shape[-3:]) for x in additions])
        input = input + additions
        input = input.clamp(0.0, 1.0)

    return _solarize(input, thresholds)


@perform_keep_shape_image
def posterize(input: Tensor, bits: Union[int, Tensor]) -> Tensor:
    r"""Reduce the number of bits for each color channel.

    .. image:: _static/img/posterize.png

    Non-differentiable function, ``torch.uint8`` involved.

    Args:
        input: image tensor with shape :math:`(*, C, H, W)` to posterize.
        bits: number of high bits. Must be in range [0, 8].
            If int or one element tensor, input will be posterized by this bits.
            If 1-d tensor, input will be posterized element-wisely, len(bits) == input.shape[-3].
            If n-d tensor, input will be posterized element-channel-wisely, bits.shape == input.shape[:len(bits.shape)]

    Returns:
        Image with reduced color channels with shape :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 6, 3, 3)
        >>> out = posterize(x, bits=8)
        >>> torch.testing.assert_close(x, out)

        >>> x = torch.rand(2, 6, 3, 3)
        >>> bits = torch.tensor([4, 2])
        >>> posterize(x, bits).shape
        torch.Size([2, 6, 3, 3])
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

    if not isinstance(bits, (int, Tensor)):
        raise TypeError(f"bits type is not an int or Tensor. Got {type(bits)}")

    if isinstance(bits, int):
        bits = torch.as_tensor(bits)

    # TODO: find a better way to check boundaries on tensors
    # if not torch.all((bits >= 0) * (bits <= 8)) and bits.dtype == torch.int:
    #     raise ValueError(f"bits must be integers within range [0, 8]. Got {bits}.")

    # TODO: Make a differentiable version
    # Current version:
    # Ref: https://github.com/open-mmlab/mmcv/pull/132/files#diff-309c9320c7f71bedffe89a70ccff7f3bR19
    # Ref: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L222
    # Potential approach: implementing kornia.LUT with floating points
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py#L472
    def _left_shift(input: Tensor, shift: Tensor) -> Tensor:
        return ((input * 255).to(torch.uint8) * (2**shift)).to(input.dtype) / 255.0

    def _right_shift(input: Tensor, shift: Tensor) -> Tensor:
        return (input * 255).to(torch.uint8) / (2**shift).to(input.dtype) / 255.0

    def _posterize_one(input: Tensor, bits: Tensor) -> Tensor:
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
        if bits.shape[0] != input.shape[0]:
            raise AssertionError(
                f"Batch size must be equal between bits and input. Got {bits.shape[0]}, {input.shape[0]}."
            )

        for i in range(input.shape[0]):
            res.append(_posterize_one(input[i], bits[i]))
        return torch.stack(res, dim=0)

    if bits.shape != input.shape[: len(bits.shape)]:
        raise AssertionError(
            "Batch and channel must be equal between bits and input. "
            f"Got {bits.shape}, {input.shape[:len(bits.shape)]}."
        )
    _input = input.view(-1, *input.shape[len(bits.shape) :])
    _bits = bits.flatten()
    for i in range(input.shape[0]):
        res.append(_posterize_one(_input[i], _bits[i]))
    return torch.stack(res, dim=0).reshape(*input.shape)


@perform_keep_shape_image
def sharpness(input: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Apply sharpness to the input tensor.

    .. image:: _static/img/sharpness.png

    Implemented Sharpness function from PIL using torch ops. This implementation refers to:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L326

    Args:
        input: image tensor with shape :math:`(*, C, H, W)` to sharpen.
        factor: factor of sharpness strength. Must be above 0.
            If float or one element tensor, input will be sharpened by the same factor across the whole batch.
            If 1-d tensor, input will be sharpened element-wisely, len(factor) == len(input).

    Returns:
        Sharpened image or images with shape :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 1, 5, 5)
        >>> sharpness(x, 0.5).shape
        torch.Size([1, 1, 5, 5])
    """
    if not isinstance(factor, Tensor):
        factor = torch.as_tensor(factor, device=input.device, dtype=input.dtype)

    if len(factor.size()) != 0 and factor.shape != torch.Size([input.size(0)]):
        raise AssertionError(
            "Input batch size shall match with factor size if factor is not a 0-dim tensor. "
            f"Got {input.size(0)} and {factor.shape}"
        )

    kernel = (
        torch.as_tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=input.dtype, device=input.device)
        .view(1, 1, 3, 3)
        .repeat(input.size(1), 1, 1, 1)
        / 13
    )

    # This shall be equivalent to depthwise conv2d:
    # Ref: https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2
    degenerate = torch.nn.functional.conv2d(input, kernel, bias=None, stride=1, groups=input.size(1))
    degenerate = torch.clamp(degenerate, 0.0, 1.0)

    # For the borders of the resulting image, fill in the values of the original image.
    mask = torch.ones_like(degenerate)
    padded_mask = torch.nn.functional.pad(mask, [1, 1, 1, 1])
    padded_degenerate = torch.nn.functional.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, input)

    if len(factor.size()) == 0:
        return _blend_one(result, input, factor)
    return torch.stack([_blend_one(result[i], input[i], factor[i]) for i in range(len(factor))])


def _blend_one(input1: Tensor, input2: Tensor, factor: Tensor) -> Tensor:
    r"""Blend two images into one.

    Args:
        input1: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        input2: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        factor: factor 0-dim tensor.

    Returns:
        : image tensor with the batch in the zero position.
    """
    if not isinstance(input1, Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input1}.")
    if not isinstance(input2, Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input2}.")

    if isinstance(factor, Tensor) and len(factor.size()) != 0:
        raise AssertionError(f"Factor shall be a float or single element tensor. Got {factor}.")
    if factor == 0.0:
        return input1
    if factor == 1.0:
        return input2
    diff = (input2 - input1) * factor
    res = input1 + diff
    if factor > 0.0 and factor < 1.0:
        return res
    return torch.clamp(res, 0, 1)


def _build_lut(histo: Tensor, step: Tensor) -> Tensor:
    # Compute the cumulative sum, shifting by step // 2
    # and then normalization by step.
    step_trunc = torch.div(step, 2, rounding_mode="trunc")
    lut = torch.div(torch.cumsum(histo, 0) + step_trunc, step, rounding_mode="trunc")
    # Shift lut, prepending with 0.
    lut = torch.cat([torch.zeros(1, device=lut.device, dtype=lut.dtype), lut[:-1]])
    # Clip the counts to be in range.  This is done
    # in the C code for image.point.
    return torch.clamp(lut, 0, 255)


# Code taken from: https://github.com/pytorch/vision/pull/796
def _scale_channel(im: Tensor) -> Tensor:
    r"""Scale the data in the channel to implement equalize.

    Args:
        input: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.

    Returns:
        image tensor with the batch in the zero position.
    """
    min_ = im.min()
    max_ = im.max()

    if min_.item() < 0.0 and not torch.isclose(min_, torch.as_tensor(0.0, dtype=min_.dtype)):
        raise ValueError(f"Values in the input tensor must greater or equal to 0.0. Found {min_.item()}.")

    if max_.item() > 1.0 and not torch.isclose(max_, torch.as_tensor(1.0, dtype=max_.dtype)):
        raise ValueError(f"Values in the input tensor must lower or equal to 1.0. Found {max_.item()}.")

    ndims = len(im.shape)
    if ndims not in (2, 3):
        raise TypeError(f"Input tensor must have 2 or 3 dimensions. Found {ndims}.")

    im = im * 255.0
    # Compute the histogram of the image channel.
    histo = _torch_histc_cast(im, bins=256, min=0, max=255)
    # For the purposes of computing the step, filter out the nonzeros.
    nonzero_histo = torch.reshape(histo[histo != 0], [-1])
    step = torch.div(torch.sum(nonzero_histo) - nonzero_histo[-1], 255, rounding_mode="trunc")

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    if step == 0:
        result = im
    else:
        # can't index using 2d index. Have to flatten and then reshape
        result = torch.gather(_build_lut(histo, step), 0, im.flatten().long())
        result = result.reshape_as(im)

    return result / 255.0


@perform_keep_shape_image
def equalize(input: Tensor) -> Tensor:
    r"""Apply equalize on the input tensor.

    .. image:: _static/img/equalize.png

    Implements Equalize function from PIL using PyTorch ops based on uint8 format:
    https://github.com/tensorflow/tpu/blob/5f71c12a020403f863434e96982a840578fdd127/models/official/efficientnet/autoaugment.py#L355

    Args:
        input: image tensor to equalize with shape :math:`(*, C, H, W)`.

    Returns:
        Equalized image tensor with shape :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 2, 3, 3)
        >>> equalize(x).shape
        torch.Size([1, 2, 3, 3])
    """
    res = []
    for image in input:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_image = torch.stack([_scale_channel(image[i, :, :]) for i in range(len(image))])
        res.append(scaled_image)
    return torch.stack(res)


@perform_keep_shape_video
def equalize3d(input: Tensor) -> Tensor:
    r"""Equalize the values for a 3D volumetric tensor.

    Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352

    Args:
        input: image tensor with shape :math:`(*, C, D, H, W)` to equalize.

    Returns:
        Equalized volume with shape :math:`(B, C, D, H, W)`.
    """
    res = []
    for volume in input:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_input = torch.stack([_scale_channel(volume[i, :, :, :]) for i in range(len(volume))])
        res.append(scaled_input)

    return torch.stack(res)


def invert(image: Tensor, max_val: Tensor = Tensor([1.0])) -> Tensor:
    r"""Invert the values of an input image tensor by its maximum value.

    .. image:: _static/img/invert.png

    Args:
        image: The input tensor to invert with an arbitatry shape.
        max_val: The expected maximum value in the input tensor. The shape has to
          according to the input tensor shape, or at least has to work with broadcasting.

    Example:
        >>> img = torch.rand(1, 2, 4, 4)
        >>> invert(img).shape
        torch.Size([1, 2, 4, 4])

        >>> img = 255. * torch.rand(1, 2, 3, 4, 4)
        >>> invert(img, torch.as_tensor(255.)).shape
        torch.Size([1, 2, 3, 4, 4])

        >>> img = torch.rand(1, 3, 4, 4)
        >>> invert(img, torch.as_tensor([[[[1.]]]])).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(image, Tensor):
        raise AssertionError(f"Input is not a Tensor. Got: {type(input)}")

    if not isinstance(max_val, Tensor):
        raise AssertionError(f"max_val is not a Tensor. Got: {type(max_val)}")

    return max_val.to(image) - image


class AdjustSaturation(Module):
    r"""Adjust color saturation of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        saturation_factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
        saturation_mode: The mode to adjust saturation.

    Shape:
        - Input: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        - Output: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustSaturation(2.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2)
        >>> out = AdjustSaturation(y)(x)
        >>> torch.nn.functional.mse_loss(x, out)
        tensor(0.)
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def __init__(self, saturation_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.saturation_factor: Union[float, Tensor] = saturation_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_saturation(input, self.saturation_factor)


class AdjustSaturationWithGraySubtraction(Module):
    r"""Adjust color saturation of an image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    The input image is expected to be an RGB or gray image in the range of [0, 1].

    Args:
        saturation_factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
        saturation_mode: The mode to adjust saturation.

    Shape:
        - Input: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        - Output: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustSaturationWithGraySubtraction(2.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2)
        >>> out = AdjustSaturationWithGraySubtraction(y)(x)
        >>> torch.nn.functional.mse_loss(x, out)
        tensor(0.)
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def __init__(self, saturation_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.saturation_factor: Union[float, Tensor] = saturation_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_saturation_with_gray_subtraction(input, self.saturation_factor)


class AdjustHue(Module):
    r"""Adjust hue of an image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        hue_factor: How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Shape:
        - Input: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        - Output: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustHue(3.141516)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2) * 3.141516
        >>> AdjustHue(y)(x).shape
        torch.Size([2, 3, 3, 3])
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def __init__(self, hue_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.hue_factor: Union[float, Tensor] = hue_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_hue(input, self.hue_factor)


class AdjustGamma(Module):
    r"""Perform gamma correction on an image.

    The input image is expected to be in the range of [0, 1].

    Args:
        gamma: Non negative real number, same as y\gammay in the equation.
          gamma larger than 1 make the shadows darker, while gamma smaller than 1 make
          dark regions lighter.
        gain: The constant multiplier.

    Shape:
        - Input: Image to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustGamma(1.0, 2.0)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y1 = torch.ones(2) * 1.0
        >>> y2 = torch.ones(2) * 2.0
        >>> AdjustGamma(y1, y2)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, gamma: Union[float, Tensor], gain: Union[float, Tensor] = 1.0) -> None:
        super().__init__()
        self.gamma: Union[float, Tensor] = gamma
        self.gain: Union[float, Tensor] = gain

    def forward(self, input: Tensor) -> Tensor:
        return adjust_gamma(input, self.gamma, self.gain)


class AdjustContrast(Module):
    r"""Adjust Contrast of an image.

    This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        contrast_factor: Contrast adjust factor per element
          in the batch. 0 generates a completely black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustContrast(0.5)(x)
        tensor([[[[0.5000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 0.5000]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustContrast(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, contrast_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.contrast_factor: Union[float, Tensor] = contrast_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_contrast(input, self.contrast_factor)


class AdjustContrastWithMeanSubtraction(Module):
    r"""Adjust Contrast of an image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        contrast_factor: Contrast adjust factor per element
          in the batch by subtracting its mean grayscaled version.
          0 generates a completely black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustContrastWithMeanSubtraction(0.5)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustContrastWithMeanSubtraction(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, contrast_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.contrast_factor: Union[float, Tensor] = contrast_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_contrast_with_mean_subtraction(input, self.contrast_factor)


class AdjustBrightness(Module):
    r"""Adjust Brightness of an image.

    This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        brightness_factor: Brightness adjust factor per element
          in the batch. 0 does not modify the input image while any other number modify the
          brightness.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustBrightness(1.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustBrightness(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, brightness_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.brightness_factor: Union[float, Tensor] = brightness_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_brightness(input, self.brightness_factor)


class AdjustSigmoid(Module):
    r"""Adjust the contrast of an image tensor or performs sigmoid correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
        [1]: Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions",
             http://markfairchild.org/PDFs/PAP07.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        cutoff: The cutoff of sigmoid function.
        gain: The multiplier of sigmoid function.
        inv: If is set to True the function will return the negative sigmoid correction.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> AdjustSigmoid(gain=0)(x)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])
    """

    def __init__(self, cutoff: float = 0.5, gain: float = 10, inv: bool = False) -> None:
        super().__init__()
        self.cutoff: float = cutoff
        self.gain: float = gain
        self.inv: bool = inv

    def forward(self, image: Tensor) -> Tensor:
        return adjust_sigmoid(image, cutoff=self.cutoff, gain=self.gain, inv=self.inv)


class AdjustLog(Module):
    """Adjust log correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
    [1]: http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        gain: The multiplier of logarithmic function.
        inv:  If is set to True the function will return the inverse logarithmic correction.
        clip_output: Whether to clip the output image with range of [0, 1].

    Example:
        >>> x = torch.zeros(1, 1, 2, 2)
        >>> AdjustLog(inv=True)(x)
        tensor([[[[0., 0.],
                  [0., 0.]]]])
    """

    def __init__(self, gain: float = 1, inv: bool = False, clip_output: bool = True) -> None:
        super().__init__()
        self.gain: float = gain
        self.inv: bool = inv
        self.clip_output: bool = clip_output

    def forward(self, image: Tensor) -> Tensor:
        return adjust_log(image, gain=self.gain, inv=self.inv, clip_output=self.clip_output)


class AdjustBrightnessAccumulative(Module):
    r"""Adjust Brightness of an image accumulatively.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        brightness_factor: Brightness adjust factor per element
          in the batch. 0 does not modify the input image while any other number modify the
          brightness.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustBrightnessAccumulative(1.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustBrightnessAccumulative(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, brightness_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.brightness_factor: Union[float, Tensor] = brightness_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_brightness_accumulative(input, self.brightness_factor)


class Invert(Module):
    r"""Invert the values of an input tensor by its maximum value.

    Args:
        input: The input tensor to invert with an arbitatry shape.
        max_val: The expected maximum value in the input tensor. The shape has to
          according to the input tensor shape, or at least has to work with broadcasting. Default: 1.0.

    Example:
        >>> img = torch.rand(1, 2, 4, 4)
        >>> Invert()(img).shape
        torch.Size([1, 2, 4, 4])

        >>> img = 255. * torch.rand(1, 2, 3, 4, 4)
        >>> Invert(torch.as_tensor(255.))(img).shape
        torch.Size([1, 2, 3, 4, 4])

        >>> img = torch.rand(1, 3, 4, 4)
        >>> Invert(torch.as_tensor([[[[1.]]]]))(img).shape
        torch.Size([1, 3, 4, 4])
    """

    def __init__(self, max_val: Tensor = torch.tensor(1.0)) -> None:
        super().__init__()
        if not isinstance(max_val, Parameter):
            self.register_buffer("max_val", max_val)
        else:
            self.max_val = max_val

    def forward(self, input: Tensor) -> Tensor:
        return invert(input, self.max_val)
