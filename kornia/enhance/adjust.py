# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import struct
from math import pi
from typing import ClassVar, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.color import hsv_to_rgb, rgb_to_grayscale, rgb_to_hsv
from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_COLOR_OR_GRAY,
    KORNIA_CHECK_IS_TENSOR,
    _check_async,
    are_checks_enabled,
)
from kornia.core.exceptions import TypeCheckError
from kornia.core.ops import _device_literal
from kornia.core.utils import _normalize_to_float32_or_float64, _to_device_nonblocking, is_compiling
from kornia.image.utils import _to_bcdhw, _to_bchw


def _clip_unit_fresh(image: torch.Tensor) -> torch.Tensor:
    """Clamp a tensor this module just allocated into ``[0, 1]``, in place when no gradient is recorded.

    The values equal ``image.clamp(0, 1)``. Without a recorded gradient the clamp reuses the fresh buffer instead of
    allocating another image-sized one: at batch 32 that third 25 MB temporary per call made glibc trim and re-fault
    the heap on every call. Only call it on a tensor no caller can see.

    An integer tensor, which reaches here only with checks disabled, keeps the out-of-place clamp: the float bounds
    promote its result to the default float dtype, which an in-place clamp cannot do. So does a tensor under a function
    transform: vmap has a batching rule for ``clamp`` but not for ``clamp_``, whose fallback loops over the batch and
    rejects an empty one. TorchScript cannot compile the function-transform query, so a scripted call keeps it too.
    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if (
            image.is_floating_point()
            and not (torch.is_grad_enabled() and image.requires_grad)
            and not torch._C._are_functorch_transforms_active()
        ):
            return image.clamp_(min=0.0, max=1.0)
    return image.clamp(min=0.0, max=1.0)


def _check_floating_image(image: torch.Tensor, op: str) -> None:
    """Reject an integer image in an op that expects floating-point values in ``[0, 1]``.

    These ops clamp to ``[0, 1]``, scale by 255 or index a 256-entry table, so an 8-bit image would
    otherwise come back saturated, wrapped or negative without an error (#5981). The equalize family's
    ``[0, 1]`` range check let an integer 0/1 mask through, which ``equalize_clahe`` then mapped to wrong
    values. The check reads only the dtype, so it stays ``torch.compile(fullgraph=True)``-safe, and
    ``disable_checks()`` skips it like every kornia check. A non-tensor is left to the op's own type check.
    As in the core checks, TorchScript compiles neither the checks flag nor the exception metadata, so a
    scripted op raises a plain message instead.
    """
    if not isinstance(image, torch.Tensor) or image.is_floating_point():
        return
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if not are_checks_enabled():
            return
    if torch.jit.is_scripting():
        raise TypeCheckError(
            op + " expects a floating-point image in [0, 1]; convert it first, e.g. image.float() / 255."
        )
    raise TypeCheckError(
        f"{op} expects a floating-point image in [0, 1], got {image.dtype}. "
        "Convert it first, for example image.float() / 255 for 8-bit data.",
        actual_type=image.dtype,
        expected_type=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )


def _make_factor_broadcastable(factor: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Right-pad ``factor`` with singleton dimensions so it broadcasts against ``image``.

    A ``factor`` with more dimensions than ``image`` can never reach the image rank by appending
    dimensions, so padding it would loop forever. Reject it here instead of hanging. The padding
    only lines the ranks up: shapes that still do not broadcast are reported by the op itself.
    """
    if factor.dim() > image.dim():
        if torch.jit.is_scripting():
            raise ValueError("Factor has more dimensions than the image and cannot be broadcast.")
        raise ValueError(
            f"Factor has more dimensions than the image and cannot be broadcast: got factor shape "
            f"{tuple(factor.shape)} for image shape {tuple(image.shape)}."
        )

    while factor.dim() != image.dim():
        factor = factor[..., None]

    return factor


def adjust_saturation_raw(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Scale the saturation channel of an HSV image by a factor and clamp it to [0, 1].

    Convention:
        Expects HSV, not RGB. The saturation channel is multiplied by factor
        and clamped to [0, 1]; :func:`adjust_saturation` performs RGB conversion.

    Expecting image to be in hsv format already.
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, torch.Tensor)), "Factor should be float or torch.Tensor.")

    if isinstance(factor, float):
        # TODO: figure out how to create later a torch.Tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, torch.Tensor):
        factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)

    # unpack the hsv values
    h, s, v = torch.chunk(image, chunks=3, dim=-3)

    # transform the hue value and appl module
    s_out: torch.Tensor = torch.clamp(s * factor, min=0, max=1)

    # pack back back the corrected hue
    out: torch.Tensor = torch.cat([h, s_out, v], dim=-3)

    return out


def adjust_saturation_with_gray_subtraction(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust the saturation of an RGB image by blending it with its grayscale version, as PIL and torchvision do.

    Convention:
        factor=0 selects grayscale and factor=1 preserves RGB input; grayscale
        input is returned unchanged. This differs from HSV saturation scaling.

    The image is expected to be an RGB image or a gray image in the range of [0, 1].
    If it is an RGB image, returns blending of the image with its grayscaled version.
    If it is a gray image, returns the image.

    .. note::
        this is just a convenience function to have compatibility with Pil

    Args:
        image: Image/torch.Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
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
    _check_floating_image(image, "adjust_saturation_with_gray_subtraction")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, torch.Tensor)), "Factor should be float or torch.Tensor.")
    KORNIA_CHECK_IS_COLOR_OR_GRAY(image, "Image should be an RGB or gray image")

    if image.shape[-3] == 1:
        return image

    if isinstance(factor, float):
        # TODO: figure out how to create later a torch.Tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, torch.Tensor):
        factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)

    x_other: torch.Tensor = rgb_to_grayscale(image)

    # blend the image with the grayscaled image
    x_adjusted: torch.Tensor = (1 - factor) * x_other + factor * image

    # clamp the output
    out: torch.Tensor = torch.clamp(x_adjusted, 0.0, 1.0)

    return out


def adjust_saturation(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust the saturation of an RGB image by scaling its HSV saturation channel.

    Convention:
        Expects RGB in [0, 1] with channel dimension -3. factor=1 preserves the image;
        factor=0 removes HSV saturation, so each pixel becomes max(R, G, B) rather than its luma.
        :func:`adjust_saturation_with_gray_subtraction` blends with luma, as torchvision and PIL do;
        see :func:`adjust_saturation_raw` for the HSV-domain primitive.

    .. image:: _static/img/adjust_saturation.png

    The image is expected to be an RGB image in the range of [0, 1].

    Args:
        image: Image/torch.Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.

    Return:
        Adjusted image in the shape of :math:`(*, 3, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> adjust_saturation(x, 2.).shape
        torch.Size([1, 3, 3, 3])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.tensor([1., 2.])
        >>> adjust_saturation(x, y).shape
        torch.Size([2, 3, 3, 3])

    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, torch.Tensor)), "Factor should be float or torch.Tensor.")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # Reordering the HSV arithmetic has visibly different rounding in half precision (up to
    # ~7e-3 in float16 and ~6e-2 in bfloat16 over the unit cube). Keep the established path there;
    # the direct formula below agrees to float32 precision and targets the common augmentation dtype.
    if not image.is_floating_point() or image.dtype in (torch.float16, torch.bfloat16):
        return hsv_to_rgb(adjust_saturation_raw(rgb_to_hsv(image), factor))

    if isinstance(factor, float):
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    else:
        factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)
    if factor.shape[-3] != 1:
        raise ValueError(f"Factor must hold one value per image, not per channel. Got shape {factor.shape}")

    # Scaling saturation in HSV keeps value (the maximum RGB channel) and hue fixed, so it is
    # equivalent to scaling each channel's distance from the minimum RGB channel. Expressing that
    # relation directly avoids materialising an HSV image and the expensive HSV-to-RGB sextant
    # selection. Keep rgb_to_hsv's guarded divisors so gradients stay finite at black and
    # grayscale pixels.
    max_rgb = image.amax(dim=-3, keepdim=True)
    min_rgb = image.amin(dim=-3, keepdim=True)
    delta = max_rgb - min_rgb
    value_divisor = torch.where(max_rgb == 0, torch.ones_like(max_rgb), max_rgb + 1e-8)
    saturation = delta / value_divisor
    saturation = torch.clamp(saturation * factor, min=0, max=1)

    adjusted_delta = max_rgb * saturation
    delta_divisor = torch.where(delta == 0, torch.ones_like(delta), delta)
    return (image - min_rgb) * (adjusted_delta / delta_divisor) + (max_rgb - adjusted_delta)


def adjust_hue_raw(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Shift the hue channel of an HSV image by an angle in radians, wrapping it into [0, 2π).

    Convention:
        Expects HSV. factor is an angle in radians added to the hue, and the sum is wrapped into
        [0, 2π). :func:`adjust_hue` supplies the RGB-domain wrapper.

    Expecting image to be in hsv format already.
    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(
        isinstance(factor, (float, torch.Tensor)),
        f"The factor should be a float number or torch.Tensor in the range between [-PI, PI]. Got {type(factor)}",
    )

    if isinstance(factor, float):
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)

    factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)

    # unpack the hsv values
    h, s, v = torch.chunk(image, chunks=3, dim=-3)

    # transform the hue value and appl module
    divisor: float = 2 * pi
    h_out: torch.Tensor = torch.remainder(h + factor, divisor)
    # A small negative sum rounds up to the period itself (0 - 1e-8 gives 2π in float32); fold it to 0.
    h_out = h_out.masked_fill(h_out >= divisor, 0.0)

    # pack back back the corrected hue
    out: torch.Tensor = torch.cat([h_out, s, v], dim=-3)

    return out


def adjust_hue(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Shift the hue of an RGB image by an angle in radians.

    Convention:
        Expects RGB in [0, 1] with channel dimension -3. factor is a hue angle
        in radians; see :func:`adjust_hue_raw` for the HSV operation.

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
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

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
    x_hsv: torch.Tensor = rgb_to_hsv(image)

    # perform the conversion
    x_adjusted: torch.Tensor = adjust_hue_raw(x_hsv, factor)

    # convert back to rgb
    out: torch.Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_gamma(
    input: torch.Tensor, gamma: Union[float, torch.Tensor], gain: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    r"""Perform gamma correction on an image.

    Convention:
        Applies gain * input ** gamma elementwise, then clamps to [0, 1]. For ``0 < gamma < 1`` the derivative of
        ``input ** gamma`` at a black pixel (``input == 0``) is infinite; the gradient there is 0 instead.

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
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    .. note::
       The non-negativity check on ``gamma``/``gain`` reads a CPU value during the call and raises
       :exc:`~kornia.core.exceptions.AsyncCheckError`, a :exc:`RuntimeError`. A tensor on CUDA, or on MPS
       from torch 2.13, is checked with ``torch._assert_async`` instead and raises ``RuntimeError`` by the
       next synchronization; on CUDA that is a device-side assert.

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
    _check_floating_image(input, "adjust_gamma")
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(gamma, (float, torch.Tensor)):
        raise TypeError(f"The gamma should be a positive float or torch.Tensor. Got {type(gamma)}")

    if not isinstance(gain, (float, torch.Tensor)):
        raise TypeError(f"The gain should be a positive float or torch.Tensor. Got {type(gain)}")

    # Python scalars need no device-side validation or parameter transfers. Half precision
    # keeps the tensor path: rounding a parameter before the operation matters there.
    # Keep invalid values on that path too, including negatives that round to signed zero.
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript must skip the eager-only body
        if (
            not torch.jit.is_tracing()
            and not is_compiling()
            and isinstance(gamma, float)
            and isinstance(gain, float)
            and input.dtype in (torch.float32, torch.float64)
            and input.ndim > 0
        ):
            limits = torch.finfo(input.dtype)
            if 0.0 <= gamma <= limits.max and 0.0 <= gain <= limits.max:
                # Scalar power inspects its exponent before converting to the tensor dtype.
                # Round on the host first, as tensor construction did: near zero/one this
                # affects both the value at black pixels and its gradient. Captured calls
                # retain the tensor expression, which the compiler can constant-fold.
                if input.dtype == torch.float32:
                    gamma = struct.unpack("f", struct.pack("f", gamma))[0]
                    gain = struct.unpack("f", struct.pack("f", gain))[0]
                if torch.is_grad_enabled() and input.requires_grad and 0.0 < gamma < 1.0:
                    # As in the tensor path below.
                    black = input == 0
                    power = torch.where(
                        black, torch.zeros_like(input), torch.where(black, torch.ones_like(input), input).pow(gamma)
                    )
                else:
                    power = input.pow(gamma)
                return (power * gain).clamp(0.0, 1.0)

    if isinstance(gamma, float):
        gamma = torch.tensor([gamma], device=input.device, dtype=input.dtype)

    if isinstance(gain, float):
        gain = torch.tensor([gain], device=input.device, dtype=input.dtype)

    gamma = gamma.to(input.device).to(input.dtype)
    gain = gain.to(input.device).to(input.dtype)

    _check_async(
        (gamma >= 0.0).all(),
        "Gamma must be non-negative. Clamp it first: max(gamma, 0.0) for floats, gamma.clamp_min(0.0) for tensors.",
    )
    _check_async(
        (gain >= 0.0).all(),
        "Gain must be non-negative. Clamp it first: max(gain, 0.0) for floats, gain.clamp_min(0.0) for tensors.",
    )

    gamma = _make_factor_broadcastable(gamma, input)
    gain = _make_factor_broadcastable(gain, input)

    # Apply the gamma correction. For 0 < gamma < 1, x ** gamma has an infinite derivative at x = 0, which the clamp
    # below turns into NaN (inf before torch 2.14). Black pixels then take the power of a safe base, keep their value
    # 0 ** gamma = 0 and get a zero gradient; gamma = 0 keeps 0 ** 0 = 1. The values are the same either way, so the
    # masking only runs when autograd records a gradient for the input.
    if torch.is_grad_enabled() and input.requires_grad:
        black = (input == 0) & (gamma > 0) & (gamma < 1)
        power = torch.where(
            black, torch.zeros_like(input), torch.pow(torch.where(black, torch.ones_like(input), input), gamma)
        )
    else:
        power = torch.pow(input, gamma)
    x_adjust: torch.Tensor = gain * power

    # Truncate between pixel values
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_contrast(image: torch.Tensor, factor: Union[float, torch.Tensor], clip_output: bool = True) -> torch.Tensor:
    r"""Adjust the contrast of an image by multiplying its values by a factor.

    Convention:
        Contrast is a direct multiplicative scale of RGB values; factor=1
        preserves input and clip_output controls the final [0, 1] clamp.

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
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    .. note::
       The non-negativity check on ``factor`` reads a CPU value during the call and raises
       :exc:`~kornia.core.exceptions.AsyncCheckError`, a :exc:`RuntimeError`. A tensor on CUDA, or on MPS
       from torch 2.13, is checked with ``torch._assert_async`` instead and raises ``RuntimeError`` by the
       next synchronization; on CUDA that is a device-side assert.

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
    _check_floating_image(image, "adjust_contrast")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, torch.Tensor)), "Factor should be float or torch.Tensor.")

    if isinstance(factor, float):
        # TODO: figure out how to create later a torch.Tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, torch.Tensor):
        factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)

    _check_async(
        (factor >= 0).all(),
        "Contrast factor must be non-negative. Clamp it first: max(factor, 0.0) for floats, "
        "factor.clamp_min(0.0) for tensors.",
    )

    # Apply contrast factor to each channel
    img_adjust: torch.Tensor = image * factor

    # Truncate between pixel values
    if clip_output:
        img_adjust = _clip_unit_fresh(img_adjust)

    return img_adjust


def adjust_contrast_with_mean_subtraction(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust the contrast of an image by blending it with its mean gray level, as PIL and torchvision do.

    .. note::
        this is just a convenience function to have compatibility with Pil. For exact
        definition of image contrast adjustment consider using :func:`kornia.enhance.adjust_gamma`.

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        factor: Contrast adjust factor per element
            in the batch. 0 gives a uniform image at the mean gray level, 1 does not modify
            the input image and a larger factor increases the contrast.

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
    _check_floating_image(image, "adjust_contrast_with_mean_subtraction")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, torch.Tensor)), "Factor should be float or torch.Tensor.")

    if isinstance(factor, float):
        # TODO: figure out how to create later a torch.Tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, torch.Tensor):
        factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)

    # KORNIA_CHECK(any(factor >= 0), "Contrast factor must be positive.")

    img_mean = rgb_to_grayscale(image).mean((-2, -1), True) if image.shape[-3] == 3 else image.mean((-3, -2, -1), True)

    # Apply contrast factor subtracting the mean
    img_adjust: torch.Tensor = image * factor + img_mean * (1 - factor)

    return img_adjust.clamp(min=0.0, max=1.0)


def adjust_brightness(
    image: torch.Tensor, factor: Union[float, torch.Tensor], clip_output: bool = True
) -> torch.Tensor:
    r"""Adjust the brightness of an image torch.Tensor.

    Convention:
        Adds factor to RGB values; clip_output controls the final [0, 1] clamp.

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
        clip_output: Whether to clip output to be in [0,1].

    Return:
        Adjusted torch.Tensor in the shape of :math:`(*, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

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
    _check_floating_image(image, "adjust_brightness")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, torch.Tensor)), "Factor should be float or torch.Tensor.")

    # convert factor to a torch.Tensor
    if isinstance(factor, float):
        # TODO: figure out how to create later a torch.Tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, torch.Tensor):
        factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)

    # shift pixel values
    img_adjust: torch.Tensor = image + factor

    # truncate between pixel values
    if clip_output:
        img_adjust = _clip_unit_fresh(img_adjust)

    return img_adjust


def adjust_brightness_accumulative(
    image: torch.Tensor, factor: Union[float, torch.Tensor], clip_output: bool = True
) -> torch.Tensor:
    r"""Adjust the brightness accumulatively of an image torch.Tensor.

    Convention:
        Multiplies each image by factor; factor=1 preserves it and factor=0
        produces black, unlike :func:`adjust_brightness` which adds factor.

    This implementation follows PIL convention.

    The input image and factor is expected to be in the range of [0, 1].

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        factor: Brightness factor per element in the batch. The image is multiplied by it, so ``1``
            leaves an image in ``[0, 1]`` unchanged, ``0`` gives a black image and a factor above ``1``
            brightens it.
        clip_output: Whether to clip output to be in [0,1].

    Return:
        Adjusted torch.Tensor in the shape of :math:`(*, H, W)`.

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
    _check_floating_image(image, "adjust_brightness_accumulative")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(isinstance(factor, (float, torch.Tensor)), "Factor should be float or torch.Tensor.")

    # convert factor to a torch.Tensor
    if isinstance(factor, float):
        # TODO: figure out how to create later a torch.Tensor without importing torch
        factor = torch.as_tensor(factor, device=image.device, dtype=image.dtype)
    elif isinstance(factor, torch.Tensor):
        factor = _to_device_nonblocking(factor, image.device, image.dtype)

    factor = _make_factor_broadcastable(factor, image)

    # shift pixel values
    img_adjust: torch.Tensor = image * factor

    # truncate between pixel values
    if clip_output:
        img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    return img_adjust


def adjust_sigmoid(image: torch.Tensor, cutoff: float = 0.5, gain: float = 10, inv: bool = False) -> torch.Tensor:
    """Adjust sigmoid correction on the input image torch.Tensor.

    Convention:
        Applies a sigmoid around cutoff; inv selects its complementary curve.

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
         Adjusted torch.Tensor in the shape of :math:`(*, H, W)`.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_sigmoid(x, gain=0)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])

    """
    _check_floating_image(image, "adjust_sigmoid")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")

    return 1 - 1 / (1 + (gain * (cutoff - image)).exp()) if inv else 1 / (1 + (gain * (cutoff - image)).exp())


def adjust_log(image: torch.Tensor, gain: float = 1, inv: bool = False, clip_output: bool = True) -> torch.Tensor:
    """Adjust log correction on the input image torch.Tensor.

    Convention:
        inv selects exponential instead of logarithmic correction; clip_output
        controls the final [0, 1] clamp.

    The input image is expected to be in the range of [0, 1].

    Reference:
    [1]: http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        gain: The multiplier of logarithmic function.
        inv:  If is set to True the function will return the inverse logarithmic correction.
        clip_output: Whether to clip the output image with range of [0, 1].

    Returns:
        Adjusted torch.Tensor in the shape of :math:`(*, H, W)`.

    Example:
        >>> x = torch.zeros(1, 1, 2, 2)
        >>> adjust_log(x, inv=True)
        tensor([[[[0., 0.],
                  [0., 0.]]]])

    """
    _check_floating_image(image, "adjust_log")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")

    img_adjust = (2**image - 1) * gain if inv else (1 + image).log2() * gain

    # truncate between pixel values
    if clip_output:
        img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    return img_adjust


def _solarize(input: torch.Tensor, thresholds: Union[float, torch.Tensor] = 0.5) -> torch.Tensor:
    r"""For each pixel in the image, select the pixel if the value is less than the threshold.

    Otherwise, replace it with ``1.0 - value``.

    Args:
        input: image or batched images to solarize.
        thresholds: solarize thresholds.
            If int or one element torch.Tensor, input will be solarized across the whole batch.
            If 1-d torch.Tensor, input will be solarized element-wise, len(thresholds) == len(input).

    Returns:
        Solarized images.

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, torch.Tensor)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. Got {type(thresholds)}")

    if isinstance(thresholds, torch.Tensor) and len(thresholds.shape) != 0:
        if not (input.size(0) == len(thresholds) and len(thresholds.shape) == 1):
            raise AssertionError(f"thresholds must be a 1-d vector of shape ({input.size(0)},). Got {thresholds}")
        # Reshape the per-sample threshold to broadcast over (C, H, W) rather than iterating the
        # tensor (`for x in thresholds` breaks torch.compile); the `where` below broadcasts identically.
        thresholds = _to_device_nonblocking(thresholds, input.device).to(input.dtype)
        thresholds = thresholds.view([-1] + [1] * (input.ndim - 1))

    return torch.where(input < thresholds, input, 1.0 - input)


def solarize(
    input: torch.Tensor,
    thresholds: Union[float, torch.Tensor] = 0.5,
    additions: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Add an amount to every pixel, clamp, then invert the pixels at or above a threshold.

    Convention:
        additions are applied and clamped before values at or above thresholds
        are inverted about one.

    .. image:: _static/img/solarize.png

    ``additions`` is added to the **whole** image and the sum is clamped into ``[0, 1]``; only then is each
    pixel compared against ``thresholds``, and every pixel at or above it is replaced by ``1 - value``.
    The addition is therefore applied on both sides of the threshold, and it can move a pixel across it:
    with ``thresholds=0.5`` and ``additions=0.2``, ``[0.1, 0.4, 0.6, 0.9]`` comes back as
    ``[0.3, 0.4, 0.2, 0.0]``. The value of 'addition' is in the closed range [-0.5, 0.5].

    Args:
        input: image torch.Tensor with shapes like :math:`(*, C, H, W)` to solarize.
        thresholds: solarize thresholds.
            If int or one element torch.Tensor, input will be solarized across the whole batch.
            If 1-d torch.Tensor, input will be solarized element-wise, len(thresholds) == len(input).
        additions: in the closed range [-0.5, 0.5], endpoints included.
            If None, no addition will be performed.
            If int or one element torch.Tensor, same addition will be added across the whole batch.
            If 1-d torch.Tensor, additions will be added element-wisely, len(additions) == len(input).

    Returns:
        The solarized images with shape :math:`(*, C, H, W)`.

    .. note::
       The range check on ``additions`` reads a float or a CPU tensor during the call and raises
       :exc:`~kornia.core.exceptions.AsyncCheckError`, a :exc:`RuntimeError`, whatever the device of the image.
       A tensor on CUDA, or on MPS from torch 2.13, is checked with ``torch._assert_async`` instead and raises
       ``RuntimeError`` by the next synchronization; on CUDA that is a device-side assert.

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
    _check_floating_image(input, "solarize")
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, torch.Tensor)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. Got {type(thresholds)}")

    if isinstance(thresholds, float):
        thresholds = torch.as_tensor(thresholds, device=input.device, dtype=input.dtype)

    if additions is not None:
        if not isinstance(additions, (float, torch.Tensor)):
            raise TypeError(f"The factor should be either a float or torch.Tensor. Got {type(additions)}")

        if isinstance(additions, float):
            # Check the Python value on the host, whatever the default device: rounded to float16/bfloat16, a
            # value just past the bound passes.
            in_range = torch.tensor(-0.5 <= additions <= 0.5, device="cpu")
            additions = torch.as_tensor(additions, device=input.device, dtype=input.dtype)
        else:
            in_range = ((additions <= 0.5) & (additions >= -0.5)).all()

        _check_async(
            in_range,
            "The addition must be in the closed range [-0.5, 0.5]. Clamp it first: min(max(additions, -0.5), 0.5) "
            "for floats, additions.clamp(-0.5, 0.5) for tensors.",
        )

        if isinstance(additions, torch.Tensor) and len(additions.shape) != 0:
            if not (input.size(0) == len(additions) and len(additions.shape) == 1):
                raise AssertionError(f"additions must be a 1-d vector of shape ({input.size(0)},). Got {additions}")
            # Broadcast the per-sample scalar over (C, H, W) by reshaping instead of iterating the
            # tensor (`for x in additions` breaks the graph); numerically identical.
            additions = _to_device_nonblocking(additions, input.device).to(input.dtype)
            additions = additions.view([-1] + [1] * (input.ndim - 1))
        input = input + additions
        input = input.clamp(0.0, 1.0)

    return _solarize(input, thresholds)


def _prepare_intensity_input(input: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """Batch image or volume inputs without accepting empty channels or spatial dimensions."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    if 0 in input.shape[-(spatial_dims + 1) :]:
        raise ValueError(f"Input channel and spatial dimensions must be non-zero. Got {input.shape}.")
    return _to_bcdhw(input) if spatial_dims == 3 else _to_bchw(input)


def posterize(input: torch.Tensor, bits: Union[int, torch.Tensor]) -> torch.Tensor:
    r"""Reduce the number of bits for each color channel.

    Convention:
        Expects values in [0, 1] and quantizes them through 8-bit integers; bits=8 returns the
        input unchanged. bits is a scalar, (1,), or a leading prefix of the input shape such as
        (B,) or (B, C), with arbitrary leading image dimensions flattened into B.
        Empty leading dimensions are supported; channels and spatial dimensions must be non-empty.

    .. image:: _static/img/posterize.png

    Non-differentiable function, ``torch.uint8`` involved.

    Args:
        input: image torch.Tensor with shape :math:`(*, C, H, W)` to posterize.
        bits: number of high bits. Must be in range [0, 8].
            If int or one-element torch.Tensor, the same bits value is used across the whole batch.
            A 1-d tensor with multiple elements specifies one value per sample, not per channel;
            its length must equal the normalized batch size ``B`` after flattening leading dimensions.
            Any other length raises
            ``Batch size must be equal between bits and input``.
            A tensor with multiple dimensions specifies element-channel-wise values; its shape must
            match the corresponding leading dimensions of the normalized ``(B, C, H, W)`` input.

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
    _check_floating_image(input, "posterize")
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    input_shape = input.shape
    input = _prepare_intensity_input(input, 2)

    if not isinstance(bits, (int, torch.Tensor)):
        raise TypeError(f"bits type is not an int or torch.Tensor. Got {type(bits)}")

    if isinstance(bits, int):
        bits = torch.as_tensor(bits, device=input.device)

    # A user-supplied CPU bits tensor mixes with any device only when 0-dim (PyTorch's "wrapped
    # number" rule); multi-element CPU tensors hit a device mismatch on MPS/CUDA, so move bits
    # onto input's device explicitly (no-op when already there).
    bits = bits.to(device=input.device)

    # TODO: find a better way to check boundaries on tensors
    # if not torch.all((bits >= 0) * (bits <= 8)) and bits.dtype == torch.int:
    #     raise ValueError(f"bits must be integers within range [0, 8]. Got {bits}.")

    # TODO: Make a differentiable version
    # Current version:
    # Ref: https://github.com/open-mmlab/mmcv/pull/132/files#diff-309c9320c7f71bedffe89a70ccff7f3bR19
    # Ref: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L222
    # Potential approach: implementing kornia.LUT with floating points
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py#L472
    def _left_shift(input: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return ((input * 255).to(torch.uint8) * (2**shift)).to(input.dtype) / 255.0

    def _right_shift(input: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return (input * 255).to(torch.uint8) / (2**shift).to(input.dtype) / 255.0

    def _posterize_one(input: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        # Branchless for torch.compile: the bits==0 / bits==8 special cases guard a uint8
        # overflow in the shift math, so compute the shift unconditionally and select with
        # torch.where (verified numerically identical to the branched version for bits 0..8).
        shift = 8 - bits
        shifted = _left_shift(_right_shift(input, shift), shift)
        out = torch.where(bits == 0, torch.zeros_like(input), shifted)
        return torch.where(bits == 8, input, out)

    if len(bits.shape) == 0 or (len(bits.shape) == 1 and len(bits) == 1):
        return _posterize_one(input, bits).reshape(input_shape)

    if len(bits.shape) == 1:
        if bits.shape[0] != input.shape[0]:
            raise AssertionError(
                f"Batch size must be equal between bits and input. Got {bits.shape[0]}, {input.shape[0]}."
            )

        # Broadcast the per-sample bits over (B, C, H, W) and posterize in one vectorised call
        # instead of a Python loop over the batch (which launched a kernel per sample).
        return _posterize_one(input, bits.view(-1, *([1] * (input.ndim - 1)))).reshape(input_shape)

    if bits.shape != input.shape[: len(bits.shape)]:
        raise AssertionError(
            "Batch and channel must be equal between bits and input. "
            f"Got {bits.shape}, {input.shape[: len(bits.shape)]}."
        )
    # Broadcast per-(batch, channel) bits over the spatial dims and posterize in one call.
    _bits = bits.view(*bits.shape, *([1] * (input.ndim - bits.ndim)))
    return _posterize_one(input, _bits).reshape(input_shape)


def sharpness(input: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Apply sharpness to the input torch.Tensor.

    Convention:
        factor=0 selects the blurred image and factor=1 preserves the input;
        factor is a float or a (B,) tensor with one value per image, with arbitrary leading image
        dimensions flattened into B. Empty leading dimensions are supported; channels must be
        non-empty and both spatial dimensions must be at least 3 pixels.

    .. image:: _static/img/sharpness.png

    Implemented Sharpness function from PIL using torch ops. This implementation refers to:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L326

    Args:
        input: image torch.Tensor with shape :math:`(*, C, H, W)` to sharpen.
        factor: blend factor between the smoothed copy and the input. ``0`` returns the smoothed image,
            ``1`` returns the input, and a factor above ``1`` sharpens. It is not validated: ``0`` and
            negative factors are accepted and extrapolate past the smoothed image.
            A scalar or 0-d tensor applies the same factor across the whole batch.
            A 1-d tensor specifies one factor per sample; its length must equal the normalized batch
            size ``B`` after flattening leading dimensions. A length-one vector does not broadcast
            when ``B`` is greater than one.

    Returns:
        Sharpened image or images with shape :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 1, 5, 5)
        >>> sharpness(x, 0.5).shape
        torch.Size([1, 1, 5, 5])

    """
    _check_floating_image(input, "sharpness")
    batched_input = _prepare_intensity_input(input, 2)
    input_shape = input.shape
    input = batched_input

    if isinstance(factor, (int, float)):
        # Filled on the device: ``torch.as_tensor(..., device=cuda)`` copies from the host and waits for every
        # queued kernel.
        factor = _device_literal(factor, device=input.device, dtype=input.dtype)
    elif not isinstance(factor, torch.Tensor):
        factor = torch.as_tensor(factor, device=input.device, dtype=input.dtype)
    else:
        # Match the input's device/dtype (as adjust_brightness/contrast do). Augmentation
        # generators may sample the factor on a different device (e.g. CPU) than the image.
        factor = _to_device_nonblocking(factor, input.device, input.dtype)

    if len(factor.size()) != 0 and factor.shape != torch.Size([input.size(0)]):
        raise AssertionError(
            "Input batch size shall match with factor size if factor is not a 0-dim torch.Tensor. "
            f"Got {input.size(0)} and {factor.shape}"
        )

    kernel = (
        _device_literal([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=input.dtype, device=input.device)
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
    padded_mask = F.pad(mask, [1, 1, 1, 1])
    padded_degenerate = F.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, input)

    # Blend the sharpened result with the input. factor may be 0-dim (whole batch) or
    # 1-d (per-sample); reshape to broadcast, then blend branchlessly so the op stays
    # torch.compile fullgraph-safe (no per-sample Python loop, no factor==0/1 branches).
    # Equivalent to the old _blend_one: clamp is a no-op for factor in [0, 1] and only
    # bites for extrapolation, matching the previous behavior.
    if len(factor.size()) != 0:
        factor = factor.view(-1, *([1] * (result.dim() - 1)))
    return torch.clamp(result + (input - result) * factor, 0.0, 1.0).reshape(input_shape)


def _scale_channel_batched(input: torch.Tensor) -> torch.Tensor:
    r"""Vectorized histogram equalization over the leading ``(B, C)`` planes.

    Computes all per-plane histograms/LUTs in a single batched pass instead of a Python loop of
    ``B * C`` serial calls. Each pixel is counted in the LUT entry it reads, its 8-bit level
    ``(x * 255).long()`` computed in the input dtype, so a plane whose LUT step is not 0 comes out as
    ``PIL.ImageOps.equalize`` of those levels while the counts and their running sums are exact (a plane of
    up to ``2**24`` pixels in float32); a plane whose step is 0 is returned unchanged.

    Args:
        input: image tensor shaped ``(B, C, *spatial)`` with values in ``[0, 1]``.

    Returns:
        The equalized tensor, same shape as ``input``.
    """
    shape = input.shape
    n = shape[0] * shape[1]
    scaled = input.flatten(0, 1).flatten(1) * 255.0  # (N, P)

    # Input is expected in [0, 1]. The histogram and the LUT lookup both index with ``scaled.long()``,
    # which is out of bounds unless ``scaled`` is in (-1, 256).
    # Check that domain without ``.item()``, so there is no device sync and fullgraph still compiles;
    # on MPS before torch 2.13 it costs one host sync instead (see ``_check_async``). Inputs the
    # lookup can index keep working exactly as before. The check keeps the index in bounds, so it runs
    # with checks disabled too: a compiled CPU scatter with an index out of range terminates the process.
    _check_async(
        ((scaled > -1.0) & (scaled < 256.0)).all(),
        "equalize expects input values in [0, 1]. Scale the image into that range first, "
        "for example image / 255.0 for 8-bit data.",
        guard=True,
    )

    # Count in float32 (float64 stays float64): a count grows with the plane size, which half precision
    # cannot hold. float16 rounds counts above 2048 and their total overflows to inf past 65504, and
    # bfloat16 rounds counts above 256.
    count_dtype = _normalize_to_float32_or_float64(scaled.dtype)
    # Per-plane 256-bin histogram on the index the lookup below reads, so every pixel is counted in its own
    # entry, as PIL counts and reads one 8-bit level. ``torch.histc(x, 256, 0, 255)`` bins in ``floor(256 x)``,
    # which puts every value in ``[k/256, k/255)``, about half of a continuous image, one bin above that entry.
    index = scaled.long()
    histo = torch.zeros(n, 256, device=input.device, dtype=count_dtype)
    histo.scatter_add_(1, index, torch.ones_like(scaled, dtype=count_dtype))

    # step = (sum(nonzero) - last_nonzero) // 255, per plane.
    total = histo.sum(1)
    ar = torch.arange(256, device=input.device)
    last_idx = torch.where(histo > 0, ar, torch.zeros_like(ar)).amax(1)
    last_count = histo.gather(1, last_idx.unsqueeze(1)).squeeze(1)
    step = torch.div(total - last_count, 255, rounding_mode="trunc")  # (N,)

    # Build the LUT (cumsum shifted by step // 2, normalized by step); guard step == 0 for the div
    # and select the untouched plane afterwards, mirroring the scalar ``if step == 0`` branch.
    step_col = step.unsqueeze(1)
    step_trunc = torch.div(step, 2, rounding_mode="trunc").unsqueeze(1)
    lut = torch.div(histo.cumsum(1) + step_trunc, step_col.clamp(min=1), rounding_mode="trunc")
    lut = torch.cat([torch.zeros(n, 1, device=input.device, dtype=lut.dtype), lut[:, :-1]], 1)
    # The LUT holds integers in [0, 255], which every floating dtype represents exactly.
    lut = torch.clamp(lut, 0, 255).to(scaled.dtype)

    result = lut.gather(1, index)
    result = torch.where(step_col == 0, scaled, result)
    return (result / 255.0).reshape(shape)


def equalize(input: torch.Tensor) -> torch.Tensor:
    r"""Apply equalize on the input torch.Tensor.

    Convention:
        Equalizes each channel of each image independently over [0, 1].
        :func:`equalize3d` applies the corresponding operation to volumes.
        Empty leading dimensions are supported; channels and spatial dimensions must be non-empty.

    .. image:: _static/img/equalize.png

    Implements Equalize function from PIL using PyTorch ops based on uint8 format:
    https://github.com/tensorflow/tpu/blob/5f71c12a020403f863434e96982a840578fdd127/models/official/efficientnet/autoaugment.py#L355

    Args:
        input: image torch.Tensor to equalize with shape :math:`(*, C, H, W)`.

    Returns:
        Equalized image torch.Tensor with shape :math:`(*, C, H, W)`.

    .. note::
       The input is expected as a floating-point image in :math:`[0, 1]`, and each channel is equalized from
       a 256-bin histogram. Each pixel is counted in the lookup-table entry it reads, its 8-bit level
       ``(x * 255).long()`` in the input dtype, so a float32 or float64 image gives ``PIL.ImageOps.equalize``
       of those levels. A channel whose lookup step is 0 is returned unchanged, and a float32 image counts in
       float32, which is exact while a channel holds at most ``2**24`` pixels (4096 x 4096). An integer or
       boolean image raises
       :exc:`~kornia.core.exceptions.TypeCheckError` (a :exc:`TypeError`); convert 8-bit data with
       ``image.float() / 255`` first. Values the 256-bin lookup cannot index (outside roughly :math:`[0, 1]`)
       raise a ``RuntimeError`` naming the range, during the call for a CPU image, even when checks are
       disabled, since the check keeps the lookup in bounds. On CUDA, and on MPS from torch ``2.13``, the check
       is ``torch._assert_async`` and raises by the next synchronization; on an older MPS release the condition
       is read on the host instead, one device sync per call, and is skipped there under ``torch.compile``.

    Example:
        >>> x = torch.rand(1, 2, 3, 3)
        >>> equalize(x).shape
        torch.Size([1, 2, 3, 3])

    """
    _check_floating_image(input, "equalize")
    # Scales each channel independently, batched over (B, C) in a single pass.
    return _scale_channel_batched(_prepare_intensity_input(input, 2)).reshape(input.shape)


def equalize3d(input: torch.Tensor) -> torch.Tensor:
    r"""Equalize the values for a 3D volumetric torch.Tensor.

    Convention:
        Equalizes each channel of each volume independently over [0, 1]; see
        :func:`equalize` for the image variant and :func:`equalize_clahe` for local tiling.
        Empty leading dimensions are supported; channels and spatial dimensions must be non-empty.

    Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352

    Args:
        input: image torch.Tensor with shape :math:`(*, C, D, H, W)` to equalize.

    Returns:
        Equalized volume with the same shape as the input.

    .. note::
       The input is expected as a floating-point volume in :math:`[0, 1]`, and each channel's whole
       :math:`(D, H, W)` volume is equalized from one 256-bin histogram. The lookup step is an integer
       division by 255, so a volume with no more than 255 voxels per channel is returned unchanged up to
       floating-point roundoff; just above that, whether it changes depends on the values. An integer or
       boolean volume raises :exc:`~kornia.core.exceptions.TypeCheckError`, as for :func:`equalize`. Values
       the 256-bin lookup cannot index (outside roughly :math:`[0, 1]`) raise a ``RuntimeError`` naming the
       range, during the call for a CPU volume, even when checks are disabled, as for :func:`equalize`. On CUDA,
       and on MPS from torch ``2.13``, the check is ``torch._assert_async`` and raises by the next
       synchronization; on an older MPS release the condition is read on the host instead, one device sync per
       call, and is skipped there under ``torch.compile``.

    """
    _check_floating_image(input, "equalize3d")
    # Scales each channel independently (each (D, H, W) volume), batched over (B, C).
    return _scale_channel_batched(_prepare_intensity_input(input, 3)).reshape(input.shape)


def invert(image: torch.Tensor, max_val: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Invert the values of an input image torch.Tensor by its maximum value.

    Convention:
        Returns max_val - image elementwise. Omitting max_val uses one in the
        input dtype and device; pass max_val=torch.tensor(255) for uint8 images.

    .. image:: _static/img/invert.png

    Args:
        image: The input torch.Tensor to invert with an arbitatry shape.
        max_val: The expected maximum value in the input torch.Tensor. The shape has to
          according to the input torch.Tensor shape, or at least has to work with broadcasting.

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
    if not isinstance(image, torch.Tensor):
        raise AssertionError(f"Input is not a torch.Tensor. Got: {type(input)}")

    _max_val = torch.tensor([1.0]) if max_val is None else max_val
    if not isinstance(_max_val, torch.Tensor):
        raise AssertionError(f"max_val is not a torch.Tensor. Got: {type(_max_val)}")

    return _max_val.to(image) - image


class AdjustSaturation(nn.Module):
    r"""Adjust the saturation of an RGB image by scaling its HSV saturation channel.

    Convention:
        See :func:`adjust_saturation` for the RGB input and factor policy.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        saturation_factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.

    Shape:
        - Input: Image/torch.Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
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

    def __init__(self, saturation_factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.saturation_factor: Union[float, torch.Tensor] = saturation_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adjust RGB saturation using the module's configured factor.

        Args:
            input: Input RGB tensor, typically shaped :math:`(*, 3, H, W)`.

        Returns:
            Saturation-adjusted tensor with the same shape as ``input``.
        """
        return adjust_saturation(input, self.saturation_factor)


class AdjustSaturationWithGraySubtraction(nn.Module):
    r"""Adjust the saturation of an RGB image by blending it with its grayscale version, as PIL and torchvision do.

    Convention:
        See :func:`adjust_saturation_with_gray_subtraction` for blending.

    It blends the image with its grayscale, ``factor * image + (1 - factor) * gray``, as torchvision and PIL do.
    The input image is expected to be in the range of [0, 1].

    The input image is expected to be an RGB or gray image in the range of [0, 1].

    Args:
        saturation_factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.

    Shape:
        - Input: Image/torch.Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
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

    def __init__(self, saturation_factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.saturation_factor: Union[float, torch.Tensor] = saturation_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adjust saturation using gray-subtraction interpolation behavior.

        This forwards to :func:`adjust_saturation_with_gray_subtraction`,
        which follows PIL-like saturation behavior.

        Args:
            input: Input image tensor, typically shaped :math:`(*, 3, H, W)`.

        Returns:
            Saturation-adjusted tensor with the same shape as ``input``.
        """
        return adjust_saturation_with_gray_subtraction(input, self.saturation_factor)


class AdjustHue(nn.Module):
    r"""Shift the hue of an RGB image by an angle in radians.

    Convention:
        See :func:`adjust_hue` for RGB input and radians convention.

    ``factor`` is in radians: it matches torchvision's ``adjust_hue`` with ``hue_factor = factor / (2 * pi)``.
    The input image is expected to be in the range of [0, 1].

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        hue_factor: How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Shape:
        - Input: Image/torch.Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
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

    def __init__(self, hue_factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.hue_factor: Union[float, torch.Tensor] = hue_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Shift image hue by the module's configured hue factor.

        Args:
            input: Input RGB tensor in the expected image range, usually shaped
                :math:`(*, 3, H, W)`.

        Returns:
            Hue-adjusted tensor with the same shape as ``input``.
        """
        return adjust_hue(input, self.hue_factor)


class AdjustGamma(nn.Module):
    r"""Perform gamma correction on an image.

    Convention:
        See :func:`adjust_gamma` for the elementwise power convention.

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

    def __init__(self, gamma: Union[float, torch.Tensor], gain: Union[float, torch.Tensor] = 1.0) -> None:
        super().__init__()
        self.gamma: Union[float, torch.Tensor] = gamma
        self.gain: Union[float, torch.Tensor] = gain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply gamma correction using this module's ``gamma`` and ``gain``.

        Args:
            input: Input tensor to correct, commonly image-shaped
                :math:`(*, C, H, W)`.

        Returns:
            Gamma-corrected tensor with the same shape as ``input``.
        """
        return adjust_gamma(input, self.gamma, self.gain)


class AdjustContrast(nn.Module):
    r"""Adjust the contrast of an image by multiplying its values by a factor.

    Convention:
        See :func:`adjust_contrast` for the multiplicative scale; this module always clips to [0, 1].

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

    def __init__(self, contrast_factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.contrast_factor: Union[float, torch.Tensor] = contrast_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adjust contrast using the configured contrast factor.

        Args:
            input: Input tensor to adjust, typically shaped :math:`(*, C, H, W)`.

        Returns:
            Contrast-adjusted tensor with the same shape as ``input``.
        """
        return adjust_contrast(input, self.contrast_factor)


class AdjustContrastWithMeanSubtraction(nn.Module):
    r"""Adjust the contrast of an image by blending it with its mean gray level, as PIL and torchvision do.

    Convention:
        See :func:`adjust_contrast_with_mean_subtraction` for its mean blend.

    For an RGB image it blends the image with its grayscale mean, ``factor * image + (1 - factor) * mean``, as
    torchvision and PIL do.
    The input image is expected to be in the range of [0, 1].

    Args:
        contrast_factor: Contrast adjust factor per element
          in the batch by subtracting its mean grayscaled version.
          0 gives a uniform image at the mean gray level, 1 does not modify
          the input image and a larger factor increases the contrast.

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

    def __init__(self, contrast_factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.contrast_factor: Union[float, torch.Tensor] = contrast_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adjust contrast with mean-subtraction based interpolation.

        This forwards to :func:`adjust_contrast_with_mean_subtraction`, which
        applies PIL-like contrast behavior.

        Args:
            input: Input tensor to adjust, typically shaped :math:`(*, C, H, W)`.

        Returns:
            Contrast-adjusted tensor with the same shape as ``input``.
        """
        return adjust_contrast_with_mean_subtraction(input, self.contrast_factor)


class AdjustBrightness(nn.Module):
    r"""Adjust Brightness of an image.

    Convention:
        See :func:`adjust_brightness` for the additive factor; this module always clips to [0, 1].

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

    def __init__(self, brightness_factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.brightness_factor: Union[float, torch.Tensor] = brightness_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adjust image brightness using the configured brightness factor.

        Args:
            input: Input tensor to adjust, commonly shaped :math:`(*, C, H, W)`.

        Returns:
            Brightness-adjusted tensor with the same shape as ``input``.
        """
        return adjust_brightness(input, self.brightness_factor)


class AdjustSigmoid(nn.Module):
    r"""Adjust the contrast of an image torch.Tensor or performs sigmoid correction on the input image torch.Tensor.

    Convention:
        See :func:`adjust_sigmoid` for cutoff and inverse-curve policy.

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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid-based intensity remapping to the input image.

        Args:
            image: Input image tensor to remap, typically shaped
                :math:`(*, C, H, W)` or :math:`(*, H, W)`.

        Returns:
            Tensor with sigmoid correction applied, preserving input shape.
        """
        return adjust_sigmoid(image, cutoff=self.cutoff, gain=self.gain, inv=self.inv)


class AdjustLog(nn.Module):
    """Adjust log correction on the input image torch.Tensor.

    Convention:
        See :func:`adjust_log` for logarithmic and inverse behavior.

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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply logarithmic or inverse-logarithmic intensity correction.

        Args:
            image: Input image tensor to transform, typically shaped
                :math:`(*, C, H, W)` or :math:`(*, H, W)`.

        Returns:
            Log-corrected tensor with the same shape as ``image``.
        """
        return adjust_log(image, gain=self.gain, inv=self.inv, clip_output=self.clip_output)


class AdjustBrightnessAccumulative(nn.Module):
    r"""Adjust Brightness of an image accumulatively.

    Convention:
        See :func:`adjust_brightness_accumulative` for the multiplicative factor; this module always clips
        to [0, 1].

    It multiplies the image by ``factor``, as torchvision's and PIL's brightness does.
    The input image is expected to be in the range of [0, 1].

    Args:
        brightness_factor: Brightness factor per element in the batch. The image is multiplied by it and
          clamped into ``[0, 1]``, so ``1`` leaves an image in ``[0, 1]`` unchanged, ``0`` gives a black image
          and a factor above ``1`` brightens it.

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

    def __init__(self, brightness_factor: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.brightness_factor: Union[float, torch.Tensor] = brightness_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adjust brightness with accumulative (PIL-like) behavior.

        Args:
            input: Input tensor to adjust, commonly shaped :math:`(*, C, H, W)`.

        Returns:
            Brightness-adjusted tensor with the same shape as ``input``.
        """
        return adjust_brightness_accumulative(input, self.brightness_factor)


class Invert(nn.Module):
    r"""Invert the values of an input torch.Tensor by its maximum value.

    Convention:
        See :func:`invert` for max-value subtraction behavior.

    Args:
        input: The input torch.Tensor to invert with an arbitatry shape.
        max_val: The expected maximum value in the input torch.Tensor. The shape has to
          according to the input torch.Tensor shape, or at least has to work with broadcasting. Default: 1.0.

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

    def __init__(self, max_val: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if max_val is None:
            max_val = torch.tensor(1.0)
        if not isinstance(max_val, nn.Parameter):
            self.register_buffer("max_val", max_val)
        else:
            self.max_val = max_val

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Invert values using the module's configured maximum reference.

        Args:
            input: Tensor to invert. Any shape is accepted as long as it is
                broadcast-compatible with ``self.max_val``.

        Returns:
            Inverted tensor with the same shape as ``input``.
        """
        return invert(input, self.max_val)
