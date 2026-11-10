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
from kornia.core.utils import (
    _clamp_unit_range,
    _is_tracing_or_exporting,
    _needs_derivatives,
    _normalize_to_float32_or_float64,
    _to_device_nonblocking,
    is_compiling,
)
from kornia.image.utils import _to_bcdhw, _to_bchw


def _clip_unit_fresh(image: torch.Tensor) -> torch.Tensor:
    """Clamp a tensor this module just allocated into ``[0, 1]``, in place when no derivative can be asked for.

    The values equal ``image.clamp(0, 1)``. Without a recorded gradient or a forward-mode tangent the clamp reuses the
    fresh buffer instead of allocating another image-sized one: at batch 32 that third 25 MB temporary per call made
    glibc trim and re-fault the heap on every call. With one it is ``_clamp_unit_range``, whose derivative at a bound
    does not depend on the torch version. A captured graph takes the same path whatever the grad mode, so that
    ``torch.jit.trace``'s check under ``no_grad`` records the same clamp. Only call it on a tensor no caller can see.

    An integer tensor, which reaches here only with checks disabled, keeps the out-of-place clamp: the float bounds
    promote its result to the default float dtype, which an in-place clamp cannot do. So does a tensor under a function
    transform: vmap has a batching rule for ``clamp`` but not for ``clamp_``, whose fallback loops over the batch and
    rejects an empty one. TorchScript cannot compile the function-transform query, so a scripted call keeps it too.
    """
    if not torch.jit.is_scripting():  # noqa: SIM102 - TorchScript only skips a bare is_scripting() guard
        if (
            image.is_floating_point()
            and not _is_tracing_or_exporting()
            and not _needs_derivatives(image)
            and not torch._C._are_functorch_transforms_active()
        ):
            return image.clamp_(min=0.0, max=1.0)
    return _clamp_unit_range(image)


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


def _level_scale(dtype: torch.dtype, top: float = 255.0) -> float:
    """Return the factor that turns a value in ``[0, 1]`` into the level in ``0 .. top`` its truncation reads.

    ``(x * _level_scale(x.dtype)).long()`` is the 8-bit level ``(x * 255).long()``, except that ``k / 255`` reads
    level ``k`` even when it was rounded a few units in the last place below ``k / 255``. CUDA divides by a Python
    number by multiplying with its rounded reciprocal, as ``x * (1 / 255)`` does on any device, which leaves 24 of the
    256 float64 levels one unit below ``k / 255``; truncating ``x * 255`` read those as ``k - 1``. For float32 and
    float64 the factor is ``top`` raised by four float32 epsilons, relative (``2**-21``, eight units in the last place
    of 255 in float32), so the margin grows with the level as that roundoff does, and a continuous value moves up a
    level only within that margin below a level boundary, about 7 in 100000 values of a uniform image. float64 takes
    the float32 margin rather than its own, so the same values read the same levels in both dtypes, and float32
    roundoff carried into a float64 image is covered too. Half precision gets ``top`` itself, so the levels it reads
    are unchanged: ``x * 255`` is rounded to the half-precision grid, which already gives ``k`` back for ``k / 255``
    rounded to half precision.
    """
    if dtype in (torch.float32, torch.float64):
        return top * (1.0 + 2.0**-21)
    return top


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

    ``out = adjust_saturation_raw(image, factor)`` multiplies channel 1 (saturation) of a ``(*, 3, H, W)`` HSV image by
    ``factor``, clamps it to :math:`[0, 1]` and leaves hue and value as they are. For an RGB image, use
    :func:`adjust_saturation`.

    Args:
        image: HSV image with shape ``(*, 3, H, W)``.
        factor: saturation multiplier, a float or a tensor with shape ``(B,)`` for one value per image.

    Returns:
        HSV image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`adjust_saturation`: the same scaling on an RGB image.
        - :func:`adjust_hue_raw`: shifts the hue of an HSV image.

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
    s_out: torch.Tensor = s * factor
    # Integer bounds keep the dtype of an integer image, which this HSV primitive does not reject.
    s_out = _clamp_unit_range(s_out) if s_out.is_floating_point() else s_out.clamp(min=0, max=1)

    # pack back back the corrected hue
    out: torch.Tensor = torch.cat([h, s_out, v], dim=-3)

    return out


def adjust_saturation_with_gray_subtraction(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust the saturation of an RGB image by blending it with its grayscale version, as PIL does.

    ``out = adjust_saturation_with_gray_subtraction(image, factor)`` returns ``factor * image + (1 - factor) * gray``
    for a ``(*, 3, H, W)`` RGB image in :math:`[0, 1]`, clamped to :math:`[0, 1]`: ``0`` gives the gray image and ``1``
    the input. A single-channel image is returned unchanged.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``, or gray image with shape ``(*, 1, H, W)``.
        factor: weight of the image, a float or a tensor with shape ``(B,)`` for one value per image.

    Returns:
        Adjusted image with the shape of ``image``.

    See Also:
        - :func:`adjust_saturation`: scales HSV saturation, so ``0`` gives max(R, G, B) instead of the gray level.
        - :class:`AdjustSaturationWithGraySubtraction`: the module form.

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
    out: torch.Tensor = _clamp_unit_range(x_adjusted)

    return out


def adjust_saturation(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust the saturation of an RGB image by scaling its HSV saturation channel.

    ``out = adjust_saturation(image, factor)`` multiplies the HSV saturation of a ``(*, 3, H, W)`` RGB image in
    :math:`[0, 1]` by ``factor``, clamps it to :math:`[0, 1]` and returns RGB. ``0`` gives gray pixels equal to
    max(R, G, B), not the gray level; for PIL's blend with the gray level, use
    :func:`adjust_saturation_with_gray_subtraction`.

    .. image:: _static/img/adjust_saturation.png

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.
        factor: saturation multiplier, a float or a tensor with shape ``(B,)`` for one value per image.

    Returns:
        Adjusted image with shape ``(*, 3, H, W)``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    See Also:
        - :func:`adjust_saturation_with_gray_subtraction`: blends with the gray image instead of scaling the HSV
          saturation.
        - :func:`adjust_saturation_raw`: the same scaling on an HSV image.
        - :class:`AdjustSaturation`: the module form.

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
    saturation = _clamp_unit_range(saturation * factor)

    adjusted_delta = max_rgb * saturation
    delta_divisor = torch.where(delta == 0, torch.ones_like(delta), delta)
    return (image - min_rgb) * (adjusted_delta / delta_divisor) + (max_rgb - adjusted_delta)


def adjust_hue_raw(image: torch.Tensor, factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Shift the hue channel of an HSV image by an angle in radians, wrapping it into [0, 2π).

    ``out = adjust_hue_raw(image, factor)`` adds ``factor`` radians to channel 0 (hue) of a ``(*, 3, H, W)`` HSV image
    and wraps the sum into :math:`[0, 2\pi)`, leaving saturation and value as they are. For an RGB image, use
    :func:`adjust_hue`.

    Args:
        image: HSV image with shape ``(*, 3, H, W)``, hue in radians.
        factor: angle in radians, a number or a tensor with shape ``(B,)`` for one value per image.

    Returns:
        HSV image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`adjust_hue`: the same shift on an RGB image.
        - :func:`adjust_saturation_raw`: scales the saturation of an HSV image.
        - :func:`~kornia.color.rgb_to_hsv`: makes the HSV image this takes, with hue in radians.

    """
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")
    KORNIA_CHECK(
        isinstance(factor, (int, float, torch.Tensor)) and not isinstance(factor, bool),
        f"The factor should be an int, a float or a torch.Tensor. Got {type(factor)}",
    )

    if isinstance(factor, (int, float)):
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

    ``out = adjust_hue(image, factor)`` rotates the hue of a ``(*, 3, H, W)`` RGB image in :math:`[0, 1]` by ``factor``
    radians and returns RGB.

    .. image:: _static/img/adjust_hue.png

    Convention:
        - ``factor`` is any angle in radians, taken modulo :math:`2\pi`: ``factor`` and ``factor + 2π`` agree, ``±π``
          gives the complementary colors, and a positive angle turns red towards green.
        - torchvision's ``adjust_hue`` takes a fraction of a turn in :math:`[-0.5, 0.5]`: its ``hue_factor`` is
          ``factor / (2 * pi)`` for ``factor`` in :math:`[-\pi, \pi]`.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.
        factor: hue shift in radians, a number or a tensor with shape ``(B,)`` for one value per image.

    Returns:
        Adjusted image with shape ``(*, 3, H, W)``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    See Also:
        - :func:`adjust_hue_raw`: the same shift on an HSV image.
        - :class:`AdjustHue`: the module form.

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
    r"""Apply gamma correction ``gain * input ** gamma`` to an image.

    ``out = adjust_gamma(input, gamma, gain)`` takes an image in :math:`[0, 1]` and clamps the result to
    :math:`[0, 1]`; ``gamma > 1`` darkens the shadows and ``gamma < 1`` lightens them.

    .. image:: _static/img/adjust_gamma.png

    Convention:
        For ``0 < gamma < 1`` the derivative of ``input ** gamma`` at a black pixel (``input == 0``) is infinite; the
        gradient there is 0 instead.

    Args:
        input: image with shape ``(*, H, W)``.
        gamma: non-negative exponent, a float or a tensor with shape ``(B,)`` for one value per image.
        gain: non-negative multiplier, a float or a tensor with shape ``(B,)``.

    Returns:
        Adjusted image with the shape of ``input``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    .. note::
       The non-negativity check on ``gamma``/``gain`` reads a CPU value during the call and raises
       :exc:`~kornia.core.exceptions.AsyncCheckError`, a :exc:`RuntimeError`. A tensor on CUDA, or on MPS
       from torch 2.13, is checked with ``torch._assert_async`` instead and raises ``RuntimeError`` by the
       next synchronization; on CUDA that is a device-side assert.

    See Also:
        - :class:`AdjustGamma`: the module form.
        - :func:`adjust_sigmoid`: an S-shaped curve around a cutoff.
        - :func:`adjust_log`: a logarithmic curve.

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
                if 0.0 < gamma < 1.0 and _needs_derivatives(input):
                    # As in the tensor path below.
                    black = input == 0
                    power = torch.where(
                        black, torch.zeros_like(input), torch.where(black, torch.ones_like(input), input).pow(gamma)
                    )
                else:
                    power = input.pow(gamma)
                return _clamp_unit_range(power * gain)

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
    # below would pass on. Black pixels then take the power of a safe base, keep their value 0 ** gamma = 0 and get a
    # zero gradient; gamma = 0 keeps 0 ** 0 = 1. The values are the same either way, so the masking only runs when
    # autograd can ask for a derivative through the input. The mask costs 5 times or more without a derivative.
    if _needs_derivatives(input):
        black = (input == 0) & (gamma > 0) & (gamma < 1)
        power = torch.where(
            black, torch.zeros_like(input), torch.pow(torch.where(black, torch.ones_like(input), input), gamma)
        )
    else:
        power = torch.pow(input, gamma)
    x_adjust: torch.Tensor = gain * power

    # Truncate between pixel values
    out: torch.Tensor = _clamp_unit_range(x_adjust)

    return out


def adjust_contrast(image: torch.Tensor, factor: Union[float, torch.Tensor], clip_output: bool = True) -> torch.Tensor:
    r"""Adjust the contrast of an image by multiplying its values by a factor.

    ``out = adjust_contrast(image, factor)`` multiplies an image in :math:`[0, 1]` by ``factor``: ``0`` gives black and
    ``1`` leaves it unchanged. To blend with the mean gray level instead, as PIL does, use
    :func:`adjust_contrast_with_mean_subtraction`.

    .. image:: _static/img/adjust_contrast.png

    Convention:
        Multiplicative, as in Szeliski's book and OpenCV's ``convertScaleAbs(alpha=factor)``.

    Args:
        image: image with shape ``(*, H, W)``.
        factor: non-negative multiplier, a float or a tensor with shape ``(B,)`` for one value per image.
        clip_output: clamp the result to :math:`[0, 1]`.

    Returns:
        Adjusted image with the shape of ``image``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    .. note::
       The non-negativity check on ``factor`` reads a CPU value during the call and raises
       :exc:`~kornia.core.exceptions.AsyncCheckError`, a :exc:`RuntimeError`. A tensor on CUDA, or on MPS
       from torch 2.13, is checked with ``torch._assert_async`` instead and raises ``RuntimeError`` by the
       next synchronization; on CUDA that is a device-side assert.

    See Also:
        - :func:`adjust_contrast_with_mean_subtraction`: blends with the mean gray level instead of scaling.
        - :func:`adjust_brightness_accumulative`: the same product, with a negative factor accepted.
        - :func:`adjust_gamma`: a power curve, which maps :math:`[0, 1]` onto itself at ``gain=1``.
        - :class:`AdjustContrast`: the module form.

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
    r"""Adjust the contrast of an image by blending it with its mean gray level, as PIL does.

    ``out = adjust_contrast_with_mean_subtraction(image, factor)`` returns ``factor * image + (1 - factor) * mean``,
    clamped to :math:`[0, 1]`, for an image in :math:`[0, 1]`. ``mean`` is the mean of the image's grayscale for 3
    channels and of all its values otherwise; ``0`` gives a uniform image and ``1`` the input.

    Args:
        image: image with shape ``(*, C, H, W)``.
        factor: weight of the image, a float or a tensor with shape ``(B,)`` for one value per image.

    Returns:
        Adjusted image with the shape of ``image``.

    See Also:
        - :func:`adjust_contrast`: multiplies the values, so the mean moves with them.
        - :func:`adjust_gamma`: a power curve, which maps :math:`[0, 1]` onto itself at ``gain=1``.
        - :class:`AdjustContrastWithMeanSubtraction`: the module form.

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

    # A 2-D image is one gray plane (#6645); only three or more dimensions have a channel axis.
    if image.ndim < 3:
        img_mean = image.mean((-2, -1), True)
    elif image.shape[-3] == 3:
        img_mean = rgb_to_grayscale(image).mean((-2, -1), True)
    else:
        img_mean = image.mean((-3, -2, -1), True)

    # Apply contrast factor subtracting the mean
    img_adjust: torch.Tensor = image * factor + img_mean * (1 - factor)

    return _clamp_unit_range(img_adjust)


def adjust_brightness(
    image: torch.Tensor, factor: Union[float, torch.Tensor], clip_output: bool = True
) -> torch.Tensor:
    r"""Adjust the brightness of an image by adding a factor to its values.

    ``out = adjust_brightness(image, factor)`` adds ``factor`` to every value of an image in :math:`[0, 1]`; ``0``
    leaves it unchanged. PIL multiplies instead, which is :func:`adjust_brightness_accumulative`.

    .. image:: _static/img/adjust_brightness.png

    Args:
        image: image with shape ``(*, H, W)``.
        factor: value to add, a float or a tensor with shape ``(B,)`` for one value per image.
        clip_output: clamp the result to :math:`[0, 1]`.

    Returns:
        Adjusted image with the shape of ``image``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/image_enhancement.html>`__.

    See Also:
        - :func:`adjust_brightness_accumulative`: multiplies instead of adding.
        - :func:`adjust_gamma`: a power curve, which maps :math:`[0, 1]` onto itself at ``gain=1``.
        - :func:`shift_rgb`: adds a different shift to each color channel.
        - :class:`AdjustBrightness`: the module form.

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
    r"""Multiply an image by a factor to brighten or darken it, as PIL's ``ImageEnhance.Brightness`` does.

    ``out = adjust_brightness_accumulative(image, factor)`` multiplies an image in :math:`[0, 1]` by ``factor``: ``0``
    gives black, ``1`` leaves it unchanged and a larger factor brightens it.

    Args:
        image: image with shape ``(*, H, W)``.
        factor: multiplier, a float or a tensor with shape ``(B,)`` for one value per image.
        clip_output: clamp the result to :math:`[0, 1]`.

    Returns:
        Adjusted image with the shape of ``image``.

    See Also:
        - :func:`adjust_brightness`: adds the factor.
        - :func:`adjust_contrast`: the same product, but a negative factor raises.
        - :class:`AdjustBrightnessAccumulative`: the module form.

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
        img_adjust = _clamp_unit_range(img_adjust)

    return img_adjust


def adjust_sigmoid(image: torch.Tensor, cutoff: float = 0.5, gain: float = 10, inv: bool = False) -> torch.Tensor:
    r"""Apply sigmoid correction around a cutoff to an image.

    ``out = adjust_sigmoid(image, cutoff, gain, inv)`` maps an image in :math:`[0, 1]` through the curve :math:`1 / (1 +
    e^{\text{gain}\,(\text{cutoff} - x)})`, which equals 0.5 at ``cutoff``. With ``inv=True`` it returns one minus that
    curve.

    Reference:
        [1]: Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions",
             http://markfairchild.org/PDFs/PAP07.pdf

    Args:
        image: image with shape ``(*, H, W)``.
        cutoff: input value at which the curve equals 0.5.
        gain: steepness of the curve.
        inv: return one minus the curve.

    Returns:
        Adjusted image with the shape of ``image``.

    See Also:
        - :class:`AdjustSigmoid`: the module form.
        - :func:`adjust_gamma`: a power curve.
        - :func:`adjust_log`: a logarithmic curve.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_sigmoid(x, gain=0)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])

    """
    _check_floating_image(image, "adjust_sigmoid")
    KORNIA_CHECK_IS_TENSOR(image, "Expected shape (*, H, W)")

    # Half exp can overflow even for ordinary image values. Tensor parameters keep their own promotion.
    promote = image.dtype == torch.float16 and isinstance(gain, (int, float)) and isinstance(cutoff, (int, float))
    if promote:
        image = image.float()
        bound = abs(gain) * max(abs(cutoff), abs(1.0 - cutoff))
        # On [0, 1], a bound of 16 keeps exp and the reciprocal cube in double backward normal in float32.
        # At larger bounds, evaluate sigmoid on its nonpositive side to avoid positive-side saturation in backward.
        if bound > 16.0:
            logits = gain * (image - cutoff)
            positive = logits >= 0
            tail = torch.where(positive, -logits, logits).sigmoid()
            result = torch.where(positive, tail, 1 - tail) if inv else torch.where(positive, 1 - tail, tail)
            return result.half()
    result = 1 - 1 / (1 + (gain * (cutoff - image)).exp()) if inv else 1 / (1 + (gain * (cutoff - image)).exp())
    return result.half() if promote else result


def adjust_log(image: torch.Tensor, gain: float = 1, inv: bool = False, clip_output: bool = True) -> torch.Tensor:
    r"""Apply logarithmic correction to an image, or exponential correction with ``inv``.

    ``out = adjust_log(image, gain, inv, clip_output)`` returns ``gain * log2(1 + image)`` for an image in
    :math:`[0, 1]`, or ``gain * (2 ** image - 1)`` with ``inv=True``.

    Reference:
        [1]: http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    Args:
        image: image with shape ``(*, H, W)``.
        gain: multiplier of the curve.
        inv: use the exponential curve.
        clip_output: clamp the result to :math:`[0, 1]`.

    Returns:
        Adjusted image with the shape of ``image``.

    See Also:
        - :class:`AdjustLog`: the module form.
        - :func:`adjust_gamma`: a power curve.
        - :func:`adjust_sigmoid`: an S-shaped curve around a cutoff.

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
        img_adjust = _clamp_unit_range(img_adjust)

    return img_adjust


def _solarize(input: torch.Tensor, thresholds: Union[float, torch.Tensor] = 0.5) -> torch.Tensor:
    r"""For each pixel in the image, select the pixel if the value is less than the threshold.

    Otherwise, replace it with ``1.0 - value``.

    Args:
        input: image or batched images to solarize.
        thresholds: solarize thresholds.
            If a float or one-element torch.Tensor, input will be solarized across the whole batch.
            If 1-d torch.Tensor, input will be solarized element-wise, len(thresholds) == len(input).

    Returns:
        Solarized images.

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, torch.Tensor)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. Got {type(thresholds)}")

    if isinstance(thresholds, torch.Tensor) and thresholds.numel() == 1:
        # One threshold serves the whole batch, as documented, whatever its shape (#6547). It is
        # moved and cast as the per-sample path below does, which a (1,) tensor with a batch of 1 took before.
        if len(thresholds.shape) != 0:
            thresholds = _to_device_nonblocking(thresholds, input.device).to(input.dtype)
        thresholds = thresholds.reshape(())
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
    r"""Invert the pixels at or above a threshold, optionally after adding an amount to every pixel.

    ``out = solarize(input, thresholds, additions)`` replaces every value of an image in :math:`[0, 1]` that is at or
    above ``thresholds`` by ``1 - value``. Without ``additions`` it equals PIL's ``ImageOps.solarize``.

    .. image:: _static/img/solarize.png

    Convention:
        ``additions`` is added to the whole image and the sum clamped to :math:`[0, 1]` before the comparison, so it can
        move a pixel across the threshold: with ``thresholds=0.5`` and ``additions=0.2``, ``[0.1, 0.4, 0.6, 0.9]`` comes
        back as ``[0.3, 0.4, 0.2, 0.0]``.

    Args:
        input: image with shape ``(*, C, H, W)``.
        thresholds: threshold in :math:`[0, 1]`, a float or a tensor. One value, as a float or a one-element tensor,
            applies to the whole batch, and a tensor with shape ``(B,)`` gives one value per image.
        additions: amount in :math:`[-0.5, 0.5]`, endpoints included, with the shapes of ``thresholds``. ``None`` adds
            nothing.

    Returns:
        The solarized image with shape ``(*, C, H, W)``.

    .. note::
       The range check on ``additions`` reads a float or a CPU tensor during the call and raises
       :exc:`~kornia.core.exceptions.AsyncCheckError`, a :exc:`RuntimeError`, whatever the device of the image.
       A tensor on CUDA, or on MPS from torch 2.13, is checked with ``torch._assert_async`` instead and raises
       ``RuntimeError`` by the next synchronization; on CUDA that is a device-side assert.

    See Also:
        - :class:`~kornia.augmentation.RandomSolarize`: draws ``thresholds`` and ``additions`` at random per image.
        - :func:`posterize`: reduces the bit depth instead of inverting.
        - :func:`invert`: inverts every pixel.

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

        if isinstance(additions, torch.Tensor) and additions.numel() == 1:
            # One addition serves the whole batch, as documented, whatever its shape (#6547). It is
            # moved and cast as the per-sample path below does, which a (1,) tensor with a batch of 1 took before.
            if len(additions.shape) != 0:
                additions = _to_device_nonblocking(additions, input.device).to(input.dtype)
            additions = additions.reshape(())
        if isinstance(additions, torch.Tensor) and len(additions.shape) != 0:
            if not (input.size(0) == len(additions) and len(additions.shape) == 1):
                raise AssertionError(f"additions must be a 1-d vector of shape ({input.size(0)},). Got {additions}")
            # Broadcast the per-sample scalar over (C, H, W) by reshaping instead of iterating the
            # tensor (`for x in additions` breaks the graph); numerically identical.
            additions = _to_device_nonblocking(additions, input.device).to(input.dtype)
            additions = additions.view([-1] + [1] * (input.ndim - 1))
        input = _clamp_unit_range(input + additions)

    return _solarize(input, thresholds)


def _prepare_intensity_input(input: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """Batch image or volume inputs without accepting empty channels or spatial dimensions."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    if 0 in input.shape[-(spatial_dims + 1) :]:
        raise ValueError(f"Input channel and spatial dimensions must be non-zero. Got {input.shape}.")
    return _to_bcdhw(input) if spatial_dims == 3 else _to_bchw(input)


def posterize(input: torch.Tensor, bits: Union[int, torch.Tensor]) -> torch.Tensor:
    r"""Keep only the highest ``bits`` of the 8 bits of each color level, as PIL's ``ImageOps.posterize`` does.

    ``out = posterize(input, bits)`` quantizes an image in :math:`[0, 1]` to its ``bits`` highest bits per value:
    ``bits=8`` returns the input and ``bits=0`` gives black.

    .. image:: _static/img/posterize.png

    Convention:
        - Each value is read at its 8-bit level ``(x * 255).long()``, with the 255 raised by four float32 epsilons,
          relative, in float32 and float64, so ``k / 255`` reads level ``k`` even when its division was rounded a unit
          low, as dividing by 255 on CUDA does.
        - The gradient is zero, except for ``bits=8``, which returns the input.
        - Empty leading dimensions are supported; channels and spatial dimensions must be non-empty.

    Args:
        input: image with shape ``(*, C, H, W)``.
        bits: number of high bits to keep, in ``[0, 8]``. An integer or a one-element tensor applies to the whole batch.
            A tensor with shape ``(B,)`` gives one value per image, one with shape ``(B, C)`` one per channel, and a
            longer prefix of ``(B, C, H, W)`` element-wise values, where ``B`` is the product of the leading dimensions
            of ``input``.

    Returns:
        Image with the shape of ``input``.

    Raises:
        AssertionError: if a ``bits`` tensor does not match the leading dimensions of ``input``.
        AsyncCheckError: if ``bits`` is outside ``[0, 8]``; it is a :exc:`ValueError` and a :exc:`RuntimeError`. A
            ``bits`` tensor on CUDA, or on MPS from torch 2.13, is checked with ``torch._assert_async`` instead and
            raises ``RuntimeError`` by the next synchronization.

    See Also:
        - :class:`~kornia.augmentation.RandomPosterize`: draws ``bits`` at random per image.
        - :func:`solarize`: inverts pixels above a threshold instead.

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
        in_range = torch.tensor(0 <= bits <= 8, device="cpu")
        bits = torch.as_tensor(bits, device=input.device)
    else:
        in_range = ((bits >= 0) & (bits <= 8)).all()
    _check_async(in_range, "bits must be in the closed range [0, 8].")

    # A user-supplied CPU bits tensor mixes with any device only when 0-dim (PyTorch's "wrapped
    # number" rule); multi-element CPU tensors hit a device mismatch on MPS/CUDA, so move bits
    # onto input's device explicitly (no-op when already there).
    bits = bits.to(device=input.device)

    # TODO: Make a differentiable version
    # Current version:
    # Ref: https://github.com/open-mmlab/mmcv/pull/132/files#diff-309c9320c7f71bedffe89a70ccff7f3bR19
    # Ref: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L222
    # Potential approach: implementing kornia.LUT with floating points
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py#L472
    # Both shifts read an 8-bit level by truncation; ``_level_scale`` keeps ``k / 255`` rounded a unit low at ``k``.
    scale = _level_scale(input.dtype)

    def _left_shift(input: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return ((input * scale).to(torch.uint8) * (2**shift)).to(input.dtype) / 255.0

    def _right_shift(input: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return (input * scale).to(torch.uint8) / (2**shift).to(input.dtype) / 255.0

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
    r"""Blend an image with a smoothed copy to sharpen or blur it, as PIL's ``ImageEnhance.Sharpness`` does.

    ``out = sharpness(input, factor)`` blends a ``(*, C, H, W)`` image in :math:`[0, 1]` with a copy smoothed by a 3x3
    kernel: ``0`` returns the smoothed image, ``1`` the input, and a factor above ``1`` sharpens. The result is clamped
    to :math:`[0, 1]`.

    .. image:: _static/img/sharpness.png

    Convention:
        - The smoothing kernel is :math:`\tfrac{1}{13}[[1, 1, 1], [1, 5, 1], [1, 1, 1]]`, and the one-pixel border keeps
          the input's values.
        - Empty leading dimensions are supported; channels must be non-empty and both spatial dimensions at least 3
          pixels.

    Args:
        input: image with shape ``(*, C, H, W)``.
        factor: blend weight of the input against the smoothed copy. A float or a 0-d tensor applies to the whole batch,
            and a tensor with shape ``(B,)`` gives one value per image, where ``B`` is the product of the leading
            dimensions of ``input``. A ``(1,)`` tensor applies to every image. Negative factors are accepted and
            extrapolate past the smoothed image.

    Returns:
        Sharpened image with the shape of ``input``.

    See Also:
        - :func:`~kornia.filters.unsharp_mask`: adds back the detail a Gaussian blur removes, with a fixed gain and no
          clamp.
        - :class:`~kornia.augmentation.RandomSharpness`: draws ``factor`` at random per image.

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

    if len(factor.size()) != 0 and factor.shape != torch.Size([input.size(0)]) and factor.shape != torch.Size([1]):
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
    degenerate = _clamp_unit_range(degenerate)

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
    return _clamp_unit_range(result + (input - result) * factor).reshape(input_shape)


def _scale_channel_batched(input: torch.Tensor) -> torch.Tensor:
    r"""Vectorized histogram equalization over the leading ``(B, C)`` planes.

    Computes all per-plane histograms/LUTs in a single batched pass instead of a Python loop of
    ``B * C`` serial calls. Each pixel is counted in the LUT entry it reads, its 8-bit level
    ``(x * _level_scale(x.dtype)).long()`` computed in the input dtype, so a plane whose LUT step is not 0 comes
    out as ``PIL.ImageOps.equalize`` of those levels while the counts and their running sums are exact (a plane
    of up to ``2**24`` pixels in float32); a plane whose step is 0 is returned as it is.

    Args:
        input: image tensor shaped ``(B, C, *spatial)`` with values in ``[0, 1]``.

    Returns:
        The equalized tensor, same shape as ``input``.
    """
    shape = input.shape
    n = shape[0] * shape[1]
    flat = input.flatten(0, 1).flatten(1)  # (N, P)
    # The level each pixel reads, before truncation. The factor is 255 raised by four float32 epsilons, relative, in
    # float32 and float64, so k / 255 rounded a unit low, as dividing by 255 on CUDA leaves 24 float64 levels,
    # still reads level k (see ``_level_scale``); half precision keeps 255.
    levels = flat * _level_scale(flat.dtype)

    # Input is expected in [0, 1]. The histogram and the LUT lookup both index with ``levels.long()``,
    # which is out of bounds unless ``levels`` is in (-1, 256).
    # Check that domain without ``.item()``, so there is no device sync and fullgraph still compiles;
    # on MPS before torch 2.13 it costs one host sync instead (see ``_check_async``). Inputs the
    # lookup can index keep working exactly as before. The check keeps the index in bounds, so it runs
    # with checks disabled too: a compiled CPU scatter with an index out of range terminates the process.
    _check_async(
        ((levels > -1.0) & (levels < 256.0)).all(),
        "equalize expects input values in [0, 1]. Scale the image into that range first, "
        "for example image / 255.0 for 8-bit data.",
        guard=True,
    )

    # Count in float32 (float64 stays float64): a count grows with the plane size, which half precision
    # cannot hold. float16 rounds counts above 2048 and their total overflows to inf past 65504, and
    # bfloat16 rounds counts above 256.
    count_dtype = _normalize_to_float32_or_float64(flat.dtype)
    # Per-plane 256-bin histogram on the index the lookup below reads, so every pixel is counted in its own
    # entry, as PIL counts and reads one 8-bit level. ``torch.histc(x, 256, 0, 255)`` bins in ``floor(256 x)``,
    # which puts every value in ``[k/256, k/255)``, about half of a continuous image, one bin above that entry.
    index = levels.long()
    histo = torch.zeros(n, 256, device=input.device, dtype=count_dtype)
    histo.scatter_add_(1, index, torch.ones_like(levels, dtype=count_dtype))

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
    lut = torch.clamp(lut, 0, 255).to(flat.dtype)

    # A plane whose step is 0 comes back as it is, not as (x * 255) / 255, which can be a unit off; it also keeps
    # ``x * 255`` from being one more image-sized temporary next to ``levels``.
    result = lut.gather(1, index) / 255.0
    return torch.where(step_col == 0, flat, result).reshape(shape)


def equalize(input: torch.Tensor) -> torch.Tensor:
    r"""Equalize the histogram of each channel of an image, as PIL's ``ImageOps.equalize`` does.

    ``out = equalize(input)`` equalizes each channel of each image in a ``(*, C, H, W)`` batch in :math:`[0, 1]` on its
    own.

    .. image:: _static/img/equalize.png

    Convention:
        - Each pixel is counted in the lookup-table entry it reads, its 8-bit level ``(x * 255).long()`` in the input
          dtype, so a float32 or float64 image gives ``PIL.ImageOps.equalize`` of those levels. In float32 and float64
          the 255 is raised by four float32 epsilons, relative (``2**-21``), so ``k / 255`` reads level ``k`` even when
          its division was rounded a unit low. Dividing by 255 on CUDA leaves 24 of the 256 float64 levels a unit low.
          A continuous value moves up a level only within that margin below a level boundary.
        - The lookup step is the pixel count minus the count at the highest occupied level, integer-divided by 255. A
          channel whose step is 0, which includes every channel of up to 255 pixels, is returned unchanged.
        - A float32 image counts in float32, which is exact while a channel holds at most ``2**24`` pixels (a 4096 by
          4096 image).
        - Empty leading dimensions are supported; channels and spatial dimensions must be non-empty.

    Args:
        input: image with shape ``(*, C, H, W)``.

    Returns:
        Equalized image with shape ``(*, C, H, W)``.

    .. note::
       An integer or boolean image raises :exc:`~kornia.core.exceptions.TypeCheckError` (a :exc:`TypeError`); convert
       8-bit data with ``image.float() / 255`` first. Values the 256-bin lookup cannot index, those whose level
       before truncation lies outside ``(-1, 256)``, raise a ``RuntimeError`` naming the range, during the call for a
       CPU image, even when checks are disabled, since the check keeps the lookup in bounds. On CUDA, and on MPS from
       torch ``2.13``, the check is ``torch._assert_async`` and raises by the next synchronization; on an older MPS
       release the condition is read on the host instead, one device sync per call, and is skipped there under
       ``torch.compile``.

    See Also:
        - :func:`equalize3d`: the volume twin, with one histogram per channel volume.
        - :func:`equalize_clahe`: equalizes each tile of a grid with a clipped histogram.
        - :class:`~kornia.augmentation.RandomEqualize`: applies it at random per image.

    Example:
        >>> x = torch.rand(1, 2, 3, 3)
        >>> equalize(x).shape
        torch.Size([1, 2, 3, 3])

    """
    _check_floating_image(input, "equalize")
    # Scales each channel independently, batched over (B, C) in a single pass.
    return _scale_channel_batched(_prepare_intensity_input(input, 2)).reshape(input.shape)


def equalize3d(input: torch.Tensor) -> torch.Tensor:
    r"""Equalize the histogram of each channel of a 3D volume.

    ``out = equalize3d(input)`` equalizes each channel of each volume in a ``(*, C, D, H, W)`` batch in :math:`[0, 1]`
    from one histogram over its whole ``(D, H, W)`` volume.

    Convention:
        - Each voxel is read at its 8-bit level as :func:`equalize` reads a pixel.
        - The lookup step and the unchanged-channel rule are those of :func:`equalize`, counted over the voxels of a
          channel volume.
        - Empty leading dimensions are supported; channels and spatial dimensions must be non-empty.

    Args:
        input: volume with shape ``(*, C, D, H, W)``.

    Returns:
        Equalized volume with the shape of ``input``.

    .. note::
       An integer or boolean volume raises :exc:`~kornia.core.exceptions.TypeCheckError`, and a value whose level
       before truncation lies outside ``(-1, 256)`` raises a ``RuntimeError``, as for :func:`equalize`, with the same
       device behavior.

    See Also:
        - :func:`equalize`: the image twin, with one histogram per channel plane.

    """
    _check_floating_image(input, "equalize3d")
    # Scales each channel independently (each (D, H, W) volume), batched over (B, C).
    return _scale_channel_batched(_prepare_intensity_input(input, 3)).reshape(input.shape)


def invert(image: torch.Tensor, max_val: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Invert an image by subtracting it from its maximum value.

    ``out = invert(image, max_val)`` returns ``max_val - image``.

    .. image:: _static/img/invert.png

    Args:
        image: tensor of any shape.
        max_val: tensor the image is subtracted from, broadcastable to ``image``. The default is ``1``, so for a
            ``uint8`` image pass ``torch.tensor(255)``.

    Returns:
        Inverted tensor with the shape of ``image``.

    See Also:
        - :class:`Invert`: the module form.
        - :func:`solarize`: inverts only the pixels at or above a threshold.
        - :class:`~kornia.augmentation.RandomInvert`: inverts at random per image.

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
        raise AssertionError(f"Input is not a torch.Tensor. Got: {type(image)}")

    _max_val = torch.tensor(1.0) if max_val is None else max_val
    if not isinstance(_max_val, torch.Tensor):
        raise AssertionError(f"max_val is not a torch.Tensor. Got: {type(_max_val)}")

    return _max_val.to(image) - image


class AdjustSaturation(nn.Module):
    r"""Adjust the saturation of an RGB image by scaling its HSV saturation (module form).

    ``adjust = AdjustSaturation(saturation_factor)`` stores the factor, and ``out = adjust(image)`` is
    :func:`adjust_saturation` with it.

    Args:
        saturation_factor: saturation multiplier, a float or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, 3, H, W)``
        - Output: ``(*, 3, H, W)``

    See Also:
        - :func:`adjust_saturation`: the one-call form.
        - :class:`AdjustSaturationWithGraySubtraction`: blends with the gray image instead of scaling the HSV
          saturation.

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
            input: Input RGB tensor, typically shaped ``(*, 3, H, W)``.

        Returns:
            Saturation-adjusted tensor with the same shape as ``input``.
        """
        return adjust_saturation(input, self.saturation_factor)


class AdjustSaturationWithGraySubtraction(nn.Module):
    r"""Adjust the saturation of an RGB image by blending it with its grayscale version (module form).

    ``adjust = AdjustSaturationWithGraySubtraction(saturation_factor)`` stores the factor, and ``out = adjust(image)``
    is :func:`adjust_saturation_with_gray_subtraction` with it.

    Args:
        saturation_factor: weight of the image, a float or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, 3, H, W)`` or ``(*, 1, H, W)``
        - Output: same as the input

    See Also:
        - :func:`adjust_saturation_with_gray_subtraction`: the one-call form.
        - :class:`AdjustSaturation`: scales the HSV saturation instead.

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
            input: Input image tensor, typically shaped ``(*, 3, H, W)``.

        Returns:
            Saturation-adjusted tensor with the same shape as ``input``.
        """
        return adjust_saturation_with_gray_subtraction(input, self.saturation_factor)


class AdjustHue(nn.Module):
    r"""Shift the hue of an RGB image by an angle in radians (module form).

    ``adjust = AdjustHue(hue_factor)`` stores the angle, and ``out = adjust(image)`` is :func:`adjust_hue` with it.

    Convention:
        - See the convention block of :func:`adjust_hue`.

    Args:
        hue_factor: hue shift in radians, a number or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, 3, H, W)``
        - Output: ``(*, 3, H, W)``

    See Also:
        - :func:`adjust_hue`: the one-call form.

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
                ``(*, 3, H, W)``.

        Returns:
            Hue-adjusted tensor with the same shape as ``input``.
        """
        return adjust_hue(input, self.hue_factor)


class AdjustGamma(nn.Module):
    r"""Apply gamma correction with a stored exponent and gain (module form).

    ``adjust = AdjustGamma(gamma, gain)`` stores both, and ``out = adjust(image)`` is :func:`adjust_gamma` with them.

    Convention:
        - See the convention block of :func:`adjust_gamma`.

    Args:
        gamma: non-negative exponent, a float or a tensor with shape ``(B,)``.
        gain: non-negative multiplier, a float or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, H, W)``
        - Output: ``(*, H, W)``

    See Also:
        - :func:`adjust_gamma`: the one-call form.

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
                ``(*, C, H, W)``.

        Returns:
            Gamma-corrected tensor with the same shape as ``input``.
        """
        return adjust_gamma(input, self.gamma, self.gain)


class AdjustContrast(nn.Module):
    r"""Adjust the contrast of an image by multiplying its values by a stored factor (module form).

    ``adjust = AdjustContrast(contrast_factor)`` stores the factor, and ``out = adjust(image)`` is
    :func:`adjust_contrast` with it and ``clip_output=True``.

    Convention:
        - See the convention block of :func:`adjust_contrast`.

    Args:
        contrast_factor: non-negative multiplier, a float or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, H, W)``
        - Output: ``(*, H, W)``

    See Also:
        - :func:`adjust_contrast`: the one-call form, with ``clip_output``.
        - :class:`AdjustContrastWithMeanSubtraction`: blends with the mean gray level instead of scaling.

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
            input: Input tensor to adjust, typically shaped ``(*, C, H, W)``.

        Returns:
            Contrast-adjusted tensor with the same shape as ``input``.
        """
        return adjust_contrast(input, self.contrast_factor)


class AdjustContrastWithMeanSubtraction(nn.Module):
    r"""Adjust the contrast of an image by blending it with its mean gray level (module form).

    ``adjust = AdjustContrastWithMeanSubtraction(contrast_factor)`` stores the factor, and ``out = adjust(image)`` is
    :func:`adjust_contrast_with_mean_subtraction` with it.

    Args:
        contrast_factor: weight of the image, a float or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, C, H, W)``
        - Output: ``(*, C, H, W)``

    See Also:
        - :func:`adjust_contrast_with_mean_subtraction`: the one-call form.
        - :class:`AdjustContrast`: multiplies the values instead.

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
            input: Input tensor to adjust, typically shaped ``(*, C, H, W)``.

        Returns:
            Contrast-adjusted tensor with the same shape as ``input``.
        """
        return adjust_contrast_with_mean_subtraction(input, self.contrast_factor)


class AdjustBrightness(nn.Module):
    r"""Adjust the brightness of an image by adding a stored factor (module form).

    ``adjust = AdjustBrightness(brightness_factor)`` stores the factor, and ``out = adjust(image)`` is
    :func:`adjust_brightness` with it and ``clip_output=True``.

    Args:
        brightness_factor: value to add, a float or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, H, W)``
        - Output: ``(*, H, W)``

    See Also:
        - :func:`adjust_brightness`: the one-call form, with ``clip_output``.
        - :class:`AdjustBrightnessAccumulative`: multiplies instead of adding.

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
            input: Input tensor to adjust, commonly shaped ``(*, C, H, W)``.

        Returns:
            Brightness-adjusted tensor with the same shape as ``input``.
        """
        return adjust_brightness(input, self.brightness_factor)


class AdjustSigmoid(nn.Module):
    r"""Apply sigmoid correction with a stored cutoff and gain (module form).

    ``adjust = AdjustSigmoid(cutoff, gain, inv)`` stores its arguments, and ``out = adjust(image)`` is
    :func:`adjust_sigmoid` with them.

    Reference:
        [1]: Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions",
             http://markfairchild.org/PDFs/PAP07.pdf

    Args:
        cutoff: input value at which the curve equals 0.5.
        gain: steepness of the curve.
        inv: return one minus the curve.

    Shape:
        - Input: ``(*, H, W)``
        - Output: ``(*, H, W)``

    See Also:
        - :func:`adjust_sigmoid`: the one-call form.

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
                ``(*, C, H, W)`` or ``(*, H, W)``.

        Returns:
            Tensor with sigmoid correction applied, preserving input shape.
        """
        return adjust_sigmoid(image, cutoff=self.cutoff, gain=self.gain, inv=self.inv)


class AdjustLog(nn.Module):
    r"""Apply logarithmic correction with a stored gain (module form).

    ``adjust = AdjustLog(gain, inv, clip_output)`` stores its arguments, and ``out = adjust(image)`` is
    :func:`adjust_log` with them.

    Reference:
        [1]: http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    Args:
        gain: multiplier of the curve.
        inv: use the exponential curve.
        clip_output: clamp the result to :math:`[0, 1]`.

    Shape:
        - Input: ``(*, H, W)``
        - Output: ``(*, H, W)``

    See Also:
        - :func:`adjust_log`: the one-call form.

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
                ``(*, C, H, W)`` or ``(*, H, W)``.

        Returns:
            Log-corrected tensor with the same shape as ``image``.
        """
        return adjust_log(image, gain=self.gain, inv=self.inv, clip_output=self.clip_output)


class AdjustBrightnessAccumulative(nn.Module):
    r"""Multiply an image by a stored factor to brighten or darken it (module form).

    ``adjust = AdjustBrightnessAccumulative(brightness_factor)`` stores the factor, and ``out = adjust(image)`` is
    :func:`adjust_brightness_accumulative` with it and ``clip_output=True``.

    Args:
        brightness_factor: multiplier, a float or a tensor with shape ``(B,)``.

    Shape:
        - Input: ``(*, H, W)``
        - Output: ``(*, H, W)``

    See Also:
        - :func:`adjust_brightness_accumulative`: the one-call form, with ``clip_output``.
        - :class:`AdjustBrightness`: adds the factor instead.

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
            input: Input tensor to adjust, commonly shaped ``(*, C, H, W)``.

        Returns:
            Brightness-adjusted tensor with the same shape as ``input``.
        """
        return adjust_brightness_accumulative(input, self.brightness_factor)


class Invert(nn.Module):
    r"""Invert an image by subtracting it from a stored maximum value (module form).

    ``op = Invert(max_val)`` stores ``max_val`` as a buffer, or as a parameter if it is an ``nn.Parameter``, and
    ``out = op(image)`` is :func:`invert` with it.

    Args:
        max_val: tensor the image is subtracted from, broadcastable to the input. The default is ``1``, so for a
            ``uint8`` image pass ``torch.tensor(255)``.

    Shape:
        - Input: any shape
        - Output: same as the input

    See Also:
        - :func:`invert`: the one-call form.

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
