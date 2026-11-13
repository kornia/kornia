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

import math
import warnings
from collections.abc import Callable
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from kornia.color import rgb_to_grayscale
from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_COLOR_OR_GRAY,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SHAPE,
)
from kornia.core.utils import _autocast_disabled, is_exporting

from .filter import _to_floating
from .gaussian import gaussian_blur2d
from .sobel import spatial_gradient


def _check_thresholds(low_threshold: float, high_threshold: float) -> None:
    # The thresholds are in the units of the unnormalised Sobel magnitude, which has no upper bound.
    KORNIA_CHECK(
        low_threshold <= high_threshold,
        "Invalid input thresholds. low_threshold should be smaller than or equal to the high_threshold. Got: "
        f"{low_threshold}>{high_threshold}",
    )
    KORNIA_CHECK(0 < low_threshold, f"Invalid low threshold. Should be positive. Got: {low_threshold}")


def _hysteresis_step(strong: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    # One round of edge tracking: a weak pixel (low & ~strong) with a strong (8-connected) neighbour becomes strong.
    # The 8-connected dilation is separable: a 1D horizontal pass followed by a 1D vertical pass.
    p = F.pad(strong, (1, 1, 1, 1))
    h = p[..., 1:-1] | p[..., :-2] | p[..., 2:]
    return (h[..., 1:-1, :] | h[..., :-2, :] | h[..., 2:, :]) & low


def _hysteresis_loop(strong: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    # Repeat the round until nothing changes, for torch.jit.trace: written in the subset of Python that TorchScript
    # compiles, so _hysteresis_loop_traced can hand a trace a scripted copy. canny keeps its eager loop inline: under
    # torch.compile the loop condition breaks the graph, and a break inside a called function splits canny once more.
    cur = strong
    while True:
        prev = cur
        cur = _hysteresis_step(cur, low)
        if not bool((cur != prev).any()):
            break
    return cur


@lru_cache(maxsize=1)
def _scripted_hysteresis_loop() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    # Compiled on the first trace, not at import, so only a trace pays for TorchScript.
    with warnings.catch_warnings():
        # torch 2.14 emits a FutureWarning that torch.jit.script is deprecated. The caller never called it
        # and torch.jit.trace already warns for them, so this would be noise.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        return torch.jit.script(_hysteresis_loop)


def _hysteresis_loop_traced(strong: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    # torch.jit.trace unrolls a data-dependent loop to the rounds the traced image needed. A scripted
    # function called from a trace keeps its control flow (and becomes an ONNX Loop).
    try:
        scripted_loop = _scripted_hysteresis_loop()
    except Exception as error:  # noqa: BLE001 - TorchScript may be absent, or unable to read the source
        warnings.warn(
            "canny with hysteresis=True could not compile its hysteresis loop with TorchScript, so the traced "
            "graph repeats the hysteresis round as many times as the traced input needed, on every input: a "
            "longer edge chain is cut short without an error. Export with torch.export or the dynamo ONNX "
            f"exporter instead. torch.jit.script raised {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=3,
        )
        return _hysteresis_loop(strong, low)
    return scripted_loop(strong, low)


def canny(
    input: torch.Tensor,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    kernel_size: tuple[int, int] | int = (5, 5),
    sigma: tuple[float, float] | torch.Tensor = (1, 1),
    hysteresis: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Find the edges of an image with the Canny detector.

    ``magnitude, edges = canny(input)`` blurs a grayscale or RGB image, thins the Sobel gradient magnitude by
    non-maximum suppression, and thresholds it with hysteresis into an edge map.

    .. image:: _static/img/canny.png

    Convention:
        - A 3-channel input is converted with :func:`~kornia.color.rgb_to_grayscale`, which reads channel 0 as
          red. Both outputs are ``(B, 1, H, W)``, in the input's dtype for a floating input and in float32 for
          an integer one, which is processed in float32 on its own scale.
        - The image is blurred with ``gaussian_blur2d(input, kernel_size, sigma)``; the Convention block on
          :func:`~kornia.filters.gaussian_blur2d` gives the order of both pairs, and ``kernel_size=1`` skips the
          blur. The magnitude is :math:`\sqrt{g_x^2 + g_y^2 + \epsilon}` of the **unnormalised** Sobel gradient
          ``spatial_gradient(blurred, normalized=False)``: an axis-aligned ramp of signed slope ``s`` gives
          :math:`\sqrt{(8 s)^2 + \epsilon}` at an interior pixel, about ``8 * abs(s)``. This is about eight times
          what :func:`~kornia.filters.sobel` returns by default, up to the ``eps`` inside the square root. On an
          image in :math:`[0, 1]` it is not bounded by 1: a unit step reaches 4 without the blur and about 2.59 with
          the default one. Float16 and bfloat16 inputs are computed in float32, with enclosing autocast disabled;
          only the final magnitude and edges are rounded to the input dtype.
        - The thresholds compare against that magnitude and are strict: a pixel is weak above ``low_threshold`` and
          strong above ``high_threshold``. :ref:`Filtering <filtering-conventions>` maps them onto OpenCV and
          scikit-image.
        - Non-maximum suppression compares each pixel with its two neighbours along the gradient direction,
          rounded to a multiple of 45 degrees. As in OpenCV's ``cv2.Canny``, along a horizontal (vertical) gradient
          a pixel must be strictly greater than its left (upper) neighbour and greater than or equal to its right
          (lower) one, so of two pixels of equal magnitude across a step edge the left (upper) one is kept. Along a
          diagonal it must be strictly greater than both. A pixel with zero gradient is never kept. The returned
          magnitude is taken after this step, so it is zero off the edge ridges; it is differentiable with respect
          to the input, and the edge map is not.
        - Hysteresis keeps a weak pixel connected to a strong one through any of its 8 neighbours, repeated until
          nothing changes, and returns edges of 0 and 1. Under ``torch.export`` the loop is a ``torch.while_loop``,
          so the exported graph also repeats until nothing changes, for any input; the dynamo ONNX exporter writes
          it as an ONNX ``Loop`` from torch 2.11 and cannot export it before.
          Under ``torch.jit.trace``, and so the legacy TorchScript ONNX exporter (``torch.onnx.export(...,
          dynamo=False)``), the loop is a scripted function that the trace calls, so the traced graph also repeats
          until nothing changes, for any input, and the exporter writes it as an ONNX ``Loop``. A trace cannot
          record a loop whose condition depends on the data on its own: if TorchScript cannot compile the function,
          a ``RuntimeWarning`` says so and the traced graph repeats the rounds the traced image needed on every input,
          which cuts a longer edge chain short without an error.
        - An integer image is promoted to float32 before the grayscale conversion, so a 3-channel integer image
          is converted with the floating weights rather than rounded to an integer grey.

    Args:
        input: image with shape ``(B, C, H, W)``, with ``C`` equal to 1, or to 3 for an RGB image, which is
            converted to grayscale.
        low_threshold: lower threshold for the hysteresis procedure, in the units of the magnitude above. It must be
            positive and at most ``high_threshold``.
        high_threshold: upper threshold for the hysteresis procedure, in the same units. It has no upper bound.
        kernel_size: Gaussian blur window ``(kH, kW)`` in pixels, or one integer for both.
        sigma: Gaussian blur standard deviation ``(sigma_y, sigma_x)`` in pixels, as in
            :func:`~kornia.filters.gaussian_blur2d`.
        hysteresis: if ``True``, keep a weak edge only when it connects to a strong one. If ``False``, the edge map
            holds 0, 0.5 for weak edges and 1 for strong edges.
        eps: added under the square root of the magnitude so that the gradient of a flat pixel is finite.

    Returns:
        - the gradient magnitude after non-maximum suppression, with shape ``(B, 1, H, W)``.
        - the edge map, with shape ``(B, 1, H, W)``, holding 0 and 1 with ``hysteresis=True``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/canny.html>`__.

    See Also:
        - :class:`Canny`: the module form.
        - :func:`sobel`: the gradient magnitude alone, without blur, suppression or thresholds.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = canny(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK_IS_COLOR_OR_GRAY(input, "canny expects 1 or 3 channels.")
    _check_thresholds(low_threshold, high_threshold)

    # an integer image is processed in float32, so both outputs are float32 (#5155); promoting before the grayscale
    # conversion keeps the fractional grey a uint8 conversion would round away
    input = _to_floating(input)

    output_dtype = input.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        input = input.float()
        if torch.jit.is_scripting():
            gradients = _canny_gradients(input, kernel_size, sigma)
        else:
            with _autocast_disabled(input.device.type):
                gradients = _canny_gradients(input, kernel_size, sigma)
    else:
        gradients = _canny_gradients(input, kernel_size, sigma)
    dtype = input.dtype

    # Unpack the edges
    gx: torch.Tensor = gradients[:, :, 0]
    gy: torch.Tensor = gradients[:, :, 1]

    # Compute gradient magnitude. The magnitude is computed one pixel beyond the image, where there is no
    # gradient: the border then has the magnitude sqrt(eps) of a flat pixel, so a flat pixel never beats it.
    gx_padded: torch.Tensor = F.pad(gx, (1, 1, 1, 1))
    gy_padded: torch.Tensor = F.pad(gy, (1, 1, 1, 1))
    padded: torch.Tensor = torch.sqrt(gx_padded * gx_padded + gy_padded * gy_padded + eps)
    magnitude: torch.Tensor = padded[..., 1:-1, 1:-1]

    # Preserve the legacy rounded-atan2 sectors: direct tangent comparisons differ at representable values
    # around 22.5/67.5 degrees. Keep the sliced neighbour comparisons below to avoid the old 9-channel window.
    # The legacy ONNX exporter produces NaN for atan2(0, 0); that flat pixel belongs to sector zero.
    angle = torch.nan_to_num(torch.atan2(gy, gx))
    sector = (angle * (4 / math.pi)).round().remainder(4)
    is_0 = sector == 0
    is_2 = sector == 2
    is_diag = ~(is_0 | is_2)
    same_sign = sector == 1

    # As in OpenCV's Canny, a pixel must be strictly greater than its left (upper) neighbour but only greater than or
    # equal to its right (lower) one along a horizontal (vertical) gradient, so of two equal pixels across a step the
    # left (upper) one is kept. Along a diagonal both comparisons are strict: the neighbours along it are two lines
    # apart, and the line between a tied pair carries the edge.
    max_0 = is_0 & (magnitude > padded[..., 1:-1, :-2]) & (magnitude >= padded[..., 1:-1, 2:])
    max_2 = is_2 & (magnitude > padded[..., :-2, 1:-1]) & (magnitude >= padded[..., 2:, 1:-1])
    max_1 = is_diag & same_sign & (magnitude > padded[..., :-2, :-2]) & (magnitude > padded[..., 2:, 2:])
    max_3 = is_diag & (~same_sign) & (magnitude > padded[..., :-2, 2:]) & (magnitude > padded[..., 2:, :-2])
    is_max = max_0 | max_2 | max_1 | max_3

    # A pixel no larger than the border, which has no gradient, is never kept. The comparisons above already exclude it,
    # since one of them is strict for every direction; this states the rule independently of the angle.
    is_max = is_max & (magnitude > padded[..., :1, :1])

    magnitude = magnitude * is_max

    # Threshold
    low: torch.Tensor = magnitude > low_threshold
    high: torch.Tensor = magnitude > high_threshold

    # Hysteresis
    if hysteresis:
        if is_exporting():
            # Graph capture cannot branch on the data, so the loop is a while_loop, which the dynamo ONNX exporter
            # writes as a Loop node from torch 2.11. The body must not return its input, hence the clone.
            def _changed(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
                return (old != new).any()

            def _step(old: torch.Tensor, new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return new.clone(), _hysteresis_step(new, low)

            _, strong = torch.while_loop(_changed, _step, (~high, high))
        elif torch.jit.is_tracing():
            strong = _hysteresis_loop_traced(high, low)
        else:
            # A tensor loop condition, not `if not bool(...): break`: Dynamo resumes after a graph break on the
            # condition, while a break inside the loop body makes it run the whole of canny eagerly.
            # The loop carries uint8, not bool. Inductor's CPU code (torch 2.14.0 and 2.15.0.dev20260917) reads a bool
            # input of the step's pad with an unmasked vector load starting one pixel before the image (and ending one
            # after): SIGBUS when the mask borders an unmapped page, as large allocations on macOS can. A uint8 input
            # gets a masked load. Casting inside the step is not enough, since Inductor folds the cast into the bool
            # load; the TorchScript, trace and export paths keep bool, as the legacy ONNX exporter rejects a uint8 OR.
            # Retire once the supported torch range has a fixed Inductor; test_dynamo_hysteresis_reads_inside_the_image
            # pins it. Cost on an M1 Pro, 1 or 32 x 256 x 256: compiled CPU canny +18% / +44% (the masked load), eager
            # CPU and MPS and compiled MPS within 2%; compiled CPU is still faster than 0.9.0's convolution.
            low = low.to(torch.uint8)
            strong = high.to(torch.uint8)
            strong_old = strong ^ 1
            while (strong_old != strong).any():
                strong_old = strong
                strong = _hysteresis_step(strong, low)

        edges = strong.to(dtype)
    else:
        # Cast before scaling: a boolean times a Python float exports as aten.mul.Scalar, which onnxscript < 0.7.2
        # cannot translate. The result is 0, 0.5 or 1, exact in every floating dtype.
        edges = ((low.to(dtype) + high.to(dtype)) * 0.5).to(dtype)

    return magnitude.to(output_dtype), edges.to(output_dtype)


def _canny_gradients(
    input: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | torch.Tensor,
) -> torch.Tensor:
    """Convert to grayscale, blur and differentiate in the working dtype."""
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)
    blurred = gaussian_blur2d(input, kernel_size, sigma)
    return spatial_gradient(blurred, normalized=False)


class Canny(nn.Module):
    r"""Find the edges of an image with stored Canny settings (module form).

    ``detector = Canny(low_threshold, high_threshold, ...)`` stores its arguments and checks the thresholds, and
    ``magnitude, edges = detector(input)`` is :func:`canny` on a ``(B, C, H, W)`` image.

    Convention:
        See the Convention block on :func:`~kornia.filters.canny`.

    Args:
        low_threshold: lower hysteresis threshold, in the units of the unnormalised Sobel magnitude. It must be
            positive and at most ``high_threshold``.
        high_threshold: upper hysteresis threshold, in the same units. It has no upper bound.
        kernel_size: Gaussian blur window ``(kH, kW)`` in pixels, or one integer for both.
        sigma: Gaussian blur standard deviation ``(sigma_y, sigma_x)`` in pixels.
        hysteresis: whether to keep a weak edge only when it connects to a strong one, as in :func:`canny`.
        eps: added under the square root of the magnitude, as in :func:`canny`.

    Shape:
        - Input: ``(B, C, H, W)``, with ``C`` equal to 1, or to 3 for an RGB image.
        - Output: the magnitude and the edge map, each ``(B, 1, H, W)``.

    See Also:
        - :func:`canny`: the one-call form.
        - :class:`Sobel`: the gradient magnitude alone.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])

    """

    # The dynamo ONNX exporter writes the hysteresis loop as an ONNX Loop from torch 2.11.
    ONNX_EXPORTABLE = True

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: tuple[int, int] | int = (5, 5),
        sigma: tuple[float, float] | torch.Tensor = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        _check_thresholds(low_threshold, high_threshold)

        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Double threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Hysteresis
        self.hysteresis = hysteresis

        self.eps: float = eps

    def __repr__(self) -> str:
        return "".join(
            (
                f"{type(self).__name__}(",
                ", ".join(
                    f"{name}={getattr(self, name)}" for name in sorted(self.__dict__) if not name.startswith("_")
                ),
                ")",
            )
        )

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect image edges with the Canny pipeline.

        The Canny detector smooths the image, estimates spatial gradients,
        performs non-maximum suppression, and thresholds candidate edges using
        the configured low and high thresholds. When hysteresis is enabled,
        weak edge pixels are kept only when they are connected to strong edge
        pixels.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the image height, and :math:`W` is the
                image width. :math:`C` must be 1, or 3 for an RGB image, which is
                converted to grayscale.

        Returns:
            Tuple ``(magnitude, edges)``. ``magnitude`` contains the gradient
            magnitude after non-maximum suppression, and ``edges`` contains the final
            thresholded edge map. Both tensors follow the layout returned by
            :func:`canny` and keep the same batch and spatial dimensions as the
            input.
        """
        return canny(
            input, self.low_threshold, self.high_threshold, self.kernel_size, self.sigma, self.hysteresis, self.eps
        )
