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
from kornia.core.utils import is_exporting

from .gaussian import gaussian_blur2d
from .kernels import get_hysteresis_kernel
from .sobel import spatial_gradient


def _check_thresholds(low_threshold: float, high_threshold: float) -> None:
    # The thresholds are in the units of the unnormalised Sobel magnitude, which has no upper bound.
    KORNIA_CHECK(
        low_threshold <= high_threshold,
        "Invalid input thresholds. low_threshold should be smaller than or equal to the high_threshold. Got: "
        f"{low_threshold}>{high_threshold}",
    )
    KORNIA_CHECK(0 < low_threshold, f"Invalid low threshold. Should be positive. Got: {low_threshold}")


# (dy, dx) of the neighbour in direction k = 0..7, counted from +x (right) towards +y (down) in steps of 45 degrees,
# followed by the pixel itself at index 8.
_WINDOW_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0))


def _beats(magnitude: torch.Tensor, neighbour: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    # Directions 0 (right) and 2 (down) accept a tie; the other six need a strictly larger magnitude.
    accepts_tie = (direction == 0) | (direction == 2)
    return (magnitude > neighbour) | (accepts_tie & (magnitude == neighbour))


def _hysteresis_step(edges: torch.Tensor, hysteresis_kernels: torch.Tensor) -> torch.Tensor:
    # One round of edge tracking: a weak pixel (0.5) with a strong (1) 8-neighbour becomes strong.
    weak = (edges == 0.5).to(edges.dtype)
    strong = (edges == 1).to(edges.dtype)
    hysteresis_magnitude = F.conv2d(edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2)
    hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(edges.dtype)
    hysteresis_magnitude = hysteresis_magnitude * weak + strong
    return hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5


def _hysteresis_loop(edges: torch.Tensor, hysteresis_kernels: torch.Tensor) -> torch.Tensor:
    # Repeat the round until nothing changes. Written in the subset of Python that TorchScript compiles;
    # eager mode runs it as is, and _hysteresis_loop_traced hands a trace a scripted copy.
    edges_old = -torch.ones_like(edges)
    while bool((edges_old - edges).abs().ne(0).any()):
        edges_old = edges
        edges = _hysteresis_step(edges, hysteresis_kernels)
    return edges


@lru_cache(maxsize=1)
def _scripted_hysteresis_loop() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    # Compiled on the first trace, not at import, so only a trace pays for TorchScript.
    with warnings.catch_warnings():
        # torch 2.14 emits a FutureWarning that torch.jit.script is deprecated. The caller never called it
        # and torch.jit.trace already warns for them, so this would be noise.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        return torch.jit.script(_hysteresis_loop)


def _hysteresis_loop_traced(edges: torch.Tensor, hysteresis_kernels: torch.Tensor) -> torch.Tensor:
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
        return _hysteresis_loop(edges, hysteresis_kernels)
    return scripted_loop(edges, hysteresis_kernels)


def canny(
    input: torch.Tensor,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    kernel_size: tuple[int, int] | int = (5, 5),
    sigma: tuple[float, float] | torch.Tensor = (1, 1),
    hysteresis: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Find edges of the input image and filters them using the Canny algorithm.

    .. image:: _static/img/canny.png

    Convention:
        - A 3-channel input is converted with :func:`~kornia.color.rgb_to_grayscale`, which reads channel 0 as
          red. For a floating input both outputs are :math:`(B, 1, H, W)` in the input's dtype.
        - The image is blurred with ``gaussian_blur2d(input, kernel_size, sigma)``; the Convention block on
          :func:`~kornia.filters.gaussian_blur2d` gives the order of both pairs, and ``kernel_size=1`` skips the
          blur. The magnitude is :math:`\sqrt{g_x^2 + g_y^2 + \epsilon}` of the **unnormalised** Sobel gradient
          ``spatial_gradient(blurred, normalized=False)``: an axis-aligned ramp of signed slope ``s`` gives
          :math:`\sqrt{(8 s)^2 + \epsilon}` at an interior pixel, about ``8 * abs(s)``. This is about eight times
          what :func:`~kornia.filters.sobel` returns by default, up to the ``eps`` inside the square root. On an
          image in :math:`[0, 1]` it is not bounded by 1: a unit step reaches 4 without the blur and about 2.59 with
          the default one. In float16 the squared gradient overflows past 65504, which a step of 64 already reaches
          without the blur, so keep a float16 image in :math:`[0, 1]` rather than scaling it and the thresholds up.
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
        - Known defect: an integer input is not converted to a floating dtype. With the default blur a 1-channel
          signed integer image blurs to zeros and yields no edge; a uint8 image, a 3-channel integer image, or any
          integer image where torch has no integer convolution raises
          (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        input: input image torch.Tensor with shape :math:`(B,C,H,W)`, with :math:`C` equal to 1, or to 3 for an RGB
            image, which is converted to grayscale.
        low_threshold: lower threshold for the hysteresis procedure, in the units of the magnitude above. It must be
            positive and at most ``high_threshold``.
        high_threshold: upper threshold for the hysteresis procedure, in the same units. It has no upper bound.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the gradient magnitude after non-maximum suppression, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/canny.html>`__.

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
    KORNIA_CHECK_IS_COLOR_OR_GRAY(input, f"canny expects 1 or 3 channels. Got: {input.shape[1]}")
    _check_thresholds(low_threshold, high_threshold)

    device = input.device
    dtype = input.dtype

    # To Grayscale
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)

    # Gaussian filter
    blurred: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)

    # Compute the gradients
    gradients: torch.Tensor = spatial_gradient(blurred, normalized=False)

    # Unpack the edges
    gx: torch.Tensor = gradients[:, :, 0]
    gy: torch.Tensor = gradients[:, :, 1]

    # Compute gradient magnitude and angle. The magnitude is computed one pixel beyond the image, where there is no
    # gradient: the border then has the magnitude sqrt(eps) of a flat pixel, so a flat pixel never beats it.
    gx_padded: torch.Tensor = F.pad(gx, (1, 1, 1, 1))
    gy_padded: torch.Tensor = F.pad(gy, (1, 1, 1, 1))
    padded: torch.Tensor = torch.sqrt(gx_padded * gx_padded + gy_padded * gy_padded + eps)
    # The legacy ONNX exporter's atan2 returns NaN at (0, 0), which casts to an out-of-range index on x86-64.
    angle: torch.Tensor = torch.nan_to_num(torch.atan2(gy, gx))

    # Radians to degrees and round to nearest 45 degree
    # degrees = angle * (180.0 / math.pi)
    # angle = torch.round(degrees / 45) * 45
    angle_45 = (angle * (4 / math.pi)).round()

    # Non-maximal suppression: entry k < 8 along dimension 2 of window is the magnitude of the neighbour in direction k
    # and entry 8 that of the pixel itself. Slicing copies the values, so a tie compares exactly equal, which a
    # convolution does not guarantee, and reading the pixel and its neighbours from one tensor keeps them in the same
    # precision under torch.compile.
    height, width = gx.shape[-2:]
    window: torch.Tensor = torch.stack(
        [padded[..., 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] for dy, dx in _WINDOW_OFFSETS], 2
    )
    magnitude: torch.Tensor = window[:, :, 8]

    # Get the indices for both directions
    positive_idx: torch.Tensor = angle_45 % 8
    positive_idx = positive_idx.long()

    negative_idx: torch.Tensor = (angle_45 + 4) % 8
    negative_idx = negative_idx.long()

    # The two neighbours along the gradient direction, read with one gather per channel. Inductor on MPS miscompiles
    # two separate gathers when the pixel is read from the padded magnitude rather than from the window: the second
    # comparison recomputes the pixel's magnitude from gx alone (pytorch/pytorch#199642)
    neighbour_both: torch.Tensor = torch.gather(window, 2, torch.stack([positive_idx, negative_idx], 2))
    neighbour_positive: torch.Tensor = neighbour_both[:, :, 0]
    neighbour_negative: torch.Tensor = neighbour_both[:, :, 1]

    # As in OpenCV's Canny, a pixel must be strictly greater than its left (upper) neighbour but only greater than or
    # equal to its right (lower) one along a horizontal (vertical) gradient, so of two equal pixels across a step the
    # left (upper) one is kept. Along a diagonal both comparisons are strict: the neighbours along it are two lines
    # apart, and the line between a tied pair carries the edge.
    is_max: torch.Tensor = _beats(magnitude, neighbour_positive, positive_idx) & _beats(
        magnitude, neighbour_negative, negative_idx
    )

    # A pixel no larger than the border, which has no gradient, is never kept. The comparisons above already exclude it,
    # since one of them is strict for every direction; this states the rule independently of the angle.
    is_max = is_max & (magnitude > padded[..., :1, :1])

    magnitude = magnitude * is_max

    # Threshold
    edges: torch.Tensor = F.threshold(magnitude, low_threshold, 0.0)

    low: torch.Tensor = magnitude > low_threshold
    high: torch.Tensor = magnitude > high_threshold

    # Cast before scaling: a boolean times a Python float exports as aten.mul.Scalar, which onnxscript < 0.7.2 cannot
    # translate. The result is 0, 0.5 or 1, exact in every floating dtype; the last cast keeps an integer input's dtype.
    edges = ((low.to(dtype) + high.to(dtype)) * 0.5).to(dtype)

    # Hysteresis
    if hysteresis:
        hysteresis_kernels: torch.Tensor = get_hysteresis_kernel(device, dtype)

        if is_exporting():
            edges_old: torch.Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)

            # Graph capture cannot branch on the data, so the loop is a while_loop, which the dynamo ONNX exporter
            # writes as a Loop node from torch 2.11. The body must not return its input, hence the clone.
            def _changed(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
                return ((old - new).abs() != 0).any()

            def _step(old: torch.Tensor, new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return new.clone(), _hysteresis_step(new, hysteresis_kernels)

            _, edges = torch.while_loop(_changed, _step, (edges_old, edges))
        elif torch.jit.is_tracing():
            edges = _hysteresis_loop_traced(edges, hysteresis_kernels)
        else:
            edges = _hysteresis_loop(edges, hysteresis_kernels)

        # The weak pixels left touch no strong pixel.
        edges = (edges == 1).to(dtype)

    return magnitude, edges


class Canny(nn.Module):
    r"""nn.Module that finds edges of the input image and filters them using the Canny algorithm.

    Convention:
        See the Convention block on :func:`~kornia.filters.canny`. The thresholds are validated at construction.

    Args:
        low_threshold: lower threshold for the hysteresis procedure, in the units of the unnormalised Sobel
            magnitude. It must be positive and at most ``high_threshold``.
        high_threshold: upper threshold for the hysteresis procedure, in the same units. It has no upper bound.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the gradient magnitude after non-maximum suppression, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

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
            input: Image tensor with shape :math:`(B, C, H, W)`, where
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
