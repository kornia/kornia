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

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_SIZE, KORNIA_CHECK_SHAPE
from kornia.core.utils import is_compiling, is_exporting

from .filter import _to_floating
from .kernels import _check_kernel_size, _unpack_2d_ks, get_gaussian_kernel2d
from .median import _compute_zero_padding


def _check_sigma_batch(name: str, sigma: torch.Tensor, input: torch.Tensor) -> None:
    """Check that a tensor sigma has a batch of 1, shared by the input, or the input batch."""
    KORNIA_CHECK_SAME_SIZE(sigma, input, allow_one=True, msg=f"{name} must have a batch of 1 or the input batch.")


# Measured on one image, 3 x 3 to 9 x 9 kernels, float32 and float64, with no page faults on either side: the tap loop
# overtakes the unfold path between 20k and 36k values (earlier for 3 x 3 kernels), so 65,536 values leaves a margin.
_BILATERAL_CPU_MIN_NUMEL = 65536
# The eager tap loop works through the batch in chunks of about this many input bytes, so that every tap's temporaries
# stay in cache: on 256 x 256 images, 2 MB chunks took 0.3x to 0.7x the time of one pass over a batch of 8 or 32.
_BILATERAL_CPU_CHUNK_BYTES = 2 * 1024 * 1024


def _records_gradient(
    input: torch.Tensor,
    guidance: Optional[torch.Tensor],
    sigma_color: float | torch.Tensor,
    sigma_space: tuple[float, float] | torch.Tensor,
) -> bool:
    """Whether autograd records a reverse-mode gradient through the filter."""
    return torch.is_grad_enabled() and (
        input.requires_grad
        or (guidance is not None and guidance.requires_grad)
        or (isinstance(sigma_color, torch.Tensor) and sigma_color.requires_grad)
        or (isinstance(sigma_space, torch.Tensor) and sigma_space.requires_grad)
    )


def _bilateral_blur_cpu_eligible(
    input: torch.Tensor,
    guidance: Optional[torch.Tensor],
    sigma_color: float | torch.Tensor,
    sigma_space: tuple[float, float] | torch.Tensor,
) -> bool:
    """Select large forward-only native-precision CPU inputs, where a tap loop beats unfolding every window."""
    # Keep the unfold path where it is what the caller captured or differentiates: the exporters and the legacy tracer
    # export it, and its reverse mode is one unfold backward instead of a chain of per-tap backwards.
    return (
        input.device.type == "cpu"
        and input.dtype in (torch.float32, torch.float64)
        and (guidance is None or guidance.device.type == "cpu")
        # Before the size check: torch.export would turn it into a guard of the exported program, which then fails at
        # run time for an image on the other side of the threshold although its dynamic shapes allow it.
        and not is_exporting()
        and input.numel() >= _BILATERAL_CPU_MIN_NUMEL
        and not torch.jit.is_tracing()
        and not _records_gradient(input, guidance, sigma_color, sigma_space)
    )


def _bilateral_blur_taps(
    image: torch.Tensor,
    guide: Optional[torch.Tensor],
    kernel_size: tuple[int, int],
    color_scale: float | torch.Tensor,
    space_kernel: torch.Tensor,
    border_type: str,
    color_distance_type: str,
    in_place: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Filter one batch chunk tap by tap, reusing its own temporaries ``in_place``, into ``out`` if given."""
    ky, kx = kernel_size
    height, width = image.shape[-2], image.shape[-1]
    pad_y, pad_x = (ky - 1) // 2, (kx - 1) // 2
    # The unfold path returned a contiguous result for any input layout.
    image = image.contiguous()
    padded_image = F.pad(image, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    if guide is None:
        guide, padded_guide = image, padded_image
    else:
        guide = guide.contiguous()
        padded_guide = F.pad(guide, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    if color_distance_type not in ("l1", "l2"):
        raise ValueError("color_distance_type only accepts l1 or l2")

    def tap_weight(tap: int) -> tuple[torch.Tensor, torch.Tensor]:
        # The unfold path's kernel entry for this tap, computed with the same operations in the same order.
        dy, dx = tap // kx, tap % kx
        diff = padded_guide[..., dy : dy + height, dx : dx + width] - guide
        if diff.shape[1] == 1:
            # One channel's sum is the value itself, and |d|^2 == d^2: both distances are the squared difference.
            color_distance_sq = diff.square_() if in_place else diff.square()
        elif color_distance_type == "l1":
            color_distance_sq = (diff.abs_() if in_place else diff.abs()).sum(1, keepdim=True)
            color_distance_sq = color_distance_sq.square_() if in_place else color_distance_sq.square()
        else:
            color_distance_sq = (diff.square_() if in_place else diff.square()).sum(1, keepdim=True)
        space_weight = space_kernel[:, tap : tap + 1]
        if in_place:
            weight = color_distance_sq.mul_(color_scale).exp_().mul_(space_weight)
        else:
            weight = space_weight * (color_scale * color_distance_sq).exp()
        return padded_image[..., dy : dy + height, dx : dx + width], weight

    values, weight = tap_weight(0)
    weighted = values * weight if out is None else out.copy_(values).mul_(weight)
    normalizer = weight
    for tap in range(1, ky * kx):
        values, weight = tap_weight(tap)
        if in_place:
            weighted.addcmul_(values, weight)
            normalizer.add_(weight)
        else:
            weighted = weighted.addcmul(values, weight)
            normalizer = normalizer + weight
    return weighted.div_(normalizer) if in_place else weighted / normalizer


def _bilateral_blur_cpu(
    input: torch.Tensor,
    guidance: Optional[torch.Tensor],
    kernel_size: tuple[int, int] | int,
    sigma_color: float | torch.Tensor,
    sigma_space: tuple[float, float] | torch.Tensor,
    border_type: str,
    color_distance_type: str,
) -> torch.Tensor:
    """Bilateral filter as a loop over the kernel taps, without unfolding every window.

    The unfold path materialises several :math:`(B, C, H, W, k_y, k_x)` tensors, :math:`k_y k_x` times the image; on a
    large CPU batch they are fresh memory, faulted in on every call. Here every tap takes shifted slices of the padded
    guidance and input, computes its weights with the unfold path's expression, so they are the same numbers, and adds
    them to the weighted sum and to the normalizer; only the order of the :math:`k_y k_x` terms of those two sums
    differs. Tensor sigmas have the input's dtype and device and a batch of 1 or the input's.
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, ky * kx, 1, 1)
    color_scale = -0.5 / sigma_color**2
    if isinstance(color_scale, torch.Tensor):
        color_scale = color_scale.view(-1, 1, 1, 1)
    # In place and chunk by chunk only where nothing records the arithmetic: autograd needs the temporaries that the
    # in-place operations overwrite, vmap has no batching rule for addcmul_, and compilers fuse the functional loop over
    # the whole batch without these allocations. Every value is computed the same way in both forms.
    in_place = (
        not is_compiling()
        and not torch._C._are_functorch_transforms_active()
        and not _records_gradient(input, guidance, sigma_color, sigma_space)
    )
    if not in_place:
        return _bilateral_blur_taps(
            input, guidance, (ky, kx), color_scale, space_kernel, border_type, color_distance_type, False, None
        )
    batch = input.shape[0]
    channels = input.shape[1] if guidance is None else max(input.shape[1], guidance.shape[1])
    chunk = max(1, _BILATERAL_CPU_CHUNK_BYTES // (channels * input.shape[-2] * input.shape[-1] * input.element_size()))
    if chunk >= batch:
        return _bilateral_blur_taps(
            input, guidance, (ky, kx), color_scale, space_kernel, border_type, color_distance_type, True, None
        )
    output = torch.empty(input.shape, device=input.device, dtype=input.dtype)
    for start in range(0, batch, chunk):
        stop = min(start + chunk, batch)
        scale = color_scale
        if isinstance(scale, torch.Tensor) and scale.shape[0] != 1:
            scale = scale[start:stop]
        space = space_kernel if space_kernel.shape[0] == 1 else space_kernel[start:stop]
        guide = None if guidance is None else guidance[start:stop]
        _bilateral_blur_taps(
            input[start:stop],
            guide,
            (ky, kx),
            scale,
            space,
            border_type,
            color_distance_type,
            True,
            output[start:stop],
        )
    return output


def _bilateral_blur(
    input: torch.Tensor,
    guidance: Optional[torch.Tensor],
    kernel_size: tuple[int, int] | int,
    sigma_color: float | torch.Tensor,
    sigma_space: tuple[float, float] | torch.Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> torch.Tensor:
    """Single implementation for both Bilateral Filter and Joint Bilateral Filter."""
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    if guidance is not None:
        # NOTE: allow guidance and input having different number of channels
        KORNIA_CHECK_IS_TENSOR(guidance)
        KORNIA_CHECK_SHAPE(guidance, ["B", "C", "H", "W"])
        KORNIA_CHECK(
            (guidance.shape[0] == input.shape[0]) and (guidance.shape[-2:] == input.shape[-2:]),
            "guidance and input should have the same batch size and spatial dimensions",
        )

    # integer values are differenced and weighted in float32: in their own dtype uint8 differences wrap (#5155)
    input = _to_floating(input)
    if guidance is not None and not guidance.is_floating_point():
        guidance = guidance.to(input.dtype)

    if isinstance(sigma_color, torch.Tensor):
        KORNIA_CHECK_SHAPE(sigma_color, ["B"])
        _check_sigma_batch("sigma_color", sigma_color, input)
        # `bool()` on a tensor is untraceable by dynamo; skip the data-dependent check under compile.
        if not is_compiling() and not bool((sigma_color > 0).all()):
            KORNIA_CHECK(False, f"sigma_color must be positive. Got {sigma_color}")
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype)
    elif not sigma_color > 0:
        KORNIA_CHECK(False, f"sigma_color must be positive. Got {sigma_color}")

    if isinstance(sigma_space, torch.Tensor):
        KORNIA_CHECK_SHAPE(sigma_space, ["B", "2"])
        _check_sigma_batch("sigma_space", sigma_space, input)
        # the spatial kernel takes the dtype of a tensor sigma, so match a floating input as for sigma_color
        if input.is_floating_point():
            sigma_space = sigma_space.to(device=input.device, dtype=input.dtype)

    ky, kx = _unpack_2d_ks(kernel_size)
    _check_kernel_size((ky, kx))

    # the colour kernel takes the guidance's dtype, so a wider guidance would promote the output
    if guidance is not None and input.is_floating_point() and guidance.is_floating_point():
        guidance = guidance.to(input.dtype)

    if _bilateral_blur_cpu_eligible(input, guidance, sigma_color, sigma_space):
        return _bilateral_blur_cpu(
            input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type
        )

    if isinstance(sigma_color, torch.Tensor):
        sigma_color = sigma_color.view(-1, 1, 1, 1, 1, 1)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    # Keep both patch axes: flattening them would copy every overlapping window.
    padded_input = F.pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1)  # (B, C, H, W, Ky, Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = F.pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1)  # (B, C, H, W, Ky, Kx)

    diff = unfolded_guidance - guidance[..., None, None]
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only accepts l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky, Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, ky, kx)

    kernel = space_kernel * color_kernel
    return (unfolded_input * kernel).sum((-2, -1)) / kernel.sum((-2, -1))


def bilateral_blur(
    input: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | torch.Tensor,
    sigma_space: tuple[float, float] | torch.Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> torch.Tensor:
    r"""Blur a torch.Tensor using a Bilateral filter.

    .. image:: _static/img/bilateral_blur.png

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Convention:
        - A neighbour weighs :math:`g \, \exp(-d^2 / (2 \sigma_{color}^2))`. :math:`g` is the Gaussian of
          :func:`~kornia.filters.gaussian_blur2d` with ``sigma = sigma_space``, :math:`(\sigma_y, \sigma_x)`, over
          the whole ``kernel_size`` rectangle, and :math:`d` is the neighbour's colour distance from the centre
          pixel: the sum of the absolute channel differences for ``'l1'``, their Euclidean norm for ``'l2'``.
          :ref:`Filtering <filtering-conventions>` compares this with OpenCV.
        - ``sigma_color`` is in the units of the input values: an image scaled by ``s > 0``, filtered with
          ``sigma_color * s``, gives the result scaled by ``s``.
        - The border modes are :func:`~kornia.filters.filter2d`'s, but only in lower case; see its Convention block.
        - A floating input keeps its dtype: a tensor ``sigma_color`` or ``sigma_space`` is cast to it and moved to
          its device.
        - An integer or bool input is differenced and averaged in float32 and the output is float32, as in
          :func:`~kornia.filters.filter2d`, so uint8 differences do not wrap.
        - On the CPU, a float32 or float64 input (after that conversion) with at least 65,536 values is filtered tap
          by tap from shifted slices of the padded input instead of from its unfolded windows, which avoids
          temporaries of ``kernel_size`` times the image, unless a reverse-mode gradient is recorded through any
          argument, or ``torch.export`` or ``torch.jit.trace`` captures the call, so an exported program keeps the
          windows for every input size. Both forms compute the same weights and add up the weighted values in a
          different order, so they agree to rounding.

    Arguments:
        input: the input torch.Tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel. Each entry must be a positive odd integer.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges. It must be positive. A float is shared by the batch; a
          torch.Tensor has shape :math:`(1,)`, shared by the batch, or :math:`(B,)`, one value per sample.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`. A tuple of two floats is shared by the batch;
          a torch.Tensor has shape :math:`(1, 2)`, shared by the batch, or :math:`(B, 2)`, one row per sample.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Default: ``'l1'``.

    Returns:
        the blurred torch.Tensor with shape :math:`(B, C, H, W)`.

    Raises:
        BaseError: if an entry of ``kernel_size`` is even or not positive.
        BaseError: if ``sigma_color`` is not positive.
        BaseError: if the batch of a tensor ``sigma_color`` or ``sigma_space`` is neither 1 nor the input batch.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = bilateral_blur(input, (3, 3), 0.1, (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """
    return _bilateral_blur(input, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def joint_bilateral_blur(
    input: torch.Tensor,
    guidance: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | torch.Tensor,
    sigma_space: tuple[float, float] | torch.Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> torch.Tensor:
    r"""Blur a torch.Tensor using a Joint Bilateral filter.

    .. image:: _static/img/joint_bilateral_blur.png

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :func:`bilateral_blur()` for more information.

    Convention:
        - See the Convention block on :func:`~kornia.filters.bilateral_blur`; the colour distances are taken in
          ``guidance``, and ``input`` is what gets averaged.
        - ``input`` comes first and ``guidance`` second, the opposite of :func:`~kornia.filters.guided_blur`.
        - ``guidance`` may have its own channel count; its batch size and :math:`(H, W)` must equal ``input``'s.
        - ``guidance`` is cast to the dtype of a floating ``input``, so the output keeps ``input``'s dtype. An integer
          ``input`` is filtered in float32, as by :func:`~kornia.filters.bilateral_blur`, and its ``guidance`` is
          cast to float32.

    Arguments:
        input: the input torch.Tensor with shape :math:`(B,C,H,W)`.
        guidance: the guidance torch.Tensor with shape :math:`(B,C_g,H,W)`.
        kernel_size: the size of the kernel. Each entry must be a positive odd integer.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges. It must be positive. A float is shared by the batch; a
          torch.Tensor has shape :math:`(1,)`, shared by the batch, or :math:`(B,)`, one value per sample.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`. A tuple of two floats is shared by the batch;
          a torch.Tensor has shape :math:`(1, 2)`, shared by the batch, or :math:`(B, 2)`, one row per sample.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Default: ``'l1'``.

    Returns:
        the blurred torch.Tensor with shape :math:`(B, C, H, W)`.

    Raises:
        BaseError: if an entry of ``kernel_size`` is even or not positive.
        BaseError: if ``sigma_color`` is not positive.
        BaseError: if the batch of a tensor ``sigma_color`` or ``sigma_space`` is neither 1 nor the input batch.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> guidance = torch.rand(2, 4, 5, 5)
        >>> output = joint_bilateral_blur(input, guidance, (3, 3), 0.1, (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """
    return _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


# trick to make mypy not throw errors about difference in .forward() signatures of subclass and superclass
class _BilateralBlur(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | torch.Tensor,
        sigma_space: tuple[float, float] | torch.Tensor,
        border_type: str = "reflect",
        color_distance_type: str = "l1",
    ) -> None:
        super().__init__()
        _check_kernel_size(_unpack_2d_ks(kernel_size))
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )


class BilateralBlur(_BilateralBlur):
    r"""Blur a torch.Tensor using a Bilateral filter.

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Convention:
        See the Convention block on :func:`~kornia.filters.bilateral_blur`.

    Arguments:
        kernel_size: the size of the kernel. Each entry must be a positive odd integer.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges. It must be positive. A float is shared by the batch; a
          torch.Tensor has shape :math:`(1,)`, shared by the batch, or :math:`(B,)`, one value per sample.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`. A tuple of two floats is shared by the batch;
          a torch.Tensor has shape :math:`(1, 2)`, shared by the batch, or :math:`(B, 2)`, one row per sample.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Default: ``'l1'``.

    Returns:
        the blurred input torch.Tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Raises:
        BaseError: if an entry of ``kernel_size`` is even or not positive; raised from the constructor.
        BaseError: if ``sigma_color`` is not positive; raised from ``forward``.
        BaseError: if the batch of a tensor ``sigma_color`` or ``sigma_space`` is neither 1 nor the input batch;
          raised from ``forward``.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> blur = BilateralBlur((3, 3), 0.1, (1.5, 1.5))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Smooth an image while keeping strong intensity edges visible.

        Bilateral filtering combines two weights for every pixel in the local
        window: a spatial weight based on distance in the image plane and a
        range weight based on color or intensity similarity. Nearby pixels with
        similar values contribute strongly, while pixels across an edge are
        suppressed even if they are spatially close.

        Args:
            input: Input image tensor with shape :math:`(B, C, H, W)`,
                where :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the image height, and :math:`W` is the
                image width.

        Returns:
            Tensor with shape :math:`(B, C, H, W)` containing the
            edge-preserving smoothed image. The output keeps the layout and
            device of ``input``, and the dtype of a floating ``input`` unless a
            tensor ``sigma_space`` of a wider dtype promotes it, while reducing
            small local variations according to the configured kernel size and
            sigma values.
        """
        return bilateral_blur(
            input, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type
        )


class JointBilateralBlur(_BilateralBlur):
    r"""Blur a torch.Tensor using a Joint Bilateral filter.

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :class:`BilateralBlur` for more information.

    Convention:
        See the Convention block on :func:`~kornia.filters.joint_bilateral_blur`.

    Arguments:
        kernel_size: the size of the kernel. Each entry must be a positive odd integer.
        sigma_color: the standard deviation for intensity/color Gaussian kernel.
          Smaller values preserve more edges. It must be positive. A float is shared by the batch; a
          torch.Tensor has shape :math:`(1,)`, shared by the batch, or :math:`(B,)`, one value per sample.
        sigma_space: the standard deviation for spatial Gaussian kernel.
          This is similar to ``sigma`` in :func:`gaussian_blur2d()`. A tuple of two floats is shared by the batch;
          a torch.Tensor has shape :math:`(1, 2)`, shared by the batch, or :math:`(B, 2)`, one row per sample.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        color_distance_type: the type of distance to calculate intensity/color
          difference. Only ``'l1'`` or ``'l2'`` is allowed. Default: ``'l1'``.

    Returns:
        the blurred input torch.Tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B, C_g, H, W)`
        - Output: :math:`(B, C, H, W)`

    Raises:
        BaseError: if an entry of ``kernel_size`` is even or not positive; raised from the constructor.
        BaseError: if ``sigma_color`` is not positive; raised from ``forward``.
        BaseError: if the batch of a tensor ``sigma_color`` or ``sigma_space`` is neither 1 nor the input batch;
          raised from ``forward``.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> guidance = torch.rand(2, 4, 5, 5)
        >>> blur = JointBilateralBlur((3, 3), 0.1, (1.5, 1.5))
        >>> output = blur(input, guidance)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """

    def forward(self, input: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """Smooth an input image using edges measured from a guidance image.

        Joint bilateral filtering is useful when the image being smoothed and
        the image that should define the edges are different tensors. The
        spatial kernel is applied around each location of ``input``, but the
        range similarity term is computed from ``guidance``. This lets a clean
        or higher-quality reference image preserve boundaries in another
        signal, such as a depth map, mask, or noisy feature image.

        Args:
            input: Tensor to smooth with shape :math:`(B, C, H, W)`, where
                :math:`B` is the batch size, :math:`C` is the number of input
                channels, :math:`H` is the height, and :math:`W` is the width.
            guidance: Tensor used to compute range weights. Its batch and
                spatial dimensions must equal those of ``input``; its
                channel count may differ when the guidance signal uses a
                different representation.

        Returns:
            Tensor with shape :math:`(B, C, H, W)` containing the filtered
            ``input`` values. Edges present in ``guidance`` reduce mixing
            across boundaries, while smooth regions are averaged more strongly.
        """
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )
