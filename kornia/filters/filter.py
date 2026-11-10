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

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_SIZE, KORNIA_CHECK_SHAPE
from kornia.core.utils import _to_device_nonblocking
from kornia.filters.kernels import normalize_kernel2d

_VALID_BORDERS = {"constant", "reflect", "replicate", "circular"}
_VALID_PADDING = {"valid", "same"}
_VALID_BEHAVIOUR = {"conv", "corr"}


def _pad(input: torch.Tensor, padding: list[int], border_type: str) -> torch.Tensor:
    """Extend reflect101 and wrap borders beyond a spatial axis, retaining native supported padding."""
    if border_type not in ("reflect", "circular"):
        return F.pad(input, padding, mode=border_type)
    # Match the native limits so ordinary calls need no extension or extra export guards.
    limit = 1 if border_type == "reflect" else 0
    shape = input.shape
    if len(padding) > 2 * len(shape):
        return F.pad(input, padding, mode=border_type)
    needs_extension = False
    for i, pad in enumerate(padding):
        if pad > shape[-(i // 2) - 1] - limit:
            needs_extension = True
            break
    if not needs_extension:
        return F.pad(input, padding, mode=border_type)

    dimensions = len(padding) // 2
    # Leave cropping, empty axes and unsupported ranks to the backend's existing contract.
    if (
        len(padding) not in (2, 4, 6)
        or input.ndim not in (dimensions + 1, dimensions + 2)
        or any(p < 0 for p in padding)
        or any(input.shape[-axis - 1] == 0 for axis in range(dimensions))
    ):
        return F.pad(input, padding, mode=border_type)
    for axis in range(dimensions):
        before, after = padding[2 * axis], padding[2 * axis + 1]
        if before == 0 and after == 0:
            continue
        size = input.shape[-axis - 1]
        indices = torch.arange(-before, size + after, device=input.device)
        if border_type == "circular":
            indices = indices.remainder(size)
        else:
            # Keep the singleton period symbolic when tracing across image sizes.
            period = 2 * size - 2 + (size == 1)
            indices = indices.remainder(period)
            indices = torch.minimum(indices, period - indices)
        input = input.index_select(-axis - 1, indices)
    return input


def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _to_floating(input: torch.Tensor) -> torch.Tensor:
    """Promote an integer or bool input to float32, the compute dtype for filtering.

    A kernel cast to an integer dtype truncates its fractional taps and wraps its sums, so an integer image is filtered
    in float32 and the result stays float32, as the color conversions do for integer images (#4053, #5155).
    """
    if input.is_floating_point() or input.is_complex():
        return input
    return input.to(torch.float32)


def _check_kernel_batch(input: torch.Tensor, kernel: torch.Tensor) -> None:
    """Check that the kernel batch is 1 or the input batch."""
    KORNIA_CHECK_SAME_SIZE(kernel, input, allow_one=True, msg="The kernel batch must be 1 or the input batch.")


def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
    behaviour: str = "corr",
) -> torch.Tensor:
    r"""Filter an image with a 2D kernel, by correlation or convolution.

    ``out = filter2d(input, kernel)`` correlates each channel of a ``(B, C, H, W)`` image with a ``(1, kH, kW)`` or
    ``(B, kH, kW)`` kernel. ``behaviour='conv'`` convolves instead, and ``padding='valid'`` drops the border instead
    of padding it. To blur with a Gaussian or box kernel, use :func:`gaussian_blur2d` or :func:`box_blur`.

    Convention:
        - The default ``behaviour='corr'`` is cross-correlation,
          :math:`out[y, x] = \sum_{i, j} k[i, j] \, input[y + i - a_y, x + j - a_x]`, anchored at
          :math:`(a_y, a_x)` = ``((kH - 1) // 2, (kW - 1) // 2)``: the centre of an odd kernel and, for an even size,
          the tap before the middle. ``behaviour='conv'`` flips the kernel on both axes and keeps the anchor.
        - ``border_type`` takes the :func:`torch.nn.functional.pad` mode names, and the default ``'reflect'`` mirrors
          about the edge pixel without repeating it. :ref:`Filtering <filtering-conventions>` maps the modes and the
          anchor onto scipy and OpenCV. Reflect and circular borders extend to any positive axis length: reflection
          repeats with period ``2 * (n - 1)`` on an axis of length ``n > 1`` and repeats the pixel when ``n = 1``;
          ``'circular'`` wraps with period ``n``. Working backend padding calls retain their native path.
        - A ``(1, kH, kW)`` kernel is shared by the whole batch, and a ``(B, kH, kW)`` kernel gives each
          sample its own, shared by the sample's channels; there are no per-channel kernels. Any other kernel batch
          raises.
          An empty batch accepts either a shared kernel or an empty per-sample kernel and retains the usual output
          spatial size, dtype promotion and gradient connections to both input and kernel.
        - ``normalized=True`` divides each kernel by the sum of its absolute values, so a zero-sum derivative kernel
          keeps its sign.
        - ``border_type``, ``padding`` and ``behaviour`` are case-insensitive: ``'REFLECT'``, ``'SAME'`` and
          ``'CONV'`` are ``'reflect'``, ``'same'`` and ``'conv'``.
        - For a floating input, the kernel is cast to the input's dtype and device and stays differentiable, and the
          output has the input's dtype.
        - An integer or bool input is filtered in float32 and the output is float32, on the input's scale: a uint8
          image is not rescaled to :math:`[0, 1]`, and the result is not rounded or clamped back to the integer
          range (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        input: image with shape ``(B, C, H, W)``.
        kernel: kernel with shape ``(1, kH, kW)`` or ``(B, kH, kW)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``; unused with
          ``padding='valid'``.
        normalized: whether to L1-normalize the kernel.
        padding: ``'same'`` pads the input so the output keeps its size, ``'valid'`` does not pad.
        behaviour: ``'corr'`` correlates, ``'conv'`` flips the kernel first.

    Return:
        the filtered image. With ``padding='same'`` it has the input's shape ``(B, C, H, W)``. With
        ``padding='valid'`` it has shape ``(B, C, H - kH + 1, W - kW + 1)``, and its pixel ``(0, 0)`` is the
        ``'same'`` output at the anchor ``(a_y, a_x)``.

    See Also:
        - :func:`correlate2d`: ``behaviour`` fixed to ``'corr'``.
        - :func:`convolve2d`: ``behaviour`` fixed to ``'conv'``.
        - :func:`filter2d_separable`: two 1D kernels, for a kernel that is their outer product.
        - :func:`filter3d`: the same filter for volumes.
        - :func:`fft_conv`: the same filter computed with FFTs, for large kernels.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ["B", "H", "W"])
    _check_kernel_batch(input, kernel)

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, {border_type}. Expected one of {_VALID_BORDERS}",
    )
    KORNIA_CHECK(
        str(padding).lower() in _VALID_PADDING,
        f"Invalid padding mode, {padding}. Expected one of {_VALID_PADDING}",
    )
    KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f"Invalid behaviour mode, {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )
    # the checks are case-insensitive, so dispatch on the lower-case spelling as well
    border_type, padding, behaviour = str(border_type).lower(), str(padding).lower(), str(behaviour).lower()

    input = _to_floating(input)

    # prepare kernel
    b, c, h, w = input.shape
    if behaviour == "conv":
        tmp_kernel = _to_device_nonblocking(kernel.flip((-2, -1))[:, None, ...], input.device, input.dtype)
    else:
        tmp_kernel = _to_device_nonblocking(kernel[:, None, ...], input.device, input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    # The legacy ONNX tracer records ``input.shape`` reads as graph values, so with a dynamic batch or size the
    # expanded kernel, and with it the convolution weight, has no static shape and the export fails (#5222).
    # The channel count sets the convolution's ``groups``, which ONNX needs as a constant anyway: read it as an int.
    tmp_kernel = tmp_kernel.expand(-1, int(c) if torch.jit.is_tracing() else c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape: list[int] = _compute_padding([height, width])
        input = _pad(input, padding_shape, border_type)

    # An empty per-sample kernel needs one structural row for nonzero convolution groups. Reduce after
    # normalization to avoid normalizing a zero kernel, while keeping the original kernel in the autograd graph.
    if tmp_kernel.shape[0] == 0:
        tmp_kernel = tmp_kernel.sum(dim=0, keepdim=True)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.reshape(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, h, w) if padding == "same" else output.view(b, c, h - height + 1, w - width + 1)


def filter2d_separable(
    input: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Filter an image with two 1D kernels, one along x and one along y.

    ``out = filter2d_separable(input, kernel_x, kernel_y)`` correlates each channel of a ``(B, C, H, W)`` image with
    ``kernel_x`` along ``W`` and then with ``kernel_y`` along ``H``. It is :func:`filter2d` with the outer-product
    kernel.

    Convention:
        See the Convention block on :func:`~kornia.filters.filter2d`. The outer-product kernel is
        ``kernel_y[:, :, None] * kernel_x[:, None, :]``, and the even-size anchor is the same. There is no
        ``behaviour`` argument: the kernels are always correlated. ``normalized=True`` normalizes each 1D kernel.

    Args:
        input: image with shape ``(B, C, H, W)``.
        kernel_x: kernel along ``W``, with shape ``(1, kW)`` or ``(B, kW)``.
        kernel_y: kernel along ``H``, with shape ``(1, kH)`` or ``(B, kH)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``; unused with
          ``padding='valid'``.
        normalized: whether to L1-normalize each kernel.
        padding: ``'same'`` pads the input so the output keeps its size, ``'valid'`` does not pad.

    Return:
        the filtered image, with the shape :func:`~kornia.filters.filter2d` returns for the same ``padding``:
        ``(B, C, H, W)`` with ``padding='same'``.

    See Also:
        - :func:`filter2d`: any 2D kernel, with ``behaviour`` and a kernel that need not be separable.
        - :func:`gaussian_blur2d`: a Gaussian blur from ``kernel_size`` and ``sigma``, which builds the kernels itself.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3)

        >>> filter2d_separable(input, kernel, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])

    """
    out_x = filter2d(input, kernel_x[..., None, :], border_type, normalized, padding)
    return filter2d(out_x, kernel_y[..., None], border_type, normalized, padding)


def filter3d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "replicate",
    normalized: bool = False,
    behaviour: str = "corr",
) -> torch.Tensor:
    r"""Filter a volume with a 3D kernel, by correlation or convolution.

    ``out = filter3d(input, kernel)`` correlates each channel of a ``(B, C, D, H, W)`` volume with a
    ``(1, kD, kH, kW)`` or ``(B, kD, kH, kW)`` kernel, after padding it by ``border_type``.
    ``behaviour='conv'`` convolves instead.

    Convention:
        - See the Convention block on :func:`~kornia.filters.filter2d`, applied to the ``(D, H, W)`` axes with the
          anchor ``((kD - 1) // 2, (kH - 1) // 2, (kW - 1) // 2)``.
        - The default ``border_type`` is ``'replicate'``, where :func:`~kornia.filters.filter2d` defaults to
          ``'reflect'``. There is no ``padding`` argument: the output always has the input's shape.

    Args:
        input: volume with shape ``(B, C, D, H, W)``.
        kernel: kernel with shape ``(1, kD, kH, kW)`` or ``(B, kD, kH, kW)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: whether to L1-normalize the kernel.
        behaviour: ``'corr'`` correlates, ``'conv'`` flips the kernel first.

    Return:
        the filtered volume, with shape ``(B, C, D, H, W)``.

    See Also:
        - :func:`correlate3d`: ``behaviour`` fixed to ``'corr'``.
        - :func:`convolve3d`: ``behaviour`` fixed to ``'conv'``.
        - :func:`filter2d`: the same filter for images, with ``padding`` and a ``'reflect'`` default border.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3d(input, kernel)
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "D", "H", "W"])
    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ["B", "D", "H", "W"])
    _check_kernel_batch(input, kernel)

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, got {border_type}. Expected one of {_VALID_BORDERS}",
    )

    KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f"Invalid behaviour mode, got {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )
    # the checks are case-insensitive, so dispatch on the lower-case spelling as well
    border_type, behaviour = str(border_type).lower(), str(behaviour).lower()

    input = _to_floating(input)

    # prepare kernel
    b, c, d, h, w = input.shape
    if behaviour == "conv":
        tmp_kernel = _to_device_nonblocking(kernel.flip((-3, -2, -1))[:, None, ...], input.device, input.dtype)
    else:
        tmp_kernel = _to_device_nonblocking(kernel[:, None, ...], input.device, input.dtype)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.reshape(bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: list[int] = _compute_padding([depth, height, width])
    input_pad = _pad(input, padding_shape, border_type)

    # As in filter2d, preserve the empty kernel's gradient connection without using zero convolution groups.
    if tmp_kernel.shape[0] == 0:
        tmp_kernel = tmp_kernel.sum(dim=0, keepdim=True)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.reshape(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv3d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, d, h, w)


def fft_conv(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
    behaviour: str = "corr",
) -> torch.Tensor:
    r"""Filter an image with a 2D kernel through the FFT, for large kernels.

    ``out = fft_conv(input, kernel)`` filters each channel of a ``(B, C, H, W)`` image with a ``(1, kH, kW)`` or
    ``(B, kH, kW)`` kernel, as :func:`filter2d` does, by multiplying in the frequency domain. Its cost barely grows
    with the kernel size, so it pays off only for large kernels.

    Convention:
        - See the Convention block on :func:`~kornia.filters.filter2d`: for the same arguments and a finite real input
          ``fft_conv`` returns the same result, to roundoff. A NaN or inf in a channel of the input makes that
          channel's whole output non-finite, where :func:`~kornia.filters.filter2d` keeps it to the pixels whose
          window reaches it.
        - An integer or bool input is filtered in float32 and the output is float32, as in
          :func:`~kornia.filters.filter2d`.
        - A float16 or bfloat16 input is transformed in float32 on every device, and the result is cast back to the
          input's dtype.

    Args:
        input: image with shape ``(B, C, H, W)``.
        kernel: kernel with shape ``(1, kH, kW)`` or ``(B, kH, kW)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``; unused with
          ``padding='valid'``.
        normalized: whether to L1-normalize the kernel.
        padding: ``'same'`` pads the input so the output keeps its size, ``'valid'`` does not pad.
        behaviour: ``'corr'`` correlates, ``'conv'`` flips the kernel first.

    Returns:
        the filtered image. With ``padding='same'`` it has the input's shape ``(B, C, H, W)``. With
        ``padding='valid'`` it has shape ``(B, C, H - kH + 1, W - kW + 1)``.

    See Also:
        - :func:`filter2d`: the same filter as a direct sliding window.

    Example:
        >>> input = torch.tensor([[[[
        ...     0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> fft_conv(input, kernel, padding="same")  # doctest: +SKIP
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ["B", "H", "W"])
    _check_kernel_batch(input, kernel)

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, {border_type}. Expected one of {_VALID_BORDERS}",
    )

    KORNIA_CHECK(
        str(padding).lower() in _VALID_PADDING,
        f"Invalid padding mode, {padding}. Expected one of {_VALID_PADDING}",
    )

    KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f"Invalid behaviour mode, {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )
    # the checks are case-insensitive, so dispatch on the lower-case spelling as well
    border_type, padding, behaviour = str(border_type).lower(), str(padding).lower(), str(behaviour).lower()

    input = _to_floating(input)

    _, c, h, w = input.shape
    kh, kw = kernel.shape[-2:]

    # without padding the kernel has to fit inside the input, as F.conv2d requires in filter2d
    KORNIA_CHECK(
        padding == "same" or (kh <= h and kw <= w),
        lambda: (
            f"With padding='valid' the kernel must not be larger than the input. Got a {kh} x {kw} kernel for a "
            f"{h} x {w} input"
        ),
    )

    if behaviour == "conv":
        tmp_kernel = _to_device_nonblocking(kernel.flip((-2, -1))[:, None, ...], input.device, input.dtype)
    else:
        tmp_kernel = _to_device_nonblocking(kernel[:, None, ...], input.device, input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    # Expand kernel across channels
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # Padding (spatial domain)
    if padding == "same":
        padding_shape = _compute_padding([kh, kw])
        input_padded = _pad(input, padding_shape, border_type)
    else:
        input_padded = input

    padded_h, padded_w = input_padded.shape[-2:]

    input_padded = input_padded.contiguous()
    tmp_kernel = tmp_kernel.contiguous()

    # The MKL FFT backend raises on an empty batch (pytorch/pytorch#174984), so there is nothing to transform: crop
    # the empty input to the output size. The zero-weighted kernel sum keeps the kernel in the autograd graph, as
    # filter2d does.
    if input_padded.shape[0] == 0:
        return input_padded[..., : padded_h - kh + 1, : padded_w - kw + 1] + tmp_kernel.sum() * 0

    # torch has no half-precision FFT that covers every input: the CPU and MPS kernels support neither float16 nor
    # bfloat16, cuFFT computes float16 only for power-of-two sizes, and a CUDA bfloat16 FFT raises before torch 2.14
    # and returns float32 from 2.14 on. Half inputs are therefore transformed in float32 on every device and cast
    # back once.
    is_half = input.dtype in (torch.float16, torch.bfloat16)
    if is_half:
        input_padded = input_padded.float()
        tmp_kernel = tmp_kernel.float()

    # FFT
    input_fr = torch.fft.rfftn(input_padded, dim=(-2, -1))
    kernel_fr = torch.fft.rfftn(tmp_kernel, s=(padded_h, padded_w), dim=(-2, -1))

    # Correlation via conjugation
    output_fr = input_fr * torch.conj(kernel_fr)

    # Inverse FFT
    output = torch.fft.irfftn(output_fr, s=(padded_h, padded_w), dim=(-2, -1))

    # Crop to valid region
    crop_h = padded_h - kh + 1
    crop_w = padded_w - kw + 1
    output = output[..., :crop_h, :crop_w].contiguous()

    return output.to(input.dtype) if is_half else output


def correlate2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Correlate an image with a 2D kernel, without flipping the kernel.

    ``out = correlate2d(input, kernel)`` is :func:`filter2d` with ``behaviour='corr'``: each output pixel is the sum of
    the kernel times the pixels under it. It equals :func:`convolve2d` for a kernel that is its own 180-degree
    rotation.

    Convention:
        See the Convention block on :func:`~kornia.filters.filter2d`.

    Args:
        input: image with shape ``(B, C, H, W)``.
        kernel: kernel with shape ``(1, kH, kW)`` or ``(B, kH, kW)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``; unused with
          ``padding='valid'``.
        normalized: whether to L1-normalize the kernel.
        padding: ``'same'`` pads the input so the output keeps its size, ``'valid'`` does not pad.

    Return:
        the correlated image. With ``padding='same'`` it has the input's shape ``(B, C, H, W)``. With
        ``padding='valid'`` it has shape ``(B, C, H - kH + 1, W - kW + 1)``.

    See Also:
        - :func:`convolve2d`: the same with the kernel flipped.
        - :func:`correlate3d`: the same for volumes, with a ``'replicate'`` default border and no ``padding``.
        - :func:`filter2d`: both in one function, through ``behaviour``.

    Example:
        Correlating an impulse gives the kernel rotated by 180 degrees; :func:`convolve2d` gives the kernel itself.

        >>> input = torch.zeros(1, 1, 5, 5)
        >>> input[..., 2, 2] = 1.0
        >>> kernel = torch.arange(1.0, 10.0).reshape(1, 3, 3)
        >>> correlate2d(input, kernel)
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 9., 8., 7., 0.],
                  [0., 6., 5., 4., 0.],
                  [0., 3., 2., 1., 0.],
                  [0., 0., 0., 0., 0.]]]])

        With ``padding='valid'`` the output loses ``kH - 1`` rows and ``kW - 1`` columns:

        >>> correlate2d(input, kernel, padding='valid').shape
        torch.Size([1, 1, 3, 3])
    """
    return filter2d(input, kernel, border_type=border_type, normalized=normalized, padding=padding, behaviour="corr")


def convolve2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Convolve an image with a 2D kernel, flipping the kernel along ``H`` and ``W``.

    ``out = convolve2d(input, kernel)`` is :func:`filter2d` with ``behaviour='conv'``: a bright pixel in the image
    leaves a copy of the kernel in the output, not its rotation.

    Convention:
        See the Convention block on :func:`~kornia.filters.filter2d`.

    Args:
        input: image with shape ``(B, C, H, W)``.
        kernel: kernel with shape ``(1, kH, kW)`` or ``(B, kH, kW)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``; unused with
          ``padding='valid'``.
        normalized: whether to L1-normalize the kernel.
        padding: ``'same'`` pads the input so the output keeps its size, ``'valid'`` does not pad.

    Return:
        the convolved image. With ``padding='same'`` it has the input's shape ``(B, C, H, W)``. With
        ``padding='valid'`` it has shape ``(B, C, H - kH + 1, W - kW + 1)``.

    See Also:
        - :func:`correlate2d`: the same without flipping the kernel.
        - :func:`convolve3d`: the same for volumes, with a ``'replicate'`` default border and no ``padding``.
        - :func:`filter2d`: both in one function, through ``behaviour``.

    Example:
        Convolving an impulse gives the kernel itself; :func:`correlate2d` gives it rotated by 180 degrees.

        >>> input = torch.zeros(1, 1, 5, 5)
        >>> input[..., 2, 2] = 1.0
        >>> kernel = torch.arange(1.0, 10.0).reshape(1, 3, 3)
        >>> convolve2d(input, kernel)
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 1., 2., 3., 0.],
                  [0., 4., 5., 6., 0.],
                  [0., 7., 8., 9., 0.],
                  [0., 0., 0., 0., 0.]]]])

        With ``padding='valid'`` the output loses ``kH - 1`` rows and ``kW - 1`` columns:

        >>> convolve2d(input, kernel, padding='valid').shape
        torch.Size([1, 1, 3, 3])
    """
    return filter2d(input, kernel, border_type=border_type, normalized=normalized, padding=padding, behaviour="conv")


def correlate3d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "replicate",
    normalized: bool = False,
) -> torch.Tensor:
    r"""Correlate a volume with a 3D kernel, without flipping the kernel.

    ``out = correlate3d(input, kernel)`` is :func:`filter3d` with ``behaviour='corr'``. It equals :func:`convolve3d`
    for a kernel that flipping all three axes leaves unchanged.

    Convention:
        See the Convention block on :func:`~kornia.filters.filter3d`.

    Args:
        input: volume with shape ``(B, C, D, H, W)``.
        kernel: kernel with shape ``(1, kD, kH, kW)`` or ``(B, kD, kH, kW)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: whether to L1-normalize the kernel.

    Return:
        the correlated volume, with the input's shape ``(B, C, D, H, W)``.

    See Also:
        - :func:`convolve3d`: the same with the kernel flipped.
        - :func:`filter3d`: both in one function, through ``behaviour``.
        - :func:`correlate2d`: the same for images, with ``padding`` and a ``'reflect'`` default border.

    Example:
        Correlation applies the kernel as given. A single tap at the kernel's first corner, index ``(0, 0, 0)`` in
        ``(D, H, W)``, makes each output voxel copy the input voxel one step back along all three axes. An impulse at
        the centre ``(1, 1, 1)`` of a 3x3x3 volume therefore lands in the volume's last corner, ``(2, 2, 2)``.
        :func:`convolve3d` puts it in the first corner, ``(0, 0, 0)``.

        >>> input = torch.zeros(1, 1, 3, 3, 3)
        >>> input[0, 0, 1, 1, 1] = 1.0  # the impulse, at the centre of the volume
        >>> kernel = torch.zeros(1, 3, 3, 3)
        >>> kernel[0, 0, 0, 0] = 1.0  # the tap, at the kernel's first corner
        >>> output = correlate3d(input, kernel)
        >>> output[0, 0].nonzero()  # (D, H, W) index of the one non-zero output voxel
        tensor([[2, 2, 2]])
    """
    return filter3d(input, kernel, border_type=border_type, normalized=normalized, behaviour="corr")


def convolve3d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "replicate",
    normalized: bool = False,
) -> torch.Tensor:
    r"""Convolve a volume with a 3D kernel, flipping the kernel along ``D``, ``H`` and ``W``.

    ``out = convolve3d(input, kernel)`` is :func:`filter3d` with ``behaviour='conv'``: a bright voxel in the volume
    leaves a copy of the kernel in the output, not its mirror image through the centre.

    Convention:
        See the Convention block on :func:`~kornia.filters.filter3d`.

    Args:
        input: volume with shape ``(B, C, D, H, W)``.
        kernel: kernel with shape ``(1, kD, kH, kW)`` or ``(B, kD, kH, kW)``.
        border_type: padding mode, ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: whether to L1-normalize the kernel.

    Return:
        the convolved volume, with the input's shape ``(B, C, D, H, W)``.

    See Also:
        - :func:`correlate3d`: the same without flipping the kernel.
        - :func:`filter3d`: both in one function, through ``behaviour``.
        - :func:`convolve2d`: the same for images, with ``padding`` and a ``'reflect'`` default border.

    Example:
        Convolution flips the kernel along all three axes, which moves a tap at the kernel's first corner, index
        ``(0, 0, 0)`` in ``(D, H, W)``, to its last corner. Each output voxel then copies the input voxel one step
        ahead along all three axes, so an impulse at the centre ``(1, 1, 1)`` of a 3x3x3 volume lands in the volume's
        first corner, ``(0, 0, 0)``. :func:`correlate3d` puts it in the last corner, ``(2, 2, 2)``.

        >>> input = torch.zeros(1, 1, 3, 3, 3)
        >>> input[0, 0, 1, 1, 1] = 1.0  # the impulse, at the centre of the volume
        >>> kernel = torch.zeros(1, 3, 3, 3)
        >>> kernel[0, 0, 0, 0] = 1.0  # the tap, at the kernel's first corner
        >>> output = convolve3d(input, kernel)
        >>> output[0, 0].nonzero()  # (D, H, W) index of the one non-zero output voxel
        tensor([[0, 0, 0]])
    """
    return filter3d(input, kernel, border_type=border_type, normalized=normalized, behaviour="conv")
