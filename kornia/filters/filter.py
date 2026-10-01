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

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters.kernels import normalize_kernel2d

_VALID_BORDERS = {"constant", "reflect", "replicate", "circular"}
_VALID_PADDING = {"valid", "same"}
_VALID_BEHAVIOUR = {"conv", "corr"}


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


def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
    behaviour: str = "corr",
) -> torch.Tensor:
    r"""Filter a tensor with a 2d kernel, by cross-correlation unless ``behaviour='conv'``.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each channel of the tensor. With ``padding='same'`` the
    input is first padded according to ``border_type``, so that the output
    keeps the input's height and width.

    Convention:
        - The default ``behaviour='corr'`` is cross-correlation,
          :math:`out[y, x] = \sum_{i, j} k[i, j] \, input[y + i - a_y, x + j - a_x]`, anchored at
          :math:`(a_y, a_x)` = ``((kH - 1) // 2, (kW - 1) // 2)``: the centre of an odd kernel and, for an even size,
          the tap before the middle. ``behaviour='conv'`` flips the kernel on both axes and keeps the anchor.
        - ``border_type`` takes the :func:`torch.nn.functional.pad` mode names, and the default ``'reflect'`` mirrors
          about the edge pixel without repeating it. :ref:`Filtering <filtering-conventions>` maps the modes and the
          anchor onto scipy and OpenCV. With ``padding='same'``, ``'reflect'`` needs each axis longer than
          ``k // 2`` for a kernel ``k`` taps long along it, and ``'circular'`` at least that long; a shorter axis
          raises.
        - A :math:`(1, kH, kW)` kernel is shared by the whole batch, and a :math:`(B, kH, kW)` kernel gives each
          sample its own, shared by the sample's channels; there are no per-channel kernels.
        - ``normalized=True`` divides each kernel by the sum of its absolute values, so a zero-sum derivative kernel
          keeps its sign.
        - The kernel is cast to the input's dtype and device and stays differentiable; the output has the input's
          dtype.
        - Known defects:

          - a kernel batch that divides the input batch without matching it is not rejected: with 2 kernels for 4
            samples, sample ``i`` is filtered with kernel ``i % 2``
            (`#5154 <https://github.com/kornia/kornia/issues/5154>`_).
          - an integer input casts the kernel to its dtype, so a fractional kernel truncates to 0: a uint8 image
            filtered with a box kernel comes back as zeros, or the call raises where torch has no integer
            convolution (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).
          - ``padding`` and ``border_type`` are checked case-insensitively but used as given: ``padding='SAME'``
            returns the ``'valid'`` output and ``border_type='REFLECT'`` raises
            (`#5156 <https://github.com/kornia/kornia/issues/5156>`_).

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.
        behaviour: defines the convolution mode -- correlation (default), using pytorch conv2d,
          or true convolution (kernel is flipped). 2 modes available ``'corr'`` or ``'conv'``.

    Return:
        the filtered tensor. With ``padding='same'`` it has the input's shape :math:`(B, C, H, W)`. With
        ``padding='valid'`` it has shape :math:`(B, C, H - kH + 1, W - kW + 1)`, and its pixel ``(0, 0)`` is the
        ``'same'`` output at the anchor ``(a_y, a_x)``.

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
        f"Invalid padding mode, {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )
    # prepare kernel
    b, c, h, w = input.shape
    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(device=input.device, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape: list[int] = _compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def filter2d_separable(
    input: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Correlate a tensor with two 1d kernels, one along x and one along y.

    The function applies the given kernels to a tensor. They are applied
    independently at each channel of the tensor. With ``padding='same'`` the
    input is first padded according to ``border_type``, so that the output
    keeps the input's height and width.

    Convention:
        See the Convention block on :func:`~kornia.filters.filter2d`. The result equals
        :func:`~kornia.filters.filter2d` with the outer-product kernel ``kernel_y[:, :, None] * kernel_x[:, None, :]``,
        the even-size anchor included: ``kernel_x`` runs along ``W`` and comes first. There is no ``behaviour``
        argument, so the kernels are always correlated. ``normalized=True`` normalizes each 1d kernel.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel_x: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kW)` or :math:`(B, kW)`.
        kernel_y: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH)` or :math:`(B, kH)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        the filtered tensor, of the shape :func:`~kornia.filters.filter2d` returns for the same ``padding``:
        :math:`(B, C, H, W)` with ``padding='same'``.

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
    r"""Filter a tensor with a 3d kernel, by cross-correlation unless ``behaviour='conv'``.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Convention:
        - See the Convention block on :func:`~kornia.filters.filter2d`, applied to the ``(D, H, W)`` axes with the
          anchor ``((kD - 1) // 2, (kH - 1) // 2, (kW - 1) // 2)``.
        - The default ``border_type`` is ``'replicate'``, where :func:`~kornia.filters.filter2d` defaults to
          ``'reflect'``. There is no ``padding`` argument: the output always has the input's shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, D, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD, kH, kW)`  or :math:`(B, kD, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        behaviour: defines the convolution mode -- correlation (default), using pytorch conv3d,
            or true convolution (kernel is flipped). The expected values are: ``'corr'``, ``'conv'``.

    Return:
        the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, D, H, W)`.

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

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, gotcha {border_type}. Expected one of {_VALID_BORDERS}",
    )

    KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f"Invalid behaviour mode, gotcha {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )

    # prepare kernel
    b, c, d, h, w = input.shape
    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-3, -2, -1))[:, None, ...].to(device=input.device, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.reshape(bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: list[int] = _compute_padding([depth, height, width])
    input_pad = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

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
    r"""Apply a 2D convolution (or correlation) using an FFT-based backend.

    This function applies a spatial kernel to a batched tensor using the
    convolution theorem, i.e., convolution in the spatial domain is performed
    as element-wise multiplication in the frequency domain.

    The kernel is applied independently to each channel of the input tensor.
    Depending on the selected padding mode, the output can either preserve
    the input spatial resolution (`'same'`) or return only the valid region
    (`'valid'`). Boundary handling is performed in the spatial domain prior
    to the FFT.

    Unlike :func:`~kornia.filters.filter2d`'s, its cost barely grows with the
    kernel size, so it pays off only for large kernels; the crossover depends
    on the device and the image size.

    Convention:
        - See the Convention block on :func:`~kornia.filters.filter2d`: for the same arguments and a finite input
          ``fft_conv`` returns the same result, to roundoff. A NaN or inf in a channel of the input makes that
          channel's whole output non-finite, where :func:`~kornia.filters.filter2d` keeps it to the pixels whose
          window reaches it.
        - Known defects:

          - one input sample with a batch of kernels is broadcast, one output per kernel, where
            :func:`~kornia.filters.filter2d` raises (`#5154 <https://github.com/kornia/kornia/issues/5154>`_).
          - an integer input truncates a fractional kernel to 0, as in :func:`~kornia.filters.filter2d`, and the
            result is in torch's default floating dtype (float32) instead of the input's
            (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).
          - with ``padding='valid'`` and a kernel taller or wider than the input, it returns a wrongly sized tensor
            where :func:`~kornia.filters.filter2d` raises (`#5285 <https://github.com/kornia/kornia/issues/5285>`_).

    Args:
        input: Input tensor of shape :math:`(B, C, H, W)`.
        kernel: Convolution kernel of shape :math:`(1, kH, kW)`, shared by the
            whole batch, or :math:`(B, kH, kW)`, where each batch element
            provides one kernel, which is shared across all channels of the
            corresponding input batch.
        border_type: Padding mode applied to the input before convolution.
            Supported values are ``'constant'``, ``'reflect'``,
            ``'replicate'``, and ``'circular'``.
        normalized: If ``True``, the kernel is L1-normalized before applying
            the convolution.
        padding: Padding strategy to use. Supported values are:
            ``'same'`` (output has the same spatial size as the input) or
            ``'valid'`` (no implicit padding).
        behaviour: Convolution mode. If ``'corr'`` (default), performs
            cross-correlation. If ``'conv'``, performs true convolution
            by flipping the kernel spatially.

    Returns:
        Tensor: The filtered tensor. If ``padding='same'``, the output shape
        is :math:`(B, C, H, W)`. If ``padding='valid'``, the output shape is
        :math:`(B, C, H - kH + 1, W - kW + 1)`.

    Note:
        - Internally, the function performs zero-padding of the kernel to
          match the input size and uses real-valued FFTs (`rfftn` / `irfftn`).
        - CPU float16 and bfloat16 inputs use float32 FFTs, with the result
          cast back to the input dtype.
        - This implementation computes linear convolution via FFT by
          appropriate spatial padding and cropping, avoiding circular
          convolution artifacts.
        - No stride or dilation is supported.

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

    _, c, _, _ = input.shape
    kh, kw = kernel.shape[-2:]

    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(device=input.device, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    # Expand kernel across channels
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # Padding (spatial domain)
    if padding == "same":
        padding_shape = _compute_padding([kh, kw])
        input_padded = F.pad(input, padding_shape, mode=border_type)
    else:
        input_padded = input

    padded_h, padded_w = input_padded.shape[-2:]

    input_padded = input_padded.contiguous()
    tmp_kernel = tmp_kernel.contiguous()

    # CPU FFT kernels do not support float16 or bfloat16.
    is_cpu_half = input.device.type == "cpu" and input.dtype in (torch.float16, torch.bfloat16)
    if is_cpu_half:
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

    return output.to(input.dtype) if is_cpu_half else output


def correlate2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Correlate a tensor with a 2d kernel.

    Convenience alias for :func:`filter2d` with ``behaviour='corr'`` (cross-correlation).
    See :func:`filter2d` for full documentation.

    .. seealso:: :func:`convolve2d`, :func:`filter2d`

    Args:
        input: the input tensor with shape of :math:`(B, C, H, W)`.
        kernel: the kernel to be correlated with the input tensor.
            The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding. 2 modes available ``'same'`` or ``'valid'``.

    Return:
        the correlated tensor. With ``padding='same'`` it has the input's shape :math:`(B, C, H, W)`. With
        ``padding='valid'`` it has shape :math:`(B, C, H - kH + 1, W - kW + 1)`.

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
    r"""Convolve a tensor with a 2d kernel using true convolution.

    Convenience alias for :func:`filter2d` with ``behaviour='conv'`` (true convolution,
    where the kernel is flipped along both spatial axes, ``H`` and ``W``, before applying).
    See :func:`filter2d` for full documentation.

    .. seealso:: :func:`correlate2d`, :func:`filter2d`

    Args:
        input: the input tensor with shape of :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input tensor.
            The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding. 2 modes available ``'same'`` or ``'valid'``.

    Return:
        the convolved tensor. With ``padding='same'`` it has the input's shape :math:`(B, C, H, W)`. With
        ``padding='valid'`` it has shape :math:`(B, C, H - kH + 1, W - kW + 1)`.

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
    r"""Correlate a tensor with a 3d kernel.

    Convenience alias for :func:`filter3d` with ``behaviour='corr'`` (cross-correlation).
    See :func:`filter3d` for full documentation.

    .. seealso:: :func:`convolve3d`, :func:`filter3d`

    Args:
        input: the input tensor with shape of :math:`(B, C, D, H, W)`.
        kernel: the kernel to be correlated with the input tensor.
            The kernel shape must be :math:`(1, kD, kH, kW)` or :math:`(B, kD, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.

    Return:
        the correlated tensor, with the input's shape :math:`(B, C, D, H, W)`.

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
    r"""Convolve a tensor with a 3d kernel using true convolution.

    Convenience alias for :func:`filter3d` with ``behaviour='conv'`` (true convolution,
    where the kernel is flipped along all three axes, ``D``, ``H`` and ``W``, before applying).
    See :func:`filter3d` for full documentation.

    .. seealso:: :func:`correlate3d`, :func:`filter3d`

    Args:
        input: the input tensor with shape of :math:`(B, C, D, H, W)`.
        kernel: the kernel to be convolved with the input tensor.
            The kernel shape must be :math:`(1, kD, kH, kW)` or :math:`(B, kD, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.

    Return:
        the convolved tensor, with the input's shape :math:`(B, C, D, H, W)`.

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
