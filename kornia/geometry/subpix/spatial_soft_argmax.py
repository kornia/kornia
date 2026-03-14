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

from kornia.filters.sobel import spatial_gradient3d
from kornia.geometry.conversions import normalize_pixel_coordinates, normalize_pixel_coordinates3d
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d

from .dsnt import spatial_expectation2d, spatial_softmax2d
from .nms import nms3d


def _get_window_grid_kernel2d(h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
    r"""Generate a kernel to with window coordinates, residual to window center.

    Args:
         h: kernel height.
         w: kernel width.
         device: device, on which generate.

    Returns:
        conv_kernel [2x1xhxw]

    """
    if device is None:
        device = torch.device("cpu")
    window_grid2d = create_meshgrid(h, w, False, device=device)
    window_grid2d = normalize_pixel_coordinates(window_grid2d, h, w)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2)
    return conv_kernel


def _get_center_kernel2d(h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
    r"""Generate a kernel to return center coordinates, when applied with F.conv2d to 2d coordinates grid.

    Args:
        h: kernel height.
        w: kernel width.
        device: device, on which generate.

    Returns:
        conv_kernel [2x2xhxw].

    """
    if device is None:
        device = torch.device("cpu")
    center_kernel = torch.zeros(2, 2, h, w, device=device)

    #  If the size is odd, we have one pixel for center, if even - 2
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = (h // 2) + 1
    else:
        h_i1 = (h // 2) - 1
        h_i2 = (h // 2) + 1
    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = (w // 2) + 1
    else:
        w_i1 = (w // 2) - 1
        w_i2 = (w // 2) + 1
    center_kernel[(0, 1), (0, 1), h_i1:h_i2, w_i1:w_i2] = 1.0 / float((h_i2 - h_i1) * (w_i2 - w_i1))
    return center_kernel


def _get_center_kernel3d(d: int, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
    r"""Generate a kernel to return center coordinates, when applied with F.conv2d to 3d coordinates grid.

    Args:
        d: kernel depth.
        h: kernel height.
        w: kernel width.
        device: device, on which generate.

    Returns:
        conv_kernel [3x3xdxhxw].

    """
    if device is None:
        torch.device("cpu")
    center_kernel = torch.zeros(3, 3, d, h, w, device=device)
    #  If the size is odd, we have one pixel for center, if even - 2
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = (h // 2) + 1
    else:
        h_i1 = (h // 2) - 1
        h_i2 = (h // 2) + 1
    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = (w // 2) + 1
    else:
        w_i1 = (w // 2) - 1
        w_i2 = (w // 2) + 1
    if d % 2 != 0:
        d_i1 = d // 2
        d_i2 = (d // 2) + 1
    else:
        d_i1 = (d // 2) - 1
        d_i2 = (d // 2) + 1
    center_num = float((h_i2 - h_i1) * (w_i2 - w_i1) * (d_i2 - d_i1))
    center_kernel[(0, 1, 2), (0, 1, 2), d_i1:d_i2, h_i1:h_i2, w_i1:w_i2] = 1.0 / center_num
    return center_kernel


def _get_window_grid_kernel3d(d: int, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
    r"""Generate a kernel to return coordinates, residual to window center.

    Args:
        d: kernel depth.
        h: kernel height.
        w: kernel width.
        device: device, on which generate.

    Returns:
        conv_kernel [3x1xdxhxw]

    """
    if device is None:
        device = torch.device("cpu")
    grid2d = create_meshgrid(h, w, True, device=device)
    if d > 1:
        z = torch.linspace(-1, 1, d, device=device).view(d, 1, 1, 1)
    else:  # only onr channel with index == 0
        z = torch.zeros(1, 1, 1, 1, device=device)
    grid3d = torch.cat([z.repeat(1, h, w, 1).contiguous(), grid2d.repeat(d, 1, 1, 1)], 3)
    conv_kernel = grid3d.permute(3, 0, 1, 2).unsqueeze(1)
    return conv_kernel


class ConvSoftArgmax2d(nn.Module):
    r"""nn.Module that calculates soft argmax 2d per window.

    See
    :func: `~kornia.geometry.subpix.conv_soft_argmax2d` for details.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
        temperature: torch.Tensor | float = 1.0,
        normalized_coordinates: bool = True,
        eps: float = 1e-8,
        output_value: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"temperature={self.temperature}, "
            f"normalized_coordinates={self.normalized_coordinates}, "
            f"eps={self.eps}, "
            f"output_value={self.output_value})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return conv_soft_argmax2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.temperature,
            self.normalized_coordinates,
            self.eps,
            self.output_value,
        )


class ConvSoftArgmax3d(nn.Module):
    r"""nn.Module that calculates soft argmax 3d per window.

    See
    :func: `~kornia.geometry.subpix.conv_soft_argmax3d` for details.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (1, 1, 1),
        temperature: torch.Tensor | float = 1.0,
        normalized_coordinates: bool = False,
        eps: float = 1e-8,
        output_value: bool = True,
        strict_maxima_bonus: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value
        self.strict_maxima_bonus = strict_maxima_bonus

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"temperature={self.temperature}, "
            f"normalized_coordinates={self.normalized_coordinates}, "
            f"eps={self.eps}, "
            f"strict_maxima_bonus={self.strict_maxima_bonus}, "
            f"output_value={self.output_value})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return conv_soft_argmax3d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.temperature,
            self.normalized_coordinates,
            self.eps,
            self.output_value,
            self.strict_maxima_bonus,
        )


def conv_soft_argmax2d(
    input: torch.Tensor,
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (1, 1),
    temperature: torch.Tensor | float = 1.0,
    normalized_coordinates: bool = True,
    eps: float = 1e-8,
    output_value: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the convolutional spatial Soft-Argmax 2D over the windows of a given heatmap.

    .. math::
        ij(X) = \frac{\sum{(i,j)} * exp(x / T)  \in X} {\sum{exp(x / T)  \in X}}

    .. math::
        val(X) = \frac{\sum{x * exp(x / T)  \in X}} {\sum{exp(x / T)  \in X}}

    where :math:`T` is temperature.

    Args:
        input: the given heatmap with shape :math:`(N, C, H_{in}, W_{in})`.
        kernel_size: the size of the window.
        stride: the stride of the window.
        padding: input zero padding.
        temperature: factor to apply to input.
        normalized_coordinates: whether to return the coordinates normalized in the range of :math:`[-1, 1]`.
            Otherwise, it will return the coordinates in the range of the input shape.
        eps: small value to avoid zero division.
        output_value: if True, val is output, if False, only ij.

    Returns:
        Function has two outputs - argmax coordinates and the softmaxpooled heatmap values themselves.
        On each window, the function computed returns with shapes :math:`(N, C, 2, H_{out},
        W_{out})`, :math:`(N, C, H_{out}, W_{out})`,

        where

         .. math::
             H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
               (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

         .. math::
             W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
               (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Examples:
        >>> input = torch.randn(20, 16, 50, 32)
        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3,3), (2,2), (1,1), output_value=True)

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if temperature <= 0:
        raise ValueError(f"Temperature should be positive float or torch.Tensor. Got: {temperature}")

    b, c, h, w = input.shape
    ky, kx = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, h, w)

    center_kernel: torch.Tensor = _get_center_kernel2d(ky, kx, device).to(dtype)
    window_kernel: torch.Tensor = _get_window_grid_kernel2d(ky, kx, device).to(dtype)

    # applies exponential normalization trick
    # https://timvieira.github.io/blog/post/2014/02/11/exp-F.normalize-trick/
    # https://github.com/pytorch/pytorch/blob/bcb0bb7e0e03b386ad837015faba6b4b16e3bfb9/aten/src/ATen/native/SoftMax.cpp#L44
    x_max = F.adaptive_max_pool2d(input, (1, 1))

    # max is detached to prevent undesired backprop loops in the graph
    x_exp = ((input - x_max.detach()) / temperature).exp()

    # F.avg_pool2d(.., divisor_override = 1.0) - proper way for sum pool in PyTorch 1.2.
    # Not available yet in version 1.0, so let's do manually
    pool_coef: float = float(kx * ky)

    # F.softmax denominator
    den = pool_coef * F.avg_pool2d(x_exp, kernel_size, stride=stride, padding=padding) + eps

    x_softmaxpool = pool_coef * F.avg_pool2d(x_exp * input, kernel_size, stride=stride, padding=padding) / den
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3))

    # We need to output also coordinates
    # Pooled window center coordinates
    grid_global: torch.Tensor = create_meshgrid(h, w, False, device).to(dtype).permute(0, 3, 1, 2)

    grid_global_pooled = F.conv2d(grid_global, center_kernel, stride=stride, padding=padding)

    # Coordinates of maxima residual to window center
    # prepare kernel
    coords_max: torch.Tensor = F.conv2d(x_exp, window_kernel, stride=stride, padding=padding)

    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)
    # [:,:, 0, ...] is x
    # [:,:, 1, ...] is y

    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates(coords_max.permute(0, 2, 3, 1), h, w)
        coords_max = coords_max.permute(0, 3, 1, 2)

    # Back B*C -> (b, c)
    coords_max = coords_max.view(b, c, 2, coords_max.size(2), coords_max.size(3))

    if output_value:
        return coords_max, x_softmaxpool
    return coords_max


def conv_soft_argmax3d(
    input: torch.Tensor,
    kernel_size: tuple[int, int, int] = (3, 3, 3),
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (1, 1, 1),
    temperature: torch.Tensor | float = 1.0,
    normalized_coordinates: bool = False,
    eps: float = 1e-8,
    output_value: bool = True,
    strict_maxima_bonus: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the convolutional spatial Soft-Argmax 3D over the windows of a given heatmap.

    .. math::
             ijk(X) = \frac{\sum{(i,j,k)} * exp(x / T)  \in X} {\sum{exp(x / T)  \in X}}

    .. math::
             val(X) = \frac{\sum{x * exp(x / T)  \in X}} {\sum{exp(x / T)  \in X}}

    where ``T`` is temperature.

    Args:
        input: the given heatmap with shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
        kernel_size:  size of the window.
        stride: stride of the window.
        padding: input zero padding.
        temperature: factor to apply to input.
        normalized_coordinates: whether to return the coordinates normalized in the range of :math:[-1, 1]`.
            Otherwise, it will return the coordinates in the range of the input shape.
        eps: small value to avoid zero division.
        output_value: if True, val is output, if False, only ij.
        strict_maxima_bonus: pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
          This is needed for mimic behavior of strict NMS in classic local features

    Returns:
        Function has two outputs - argmax coordinates and the softmaxpooled heatmap values themselves.
        On each window, the function computed returns with shapes :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`,
        :math:`(N, C, D_{out}, H_{out}, W_{out})`,

        where

         .. math::
             D_{out} = \left\lfloor\frac{D_{in}  + 2 \times \text{padding}[0] -
             (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

         .. math::
             H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[1] -
             (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

         .. math::
             W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[2] -
             (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples:
        >>> input = torch.randn(20, 16, 3, 50, 32)
        >>> nms_coords, nms_val = conv_soft_argmax3d(input, (3, 3, 3), (1, 2, 2), (0, 1, 1))

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}")

    if temperature <= 0:
        raise ValueError(f"Temperature should be positive float or torch.Tensor. Got: {temperature}")

    b, c, d, h, w = input.shape
    kz, ky, kx = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, d, h, w)

    center_kernel: torch.Tensor = _get_center_kernel3d(kz, ky, kx, device).to(dtype)
    window_kernel: torch.Tensor = _get_window_grid_kernel3d(kz, ky, kx, device).to(dtype)

    # applies exponential normalization trick
    # https://timvieira.github.io/blog/post/2014/02/11/exp-F.normalize-trick/
    # https://github.com/pytorch/pytorch/blob/bcb0bb7e0e03b386ad837015faba6b4b16e3bfb9/aten/src/ATen/native/SoftMax.cpp#L44
    x_max = F.adaptive_max_pool3d(input, (1, 1, 1))

    # max is detached to prevent undesired backprop loops in the graph
    x_exp = ((input - x_max.detach()) / temperature).exp()

    pool_coef: float = float(kx * ky * kz)

    # F.softmax denominator
    den = pool_coef * F.avg_pool3d(x_exp.view_as(input), kernel_size, stride=stride, padding=padding) + eps

    # We need to output also coordinates
    # Pooled window center coordinates
    grid_global: torch.Tensor = create_meshgrid3d(d, h, w, False, device=device).to(dtype).permute(0, 4, 1, 2, 3)

    grid_global_pooled = F.conv3d(grid_global, center_kernel, stride=stride, padding=padding)

    # Coordinates of maxima residual to window center
    # prepare kernel
    coords_max: torch.Tensor = F.conv3d(x_exp, window_kernel, stride=stride, padding=padding)

    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)
    # [:,:, 0, ...] is depth (scale)
    # [:,:, 1, ...] is x
    # [:,:, 2, ...] is y

    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates3d(coords_max.permute(0, 2, 3, 4, 1), d, h, w)
        coords_max = coords_max.permute(0, 4, 1, 2, 3)

    # Back B*C -> (b, c)
    coords_max = coords_max.view(b, c, 3, coords_max.size(2), coords_max.size(3), coords_max.size(4))

    if not output_value:
        return coords_max

    x_softmaxpool = (
        pool_coef * F.avg_pool3d(x_exp.view(input.size()) * input, kernel_size, stride=stride, padding=padding) / den
    )
    if strict_maxima_bonus > 0:
        in_levels: int = input.size(2)
        out_levels: int = x_softmaxpool.size(2)
        skip_levels: int = (in_levels - out_levels) // 2
        strict_maxima: torch.Tensor = F.avg_pool3d(nms3d(input, kernel_size), 1, stride, 0)
        strict_maxima = strict_maxima[:, :, skip_levels : out_levels - skip_levels]
        x_softmaxpool *= 1.0 + strict_maxima_bonus * strict_maxima
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3), x_softmaxpool.size(4))
    return coords_max, x_softmaxpool


def spatial_soft_argmax2d(
    input: torch.Tensor, temperature: Optional[torch.Tensor] = None, normalized_coordinates: bool = True
) -> torch.Tensor:
    r"""Compute the Spatial Soft-Argmax 2D of a given input heatmap.

    Args:
        input: the given heatmap with shape :math:`(B, N, H, W)`.
        temperature: factor to apply to input.
        normalized_coordinates: whether to return the coordinates normalized in the range of :math:`[-1, 1]`.
            Otherwise, it will return the coordinates in the range of the input shape.

    Returns:
        the index of the maximum 2d coordinates of the give map :math:`(B, N, 2)`.
        The output order is x-coord and y-coord.

    Examples:
        >>> input = torch.tensor([[[
        ... [0., 0., 0.],
        ... [0., 10., 0.],
        ... [0., 0., 0.]]]])
        >>> spatial_soft_argmax2d(input, normalized_coordinates=False)
        tensor([[[1.0000, 1.0000]]])

    """
    if temperature is None:
        temperature = torch.tensor(1.0)
    input_soft: torch.Tensor = spatial_softmax2d(input, temperature)
    output: torch.Tensor = spatial_expectation2d(input_soft, normalized_coordinates)
    return output


class SpatialSoftArgmax2d(nn.Module):
    r"""Compute the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.geometry.subpix.spatial_soft_argmax2d` for details.
    """

    def __init__(self, temperature: Optional[torch.Tensor] = None, normalized_coordinates: bool = True) -> None:
        super().__init__()
        if temperature is None:
            temperature = torch.tensor(1.0)
        self.temperature: torch.Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"temperature={self.temperature}, "
            f"normalized_coordinates={self.normalized_coordinates})"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return spatial_soft_argmax2d(input, self.temperature, self.normalized_coordinates)


def _solve_cramer_sym3x3(
    dxx: torch.Tensor,
    dyy: torch.Tensor,
    dss: torch.Tensor,
    dxy: torch.Tensor,
    dxs: torch.Tensor,
    dys: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve H * [sx, sy, ss]^T = [r0, r1, r2]^T via Cramer's rule.

    H is symmetric: H = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]].
    All inputs are batched 1-D tensors of length N.

    Returns:
        (sx, sy, ss, solved) where ``solved`` is a bool mask for non-singular systems.
        Entries where ``solved`` is False are set to zero.
    """
    cf00 = dyy * dss - dys * dys  # cofactor M00
    cf01 = dxy * dss - dys * dxs  # cofactor M01
    cf02 = dxy * dys - dyy * dxs  # cofactor M02
    det = dxx * cf00 - dxy * cf01 + dxs * cf02
    solved = det.abs() > 0.0
    # Avoid division by zero for singular cases; results are discarded via the solved mask.
    safe_det = torch.where(solved, det, torch.ones_like(det))
    sx = (r0 * cf00 - dxy * (r1 * dss - dys * r2) + dxs * (r1 * dys - dyy * r2)) / safe_det
    sy = (dxx * (r1 * dss - dys * r2) - r0 * cf01 + dxs * (dxy * r2 - r1 * dxs)) / safe_det
    ss = (dxx * (dyy * r2 - r1 * dys) - dxy * (dxy * r2 - r1 * dxs) + r0 * cf02) / safe_det
    return sx, sy, ss, solved


def iterative_quad_interp3d(
    input: torch.Tensor,
    n_iters: int = 5,
    strict_maxima_bonus: float = 10.0,
    max_subpixel_shift: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Iterative subpixel localization of 3D extrema via quadratic interpolation.

    Unlike :func:`conv_quad_interp3d`, which performs a single-pass sliding-window
    solve using spatial gradient convolutions, this function explicitly extracts the
    :math:`3 \times 3 \times 3` patch at each NMS maximum and iterates up to
    ``n_iters`` times. When the estimated subpixel shift along any spatial or scale
    axis exceeds ``max_subpixel_shift`` the integer center is moved one step in that
    direction and the solve is repeated—matching the localization loop from the
    HessAff / SIFT family of detectors.

    Args:
        input: response pyramid with shape :math:`(B, C, D, H, W)`.
        n_iters: maximum number of localization iterations per keypoint.
        strict_maxima_bonus: value added to ``y_max`` at NMS-maximum positions so
            that strict maxima are preferred when selecting the top-K keypoints.
        max_subpixel_shift: if the estimated shift along any axis is larger than this
            threshold the integer center is displaced and another iteration is run.

    Returns:
        A tuple ``(coords_max, y_max)`` where

        * ``coords_max`` has shape :math:`(B, C, 3, D, H, W)` and stores the refined
          coordinates ``[scale, x, y]`` for every position in the input.
          Non-NMS positions keep their original grid coordinates.
        * ``y_max`` has shape :math:`(B, C, D, H, W)` and stores the quadratically
          corrected response values (with the optional strict-maxima bonus added).

    Example:
        >>> input = torch.randn(2, 3, 3, 8, 8)
        >>> coords, vals = iterative_quad_interp3d(input, n_iters=5)
        >>> coords.shape
        torch.Size([2, 3, 3, 3, 8, 8])
        >>> vals.shape
        torch.Size([2, 3, 3, 8, 8])

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    if input.ndim != 5:
        raise ValueError(f"Invalid input shape, expected BxCxDxHxW. Got: {input.shape}")

    B, C, D, H, W = input.shape
    device = input.device
    dtype = input.dtype

    # Build coords_max with the same coordinate ordering as conv_quad_interp3d:
    # dim-2: [scale(z), x(width), y(height)].
    # Use empty + broadcast-fill: 23x faster than expand(...).clone() on large tensors.
    coords_max = torch.empty(B, C, 3, D, H, W, device=device, dtype=dtype)
    coords_max[:, :, 0] = torch.arange(D, device=device, dtype=dtype).view(D, 1, 1)
    coords_max[:, :, 1] = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    coords_max[:, :, 2] = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    y_max = input.clone()

    if D < 3 or H < 3 or W < 3:
        return coords_max, y_max

    # Flatten B and C into a single leading dimension for indexing convenience.
    inp = input.view(B * C, D, H, W)
    DHW = D * H * W
    HW = H * W

    nms_mask = nms3d(input, (3, 3, 3), True)  # (B, C, D, H, W) bool
    nms_flat = nms_mask.view(B * C, D, H, W)

    bc_idx, d_idx, h_idx, w_idx = torch.where(nms_flat)
    N = bc_idx.shape[0]
    if N == 0:
        return coords_max, y_max

    # Pre-compute flat offsets for gathering a full 3×3×3 neighborhood in one shot.
    # Layout: patch[k] = inp[bc, d+dd, h+dh, w+dw] where
    #   k = (dd+1)*9 + (dh+1)*3 + (dw+1), center k=13.
    _dd = torch.tensor(
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        device=device,
        dtype=torch.long,
    )
    _dh = torch.tensor(
        [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1],
        device=device,
        dtype=torch.long,
    )
    _dw = torch.tensor(
        [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
        device=device,
        dtype=torch.long,
    )
    patch_offsets = _dd * HW + _dh * W + _dw  # (27,)

    # Integer center positions — updated across iterations.
    d_cur = d_idx.clone()
    h_cur = h_idx.clone()
    w_cur = w_idx.clone()

    # Track which keypoints remain valid (in bounds, system solvable).
    valid = torch.ones(N, dtype=torch.bool, device=device)

    # Final subpixel offsets relative to the (possibly moved) integer center.
    shift_x = torch.zeros(N, device=device, dtype=dtype)
    shift_y = torch.zeros(N, device=device, dtype=dtype)
    shift_s = torch.zeros(N, device=device, dtype=dtype)
    grad_dot_shift = torch.zeros(N, device=device, dtype=dtype)

    inp_flat = inp.reshape(-1)  # flat view for gather indexing
    bc_base = bc_idx * DHW  # per-keypoint base offset (constant across iterations)

    for _ in range(n_iters):
        # Clamp so we can always safely access ±1 neighbours.
        d_s = d_cur.clamp(1, D - 2)
        h_s = h_cur.clamp(1, H - 2)
        w_s = w_cur.clamp(1, W - 2)

        # Gather full 3×3×3 patch in a single vectorized op: (N, 27)
        patch = inp_flat[(bc_base + d_s * HW + h_s * W + w_s).unsqueeze(1) + patch_offsets.unsqueeze(0)]

        # Unpack patch neighbors by name.  Flat index: k = (dd+1)*9 + (dh+1)*3 + (dw+1)
        # where dd/dh/dw are the offsets in scale/height/width, each in {-1, 0, +1}.
        # Naming convention: xm/xp = width-1/+1, ym/yp = height-1/+1, sm/sp = scale-1/+1.
        c000 = patch[:, 13]  # center (dd=0, dh=0, dw=0)
        # axis neighbors
        p_xm  = patch[:, 12]  # (dd= 0, dh= 0, dw=-1)
        p_xp  = patch[:, 14]  # (dd= 0, dh= 0, dw=+1)
        p_ym  = patch[:, 10]  # (dd= 0, dh=-1, dw= 0)
        p_yp  = patch[:, 16]  # (dd= 0, dh=+1, dw= 0)
        p_sm  = patch[:,  4]  # (dd=-1, dh= 0, dw= 0)
        p_sp  = patch[:, 22]  # (dd=+1, dh= 0, dw= 0)
        # diagonal neighbors used for mixed partials
        p_xm_ym = patch[:,  9]  # (dd= 0, dh=-1, dw=-1)
        p_xp_ym = patch[:, 11]  # (dd= 0, dh=-1, dw=+1)
        p_xm_yp = patch[:, 15]  # (dd= 0, dh=+1, dw=-1)
        p_xp_yp = patch[:, 17]  # (dd= 0, dh=+1, dw=+1)
        p_xm_sm = patch[:,  3]  # (dd=-1, dh= 0, dw=-1)
        p_xp_sm = patch[:,  5]  # (dd=-1, dh= 0, dw=+1)
        p_xm_sp = patch[:, 21]  # (dd=+1, dh= 0, dw=-1)
        p_xp_sp = patch[:, 23]  # (dd=+1, dh= 0, dw=+1)
        p_ym_sm = patch[:,  1]  # (dd=-1, dh=-1, dw= 0)
        p_yp_sm = patch[:,  7]  # (dd=-1, dh=+1, dw= 0)
        p_ym_sp = patch[:, 19]  # (dd=+1, dh=-1, dw= 0)
        p_yp_sp = patch[:, 25]  # (dd=+1, dh=+1, dw= 0)

        # First-order finite differences (0.5 * (next - prev)), matches C++ convention.
        gx = 0.5 * (p_xp - p_xm)  # x/width direction
        gy = 0.5 * (p_yp - p_ym)  # y/height direction
        gs = 0.5 * (p_sp - p_sm)  # scale direction

        # Second-order finite differences.
        dxx = p_xp - 2.0 * c000 + p_xm
        dyy = p_yp - 2.0 * c000 + p_ym
        dss = p_sp - 2.0 * c000 + p_sm
        dxy = 0.25 * (p_xp_yp - p_xm_yp - p_xp_ym + p_xm_ym)
        dxs = 0.25 * (p_xp_sp - p_xm_sp - p_xp_sm + p_xm_sm)
        dys = 0.25 * (p_yp_sp - p_ym_sp - p_yp_sm + p_ym_sm)

        sx, sy, ss, solved = _solve_cramer_sym3x3(dxx, dyy, dss, dxy, dxs, dys, -gx, -gy, -gs)
        # Use non-inplace ops so that tensors saved by torch.where for autograd are not mutated.
        valid = valid & solved

        # Zero out invalid solutions to avoid NaN propagation.
        valid_f = valid.to(dtype)
        sx = sx * valid_f
        sy = sy * valid_f
        ss = ss * valid_f

        # Save best estimates for valid keypoints.
        shift_x = torch.where(valid, sx, shift_x)
        shift_y = torch.where(valid, sy, shift_y)
        shift_s = torch.where(valid, ss, shift_s)
        grad_dot_shift = torch.where(valid, gx * sx + gy * sy + gs * ss, grad_dot_shift)

        # Determine which axes need an integer step and move the center.
        move_pos_x = valid & (sx > max_subpixel_shift)
        move_neg_x = valid & (sx < -max_subpixel_shift)
        new_w = w_cur + move_pos_x.long() - move_neg_x.long()
        valid = valid & (new_w >= 1) & (new_w <= W - 2)
        w_cur = new_w.clamp(0, W - 1)

        move_pos_y = valid & (sy > max_subpixel_shift)
        move_neg_y = valid & (sy < -max_subpixel_shift)
        new_h = h_cur + move_pos_y.long() - move_neg_y.long()
        valid = valid & (new_h >= 1) & (new_h <= H - 2)
        h_cur = new_h.clamp(0, H - 1)

        move_pos_s = valid & (ss > max_subpixel_shift)
        move_neg_s = valid & (ss < -max_subpixel_shift)
        new_d = d_cur + move_pos_s.long() - move_neg_s.long()
        valid = valid & (new_d >= 1) & (new_d <= D - 2)
        d_cur = new_d.clamp(0, D - 1)

    # Invalidate keypoints whose final shift is still too large (C++ finalThreshold check).
    valid = valid & (shift_x.abs() <= 1.5) & (shift_y.abs() <= 1.5) & (shift_s.abs() <= 1.5)

    # --- Write refined coordinates into the dense output map ----------------
    # coords_max layout: dim-2 = [scale, x(width), y(height)] — same as conv_quad_interp3d.
    b_idx = bc_idx // C
    c_idx = bc_idx % C

    # For valid keypoints: integer center + subpixel offset.
    # For invalid keypoints: fall back to original NMS position.
    final_s = torch.where(valid, d_cur.to(dtype) + shift_s, d_idx.to(dtype))
    final_x = torch.where(valid, w_cur.to(dtype) + shift_x, w_idx.to(dtype))
    final_y = torch.where(valid, h_cur.to(dtype) + shift_y, h_idx.to(dtype))

    coords_max[b_idx, c_idx, 0, d_idx, h_idx, w_idx] = final_s
    coords_max[b_idx, c_idx, 1, d_idx, h_idx, w_idx] = final_x
    coords_max[b_idx, c_idx, 2, d_idx, h_idx, w_idx] = final_y

    # Quadratically corrected response value: val = center + 0.5 * grad^T * shift.
    val_correction = 0.5 * torch.where(valid, grad_dot_shift, torch.zeros_like(grad_dot_shift))
    val_center = inp[bc_idx, d_idx, h_idx, w_idx]
    y_max[b_idx, c_idx, d_idx, h_idx, w_idx] = val_center + val_correction

    if strict_maxima_bonus > 0:
        y_max[b_idx, c_idx, d_idx, h_idx, w_idx] += strict_maxima_bonus * valid.to(dtype)

    return coords_max, y_max


class IterativeQuadInterp3d(nn.Module):
    r"""Iterative subpixel localization of 3D extrema via quadratic interpolation.

    See :func:`~kornia.geometry.subpix.iterative_quad_interp3d` for details.
    """

    def __init__(
        self,
        n_iters: int = 5,
        strict_maxima_bonus: float = 10.0,
        max_subpixel_shift: float = 0.6,
    ) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.strict_maxima_bonus = strict_maxima_bonus
        self.max_subpixel_shift = max_subpixel_shift

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_iters={self.n_iters}, "
            f"strict_maxima_bonus={self.strict_maxima_bonus}, "
            f"max_subpixel_shift={self.max_subpixel_shift})"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return iterative_quad_interp3d(x, self.n_iters, self.strict_maxima_bonus, self.max_subpixel_shift)


def conv_quad_interp3d(
    input: torch.Tensor, strict_maxima_bonus: float = 10.0, eps: float = 1e-7
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the single iteration of quadratic interpolation of the extremum (max or min).

    Args:
        input: the given heatmap with shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
        strict_maxima_bonus: pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
          This is needed for mimic behavior of strict NMS in classic local features
        eps: parameter to control the hessian matrix ill-condition number.

    Returns:
        the location and value per each 3x3x3 window which contains strict extremum, similar to one done is SIFT.
        :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`,

        where

         .. math::
             D_{out} = \left\lfloor\frac{D_{in}  + 2 \times \text{padding}[0] -
             (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

         .. math::
             H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[1] -
             (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

         .. math::
             W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[2] -
             (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples:
        >>> input = torch.randn(20, 16, 3, 50, 32)
        >>> nms_coords, nms_val = conv_quad_interp3d(input, 1.0)

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}")

    B, CH, D, H, W = input.shape
    grid_global: torch.Tensor = create_meshgrid3d(D, H, W, False, device=input.device).permute(0, 4, 1, 2, 3)
    grid_global = grid_global.to(input.dtype)

    # to determine the location we are solving system of linear equations Ax = b, where b is 1st order gradient
    # and A is Hessian matrix
    b: torch.Tensor = spatial_gradient3d(input, order=1, mode="diff")
    b = b.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3, 1)
    A: torch.Tensor = spatial_gradient3d(input, order=2, mode="diff")
    A = A.permute(0, 1, 3, 4, 5, 2).reshape(-1, 6)
    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]
    dxy = 0.25 * A[..., 3]  # normalization to match OpenCV implementation
    dys = 0.25 * A[..., 4]  # normalization to match OpenCV implementation
    dxs = 0.25 * A[..., 5]  # normalization to match OpenCV implementation

    nms_mask: torch.Tensor = nms3d(input, (3, 3, 3), True)
    x_solved: torch.Tensor = torch.zeros_like(b)

    # Solve H * x = b for NMS points via Cramer's rule (avoids LU factorization + matrix construction).
    nms_flat = nms_mask.view(-1)
    b_nms = b[nms_flat, :, 0]  # (N_nms, 3)
    sx, sy, ss, solved_correctly = _solve_cramer_sym3x3(
        dxx[nms_flat],
        dyy[nms_flat],
        dss[nms_flat],
        dxy[nms_flat],
        dxs[nms_flat],
        dys[nms_flat],
        b_nms[:, 0],
        b_nms[:, 1],
        b_nms[:, 2],
    )
    x_solved_masked = torch.stack([sx, sy, ss], dim=-1).unsqueeze(-1)  # (N_nms, 3, 1)

    #  Kill those points, where we cannot solve
    new_nms_mask = nms_mask.masked_scatter(nms_mask, solved_correctly)

    x_solved[torch.where(new_nms_mask.view(-1, 1, 1))[0]] = x_solved_masked[solved_correctly]

    dx: torch.Tensor = -x_solved

    # Ignore torch.ones, which are far from window center
    mask1 = dx.abs().max(dim=1, keepdim=True)[0] > 0.7
    dx.masked_fill_(mask1.expand_as(dx), 0)
    dy: torch.Tensor = 0.5 * torch.bmm(b.permute(0, 2, 1), dx)
    y_max = input + dy.view(B, CH, D, H, W)
    if strict_maxima_bonus > 0:
        y_max += strict_maxima_bonus * new_nms_mask.to(input.dtype)

    dx_res: torch.Tensor = dx.flip(1).reshape(B, CH, D, H, W, 3).permute(0, 1, 5, 2, 3, 4)
    dx_res[:, :, (1, 2)] = dx_res[:, :, (2, 1)]

    coords_max: torch.Tensor = grid_global.repeat(B, 1, 1, 1, 1).unsqueeze(1)
    coords_max = coords_max + dx_res

    return coords_max, y_max


class ConvQuadInterp3d(nn.Module):
    r"""Calculate soft argmax 3d per window.

    See
    :func: `~kornia.geometry.subpix.conv_quad_interp3d` for details.
    """

    def __init__(self, strict_maxima_bonus: float = 10.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strict_maxima_bonus={self.strict_maxima_bonus})"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps)
