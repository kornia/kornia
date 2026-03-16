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

from kornia.geometry.conversions import normalize_pixel_coordinates, normalize_pixel_coordinates3d
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d

from .dsnt import spatial_expectation2d, spatial_softmax2d
from .nms import nms3d

# Flat offsets for gathering the full 3x3x3 neighbourhood of a voxel.
# Layout: patch[k] = inp[bc, d+dd, h+dh, w+dw]  where
#   k = (dd+1)*9 + (dh+1)*3 + (dw+1),  center k=13.
# Defined once at module level to avoid per-call allocation.
_PATCH_DD = torch.tensor(
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    dtype=torch.long,
)
_PATCH_DH = torch.tensor(
    [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1],
    dtype=torch.long,
)
_PATCH_DW = torch.tensor(
    [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
    dtype=torch.long,
)


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
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve H * [sx, sy, ss]^T = [r0, r1, r2]^T via Cramer's rule.

    H is symmetric: H = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]].
    All inputs are batched 1-D tensors of length N.

    Args:
        dxx: diagonal Hessian element (d²/dx²).
        dyy: diagonal Hessian element (d²/dy²).
        dss: diagonal Hessian element (d²/ds²).
        dxy: off-diagonal Hessian element (d²/dxdy).
        dxs: off-diagonal Hessian element (d²/dxds).
        dys: off-diagonal Hessian element (d²/dyds).
        r0: right-hand side component along x.
        r1: right-hand side component along y.
        r2: right-hand side component along s.
        eps: determinant magnitude below which the system is treated as singular.
            Near-singular systems can produce numerically unstable (huge) shifts.

    Returns:
        (sx, sy, ss, solved) where ``solved`` is a bool mask for well-conditioned
        systems (``|det| > eps``).  Outputs for unsolved entries are numerically
        meaningless and should be discarded by the caller.
    """
    cf00 = dyy * dss - dys * dys  # cofactor M00
    cf01 = dxy * dss - dys * dxs  # cofactor M01
    cf02 = dxy * dys - dyy * dxs  # cofactor M02
    det = dxx * cf00 - dxy * cf01 + dxs * cf02
    solved = det.abs() > eps
    # Avoid division by zero for singular/near-singular cases; outputs are discarded via solved.
    safe_det = torch.where(solved, det, torch.ones_like(det))
    sx = (r0 * cf00 - dxy * (r1 * dss - dys * r2) + dxs * (r1 * dys - dyy * r2)) / safe_det
    sy = (dxx * (r1 * dss - dys * r2) - r0 * cf01 + dxs * (dxy * r2 - r1 * dxs)) / safe_det
    ss = (dxx * (dyy * r2 - r1 * dys) - dxy * (dxy * r2 - r1 * dxs) + r0 * cf02) / safe_det
    return sx, sy, ss, solved


def conv_quad_interp3d(
    input: torch.Tensor,
    n_iters: int = 5,
    strict_maxima_bonus: float = 10.0,
    max_subpixel_shift: float = 0.6,
    precomputed_nms_mask: Optional[torch.Tensor] = None,
    dilation_radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subpixel localization of 3D scale-space extrema via quadratic interpolation.

    For each NMS maximum the function fits a 3-D quadratic to the local
    :math:`3 \times 3 \times 3` neighbourhood and solves for the sub-voxel
    shift that maximises the fit.  When the shift along any axis exceeds
    ``max_subpixel_shift`` the integer centre is moved one step in that direction
    and the solve is repeated — up to ``n_iters`` times.

    Unlike a naive iterative approach, all Hessian solves are precomputed once at
    the start for every voxel that any keypoint could possibly visit (the
    **dilated NMS neighbourhood**, an L\ :math:`\infty` ball of radius
    ``dilation_radius`` around each maximum).  The subsequent iteration loop contains
    **no data-dependent Python control flow** and no GPU→CPU synchronisation,
    making the function fully compatible with ``torch.compile`` / CUDA graphs.

    The ``dilation_radius`` controls the precompute footprint and should be set
    to the maximum number of integer-centre moves expected per keypoint.  With the
    default ``max_subpixel_shift=0.6`` almost all keypoints converge within 1 move,
    so the default ``dilation_radius=1`` (i.e. :math:`3^3 = 27` positions per
    maximum) is sufficient.  Use ``dilation_radius=2`` (:math:`5^3 = 125`) for
    extra safety.  Setting it equal to ``n_iters`` recovers the original behaviour
    but is much slower on large images.

    Args:
        input: response pyramid with shape :math:`(B, C, D, H, W)`.
        n_iters: maximum number of localization iterations per keypoint.
        strict_maxima_bonus: value added to ``y_max`` at NMS-maximum positions
            so that strict maxima are preferred during top-K selection.
        max_subpixel_shift: threshold above which the integer centre is
            moved one step and another iteration is run.
        precomputed_nms_mask: optional bool tensor of shape
            :math:`(B, C, D, H, W)` — pass the result of
            :func:`~kornia.geometry.subpix.nms3d` to skip the internal NMS call.
        dilation_radius: L\ :math:`\infty` radius (in voxels) of the neighbourhood
            around each NMS maximum where the Hessian solve is precomputed.
            Keypoints that attempt to move farther than this are marked invalid.

    Returns:
        Tuple ``(coords_max, y_max)``:

        * ``coords_max`` — shape :math:`(B, C, 3, D, H, W)`, refined
          ``[scale, x(width), y(height)]`` coordinates for each NMS maximum;
          non-maximum positions keep their grid coordinates.
        * ``y_max`` — shape :math:`(B, C, D, H, W)`, quadratically corrected
          response with optional strict-maxima bonus.

    Example:
        >>> input = torch.randn(2, 3, 5, 64, 64)
        >>> coords, vals = conv_quad_interp3d(input, n_iters=5)
        >>> coords.shape
        torch.Size([2, 3, 3, 5, 64, 64])
        >>> vals.shape
        torch.Size([2, 3, 5, 64, 64])

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    if input.ndim != 5:
        raise ValueError(f"Invalid input shape, expected BxCxDxHxW. Got: {input.shape}")

    B, C, D, H, W = input.shape
    device = input.device
    dtype = input.dtype
    BC = B * C
    DHW = D * H * W
    HW = H * W

    coords_max = torch.empty(B, C, 3, D, H, W, device=device, dtype=dtype)
    coords_max[:, :, 0] = torch.arange(D, device=device, dtype=dtype).view(D, 1, 1)
    coords_max[:, :, 1] = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    coords_max[:, :, 2] = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    y_max = input.clone()

    if D < 3 or H < 3 or W < 3:
        return coords_max, y_max

    # ── Step 1: NMS maxima ────────────────────────────────────────────────────
    nms_mask = precomputed_nms_mask if precomputed_nms_mask is not None else nms3d(input, (3, 3, 3), True)
    bc_idx, d_idx, h_idx, w_idx = torch.where(nms_mask.view(BC, D, H, W))
    N = bc_idx.shape[0]
    if N == 0:
        return coords_max, y_max

    # ── Step 2: dilate NMS positions — L∞ ball of radius dilation_radius ────────
    # Generates all voxels a keypoint could visit across n_iters shift steps.
    # With max_subpixel_shift=0.6, almost all keypoints converge in ≤1 integer move,
    # so dilation_radius=1 (27 positions per max) is sufficient in practice.
    r = dilation_radius
    offs = torch.arange(-r, r + 1, device=device, dtype=torch.long)  # (2r+1,)
    od, oh, ow = torch.meshgrid(offs, offs, offs, indexing="ij")  # each: (2r+1)³
    od = od.reshape(-1)  # (K,)  K = (2r+1)³
    oh = oh.reshape(-1)
    ow = ow.reshape(-1)
    K = od.shape[0]

    # Broadcast expand: (N,1) + (1,K) → (N*K,)
    d_dil = (d_idx.unsqueeze(1) + od.unsqueeze(0)).reshape(-1)
    h_dil = (h_idx.unsqueeze(1) + oh.unsqueeze(0)).reshape(-1)
    w_dil = (w_idx.unsqueeze(1) + ow.unsqueeze(0)).reshape(-1)
    bc_dil = bc_idx.unsqueeze(1).expand(-1, K).reshape(-1)

    # Keep only interior positions (Hessian needs ±1 neighbours in all dims).
    keep = (d_dil >= 1) & (d_dil <= D - 2) & (h_dil >= 1) & (h_dil <= H - 2) & (w_dil >= 1) & (w_dil <= W - 2)
    d_dil = d_dil[keep]
    h_dil = h_dil[keep]
    w_dil = w_dil[keep]
    bc_dil = bc_dil[keep]

    # Deduplicate: multiple maxima may share dilated positions.
    flat_u = torch.unique(bc_dil * DHW + d_dil * HW + h_dil * W + w_dil, sorted=False)
    bc_u = flat_u // DHW
    rem = flat_u % DHW
    d_u = rem // HW
    rem = rem % HW
    h_u = rem // W
    w_u = rem % W

    # ── Step 3: gather 3×3×3 neighbourhood for all unique dilated positions ──
    inp_flat = input.view(-1)
    patch_offsets = _PATCH_DD.to(device) * HW + _PATCH_DH.to(device) * W + _PATCH_DW.to(device)  # (27,)
    center_flat = bc_u * DHW + d_u * HW + h_u * W + w_u
    patch = inp_flat[center_flat.unsqueeze(1) + patch_offsets.unsqueeze(0)]  # (NU, 27)

    # Named patch elements.  Flat index: k = (dd+1)*9 + (dh+1)*3 + (dw+1), center k=13.
    c000 = patch[:, 13]
    p_xm = patch[:, 12]
    p_xp = patch[:, 14]
    p_ym = patch[:, 10]
    p_yp = patch[:, 16]
    p_sm = patch[:, 4]
    p_sp = patch[:, 22]
    p_xm_ym = patch[:, 9]
    p_xp_ym = patch[:, 11]
    p_xm_yp = patch[:, 15]
    p_xp_yp = patch[:, 17]
    p_xm_sm = patch[:, 3]
    p_xp_sm = patch[:, 5]
    p_xm_sp = patch[:, 21]
    p_xp_sp = patch[:, 23]
    p_ym_sm = patch[:, 1]
    p_yp_sm = patch[:, 7]
    p_ym_sp = patch[:, 19]
    p_yp_sp = patch[:, 25]

    # ── Step 4: compute gradients + Hessian + solve (all unique positions) ───
    gx = 0.5 * (p_xp - p_xm)
    gy = 0.5 * (p_yp - p_ym)
    gs = 0.5 * (p_sp - p_sm)
    dxx = p_xp - 2.0 * c000 + p_xm
    dyy = p_yp - 2.0 * c000 + p_ym
    dss = p_sp - 2.0 * c000 + p_sm
    dxy = 0.25 * (p_xp_yp - p_xm_yp - p_xp_ym + p_xm_ym)
    dxs = 0.25 * (p_xp_sp - p_xm_sp - p_xp_sm + p_xm_sm)
    dys = 0.25 * (p_yp_sp - p_ym_sp - p_yp_sm + p_ym_sm)

    sx_u, sy_u, ss_u, sol_u = _solve_cramer_sym3x3(dxx, dyy, dss, dxy, dxs, dys, -gx, -gy, -gs)
    # Precompute gradient·shift for the response correction (avoids storing gx/gy/gs tables).
    gds_u = gx * sx_u + gy * sy_u + gs * ss_u

    # ── Step 5: scatter solutions to dense lookup tables ─────────────────────
    # Tables are (BC, D, H, W): only the ~N_dilated positions are filled;
    # all others stay at zero / False and will mark migrated keypoints as invalid.
    sx_f = torch.zeros(BC, D, H, W, device=device, dtype=dtype)
    sy_f = torch.zeros_like(sx_f)
    ss_f = torch.zeros_like(sx_f)
    gds_f = torch.zeros_like(sx_f)
    sol_f = torch.zeros(BC, D, H, W, device=device, dtype=torch.bool)
    sx_f[bc_u, d_u, h_u, w_u] = sx_u
    sy_f[bc_u, d_u, h_u, w_u] = sy_u
    ss_f[bc_u, d_u, h_u, w_u] = ss_u
    gds_f[bc_u, d_u, h_u, w_u] = gds_u
    sol_f[bc_u, d_u, h_u, w_u] = sol_u

    # ── Step 6: iterative lookup — no .any().item() sync ─────────────────────
    d_cur = d_idx.clone()
    h_cur = h_idx.clone()
    w_cur = w_idx.clone()
    valid = torch.ones(N, dtype=torch.bool, device=device)
    shift_x = torch.zeros(N, device=device, dtype=dtype)
    shift_y = torch.zeros(N, device=device, dtype=dtype)
    shift_s = torch.zeros(N, device=device, dtype=dtype)
    grad_dot_shift = torch.zeros(N, device=device, dtype=dtype)

    for _ in range(n_iters):
        di = d_cur.clamp(1, D - 2)
        hi = h_cur.clamp(1, H - 2)
        wi = w_cur.clamp(1, W - 2)

        sx = sx_f[bc_idx, di, hi, wi]
        sy = sy_f[bc_idx, di, hi, wi]
        ss = ss_f[bc_idx, di, hi, wi]
        sol = sol_f[bc_idx, di, hi, wi]
        gds = gds_f[bc_idx, di, hi, wi]

        valid = valid & sol
        vf = valid.to(dtype)
        sx = sx * vf
        sy = sy * vf
        ss = ss * vf

        shift_x = torch.where(valid, sx, shift_x)
        shift_y = torch.where(valid, sy, shift_y)
        shift_s = torch.where(valid, ss, shift_s)
        grad_dot_shift = torch.where(valid, gds, grad_dot_shift)

        move_px = valid & (sx > max_subpixel_shift)
        move_nx = valid & (sx < -max_subpixel_shift)
        new_w = w_cur + move_px.long() - move_nx.long()
        valid = valid & (new_w >= 1) & (new_w <= W - 2)
        w_cur = new_w.clamp(0, W - 1)

        move_py = valid & (sy > max_subpixel_shift)
        move_ny = valid & (sy < -max_subpixel_shift)
        new_h = h_cur + move_py.long() - move_ny.long()
        valid = valid & (new_h >= 1) & (new_h <= H - 2)
        h_cur = new_h.clamp(0, H - 1)

        move_ps = valid & (ss > max_subpixel_shift)
        move_ns = valid & (ss < -max_subpixel_shift)
        new_d = d_cur + move_ps.long() - move_ns.long()
        valid = valid & (new_d >= 1) & (new_d <= D - 2)
        d_cur = new_d.clamp(0, D - 1)

    valid = valid & (shift_x.abs() <= 1.5) & (shift_y.abs() <= 1.5) & (shift_s.abs() <= 1.5)

    # ── Write refined coordinates and corrected response ──────────────────────
    b_idx = bc_idx // C
    c_idx = bc_idx % C

    coords_max[b_idx, c_idx, 0, d_idx, h_idx, w_idx] = torch.where(valid, d_cur.to(dtype) + shift_s, d_idx.to(dtype))
    coords_max[b_idx, c_idx, 1, d_idx, h_idx, w_idx] = torch.where(valid, w_cur.to(dtype) + shift_x, w_idx.to(dtype))
    coords_max[b_idx, c_idx, 2, d_idx, h_idx, w_idx] = torch.where(valid, h_cur.to(dtype) + shift_y, h_idx.to(dtype))

    val_correction = 0.5 * torch.where(valid, grad_dot_shift, torch.zeros_like(grad_dot_shift))
    val_center = input.view(BC, D, H, W)[bc_idx, d_idx, h_idx, w_idx]
    y_max[b_idx, c_idx, d_idx, h_idx, w_idx] = val_center + val_correction
    if strict_maxima_bonus > 0:
        y_max[b_idx, c_idx, d_idx, h_idx, w_idx] += strict_maxima_bonus * valid.to(dtype)

    return coords_max, y_max


class ConvQuadInterp3d(nn.Module):
    r"""Subpixel localization of 3D scale-space extrema via quadratic interpolation.

    Wraps :func:`~kornia.geometry.subpix.conv_quad_interp3d`.  The Hessian system
    is solved once for each voxel in the dilated NMS neighbourhood (no dense
    precomputation over the whole volume), then the shift chain is followed by
    table lookup with no GPU→CPU synchronisation — making the module compatible
    with ``torch.compile`` and CUDA graphs.

    Args:
        n_iters: maximum localization iterations per keypoint.
        strict_maxima_bonus: score bonus at NMS-maximum positions.
        max_subpixel_shift: shift threshold that triggers integer centre move.
    """

    def __init__(
        self,
        n_iters: int = 5,
        strict_maxima_bonus: float = 10.0,
        max_subpixel_shift: float = 0.6,
        dilation_radius: int = 1,
    ) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.strict_maxima_bonus = strict_maxima_bonus
        self.max_subpixel_shift = max_subpixel_shift
        self.dilation_radius = dilation_radius

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_iters={self.n_iters}, "
            f"strict_maxima_bonus={self.strict_maxima_bonus}, "
            f"max_subpixel_shift={self.max_subpixel_shift}, "
            f"dilation_radius={self.dilation_radius})"
        )

    def forward(
        self, x: torch.Tensor, precomputed_nms_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return conv_quad_interp3d(
            x,
            self.n_iters,
            self.strict_maxima_bonus,
            self.max_subpixel_shift,
            precomputed_nms_mask,
            self.dilation_radius,
        )


def iterative_quad_interp3d(
    input: torch.Tensor,
    n_iters: int = 5,
    strict_maxima_bonus: float = 10.0,
    max_subpixel_shift: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Iterative subpixel localization of 3D extrema via quadratic interpolation.

    Unlike :func:`conv_quad_interp3d`, which pre-computes the Hessian solve for all
    voxels reachable from NMS maxima and then follows shifts by table lookup, this
    function explicitly re-extracts the :math:`3 \times 3 \times 3` patch at each
    NMS maximum and iterates up to ``n_iters`` times. When the estimated subpixel
    shift along any spatial or scale axis exceeds ``max_subpixel_shift`` the integer
    center is moved one step in that direction and the solve is repeated — matching
    the localization loop from the HessAff / SIFT family of detectors.

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

    coords_max = torch.empty(B, C, 3, D, H, W, device=device, dtype=dtype)
    coords_max[:, :, 0] = torch.arange(D, device=device, dtype=dtype).view(D, 1, 1)
    coords_max[:, :, 1] = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    coords_max[:, :, 2] = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    y_max = input.clone()

    if D < 3 or H < 3 or W < 3:
        return coords_max, y_max

    inp = input.reshape(B * C, D, H, W)
    DHW = D * H * W
    HW = H * W

    nms_mask = nms3d(input, (3, 3, 3), True)
    nms_flat = nms_mask.view(B * C, D, H, W)

    bc_idx, d_idx, h_idx, w_idx = torch.where(nms_flat)
    N = bc_idx.shape[0]
    if N == 0:
        return coords_max, y_max

    patch_offsets = _PATCH_DD.to(device) * HW + _PATCH_DH.to(device) * W + _PATCH_DW.to(device)

    d_cur = d_idx.clone()
    h_cur = h_idx.clone()
    w_cur = w_idx.clone()

    valid = torch.ones(N, dtype=torch.bool, device=device)

    shift_x = torch.zeros(N, device=device, dtype=dtype)
    shift_y = torch.zeros(N, device=device, dtype=dtype)
    shift_s = torch.zeros(N, device=device, dtype=dtype)
    grad_dot_shift = torch.zeros(N, device=device, dtype=dtype)

    inp_flat = inp.reshape(-1)
    bc_base = bc_idx * DHW

    for _ in range(n_iters):
        d_s = d_cur.clamp(1, D - 2)
        h_s = h_cur.clamp(1, H - 2)
        w_s = w_cur.clamp(1, W - 2)

        patch = inp_flat[(bc_base + d_s * HW + h_s * W + w_s).unsqueeze(1) + patch_offsets.unsqueeze(0)]

        c000 = patch[:, 13]
        p_xm = patch[:, 12]
        p_xp = patch[:, 14]
        p_ym = patch[:, 10]
        p_yp = patch[:, 16]
        p_sm = patch[:, 4]
        p_sp = patch[:, 22]
        p_xm_ym = patch[:, 9]
        p_xp_ym = patch[:, 11]
        p_xm_yp = patch[:, 15]
        p_xp_yp = patch[:, 17]
        p_xm_sm = patch[:, 3]
        p_xp_sm = patch[:, 5]
        p_xm_sp = patch[:, 21]
        p_xp_sp = patch[:, 23]
        p_ym_sm = patch[:, 1]
        p_yp_sm = patch[:, 7]
        p_ym_sp = patch[:, 19]
        p_yp_sp = patch[:, 25]

        gx = 0.5 * (p_xp - p_xm)
        gy = 0.5 * (p_yp - p_ym)
        gs = 0.5 * (p_sp - p_sm)

        dxx = p_xp - 2.0 * c000 + p_xm
        dyy = p_yp - 2.0 * c000 + p_ym
        dss = p_sp - 2.0 * c000 + p_sm
        dxy = 0.25 * (p_xp_yp - p_xm_yp - p_xp_ym + p_xm_ym)
        dxs = 0.25 * (p_xp_sp - p_xm_sp - p_xp_sm + p_xm_sm)
        dys = 0.25 * (p_yp_sp - p_ym_sp - p_yp_sm + p_ym_sm)

        sx, sy, ss, solved = _solve_cramer_sym3x3(dxx, dyy, dss, dxy, dxs, dys, -gx, -gy, -gs)
        valid = valid & solved

        valid_f = valid.to(dtype)
        sx = sx * valid_f
        sy = sy * valid_f
        ss = ss * valid_f

        shift_x = torch.where(valid, sx, shift_x)
        shift_y = torch.where(valid, sy, shift_y)
        shift_s = torch.where(valid, ss, shift_s)
        grad_dot_shift = torch.where(valid, gx * sx + gy * sy + gs * ss, grad_dot_shift)

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

    valid = valid & (shift_x.abs() <= 1.5) & (shift_y.abs() <= 1.5) & (shift_s.abs() <= 1.5)

    b_idx = bc_idx // C
    c_idx = bc_idx % C

    final_s = torch.where(valid, d_cur.to(dtype) + shift_s, d_idx.to(dtype))
    final_x = torch.where(valid, w_cur.to(dtype) + shift_x, w_idx.to(dtype))
    final_y = torch.where(valid, h_cur.to(dtype) + shift_y, h_idx.to(dtype))

    coords_max[b_idx, c_idx, 0, d_idx, h_idx, w_idx] = final_s
    coords_max[b_idx, c_idx, 1, d_idx, h_idx, w_idx] = final_x
    coords_max[b_idx, c_idx, 2, d_idx, h_idx, w_idx] = final_y

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


class AdaptiveQuadInterp3d(nn.Module):
    r"""Subpixel localization of 3D scale-space extrema with automatic backend selection.

    Wraps :func:`~kornia.geometry.subpix.conv_quad_interp3d` and
    :func:`~kornia.geometry.subpix.iterative_quad_interp3d`, choosing the faster
    backend based on the input device and the requested ``mode``.

    Benchmarks show:

    * **GPU** — :func:`conv_quad_interp3d` is 1.5-2x faster due to better
      parallelism on the batched gather+solve.
    * **CPU** — :func:`iterative_quad_interp3d` is faster for large images because
      it processes only the NMS maxima directly without any dilation/dedup overhead.

    Args:
        mode: backend selection strategy.

            * ``"patch"`` — always use :func:`iterative_quad_interp3d`.
            * ``"conv"``  — always use :func:`conv_quad_interp3d`.
            * ``"auto"``  — use ``"conv"`` when the input is on a CUDA device,
              ``"patch"`` otherwise.

        n_iters: maximum localization iterations per keypoint.
        strict_maxima_bonus: score bonus added at NMS-maximum positions.
        max_subpixel_shift: integer-centre move threshold.
        dilation_radius: L\ :math:`\infty` precompute radius for ``"conv"`` mode
            (ignored in ``"patch"`` mode).

    Example:
        >>> inp = torch.randn(1, 1, 3, 64, 64)
        >>> subpix = AdaptiveQuadInterp3d(mode="auto")
        >>> coords, vals = subpix(inp)
        >>> coords.shape
        torch.Size([1, 1, 3, 3, 64, 64])
        >>> vals.shape
        torch.Size([1, 1, 3, 64, 64])

    """

    MODES = ("patch", "conv", "auto")

    def __init__(
        self,
        mode: str = "auto",
        n_iters: int = 5,
        strict_maxima_bonus: float = 10.0,
        max_subpixel_shift: float = 0.6,
        dilation_radius: int = 3,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        self.mode = mode
        self.n_iters = n_iters
        self.strict_maxima_bonus = strict_maxima_bonus
        self.max_subpixel_shift = max_subpixel_shift
        self.dilation_radius = dilation_radius

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', "
            f"n_iters={self.n_iters}, "
            f"strict_maxima_bonus={self.strict_maxima_bonus}, "
            f"max_subpixel_shift={self.max_subpixel_shift}, "
            f"dilation_radius={self.dilation_radius})"
        )

    def forward(
        self,
        x: torch.Tensor,
        precomputed_nms_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_conv = self.mode == "conv" or (self.mode == "auto" and x.is_cuda)
        if use_conv:
            return conv_quad_interp3d(
                x,
                self.n_iters,
                self.strict_maxima_bonus,
                self.max_subpixel_shift,
                precomputed_nms_mask,
                self.dilation_radius,
            )
        return iterative_quad_interp3d(x, self.n_iters, self.strict_maxima_bonus, self.max_subpixel_shift)
