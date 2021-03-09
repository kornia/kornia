from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from kornia.geometry.subpix import dsnt
from kornia.utils import create_meshgrid, create_meshgrid3d
from kornia.geometry import normalize_pixel_coordinates, normalize_pixel_coordinates3d


def _get_window_grid_kernel2d(h: int, w: int,
                              device: torch.device = torch.device('cpu')) -> torch.Tensor:
    r"""Helper function, which generates a kernel to with window coordinates,
       residual to window center.

    Args:
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [2x1xhxw]
    """
    window_grid2d = create_meshgrid(h, w, False, device=device)
    window_grid2d = normalize_pixel_coordinates(window_grid2d, h, w)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2)
    return conv_kernel


def _get_center_kernel2d(h: int, w: int,
                         device: torch.device = torch.device('cpu')) -> torch.Tensor:
    r"""Helper function, which generates a kernel to return center coordinates,
       when applied with F.conv2d to 2d coordinates grid.

    Args:
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [2x2xhxw]
    """
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
    center_kernel[(0, 1), (0, 1), h_i1: h_i2, w_i1: w_i2] = 1.0 / float(((h_i2 - h_i1) * (w_i2 - w_i1)))
    return center_kernel


def _get_center_kernel3d(d: int, h: int, w: int,
                         device: torch.device = torch.device('cpu')) -> torch.Tensor:
    r"""Helper function, which generates a kernel to return center coordinates,
       when applied with F.conv2d to 3d coordinates grid.

    Args:
         d (int): kernel depth.
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [3x3xdxhxw]
    """
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
    center_kernel[(0, 1, 2), (0, 1, 2), d_i1: d_i2, h_i1: h_i2, w_i1: w_i2] = 1.0 / center_num
    return center_kernel


def _get_window_grid_kernel3d(d: int, h: int, w: int,
                              device: torch.device = torch.device('cpu')) -> torch.Tensor:
    r"""Helper function, which generates a kernel to return coordinates,
       residual to window center.

    Args:
         d (int): kernel depth.
         h (int): kernel height.
         w (int): kernel width.
         device (torch.device): device, on which generate.

    Returns:
        conv_kernel (torch.Tensor) [3x1xdxhxw]
    """
    grid2d = create_meshgrid(h, w, True, device=device)
    if d > 1:
        z = torch.linspace(-1, 1, d, device=device).view(d, 1, 1, 1)
    else:  # only onr channel with index == 0
        z = torch.zeros(1, 1, 1, 1, device=device)
    grid3d = torch.cat([z.repeat(1, h, w, 1).contiguous(), grid2d.repeat(d, 1, 1, 1)], dim=3)
    conv_kernel = grid3d.permute(3, 0, 1, 2).unsqueeze(1)
    return conv_kernel


class ConvSoftArgmax2d(nn.Module):
    r"""Module that calculates soft argmax 2d per window.

    See :func:`~kornia.geometry.subpix.conv_soft_argmax2d` for details.
    """

    def __init__(self,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1),
                 temperature: Union[torch.Tensor, float] = torch.tensor(1.0),
                 normalized_coordinates: bool = True,
                 eps: float = 1e-8,
                 output_value: bool = False) -> None:
        super(ConvSoftArgmax2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(' + 'kernel_size=' + str(self.kernel_size) +\
            ', ' + 'stride=' + str(self.stride) +\
            ', ' + 'padding=' + str(self.padding) +\
            ', ' + 'temperature=' + str(self.temperature) +\
            ', ' + 'normalized_coordinates=' + str(self.normalized_coordinates) +\
            ', ' + 'eps=' + str(self.eps) +\
            ', ' + 'output_value=' + str(self.output_value) + ')'

    def forward(self, x: torch.Tensor):  # type: ignore
        return conv_soft_argmax2d(x,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.temperature,
                                  self.normalized_coordinates,
                                  self.eps,
                                  self.output_value)


class ConvSoftArgmax3d(nn.Module):
    r"""Module that calculates soft argmax 3d per window.

    See :func:`~kornia.geometry.subpix.conv_soft_argmax3d` for details.
    """

    def __init__(self,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 temperature: Union[torch.Tensor, float] = torch.tensor(1.0),
                 normalized_coordinates: bool = False,
                 eps: float = 1e-8,
                 output_value: bool = True,
                 strict_maxima_bonus: float = 0.0) -> None:
        super(ConvSoftArgmax3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value
        self.strict_maxima_bonus = strict_maxima_bonus
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(' + 'kernel_size=' + str(self.kernel_size) +\
            ', ' + 'stride=' + str(self.stride) +\
            ', ' + 'padding=' + str(self.padding) +\
            ', ' + 'temperature=' + str(self.temperature) +\
            ', ' + 'normalized_coordinates=' + str(self.normalized_coordinates) +\
            ', ' + 'eps=' + str(self.eps) +\
            ', ' + 'strict_maxima_bonus=' + str(self.strict_maxima_bonus) +\
            ', ' + 'output_value=' + str(self.output_value) + ')'

    def forward(self, x: torch.Tensor):  # type: ignore
        return conv_soft_argmax3d(x,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.temperature,
                                  self.normalized_coordinates,
                                  self.eps,
                                  self.output_value,
                                  self.strict_maxima_bonus)


def conv_soft_argmax2d(input: torch.Tensor,
                       kernel_size: Tuple[int, int] = (3, 3),
                       stride: Tuple[int, int] = (1, 1),
                       padding: Tuple[int, int] = (1, 1),
                       temperature: Union[torch.Tensor, float] = torch.tensor(1.0),
                       normalized_coordinates: bool = True,
                       eps: float = 1e-8,
                       output_value: bool = False) -> Union[torch.Tensor,
                                                            Tuple[torch.Tensor, torch.Tensor]]:
    r"""Function that computes the convolutional spatial Soft-Argmax 2D over the windows
    of a given input heatmap. Function has two outputs: argmax coordinates and the softmaxpooled heatmap values
    themselves. On each window, the function computed is

    .. math::
             ij(X) = \frac{\sum{(i,j)} * exp(x / T)  \in X} {\sum{exp(x / T)  \in X}}

    .. math::
             val(X) = \frac{\sum{x * exp(x / T)  \in X}} {\sum{exp(x / T)  \in X}}

    where T is temperature.

    Args:
        kernel_size (Tuple[int,int]): the size of the window
        stride  (Tuple[int,int]): the stride of the window.
        padding (Tuple[int,int]): input zero padding
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the coordinates normalized in the range of [-1, 1]. Otherwise,
                                       it will return the coordinates in the range of the input shape. Default is True.
        eps (float): small value to avoid zero division. Default is 1e-8.
        output_value (bool): if True, val is output, if False, only ij

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, 2, H_{out}, W_{out})`, :math:`(N, C, H_{out}, W_{out})`, where

         .. math::
                  H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                  (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

         .. math::
                  W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                  (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3,3), (2,2), (1,1), output_value=True)
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if temperature <= 0:
        raise ValueError("Temperature should be positive float or tensor. Got: {}"
                         .format(temperature))

    b, c, h, w = input.shape
    kx, ky = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, h, w)

    center_kernel: torch.Tensor = _get_center_kernel2d(kx, ky, device).to(dtype)
    window_kernel: torch.Tensor = _get_window_grid_kernel2d(kx, ky, device).to(dtype)

    # applies exponential normalization trick
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # https://github.com/pytorch/pytorch/blob/bcb0bb7e0e03b386ad837015faba6b4b16e3bfb9/aten/src/ATen/native/SoftMax.cpp#L44
    x_max = F.adaptive_max_pool2d(input, (1, 1))

    # max is detached to prevent undesired backprop loops in the graph
    x_exp = ((input - x_max.detach()) / temperature).exp()

    # F.avg_pool2d(.., divisor_override = 1.0) - proper way for sum pool in PyTorch 1.2.
    # Not available yet in version 1.0, so let's do manually
    pool_coef: float = float(kx * ky)

    # softmax denominator
    den = pool_coef * F.avg_pool2d(x_exp, kernel_size, stride=stride, padding=padding) + eps

    x_softmaxpool = pool_coef * F.avg_pool2d(x_exp * input,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding) / den
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3))

    # We need to output also coordinates
    # Pooled window center coordinates
    grid_global: torch.Tensor = create_meshgrid(h, w, False, device).to(
        dtype).permute(0, 3, 1, 2)

    grid_global_pooled = F.conv2d(grid_global,
                                  center_kernel,
                                  stride=stride,
                                  padding=padding)

    # Coordinates of maxima residual to window center
    # prepare kernel
    coords_max: torch.Tensor = F.conv2d(x_exp,
                                        window_kernel,
                                        stride=stride,
                                        padding=padding)

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


def conv_soft_argmax3d(input: torch.Tensor,
                       kernel_size: Tuple[int, int, int] = (3, 3, 3),
                       stride: Tuple[int, int, int] = (1, 1, 1),
                       padding: Tuple[int, int, int] = (1, 1, 1),
                       temperature: Union[torch.Tensor, float] = torch.tensor(1.0),
                       normalized_coordinates: bool = False,
                       eps: float = 1e-8,
                       output_value: bool = True,
                       strict_maxima_bonus: float = 0.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Function that computes the convolutional spatial Soft-Argmax 3D over the windows
    of a given input heatmap. Function has two outputs: argmax coordinates and the softmaxpooled heatmap values
    themselves.
    On each window, the function computed is:

    .. math::
             ijk(X) = \frac{\sum{(i,j,k)} * exp(x / T)  \in X} {\sum{exp(x / T)  \in X}}

    .. math::
             val(X) = \frac{\sum{x * exp(x / T)  \in X}} {\sum{exp(x / T)  \in X}}

    where T is temperature.

    Args:
        kernel_size (Tuple[int,int,int]):  size of the window
        stride (Tuple[int,int,int]): stride of the window.
        padding (Tuple[int,int,int]): input zero padding
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the coordinates normalized in the range of [-1, 1]. Otherwise,
                                       it will return the coordinates in the range of the input shape. Default is False.
        eps (float): small value to avoid zero division. Default is 1e-8.
        output_value (bool): if True, val is output, if False, only ij
        strict_maxima_bonus (float): pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
                                     This is needed for mimic behavior of strict NMS in classic local features
    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

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
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if temperature <= 0:
        raise ValueError("Temperature should be positive float or tensor. Got: {}"
                         .format(temperature))

    b, c, d, h, w = input.shape
    kx, ky, kz = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, d, h, w)

    center_kernel: torch.Tensor = _get_center_kernel3d(kx, ky, kz, device).to(dtype)
    window_kernel: torch.Tensor = _get_window_grid_kernel3d(kx, ky, kz, device).to(dtype)

    # applies exponential normalization trick
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # https://github.com/pytorch/pytorch/blob/bcb0bb7e0e03b386ad837015faba6b4b16e3bfb9/aten/src/ATen/native/SoftMax.cpp#L44
    x_max = F.adaptive_max_pool3d(input, (1, 1, 1))

    # max is detached to prevent undesired backprop loops in the graph
    x_exp = ((input - x_max.detach()) / temperature).exp()

    pool_coef: float = float(kx * ky * kz)

    # softmax denominator
    den = pool_coef * F.avg_pool3d(x_exp.view_as(input),
                                   kernel_size,
                                   stride=stride,
                                   padding=padding) + eps

    # We need to output also coordinates
    # Pooled window center coordinates
    grid_global: torch.Tensor = create_meshgrid3d(
        d, h, w, False, device=device).to(dtype).permute(0, 4, 1, 2, 3)

    grid_global_pooled = F.conv3d(grid_global,
                                  center_kernel,
                                  stride=stride,
                                  padding=padding)

    # Coordinates of maxima residual to window center
    # prepare kernel
    coords_max: torch.Tensor = F.conv3d(x_exp,
                                        window_kernel,
                                        stride=stride,
                                        padding=padding)

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
    x_softmaxpool = pool_coef * F.avg_pool3d(x_exp.view(input.size()) * input,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding) / den
    if strict_maxima_bonus > 0:
        in_levels: int = input.size(2)
        out_levels: int = x_softmaxpool.size(2)
        skip_levels: int = (in_levels - out_levels) // 2
        strict_maxima: torch.Tensor = F.avg_pool3d(kornia.feature.nms3d(input, kernel_size), 1, stride, 0)
        strict_maxima = strict_maxima[:, :, skip_levels:out_levels - skip_levels]
        x_softmaxpool *= 1.0 + strict_maxima_bonus * strict_maxima
    x_softmaxpool = x_softmaxpool.view(b,
                                       c,
                                       x_softmaxpool.size(2),
                                       x_softmaxpool.size(3),
                                       x_softmaxpool.size(4))
    return coords_max, x_softmaxpool


def spatial_soft_argmax2d(
        input: torch.Tensor,
        temperature: torch.Tensor = torch.tensor(1.0),
        normalized_coordinates: bool = True,
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        temperature (torch.Tensor): factor to apply to input. Default is 1.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
        eps (float): small value to avoid zero division. Default is 1e-8.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples:
        >>> input = torch.tensor([[[
        ... [0., 0., 0.],
        ... [0., 10., 0.],
        ... [0., 0., 0.]]]])
        >>> spatial_soft_argmax2d(input, normalized_coordinates=False)
        tensor([[[1.0000, 1.0000]]])
    """
    input_soft: torch.Tensor = dsnt.spatial_softmax2d(input, temperature)
    output: torch.Tensor = dsnt.spatial_expectation2d(input_soft, normalized_coordinates)
    return output


class SpatialSoftArgmax2d(nn.Module):
    r"""Module that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.geometry.subpix.spatial_soft_argmax2d` for details.
    """

    def __init__(self,
                 temperature: torch.Tensor = torch.tensor(1.0),
                 normalized_coordinates: bool = True,
                 eps: float = 1e-8) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.temperature: torch.Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates
        self.eps: float = eps

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(temperature=' + str(self.temperature) + ', ' +\
            'normalized_coordinates=' + str(self.normalized_coordinates) + ', ' + \
            'eps=' + str(self.eps) + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return spatial_soft_argmax2d(input, self.temperature,
                                     self.normalized_coordinates, self.eps)


def _get_extrema(b: torch.Tensor,
                 Hes: torch.Tensor,
                 where_to_calculate_mask: torch.Tensor,
                 eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    x_solved: torch.Tensor = torch.zeros_like(b)
    x_solved_masked, _ = torch.solve(b[where_to_calculate_mask.view(-1)],
                                     Hes[where_to_calculate_mask.view(-1)])
    x_solved.masked_scatter_(where_to_calculate_mask.view(-1, 1, 1), x_solved_masked)
    dx: torch.Tensor = -x_solved
    return dx, -x_solved_masked


def _update_converged_mask(dx_masked: torch.Tensor,
                           converged_all: torch.Tensor,
                           nms_mask: torch.Tensor) -> Tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
    current_converged_masked = (dx_masked.abs().max(dim=1, keepdim=True)[0] < 0.5)
    current_non_converged_masked = ~current_converged_masked
    converged_to_add = converged_all.clone().view(-1).masked_scatter_(nms_mask.view(-1),  # noqa
                                                                      current_converged_masked.view(-1)).view_as(nms_mask) # noqa
    converged_all = converged_all.clone() | converged_to_add
    return converged_all, current_converged_masked, current_non_converged_masked


def _get_new_mask(in_progress_mask: torch.Tensor,
                  current_converged_masked: torch.Tensor,
                  current_non_converged_masked: torch.Tensor,
                  dx_masked: torch.Tensor,
                  B: int, CH: int, D: int, H: int, W: int) -> torch.Tensor:
    in_progress_mask = in_progress_mask.view(-1).clone().masked_scatter_(in_progress_mask.view(-1).clone(), # noqa
                                                                         ~current_converged_masked).view(B, CH, D, H, W) # noqa
    new_nms_mask = torch.zeros_like(in_progress_mask)
    new_positions = in_progress_mask.view(B, CH, D, H, W).nonzero().float()
    if len(new_positions) == 0:
        return new_nms_mask
    new_positions[:, 2:] = (new_positions[:, 2:].clone() +
                            dx_masked[current_non_converged_masked.repeat(1, 3, 1)].view(-1, 3))
    new_positions = new_positions.round().long()
    bad_idxs = ((new_positions < 0).max(dim=1)[0] |
                (new_positions[:, 4] >= W) |
                (new_positions[:, 3] >= H) |
                (new_positions[:, 2] >= D))
    new_positions_to_try = new_positions[~bad_idxs]
    new_positions_to_try_idxs = torch.split(new_positions_to_try, 1, dim=1)
    new_nms_mask[new_positions_to_try_idxs] = 1
    #new_nms_mask.masked_fill_(new_positions_to_try_idxs, 1)
    return new_nms_mask


def _filter_new_mask(new_nms_mask: torch.Tensor,
                     already_tried: torch.Tensor) -> torch.Tensor:
    return new_nms_mask & (~already_tried)


def conv_quad_interp3d(input: torch.Tensor,
                       strict_maxima_bonus: float = 10.0,
                       eps: float = 1e-7,
                       n_iter: int = 1):
    r"""Function that computes the single iteration of quadratic interpolation of of the extremum (max or min) location
    and value per each 3x3x3 window which contains strict extremum, similar to one done is SIFT

    Args:
        strict_maxima_bonus (float): pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
                                     This is needed for mimic behavior of strict NMS in classic local features
        eps (float): parameter to control the hessian matrix ill-condition number.
        n_iter (float): number of iterations.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

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
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))
    B, CH, D, H, W = input.shape
    dev: torch.device = input.device
    grid_global: torch.Tensor = create_meshgrid3d(D, H, W, False,
                                                  device=input.device).permute(0, 4, 1, 2, 3)
    grid_global = grid_global.to(input.dtype)

    # to determine the location we are solving system of linear equations Ax = b, where b is 1st order gradient
    # and A is Hessian matrix
    b: torch.Tensor = kornia.filters.spatial_gradient3d(input, order=1, mode='diff')  #
    b = b.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3, 1).flip(1)
    A: torch.Tensor = kornia.filters.spatial_gradient3d(input, order=2, mode='diff')
    A = A.permute(0, 1, 3, 4, 5, 2).reshape(-1, 6)
    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]
    dxy = 0.25 * A[..., 3]  # normalization to match OpenCV implementation
    dys = 0.25 * A[..., 4]  # normalization to match OpenCV implementation
    dxs = 0.25 * A[..., 5]  # normalization to match OpenCV implementation
    Hes = torch.stack([dss, dys, dxs,
                       dys, dyy, dxy,
                       dxs, dxy, dss], dim=-1).view(-1, 3, 3)

    # The following is needed to avoid singular cases
    Hes += torch.rand(Hes[0].size(), device=Hes.device).abs()[None] * eps

    nms_mask: torch.Tensor = kornia.feature.nms3d(input, (3, 3, 3), True)
    num_nc = nms_mask.sum().item()
    if num_nc == 0:  # no extrema, abort
        return grid_global[None].expand(B, CH, 3, D, H, W), input

    converged_all = torch.zeros_like(nms_mask)
    in_progress_mask = nms_mask.clone()
    already_tried = nms_mask.clone()

    #  Find extrema around existing mask
    delta_coords_all, dx_masked = _get_extrema(b, Hes, in_progress_mask, eps)

    #  Save those extrema, which are converged
    (converged_all,
     current_converged_masked,
     current_non_converged_masked) = _update_converged_mask(dx_masked, converged_all, nms_mask)

    # Generate new mask from the non-converging extremas
    new_nms_mask = _get_new_mask(in_progress_mask,
                                 current_converged_masked,
                                 current_non_converged_masked,
                                 dx_masked,
                                 B, CH, D, H, W)

    # Exclude from the new mask those, which are already checked
    in_progress_mask = _filter_new_mask(new_nms_mask, already_tried)

    # Iterate
    for iter_idx in range(max(0, n_iter - 1)):
        num_nc = in_progress_mask.sum().item()
        if num_nc == 0:
            break
        dx_upd, dx_masked_upd = _get_extrema(b, Hes, in_progress_mask, eps)
        already_tried = already_tried | in_progress_mask
        delta_coords_all = delta_coords_all.clone() + dx_upd
        (converged_all,
         current_converged_masked,
         current_non_converged_masked) = _update_converged_mask(dx_masked_upd,
                                                                converged_all,
                                                                in_progress_mask)
        if current_non_converged_masked.sum().item() == 0:
            break
        new_nms_mask = _get_new_mask(in_progress_mask,
                                     current_converged_masked,
                                     current_non_converged_masked,
                                     dx_masked_upd,
                                     B, CH, D, H, W)
        in_progress_mask = _filter_new_mask(new_nms_mask, already_tried)
    delta_coords_all.masked_fill_((~converged_all).view(-1, 1, 1).expand_as(delta_coords_all), 0.0)
    #delta_coords_final = converged_all.view(-1, 1, 1).expand_as(delta_coords_all).to(delta_coords_all.dtype) * 
    #delta_coords_final.masked_scatter_(where_to_calculate_mask.view(-1, 1, 1), x_solved_masked)
    
    dy_all = 0.5 * torch.bmm(b.permute(0, 2, 1), delta_coords_all)
    y_max = input + dy_all.view(B, CH, D, H, W)
    if strict_maxima_bonus > 0:
        y_max += strict_maxima_bonus * converged_all.to(input.dtype)
    delta_coords_all = delta_coords_all.view(B, CH, D, H, W, 3).permute(0, 1, 5, 2, 3, 4)
    delta_coords_all += grid_global[None].expand_as(delta_coords_all)
    return delta_coords_all, y_max


class ConvQuadInterp3d(nn.Module):
    r"""Module that calculates soft argmax 3d per window
    See :func:`~kornia.geometry.subpix.conv_quad_interp3d` for details.
    """

    def __init__(self,
                 strict_maxima_bonus: float = 10.0,
                 eps: float = 1e-7,
                 n_iter: int = 1) -> None:
        super(ConvQuadInterp3d, self).__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps
        self.n_iter = n_iter
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' + 'strict_maxima_bonus=' + str(self.strict_maxima_bonus) + ', n_iter=' + str(self.n_iter) + ')' # noqa

    def forward(self, x: torch.Tensor):  # type: ignore
        return conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps, self.n_iter)
