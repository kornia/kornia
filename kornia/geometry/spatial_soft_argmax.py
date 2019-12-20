import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry import dsnt
from kornia.utils import create_meshgrid, create_meshgrid3d
from kornia.geometry import normalize_pixel_coordinates, normalize_pixel_coordinates3d
from typing import Tuple, Union


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

    See :func:`~kornia.geometry.conv_soft_argmax2d` for details.
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

    See :func:`~kornia.geometry.conv_soft_argmax3d` for details.
    """

    def __init__(self,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 temperature: Union[torch.Tensor, float] = torch.tensor(1.0),
                 normalized_coordinates: bool = False,
                 eps: float = 1e-8,
                 output_value: bool = True) -> None:
        super(ConvSoftArgmax3d, self).__init__()
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
        return conv_soft_argmax3d(x,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.temperature,
                                  self.normalized_coordinates,
                                  self.eps,
                                  self.output_value)


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
        output_value (bool): if True, val is outputed, if False, only ij

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
        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3,3), (2,2), (1,1))
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
                       output_value: bool = True) -> Union[torch.Tensor,
                                                           Tuple[torch.Tensor, torch.Tensor]]:
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
        output_value (bool): if True, val is outputed, if False, only ij

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
        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3, 3, 3), (1, 2, 2), (0, 1, 1))
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

    x_softmaxpool = pool_coef * F.avg_pool3d(x_exp.view_as(input) * input,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding) / den
    x_softmaxpool = x_softmaxpool.view(b,
                                       c,
                                       x_softmaxpool.size(2),
                                       x_softmaxpool.size(3),
                                       x_softmaxpool.size(4))

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

    if output_value:
        return coords_max, x_softmaxpool
    return coords_max


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
            [0., 0., 0.],
            [0., 10., 0.],
            [0., 0., 0.]]]])
        >>> coords = kornia.spatial_soft_argmax2d(input, False)
        tensor([[[1.0000, 1.0000]]])
    """
    input_soft: torch.Tensor = dsnt.spatial_softmax_2d(input, temperature)
    output: torch.Tensor = dsnt.spatial_softargmax_2d(input_soft,
                                                      normalized_coordinates)
    return output


class SpatialSoftArgmax2d(nn.Module):
    r"""Module that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.contrib.spatial_soft_argmax2d` for details.
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
