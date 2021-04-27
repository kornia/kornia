from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

import kornia
from kornia.filters import GaussianBlur2d

def sharpen(
        input: np.ndarray,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = 'reflect') -> np.ndarray:
    r"""Creates an operator that blurs a tensor using the existing Gaussian filter available with the Kornia library.


    Arguments:
        input (np.ndarray): the input image.
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        np.ndarray: the sharpened image.

    Examples:
        >>> input = cv2.imread(path_to_image)
        >>> output = sharpen(input, (3, 3), (1.5, 1.5))
        >>> type(output)
        <class 'numpy.ndarray'>
    """
    data: torch.tensor = kornia.image_to_tensor(input, keepdim=False)
    data=data.float()/255.
    gauss = kornia.filters.GaussianBlur2d(kernel_size, sigma)
    data_blur: torch.tensor = gauss(data)
    data_new = data + (data - data_blur)
    img_new = kornia.tensor_to_image(data_new) * 255
    return img_new
    




class unsharp_mask(nn.Module):
    r"""Creates an operator that sharpens image using the existing Gaussian filter available with the Kornia library..


    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        Tensor: the sharpened image.

    Examples:

        >>> input = cv2.imread(path_to_image)
        >>> sharpen = unsharp_mask((3, 3), (1.5, 1.5))
        >>> output = sharpen(input)
        >>> type(output)
        <class 'numpy.ndarray'>
        
    """
    def __init__(self,kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:
        super(unsharp_mask, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.border_type = border_type
    

    def forward(self, input: np.ndarray) -> np.ndarray:
        return sharpen(input, self.kernel_size, self.sigma, self.border_type)
