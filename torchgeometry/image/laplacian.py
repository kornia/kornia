#from typing import Tuple

import math
import torch
import torch.nn as nn
from torch.nn.functional import conv2d


def laplacian(window_size):
	r"""One could also use the Laplacian of Gaussian formula
		to design the filter.
	"""

	laplacian = torch.ones(window_size)
	laplacian[int(math.floor(window_size/2))] = 1 - window_size
	return laplacian


def get_laplacian_kernel(ksize: int):
	r"""Function that returns the coefficients of a 1D Laplacian filter

	Args:
		ksize (int): filter size. It should be odd and positive.

	Returns:
		Tensor (float): 1D tensor with laplacian filter coefficients.

	Shape:
		- Output: math:`(ksize, 0)`

	Examples::


	"""
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = laplacian(ksize)
    return window_1d



kk =get_laplacian_kernel(7)
print(kk)