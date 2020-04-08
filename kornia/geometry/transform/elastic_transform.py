from typing import Tuple, Optional

import torch 
import math
import kornia

import torch.nn.functional as F


def elastic_transform_2d(tensor: torch.Tensor, 
                         kernel_size: Tuple[int, int] = (3,3), 
                         sigma: Tuple[float, float] = (4.,4.), 
                         alpha: Tuple[float, float] = (32.,32.), 
                         random_seed: Optional = None) -> torch.Tensor:
    r"""Applies elastic transform of images as described in [Simard2003]_.

    Args:
        img (torch.Tensor): input image.
        kernel_size (Tuple[int, int]): the size of the Gaussian kernel. Default:(3,3).
        sigma (Tuple[float, float]): the standard deviation of the Gaussian in the y and x directions, respecitvely. 
                                     Larger sigma results in smaller pixel displacements. Default:(4,4).
        alpha (Tuple[float, float]):  the scaling factor that controls the intensity of the deformation
                                  in the y and x directions, respectively. Default:32.
        random_seed (Optional): random seed for generating the displacement vector. Default:None.
        

    Returns:
        img (torch.Tensor): the elastically transformed input image.

    References:
        [Simard2003]: Simard, Steinkraus and Platt, "Best Practices for
                      Convolutional Neural Networks applied to Visual Document Analysis", in
                      Proc. of the International Conference on Document Analysis and
                      Recognition, 2003.
    """
    generator = torch.Generator(device='cpu')
    if random_seed is not None:
        generator.manual_seed(random_seed)

    n, c, h, w = tensor.shape
    
    # Get Gaussian kernel for 'y' and 'x' displacement
    kernel_y = kornia.filters.get_gaussian_kernel2d(kernel_size, (sigma[0], sigma[0]))[None].to(device=tensor.device)
    kernel_x = kornia.filters.get_gaussian_kernel2d(kernel_size, (sigma[1], sigma[1]))[None].to(device=tensor.device)

    # Convolve over a random displacement matrix and scale them with 'alpha'
    displacement = torch.rand(n, 2, h, w, generator=generator).to(device=tensor.device) * 0.1
    disp_y = kornia.filters.filter2D(displacement[:,[0]], kernel=kernel_y, border_type='reflect') * alpha[0]
    disp_x = kornia.filters.filter2D(displacement[:,[1]], kernel=kernel_x, border_type='reflect') * alpha[1]

    # stack and normalize displacement
    disp = torch.cat([disp_y, disp_x], dim=1).squeeze(0).permute(0,2,3,1)
    
    # Warp image based on displacement matrix
    grid = kornia.utils.create_meshgrid(h, w).to(device=tensor.device)
    warped =  F.grid_sample(tensor, (grid + disp).clamp(-1,1), align_corners=True)
    
    return warped

