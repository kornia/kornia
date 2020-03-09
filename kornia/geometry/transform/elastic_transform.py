"""Elastic Transform
"""

# Authors: Issam H. Laradji <issam.laradji@gmail.com>

import torch 
import math
import kornia

import torch.nn.functional as F


def elastic_transform_2d(img, kernel_size=(3,3), sigma=(4,4), alpha=(32,32), random_seed=None):
    r"""Applies elastic transform of images as described in [Simard2003]_.

    Args:
        img (torch.Tensor): input image.
        kernel_size (Tuple[int, int]): the size of the Gaussian kernel. Default:(3,3).
        sigma (Tuple[float, float]): the standard deviation of the Gaussian in the y and x directions, respecitvely. 
                                     Larger sigma results in smaller pixel displacements. Default:(4,4).
        alpha (Tuple[float, float]):  the scaling factor that controls the intensity of the deformation
                                  in the y and x directions, respectively. Default:32.
        

    Returns:
        img (torch.Tensor): the elastically transformed input image.

    References:
        [Simard2003]: Simard, Steinkraus and Platt, "Best Practices for
                      Convolutional Neural Networks applied to Visual Document Analysis", in
                      Proc. of the International Conference on Document Analysis and
                      Recognition, 2003.
    """
    generator = torch.Generator(device=img.device)
    if random_seed is not None:
        generator.manual_seed(random_seed)

    H, W = img.shape[-2:]
    
    # Get Gaussian kernel for 'y' and 'x' displacement
    kernel_y = kornia.filters.get_gaussian_kernel2d(kernel_size, (sigma[0], sigma[0]))[None, None]
    kernel_x = kornia.filters.get_gaussian_kernel2d(kernel_size, (sigma[1], sigma[1]))[None, None]

    # Convolve over a random displacement matrix and scale them with 'alpha'
    displacement = torch.rand(2, H, W, generator=generator)*0.1
    disp_y = F.conv2d(displacement[0][None, None], weight=kernel_y, padding=kernel_size[0]//2) * alpha[0]
    disp_x = F.conv2d(displacement[1][None, None], weight=kernel_x, padding=kernel_size[0]//2) * alpha[1]

    # stack and normalize displacement
    disp = torch.stack([disp_y, disp_x], dim=4).squeeze(0) 
    
    # Warp image based on displacement matrix
    grid = kornia.utils.create_meshgrid(H, W)
    warped =  F.grid_sample(img, (grid + disp).clamp(-1,1), align_corners=True)
    
    return warped
