from typing import Union, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.constants import pi
from kornia.utils import create_meshgrid
from kornia.geometry.conversions import cart2pol
from kornia.filters import SpatialGradient, GaussianBlur2d


# Precomputed coefficients for Von Mises kernel, given N and K(appa).
sqrt2: float = 1.4142135623730951
COEFFS_N1_K1: List[float] = [0.38214156, 0.48090413]
COEFFS_N2_K8: List[float] = [0.14343168, 0.268285, 0.21979234]
COEFFS_N3_K8: List[float] = [0.14343168, 0.268285, 0.21979234, 0.15838885]
COEFFS:Dict[str, List[float]] = {'xy':COEFFS_N1_K1,
                                 'rhophi':COEFFS_N2_K8,
                                 'theta':COEFFS_N3_K8}

urls: Dict[str, str] = {k:f'https://github.com/manyids2/mkd_pytorch/raw/master/mkd_pytorch/mkd-{k}-64.pth'
                        for k in ['cart', 'polar', 'concat']}


def get_grid_dict(patch_size:int = 32) -> Dict[str, torch.Tensor]:
    """Gets cartesian and polar parametrizations of grid. """
    kgrid = create_meshgrid(height=patch_size,
                            width=patch_size,
                            normalized_coordinates=True)
    x = kgrid[0,:,:,0]
    y = kgrid[0,:,:,1]
    rho, phi = cart2pol(x, y)
    grid_dict = {'x':x, 'y':y, 'rho':rho, 'phi':phi}
    return grid_dict


def get_kron_order(d1: int, d2: int) -> torch.Tensor:
    """Gets order for doing kronecker product. """
    kron_order = torch.zeros([d1 * d2, 2], dtype=torch.int64)
    for i in range(d1):
        for j in range(d2):
            kron_order[i * d2 + j, 0] = i
            kron_order[i * d2 + j, 1] = j
    return kron_order


class MKDGradients(nn.Module):
    r"""
    Module, which computes gradients of given patches,
    stacked as [magnitudes, orientations].
    Given gradients $g_x$, $g_y$ with respect to $x$, $y$ respectively,
      - $\mathbox{mags} = $\sqrt{g_x^2 + g_y^2 + eps}$
      - $\mathbox{oris} = $\mbox{tan}^{-1}(\nicefrac{g_y}{g_x})$.
    Args:
        patch_size: (int) Input patch size in pixels (32 is default)
    Returns:
        Tensor: gradients of given patches
    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, 2, patch_size, patch_size)
    Examples::
        >>> patches = torch.rand(23, 1, 32, 32)
        >>> gradient = kornia.feature.mkd.MKDGradients(patch_size=32)
        >>> g = gradient(patches) # 23x2x32x32
    """

    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-8

        # Modify 'diff' gradient.
        grad_fn = SpatialGradient(mode='diff', order=1, normalized=False)
        grad_fn.kernel = -1 * grad_fn.kernel
        self.grad = grad_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grads_xy = self.grad(x)[:,0,:,:,:]
        gx = grads_xy[:,0,:,:].unsqueeze(1)
        gy = grads_xy[:,1,:,:].unsqueeze(1)
        mags = torch.sqrt(torch.pow(gx, 2) + torch.pow(gy, 2) + self.eps)
        oris = torch.atan2(gy, gx)
        y = torch.cat([mags, oris], dim=1)
        return y

    def __repr__(self) -> str:
        return self.__class__.__name__
