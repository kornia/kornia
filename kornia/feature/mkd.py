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


class VonMisesKernel(nn.Module):
    """
    Module, which computes parameters of Von Mises kernel given coefficients,
    and embeds given patches.
    Args:
        patch_size: (int) Input patch size in pixels (32 is default)
        coeffs: (list) List of coefficients
              Some examples are hardcoded in COEFFS
    Returns:
        Tensor: Von Mises embedding of given parametrization
    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, d, patch_size, patch_size)
    Examples::
        >>> oris = torch.rand(23, 1, 32, 32)
        >>> vm = kornia.feature.mkd.VonMisesKernel(patch_size=32,
                                                   coeffs=[0.14343168,
                                                           0.268285,
                                                           0.21979234])
        >>> emb = vm(oris) # 23x7x32x32
    """

    def __init__(self,
                 patch_size: int,
                 coeffs: Union[list, tuple]) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.register_buffer('coeffs', torch.Tensor(coeffs).float())

        # Compute parameters.
        n = self.coeffs.shape[0] - 1
        self.n = n
        self.d = 2 * n + 1

        # Precompute helper variables.
        emb0 = torch.ones([1, 1, patch_size, patch_size]).float()
        frange = torch.arange(n).float() + 1
        frange = frange.reshape(-1, 1, 1).float()
        weights = torch.zeros([2 * n + 1]).float()
        weights[:n + 1] = torch.sqrt(self.coeffs)
        weights[n + 1:] = torch.sqrt(self.coeffs[1:])
        weights = weights.reshape(-1, 1, 1).float()
        self.register_buffer('emb0', emb0)
        self.register_buffer('frange', frange)
        self.register_buffer('weights', weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb0 = self.emb0.repeat(x.size(0), 1, 1, 1)
        frange = self.frange * x
        emb1 = torch.cos(frange)
        emb2 = torch.sin(frange)
        embedding = torch.cat([emb0, emb1, emb2], dim=1)
        embedding = self.weights * embedding
        return embedding

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(' + 'patch_size=' + str(self.patch_size) +\
            ', ' + 'n=' + str(self.n) +\
            ', ' + 'd=' + str(self.d) +\
            ', ' + 'coeffs=' + str(self.coeffs) + ')'


class EmbedGradients(nn.Module):
    """
    Module that computes gradient embedding,
    weighted by sqrt of magnitudes of given patches.
    Args:
        patch_size: (int) Input patch size in pixels (32 is default)
        relative: (bool) absolute or relative gradients (False is default)
    Returns:
        Tensor: Gradient embedding
    Shape:
        - Input: (B, 2, patch_size, patch_size)
        - Output: (B, 7, patch_size, patch_size)
    Examples::
        >>> grads = torch.rand(23, 2, 32, 32)
        >>> emb_grads = kornia.feature.mkd.EmbedGradients(patch_size=32,
                                                          relative=False)
        >>> emb = emb_grads(grads) # 23x7x32x32
    """

    def __init__(self,
                 patch_size: int = 32,
                 relative: bool = False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.relative = relative
        self.eps = 1e-8

        # Theta kernel for gradients.
        self.kernel = VonMisesKernel(patch_size=patch_size,
                                     coeffs=COEFFS['theta'])

        # Relative gradients.
        kgrid = create_meshgrid(height=patch_size,
                                width=patch_size,
                                normalized_coordinates=True)
        _, phi = cart2pol(kgrid[:,:,:,0], kgrid[:,:,:,1])
        self.register_buffer('phi', phi.float())

    def emb_mags(self, mags: torch.Tensor) -> torch.Tensor:
        """Embed square roots of magnitudes with eps for numerical reasons. """
        mags = torch.sqrt(mags + self.eps)
        return mags

    def forward(self, grads: torch.Tensor) -> torch.Tensor:
        mags = grads[:, :1, :, :]
        oris = grads[:, 1:, :, :]
        if self.relative:
            oris = oris - self.phi
        y = self.kernel(oris) * self.emb_mags(mags)
        return y

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(' + 'patch_size=' + str(self.patch_size) +\
            ', ' + 'relative=' + str(self.relative) + ')'

def spatial_kernel_embedding(dtype, grids: dict) -> torch.Tensor:
    """Compute embeddings for cartesian and polar parametrizations. """
    factors = {"phi": 1.0, "rho": pi / sqrt2, "x": pi / 2, "y": pi / 2}
    if dtype == 'cart':
        coeffs_ = 'xy'
        params_ = ['x', 'y']
    elif dtype == 'polar':
        coeffs_ = 'rhophi'
        params_ = ['phi', 'rho']

    # Infer patch_size.
    keys = list(grids.keys())
    patch_size = grids[keys[0]].shape[-1]

    # Scale appropriately.
    grids_normed = {k:v * factors[k] for k,v in grids.items()}
    grids_normed = {k:v.unsqueeze(0).unsqueeze(0).float()
        for k,v in grids_normed.items()}

    # x,y/rho,phi kernels.
    vm_a = VonMisesKernel(patch_size=patch_size, coeffs=COEFFS[coeffs_])
    vm_b = VonMisesKernel(patch_size=patch_size, coeffs=COEFFS[coeffs_])

    emb_a = vm_a(grids_normed[params_[0]]).squeeze()
    emb_b = vm_b(grids_normed[params_[1]]).squeeze()

    # Final precomputed position embedding.
    kron_order = get_kron_order(vm_a.d, vm_b.d)
    spatial_kernel = emb_a.index_select(0,
        kron_order[:,0]) * emb_b.index_select(0, kron_order[:,1])
    return spatial_kernel


class ExplicitSpacialEncoding(nn.Module):
    """
    Module that computes explicit cartesian or polar embedding.
    Args:
        dtype: (str) Parametrization of kernel.
                     'polar', 'cart' ('polar' is default)
        fmap_size: (int) Input feature map size in pixels (32 is default)
        in_dims: (int) Dimensionality of input feature map (7 is default)
        do_gmask: (bool) Apply gaussian mask (True is default)
        do_l2: (bool) Apply l2-normalization (True is default)
    Returns:
        Tensor: Explicit cartesian or polar embedding
    Shape:
        - Input: (B, in_dims, fmap_size, fmap_size)
        - Output: (B, out_dims, fmap_size, fmap_size)
    Examples::
        >>> emb_ori = torch.rand(23, 7, 32, 32)
        >>> ese = kornia.feature.mkd.ExplicitSpacialEncoding(dtype='polar',
                                                             fmap_size=32,
                                                             in_dims=7,
                                                             do_gmask=True,
                                                             do_l2=True)
        >>> desc = ese(emb_ori) # 23x175x32x32
    """

    def __init__(self,
                 dtype: str = 'polar',
                 fmap_size: int = 32,
                 in_dims: int = 7,
                 do_gmask: bool = True,
                 do_l2: bool = True) -> None:
        super().__init__()

        self.dtype = dtype
        self.fmap_size = fmap_size
        self.in_dims = in_dims
        self.do_gmask = do_gmask
        self.do_l2 = do_l2
        self.grid = get_grid_dict(fmap_size)
        self.gmask = None

        # Precompute embedding.
        if self.dtype == 'cart':
            emb = spatial_kernel_embedding('cart', self.grid)
        elif self.dtype == 'polar':
            emb = spatial_kernel_embedding('polar', self.grid)
        else:
            raise NotImplementedError(f'{self.dtype} is not implemented.')

        # Gaussian mask.
        if self.do_gmask:
            self.gmask = self.get_gmask(sigma=1.0)
            emb = emb * self.gmask

        # Store precomputed embedding.
        self.register_buffer('emb', emb.unsqueeze(0))
        self.d_emb = self.emb.shape[1]
        self.out_dims = self.in_dims * self.d_emb
        self.odims = self.out_dims

        # Store kronecker form.
        emb2, idx1 = self.init_kron()
        self.register_buffer('emb2', emb2)
        self.register_buffer('idx1', idx1)

    def get_gmask(self, sigma: float) -> torch.Tensor:
        """Compute Gaussian mask. """
        norm_rho = self.grid['rho'] / self.grid['rho'].max()
        gmask = torch.exp(-1 * norm_rho**2 / sigma**2)
        return gmask

    def init_kron(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize helper variables to calculate kronecker. """
        kron = get_kron_order(self.in_dims, self.d_emb)
        emb2 = torch.index_select(self.emb, 1, kron[:, 1])
        return emb2, kron[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb1 = torch.index_select(x, 1, self.idx1)
        output = emb1 * self.emb2
        output = output.sum(dim=(2, 3))
        if self.do_l2:
            output = F.normalize(output, dim=1)
        return output

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(' + 'dtype=' + str(self.dtype) +\
            ', ' + 'fmap_size=' + str(self.fmap_size) +\
            ', ' + 'in_dims=' + str(self.in_dims) +\
            ', ' + 'out_dims=' + str(self.out_dims) +\
            ', ' + 'do_gmask=' + str(self.do_gmask) +\
            ', ' + 'do_l2=' + str(self.do_l2) + ')'
