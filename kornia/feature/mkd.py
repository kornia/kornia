from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.constants import pi
from kornia.core import Tensor, cos, sin, tensor, zeros
from kornia.filters import GaussianBlur2d, SpatialGradient
from kornia.geometry.conversions import cart2pol
from kornia.utils import create_meshgrid

# Precomputed coefficients for Von Mises kernel, given N and K(appa).
sqrt2: float = 1.4142135623730951
COEFFS_N1_K1: List[float] = [0.38214156, 0.48090413]
COEFFS_N2_K8: List[float] = [0.14343168, 0.268285, 0.21979234]
COEFFS_N3_K8: List[float] = [0.14343168, 0.268285, 0.21979234, 0.15838885]
COEFFS: Dict[str, List[float]] = {"xy": COEFFS_N1_K1, "rhophi": COEFFS_N2_K8, "theta": COEFFS_N3_K8}

urls: Dict[str, str] = {
    k: f"https://github.com/manyids2/mkd_pytorch/raw/master/mkd_pytorch/mkd-{k}-64.pth"
    for k in ["cart", "polar", "concat"]
}


def get_grid_dict(patch_size: int = 32) -> Dict[str, Tensor]:
    r"""Get cartesian and polar parametrizations of grid."""
    kgrid = create_meshgrid(height=patch_size, width=patch_size, normalized_coordinates=True)
    x = kgrid[0, :, :, 0]
    y = kgrid[0, :, :, 1]
    rho, phi = cart2pol(x, y)
    grid_dict = {"x": x, "y": y, "rho": rho, "phi": phi}
    return grid_dict


def get_kron_order(d1: int, d2: int) -> Tensor:
    r"""Get order for doing kronecker product."""
    kron_order = zeros([d1 * d2, 2], dtype=torch.int64)
    for i in range(d1):
        for j in range(d2):
            kron_order[i * d2 + j, 0] = i
            kron_order[i * d2 + j, 1] = j
    return kron_order


class MKDGradients(nn.Module):
    r"""Module, which computes gradients of given patches, stacked as [magnitudes, orientations].

    Given gradients $g_x$, $g_y$ with respect to $x$, $y$ respectively,
      - $\mathbox{mags} = $\sqrt{g_x^2 + g_y^2 + eps}$
      - $\mathbox{oris} = $\mbox{tan}^{-1}(\nicefrac{g_y}{g_x})$.

    Args:
        patch_size: Input patch size in pixels.

    Returns:
        gradients of given patches.

    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, 2, patch_size, patch_size)

    Example:
        >>> patches = torch.rand(23, 1, 32, 32)
        >>> gradient = MKDGradients()
        >>> g = gradient(patches) # 23x2x32x32
    """

    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-8

        self.grad = SpatialGradient(mode="diff", order=1, normalized=False)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(x)}")
        if not len(x.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect Bx1xHxW. Got: {x.shape}")
        # Modify 'diff' gradient. Before we had lambda function, but it is not jittable
        grads_xy = -self.grad(x)
        gx = grads_xy[:, :, 0, :, :]
        gy = grads_xy[:, :, 1, :, :]
        y = torch.cat(cart2pol(gx, gy, self.eps), dim=1)
        return y

    def __repr__(self) -> str:
        return self.__class__.__name__


class VonMisesKernel(nn.Module):
    r"""Module, which computes parameters of Von Mises kernel given coefficients, and embeds given patches.

    Args:
        patch_size: Input patch size in pixels.
        coeffs: List of coefficients. Some examples are hardcoded in COEFFS,

    Returns:
        Von Mises embedding of given parametrization.

    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, d, patch_size, patch_size)

    Examples:
        >>> oris = torch.rand(23, 1, 32, 32)
        >>> vm = VonMisesKernel(patch_size=32,
        ...                     coeffs=[0.14343168,
        ...                             0.268285,
        ...                             0.21979234])
        >>> emb = vm(oris) # 23x7x32x32
    """

    def __init__(self, patch_size: int, coeffs: Union[List[Union[float, int]], Tuple[Union[float, int], ...]]) -> None:
        super().__init__()

        self.patch_size = patch_size
        b_coeffs = tensor(coeffs)
        self.register_buffer("coeffs", b_coeffs)

        # Compute parameters.
        n = len(coeffs) - 1
        self.n = n
        self.d = 2 * n + 1

        # Precompute helper variables.
        emb0 = torch.ones([1, 1, patch_size, patch_size])
        frange = torch.arange(n) + 1
        frange = frange.reshape(-1, 1, 1)
        weights = zeros([2 * n + 1])
        weights[: n + 1] = torch.sqrt(b_coeffs)
        weights[n + 1 :] = torch.sqrt(b_coeffs[1:])
        weights = weights.reshape(-1, 1, 1)
        self.register_buffer("emb0", emb0)
        self.register_buffer("frange", frange)
        self.register_buffer("weights", weights)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(x)}")

        if not len(x.shape) == 4 or x.shape[1] != 1:
            raise ValueError(f"Invalid input shape, we expect Bx1xHxW. Got: {x.shape}")

        if not isinstance(self.emb0, Tensor):
            raise TypeError(f"Emb0 type is not a Tensor. Got {type(x)}")

        emb0 = self.emb0.to(x).repeat(x.size(0), 1, 1, 1)
        frange = self.frange.to(x) * x
        emb1 = cos(frange)
        emb2 = sin(frange)
        embedding = torch.cat([emb0, emb1, emb2], dim=1)
        embedding = self.weights * embedding
        return embedding

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patch_size={self.patch_size}, n={self.n}, d={self.d}, coeffs={self.coeffs})"


class EmbedGradients(nn.Module):
    r"""Module that computes gradient embedding, weighted by sqrt of magnitudes of given patches.

    Args:
        patch_size: Input patch size in pixels.
        relative: absolute or relative gradients.

    Returns:
        Gradient embedding.

    Shape:
        - Input: (B, 2, patch_size, patch_size)
        - Output: (B, 7, patch_size, patch_size)

    Examples:
        >>> grads = torch.rand(23, 2, 32, 32)
        >>> emb_grads = EmbedGradients(patch_size=32,
        ...                            relative=False)
        >>> emb = emb_grads(grads) # 23x7x32x32
    """

    def __init__(self, patch_size: int = 32, relative: bool = False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.relative = relative
        self.eps = 1e-8

        # Theta kernel for gradients.
        self.kernel = VonMisesKernel(patch_size=patch_size, coeffs=COEFFS["theta"])

        # Relative gradients.
        kgrid = create_meshgrid(height=patch_size, width=patch_size, normalized_coordinates=True)
        _, phi = cart2pol(kgrid[:, :, :, 0], kgrid[:, :, :, 1])
        self.register_buffer("phi", phi)

    def emb_mags(self, mags: Tensor) -> Tensor:
        """Embed square roots of magnitudes with eps for numerical reasons."""
        mags = torch.sqrt(mags + self.eps)
        return mags

    def forward(self, grads: Tensor) -> Tensor:
        if not isinstance(grads, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(grads)}")
        if not len(grads.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect Bx2xHxW. Got: {grads.shape}")
        mags = grads[:, :1, :, :]
        oris = grads[:, 1:, :, :]
        if self.relative:
            oris = oris - self.phi.to(oris)
        y = self.kernel(oris) * self.emb_mags(mags)
        return y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patch_size={self.patch_size}, relative={self.relative})"


def spatial_kernel_embedding(kernel_type: str, grids: Dict[str, Tensor]) -> Tensor:
    r"""Compute embeddings for cartesian and polar parametrizations."""
    factors = {"phi": 1.0, "rho": pi / sqrt2, "x": pi / 2, "y": pi / 2}
    if kernel_type == "cart":
        coeffs_ = "xy"
        params_ = ["x", "y"]
    elif kernel_type == "polar":
        coeffs_ = "rhophi"
        params_ = ["phi", "rho"]

    # Infer patch_size.
    keys = list(grids.keys())
    patch_size = grids[keys[0]].shape[-1]

    # Scale appropriately.
    grids_normed = {k: v * factors[k] for k, v in grids.items()}
    grids_normed = {k: v.unsqueeze(0).unsqueeze(0).float() for k, v in grids_normed.items()}

    # x,y/rho,phi kernels.
    vm_a = VonMisesKernel(patch_size=patch_size, coeffs=COEFFS[coeffs_])
    vm_b = VonMisesKernel(patch_size=patch_size, coeffs=COEFFS[coeffs_])

    emb_a = vm_a(grids_normed[params_[0]]).squeeze()
    emb_b = vm_b(grids_normed[params_[1]]).squeeze()

    # Final precomputed position embedding.
    kron_order = get_kron_order(vm_a.d, vm_b.d)
    spatial_kernel = emb_a.index_select(0, kron_order[:, 0]) * emb_b.index_select(0, kron_order[:, 1])
    return spatial_kernel


class ExplicitSpacialEncoding(nn.Module):
    r"""Module that computes explicit cartesian or polar embedding.

    Args:
        kernel_type: Parametrization of kernel ``'polar'`` or ``'cart'``.
        fmap_size: Input feature map size in pixels.
        in_dims: Dimensionality of input feature map.
        do_gmask: Apply gaussian mask.
        do_l2: Apply l2-normalization.

    Returns:
        Explicit cartesian or polar embedding.

    Shape:
        - Input: (B, in_dims, fmap_size, fmap_size)
        - Output: (B, out_dims, fmap_size, fmap_size)

    Example:
        >>> emb_ori = torch.rand(23, 7, 32, 32)
        >>> ese = ExplicitSpacialEncoding(kernel_type='polar',
        ...                               fmap_size=32,
        ...                               in_dims=7,
        ...                               do_gmask=True,
        ...                               do_l2=True)
        >>> desc = ese(emb_ori) # 23x175x32x32
    """

    def __init__(
        self,
        kernel_type: str = "polar",
        fmap_size: int = 32,
        in_dims: int = 7,
        do_gmask: bool = True,
        do_l2: bool = True,
    ) -> None:
        super().__init__()

        if kernel_type not in ["polar", "cart"]:
            raise NotImplementedError(f"{kernel_type} is not valid, use polar or cart).")

        self.kernel_type = kernel_type
        self.fmap_size = fmap_size
        self.in_dims = in_dims
        self.do_gmask = do_gmask
        self.do_l2 = do_l2
        self.grid = get_grid_dict(fmap_size)
        self.gmask = None

        # Precompute embedding.
        emb = spatial_kernel_embedding(self.kernel_type, self.grid)

        # Gaussian mask.
        if self.do_gmask:
            self.gmask = self.get_gmask(sigma=1.0)
            emb = emb * self.gmask

        # Store precomputed embedding.
        self.register_buffer("emb", emb.unsqueeze(0))
        self.d_emb: int = emb.shape[0]
        self.out_dims: int = self.in_dims * self.d_emb
        self.odims: int = self.out_dims

        # Store kronecker form.
        emb2, idx1 = self.init_kron()
        self.register_buffer("emb2", emb2)
        self.register_buffer("idx1", idx1)

    def get_gmask(self, sigma: float) -> Tensor:
        """Compute Gaussian mask."""
        norm_rho = self.grid["rho"] / self.grid["rho"].max()
        gmask = torch.exp(-1 * norm_rho**2 / sigma**2)
        return gmask

    def init_kron(self) -> Tuple[Tensor, Tensor]:
        """Initialize helper variables to calculate kronecker."""
        kron = get_kron_order(self.in_dims, self.d_emb)
        _emb = torch.jit.annotate(Tensor, self.emb)
        emb2 = torch.index_select(_emb, 1, kron[:, 1])
        return emb2, kron[:, 0]

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(x)}")
        if not ((len(x.shape) == 4) | (x.shape[1] == self.in_dims)):
            raise ValueError(f"Invalid input shape, we expect Bx{self.in_dims}xHxW. Got: {x.shape}")
        idx1 = torch.jit.annotate(Tensor, self.idx1)
        emb1 = torch.index_select(x, 1, idx1)
        output = emb1 * self.emb2
        output = output.sum(dim=(2, 3))
        if self.do_l2:
            output = F.normalize(output, dim=1)
        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"kernel_type={self.kernel_type}, "
            f"fmap_size={self.fmap_size}, "
            f"in_dims={self.in_dims}, "
            f"out_dims={self.out_dims}, "
            f"do_gmask={self.do_gmask}, "
            f"do_l2={self.do_l2})"
        )


class Whitening(nn.Module):
    r"""Module, performs supervised or unsupervised whitening.

    This is based on the paper "Understanding and Improving Kernel Local Descriptors".
    See :cite:`mukundan2019understanding` for more details.

    Args:
        xform: Variant of whitening to use. None, 'lw', 'pca', 'pcaws', 'pcawt'.
        whitening_model: Dictionary with keys 'mean', 'eigvecs', 'eigvals' holding Tensors.
        in_dims: Dimensionality of input descriptors.
        output_dims: (int) Dimensionality reduction.
        keval: Shrinkage parameter.
        t: Attenuation parameter.

    Returns:
        l2-normalized, whitened descriptors.

    Shape:
        - Input: (B, in_dims, fmap_size, fmap_size)
        - Output: (B, out_dims, fmap_size, fmap_size)

    Examples:
        >>> descs = torch.rand(23, 238)
        >>> whitening_model = {'pca': {'mean': torch.zeros(238),
        ...                            'eigvecs': torch.eye(238),
        ...                            'eigvals': torch.ones(238)}}
        >>> whitening = Whitening(xform='pcawt',
        ...                       whitening_model=whitening_model,
        ...                       in_dims=238,
        ...                       output_dims=128,
        ...                       keval=40,
        ...                       t=0.7)
        >>> wdescs = whitening(descs) # 23x128
    """

    def __init__(
        self,
        xform: str,
        whitening_model: Union[Dict[str, Dict[str, Tensor]], None],
        in_dims: int,
        output_dims: int = 128,
        keval: int = 40,
        t: float = 0.7,
    ) -> None:
        super().__init__()

        self.xform = xform
        self.in_dims = in_dims
        self.keval = keval
        self.t = t
        self.pval = 1.0

        # Compute true output_dims.
        output_dims = min(output_dims, in_dims)
        self.output_dims = output_dims

        # Initialize identity transform.
        self.mean = nn.Parameter(zeros(in_dims), requires_grad=True)
        self.evecs = nn.Parameter(torch.eye(in_dims)[:, :output_dims], requires_grad=True)
        self.evals = nn.Parameter(torch.ones(in_dims)[:output_dims], requires_grad=True)

        if whitening_model is not None:
            self.load_whitening_parameters(whitening_model)

    def load_whitening_parameters(self, whitening_model: Dict[str, Dict[str, Tensor]]) -> None:
        algo = "lw" if self.xform == "lw" else "pca"
        wh_model = whitening_model[algo]
        self.mean.data = wh_model["mean"]
        self.evecs.data = wh_model["eigvecs"][:, : self.output_dims]
        self.evals.data = wh_model["eigvals"][: self.output_dims]

        modifications = {
            "pca": self._modify_pca,
            "lw": self._modify_lw,
            "pcaws": self._modify_pcaws,
            "pcawt": self._modify_pcawt,
        }

        # Call modification.
        modifications[self.xform]()

    def _modify_pca(self) -> None:
        """Modify powerlaw parameter."""
        self.pval = 0.5

    def _modify_lw(self) -> None:
        """No modification required."""

    def _modify_pcaws(self) -> None:
        """Shrinkage for eigenvalues."""
        alpha = self.evals[self.keval]
        evals = ((1 - alpha) * self.evals) + alpha
        self.evecs.data = self.evecs @ torch.diag(torch.pow(evals, -0.5))

    def _modify_pcawt(self) -> None:
        """Attenuation for eigenvalues."""
        m = -0.5 * self.t
        self.evecs.data = self.evecs @ torch.diag(torch.pow(self.evals, m))

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(x)}")
        if not len(x.shape) == 2:
            raise ValueError(f"Invalid input shape, we expect NxD. Got: {x.shape}")
        x = x - self.mean  # Center the data.
        x = x @ self.evecs  # Apply rotation and/or scaling.
        x = torch.sign(x) * torch.pow(torch.abs(x), self.pval)  # Powerlaw.
        return F.normalize(x, dim=1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(xform={self.xform}, in_dims={self.in_dims}, output_dims={self.output_dims})"


class MKDDescriptor(nn.Module):
    r"""Module that computes Multiple Kernel local descriptors.

    This is based on the paper "Understanding and Improving Kernel Local Descriptors".
    See :cite:`mukundan2019understanding` for more details.

    Args:
        patch_size: Input patch size in pixels.
        kernel_type: Parametrization of kernel ``'concat'``, ``'cart'``, ``'polar'``.
        whitening: Whitening transform to apply ``None``, ``'lw'``, ``'pca'``, ``'pcawt'``, ``'pcaws'``.
        training_set: Set that model was trained on ``'liberty'``, ``'notredame'``, ``'yosemite'``.
        output_dims: Dimensionality reduction.

    Returns:
        Explicit cartesian or polar embedding.

    Shape:
        - Input: :math:`(B, in_{dims}, fmap_{size}, fmap_{size})`.
        - Output: :math:`(B, out_{dims}, fmap_{size}, fmap_{size})`,

    Examples:
        >>> patches = torch.rand(23, 1, 32, 32)
        >>> mkd = MKDDescriptor(patch_size=32,
        ...                     kernel_type='concat',
        ...                     whitening='pcawt',
        ...                     training_set='liberty',
        ...                     output_dims=128)
        >>> desc = mkd(patches) # 23x128
    """

    def __init__(
        self,
        patch_size: int = 32,
        kernel_type: str = "concat",
        whitening: str = "pcawt",
        training_set: str = "liberty",
        output_dims: int = 128,
    ) -> None:
        super().__init__()

        self.patch_size: int = patch_size
        self.kernel_type: str = kernel_type
        self.whitening: str = whitening
        self.training_set: str = training_set

        self.sigma = 1.4 * (patch_size / 64)
        self.smoothing = GaussianBlur2d((5, 5), (self.sigma, self.sigma), "replicate")
        self.gradients = MKDGradients()
        # This stupid thing needed for jitting...
        polar_s: str = "polar"
        cart_s: str = "cart"
        self.parametrizations = [polar_s, cart_s] if self.kernel_type == "concat" else [self.kernel_type]

        # Initialize cartesian/polar embedding with absolute/relative gradients.
        self.odims: int = 0
        relative_orientations = {polar_s: True, cart_s: False}
        self.feats = {}
        for parametrization in self.parametrizations:
            gradient_embedding = EmbedGradients(patch_size=patch_size, relative=relative_orientations[parametrization])
            spatial_encoding = ExplicitSpacialEncoding(
                kernel_type=parametrization, fmap_size=patch_size, in_dims=gradient_embedding.kernel.d
            )

            self.feats[parametrization] = nn.Sequential(gradient_embedding, spatial_encoding)
            self.odims += spatial_encoding.odims
        # Compute true output_dims.
        self.output_dims: int = min(output_dims, self.odims)

        # Load supervised(lw)/unsupervised(pca) model trained on training_set.
        if self.whitening is not None:
            whitening_models = torch.hub.load_state_dict_from_url(
                urls[self.kernel_type], map_location=torch.device("cpu")
            )
            whitening_model = whitening_models[training_set]
            self.whitening_layer = Whitening(
                whitening, whitening_model, in_dims=self.odims, output_dims=self.output_dims
            )
            self.odims = self.output_dims
        self.eval()

    def forward(self, patches: Tensor) -> Tensor:
        if not isinstance(patches, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(patches)}")
        if not len(patches.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect Bx1xHxW. Got: {patches.shape}")
        # Extract gradients.
        g = self.smoothing(patches)
        g = self.gradients(g)

        # Extract polar/cart features.
        features = []
        for parametrization in self.parametrizations:
            self.feats[parametrization].to(g.device)
            features.append(self.feats[parametrization](g))

        # Concatenate.
        y = torch.cat(features, dim=1)

        # l2-normalize.
        y = F.normalize(y, dim=1)

        # Whiten descriptors.
        if self.whitening is not None:
            y = self.whitening_layer(y)

        return y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"patch_size={self.patch_size}, "
            f"kernel_type={self.kernel_type}, "
            f"whitening={self.whitening}, "
            f"training_set={self.training_set}, "
            f"output_dims={self.output_dims})"
        )


def load_whitening_model(kernel_type: str, training_set: str) -> Dict[str, Any]:
    whitening_models = torch.hub.load_state_dict_from_url(urls[kernel_type], map_location=torch.device("cpu"))
    whitening_model = whitening_models[training_set]
    return whitening_model


class SimpleKD(nn.Module):
    """Example to write custom Kernel Descriptors."""

    def __init__(
        self,
        patch_size: int = 32,
        kernel_type: str = "polar",  # 'cart' 'polar'
        whitening: str = "pcawt",  # 'lw', 'pca', 'pcaws', 'pcawt
        training_set: str = "liberty",  # 'liberty', 'notredame', 'yosemite'
        output_dims: int = 128,
    ) -> None:
        super().__init__()

        relative: bool = kernel_type == "polar"
        sigma: float = 1.4 * (patch_size / 64)
        self.patch_size = patch_size
        # Sequence of modules.
        smoothing = GaussianBlur2d((5, 5), (sigma, sigma), "replicate")
        gradients = MKDGradients()
        ori = EmbedGradients(patch_size=patch_size, relative=relative)
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=patch_size, in_dims=ori.kernel.d)
        wh = Whitening(
            whitening, load_whitening_model(kernel_type, training_set), in_dims=ese.odims, output_dims=output_dims
        )

        self.features = nn.Sequential(smoothing, gradients, ori, ese, wh)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)
