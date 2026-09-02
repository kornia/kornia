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

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import angle_to_rotation_matrix, convert_points_from_homogeneous
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import pyrdown


def get_laf_scale(LAF: torch.Tensor) -> torch.Tensor:
    """Return a scale of the LAFs.

    Args:
        LAF: :math:`(B, N, 2, 3)`

    Returns:
        scale :math:`(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_scale(input)  # BxNx1x1

    """
    KORNIA_CHECK_LAF(LAF)
    eps = 1e-10
    out = LAF[..., 0:1, 0:1] * LAF[..., 1:2, 1:2] - LAF[..., 1:2, 0:1] * LAF[..., 0:1, 1:2] + eps
    return out.abs().sqrt()


def laf_is_valid(laf: torch.Tensor) -> torch.Tensor:
    """Check that each LAF is finite and has a finite, nonzero determinant.

    Args:
        laf: :math:`(B, N, 2, 3)`.

    Returns:
        validity mask :math:`(B, N)`.

    Example:
        >>> laf = torch.eye(2, 3).view(1, 1, 2, 3)
        >>> laf_is_valid(laf)
        tensor([[True]])

    """
    KORNIA_CHECK_LAF(laf)
    det = laf[..., 0, 0] * laf[..., 1, 1] - laf[..., 1, 0] * laf[..., 0, 1]
    return laf.isfinite().all(dim=-1).all(dim=-1) & det.isfinite() & (det != 0)


def get_laf_center(LAF: torch.Tensor) -> torch.Tensor:
    """Return a center (keypoint) of the LAFs.

    The convention is that center of 5-pixel image (coordinates from 0 to 4) is 2, and not 2.5.

    Args:
        LAF: :math:`(B, N, 2, 3)`

    Returns:
        xy :math:`(B, N, 2)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_center(input)  # BxNx2

    """
    KORNIA_CHECK_LAF(LAF)
    out = LAF[..., 2]
    return out


def get_laf_orientation(LAF: torch.Tensor) -> torch.Tensor:
    """Return orientation of the LAFs, in degrees.

    Args:
        LAF: :math:`(B, N, 2, 3)`

    Returns:
        angle in degrees :math:`(B, N, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_orientation(input)  # BxNx1

    """
    KORNIA_CHECK_LAF(LAF)
    angle_rad = torch.atan2(LAF[..., 0, 1], LAF[..., 0, 0])
    return torch.rad2deg(angle_rad).unsqueeze(-1)


def rotate_laf(LAF: torch.Tensor, angles_degrees: torch.Tensor) -> torch.Tensor:
    """Apply additional rotation to the LAFs.

    Compared to `set_laf_orientation`, the resulting rotation is original LAF orientation plus angles_degrees.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        angles_degrees: :math:`(B, N, 1)` in degrees.

    Returns:
        LAF oriented with angles :math:`(B, N, 2, 3)`

    """
    KORNIA_CHECK_LAF(LAF)
    B, N = LAF.shape[:2]
    rotmat = angle_to_rotation_matrix(angles_degrees).view(B * N, 2, 2)
    out_laf = LAF.clone()
    out_laf[:, :, :2, :2] = torch.bmm(LAF[:, :, :2, :2].reshape(B * N, 2, 2), rotmat).reshape(B, N, 2, 2)
    return out_laf


def set_laf_orientation(LAF: torch.Tensor, angles_degrees: torch.Tensor) -> torch.Tensor:
    """Change the orientation of the LAFs.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        angles_degrees: :math:`(B, N, 1)` in degrees.

    Returns:
        LAF oriented with angles :math:`(B, N, 2, 3)`

    """
    KORNIA_CHECK_LAF(LAF)
    _B, _N = LAF.shape[:2]
    ori = get_laf_orientation(LAF).reshape_as(angles_degrees)
    return rotate_laf(LAF, angles_degrees - ori)


def laf_from_center_scale_ori(
    xy: torch.Tensor, scale: Optional[torch.Tensor] = None, ori: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Create a LAF from keypoint center, scale and orientation.

    Useful to create kornia LAFs from OpenCV keypoints.

    Args:
        xy: :math:`(B, N, 2)`.
        scale: :math:`(B, N, 1, 1)`. If not provided, scale = 1.0 is assumed
        ori: angle in degrees :math:`(B, N, 1)`. If not provided orientation = 0 is assumed

    Returns:
        LAF :math:`(B, N, 2, 3)`

    """
    KORNIA_CHECK_SHAPE(xy, ["B", "N", "2"])
    device = xy.device
    dtype = xy.dtype
    B, N = xy.shape[:2]
    if scale is None:
        scale = torch.ones(B, N, 1, 1, device=device, dtype=dtype)
    if ori is None:
        ori = torch.zeros(B, N, 1, device=device, dtype=dtype)
    KORNIA_CHECK_SHAPE(scale, ["B", "N", "1", "1"])
    KORNIA_CHECK_SHAPE(ori, ["B", "N", "1"])
    unscaled_laf = torch.cat([angle_to_rotation_matrix(ori.squeeze(-1)), xy.unsqueeze(-1)], dim=-1)
    laf = scale_laf(unscaled_laf, scale)
    return laf


def scale_laf(laf: torch.Tensor, scale_coef: Union[float, torch.Tensor]) -> torch.Tensor:
    """Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.

    So the center, shape and orientation of the local feature stays the same, but the region area changes.

    Args:
        laf: :math:`(B, N, 2, 3)`
        scale_coef: broadcastable torch.Tensor or float.

    Returns:
        LAF :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> scale = 0.5
        >>> output = scale_laf(input, scale)  # BxNx2x3

    """
    if not isinstance(scale_coef, (float, torch.Tensor)):
        raise TypeError(f"scale_coef should be float or torch.Tensor. Got {type(scale_coef)}")
    KORNIA_CHECK_LAF(laf)
    centerless_laf = laf[:, :, :2, :2]
    return torch.cat([scale_coef * centerless_laf, laf[:, :, :, 2:]], dim=3)


def make_upright(laf: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Rectify the affine matrix, so that it becomes upright.

    Args:
        laf: :math:`(B, N, 2, 3)`
        eps: for safe division.

    Returns:
        laf: :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = make_upright(input)  #  BxNx2x3

    """
    KORNIA_CHECK_LAF(laf)
    det = get_laf_scale(laf)
    scale = det
    # The function is equivalent to doing 2x2 SVD and resetting rotation
    # matrix to an identity: U, S, V = svd(LAF); LAF_upright = U * S.
    b2a2 = torch.sqrt(laf[..., 0:1, 1:2] ** 2 + laf[..., 0:1, 0:1] ** 2) + eps
    laf1_ell = torch.cat([(b2a2 / det).contiguous(), torch.zeros_like(det)], dim=3)
    laf2_ell = torch.cat(
        [
            ((laf[..., 1:2, 1:2] * laf[..., 0:1, 1:2] + laf[..., 1:2, 0:1] * laf[..., 0:1, 0:1]) / (b2a2 * det)),
            (det / b2a2).contiguous(),
        ],
        dim=3,
    )
    laf_unit_scale = torch.cat([torch.cat([laf1_ell, laf2_ell], dim=2), laf[..., :, 2:3]], dim=3)
    return scale_laf(laf_unit_scale, scale)


def ellipse_to_laf(ells: torch.Tensor) -> torch.Tensor:
    """Convert ellipse regions to LAF format.

    Ellipse (a, b, c) and upright covariance matrix [a11 a12; 0 a22] are connected
    by inverse matrix square root: A = invsqrt([a b; b c]).

    See also https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_frame2oell.m

    Args:
        ells: torch.Tensor :math:`(B, N, 5)` of ellipses in Oxford format [x y a b c].

    Returns:
        LAF :math:`(B, N, 2, 3)`

    Note:
        A degenerate ellipse -- one whose ``a`` or ``c`` is ``0`` after rounding to ``ells.dtype`` --
        describes an unbounded strip rather than a bounded region, and makes the matrix being inverted
        singular. Its LAF is non-finite: ``inf`` always appears on the diagonal, while ``nan`` appears
        only in the sub-case where the off-diagonal ``b`` is exactly ``0`` (``0 * inf``) -- the generic
        degenerate ellipse is ``inf``-only, so screen results with :func:`laf_is_valid` rather than an
        ``isnan`` test, which misses it. :func:`get_laf_scale` of such a LAF is non-finite as
        well. The conversion does not raise. Rounding is part of the condition: in ``float16`` an ``a``
        below roughly ``3e-8`` (half the smallest subnormal) rounds to ``0``, and a backend that
        flushes subnormals to zero raises that cutoff to the smallest normal, about ``6e-5``.

    Example:
        >>> input = torch.ones(1, 10, 5)  # BxNx5
        >>> output = ellipse_to_laf(input)  #  BxNx2x3

    """
    KORNIA_CHECK_SHAPE(ells, ["B", "N", "5"])
    B, N, _ = ells.shape
    # Previous implementation was incorrectly using Cholesky decomp as matrix sqrt
    # ell_shape = torch.cat([torch.cat([ells[..., 2:3], ells[..., 3:4]], dim=2).unsqueeze(2),
    #                       torch.cat([ells[..., 3:4], ells[..., 4:5]], dim=2).unsqueeze(2)], dim=2).view(-1, 2, 2)
    # out = torch.matrix_power(torch.cholesky(ell_shape, False), -1).view(B, N, 2, 2)

    # We will calculate 2x2 matrix square root via special case formula
    # https://en.wikipedia.org/wiki/Square_root_of_a_matrix
    # "The Cholesky factorization provides another particular example of square root
    #  which should not be confused with the unique non-negative square root."
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # M = (A 0; C D)
    # R = (sqrt(A) 0; C / (sqrt(A)+sqrt(D)) sqrt(D))
    a11 = ells[..., 2:3].abs().sqrt()
    a22 = ells[..., 4:5].abs().sqrt()
    a21 = ells[..., 3:4] / (a11 + a22)
    # The matrix [[a11, 0], [a21, a22]] is lower-triangular, so its inverse is the closed form
    # [[1/a11, 0], [-a21/(a11*a22), 1/a22]] — no batched torch.inverse, which is orders of
    # magnitude slower, unsupported in float16/bfloat16 on CPU, and pathological on MPS.
    inv11 = 1.0 / a11
    inv22 = 1.0 / a22
    # Divide by the product of the roots instead of multiplying the reciprocals: every ordering of
    # `-a21 * inv11 * inv22` has an input region where an intermediate overflows to inf (or a
    # mathematically-zero off-diagonal becomes 0 * inf = nan) or flushes a representable result to
    # zero. a11 * a22 = sqrt(a) * sqrt(c) can neither overflow nor round to zero, so the single
    # division is correctly rounded (not exact) wherever the product is a normal number; when the
    # product is subnormal the result loses precision but is never a corrupted zero, inf, or nan.
    # What remains non-finite is exactly the singular ellipse, which we deliberately do not guard.
    inv21 = -a21 / (a11 * a22)
    A = torch.stack([inv11, torch.zeros_like(inv11), inv21, inv22], dim=-1).view(B, N, 2, 2)
    out = torch.cat([A, ells[..., :2].view(B, N, 2, 1)], dim=3)
    return out


def laf_to_boundary_points(LAF: torch.Tensor, n_pts: int = 50) -> torch.Tensor:
    """Convert LAFs to boundary points of the regions + center.

    Used for local features visualization, see visualize_laf function.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        n_pts: number of points to output.

    Returns:
        torch.Tensor of boundary points LAF: :math:`(B, N, n_pts, 2)`

    """
    KORNIA_CHECK_LAF(LAF)
    B, N, _, _ = LAF.size()
    pts = torch.cat(
        [
            torch.sin(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
            torch.cos(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
            torch.ones(n_pts - 1, 1),
        ],
        dim=1,
    )
    # Add origin to draw also the orientation
    pts = torch.cat([torch.tensor([0.0, 0.0, 1.0]).view(1, 3), pts], dim=0).unsqueeze(0).expand(B * N, n_pts, 3)
    pts = pts.to(LAF.device).to(LAF.dtype)
    aux = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).expand(B * N, 1, 3)
    HLAF = torch.cat([LAF.view(-1, 2, 3), aux.to(LAF.device).to(LAF.dtype)], dim=1)
    pts_h = torch.bmm(HLAF, pts.permute(0, 2, 1)).permute(0, 2, 1)
    return convert_points_from_homogeneous(pts_h.view(B, N, n_pts, 3))


def get_laf_pts_to_draw(LAF: torch.Tensor, img_idx: int = 0) -> Tuple[List[int], List[int]]:
    """Return list for drawing LAFs (local features).

    Args:
        LAF: :math:`(B, N, 2, 3)`
        img_idx: which points to output.

    Returns:
        List of boundary points x, y`

    Examples:
        x, y = get_laf_pts_to_draw(LAF, img_idx)
        plt.figure()
        plt.imshow(kornia.image.tensor_to_image(img[img_idx]))
        plt.plot(x, y, 'r')
        plt.show()

    """
    # TODO: Refactor doctest
    KORNIA_CHECK_LAF(LAF)
    pts = laf_to_boundary_points(LAF[img_idx : img_idx + 1])[0]
    pts_np = pts.detach().permute(1, 0, 2).cpu()
    return (pts_np[..., 0].tolist(), pts_np[..., 1].tolist())


def denormalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Denormalize LAFs from the [0, 1] scale to image (pixel) scale.

    The convention is that center of 5-pixel image (coordinates from 0 to 4) is 2, and not 2.5.

        B,CH,H,W = images.size()
        MIN_SIZE = min(H - 1, W -1)
        [a11 a12 x]
        [a21 a22 y]
        becomes
        [a11*MIN_SIZE a12*MIN_SIZE x*(W-1)]
        [a21*MIN_SIZE a22*MIN_SIZE y*(H-1)]

    A singleton axis (``H == 1`` or ``W == 1``) has no spatial extent and counts as one pixel, so
    the conversion stays finite and round-trips with :func:`normalize_laf`.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        images: :math:`(B, CH, H, W)`

    Returns:
        the denormalized LAF: :math:`(B, N, 2, 3)`, scale in pixels

    """
    KORNIA_CHECK_LAF(LAF)
    _, _, h, w = images.size()
    # A singleton image axis has a single valid coordinate and therefore no spatial extent.
    # Treating it as one pixel wide keeps the conversion finite and round-trippable with
    # `normalize_laf`, instead of collapsing every LAF to zero here and raising a
    # `ZeroDivisionError` there.
    wf = float(max(w - 1, 1))
    hf = float(max(h - 1, 1))
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3, dtype=LAF.dtype, device=LAF.device) * min_size
    coef[0, 0, 0, 2] = wf
    coef[0, 0, 1, 2] = hf
    return coef.expand_as(LAF) * LAF


def normalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Normalize LAFs to [0,1] scale from pixel scale.

    See below:
        B,CH,H,W = images.size()
        MIN_SIZE =  min(H - 1, W -1)
        [a11 a12 x]
        [a21 a22 y]
        becomes:
        [a11/MIN_SIZE a12/MIN_SIZE x/(W-1)]
        [a21/MIN_SIZE a22/MIN_SIZE y/(H-1)]

    A singleton axis (``H == 1`` or ``W == 1``) has no spatial extent and counts as one pixel, so
    the conversion stays finite instead of dividing by zero.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        images: :math:`(B, CH, H, W)`

    Returns:
        the normalized LAF: :math:`(B, N, 2, 3)`, scale in image percentage (0, 1)

    """
    KORNIA_CHECK_LAF(LAF)
    _, _, h, w = images.size()
    # See `denormalize_laf`: a singleton axis counts as one pixel of extent, so a 1-pixel-wide or
    # 1-pixel-tall image normalizes finitely instead of dividing by zero.
    wf = float(max(w - 1, 1))
    hf = float(max(h - 1, 1))
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3, dtype=LAF.dtype, device=LAF.device) / min_size
    coef[0, 0, 0, 2] = 1.0 / wf
    coef[0, 0, 1, 2] = 1.0 / hf
    return coef.expand_as(LAF) * LAF


def generate_patch_grid_from_normalized_LAF(img: torch.Tensor, LAF: torch.Tensor, PS: int = 32) -> torch.Tensor:
    """Generate affine grid.

    Args:
        img: image torch.Tensor of shape :math:`(B, CH, H, W)`.
        LAF: laf with shape :math:`(B, N, 2, 3)`.
        PS: patch size to be extracted.

    Returns:
        grid :math:`(B*N, PS, PS, 2)`

    """
    KORNIA_CHECK_LAF(LAF)
    B, N, _, _ = LAF.size()
    _, ch, h, w = img.size()

    # norm, then renorm is needed for allowing detection on one resolution
    # and extraction at arbitrary other
    LAF_renorm = denormalize_laf(LAF, img)

    grid = F.affine_grid(LAF_renorm.view(B * N, 2, 3), [B * N, ch, PS, PS], align_corners=False)
    # A singleton axis has no spatial extent; one pixel of denominator keeps the grid finite and
    # lets the border padding return that single pixel (see `denormalize_laf`).
    grid[..., :, 0] = 2.0 * grid[..., :, 0].clone() / float(max(w - 1, 1)) - 1.0
    grid[..., :, 1] = 2.0 * grid[..., :, 1].clone() / float(max(h - 1, 1)) - 1.0
    return grid


def _clamp_grid_to_pixel_centers(grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    r"""Clamp a normalized sampling grid to the outermost pixel centers.

    MPS does not implement ``padding_mode="border"`` for :func:`torch.nn.functional.grid_sample`,
    so the border behavior has to be emulated with zero padding plus a clamped grid. Clamping to
    :math:`\pm 1` is *not* equivalent: with ``align_corners=False`` that is the outer **edge** of
    the border pixel, i.e. pixel index :math:`-0.5`, where bilinear sampling blends the border
    pixel with the zero padding and returns roughly half its value. The outermost pixel **center**
    sits at :math:`\pm (1 - 1/\text{size})`, and clamping there reproduces ``padding_mode="border"``
    exactly.

    Args:
        grid: sampling grid :math:`(..., 2)` with any leading batch dimensions, last dimension ordered
            ``(x, y)``. Only 2-D sampling is handled; a 3-D grid would need a third bound from the
            depth of the sampled volume.
        h: height of the sampled image.
        w: width of the sampled image.

    Returns:
        the clamped grid, same shape as ``grid``.

    """
    x = grid[..., 0].clamp(-1.0 + 1.0 / float(w), 1.0 - 1.0 / float(w))
    y = grid[..., 1].clamp(-1.0 + 1.0 / float(h), 1.0 - 1.0 / float(h))
    return torch.stack([x, y], dim=-1)


def _grid_sample_patches(img: torch.Tensor, grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    r"""Run ``grid_sample`` with border padding, robust across devices.

    MPS does not implement ``padding_mode="border"``; it is emulated with a clamped grid plus
    zero padding (see :func:`_clamp_grid_to_pixel_centers`). ``img`` and ``grid`` share a dtype:
    the extractors upcast reduced-precision inputs to float32 once, before their chunk loops.
    That upcast is deliberate on every torch version, not only the torch <= 2.9 builds whose
    float16/bfloat16 CPU kernel reads out of bounds at the border — half-precision sampling
    coordinates also quantize to whole pixels on large images. It is a real trade on CUDA, where
    the native half kernels are fine: float16 ``extract_patches_simple`` pays roughly a 2x
    slowdown at high N for the accuracy.
    """
    if img.device.type == "mps":
        return F.grid_sample(img, _clamp_grid_to_pixel_centers(grid, h, w), padding_mode="zeros", align_corners=False)
    return F.grid_sample(img, grid, padding_mode="border", align_corners=False)


def _grid_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the sampling-grid dtype for an image or LAF dtype."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _promoted_grid_dtype(image_dtype: torch.dtype, laf_dtype: torch.dtype) -> torch.dtype:
    """Return a grid dtype that never discards image or LAF coordinate precision."""
    return _grid_dtype(torch.promote_types(image_dtype, laf_dtype))


def _grid_elem_bytes(dtype: torch.dtype) -> int:
    """Bytes per element after applying the sampling-grid dtype policy."""
    return 8 if _grid_dtype(dtype) == torch.float64 else 4


def _grid_chunk_lafs(B: int, N: int, ch: int, PS: int, elem_size: int, budget: int = 64 * 1024 * 1024) -> int:
    r"""Largest LAF count whose folded grid and sampled chunk each fit the byte budget.

    The sampling grid and its pointwise intermediates scale with :math:`B \cdot N \cdot PS^2`,
    so extraction is chunked along N to bound peak memory, in the spirit of
    :func:`kornia.core.utils.batched_forward`. Both the two-coordinate folded grid and the
    channel-scaled ``grid_sample`` result are individually kept within the budget; pointwise remap
    temporaries can still make the real peak a multiple of it. At least one LAF per chunk; when
    everything fits the budget the loop degenerates to the single-call fast path. The default
    budget is a defaulted argument rather than a module constant because TorchScript cannot close
    over module-level ints; tests monkeypatch this function to force multi-chunk sampling on small
    inputs.
    """
    per_laf = B * PS * PS * max(2, ch) * elem_size
    return min(N, max(1, budget // per_laf))


def _pyramid_atlas_fits(atlas_bytes: int, budget: int = 128 * 1024 * 1024) -> bool:
    """Return whether a packed pyramid atlas fits its default memory budget.

    The budget is a defaulted argument because TorchScript cannot close over module-level integer
    constants. Keeping the policy in a helper also gives tests a narrow seam for forcing either
    extraction path.
    """
    return atlas_bytes <= budget


def _sample_patches(img: torch.Tensor, grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    r"""Sample one patch per LAF with a single ``grid_sample`` call.

    The per-LAF grids :math:`(B, N, PS, PS, 2)` are folded to :math:`(B, N \cdot PS, PS, 2)` so
    one call covers the whole batch, without a Python loop over B or a ``(B*N, CH, H, W)`` input
    copy. ``grid`` must already live on ``img``'s device: the extractors move a cross-device LAF
    to the image's device up front, so the grid is built there.

    Returns:
        patches :math:`(B, N, CH, PS, PS)` as a permuted view; assigning it or calling
        ``.contiguous()`` makes the copy.
    """
    B, N, PS = grid.size(0), grid.size(1), grid.size(2)
    ch = img.size(1)
    folded = grid.view(B, N * PS, PS, 2)
    return _grid_sample_patches(img, folded, h, w).view(B, ch, N, PS, PS).permute(0, 2, 1, 3, 4)


def _extract_patches_from_pyramid_levelwise(
    img: torch.Tensor,
    nlaf: torch.Tensor,
    pyr_idx: torch.Tensor,
    heights: List[int],
    widths: List[int],
    PS: int,
) -> torch.Tensor:
    """Sample each pyramid level separately when a single atlas would be too large.

    Patches accumulate on the image's device and dtype, like the atlas path: the callers move a
    cross-device LAF to the image's device before the level index is derived, and
    reduced-precision levels are upcast once, before the loops, so no full level is re-cast per
    chunk. Out-of-range level indices (the -1 marking a non-finite LAF) match no level and keep
    their zero patch.
    """
    B, N = nlaf.shape[:2]
    ch = img.shape[1]
    grid_dtype = _promoted_grid_dtype(img.dtype, nlaf.dtype)
    grid_laf = nlaf.to(grid_dtype) if nlaf.dtype != grid_dtype else nlaf
    out = torch.zeros(B, N, ch, PS, PS, dtype=img.dtype, device=img.device)
    chunk = _grid_chunk_lafs(B, N, ch, PS, _grid_elem_bytes(grid_dtype))
    cur_img = img.to(grid_laf.dtype) if img.dtype != grid_laf.dtype else img
    laf_a = grid_laf[..., :2]
    t = 2.0 * grid_laf[..., :, 2] - 1.0

    # Most calls fit in one bounded chunk, so their level-independent affine grid can be reused
    # across the streaming pyramid. A multi-chunk call rebuilds each chunk's grid per level rather
    # than retaining every grid or pyramid level: the fallback exists specifically to keep peak
    # memory bounded, and either cache would make it scale with all N or another third of the image.
    base_grid_all = torch.empty(0, dtype=grid_dtype, device=nlaf.device)
    if chunk == N:
        theta = torch.cat([laf_a, torch.zeros(B, N, 2, 1, dtype=grid_dtype, device=nlaf.device)], dim=-1)
        base_grid_all = F.affine_grid(theta.view(B * N, 2, 3), [B * N, ch, PS, PS], align_corners=False).view(
            B, N, PS, PS, 2
        )

    for level_idx, (h_l, w_l) in enumerate(zip(heights, widths)):
        if level_idx > 0:
            cur_img = pyrdown(cur_img)
        for st in range(0, N, chunk):
            en = min(st + chunk, N)
            nc = en - st
            if chunk == N:
                base_grid = base_grid_all
            else:
                theta = torch.cat(
                    [laf_a[:, st:en], torch.zeros(B, nc, 2, 1, dtype=grid_dtype, device=nlaf.device)], dim=-1
                )
                base_grid = F.affine_grid(theta.view(B * nc, 2, 3), [B * nc, ch, PS, PS], align_corners=False).view(
                    B, nc, PS, PS, 2
                )
            translation = t[:, st:en].view(B, nc, 1, 1, 2)
            # Match `normalize_laf` / `denormalize_laf`: a singleton axis counts as one pixel of
            # extent. Border padding still repeats that axis's only pixel, while the other axis
            # keeps its spatial variation instead of being collapsed by a shared zero `min_l`.
            min_l = float(min(max(h_l - 1, 1), max(w_l - 1, 1)))
            k = base_grid.new_tensor([2.0 * min_l / float(max(w_l - 1, 1)), 2.0 * min_l / float(max(h_l - 1, 1))])
            grid = base_grid * k + translation
            patches = _sample_patches(cur_img, grid, h_l, w_l).to(img.dtype)
            mask = (pyr_idx[:, st:en] == level_idx).view(B, en - st, 1, 1, 1)
            out[:, st:en] = torch.where(mask, patches, out[:, st:en])
    return out


def extract_patches_simple(
    img: torch.Tensor, laf: torch.Tensor, PS: int = 32, normalize_lafs_before_extraction: bool = True
) -> torch.Tensor:
    """Extract patches defined by LAFs from image torch.Tensor.

    No smoothing applied, huge aliasing (better use extract_patches_from_pyramid).

    Args:
        img: images, LAFs are detected in  :math:`(B, CH, H, W)`.
        laf: :math:`(B, N, 2, 3)`.
        PS: patch size.
        normalize_lafs_before_extraction: if True, lafs are normalized to image size.

    Returns:
        patches with shape :math:`(B, N, CH, PS,PS)`.

    """
    KORNIA_CHECK_LAF(laf)
    KORNIA_CHECK(img.size(0) == laf.size(0), "img and laf must have the same batch size")
    # The image owns the output device and dtype, but the LAF keeps its coordinate precision.
    # Moving the tiny LAF up front also preserves mixed-precision autocast pipelines whose
    # detector emits a reduced-precision LAF for a float32 image.
    laf = laf.to(device=img.device)
    grid_dtype = _promoted_grid_dtype(img.dtype, laf.dtype)
    laf = laf.to(grid_dtype) if laf.dtype != grid_dtype else laf
    if normalize_lafs_before_extraction:
        nlaf = normalize_laf(laf, img)
    else:
        nlaf = laf
    _, ch, h, w = img.size()
    B, N, _, _ = laf.size()
    if B == 0 or N == 0:
        return torch.zeros(B, N, ch, PS, PS, device=img.device, dtype=img.dtype)
    # See `extract_patches_from_pyramid`: a non-finite value anywhere in a training-time LAF
    # frame, including only its center, would reach `grid_sample` as an invalid grid, and the CPU
    # border-padding backward kernel can segfault on it. Mark the whole frame invalid and
    # sanitize it before any grid arithmetic; the frame then contributes neither output nor
    # gradient, and its finite neighbours are untouched.
    invalid_lafs = ~torch.isfinite(nlaf).all(dim=-1).all(dim=-1)
    nlaf = nlaf.masked_fill(invalid_lafs.view(B, N, 1, 1), 0.0)
    # The image is upcast to the grid's dtype once, before the chunk loop: `_grid_sample_patches`
    # samples in one dtype, and re-casting the full image per chunk would defeat the chunk budget.
    sample_img = img.to(grid_dtype) if img.dtype != grid_dtype else img
    out = torch.empty(B, N, ch, PS, PS, device=img.device, dtype=img.dtype)
    chunk = _grid_chunk_lafs(B, N, ch, PS, _grid_elem_bytes(grid_dtype))
    for st in range(0, N, chunk):
        en = min(st + chunk, N)
        grid = generate_patch_grid_from_normalized_LAF(img, nlaf[:, st:en], PS).view(B, en - st, PS, PS, 2)
        out[:, st:en] = _sample_patches(sample_img, grid, h, w)
    # Zeroing after the loop keeps the masking unconditional, so the fullgraph `torch.compile`
    # path never takes a data-dependent Python branch.
    return out.masked_fill_(invalid_lafs.view(B, N, 1, 1, 1), 0.0)


def extract_patches_from_pyramid(
    img: torch.Tensor, laf: torch.Tensor, PS: int = 32, normalize_lafs_before_extraction: bool = True
) -> torch.Tensor:
    """Extract patches defined by LAFs from image torch.Tensor.

    Patches are extracted from the appropriate pyramid level. A LAF whose scale selects a level
    smaller than ``PS`` is sampled from the coarsest level that can still provide a full patch.

    Args:
        img: images, LAFs are detected in  :math:`(B, CH, H, W)`.
        laf: :math:`(B, N, 2, 3)`.
        PS: patch size.
        normalize_lafs_before_extraction: if True, lafs are normalized to image size.

    Returns:
        patches with shape :math:`(B, N, CH, PS,PS)`.

    """
    KORNIA_CHECK_LAF(laf)
    KORNIA_CHECK(img.size(0) == laf.size(0), "img and laf must have the same batch size")
    # See `extract_patches_simple`: the image owns the public output contract, while the much
    # smaller LAF moves to its device without discarding coordinate precision.
    laf = laf.to(device=img.device)
    grid_dtype = _promoted_grid_dtype(img.dtype, laf.dtype)
    laf = laf.to(grid_dtype) if laf.dtype != grid_dtype else laf
    if normalize_lafs_before_extraction:
        nlaf = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    _, ch, h, w = img.size()
    if B == 0 or N == 0:
        return torch.zeros(B, N, ch, PS, PS, device=img.device, dtype=img.dtype)
    # max_level is a compile-time constant for static image shapes.
    max_level = min(h, w) // PS
    # Build exactly the pyramid that can provide a PS-sized patch. ``pyrdown`` defines its output
    # size as floor(side / 2), including for odd inputs, so the atlas can be allocated before the
    # levels are materialized and each level can be released after it is copied.
    heights = [h]
    widths = [w]
    for _ in range(1, max(1, max_level)):
        # `pyrdown` applies a 5x5 Gaussian with two pixels of reflect padding, which requires
        # both source axes to be larger than two. Keep such a source as the actual coarsest
        # usable level instead of recording a target level that cannot be materialized.
        if min(heights[-1], widths[-1]) <= 2:
            break
        h_l = heights[-1] // 2
        w_l = widths[-1] // 2
        if min(h_l, w_l) < PS:
            break
        heights.append(h_l)
        widths.append(w_l)

    # A non-finite value anywhere in a training-time LAF, including only its center, can reach an
    # invalid sampling grid. The CPU border-padding backward kernel can segfault on a NaN grid.
    # Mark the whole frame invalid and sanitize it before any grid arithmetic; both paths then
    # return a zero patch and a zero gradient for that frame without disturbing finite neighbours.
    invalid_lafs = ~torch.isfinite(nlaf).all(dim=-1).all(dim=-1)
    nlaf = nlaf.masked_fill(invalid_lafs.view(B, N, 1, 1), 0.0)
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    pyr_idx = scale.log2().clamp(min=0.0, max=float(len(heights) - 1)).long().squeeze(-1).squeeze(-1)
    pyr_idx = pyr_idx.masked_fill(invalid_lafs, -1)

    # Small images and ROIs commonly have only level 0. Sampling that image directly avoids
    # allocating and remapping a one-level atlas that cannot provide any pyramid benefit.
    if len(heights) == 1:
        return _extract_patches_from_pyramid_levelwise(img, nlaf, pyr_idx, heights, widths, PS)

    # Place every level side-by-side with a one-pixel replicated guard. The guard absorbs
    # floating-point remapping error at an outer pixel center and preserves both the border value
    # and its zero outward gradient instead of leaking into the neighbouring level.
    atlas_h = h + 2
    packed_widths = [w_l + 2 for w_l in widths]
    atlas_w = sum(packed_widths)
    # A full-height atlas is counterproductive for very large or heavily batched images. Keep
    # its storage bounded; the static shape guard disappears under torch.compile, and the
    # levelwise path preserves the same clamping and reduced-precision grid semantics. The atlas
    # is built in the grid's dtype -- reduced-precision inputs are upcast once, so the replicate
    # pad, `pyrdown` and every chunk's `grid_sample` run on kernels every torch build has, and no
    # full-atlas recast is paid per chunk. A 1-pixel axis in any *built level* -- a 1-pixel input
    # image, or a coarse level that `PS == 1` lets the pyramid descend to -- would make the level
    # constants' Python `size - 1` division below raise; the levelwise sampler treats that axis
    # as having zero spatial extent instead.
    atlas_elements = B * ch * atlas_h * atlas_w
    atlas_bytes = atlas_elements * _grid_elem_bytes(grid_dtype)
    if img.dtype != grid_dtype:
        atlas_bytes += B * ch * h * w * _grid_elem_bytes(grid_dtype)  # the one-time upcast copy
    if min(heights[-1], widths[-1]) < 2 or not _pyramid_atlas_fits(atlas_bytes):
        return _extract_patches_from_pyramid_levelwise(img, nlaf, pyr_idx, heights, widths, PS)
    sample_img = img.to(grid_dtype) if img.dtype != grid_dtype else img
    atlas = sample_img.new_zeros(B, ch, atlas_h, atlas_w)
    cur_img = sample_img
    xoff = 0
    for level_idx, (h_l, w_l) in enumerate(zip(heights, widths)):
        if level_idx > 0:
            cur_img = pyrdown(cur_img)
        atlas[:, :, : h_l + 2, xoff : xoff + w_l + 2] = F.pad(cur_img, (1, 1, 1, 1), mode="replicate")
        xoff += w_l + 2

    # A patch grid's linear part is level-independent: it is generated from normalized LAF A with
    # zero translation; reduced-precision inputs need float32 grid arithmetic because the atlas is
    # wider than the original image.
    laf_a = nlaf[..., :2].to(grid_dtype)
    t = 2.0 * nlaf[..., :, 2].to(grid_dtype) - 1.0

    # Gather all level-dependent conversion constants per patch, packed as (x, y) pairs so the
    # remap below runs on the whole grid tensor step by step, never holding split-axis copies. A
    # giant LAF that nominally selects an unbuilt level is sampled from the actual coarsest
    # pyramid image.
    xoff = 0
    constants = []
    for h_l, w_l in zip(heights, widths):
        min_l = float(min(h_l - 1, w_l - 1))
        constants.append(
            (
                2.0 * min_l / float(w_l - 1),  # k: LAF frame -> level-normalized units
                2.0 * min_l / float(h_l - 1),
                -1.0 + 1.0 / float(w_l),  # lo/hi: the level's outermost pixel centers
                -1.0 + 1.0 / float(h_l),
                1.0 - 1.0 / float(w_l),
                1.0 - 1.0 / float(h_l),
                float(w_l) / float(atlas_w),  # scale/offset: level frame -> atlas frame
                float(h_l) / float(atlas_h),
                (float(w_l) + 2.0 * float(xoff + 1)) / float(atlas_w) - 1.0,
                (float(h_l) + 2.0) / float(atlas_h) - 1.0,
            )
        )
        xoff += w_l + 2
    level_constants = torch.tensor(constants, dtype=grid_dtype, device=nlaf.device).view(-1, 5, 2)

    # The folded grid and its remap intermediates scale with B*N*PS^2, so sampling is chunked
    # along N to bound peak memory; each chunk repeats exactly the single-call arithmetic, and
    # for small workloads the loop is a single iteration.
    out = torch.empty(B, N, ch, PS, PS, device=img.device, dtype=img.dtype)
    chunk = _grid_chunk_lafs(B, N, ch, PS, _grid_elem_bytes(grid_dtype))
    # Invalid (non-finite) LAFs carry level -1: they index the level-0 constants here and their
    # patches are zeroed after the loop, unconditionally -- a data-dependent Python branch would
    # break the fullgraph `torch.compile` path.
    safe_pyr_idx = pyr_idx.clamp(min=0)
    for st in range(0, N, chunk):
        en = min(st + chunk, N)
        nc = en - st
        theta = torch.cat([laf_a[:, st:en], torch.zeros(B, nc, 2, 1, dtype=grid_dtype, device=nlaf.device)], dim=-1)
        grid = F.affine_grid(theta.view(B * nc, 2, 3), [B * nc, ch, PS, PS], align_corners=False).view(B, nc, PS, PS, 2)
        k, lo, hi, level_scale, level_offset = (
            level_constants[safe_pyr_idx[:, st:en]].view(B, nc, 1, 1, 5, 2).unbind(-2)
        )
        grid = grid * k
        grid = grid + t[:, st:en].view(B, nc, 1, 1, 2)
        grid = grid.maximum(lo)
        grid = grid.minimum(hi)
        grid = grid * level_scale
        grid = grid + level_offset
        out[:, st:en] = _sample_patches(atlas, grid, atlas_h, atlas_w)
    return out.masked_fill_((pyr_idx < 0).view(B, N, 1, 1, 1), 0.0)


def laf_is_inside_image(laf: torch.Tensor, images: torch.Tensor, border: int = 0) -> torch.Tensor:
    """Check if the LAF is touching or partly outside the image boundary.

    Returns the mask of LAFs, which are fully inside the image, i.e. valid.

    Args:
        laf:  :math:`(B, N, 2, 3)`.
        images: images, lafs are detected in :math:`(B, CH, H, W)`.
        border: additional border.

    Returns:
        mask with shape :math:`(B, N)`.

    """
    KORNIA_CHECK_LAF(laf)
    _, _, h, w = images.size()
    pts = laf_to_boundary_points(laf, 12)
    # Valid pixel coordinates run 0 .. w-1 and 0 .. h-1, matching the convention documented on
    # `get_laf_center` and the `w - 1` / `h - 1` extent used by `normalize_laf` / `denormalize_laf`.
    x_max = float(w - 1) - border
    y_max = float(h - 1) - border
    good_lafs_mask = (pts[..., 0] >= border) * (pts[..., 0] <= x_max) * (pts[..., 1] >= border) * (pts[..., 1] <= y_max)
    # `.all` rather than `.min` on the bool mask: ONNX ReduceMin has no bool overload.
    good_lafs_mask = good_lafs_mask.all(dim=2)
    return good_lafs_mask


def laf_to_three_points(laf: torch.Tensor) -> torch.Tensor:
    """Convert local affine frame(LAF) to alternative representation: coordinates of LAF center, LAF-x unit vector,
    LAF-y unit vector.

    Args:
        laf:  :math:`(B, N, 2, 3)`.

    Returns:
        threepts :math:`(B, N, 2, 3)`.

    """  # noqa:D205
    KORNIA_CHECK_LAF(laf)
    three_pts = torch.stack([laf[..., 2] + laf[..., 0], laf[..., 2] + laf[..., 1], laf[..., 2]], dim=-1)
    return three_pts


def laf_from_three_points(threepts: torch.Tensor) -> torch.Tensor:
    """Convert three points to local affine frame.

    Order is (0,0), (0, 1), (1, 0).

    Args:
        threepts: :math:`(B, N, 2, 3)`.

    Returns:
        laf :math:`(B, N, 2, 3)`.

    """
    laf = torch.stack(
        [threepts[..., 0] - threepts[..., 2], threepts[..., 1] - threepts[..., 2], threepts[..., 2]], dim=-1
    )
    return laf


def perspective_transform_lafs(trans_01: torch.Tensor, lafs_1: torch.Tensor) -> torch.Tensor:
    r"""Apply perspective transformations to a set of local affine frames (LAFs).

    Args:
        trans_01: torch.Tensor for perspective transformations of shape :math:`(B, 3, 3)`.
        lafs_1: torch.Tensor of lafs of shape :math:`(B, N, 2, 3)`.

    Returns:
        torch.Tensor of N-dimensional points of shape :math:`(B, N, 2, 3)`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> lafs_1 = torch.rand(2, 4, 2, 3)  # BxNx2x3
        >>> lafs_1
        tensor([[[[0.4963, 0.7682, 0.0885],
                  [0.1320, 0.3074, 0.6341]],
        <BLANKLINE>
                 [[0.4901, 0.8964, 0.4556],
                  [0.6323, 0.3489, 0.4017]],
        <BLANKLINE>
                 [[0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000]],
        <BLANKLINE>
                 [[0.1610, 0.2823, 0.6816],
                  [0.9152, 0.3971, 0.8742]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.4194, 0.5529, 0.9527],
                  [0.0362, 0.1852, 0.3734]],
        <BLANKLINE>
                 [[0.3051, 0.9320, 0.1759],
                  [0.2698, 0.1507, 0.0317]],
        <BLANKLINE>
                 [[0.2081, 0.9298, 0.7231],
                  [0.7423, 0.5263, 0.2437]],
        <BLANKLINE>
                 [[0.5846, 0.0332, 0.1387],
                  [0.2422, 0.8155, 0.7932]]]])
        >>> trans_01 = torch.eye(3).repeat(2, 1, 1)  # Bx3x3
        >>> trans_01.shape
        torch.Size([2, 3, 3])
        >>> lafs_0 = perspective_transform_lafs(trans_01, lafs_1)  # BxNx2x3

    """
    KORNIA_CHECK_LAF(lafs_1)
    if not torch.is_tensor(trans_01):
        raise TypeError("Input type is not a torch.Tensor")

    if not trans_01.device == lafs_1.device:
        raise TypeError("torch.Tensor must be in the same device")

    if not trans_01.shape[0] == lafs_1.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")

    if (not (trans_01.shape[-1] == 3)) or (not (trans_01.shape[-2] == 3)):
        raise ValueError("Transformation should be homography")

    bs, n, _, _ = lafs_1.size()
    # First, we convert LAF to points
    threepts_1 = laf_to_three_points(lafs_1)
    points_1 = threepts_1.permute(0, 1, 3, 2).reshape(bs, n * 3, 2)

    # First, transform the points
    points_0 = transform_points(trans_01, points_1)

    # Back to LAF format
    threepts_0 = points_0.view(bs, n, 3, 2).permute(0, 1, 3, 2)
    return laf_from_three_points(threepts_0)
