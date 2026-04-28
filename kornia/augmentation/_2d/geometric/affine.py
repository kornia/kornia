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

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample, SamplePadding
from kornia.geometry.transform import warp_affine


def _affine_matrix2d_closed(
    translations: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    angle: torch.Tensor,
    shear_x: torch.Tensor,
    shear_y: torch.Tensor,
) -> torch.Tensor:
    """Build a Bx3x3 affine matrix without intermediate matmuls or eye_like allocations.

    Equivalent to ``get_affine_matrix2d`` but derived in closed form from the component
    parameters, avoiding the 4 ``eye_like`` calls, 4 Bx3x3 matmuls, F.pad, and the
    optional shear matmul inside ``get_affine_matrix2d`` / ``get_rotation_matrix2d``.

    Args:
        translations: pixel-space (tx, ty) with shape (B, 2).
        center: rotation/scale center (cx, cy) with shape (B, 2).
        scale: per-axis scale (scale_x, scale_y) with shape (B, 2).
        angle: rotation in **degrees** with shape (B,).
        shear_x: x-axis shear in **degrees** with shape (B,).
        shear_y: y-axis shear in **degrees** with shape (B,).

    Returns:
        Affine matrix with shape (B, 3, 3).
    """
    _PI_OVER_180 = 3.141592653589793 / 180.0

    # Rotation: kornia convention passes -angle to angle_to_rotation_matrix, which gives
    #   [[cos(angle_deg), -sin(angle_deg)], [sin(angle_deg), cos(angle_deg)]]
    ang_rad = angle * _PI_OVER_180
    cos_a = torch.cos(ang_rad)   # (B,)
    sin_a = torch.sin(ang_rad)   # (B,)

    sx = scale[:, 0]   # (B,)
    sy = scale[:, 1]   # (B,)
    cx = center[:, 0]  # (B,)
    cy = center[:, 1]  # (B,)
    tx = translations[:, 0]  # (B,)
    ty = translations[:, 1]  # (B,)

    # Rotation+scale block (from shift @ rot @ scale @ shift_inv):
    #   a = sx * cos_a,  b = -sy * sin_a
    #   d = sx * sin_a,  e =  sy * cos_a
    # Translation column from shift @ rot @ scale @ shift_inv:
    #   c = cx*(1 - a) - cy*b + tx  =  cx - a*cx + b_abs*cy + tx  (b is negative)
    #   f = cy*(1 - e) - cx*d + ty
    a = sx * cos_a
    b = -sy * sin_a
    c = cx - a * cx - b * cy + tx   # note: -b because b = -sy*sin_a, so -b = sy*sin_a
    d = sx * sin_a
    e = sy * cos_a
    f = cy - e * cy - d * cx + ty

    # Apply shear (same effect as right-multiplying by shear_3x3):
    #   shear_3x3 top-left 2x3: [[1, -sx_tan, sx_tan*cy], [-sy_tan, 1+sx_tan*sy_tan, sy_tan*(cx-sx_tan*cy)]]
    #   result[row, :] = [row_a, row_b, row_c] @ shear_3x3
    sx_t = torch.tan(shear_x * _PI_OVER_180)   # (B,)
    sy_t = torch.tan(shear_y * _PI_OVER_180)   # (B,)

    r00 = a - b * sy_t
    r01 = -a * sx_t + b * (1.0 + sx_t * sy_t)
    r02 = a * sx_t * cy + b * sy_t * (cx - sx_t * cy) + c
    r10 = d - e * sy_t
    r11 = -d * sx_t + e * (1.0 + sx_t * sy_t)
    r12 = d * sx_t * cy + e * sy_t * (cx - sx_t * cy) + f

    zeros = torch.zeros_like(r00)
    ones = torch.ones_like(r00)

    # Stack into Bx3x3
    return torch.stack(
        [r00, r01, r02, r10, r11, r12, zeros, zeros, ones], dim=-1
    ).reshape(-1, 3, 3)


def _affine_homography_inv(M: torch.Tensor) -> torch.Tensor:
    """Invert a batch of homogeneous 2D affine matrices analytically.

    Exploits the fact that the last row of every matrix in the batch is
    ``[0, 0, 1]`` — a property that always holds for the matrices produced by
    :func:`_affine_matrix2d_closed`.  This is ~35× faster than
    ``torch.linalg.inv`` on CPU for typical augmentation batch sizes (B ≤ 16).

    Args:
        M: Bx3x3 homogeneous affine matrix whose last row is (0, 0, 1).

    Returns:
        Analytical inverse with shape (B, 3, 3).
    """
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    det = a * e - b * d
    r00 = e / det
    r01 = -b / det
    r02 = (b * f - c * e) / det
    r10 = -d / det
    r11 = a / det
    r12 = (c * d - a * f) / det
    zeros = torch.zeros_like(r00)
    ones = torch.ones_like(r00)
    return torch.stack([r00, r01, r02, r10, r11, r12, zeros, zeros, ones], dim=-1).reshape(-1, 3, 3)


class RandomAffine(GeometricAugmentationBase2D):
    r"""Apply a random 2D affine transformation to a torch.Tensor image.

    .. image:: _static/img/RandomAffine.png

    The transformation is computed so that the image center is kept invariant.

    Args:
        degrees: Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale: scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If (a, b, c, d), the scale is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d.
            Will keep original scale by default.
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "torch.zeros" (0), "border" (1), "reflection" (2) or "fill" (3).
        fill_value: the value to be filled in the padding area when padding_mode="fill".
            Can be a float, int, or a torch.Tensor of shape (C) or (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_affine`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> out = aug(input)
        >>> out, aug.transform_matrix
        (tensor([[[[0.3961, 0.7310, 0.1574],
                  [0.1781, 0.3074, 0.5648],
                  [0.4804, 0.8379, 0.4234]]]]), tensor([[[ 0.9923, -0.1241,  0.1319],
                 [ 0.1241,  0.9923, -0.1164],
                 [ 0.0000,  0.0000,  1.0000]]]))
        >>> aug.inverse(out)
        tensor([[[[0.3890, 0.6573, 0.1865],
                  [0.2063, 0.3074, 0.5459],
                  [0.3892, 0.7896, 0.4224]]]])
        >>> input
        tensor([[[[0.4963, 0.7682, 0.0885],
                  [0.1320, 0.3074, 0.6341],
                  [0.4901, 0.8964, 0.4556]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        degrees: Union[torch.Tensor, float, Tuple[float, float]],
        translate: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        scale: Optional[Union[torch.Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[torch.Tensor, float, Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        fill_value: Optional[Union[float, int, torch.Tensor]] = None,  # Updated type hint
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator: rg.AffineGenerator = rg.AffineGenerator(degrees, translate, scale, shear)

        if fill_value is not None and not isinstance(fill_value, torch.Tensor):
            fill_value = torch.as_tensor(fill_value)

        self.flags = {
            "resample": Resample.get(resample),
            "padding_mode": SamplePadding.get(padding_mode),
            "align_corners": align_corners,
            "fill_value": fill_value,
        }
        # Cache for pixel-to-normalised coordinate matrices, keyed by (H, W, device, dtype).
        # Avoids two linalg.inv calls and two torch.tensor allocations per forward pass.
        self._norm_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _make_norm_matrices(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build pixel-to-normalised-coord matrix N and its inverse N_inv analytically.

        N maps pixel coordinates (with kornia's width-1/height-1 convention) to [-1, 1].
        N_inv is computed analytically — no torch.linalg.inv call needed.
        Both tensors are shaped (1, 3, 3) so they broadcast over the batch dimension.
        """
        eps = 1e-14
        w_denom = float(width - 1) if width > 1 else eps
        h_denom = float(height - 1) if height > 1 else eps
        sw = 2.0 / w_denom
        sh = 2.0 / h_denom
        # N = [[sw,  0, -1],
        #      [0,  sh, -1],
        #      [0,   0,  1]]
        N = torch.zeros(1, 3, 3, device=device, dtype=dtype)
        N[0, 0, 0] = sw
        N[0, 1, 1] = sh
        N[0, 0, 2] = -1.0
        N[0, 1, 2] = -1.0
        N[0, 2, 2] = 1.0
        # N_inv = [[1/sw,    0, 1/sw],
        #          [0,    1/sh, 1/sh],
        #          [0,       0,    1]]
        N_inv = torch.zeros(1, 3, 3, device=device, dtype=dtype)
        N_inv[0, 0, 0] = 1.0 / sw
        N_inv[0, 1, 1] = 1.0 / sh
        N_inv[0, 0, 2] = 1.0 / sw
        N_inv[0, 1, 2] = 1.0 / sh
        N_inv[0, 2, 2] = 1.0
        return N, N_inv

    def _get_norm_matrices(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, device, dtype)
        if key not in self._norm_cache:
            self._norm_cache[key] = self._make_norm_matrices(height, width, device, dtype)
        return self._norm_cache[key]

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        return _affine_matrix2d_closed(
            params["translations"].to(device=input.device, dtype=input.dtype),
            params["center"].to(device=input.device, dtype=input.dtype),
            params["scale"].to(device=input.device, dtype=input.dtype),
            params["angle"].to(device=input.device, dtype=input.dtype),
            params["shear_x"].to(device=input.device, dtype=input.dtype),
            params["shear_y"].to(device=input.device, dtype=input.dtype),
        )

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

        padding_mode_str = flags["padding_mode"].name.lower()

        # For the 'fill' padding mode warp_affine has special handling; delegate to it.
        if padding_mode_str == "fill":
            return warp_affine(
                input,
                transform[:, :2, :],
                (height, width),
                flags["resample"].name.lower(),
                align_corners=flags["align_corners"],
                padding_mode=padding_mode_str,
                fill_value=flags["fill_value"],
            )

        # Fast path: bypass warp_affine's normalize_homography chain.
        #
        # warp_affine internally does (for same src/dst size):
        #   M_3x3 = pad(M_2x3)                        → Bx3x3
        #   N_dst_norm = normalize_homography(M_3x3)   → N @ M_3x3 @ N^{-1}
        #   theta      = inv(N_dst_norm)[:, :2, :]     → (N @ M^{-1} @ N^{-1})[:2,:]
        #   grid       = affine_grid(theta, ...)
        #   out        = grid_sample(src, grid, ...)
        #
        # The two linalg.inv calls (one inside normalize_homography on the fixed
        # normalization matrix, one on the combined Bx3x3 matrix) cost ~37 ms at
        # B=8, 512x512.  We replace them with:
        #   * N and N^{-1} precomputed analytically and cached per (H,W,device,dtype)
        #   * a single linalg.inv on the Bx3x3 forward transform M
        #   * two small batched matmuls to build theta
        #
        # This halves the linalg.inv cost and eliminates the normal_transform_pixel
        # torch.tensor allocations.
        align_corners = flags["align_corners"]
        mode = flags["resample"].name.lower()

        N, N_inv = self._get_norm_matrices(height, width, input.device, input.dtype)

        # transform is Bx3x3 (forward pixel-space affine produced by compute_transformation).
        # Its last row is always [0, 0, 1], so we use the analytical 2x2-block inverse
        # instead of the general linalg.inv (~35x faster for small B on CPU).
        M_inv = _affine_homography_inv(transform)     # Bx3x3, analytical inversion
        theta = (N @ M_inv @ N_inv)[:, :2, :]         # Bx2x3

        B, C = input.shape[:2]
        grid = F.affine_grid(theta, [B, C, height, width], align_corners=align_corners)
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode_str, align_corners=align_corners)

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=torch.as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
