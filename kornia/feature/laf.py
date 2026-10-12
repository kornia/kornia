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
from kornia.geometry.conversions import angle_to_rotation_matrix
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import pyrdown


def _laf_product_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype to multiply two LAF entries in.

    A product of two entries of an ordinary LAF can leave the float16 range. It overflows above 65504: the determinant
    of any LAF of scale 256 px or more, and the square of a 256 px entry of an elongated LAF of a smaller scale (300 x
    20 px has a scale of 77 px). It rounds to zero for entries below about 1.7e-4. float16 therefore works in float32,
    which holds every product of two float16 values exactly. bfloat16 has the exponent range of float32 and is left as
    it is.
    """
    if dtype == torch.float16:
        return torch.float32
    return dtype


def get_laf_scale(LAF: torch.Tensor) -> torch.Tensor:
    """Return a scale of the LAFs.

    Convention:
        - See the Convention block on :func:`~kornia.feature.laf_from_center_scale_ori`, which defines the scale.
        - The determinant of a float16 LAF is formed in float32, so that a LAF whose entry products leave the float16
          range does not overflow; the scale is returned in the LAF dtype.

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
    work = LAF.to(_laf_product_dtype(LAF.dtype))
    out = work[..., 0:1, 0:1] * work[..., 1:2, 1:2] - work[..., 1:2, 0:1] * work[..., 0:1, 1:2] + eps
    scale = out.abs().sqrt()
    if work.dtype != LAF.dtype:
        scale = scale.to(LAF.dtype)
    return scale


def laf_is_valid(laf: torch.Tensor) -> torch.Tensor:
    """Check that each LAF is finite and has a finite, nonzero determinant.

    Convention:
        - A LAF with a negative determinant, a mirrored frame, is valid. The all-zero padding LAF (see
          :func:`~kornia.feature.laf_is_filled`), a rank-1 ``A`` and an ``A`` whose determinant overflows are not.
        - The determinant of a float16 LAF is formed in float32, so the LAF is not judged by an intermediate outside
          the float16 range: one whose entry products overflow float16 (for instance any LAF of scale 256 px or more)
          is valid, and so is one whose determinant is too small for float16. The determinant of a float32, float64
          or bfloat16 LAF is formed in that dtype, where it can overflow.

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
    laf = laf.to(_laf_product_dtype(laf.dtype))
    det = laf[..., 0, 0] * laf[..., 1, 1] - laf[..., 1, 0] * laf[..., 0, 1]
    return laf.isfinite().all(dim=-1).all(dim=-1) & det.isfinite() & (det != 0)


def laf_is_filled(laf: torch.Tensor) -> torch.Tensor:
    """Check which slots hold a detection rather than the zero-LAF padding.

    Convention:
        - kornia's LAF detectors pad the slots no detection filled with an all-zero LAF and a zero response, so that a
          batch is one tensor. Drop those slots before matching: the padding frames are identical to one another, so a
          mutual nearest-neighbour test does not reject them and they match each other at zero distance.
        - Only an all-zero frame is padding. Any other frame is filled: a detection at the image origin, which has a
          zero centre but a nonzero shape, and also a singular or non-finite frame, which
          :func:`~kornia.feature.laf_is_valid` screens out.
        - Occupancy cannot be read off the response instead, because a pluggable signed response may have a genuine
          maximum at exactly zero.

    Args:
        laf: :math:`(B, N, 2, 3)`.

    Returns:
        mask of the slots a detection filled :math:`(B, N)`.

    Example:
        >>> laf = torch.tensor([[[[2., 0., 5.], [0., 3., 7.]], [[0., 0., 0.], [0., 0., 0.]]]])
        >>> laf_is_filled(laf)
        tensor([[ True, False]])

    """
    KORNIA_CHECK_LAF(laf)
    return laf.ne(0).any(dim=-1).any(dim=-1)


def get_laf_center(LAF: torch.Tensor) -> torch.Tensor:
    """Return a center (keypoint) of the LAFs.

    Convention:
        The centre is ``(x, y)`` in pixels, with pixel centres at integers; see
        :ref:`Local features <local-features-conventions>`.

    Args:
        LAF: :math:`(B, N, 2, 3)`

    Returns:
        xy :math:`(B, N, 2)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_center(input)  # BxNx2

    """
    KORNIA_CHECK_LAF(LAF)
    return LAF[..., 2]


def get_laf_orientation(LAF: torch.Tensor) -> torch.Tensor:
    """Return orientation of the LAFs, in degrees.

    Convention:
        See the Convention block on :func:`~kornia.feature.laf_from_center_scale_ori`, which defines the orientation.

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

    Convention:
        - The result is ``[A @ R | c]``, with ``R`` the rotation matrix of ``angles_degrees`` from
          :func:`~kornia.geometry.conversions.angle_to_rotation_matrix`. The orientation grows by ``angles_degrees``
          (:func:`~kornia.feature.set_laf_orientation` sets it instead), and the centre and the upright factor that
          :func:`~kornia.feature.make_upright` returns are kept.

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

    Convention:
        See the Convention block on :func:`~kornia.feature.rotate_laf`. This rotates by ``angles_degrees`` minus the
        current orientation, so the orientation reads back as ``angles_degrees`` wrapped into ``[-180, 180]``.

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
    r"""Create a LAF from keypoint center, scale and orientation.

    Convention:
        - A LAF is a :math:`(B, N, 2, 3)` tensor ``[A | c]``, with the centre ``c = (x, y)`` and ``A`` in pixels.
          ``A`` maps the patch frame onto the image: its first column is the image of the patch's x axis and its
          second column that of the patch's y axis. :ref:`Local features <local-features-conventions>` gives the
          pixel frame, the patch that :func:`~kornia.feature.extract_patches_from_pyramid` samples and the
          conversion of OpenCV keypoints.
        - This function returns ``A = scale * R``, with ``R = [[cos, sin], [-sin, cos]]`` of ``ori`` in degrees, the
          matrix of :func:`~kornia.geometry.conversions.angle_to_rotation_matrix`. A positive ``ori`` turns the
          patch's x axis counter-clockwise on screen, with image y pointing down.
        - For any LAF, the scale is :math:`\sqrt{|\det A|}` (:func:`~kornia.feature.get_laf_scale`) and the
          orientation is ``atan2(a12, a11)`` in degrees (:func:`~kornia.feature.get_laf_orientation`), within
          ``[-180, 180]``: a LAF made with ``ori = 181`` reads back ``-179``. For :math:`\det A > 0`, ``A = U R``
          with ``U`` the upright ``A`` that :func:`~kornia.feature.make_upright` returns and ``R`` the rotation
          matrix of that orientation.

    Args:
        xy: :math:`(B, N, 2)`.
        scale: :math:`(B, N, 1, 1)`. If not provided, scale = 1.0 is assumed
        ori: angle in degrees :math:`(B, N, 1)`. If not provided orientation = 0 is assumed

    Returns:
        LAF :math:`(B, N, 2, 3)`

    Example:
        >>> xy = torch.tensor([[[5.0, 7.0]]])
        >>> laf_from_center_scale_ori(xy, torch.full((1, 1, 1, 1), 2.0), torch.tensor([[[30.0]]]))
        tensor([[[[ 1.7321,  1.0000,  5.0000],
                  [-1.0000,  1.7321,  7.0000]]]])

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
    return scale_laf(unscaled_laf, scale)


def scale_laf(laf: torch.Tensor, scale_coef: Union[float, torch.Tensor]) -> torch.Tensor:
    """Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.

    Convention:
        - The result is ``[k A | c]`` for ``k = scale_coef``, a product taken elementwise: the centre is kept. With one
          coefficient per LAF, the scale is multiplied by ``|k|``, and a negative ``k`` also turns the frame by 180
          degrees.
        - ``scale_coef`` is a ``float`` or a tensor that broadcasts against :math:`(B, N, 2, 2)`, such as one
          coefficient per LAF in a :math:`(B, N, 1, 1)` tensor; an ``int`` is rejected.

    Args:
        laf: :math:`(B, N, 2, 3)`
        scale_coef: broadcastable torch.Tensor or float.

    Returns:
        LAF :math:`(B, N, 2, 3)`

    Raises:
        TypeError: if ``scale_coef`` is neither a ``float`` nor a tensor.

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
    r"""Rectify the affine matrix, so that it becomes upright.

    Convention:
        - The result is ``[U | c]``: the centre is kept, and ``U`` is the lower-triangular matrix with a positive
          diagonal and :math:`U U^\top = A A^\top`. The region, the ellipse ``c + A u`` over the unit vectors ``u``,
          and the scale are therefore kept, and the orientation becomes 0.
        - For :math:`\det A > 0`, ``A = U R`` with ``R`` the rotation matrix of the LAF's orientation, as in
          :func:`~kornia.feature.laf_from_center_scale_ori`. For :math:`\det A < 0` the reflection is dropped too:
          ``U`` has the determinant :math:`|\det A|`.
        - The all-zero LAF that pads an unfilled detector slot (see :func:`~kornia.feature.laf_is_filled`) is
          returned unchanged, so the padding stays recognizable.
        - A float16 LAF is rectified in float32, so that a LAF whose entry products leave the float16 range does not
          overflow; the result is returned in the LAF dtype.

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
    dtype = laf.dtype
    laf = laf.to(_laf_product_dtype(dtype))
    # Differentiate through an identity frame in place of the zero padding: sqrt at exactly zero has an infinite
    # derivative, which torch.where below would turn into nan rather than mask.
    filled = laf_is_filled(laf)[..., None, None]
    eye = torch.eye(2, 3, device=laf.device, dtype=laf.dtype).expand_as(laf)
    padding = laf
    laf = torch.where(filled, laf, eye)
    det = get_laf_scale(laf)
    scale = det
    # The result is the lower-triangular factor of the LQ decomposition A = U Q (Q orthogonal): for the rows r0, r1
    # of A, U = [[|r0|, 0], [r1 . r0 / |r0|, |det A| / |r0|]], built at unit scale and multiplied back by the scale.
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
    upright = scale_laf(laf_unit_scale, scale)
    # The padding LAF would otherwise come back as a tiny nonzero frame (the determinant and the division carry eps).
    out = torch.where(filled, upright, padding)
    if laf.dtype != dtype:
        out = out.to(dtype)
    return out


def ellipse_to_laf(ells: torch.Tensor) -> torch.Tensor:
    r"""Convert ellipse regions to LAF format.

    Convention:
        - An ellipse ``[x, y, a, b, c]`` is the region :math:`a u^2 + 2 b u v + c v^2 \le 1` in the coordinates
          ``(u, v)`` relative to its centre ``(x, y)``, as in the Oxford affine-region files. ``(x, y)`` becomes the
          LAF centre unchanged.
        - ``A`` is the lower-triangular matrix with a positive diagonal and ``A @ A.T == inverse([[a, b], [b, c]])``:
          the Cholesky factor of the inverse ellipse matrix, as VLFeat's ``vl_frame2oell.m`` builds it, not the
          symmetric inverse square root (the two differ by a rotation). Either one maps the unit circle onto the
          ellipse; the lower-triangular one is what :func:`~kornia.feature.make_upright` returns for any LAF of that
          region.
        - The conversion checks only the shape :math:`(B, N, 5)`; it does not check the ellipse values and does not
          raise on them. An ellipse that is degenerate or indefinite (``a``, ``c`` or ``a * c - b * b`` not positive)
          is not a bounded region; its LAF has ``inf`` or ``nan`` entries, or huge finite ones where rounding leaves
          ``a - b * b / c`` barely positive. Rounding also works the other way: a nearly singular valid ellipse can
          give ``inf`` or ``nan``. :func:`~kornia.feature.laf_is_valid` screens out the ``inf`` and ``nan`` results,
          which an ``isnan`` test misses for ``inf``; the huge finite ones pass it, so check the ellipse itself where
          that matters.

    See also https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_frame2oell.m

    Args:
        ells: torch.Tensor :math:`(B, N, 5)` of ellipses in Oxford format [x y a b c].

    Returns:
        LAF :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 10, 5)  # BxNx5
        >>> output = ellipse_to_laf(input)  #  BxNx2x3

    """
    KORNIA_CHECK_SHAPE(ells, ["B", "N", "5"])
    B, N, _ = ells.shape
    # The LAF is the upright (lower-triangular) A with A @ A.T == inverse([[a, b], [b, c]]), i.e. the
    # Cholesky factor of the inverse ellipse matrix, written with the Schur complement a - b^2 / c
    # (= det / c) so that b == 0 reduces to the plain 1 / sqrt(a), 1 / sqrt(c) diagonal.
    # Scaling b by 1 / sqrt(c) first keeps b^2 out of the arithmetic (it overflows for large but valid
    # ellipses) and, for a positive definite ellipse, b / sqrt(c) <= sqrt(a) can neither overflow nor
    # flush a representable off-diagonal to zero. The off-diagonal -b / (c * sqrt(a - b^2 / c)) is then
    # a single division by sqrt(c) * sqrt(a - b^2 / c) = sqrt(det), which stays within range whenever the
    # factors do; forming it as -(b / c) / sqrt(...) or -b / (c * sqrt(...)) instead has valid inputs
    # where b / c underflows or c * sqrt(...) overflows although the result itself is representable.
    a = ells[..., 2:3]
    b = ells[..., 3:4]
    c = ells[..., 4:5]
    sqrt_c = c.sqrt()
    b_scaled = b / sqrt_c
    sqrt_schur = (a - b_scaled * b_scaled).sqrt()
    inv11 = 1.0 / sqrt_schur
    inv22 = 1.0 / sqrt_c
    inv21 = -b_scaled / (sqrt_c * sqrt_schur)
    A = torch.stack([inv11, torch.zeros_like(inv11), inv21, inv22], dim=-1).view(B, N, 2, 2)
    return torch.cat([A, ells[..., :2].view(B, N, 2, 1)], dim=3)


def laf_to_boundary_points(LAF: torch.Tensor, n_pts: int = 50) -> torch.Tensor:
    r"""Convert LAFs to boundary points of the regions + center.

    Convention:
        - The first point is the centre ``c``. The other ``n_pts - 1`` points are ``c + A @ (sin t, cos t)`` for ``t``
          from 0 to :math:`2 \pi` in equal steps: they start and end at ``c + A[:, 1]``, the tip of the patch's y
          axis, and so close the outline of the region.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        n_pts: number of points to output, the centre included.

    Returns:
        torch.Tensor of boundary points LAF: :math:`(B, N, n_pts, 2)`

    """
    KORNIA_CHECK_LAF(LAF)
    B, N, _, _ = LAF.size()
    # Build the circle on the CPU in at least float32 (the LAF's dtype when it is wider, so float64 LAFs do not carry
    # float32 rounding), then cast it once.
    build = torch.promote_types(LAF.dtype, torch.float32)
    angles = torch.linspace(0, 2 * math.pi, n_pts - 1, dtype=build)
    pts = torch.cat(
        [torch.sin(angles).unsqueeze(-1), torch.cos(angles).unsqueeze(-1), torch.ones(n_pts - 1, 1, dtype=build)], dim=1
    )
    # Add origin to draw also the orientation
    pts = torch.cat([torch.tensor([0.0, 0.0, 1.0], dtype=build).view(1, 3), pts], dim=0)
    # Move the 3-column basis before broadcasting it: expanding first and casting afterwards
    # materialized -- and on an accelerator transferred -- one copy per LAF (~12 MiB at N = 20000)
    # of a tensor with `n_pts` distinct rows.
    pts = pts.to(device=LAF.device, dtype=LAF.dtype).t().unsqueeze(0).expand(B * N, 3, n_pts)
    # The LAF's implied homogeneous row is the constant [0, 0, 1], so every output point has
    # homogeneous coordinate exactly 1 and `convert_points_from_homogeneous` would divide by 1.0.
    # Multiplying by the (2, 3) LAF directly skips both the appended row and that division, and it
    # keeps `bmm`'s M dimension at 2: measured on torch 2.14 / Apple M1, a `(B, 3, 3)` operand
    # costs 23x a `(B, 2, 3)` one for 1.5x the arithmetic. See benchmarks/README.md.
    return torch.bmm(LAF.view(-1, 2, 3), pts).permute(0, 2, 1).reshape(B, N, n_pts, 2)


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

        B,CH,H,W = images.size()
        MIN_SIZE = min(H - 1, W -1)
        [a11 a12 x]
        [a21 a22 y]
        becomes
        [a11*MIN_SIZE a12*MIN_SIZE x*(W-1)]
        [a21*MIN_SIZE a22*MIN_SIZE y*(H-1)]

    A singleton axis (``H == 1`` or ``W == 1``) has no spatial extent and counts as one pixel, so
    the conversion stays finite and round-trips with :func:`normalize_laf`.

    Convention:
        See the Convention block on :func:`~kornia.feature.normalize_laf`, which this inverts.

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

    Convention:
        - The centre becomes ``(x / (W - 1), y / (H - 1))``, which lies in ``[0, 1]`` for a centre inside the image.
          ``A`` is divided by ``min(H - 1, W - 1)`` as a whole, not per axis, and can exceed 1.
        - Taken to an image of another size with :func:`~kornia.feature.denormalize_laf`, a normalized LAF maps
          corner to corner: ``x`` becomes ``x (W' - 1) / (W - 1)``. A resize with ``align_corners=False``, the
          default of :func:`~kornia.geometry.transform.resize` in bilinear mode, maps pixel ``x`` to
          ``(x + 0.5) W' / W - 0.5`` instead, so the two disagree away from the image centre.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        images: :math:`(B, CH, H, W)`

    Returns:
        the normalized LAF: :math:`(B, N, 2, 3)`

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
    # The grid is in pixels, centres at integers (the LAF frame); `grid_sample(align_corners=False)` puts pixel x at
    # (2x + 1) / W - 1. A singleton axis maps every sample to its only pixel's centre.
    grid[..., :, 0] = (2.0 * grid[..., :, 0].clone() + 1.0) / float(w) - 1.0
    grid[..., :, 1] = (2.0 * grid[..., :, 1].clone() + 1.0) / float(h) - 1.0
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


def _pyramid_grid_frame(nlaf: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return the scale ``k`` and translation ``t`` that map a normalized LAF's patch grid to sampling coordinates.

    ``grid = affine_grid(A, align_corners=False) * k + t`` addresses, for ``grid_sample(align_corners=False)``, the
    input pixels the LAF describes: pixel :math:`x` of the :math:`(h, w)` input sits at :math:`(2x + 1) / w - 1`.
    ``pyrdown`` resizes to ``floor(side / 2)`` with ``align_corners=False``, so input pixel :math:`x` is pixel
    :math:`(x + 0.5) w_l / w - 0.5` of level :math:`l` -- which that level's ``grid_sample`` addresses at the same
    :math:`(2x + 1) / w - 1`. One grid therefore serves every pyramid level.

    Args:
        nlaf: normalized LAFs :math:`(B, N, 2, 3)` (``normalize_laf``'s frame).
        h: input height.
        w: input width.

    Returns:
        ``k`` of shape :math:`(2,)` and ``t`` of shape :math:`(B, N, 2)`, in ``nlaf``'s dtype and device.
    """
    # `denormalize_laf`'s extents: a singleton axis counts as one pixel.
    wf, hf = float(max(w - 1, 1)), float(max(h - 1, 1))
    min_size = min(wf, hf)
    k = nlaf.new_tensor([2.0 * min_size / float(w), 2.0 * min_size / float(h)])
    t = (2.0 * nlaf[..., :, 2] * nlaf.new_tensor([wf, hf]) + 1.0) / nlaf.new_tensor([float(w), float(h)]) - 1.0
    return k, t


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
    k, t = _pyramid_grid_frame(grid_laf, heights[0], widths[0])

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
            grid = base_grid * k + translation
            patches = _sample_patches(cur_img, grid, h_l, w_l).to(img.dtype)
            mask = (pyr_idx[:, st:en] == level_idx).view(B, en - st, 1, 1, 1)
            out[:, st:en] = torch.where(mask, patches, out[:, st:en])
    return out


def extract_patches_simple(
    img: torch.Tensor, laf: torch.Tensor, PS: int = 32, normalize_lafs_before_extraction: bool = True
) -> torch.Tensor:
    """Extract patches defined by LAFs from image torch.Tensor.

    Convention:
        See the Convention block on :func:`~kornia.feature.extract_patches_from_pyramid`. This function samples the
        same positions, always from ``img`` itself: a LAF larger than the patch is not smoothed first, so it aliases.

    Args:
        img: images, LAFs are detected in  :math:`(B, CH, H, W)`.
        laf: :math:`(B, N, 2, 3)`.
        PS: patch size.
        normalize_lafs_before_extraction: if True, lafs are normalized to image size; if False, ``laf`` must already
            be in the frame that :func:`~kornia.feature.normalize_laf` gives for ``img``.

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
    nlaf = normalize_laf(laf, img) if normalize_lafs_before_extraction else laf
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

    Patches are extracted from the appropriate pyramid level.

    Convention:
        - Cell ``(i, j)`` of a ``PS x PS`` patch samples the image at ``c + A @ (u_j, u_i)``, with
          ``u_k = (2 k + 1) / PS - 1`` and the LAF ``[A | c]`` in pixels, pixel centres at integers. Row 0 is on the
          patch's ``-y`` side, the top for an upright LAF, the columns run along the patch's x axis, and the cells
          tile the square ``c + A [-1, 1]^2``. For an odd ``PS``, a LAF with ``A = (PS / 2) I`` centred on a pixel
          returns the image crop around that pixel.
        - The pyramid level is ``floor(log2(2 s / PS))`` for the scale ``s = sqrt(|det A|)``, clamped to the levels
          that exist: :func:`~kornia.geometry.transform.pyrdown` builds them while the shorter side stays at least
          ``PS``, so a larger LAF is sampled from the coarsest level that can still provide a full patch. A level
          samples the same image positions as level 0, in its own pixels.
        - A cell outside the image reads the nearest border pixel of the level it samples.
        - The patches have the image's dtype. The sampling runs in the promoted dtype of the image and the LAF, and
          in at least ``float32``. A LAF with a non-finite entry gives a zero patch.

    Args:
        img: images, LAFs are detected in  :math:`(B, CH, H, W)`.
        laf: :math:`(B, N, 2, 3)`.
        PS: patch size.
        normalize_lafs_before_extraction: if True, lafs are normalized to image size; if False, ``laf`` must already
            be in the frame that :func:`~kornia.feature.normalize_laf` gives for ``img``.

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
    nlaf = normalize_laf(laf, img) if normalize_lafs_before_extraction else laf
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
    # full-atlas recast is paid per chunk.
    atlas_elements = B * ch * atlas_h * atlas_w
    atlas_bytes = atlas_elements * _grid_elem_bytes(grid_dtype)
    if img.dtype != grid_dtype:
        atlas_bytes += B * ch * h * w * _grid_elem_bytes(grid_dtype)  # the one-time upcast copy
    if not _pyramid_atlas_fits(atlas_bytes):
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
    k, t = _pyramid_grid_frame(nlaf.to(grid_dtype), h, w)

    # Gather all level-dependent conversion constants per patch, packed as (x, y) pairs so the
    # remap below runs on the whole grid tensor step by step, never holding split-axis copies. A
    # giant LAF that nominally selects an unbuilt level is sampled from the actual coarsest
    # pyramid image.
    xoff = 0
    constants = []
    for h_l, w_l in zip(heights, widths):
        constants.append(
            (
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
    level_constants = torch.tensor(constants, dtype=grid_dtype, device=nlaf.device).view(-1, 4, 2)

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
        lo, hi, level_scale, level_offset = level_constants[safe_pyr_idx[:, st:en]].view(B, nc, 1, 1, 4, 2).unbind(-2)
        grid = grid * k
        grid = grid + t[:, st:en].view(B, nc, 1, 1, 2)
        grid = grid.maximum(lo)
        grid = grid.minimum(hi)
        grid = grid * level_scale
        grid = grid + level_offset
        out[:, st:en] = _sample_patches(atlas, grid, atlas_h, atlas_w)
    return out.masked_fill_((pyr_idx < 0).view(B, N, 1, 1, 1), 0.0)


def laf_is_inside_image(laf: torch.Tensor, images: torch.Tensor, border: int = 0) -> torch.Tensor:
    """Check which LAFs lie fully inside the image.

    Convention:
        - The region of a LAF ``[A | c]`` is the ellipse ``c + A u`` over the unit vectors ``u``. Along image axis
          ``i`` (0 for x, 1 for y) it spans exactly ``c_i - r_i`` to ``c_i + r_i``, where ``r_i`` is the Euclidean
          norm of row ``i`` of ``A``. A LAF is inside when both spans lie in ``border <= x <= w - 1 - border`` and
          ``border <= y <= h - 1 - border``, so a region that touches the last valid pixel coordinate is inside.
        - The all-zero padding LAF is a point at the origin, so it counts as inside for ``border <= 0``, the default
          included; combine the mask with :func:`~kornia.feature.laf_is_filled` to drop the padding.

    Args:
        laf:  :math:`(B, N, 2, 3)`.
        images: images, lafs are detected in :math:`(B, CH, H, W)`.
        border: additional border.

    Returns:
        mask with shape :math:`(B, N)`.

    """
    KORNIA_CHECK_LAF(laf)
    _, _, h, w = images.size()
    # r_i = max over unit u of A[i, :] u, half the side of the ellipse's axis-aligned bounding box.
    extent = torch.linalg.vector_norm(laf[..., :2], dim=-1)
    lo = laf[..., 2] - extent
    hi = laf[..., 2] + extent
    # Valid pixel coordinates run 0 .. w-1 and 0 .. h-1, matching the convention documented on
    # `get_laf_center` and the `w - 1` / `h - 1` extent used by `normalize_laf` / `denormalize_laf`.
    x_max = float(w - 1) - border
    y_max = float(h - 1) - border
    return (lo[..., 0] >= border) & (hi[..., 0] <= x_max) & (lo[..., 1] >= border) & (hi[..., 1] <= y_max)


def laf_to_three_points(laf: torch.Tensor) -> torch.Tensor:
    """Convert local affine frames (LAFs) to three points: the tips of the patch's x and y axes, and the centre.

    Convention:
        - The columns of the result are the points ``c + A[:, 0]``, ``c + A[:, 1]`` and ``c``, in that order: the
          images of the patch points ``(1, 0)``, ``(0, 1)`` and ``(0, 0)``, at the frame's scale rather than of unit
          length. :func:`~kornia.feature.laf_from_three_points` is the inverse.

    Args:
        laf:  :math:`(B, N, 2, 3)`.

    Returns:
        threepts :math:`(B, N, 2, 3)`.

    """
    KORNIA_CHECK_LAF(laf)
    return torch.stack([laf[..., 2] + laf[..., 0], laf[..., 2] + laf[..., 1], laf[..., 2]], dim=-1)


def laf_from_three_points(threepts: torch.Tensor) -> torch.Tensor:
    """Convert three points to local affine frame.

    Convention:
        See the Convention block on :func:`~kornia.feature.laf_to_three_points`, which this inverts: the columns of
        ``threepts`` are the images of the patch points ``(1, 0)``, ``(0, 1)`` and ``(0, 0)``, in that order.

    Args:
        threepts: :math:`(B, N, 2, 3)`.

    Returns:
        laf :math:`(B, N, 2, 3)`.

    """
    KORNIA_CHECK_SHAPE(threepts, ["B", "N", "2", "3"])
    return torch.stack(
        [threepts[..., 0] - threepts[..., 2], threepts[..., 1] - threepts[..., 2], threepts[..., 2]], dim=-1
    )


def perspective_transform_lafs(trans_01: torch.Tensor, lafs_1: torch.Tensor) -> torch.Tensor:
    r"""Apply perspective transformations to a set of local affine frames (LAFs).

    Convention:
        - ``trans_01`` maps pixels of frame 1 to pixels of frame 0, as in
          :func:`~kornia.geometry.linalg.transform_points`, and the centre maps through it exactly. The new ``A``
          is the chord through the mapped tips of the patch axes: the three points of
          :func:`~kornia.feature.laf_to_three_points` are transformed and turned back into a LAF. For an affine
          ``trans_01`` this equals the Jacobian of ``trans_01`` at the centre times ``A``; for a perspective one the
          two differ, the more so the larger the LAF.

    Args:
        trans_01: torch.Tensor for perspective transformations of shape :math:`(B, 3, 3)`.
        lafs_1: torch.Tensor of lafs of shape :math:`(B, N, 2, 3)`.

    Returns:
        the transformed LAFs in frame 0, of shape :math:`(B, N, 2, 3)`.

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
