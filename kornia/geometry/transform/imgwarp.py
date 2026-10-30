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

"""Warp images and volumes by matrices or sampling maps, and build the matrices the warps take.

The matrix warps take channels-first images ``(B, C, H, W)`` or volumes ``(B, C, D, H, W)`` and return the size
``dsize``, ``(h, w)`` or ``(d, h, w)``. Their matrix maps **source** pixel coordinates to **destination** pixel
coordinates, with pixel centres at integers and ``x`` the column, except that :func:`homography_warp` and
:func:`homography_warp3d` take a destination→source homography, in normalized :math:`[-1, 1]` coordinates by default.
``align_corners`` defaults differ between these functions, and ``warp_affine3d``, ``warp_perspective3d`` and
``homography_warp3d`` in normalized coordinates are correct only at ``align_corners=True``
(`#4503 <https://github.com/kornia/kornia/issues/4503>`_).

Which function to use:

- An image by a pixel homography or affine matrix: :func:`warp_perspective` ``(B, 3, 3)`` or :func:`warp_affine`
  ``(B, 2, 3)``; a volume: :func:`warp_perspective3d` ``(B, 4, 4)`` or :func:`warp_affine3d` ``(B, 3, 4)``.
- An image or a volume by a destination→source homography in normalized coordinates: :func:`homography_warp`,
  :func:`homography_warp3d`.
- An image at given per-pixel source coordinates: :func:`remap`. The warped sampling grid alone, without sampling:
  :func:`warp_grid`, :func:`warp_grid3d`.
- A homography from four point pairs: :func:`get_perspective_transform`; from 3D box corners:
  :func:`get_perspective_transform3d`.
- A 2D affine matrix from parameters: rotation and scale :func:`get_rotation_matrix2d`, translation
  :func:`get_translation_matrix2d`, shear :func:`get_shear_matrix2d`, all of them :func:`get_affine_matrix2d`.
- A 3D affine matrix from parameters: rotation and scale :func:`get_projective_transform`, shear
  :func:`get_shear_matrix3d`, all of them :func:`get_affine_matrix3d`.
- The inverse of a ``(B, 2, 3)`` affine matrix: :func:`invert_affine_transform`. ``[R | t]`` from a rotation and a
  translation: :func:`projection_from_Rt`.
- To rotate, shift, scale or shear an image by parameters and keep its size, without building a matrix, use
  :func:`~kornia.geometry.transform.rotate`, :func:`~kornia.geometry.transform.translate`,
  :func:`~kornia.geometry.transform.scale` or :func:`~kornia.geometry.transform.shear`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.core.ops import eye_like
from kornia.core.utils import (
    _autocast_disabled,
    _inverse_3x3_closed_form,
    _normalize_to_float32_or_float64,
    _torch_grid_sample,
    _torch_inverse_cast,
    _torch_solve_cast,
)
from kornia.geometry.conversions import (
    angle_to_rotation_matrix,
    axis_angle_to_rotation_matrix,
    convert_affinematrix_to_homography,
    convert_affinematrix_to_homography3d,
    convert_points_from_homogeneous,
    normal_transform_pixel,
    normal_transform_pixel3d,
    normalize_homography,
    normalize_homography3d,
    normalize_pixel_coordinates,
)
from kornia.geometry.grid import create_meshgrid, create_meshgrid3d
from kornia.geometry.linalg import transform_points

__all__ = [
    "get_affine_matrix2d",
    "get_affine_matrix3d",
    "get_perspective_transform",
    "get_perspective_transform3d",
    "get_projective_transform",
    "get_rotation_matrix2d",
    "get_shear_matrix2d",
    "get_shear_matrix3d",
    "get_translation_matrix2d",
    "homography_warp",
    "homography_warp3d",
    "invert_affine_transform",
    "projection_from_Rt",
    "remap",
    "warp_affine",
    "warp_affine3d",
    "warp_grid",
    "warp_grid3d",
    "warp_perspective",
    "warp_perspective3d",
]


def _matrix_warp_grid_dtype(src: torch.Tensor, transform: torch.Tensor) -> torch.dtype:
    """Return the sampling-grid dtype the non-empty matrix-warp pipeline would build.

    ``normalize_homography`` scales ``transform`` by floating factors, so an integral or boolean
    matrix already fails there and must fail here too. Otherwise the base grid is created in
    ``src.dtype`` and combined with the normalized matrix, so the grid promotes the two: a float32
    matrix against a float64 image still samples in float64, while the reverse narrows and is
    rejected by ``grid_sample``.
    """
    if not transform.is_floating_point():
        raise RuntimeError(f"Expected a floating point transform, got {transform.dtype}.")
    return torch.promote_types(src.dtype, transform.dtype)


def _remap_grid_dtype(map_xy: torch.Tensor, normalized_coordinates: bool) -> torch.dtype:
    """Return the sampling-grid dtype the non-empty :func:`remap` pipeline would build.

    Already-normalized maps reach ``grid_sample`` untouched. Pixel maps first go through
    :func:`~kornia.geometry.conversions.normalize_pixel_coordinates`, which keeps a floating map in
    its own dtype and lifts an integral or boolean one into the default floating dtype.
    """
    if normalized_coordinates or map_xy.is_floating_point():
        return map_xy.dtype
    # Read the default floating dtype off a zero-element promotion rather than
    # ``torch.get_default_dtype()``, which TorchScript does not expose.
    return (map_xy.reshape(-1)[:0] * 1.0).dtype


def _runs_in_float32(src: torch.Tensor, other: torch.Tensor) -> bool:
    """Return whether a warp of ``src`` builds its sampling grid and samples in float32.

    torch's CPU float16/bfloat16 ``grid_sample`` returns wrong values, NaN and up to 3.4e38 once
    the source is about 128 px or larger (torch 2.9.1 and 2.14), and torch 2.5.1 has no such kernel
    at all; a half-precision sampling grid is also off by more than half a pixel at 256 px in
    bfloat16 (#5751). A CPU half warp therefore runs in float32 and returns the input dtype, as the LAF
    patch extractors already do. A mismatched ``other`` (transform or map) keeps the half path, so
    it is rejected exactly as before. Other devices and dtypes are unaffected.
    """
    return src.device.type == "cpu" and src.dtype in (torch.float16, torch.bfloat16) and other.dtype == src.dtype


def _unit_projective_scale(matrix: torch.Tensor) -> torch.Tensor:
    """Return the detached projective scale of ``matrix``: the value closest to 1 in its nonzero-magnitude range.

    A homography and any nonzero multiple of it describe the same map, yet
    :func:`kornia.geometry.convert_points_from_homogeneous` leaves a point with ``|w| <= eps``
    undivided using an *absolute* ``eps``. Feeding that routine a grid built from an arbitrarily
    scaled matrix therefore changes the sampled pixels instead of the warp. Dividing by this scale
    first puts every equivalent matrix in the same numeric range. Choosing between the largest and
    smallest nonzero magnitude, rather than just dividing by the largest, keeps a unit diagonal from
    being shrunk toward zero by large translations. The scale is a projective gauge, so detaching it
    leaves the warp's derivative unchanged.
    """
    magnitude = matrix.detach().abs()
    largest = magnitude.amax(dim=(-2, -1), keepdim=True)
    smallest = magnitude.masked_fill(magnitude == 0, float("inf")).amin(dim=(-2, -1), keepdim=True)
    scale = torch.minimum(largest, torch.ones_like(largest)).clamp_min(smallest)
    # An all-zero matrix leaves `smallest` infinite, and dividing by infinity would collapse the
    # warp's gradient to zero. Anything that is not a finite positive scale is left alone.
    usable = (scale > 0) & torch.isfinite(scale)
    return torch.where(usable, scale, torch.ones_like(scale))


def _empty_warp_output_2d(
    src: torch.Tensor,
    transform: torch.Tensor,
    dsize: tuple[int, int],
    mode: str,
    padding_mode: str,
    align_corners: bool,
    grid_dtype: torch.dtype,
    fill_value: Optional[torch.Tensor] = None,
    transform_batch_broadcast: str = "none",
    allow_fill: bool = True,
    operand: str = "transform",
) -> torch.Tensor:
    """Return an empty warp while retaining normal grid-sample validation and autograd links.

    ``operand`` names the second tensor in the error messages. ``remap`` has no ``transform``
    parameter — it delegates here with its stacked maps — so it must not be told about one.

    ``transform_batch_broadcast`` names the caller's singleton-batch rule, so the empty
    destination broadcasts and rejects exactly as its non-empty path does:
    ``"none"`` requires equal batches (``warp_perspective``), ``"src_when_larger"`` expands a
    singleton transform batch only to a source batch above one (``warp_affine``), and ``"src"``
    expands it to the source batch unconditionally, a zero batch included (``remap``).
    """
    if src.device != transform.device:
        raise RuntimeError(f"Expected src and {operand} on the same device, got {src.device} and {transform.device}.")
    # An integral ``src`` fails inside ``grid_sample`` itself on the non-empty path, so it must
    # fail here too — but name ``src`` rather than blaming the other operand for it below. (MPS is the
    # exception: its ``grid_sample`` samples an integral image back into ``int64`` instead of
    # rejecting it. Matching that would mean bilinear-sampling into an integer output, so the
    # cpu/cuda contract is the one enforced here.)
    if not src.is_floating_point():
        raise NotImplementedError(f"Expected a floating point src, got {src.dtype}.")
    # ``grid_sample`` requires the sampling grid in ``src.dtype``. ``grid_dtype`` is the dtype the
    # caller's non-empty pipeline would actually produce from ``transform``, so checking it here
    # makes a zero-sized ``dsize`` neither stricter nor laxer than a non-empty one.
    if grid_dtype != src.dtype:
        raise RuntimeError(f"Expected src and {operand} with the same dtype, got {src.dtype} and {transform.dtype}.")
    if src.dtype != transform.dtype:
        transform = transform.to(src.dtype)
    # Resolve the output batch before any clamping below hides a mismatch: the stand-ins are
    # clamped away from zero, so a 0/1 or 1/0 batch pairing would otherwise reach ``grid_sample``
    # as 1/1 and be silently accepted where the non-empty path broadcasts or rejects it.
    src_batch = src.shape[0]
    transform_batch = transform.shape[0]
    broadcasts_singleton = transform_batch_broadcast == "src" or (
        transform_batch_broadcast == "src_when_larger" and src_batch > 1
    )
    if transform_batch == 1 and broadcasts_singleton:
        transform_batch = src_batch
    if transform_batch != src_batch:
        raise RuntimeError(
            f"Expected src and {operand} with the same batch size, got {src_batch} and {transform.shape[0]}."
        )

    # ``grid_sample`` must never receive a zero-element operand here. MPS before torch 2.14 raises
    # ``[srcBuf length] > 0 ... Placeholder tensor is empty!`` for a zero-element grid or batch, even
    # against a 1x1 source, and kornia supports torch >= 2.0. Sample connected 1x1 stand-ins so the
    # remaining contracts -- dtype, device, mode and padding_mode -- are still validated by
    # ``grid_sample`` itself, then expand the sampled result to the empty destination, which keeps
    # the autograd links to ``src`` and ``transform``. The stand-ins are 1x1 in space rather than
    # ``dsize``-shaped so the work stays constant: a ``(0, 2_000_000)`` destination must not
    # materialize two million samples that the empty result then discards.
    sample_batch = max(src_batch, 1)
    sample_channels = max(src.shape[1], 1)
    grid_zero = transform.reshape(-1)[:1].sum() * 0.0
    grid = grid_zero.reshape(1, 1, 1, 1).expand(sample_batch, 1, 1, 2)

    src_zero = src.reshape(-1)[:1].sum() * 0.0
    sample_src = src_zero.reshape(1, 1, 1, 1).expand(sample_batch, sample_channels, 1, 1)

    if padding_mode == "fill" and allow_fill:
        if fill_value is None:
            fill_value = torch.zeros(sample_channels, device=src.device, dtype=src.dtype)
        sampled = _fill_and_warp(
            sample_src,
            grid,
            align_corners=align_corners,
            mode=mode,
            fill_value=fill_value,
            out_shape=[src_batch, src.shape[1], dsize[0], dsize[1]],
        )
    else:
        sampled = _torch_grid_sample(
            sample_src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode
        )
    out_zero = sampled.reshape(-1)[:1].sum() * 0.0
    return out_zero.reshape(1, 1, 1, 1).expand(src_batch, src.shape[1], dsize[0], dsize[1])


def _empty_warp_output_3d(
    src: torch.Tensor,
    transform: torch.Tensor,
    dsize: tuple[int, int, int],
    mode: str,
    padding_mode: str,
    align_corners: bool,
) -> torch.Tensor:
    """Return an empty volume warp with normal grid-sample validation and autograd links."""
    if src.device != transform.device:
        raise RuntimeError(f"Expected src and transform on the same device, got {src.device} and {transform.device}.")
    # As in the 2-D helper: an integral ``src`` cannot reach cpu/cuda ``grid_sample`` on either
    # path, so reject it by name instead of via the transform-dtype message below.
    if not src.is_floating_point():
        raise NotImplementedError(f"Expected a floating point src, got {src.dtype}.")
    if src.dtype != transform.dtype:
        raise RuntimeError(f"Expected src and transform with the same dtype, got {src.dtype} and {transform.dtype}.")
    grid_zero = transform.reshape(-1)[:1].sum() * 0.0
    grid = grid_zero.reshape(1, 1, 1, 1, 1).expand(transform.shape[0], dsize[0], dsize[1], dsize[2], 3)

    sample_src = src
    if src.shape[-3] == 0 or src.shape[-2] == 0 or src.shape[-1] == 0:
        src_zero = src.reshape(-1)[:1].sum() * 0.0
        sample_src = src_zero.reshape(1, 1, 1, 1, 1).expand(src.shape[0], src.shape[1], 1, 1, 1)
    return _torch_grid_sample(sample_src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def warp_perspective(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    fill_value: Optional[torch.Tensor] = None,  # needed for jit
) -> torch.Tensor:
    r"""Warp an image by a source→destination homography in pixel coordinates.

    ``out = warp_perspective(src, M, dsize)`` takes images ``(B, C, H, W)`` and one ``(B, 3, 3)`` homography per image
    that maps **source** pixels to **destination** pixels, and returns the ``(B, C, h, w)`` destination images for
    ``dsize = (h, w)``. Pixel centres are integers, so an identity ``M`` with a smaller ``dsize`` returns the top-left
    ``dsize`` window of the image, not a resized image. This is the warp for the homographies that
    :func:`get_perspective_transform`, :func:`~kornia.geometry.homography.find_homography_dlt` and
    :class:`~kornia.geometry.ransac.RANSAC` return; for a destination→source homography in normalized :math:`[-1, 1]`
    coordinates, use :func:`homography_warp`, and for a ``(B, 2, 3)`` affine matrix, :func:`warp_affine`.

    Each output pixel ``(x, y)`` samples the source at the inverse of ``M`` applied to it:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M^{-1}_{11} x + M^{-1}_{12} y + M^{-1}_{13}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}} ,
        \frac{M^{-1}_{21} x + M^{-1}_{22} y + M^{-1}_{23}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}}
        \right )

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for transform direction, pixel centres,
        normalized coordinates and ``align_corners`` sampling rules.

        - ``M`` is the source→destination **pixel** homography ``(B, 3, 3)``
          (contrast :func:`homography_warp`, which by default consumes destination→source normalized)
        - ``M`` has the batch of ``src``: a ``(1, 3, 3)`` ``M`` is not shared by a larger batch, as it is in
          :func:`warp_affine`
        - a nonzero scalar multiple of ``M`` describes the same warp; uniformly small or large
          nonzero entries are rescaled toward unit magnitude before the coordinate change and inversion
        - ``src`` sets the dtype: a float32 ``M`` with a float64 ``src`` samples in float64, and a float64 ``M`` with a
          float32 ``src`` raises ``RuntimeError``
        - a zero output dimension returns an autograd-connected empty tensor;
          negative output dimensions raise ``ValueError``

    Args:
        src: input image with shape ``(B, C, H, W)``.
        M: transformation matrix with shape ``(B, 3, 3)``.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``
            | ``'fill'``.
        align_corners: interpolation flag.
        fill_value: torch.Tensor of shape ``(C)`` or ``(1)`` that fills the padding area, or any shape that
            broadcasts to the output ``(B, C, h, w)`` without changing it, such as a per-sample ``(B, C, 1, 1)``
            fill. Defaults to zeros.

    Returns:
        the warped input image ``(B, C, h, w)``, spatial sizes given by ``dsize``.

    See Also:
        - :func:`homography_warp`: a destination→source homography, normalized to :math:`[-1, 1]` by default.
        - :func:`warp_perspective3d`: the same warp for ``(B, C, D, H, W)`` volumes by ``(B, 4, 4)`` homographies.
        - :func:`warp_affine`: a ``(B, 2, 3)`` affine matrix instead, where a single ``(1, 2, 3)`` one serves a batch.
        - :func:`get_perspective_transform`: the homography that maps four source points onto four destination points.
        - :func:`remap`: sample at given per-pixel source coordinates instead of by a matrix.
        - :func:`crop_by_transform_mat`: dispatches to this warp or :func:`warp_affine` by matrix shape, casting the
          matrix to the image dtype and broadcasting a batch-1 matrix.
        - :func:`~kornia.geometry.conversions.normalize_homography`: ``M`` in normalized :math:`[-1, 1]` coordinates,
          the conversion this warp applies before sampling.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> H = torch.eye(3)[None]
       >>> out = warp_perspective(img, H, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
        See a working example `here <https://www.kornia.org/tutorials/nbs/warp_perspective.html>`_.

    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

    if not len(src.shape) == 4:
        raise ValueError(f"Input src must be a BxCxHxW torch.Tensor. Got {src.shape}")

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError(f"Input M must be a Bx3x3 torch.Tensor. Got {M.shape}")

    B, _, H, W = src.size()
    h_out, w_out = dsize
    if h_out < 0 or w_out < 0:
        raise ValueError(f"Output size must be non-negative. Got {dsize}.")
    if h_out == 0 or w_out == 0:
        return _empty_warp_output_2d(
            src, M, dsize, mode, padding_mode, align_corners, _matrix_warp_grid_dtype(src, M), fill_value
        )
    out_dtype = src.dtype
    if _runs_in_float32(src, M):
        src, M = src.float(), M.float()

    # H and s*H represent the same projective map. Remove that arbitrary scale before
    # the coordinate change and inverse: the cubic determinant can overflow/underflow
    # even when every entry, the sampling positions and their gradients are finite.
    if M.is_floating_point():
        M = M / _unit_projective_scale(M)

    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(
        M, (H, W), (h_out, w_out), align_corners=align_corners
    )  # Bx3x3

    # Closed-form 3x3 inverse (pure arithmetic) instead of ``torch.linalg.inv``: numerically
    # equivalent for these well-conditioned transforms, and it runs where the LAPACK/cusolver
    # backend is unavailable (e.g. the Jetson wheel, where ``linalg.inv`` dlopen-fails and would
    # otherwise crash every warp on GPU).
    src_norm_trans_dst_norm = _inverse_3x3_closed_form(dst_norm_trans_src_norm)  # Bx3x3

    # Substitutes F.affine_grid (which only handles the affine 2x3 case) by applying the full 3x3
    # projective transform to every grid point directly.
    grid = create_meshgrid(
        h_out, w_out, normalized_coordinates=True, device=src.device, dtype=src.dtype, align_corners=align_corners
    )
    if torch.jit.is_tracing():
        # Under tracing/ONNX use the reference transform_points path (its op set exports cleanly).
        grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid.expand(B, h_out, w_out, 2))
    else:
        # Eager fast path: inline the per-point projective map (three broadcasts and a divide),
        # avoiding transform_points' repeat_interleave of the matrix to B*H copies + homogeneous
        # round-trip.
        gx0, gy0 = grid[..., 0], grid[..., 1]  # (1, H, W)
        m = src_norm_trans_dst_norm  # (B, 3, 3)
        denom = m[:, 2, 0, None, None] * gx0 + m[:, 2, 1, None, None] * gy0 + m[:, 2, 2, None, None]
        gx = (m[:, 0, 0, None, None] * gx0 + m[:, 0, 1, None, None] * gy0 + m[:, 0, 2, None, None]) / denom
        gy = (m[:, 1, 0, None, None] * gx0 + m[:, 1, 1, None, None] * gy0 + m[:, 1, 2, None, None]) / denom
        grid = torch.stack([gx, gy], dim=-1)  # (B, H, W, 2)

    if padding_mode == "fill":
        # One zero per channel, as in warp_affine. It is built here, not before the empty-destination return, whose
        # helper builds its own default for its stand-in channels (#6021).
        if fill_value is None:
            fill_value = torch.zeros(src.shape[1], device=src.device, dtype=src.dtype)
        return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value).to(out_dtype)
    return _torch_grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode).to(
        out_dtype
    )


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    fill_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Warp an image by a source→destination affine matrix in pixel coordinates.

    ``out = warp_affine(src, M, dsize)`` takes images ``(B, C, H, W)`` and ``(B, 2, 3)`` affine matrices that map
    **source** pixels to **destination** pixels, and returns the ``(B, C, h, w)`` destination images for
    ``dsize = (h, w)``. A single ``(1, 2, 3)`` matrix serves every image of a non-empty batch. Build ``M`` with
    :func:`get_rotation_matrix2d`, or take the top two rows ``M[:, :2]`` of the ``(B, 3, 3)`` matrix of
    :func:`get_affine_matrix2d`; :func:`invert_affine_transform` gives the matrix that warps back. To keep the input
    size, use :func:`affine`, or pass an angle, a shift or a scale instead of a matrix to :func:`rotate`,
    :func:`translate` or :func:`scale`; for a ``(B, 3, 3)`` homography, use :func:`warp_perspective`.

    .. image:: _static/img/warp_affine.png

    Each output pixel ``(x, y)`` samples the source at:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M^{-1}_{11} x + M^{-1}_{12} y + M^{-1}_{13} ,
        M^{-1}_{21} x + M^{-1}_{22} y + M^{-1}_{23} \right )

    where :math:`M^{-1}` is the inverse of the :math:`3 \times 3` homogeneous extension of ``M``.

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for transform direction, pixel centres,
        normalized coordinates and ``align_corners`` sampling rules.

        - ``M`` is the source→destination **pixel** affine matrix ``(B, 2, 3)``
        - ``src`` sets the dtype: a float32 ``M`` with a float64 ``src`` samples in float64, and a float64 ``M`` with a
          float32 ``src`` raises ``RuntimeError``
        - a zero output dimension returns an autograd-connected empty tensor;
          negative output dimensions raise ``ValueError``

    Args:
        src: input torch.Tensor of shape ``(B, C, H, W)``.
        M: affine transformation of shape ``(B, 2, 3)``, or ``(1, 2, 3)`` for the whole batch.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``
            | ``'fill'``.
        align_corners : mode for grid_generation.
        fill_value: torch.Tensor of shape ``(C)`` or ``(1)`` that fills the padding area, or any shape that
            broadcasts to the output ``(B, C, h, w)`` without changing it, such as a per-sample ``(B, C, 1, 1)``
            fill. Defaults to zeros.

    Returns:
        the warped torch.Tensor with shape ``(B, C, h, w)``, spatial sizes given by ``dsize``.

    See Also:
        - :func:`warp_perspective`: a ``(B, 3, 3)`` homography per image, for perspective warps.
        - :func:`warp_affine3d`: the same warp for ``(B, C, D, H, W)`` volumes by ``(B, 3, 4)`` matrices.
        - :func:`affine`: keeps the input size and also takes an unbatched ``(C, H, W)`` image.
        - :func:`invert_affine_transform`: the destination→source matrix, to warp back.
        - :func:`get_affine_matrix2d`: a ``(B, 3, 3)`` matrix from translation, rotation, scale and shear; pass its
          ``M[:, :2]``.
        - :func:`crop_by_transform_mat`: calls this warp for a ``(B, 2, 3)`` matrix after casting it to the image dtype,
          and :func:`warp_perspective` for a ``(B, 3, 3)`` one.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> A = torch.eye(2, 3)[None]
       >>> out = warp_affine(img, A, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/rotate_affine.html>`__.

    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

    if not len(src.shape) == 4:
        raise ValueError(f"Input src must be a BxCxHxW torch.Tensor. Got {src.shape}")

    if not (len(M.shape) == 3 and M.shape[-2:] == (2, 3)):
        raise ValueError(f"Input M must be a Bx2x3 torch.Tensor. Got {M.shape}")

    B, C, H, W = src.size()
    B_M = M.shape[0]
    if dsize[0] < 0 or dsize[1] < 0:
        raise ValueError(f"Output size must be non-negative. Got {dsize}.")
    if dsize[0] == 0 or dsize[1] == 0:
        return _empty_warp_output_2d(
            src,
            M,
            dsize,
            mode,
            padding_mode,
            align_corners,
            _matrix_warp_grid_dtype(src, M),
            fill_value,
            transform_batch_broadcast="src_when_larger",
        )
    out_dtype = src.dtype
    if _runs_in_float32(src, M):
        src, M = src.float(), M.float()

    M_3x3: torch.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M_3x3, (H, W), dsize, align_corners=align_corners)

    # Closed-form 3x3 inverse (see warp_perspective) — cusolver-free, so affine warps run on the
    # Jetson wheel where ``torch.linalg.inv`` dlopen-fails.
    src_norm_trans_dst_norm = _inverse_3x3_closed_form(dst_norm_trans_src_norm)

    # Generate the grid from the matrix batch. When a single shared transform is passed for a
    # batch of images (``M`` is ``1x2x3`` with ``src`` ``BxCxHxW``), build one grid and expand it
    # (a stride-0 view, no copy) instead of materializing B identical grids — this is what makes a
    # batch-shared warp as cheap as torchvision's. When ``B_M == B`` (per-sample) nothing changes.
    if torch.jit.is_tracing():
        # Under tracing/ONNX use F.affine_grid (a single op the exporter lowers cleanly).
        grid = F.affine_grid(
            src_norm_trans_dst_norm[:, :2, :], [B_M, C, dsize[0], dsize[1]], align_corners=align_corners
        )
    else:
        # Eager fast path: apply the affine matrix to a base grid directly instead of F.affine_grid
        # — three broadcast multiply-adds per axis, markedly cheaper on launch-bound hardware. The
        # base grid reproduces F.affine_grid's convention for the requested ``align_corners`` (pixel
        # corners at +/-1 vs pixel centers). A shared (1x2x3) matrix broadcasts across the batch.
        h_out, w_out = dsize
        if align_corners:
            xs = (
                torch.zeros(1, device=src.device, dtype=src.dtype)
                if w_out == 1
                else torch.linspace(-1.0, 1.0, w_out, device=src.device, dtype=src.dtype)
            )
            ys = (
                torch.zeros(1, device=src.device, dtype=src.dtype)
                if h_out == 1
                else torch.linspace(-1.0, 1.0, h_out, device=src.device, dtype=src.dtype)
            )
        else:
            xs = torch.linspace(-1.0 + 1.0 / w_out, 1.0 - 1.0 / w_out, w_out, device=src.device, dtype=src.dtype)
            ys = torch.linspace(-1.0 + 1.0 / h_out, 1.0 - 1.0 / h_out, h_out, device=src.device, dtype=src.dtype)
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        m = src_norm_trans_dst_norm  # (B_M, 3, 3); affine, so the bottom row is [0, 0, 1]
        gx = m[:, 0, 0, None, None] * base_x + m[:, 0, 1, None, None] * base_y + m[:, 0, 2, None, None]
        gy = m[:, 1, 0, None, None] * base_x + m[:, 1, 1, None, None] * base_y + m[:, 1, 2, None, None]
        grid = torch.stack([gx, gy], dim=-1)  # (B_M, H, W, 2)
    if torch.jit.is_tracing():
        # Keep both batch sizes dynamic. Using their maximum expands a shared grid for non-empty images,
        # but leaves a singleton grid intact for an empty source so grid_sample still rejects that pairing.
        batch = torch.maximum(torch._shape_as_tensor(src)[0], torch._shape_as_tensor(M)[0])
        grid = grid.expand(batch, -1, -1, -1)
    elif B_M == 1 and B > 1:
        grid = grid.expand(B, -1, -1, -1)

    if padding_mode == "fill":
        if fill_value is None:
            fill_value = torch.zeros(src.shape[1], device=src.device, dtype=src.dtype)
        return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value).to(out_dtype)

    return _torch_grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode).to(
        out_dtype
    )


def _fill_and_warp(
    src: torch.Tensor,
    grid: torch.Tensor,
    mode: str,
    align_corners: bool,
    fill_value: torch.Tensor,
    out_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    r"""Warp a mask of torch.ones, then multiply with fill_value and add to default warp.

    Args:
        src: input torch.Tensor of shape ``(B, C, H, W)``.
        grid: grid torch.Tensor from `transform_points`.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        align_corners: interpolation flag.
        fill_value: torch.Tensor of shape ``(C)`` or ``(1)`` that fills the padding area, or any shape that
            broadcasts to the output ``(B, C, H, W)`` without changing it, such as ``(B, C, 1, 1)``.
        out_shape: the output shape ``fill_value`` is checked against, when ``src`` and ``grid`` stand in for an
            empty destination; the fill is then reduced to one value. Defaults to the shape sampled from ``src``
            and ``grid``.

    Returns:
        the warped and filled torch.Tensor with shape ``(B, C, H, W)``.
    """
    stand_in = out_shape is not None
    if out_shape is None:
        out_shape = [src.shape[0], src.shape[1], grid.shape[1], grid.shape[2]]
    # 0-d and 1-D fills are per channel; any other fill broadcasts as given. Either way it must leave the output
    # shape unchanged: a fill that broadcast to more channels turned a one-channel image into three (#6021).
    fill_shape = list(fill_value.shape)
    if len(fill_shape) == 1:
        fill_shape = [1, fill_shape[0], 1, 1]
    fits = len(fill_shape) <= 4
    for i in range(1, min(len(fill_shape), 4) + 1):
        if fill_shape[-i] != 1 and fill_shape[-i] != out_shape[-i]:
            fits = False
    if not fits:
        raise ValueError(
            f"fill_value of shape {list(fill_value.shape)} does not broadcast to the warped image of shape "
            f"{out_shape} without changing it; use one value, one per channel ({out_shape[1]}), or a shape such as "
            "(B, C, 1, 1) or (B, C, H, W)."
        )
    if stand_in:
        # The stand-ins' result is discarded: a constant-size fill, still connected to autograd, keeps a spatial fill
        # from broadcasting the 1x1 stand-ins up to its own size.
        fill_value = fill_value.sum().reshape(1)
    ones_mask = torch.ones_like(src)
    fill_value = fill_value.to(ones_mask)

    if fill_value.ndim == 0:
        fill_value = fill_value.view(1, 1, 1, 1)
    elif fill_value.ndim == 1:
        fill_value = fill_value.view(1, -1, 1, 1)

    inv_ones_mask = 1 - _torch_grid_sample(
        ones_mask, grid, align_corners=align_corners, mode=mode, padding_mode="zeros"
    )

    inv_color_mask = inv_ones_mask * fill_value

    return _torch_grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode="zeros") + inv_color_mask


def warp_grid(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    """Warp a grid of 2D points by homographies, returning the coordinates without sampling an image.

    ``warped = warp_grid(grid, src_homo_dst)`` applies the homographies ``src_homo_dst`` to every ``(x, y)`` point of
    ``grid`` and returns the ``(N, H, W, 2)`` points after the homogeneous division, in the dtype of ``src_homo_dst``.
    It does not normalize: the points stay in the coordinates that ``grid`` and the homography share, so a normalized
    grid from :func:`~kornia.geometry.grid.create_meshgrid` and a normalized homography give positions that
    :func:`torch.nn.functional.grid_sample` reads directly. To warp an image, use :func:`homography_warp`, which builds
    such a grid, warps it with this function and samples the image.

    Convention:
        - ``grid`` coordinates: ``(x, y)`` (last dim), shape ``(1, H, W, 2)`` or ``(N, H, W, 2)``
        - ``src_homo_dst`` is the destination→source homography ``(1, 3, 3)``,
          ``(N, 3, 3)`` or ``(N, 1, 3, 3)``
        - only the ``(1, H, W, 2)`` grid broadcasts over homography batches; a batched grid
          requires a matching batch of ``N`` homographies
        - ``src_homo_dst`` is divided by a detached scale toward unit magnitude before the
          homogeneous division, as in :func:`warp_perspective`, so a nonzero scalar multiple of it
          returns the same grid, except at grid points whose homogeneous coordinate is within
          ``1e-8`` of zero after that rescaling: those are left undivided, as
          :func:`~kornia.geometry.conversions.convert_points_from_homogeneous` does for points at
          infinity

    Args:
        grid: Unwrapped grid of the shape ``(1, H, W, 2)``, or ``(N, H, W, 2)`` with a
          matching batch of ``N`` homographies.
        src_homo_dst: Homography or homographies (stacked) to
          transform all points in the grid. Shape of the homography
          has to be ``(1, 3, 3)``, ``(N, 3, 3)`` or ``(N, 1, 3, 3)``.

    Returns:
        the transformed grid of shape ``(N, H, W, 2)``.

    See Also:
        - :func:`warp_grid3d`: the same for a 3D ``(x, y, z)`` grid and ``(N, 4, 4)`` homographies.
        - :func:`homography_warp`: builds the grid, warps it with this function and samples the image.
        - :func:`~kornia.geometry.grid.create_meshgrid`: the ``(1, H, W, 2)`` grid to pass in, in pixels or normalized.
        - :func:`~kornia.geometry.linalg.transform_points`: transform any ``(B, N, 2)`` point set by ``(B, 3, 3)``
          matrices.

    """
    batch_size: int = src_homo_dst.size(0)
    _, height, width, _ = grid.size()
    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 3, 3)  # Nx1x3x3
    # H and s*H are the same warp, but the absolute eps inside convert_points_from_homogeneous
    # would otherwise make the sampled grid depend on the matrix's arbitrary scale.
    if src_homo_dst.is_floating_point():
        src_homo_dst = src_homo_dst / _unit_projective_scale(src_homo_dst)
    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow: torch.Tensor = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxHxWx2
    return flow.view(batch_size, height, width, 2)  # NxHxWx2


def _pixel_grid_work_dtype(src_homo_dst: torch.Tensor) -> torch.dtype:
    """Return the dtype a pixel-coordinate sampling grid is warped and normalized in: float32 for a half homography.

    Pixel coordinates reach the image size, where float16 rounds them to half pixels from 512 and to whole pixels from
    1024, and bfloat16 already from 64 and 128. The grid is therefore warped and normalized in float32, and only the
    normalized result, whose values lie near ``[-1, 1]``, returns to the homography's dtype. Other dtypes are kept.
    """
    if src_homo_dst.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return src_homo_dst.dtype


def _warp_pixel_grid(src_homo_dst: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Warp the ``(height, width)`` grid of destination pixels by ``src_homo_dst`` into source pixel coordinates.

    The grid is built and warped in :func:`_pixel_grid_work_dtype`, whatever the default dtype, with the projective
    gauge and homogeneous division of :func:`warp_grid`. The homography is applied as a sum of elementwise column
    products rather than a matrix product: a caller's autocast region would run a matrix product in half precision,
    also in a scripted call, which cannot leave it, and round the pixel coordinates of a 640 px wide grid to whole
    pixels in bfloat16. Autocast leaves elementwise products alone, so eager and scripted calls compute the same grid.
    """
    work_dtype = _pixel_grid_work_dtype(src_homo_dst)
    grid = create_meshgrid(height, width, normalized_coordinates=False, device=src_homo_dst.device, dtype=work_dtype)
    matrix = src_homo_dst.to(work_dtype)
    matrix = matrix / _unit_projective_scale(matrix)
    # Broadcasting covers both (N, 3, 3) and (N, 1, 3, 3).
    matrix = matrix.reshape(src_homo_dst.shape[0], 1, 1, 3, 3)
    warped = grid[..., :1] * matrix[..., 0] + grid[..., 1:] * matrix[..., 1] + matrix[..., 2]
    return convert_points_from_homogeneous(warped)


def _normalize_warped_pixel_grid(grid: torch.Tensor, height: int, width: int, align_corners: bool) -> torch.Tensor:
    """Map a ``(N, H, W, 2)`` grid of source pixel coordinates to the ``[-1, 1]`` coordinates ``grid_sample`` reads.

    ``height`` and ``width`` are the source's. The map is :func:`~kornia.geometry.conversions.normal_transform_pixel`
    under ``align_corners``, the one :func:`warp_perspective` conjugates its pixel homography with, applied per axis.
    """
    norm = normal_transform_pixel(height, width, device=grid.device, dtype=grid.dtype, align_corners=align_corners)[0]
    scale = torch.stack((norm[0, 0], norm[1, 1]))
    offset = torch.stack((norm[0, 2], norm[1, 2]))
    return grid * scale + offset


def warp_grid3d(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    """Warp a grid of 3D points by 4x4 homographies, returning the coordinates without sampling a volume.

    ``warped = warp_grid3d(grid, src_homo_dst)`` applies the homographies ``src_homo_dst`` to every ``(x, y, z)`` point
    of ``grid`` and returns the ``(N, D, H, W, 3)`` points after the homogeneous division, in the dtype of
    ``src_homo_dst``. It does not normalize or reorder: the last axis must already be ``(x, y, z)``, the order the
    homography and :func:`torch.nn.functional.grid_sample` read, whereas
    :func:`~kornia.geometry.grid.create_meshgrid3d` returns ``(d, x, y)``. To warp a volume, use
    :func:`homography_warp3d`, which builds and reorders the grid, warps it with this function and samples the volume.

    Convention:
        - ``grid`` coordinates: ``(x, y, z)`` (last dim), shape ``(1, D, H, W, 3)`` or
          ``(N, D, H, W, 3)``
        - ``src_homo_dst`` is the destination→source homography ``(1, 4, 4)``,
          ``(N, 4, 4)`` or ``(N, 1, 4, 4)``
        - only the ``(1, D, H, W, 3)`` grid broadcasts over homography batches; a batched
          grid requires a matching batch of ``N`` homographies
        - ``src_homo_dst`` is divided by a detached scale toward unit magnitude before the
          homogeneous division, as in :func:`warp_perspective`, so a nonzero scalar multiple of it
          returns the same grid, except at grid points whose homogeneous coordinate is within
          ``1e-8`` of zero after that rescaling: those are left undivided, as
          :func:`~kornia.geometry.conversions.convert_points_from_homogeneous` does for points at
          infinity

    Args:
        grid: Unwrapped grid of the shape ``(1, D, H, W, 3)``, or ``(N, D, H, W, 3)``
          with a matching batch of ``N`` homographies.
        src_homo_dst: Homography or homographies (stacked) to
          transform all points in the grid. Shape of the homography
          has to be ``(1, 4, 4)``, ``(N, 4, 4)`` or ``(N, 1, 4, 4)``.

    Returns:
        the transformed grid of shape ``(N, D, H, W, 3)``.

    See Also:
        - :func:`warp_grid`: the same for a 2D ``(x, y)`` grid and ``(N, 3, 3)`` homographies.
        - :func:`homography_warp3d`: builds and reorders the grid, warps it with this function and samples the volume.
        - :func:`~kornia.geometry.grid.create_meshgrid3d`: a grid in ``(d, x, y)`` order, to reorder before passing it
          in.

    """
    batch_size: int = src_homo_dst.size(0)
    _, depth, height, width, _ = grid.size()
    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1, -1)  # NxDxHxWx3
    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 4, 4)  # Nx1x3x3
    # Same projective gauge as in warp_grid: a scaled 4x4 matrix must not move the grid.
    if src_homo_dst.is_floating_point():
        src_homo_dst = src_homo_dst / _unit_projective_scale(src_homo_dst)
    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow: torch.Tensor = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxDxHxWx3
    return flow.view(batch_size, depth, height, width, 3)  # NxDxHxWx3


def _warp_pixel_grid3d(src_homo_dst: torch.Tensor, depth: int, height: int, width: int) -> torch.Tensor:
    """Warp the ``(depth, height, width)`` grid of destination voxels by ``src_homo_dst`` into source voxels.

    The 3-D counterpart of :func:`_warp_pixel_grid`: the grid is built and warped in :func:`_pixel_grid_work_dtype`,
    whatever the default dtype, with the projective gauge and homogeneous division of :func:`warp_grid3d`, as a sum of
    elementwise column products that a caller's autocast region leaves alone, scripted or not. The result is
    ``(N, D, H, W, 3)`` in the ``(x, y, z)`` order the homography and ``grid_sample`` read.
    """
    work_dtype = _pixel_grid_work_dtype(src_homo_dst)
    grid = create_meshgrid3d(
        depth, height, width, normalized_coordinates=False, device=src_homo_dst.device, dtype=work_dtype
    )
    matrix = src_homo_dst.to(work_dtype)
    matrix = matrix / _unit_projective_scale(matrix)
    # Broadcasting covers both (N, 4, 4) and (N, 1, 4, 4).
    matrix = matrix.reshape(src_homo_dst.shape[0], 1, 1, 1, 4, 4)
    # create_meshgrid3d is (d, x, y); the homography acts on (x, y, z).
    x, y, z = grid[..., 1:2], grid[..., 2:3], grid[..., 0:1]
    warped = x * matrix[..., 0] + y * matrix[..., 1] + z * matrix[..., 2] + matrix[..., 3]
    return convert_points_from_homogeneous(warped)


def _normalize_warped_pixel_grid3d(
    grid: torch.Tensor, depth: int, height: int, width: int, align_corners: bool
) -> torch.Tensor:
    """Map a ``(N, D, H, W, 3)`` grid of ``(x, y, z)`` source voxels to the ``[-1, 1]`` coordinates of ``grid_sample``.

    ``depth``, ``height`` and ``width`` are the source's. With ``align_corners=True`` the map is
    :func:`~kornia.geometry.conversions.normal_transform_pixel3d`, the one :func:`warp_perspective3d` conjugates its
    voxel homography with, applied per axis. With ``False`` it is the half-voxel map ``(2v + 1) / size - 1``.
    """
    if align_corners:
        norm = normal_transform_pixel3d(depth, height, width, device=grid.device, dtype=grid.dtype)[0]
        scale = torch.stack((norm[0, 0], norm[1, 1], norm[2, 2]))
        offset = torch.stack((norm[0, 3], norm[1, 3], norm[2, 3]))
        return grid * scale + offset
    # normal_transform_pixel3d has no align_corners yet, so the half-voxel map is written out here; #4503 gives the
    # helper align_corners, and normal_transform_pixel3d(..., align_corners=False) then replaces these lines.
    x = (2.0 * grid[..., 0:1] + 1.0) / width - 1.0
    y = (2.0 * grid[..., 1:2] + 1.0) / height - 1.0
    z = (2.0 * grid[..., 2:3] + 1.0) / depth - 1.0
    return torch.cat((x, y, z), -1)


# TODO: move to kornia.geometry.projective
# TODO: create the nn.Module -- TBD what inputs/outputs etc
# class PerspectiveTransform(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


def _unit_square_to_quad(points: torch.Tensor) -> torch.Tensor:
    """Closed-form perspective matrix mapping the unit square to a quadrilateral.

    Maps the canonical unit-square corners ``[(0,0), (1,0), (1,1), (0,1)]`` to the
    four points in ``points`` (in the same corner order). Implemented via Heckbert's
    direct formulation (no linear solve) so it lowers cleanly to ONNX.

    Args:
        points: ``(B, 4, 2)`` tensor of quadrilateral vertices.

    Returns:
        ``(B, 3, 3)`` perspective transformation matrix.
    """
    x0 = points[..., 0, 0]
    y0 = points[..., 0, 1]
    x1 = points[..., 1, 0]
    y1 = points[..., 1, 1]
    x2 = points[..., 2, 0]
    y2 = points[..., 2, 1]
    x3 = points[..., 3, 0]
    y3 = points[..., 3, 1]

    dx1 = x1 - x2
    dx2 = x3 - x2
    sx = x0 - x1 + x2 - x3
    dy1 = y1 - y2
    dy2 = y3 - y2
    sy = y0 - y1 + y2 - y3

    denom = dx1 * dy2 - dy1 * dx2
    a31 = (sx * dy2 - sy * dx2) / denom
    a32 = (dx1 * sy - dy1 * sx) / denom
    a11 = x1 - x0 + a31 * x1
    a12 = x3 - x0 + a32 * x3
    a13 = x0
    a21 = y1 - y0 + a31 * y1
    a22 = y3 - y0 + a32 * y3
    a23 = y0

    one = torch.ones_like(x0)

    row0 = torch.stack([a11, a12, a13], dim=-1)
    row1 = torch.stack([a21, a22, a23], dim=-1)
    row2 = torch.stack([a31, a32, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _get_perspective_transform_closed_form(points_src: torch.Tensor, points_dst: torch.Tensor) -> torch.Tensor:
    """Perspective transform via two unit-square (Heckbert) decompositions.

    Computes ``H = H_d @ inv(H_s)`` where ``H_s`` maps the unit square to
    ``points_src`` and ``H_d`` maps it to ``points_dst``, so ``H`` maps
    ``points_src`` onto ``points_dst`` directly. Uses only arithmetic and a
    closed-form 3x3 inverse — no ``torch.linalg.solve`` — so it needs no
    LAPACK/cusolver backend (runs on the Jetson wheel) and traces cleanly for
    ONNX. The result is normalized to ``H[2, 2] == 1`` to match the DLT
    convention, unless that entry is zero up to rounding (a homography that sends
    the source origin to infinity) or that form overflows the input dtype; then it
    is divided by the geometric mean of its largest and smallest nonzero
    magnitudes instead. Computed in float32/float64 for numerical stability, then
    cast back to the input dtype.
    """
    dtype = points_src.dtype
    work_dtype = _normalize_to_float32_or_float64(dtype)
    h_s = _unit_square_to_quad(points_src.to(work_dtype))
    h_d = _unit_square_to_quad(points_dst.to(work_dtype))
    h_s_inv = _inverse_3x3_closed_form(h_s)
    transform = h_d @ h_s_inv
    # ``H[2, 2]`` is the dot product of ``h_d``'s last row and ``h_s_inv``'s last column; below a few units of rounding
    # of that sum, its value and sign are noise, and dividing by it gives NaN, inf or an arbitrary scale. The
    # ``H[2, 2] = 1`` form must also fit the output dtype: float16 overflows at 65504. Otherwise scale by the geometric
    # mean of the largest and smallest nonzero magnitudes, which centres the entries in the dtype's range: scaling the
    # largest to 1 would flush the small ones to zero in float16. Machine epsilon is written out, as TorchScript has no
    # ``torch.finfo``, and ``x - x == 0`` tests for a finite value with ops that every exporter supports.
    eps = 1.1920928955078125e-07 if work_dtype == torch.float32 else 2.220446049250313e-16
    h22 = transform[..., 2:3, 2:3]
    h22_rounding = h_d[..., 2:3, :].abs() @ h_s_inv[..., :, 2:3].abs()
    unit_h22 = (transform / h22).to(dtype)
    fits = ((unit_h22 - unit_h22) == 0).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
    magnitude = transform.abs()
    largest = magnitude.amax(dim=(-2, -1), keepdim=True)
    smallest = magnitude.masked_fill(magnitude == 0, float("inf")).amin(dim=(-2, -1), keepdim=True)
    scale = largest.sqrt() * smallest.sqrt()
    transform = transform / torch.where((h22.abs() > 8 * eps * h22_rounding) & fits, h22, scale)
    return transform.to(dtype)


def get_perspective_transform(points_src: torch.Tensor, points_dst: torch.Tensor) -> torch.Tensor:
    r"""Compute the homography that maps four source points exactly onto four destination points.

    ``M = get_perspective_transform(points_src, points_dst)`` takes two ``(B, 4, 2)`` sets of ``(x, y)`` points,
    matched by index, and returns the ``(B, 3, 3)`` homography, scaled to ``M[2, 2] = 1`` where it can be, that maps
    each source point onto its destination point: the source→destination pixel homography that
    :func:`warp_perspective` takes. It solves the four-point system in closed form, by Heckbert's
    square-to-quadrilateral mappings, rather than with the
    `DLT <https://www.cs.cmu.edu/~16385/s17/Slides/10.2_2D_Alignment__DLT.pdf>`_, whose exact solution it equals, so it
    needs no linear solver. For more than four or noisy correspondences, use
    :func:`~kornia.geometry.homography.find_homography_dlt` or :class:`~kornia.geometry.ransac.RANSAC`.

    The matrix maps each source point onto its destination point, up to the projective scale:

    .. math::

        \begin{bmatrix}
        x^{'} \\
        y^{'} \\
        1 \\
        \end{bmatrix}
        =
        \begin{bmatrix}
        h_1 & h_2 & h_3 \\
        h_4 & h_5 & h_6 \\
        h_7 & h_8 & h_9 \\
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
        x \\
        y \\
        1 \\
        \end{bmatrix}

    Convention:
        - points: ``(x, y)``, pixel centers, origin at top-left; shape ``(B, 4, 2)``
        - a degenerate quadrilateral, such as one with three collinear points or two equal ones, does not raise: a
          degenerate source returns a non-finite matrix, a degenerate destination a non-finite or a singular one
        - ``M`` is scaled to ``M[2, 2] = 1`` where that form exists and fits the output dtype. A homography whose
          bottom-right entry is zero up to rounding, such as one that sends source pixel ``(0, 0)`` to infinity, has
          no such form although neither quadrilateral is degenerate, and one whose ``M[2, 2] = 1`` form overflows the
          dtype, such as a float16 one with entries beyond ``65504``, is divided instead by the geometric mean of its
          largest and smallest nonzero magnitudes, which keeps both in range;
          :func:`~kornia.geometry.homography.find_homography_dlt` returns the same map at another scale
        - float16 and bfloat16 points are solved in float32, and the result is cast back to their dtype
        - a caller's autocast region does not change the result, outside TorchScript

    Args:
        points_src: coordinates of quadrangle vertices in the source image with shape ``(B, 4, 2)``.
        points_dst: coordinates of the corresponding quadrangle vertices in
            the destination image with shape ``(B, 4, 2)``.

    Returns:
        the perspective transformation with shape ``(B, 3, 3)``.

    See Also:
        - :func:`warp_perspective`: warp an image by the returned homography.
        - :func:`get_perspective_transform3d`: the 3D counterpart, a ``(B, 4, 4)`` homography from box corners.
        - :func:`~kornia.geometry.homography.find_homography_dlt`: a least-squares homography from any number of
          weighted point pairs.
        - :class:`~kornia.geometry.ransac.RANSAC`: a homography that is robust to wrong correspondences.

    Example:
        >>> x1 = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
        >>> x2 = torch.tensor([[[1., 0.], [0., 0.], [0., 1.], [1., 1.]]])
        >>> x2_trans_x1 = get_perspective_transform(x1, x2)

    """
    KORNIA_CHECK_SHAPE(points_src, ["B", "4", "2"])
    KORNIA_CHECK_SHAPE(points_dst, ["B", "4", "2"])
    KORNIA_CHECK(points_src.shape == points_dst.shape, "Source data shape must match Destination data shape.")
    KORNIA_CHECK(points_src.dtype == points_dst.dtype, "Source data type must match Destination data type.")

    # Solve via two closed-form unit-square (Heckbert) decompositions rather than an 8x8
    # ``torch.linalg.solve``. Both yield the perspective transform mapping ``points_src`` onto
    # ``points_dst``; the closed form additionally needs no LAPACK/cusolver backend (so it runs on
    # the Jetson wheel, where ``linalg.solve`` dlopen-fails) and traces cleanly for ONNX.
    # A caller's autocast region would run its products in half precision: about 1 px off at the corners of a 640 x
    # 480 image. TorchScript has no context managers; a scripted call keeps the caller's precision.
    if torch.jit.is_scripting():
        return _get_perspective_transform_closed_form(points_src, points_dst)
    with _autocast_disabled(points_src.device.type):
        return _get_perspective_transform_closed_form(points_src, points_dst)


# TODO: move to kornia.geometry.affine
def get_rotation_matrix2d(center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    r"""Compute the 2D affine matrix that scales and rotates counter-clockwise, in degrees, about a centre.

    ``M = get_rotation_matrix2d(center, angle, scale)`` takes centres ``(B, 2)`` as ``(x, y)`` pixels, angles ``(B,)``
    in degrees and scale factors ``(B, 2)`` for ``x`` and ``y``, and returns the ``(B, 2, 3)`` source→destination pixel
    matrix that scales and then rotates about ``center``, which stays fixed. A positive angle turns counter-clockwise as
    displayed, with ``y`` pointing down. Pass ``M`` to :func:`warp_affine`; to add a translation and a shear in one
    ``(B, 3, 3)`` matrix, use :func:`get_affine_matrix2d`, and to rotate an image directly, :func:`rotate`.

    With equal scale factors ``scale``, the matrix is:

    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}

    where

    .. math::
        \alpha = \text{scale} \cdot \cos(\text{angle}) \\
        \beta = \text{scale} \cdot \sin(\text{angle})

    With two different factors, the left 2x2 block is the rotation times ``diag(scale[:, 0], scale[:, 1])``.

    Convention:
        - ``center`` is ``(x, y)`` in pixels, origin at top-left
        - positive ``angle`` rotates counter-clockwise as displayed (y-down image axes)
        - returns ``(B, 2, 3)`` affine matrix in pixel coordinates

    Args:
        center: center of the rotation in the source image with shape ``(B, 2)``.
        angle: rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner) with shape ``(B)``.
        scale: scale factor for x, y scaling with shape ``(B, 2)``.

    Returns:
        the affine matrix of 2D rotation with shape ``(B, 2, 3)``.

    Raises:
        TypeError: If any input is not a tensor.
        ValueError: If input shapes are invalid, or batch sizes, devices or dtypes do not match.

    See Also:
        - :func:`get_affine_matrix2d`: also translates and shears, returns ``(B, 3, 3)``, and turns clockwise for a
          positive angle (`#4408 <https://github.com/kornia/kornia/issues/4408>`_).
        - :func:`rotate`: rotate an image about its centre without building the matrix.
        - :func:`get_projective_transform`: the 3D rotation and scale, by an axis-angle vector.
        - :func:`~kornia.geometry.conversions.angle_to_rotation_matrix`: the 2x2 rotation matrix of an angle alone.

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones((1, 2))
        >>> angle = 45. * torch.ones(1)
        >>> get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])

    """
    if not isinstance(center, torch.Tensor):
        raise TypeError(f"Input center type is not a torch.Tensor. Got {type(center)}")

    if not isinstance(angle, torch.Tensor):
        raise TypeError(f"Input angle type is not a torch.Tensor. Got {type(angle)}")

    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"Input scale type is not a torch.Tensor. Got {type(scale)}")

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError(f"Input center must be a Bx2 torch.Tensor. Got {center.shape}")

    if not len(angle.shape) == 1:
        raise ValueError(f"Input angle must be a B torch.Tensor. Got {angle.shape}")

    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError(f"Input scale must be a Bx2 torch.Tensor. Got {scale.shape}")

    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError(
            f"Inputs must have same batch size dimension. Got center {center.shape}, angle {angle.shape} and scale "
            f"{scale.shape}"
        )

    if not (center.device == angle.device == scale.device):
        raise ValueError(
            f"Inputs must have same device. Got center {center.device}, angle {angle.device} and scale {scale.device}."
        )

    if not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError(
            f"Inputs must have same dtype. Got center {center.dtype}, angle {angle.dtype} and scale {scale.dtype}."
        )

    shift_m = eye_like(3, center)
    shift_m[:, :2, 2] = center

    shift_m_inv = eye_like(3, center)
    shift_m_inv[:, :2, 2] = -center

    scale_m = eye_like(3, center)
    scale_m[:, 0, 0] *= scale[:, 0]
    scale_m[:, 1, 1] *= scale[:, 1]

    rotat_m = eye_like(3, center)
    rotat_m[:, :2, :2] = angle_to_rotation_matrix(angle)

    affine_m = shift_m @ rotat_m @ scale_m @ shift_m_inv
    return affine_m[:, :2, :]  # Bx2x3


def remap(
    image: torch.Tensor,
    map_x: torch.Tensor,
    map_y: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
    normalized_coordinates: bool = False,
) -> torch.Tensor:
    r"""Sample an image at per-pixel source coordinates given as an x map and a y map.

    ``out = remap(image, map_x, map_y)`` takes images ``(B, C, H, W)`` and two ``(B, h, w)`` maps that hold, for every
    output pixel, the source column ``x`` and row ``y`` to sample, and returns ``(B, C, h, w)``. The maps are absolute
    pixel coordinates, not displacements, so the identity maps are the two channels of
    :func:`~kornia.geometry.grid.create_meshgrid` with ``normalized_coordinates=False``; with
    ``normalized_coordinates=True`` they are the :math:`[-1, 1]` coordinates of
    :func:`torch.nn.functional.grid_sample`. Maps of batch 1 serve every image. With pixel maps, pass
    ``align_corners=True``: the default resamples even an identity map. For a warp given by a matrix, use
    :func:`warp_affine` or :func:`warp_perspective`.

    .. image:: _static/img/remap.png

    .. math::
        \text{dst}(x, y) = \text{src}(map_x(x, y), map_y(x, y))

    Convention:
        - align_corners: ``None`` by default, resolved to ``False`` internally
        - pixel maps are normalized with the ``align_corners=True`` convention whatever flag
          reaches ``grid_sample``, so at ``False``/``None`` even an identity map resamples the image;
          pass ``align_corners=True`` (`#4504 <https://github.com/kornia/kornia/issues/4504>`_)
        - the output spatial size comes from the maps; a zero map axis returns an
          autograd-connected empty output, including when the matching source axis is empty

    Args:
        image: the torch.Tensor to remap with shape ``(B, C, H, W)``.
          Where C is the number of channels.
        map_x: the source x coordinate (column) to sample for each output pixel, with shape ``(B, h, w)`` or
          ``(1, h, w)``; in pixels unless ``normalized_coordinates=True``.
        map_y: the source y coordinate (row) to sample for each output pixel, with the shape of ``map_x``.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.
        normalized_coordinates: whether the input coordinates are
           normalized in the range of [-1, 1].

    Returns:
        the sampled images ``(B, C, h, w)``, with the spatial size of the maps.

    See Also:
        - :func:`warp_perspective`: warp by a ``(B, 3, 3)`` pixel homography instead of maps.
        - :func:`warp_grid`: the positions a homography gives, as a grid whose two channels are maps.
        - :func:`~kornia.geometry.grid.create_meshgrid`: the identity maps, to offset or transform.
        - :func:`warp_image_tps`: a smooth warp from thin-plate-spline control points.

    Example:
        >>> import torch
        >>> from kornia.geometry import create_meshgrid
        >>> grid = create_meshgrid(2, 2, False)  # 1x2x2x2
        >>> grid += 1  # apply offset in both directions
        >>> input = torch.ones(1, 1, 2, 2)
        >>> remap(input, grid[..., 0], grid[..., 1], align_corners=True)   # 1x1x2x2
        tensor([[[[1., 0.],
                  [0., 0.]]]])

    """
    KORNIA_CHECK_SHAPE(image, ["B", "C", "H", "W"])
    KORNIA_CHECK_SHAPE(map_x, ["B", "H", "W"])
    KORNIA_CHECK_SHAPE(map_y, ["B", "H", "W"])

    batch_size, _, height, width = image.shape

    # grid_sample need the grid between -1/1
    map_xy: torch.Tensor = torch.stack([map_x, map_y], -1)

    # Default to False if align_corners is None to avoid PyTorch warning
    if align_corners is None:
        align_corners = False

    if map_xy.shape[-3] == 0 or map_xy.shape[-2] == 0:
        return _empty_warp_output_2d(
            image,
            map_xy,
            (map_xy.shape[-3], map_xy.shape[-2]),
            mode,
            padding_mode,
            align_corners,
            _remap_grid_dtype(map_xy, normalized_coordinates),
            transform_batch_broadcast="src",
            allow_fill=False,
            operand="map",
        )
    out_dtype = image.dtype
    if _runs_in_float32(image, map_xy):
        image, map_xy = image.float(), map_xy.float()

    # F.normalize coordinates if not already normalized
    if not normalized_coordinates:
        map_xy = normalize_pixel_coordinates(map_xy, height, width)

    # simulate broadcasting since grid_sample does not support it
    map_xy = map_xy.expand(batch_size, -1, -1, -1)

    # warp the image tensor and return
    return _torch_grid_sample(image, map_xy, mode=mode, padding_mode=padding_mode, align_corners=align_corners).to(
        out_dtype
    )


def invert_affine_transform(matrix: torch.Tensor) -> torch.Tensor:
    r"""Invert 2D affine matrices of shape ``(B, 2, 3)``.

    ``M_inv = invert_affine_transform(M)`` returns, for each ``(B, 2, 3)`` affine matrix, the ``(B, 2, 3)`` inverse of
    its 3x3 extension with the last row ``(0, 0, 1)``, in the dtype of ``M``. It is pure matrix inversion, so the
    result keeps the input's coordinates, pixel or normalized: the inverse of a source→destination matrix maps
    destination to source, and warps a :func:`warp_affine` result back. For ``(B, 3, 3)`` homographies, use
    :func:`torch.linalg.inv`.

    The input is the matrix:

    .. math::
        \begin{bmatrix}
            a_{11} & a_{12} & b_{1} \\
            a_{21} & a_{22} & b_{2} \\
        \end{bmatrix}

    Convention:
        - ``matrix`` is a ``(B, 2, 3)`` affine transform in any coordinate convention
          (pixel or normalized) — pure matrix inversion; the result stays in the input's convention
        - in eager mode a singular matrix raises ``torch.linalg.LinAlgError``

    Args:
        matrix: original affine transform. The torch.Tensor must be
          in the shape of ``(B, 2, 3)``.

    Return:
        the reverse affine transform with shape ``(B, 2, 3)``.

    See Also:
        - :func:`warp_affine`: warp an image by a matrix or by its inverse.
        - :func:`get_affine_matrix2d`: a ``(B, 3, 3)`` affine matrix from translation, rotation, scale and shear;
          pass its ``M[:, :2]``.
        - :func:`~kornia.geometry.conversions.convert_affinematrix_to_homography`: the ``(B, 3, 3)`` extension, to
          compose with homographies.

    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Input matrix type is not a torch.Tensor. Got {type(matrix)}")

    if not (len(matrix.shape) == 3 and matrix.shape[-2:] == (2, 3)):
        raise ValueError(f"Input matrix must be a Bx2x3 torch.Tensor. Got {matrix.shape}")

    matrix_tmp: torch.Tensor = convert_affinematrix_to_homography(matrix)
    matrix_inv: torch.Tensor = _torch_inverse_cast(matrix_tmp)

    return matrix_inv[..., :2, :3]


def get_affine_matrix2d(
    translations: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    angle: torch.Tensor,
    sx: Optional[torch.Tensor] = None,
    sy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compose a 2D affine matrix from a translation, clockwise rotation, scale and shear about a centre.

    ``M = get_affine_matrix2d(translations, center, scale, angle, sx, sy)`` takes ``(B, 2)`` translations
    ``(dx, dy)``, centres ``(x, y)`` and scale factors for ``x`` and ``y``, ``(B,)`` angles in degrees and optional
    ``(B,)`` shear angles ``sx`` and ``sy`` in radians, and returns the ``(B, 3, 3)`` source→destination pixel matrix
    that shears, scales, rotates and then translates, the first three about ``center``. A positive ``angle`` turns
    **clockwise** as displayed, unlike :func:`get_rotation_matrix2d`, :func:`rotate` and :class:`Affine`
    (`#4408 <https://github.com/kornia/kornia/issues/4408>`_). Pass ``M[:, :2]`` to :func:`warp_affine` or ``M`` to
    :func:`warp_perspective`.

    Convention:
        - ``center`` is ``(x, y)`` in pixels, origin at top-left
        - positive ``angle`` rotates **clockwise** as displayed — this function negates ``angle``
          before delegating to :func:`get_rotation_matrix2d`, whose convention is CCW-positive
        - ``sx`` and ``sy`` are angles in radians, using :func:`get_shear_matrix2d`'s
          negative-tangent, x-then-y shear about ``center``, applied before rotation, scale and translation
        - returns ``(B, 3, 3)`` affine matrix in pixel coordinates

    Args:
        translations: torch.Tensor containing the translation vector with shape ``(B, 2)``.
        center: torch.Tensor containing the center vector with shape ``(B, 2)``.
        scale: torch.Tensor containing the scale factor with shape ``(B, 2)``.
        angle: torch.Tensor of angles in degrees ``(B)``.
        sx: torch.Tensor containing the shear angle in radians in the x-direction with shape ``(B)``.
        sy: torch.Tensor containing the shear angle in radians in the y-direction with shape ``(B)``.

    Returns:
        the affine transformation matrix ``(B, 3, 3)``.

    See Also:
        - :func:`get_rotation_matrix2d`: rotation and scale only, ``(B, 2, 3)``, counter-clockwise for a positive
          angle.
        - :func:`get_affine_matrix3d`: the 3D composition, ``(B, 4, 4)``, whose positive ``z`` angle turns
          counter-clockwise on screen.
        - :class:`Affine`: warp an image by these parameters in one module, counter-clockwise for a positive angle.
        - :func:`get_shear_matrix2d`: the shear part alone.
        - :func:`~kornia.geometry.solvers.find_affine`: an affine matrix fitted to point correspondences instead of
          built from parameters.

    """
    transform: torch.Tensor = get_rotation_matrix2d(center, -angle, scale)
    transform[..., 2] += translations  # tx/ty

    # F.pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography(transform)

    if any(s is not None for s in [sx, sy]):
        shear_mat = get_shear_matrix2d(center, sx, sy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_translation_matrix2d(translations: torch.Tensor) -> torch.Tensor:
    """Compute the 3x3 matrix of a 2D translation in pixels.

    ``M = get_translation_matrix2d(translations)`` takes ``(B, 2)`` shifts ``(dx, dy)`` in pixels and returns the
    ``(B, 3, 3)`` matrix that moves every point by them, a positive ``dx`` to the right and a positive ``dy`` down.
    Pass ``M[:, :2]`` to :func:`warp_affine` or ``M`` to :func:`warp_perspective`; to shift an image directly, use
    :func:`translate`.

    Convention:
        - ``translations`` is ``(dx, dy)`` in pixels; returns ``(B, 3, 3)`` affine matrix in pixel coordinates

    Args:
        translations: torch.Tensor containing the translation vector with shape ``(B, 2)``.

    Returns:
        the affine transformation matrix ``(B, 3, 3)``.

    See Also:
        - :func:`translate`: shift an image without building the matrix.
        - :func:`get_affine_matrix2d`: a translation composed with a rotation, a scale and a shear.

    """
    transform: torch.Tensor = eye_like(3, translations)[:, :2, :]
    transform[..., 2] += translations  # tx/ty

    # F.pad transform to get Bx3x3
    return convert_affinematrix_to_homography(transform)


def get_shear_matrix2d(
    center: torch.Tensor, sx: Optional[torch.Tensor] = None, sy: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Compute the 3x3 matrix of a 2D shear given as two angles in radians about a centre.

    ``M = get_shear_matrix2d(center, sx, sy)`` takes ``(B, 2)`` centres ``(x, y)`` in pixels and optional ``(B,)``
    shear angles ``sx`` and ``sy`` in radians (zero when omitted), and returns the ``(B, 3, 3)`` source→destination
    pixel matrix that shears along ``x`` by ``-tan(sx)`` and then along ``y`` by ``-tan(sy)``, both about ``center``.
    :class:`Affine` and :func:`get_affine_matrix2d` read shear angles this way; :func:`shear` and :class:`Shear` take
    raw factors about pixel ``(0, 0)`` instead, which shear the other way for the same values below ``pi / 2`` in
    magnitude (`#5661 <https://github.com/kornia/kornia/issues/5661>`_).

    .. math::
        \begin{bmatrix}
            1 & b \\
            a & ab + 1 \\
        \end{bmatrix}

    Here :math:`a = -\tan(sy)` and :math:`b = -\tan(sx)`.

    Convention:
        - ``center`` is ``(x, y)`` in pixels, origin at top-left
        - ``sx`` and ``sy`` are angles in radians; about ``center = (cx, cy)``, the forward mapping is
          ``x_out = x - tan(sx) * (y - cy)``, then ``y_out = y - tan(sy) * (x_out - cx)``.
          Pixel x increases rightward and y downward. This differs from :func:`shear`'s raw factors.
        - returns ``(B, 3, 3)`` affine matrix in pixel coordinates

    Args:
        center: shearing center coordinates of (x, y).
        sx: shearing angle along x axis in radians.
        sy: shearing angle along y axis in radians

    Returns:
        the shear matrix with shape ``(B, 3, 3)``, to be passed to the affine transformation.

    See Also:
        - :func:`get_shear_matrix3d`: the 3D shear from up to six angles, ``(B, 4, 4)``.
        - :func:`shear`: shear an image by raw factors about pixel ``(0, 0)``, which shear the other way below
          ``pi / 2``.
        - :func:`get_affine_matrix2d`: this shear composed with a rotation, a scale and a translation.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> sx = torch.randn(1)
        >>> sx
        tensor([1.5410])
        >>> center = torch.tensor([[0., 0.]])  # Bx2
        >>> get_shear_matrix2d(center, sx=sx)
        tensor([[[  1.0000, -33.5468,   0.0000],
                 [ -0.0000,   1.0000,   0.0000],
                 [  0.0000,   0.0000,   1.0000]]])

    """
    sx = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if sx is None else sx
    sy = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if sy is None else sy

    x, y = torch.split(center, 1, dim=-1)
    x, y = x.view(-1), y.view(-1)

    sx_tan = torch.tan(sx)
    sy_tan = torch.tan(sy)
    ones_tensor = torch.ones_like(sx)
    shear_mat = torch.stack(
        [ones_tensor, -sx_tan, sx_tan * y, -sy_tan, ones_tensor + sx_tan * sy_tan, sy_tan * (x - sx_tan * y)], dim=-1
    ).view(-1, 2, 3)

    return convert_affinematrix_to_homography(shear_mat)


def get_affine_matrix3d(
    translations: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    angles: torch.Tensor,
    sxy: Optional[torch.Tensor] = None,
    sxz: Optional[torch.Tensor] = None,
    syx: Optional[torch.Tensor] = None,
    syz: Optional[torch.Tensor] = None,
    szx: Optional[torch.Tensor] = None,
    szy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compose a 3D affine matrix from a translation, axis-angle rotation, scale and shears about a centre.

    ``M = get_affine_matrix3d(translations, center, scale, angles)`` takes ``(B, 3)`` translations ``(dx, dy, dz)``,
    centres ``(x, y, z)`` in pixels, per-axis scale factors and axis-angle vectors in degrees, plus up to six optional
    ``(B,)`` shear angles in radians, and returns the ``(B, 4, 4)`` source→destination pixel matrix that shears, scales,
    rotates and then translates, the first three about ``center``. The angles are negated before
    :func:`get_projective_transform`, so they turn the other way from that function and :func:`rotate3d`: a positive
    rotation about ``z`` turns counter-clockwise on screen, as :func:`rotate` does and :func:`get_affine_matrix2d`
    does not (`#4408 <https://github.com/kornia/kornia/issues/4408>`_). Pass ``M[:, :3]`` to :func:`warp_affine3d`
    or ``M`` to :func:`warp_perspective3d`.

    Convention:
        - ``center`` is ``(x, y, z)`` in pixels, origin at the top-left of the first depth
          slice (``z = 0``)
        - ``angles`` are negated before delegating to :func:`get_projective_transform`, whose own
          rotation convention follows the right-hand rule (see
          :func:`kornia.geometry.conversions.axis_angle_to_rotation_matrix`)
        - returns ``(B, 4, 4)`` affine matrix in pixel coordinates

    Args:
        translations: torch.Tensor containing the translation vector (dx,dy,dz) with shape ``(B, 3)``.
        center: torch.Tensor containing the center vector (x,y,z) with shape ``(B, 3)``.
        scale: per-axis scale factors ``(sx, sy, sz)`` with shape ``(B, 3)``; a ``(B, 1)`` or ``(1,)`` tensor scales
            the three axes alike. A ``(B,)`` tensor is not one factor per matrix: for ``B = 3`` it is read as
            ``(sx, sy, sz)``, and for another ``B > 1`` it raises
            (`#6398 <https://github.com/kornia/kornia/issues/6398>`_).
        angles: axis angle vector containing the rotation angles in degrees in the form
            of (rx, ry, rz) with shape ``(B, 3)``. Internally it calls Rodrigues to compute
            the rotation matrix from axis-angle.
        sxy: the shear angle ``sxy`` of :func:`get_shear_matrix3d`, in radians, with shape ``(B)``.
        sxz: the shear angle ``sxz`` of :func:`get_shear_matrix3d`, in radians, with shape ``(B)``.
        syx: the shear angle ``syx`` of :func:`get_shear_matrix3d`, in radians, with shape ``(B)``.
        syz: the shear angle ``syz`` of :func:`get_shear_matrix3d`, in radians, with shape ``(B)``.
        szx: the shear angle ``szx`` of :func:`get_shear_matrix3d`, in radians, with shape ``(B)``.
        szy: the shear angle ``szy`` of :func:`get_shear_matrix3d`, in radians, with shape ``(B)``.

    Returns:
        the 3d affine transformation matrix ``(B, 4, 4)``.

    See Also:
        - :func:`get_affine_matrix2d`: the 2D composition, whose positive angle turns clockwise on screen.
        - :func:`get_projective_transform`: rotation and scale only, ``(B, 3, 4)``, with the angles as given.
        - :func:`get_shear_matrix3d`: the shear part alone.
        - :func:`warp_affine3d`: warp a volume by ``M[:, :3]``.

    """
    transform: torch.Tensor = get_projective_transform(center, -angles, scale)
    transform[..., 3] += translations  # tx/ty/tz

    # F.pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography3d(transform)
    if any(s is not None for s in [sxy, sxz, syx, syz, szx, szy]):
        shear_mat = get_shear_matrix3d(center, sxy, sxz, syx, syz, szx, szy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_shear_matrix3d(
    center: torch.Tensor,
    sxy: Optional[torch.Tensor] = None,
    sxz: Optional[torch.Tensor] = None,
    syx: Optional[torch.Tensor] = None,
    syz: Optional[torch.Tensor] = None,
    szx: Optional[torch.Tensor] = None,
    szy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute the 4x4 matrix of a 3D shear given as up to six angles in radians about a centre.

    ``M = get_shear_matrix3d(center, sxy, sxz, syx, syz, szx, szy)`` takes ``(B, 3)`` centres ``(x, y, z)`` in pixels
    and optional ``(B,)`` shear angles in radians (zero when omitted), and returns the ``(B, 4, 4)`` source→destination
    pixel matrix below, whose translation keeps ``center`` fixed. With :math:`S_{ab} = -\tan(s_{ab})`, the angle ``sab``
    on its own adds :math:`S_{ab}` times coordinate ``a`` to coordinate ``b``. ``sxy`` and ``sxz`` form one shear,
    ``syx`` and ``syz`` a second and ``szx`` and ``szy`` a third, and the linear part is the product of the three in
    that order, so a point is sheared by ``szx`` and ``szy`` first and by ``sxy`` and ``sxz`` last.

    .. math::
        \begin{bmatrix}
            1 & o & r & x - (x + oy + rz) \\
            m & p & s & y - (mx + py + sz) \\
            n & q & t & z - (nx + qy + tz) \\
            0 & 0 & 0 & 1  \\
        \end{bmatrix}
        Where:
        m = S_{xy}
        n = S_{xz}
        o = S_{yx}
        p = S_{xy}S_{yx} + 1
        q = S_{xz}S_{yx} + S_{yz}
        r = S_{zx} + S_{yx}S_{zy}
        s = S_{xy}S_{zx} + (S_{xy}S_{yx} + 1)S_{zy}
        t = S_{xz}S_{zx} + (S_{xz}S_{yx} + S_{yz})S_{zy} + 1

    Here :math:`S_{ab} = -\tan(s_{ab})`, and :math:`(x, y, z)` is ``center``.

    Convention:
        - ``center`` is ``(x, y, z)`` in pixels, origin at the top-left of the first depth
          slice (``z = 0``); it is a fixed point of the returned matrix for every combination of shears
        - returns ``(B, 4, 4)`` affine matrix in pixel coordinates

    Args:
        center: shearing center coordinates of (x, y, z), with shape ``(B, 3)``.
        sxy: shearing angle in radians that adds :math:`S_{xy} x` to ``y``.
        sxz: shearing angle in radians that adds :math:`S_{xz} x` to ``z``.
        syx: shearing angle in radians that adds :math:`S_{yx} y` to ``x``.
        syz: shearing angle in radians that adds :math:`S_{yz} y` to ``z``.
        szx: shearing angle in radians that adds :math:`S_{zx} z` to ``x``.
        szy: shearing angle in radians that adds :math:`S_{zy} z` to ``y``.

    Returns:
        the shear matrix with shape ``(B, 4, 4)``, to be passed to the affine transformation.

    See Also:
        - :func:`get_shear_matrix2d`: the 2D shear from two angles, ``(B, 3, 3)``.
        - :func:`get_affine_matrix3d`: this shear composed with a rotation, a scale and a translation.
        - :func:`warp_perspective3d`: warp a volume by the returned matrix.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> sxy, sxz, syx, syz = torch.randn(4, 1)
        >>> sxy, sxz, syx, syz
        (tensor([1.5410]), tensor([-0.2934]), tensor([-2.1788]), tensor([0.5684]))
        >>> center = torch.tensor([[0., 0., 0.]])  # Bx3
        >>> get_shear_matrix3d(center, sxy=sxy, sxz=sxz, syx=syx, syz=syz)
        tensor([[[  1.0000,  -1.4369,   0.0000,   0.0000],
                 [-33.5468,  49.2039,   0.0000,   0.0000],
                 [  0.3022,  -1.0729,   1.0000,   0.0000],
                 [  0.0000,   0.0000,   0.0000,   1.0000]]])

    """
    sxy = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if sxy is None else sxy
    sxz = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if sxz is None else sxz
    syx = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if syx is None else syx
    syz = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if syz is None else syz
    szx = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if szx is None else szx
    szy = torch.zeros(center.size(0), device=center.device, dtype=center.dtype) if szy is None else szy

    x, y, z = torch.split(center, 1, dim=-1)
    x, y, z = x.view(-1), y.view(-1), z.view(-1)
    # Prepare parameters
    sxy_tan = torch.tan(sxy)
    sxz_tan = torch.tan(sxz)
    syx_tan = torch.tan(syx)
    syz_tan = torch.tan(syz)
    szx_tan = torch.tan(szx)
    szy_tan = torch.tan(szy)

    # shear matrix is implemented with negative values
    sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan = -sxy_tan, -sxz_tan, -syx_tan, -syz_tan, -szx_tan, -szy_tan
    m00, m10, m20, m01, m11, m21, m02, m12, m22 = _compute_shear_matrix_3d(
        sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan
    )

    # translation t = c - A c from the returned linear part A, so that the center is a fixed point
    m03 = x - (m00 * x + m01 * y + m02 * z)
    m13 = y - (m10 * x + m11 * y + m12 * z)
    m23 = z - (m20 * x + m21 * y + m22 * z)

    shear_mat = torch.stack([m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23], -1).view(-1, 3, 4)
    return convert_affinematrix_to_homography3d(shear_mat)


def _compute_shear_matrix_3d(
    sxy_tan: torch.Tensor,
    sxz_tan: torch.Tensor,
    syx_tan: torch.Tensor,
    syz_tan: torch.Tensor,
    szx_tan: torch.Tensor,
    szy_tan: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ones_tensor = torch.ones_like(sxy_tan)

    m00, m10, m20 = ones_tensor, sxy_tan, sxz_tan
    m01, m11, m21 = syx_tan, sxy_tan * syx_tan + ones_tensor, sxz_tan * syx_tan + syz_tan
    m02 = syx_tan * szy_tan + szx_tan
    m12 = sxy_tan * szx_tan + szy_tan * m11
    m22 = sxz_tan * szx_tan + szy_tan * m21 + ones_tensor
    return m00, m10, m20, m01, m11, m21, m02, m12, m22


def warp_affine3d(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: tuple[int, int, int],
    flags: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp a volume by a source→destination 3x4 affine matrix in voxel coordinates.

    ``out = warp_affine3d(src, M, dsize)`` takes volumes ``(B, C, D, H, W)`` and one ``(B, 3, 4)`` affine matrix per
    volume that maps **source** voxels ``(x, y, z)`` to **destination** voxels, and returns the ``(B, C, d, h, w)``
    destination volumes for ``dsize = (d, h, w)``; ``flags`` is the interpolation mode. Keep the default
    ``align_corners=True``: at ``False`` the matrix is normalized for corner-aligned voxels while ``grid_sample`` reads
    half-voxel ones, so a translation of ``t`` voxels moves the volume ``t * size / (size - 1)`` voxels along that
    axis (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). For a ``(B, 4, 4)`` homography, use
    :func:`warp_perspective3d`; for images, :func:`warp_affine`; to keep the input size, :func:`affine3d`.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Convention:
        - align_corners: ``True`` by default
        - padding_mode: ``'zeros'`` by default
        - ``M`` has the batch of ``src``: one matrix is not shared by a larger batch
        - an empty batch returns shape ``(0, C, d, h, w)`` with gradients connected to both inputs
          even when a source spatial dimension is zero
        - a zero output dimension returns an autograd-connected empty tensor;
          negative output dimensions raise ``ValueError``
        - a nonempty batch requires positive source spatial dimensions for a nonempty output

    Args:
        src : input torch.Tensor of shape ``(B, C, D, H, W)``.
        M: affine transformation matrix of shape ``(B, 3, 4)``.
        dsize: size of the output image (depth, height, width).
        flags: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners : mode for grid_generation.

    Returns:
        torch.Tensor: the warped 3d torch.tensor with shape ``(B, C, d, h, w)``, spatial sizes
        given by ``dsize``.

    See Also:
        - :func:`warp_affine`: the 2D warp of images by ``(B, 2, 3)`` matrices.
        - :func:`warp_perspective3d`: a ``(B, 4, 4)`` homography; its default ``align_corners=False`` resamples even an
          identity.
        - :func:`affine3d`: keeps the input size and accepts an unbatched ``(C, D, H, W)`` volume.
        - :func:`get_projective_transform`: the ``(B, 3, 4)`` matrix of a rotation and a scale about a centre.
        - :func:`crop_by_transform_mat3d`: calls this warp with the top three rows of a ``(B, 3, 4)`` or ``(B, 4, 4)``
          matrix, and broadcasts a batch-1 matrix.

    """
    if len(src.shape) != 5:
        raise AssertionError(src.shape)
    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 4)):
        raise AssertionError(M.shape)
    if len(dsize) != 3:
        raise AssertionError(dsize)
    B, C, D, H, W = src.size()
    if dsize[0] < 0 or dsize[1] < 0 or dsize[2] < 0:
        raise ValueError(f"Output size must be non-negative. Got {dsize}.")
    # affine_grid rejects a zero batch: an empty batch takes the empty-output path, which keeps the
    # sampling-option, dtype, device and batch checks and the autograd links to both inputs.
    if B == 0 or dsize[0] == 0 or dsize[1] == 0 or dsize[2] == 0:
        return _empty_warp_output_3d(src, M, dsize, flags, padding_mode, align_corners)
    out_dtype = src.dtype
    if _runs_in_float32(src, M):
        src, M = src.float(), M.float()

    size_src: tuple[int, int, int] = (D, H, W)
    size_out: tuple[int, int, int] = dsize

    M_4x4 = convert_affinematrix_to_homography3d(M)  # Bx4x4

    # we need to F.normalize the transformation since grid sample needs -1/1 coordinates
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography3d(M_4x4, size_src, size_out)  # Bx4x4

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
    P_norm: torch.Tensor = src_norm_trans_dst_norm[:, :3]  # Bx3x4

    # compute meshgrid and apply to input
    dsize_out: list[int] = [B, C, *list(size_out)]
    grid = F.affine_grid(P_norm, dsize_out, align_corners=align_corners)
    return _torch_grid_sample(src, grid, align_corners=align_corners, mode=flags, padding_mode=padding_mode).to(
        out_dtype
    )


def projection_from_Rt(rmat: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
    """Stack rotations and translations into ``[R | t]`` matrices of shape ``(*, 3, 4)``.

    ``P = projection_from_Rt(rmat, tvec)`` concatenates rotations ``(*, 3, 3)`` and translations ``(*, 3, 1)`` with the
    same leading dimensions into the ``(*, 3, 4)`` matrices :math:`P = [R | t]`. It applies no intrinsics: for a camera
    projection matrix :math:`K [R | t]`, use :func:`~kornia.geometry.epipolar.projection_from_KRt`, and for a 4x4
    pose, :func:`~kornia.geometry.conversions.Rt_to_matrix4x4`.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Convention:
        - returns the concatenation :math:`[R | t]` with shape ``(*, 3, 4)``

    Args:
       rmat: the rotation matrix with shape ``(*, 3, 3)``.
       tvec: the translation vector with shape ``(*, 3, 1)``.

    Returns:
       the projection matrix with shape ``(*, 3, 4)``.

    See Also:
        - :func:`~kornia.geometry.epipolar.projection_from_KRt`: also multiplies by the camera intrinsics ``K``.
        - :func:`~kornia.geometry.conversions.Rt_to_matrix4x4`: the homogeneous ``(B, 4, 4)`` pose instead.

    """
    if not (len(rmat.shape) >= 2 and rmat.shape[-2:] == (3, 3)):
        raise AssertionError(rmat.shape)
    if not (len(tvec.shape) >= 2 and tvec.shape[-2:] == (3, 1)):
        raise AssertionError(tvec.shape)

    return torch.cat([rmat, tvec], -1)  # Bx3x4


def get_projective_transform(center: torch.Tensor, angles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Compute the 3x4 affine matrix of a 3D scaling and axis-angle rotation about a centre.

    ``P = get_projective_transform(center, angles, scales)`` takes ``(B, 3)`` centres ``(x, y, z)`` in pixels,
    axis-angle vectors ``(rx, ry, rz)`` in degrees and per-axis scale factors, and returns the ``(B, 3, 4)``
    source→destination pixel matrix that scales and then rotates about ``center``, as :func:`warp_affine3d` takes. The
    rotation is Rodrigues' formula on ``angles`` converted to radians and follows the right-hand rule, so a
    positive rotation about ``z`` turns clockwise on screen, the opposite of :func:`get_rotation_matrix2d`. The matrix
    is affine, not projective; to add a translation and shears, use :func:`get_affine_matrix3d`, whose angles turn
    the other way (`#4408 <https://github.com/kornia/kornia/issues/4408>`_).

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Convention:
        - ``center`` is ``(x, y, z)`` in pixels, origin at the top-left of the first depth
          slice (``z = 0``)
        - rotation follows the right-hand rule (see
          :func:`kornia.geometry.conversions.axis_angle_to_rotation_matrix`); a positive rotation
          about +z is **clockwise on screen** (y-down image axes) — opposite of :func:`get_rotation_matrix2d`
        - returns the projection matrix ``(B, 3, 4)`` in pixel coordinates

    Args:
        center: center of the rotation (x,y,z) in the source with shape ``(B, 3)``.
        angles: axis angle vector containing the rotation angles in degrees in the form
            of (rx, ry, rz) with shape ``(B, 3)``. Internally it calls Rodrigues to compute
            the rotation matrix from axis-angle.
        scales: scale factor for x-y-z-directions with shape ``(B, 3)``.

    Returns:
        the projection matrix of 3D rotation with shape ``(B, 3, 4)``.

    See Also:
        - :func:`get_rotation_matrix2d`: the 2D rotation and scale, counter-clockwise on screen.
        - :func:`get_affine_matrix3d`: adds a translation and shears, and negates the angles.
        - :func:`rotate3d`: rotate a volume about its centre by such an axis-angle vector.
        - :func:`~kornia.geometry.conversions.axis_angle_to_rotation_matrix`: the 3x3 rotation of an axis-angle vector
          in radians.

    """
    if not (len(center.shape) == 2 and center.shape[-1] == 3):
        raise AssertionError(center.shape)
    if not (len(angles.shape) == 2 and angles.shape[-1] == 3):
        raise AssertionError(angles.shape)
    if center.device != angles.device:
        raise AssertionError(center.device, angles.device)
    if center.dtype != angles.dtype:
        raise AssertionError(center.dtype, angles.dtype)

    # create rotation matrix
    axis_angle_rad: torch.Tensor = torch.torch.deg2rad(angles)
    rmat: torch.Tensor = axis_angle_to_rotation_matrix(axis_angle_rad)  # Bx3x3
    scaling_matrix: torch.Tensor = eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = eye_like(4, rmat, shared_memory=False)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    # An identity plus a translation column is unit triangular, so it is always invertible.
    to_origin_mat = _torch_inverse_cast(from_origin_mat, check_errors=False)

    # append translation with torch.zeros
    proj_mat = projection_from_Rt(rmat, torch.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = convert_affinematrix_to_homography3d(proj_mat)  # Bx4x4
    proj_mat = from_origin_mat @ proj_mat @ to_origin_mat

    return proj_mat[..., :3, :]  # Bx3x4


def get_perspective_transform3d(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    r"""Compute the 3D homography that maps five source box corners exactly onto destination ones.

    ``M = get_perspective_transform3d(src, dst)`` takes two ``(B, 8, 3)`` sets of ``(x, y, z)`` box corners in
    :func:`~kornia.geometry.bbox.bbox_generator3d` order and returns the ``(B, 4, 4)`` homography, with
    ``M[3, 3] = 1``, that maps corners 0, 1, 2, 5 and 7 of ``src`` onto those of ``dst``: the source→destination voxel
    homography that :func:`warp_perspective3d` takes. The other three corners map onto theirs only when all eight pairs
    are related by one homography. For 2D points, use :func:`get_perspective_transform`.

    The function calculates the matrix of a perspective transform so that:

    .. math::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i}z_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        z_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math::
        dst(i) = (x_{i}^{'},y_{i}^{'},z_{i}^{'}), src(i) = (x_{i}, y_{i}, z_{i}), i = 0,1,2,5,7

    Concrete math is as below:

    .. math::

        \[ u_i =\frac{c_{00} * x_i + c_{01} * y_i + c_{02} * z_i + c_{03}}
            {c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]
        \[ v_i =\frac{c_{10} * x_i + c_{11} * y_i + c_{12} * z_i + c_{13}}
            {c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]
        \[ w_i =\frac{c_{20} * x_i + c_{21} * y_i + c_{22} * z_i + c_{23}}
            {c_{30} * x_i + c_{31} * y_i + c_{32} * z_i + c_{33}} \]

    .. math::

        \begin{pmatrix}
        x_0 & y_0 & z_0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_0*u_0 & -y_0*u_0 & -z_0 * u_0 \\
        x_1 & y_1 & z_1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_1*u_1 & -y_1*u_1 & -z_1 * u_1 \\
        x_2 & y_2 & z_2 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_2*u_2 & -y_2*u_2 & -z_2 * u_2 \\
        x_5 & y_5 & z_5 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_5*u_5 & -y_5*u_5 & -z_5 * u_5 \\
        x_7 & y_7 & z_7 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -x_7*u_7 & -y_7*u_7 & -z_7 * u_7 \\
        0 & 0 & 0 & 0 & x_0 & y_0 & z_0 & 1 & 0 & 0 & 0 & 0 & -x_0*v_0 & -y_0*v_0 & -z_0 * v_0 \\
        0 & 0 & 0 & 0 & x_1 & y_1 & z_1 & 1 & 0 & 0 & 0 & 0 & -x_1*v_1 & -y_1*v_1 & -z_1 * v_1 \\
        0 & 0 & 0 & 0 & x_2 & y_2 & z_2 & 1 & 0 & 0 & 0 & 0 & -x_2*v_2 & -y_2*v_2 & -z_2 * v_2 \\
        0 & 0 & 0 & 0 & x_5 & y_5 & z_5 & 1 & 0 & 0 & 0 & 0 & -x_5*v_5 & -y_5*v_5 & -z_5 * v_5 \\
        0 & 0 & 0 & 0 & x_7 & y_7 & z_7 & 1 & 0 & 0 & 0 & 0 & -x_7*v_7 & -y_7*v_7 & -z_7 * v_7 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_0 & y_0 & z_0 & 1 & -x_0*w_0 & -y_0*w_0 & -z_0 * w_0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_1 & y_1 & z_1 & 1 & -x_1*w_1 & -y_1*w_1 & -z_1 * w_1 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_2 & y_2 & z_2 & 1 & -x_2*w_2 & -y_2*w_2 & -z_2 * w_2 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_5 & y_5 & z_5 & 1 & -x_5*w_5 & -y_5*w_5 & -z_5 * w_5 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_7 & y_7 & z_7 & 1 & -x_7*w_7 & -y_7*w_7 & -z_7 * w_7 \\
        \end{pmatrix}

    Convention:
        - points: ``(x, y, z)``, pixel centers, origin at the top-left of the first depth slice; shape ``(B, 8, 3)``
        - the 15 unknowns are solved exactly from points 0, 1, 2, 5 and 7, the corners ``000``, ``100``, ``110``,
          ``101`` and ``011`` in :func:`~kornia.geometry.bbox.bbox_generator3d` order; points 3, 4 and 6 are not
          used, so they map onto their ``dst`` only when all eight pairs are related by one homography
        - returns the source→destination **pixel** homography ``(B, 4, 4)``

    Args:
        src: coordinates of the box corners in the source volume with shape ``(B, 8, 3)``.
        dst: coordinates of the corresponding box corners in
            the destination volume with shape ``(B, 8, 3)``.

    Returns:
        the perspective transformation with shape ``(B, 4, 4)``.

    See Also:
        - :func:`get_perspective_transform`: the 2D homography from four point pairs.
        - :func:`warp_perspective3d`: warp a volume by the returned homography.
        - :func:`~kornia.geometry.bbox.bbox_generator3d`: box corners in the order this function reads them.

    """
    if not isinstance(src, (torch.Tensor)):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(dst, (torch.Tensor)):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst)}")

    if not src.shape[-2:] == (8, 3):
        raise ValueError(f"Inputs must be a Bx8x3 torch.Tensor. Got {src.shape}")

    if not src.shape == dst.shape:
        raise ValueError(f"Inputs must have the same shape. Got {dst.shape}")

    if not (src.shape[0] == dst.shape[0]):
        raise ValueError(f"Inputs must have same batch size dimension. Expect {src.shape} but got {dst.shape}")

    if not (src.device == dst.device and src.dtype == dst.dtype):
        raise AssertionError(
            f"Expect `src` and `dst` to be in the same device (Got {src.dtype}, {dst.dtype}) "
            f"with the same dtype (Got {src.dtype}, {dst.dtype})."
        )

    # we build the square matrix A from 5 point correspondences, 3 equations each, and solve it exactly
    p = []

    # 000, 100, 110, 101, 011
    for i in [0, 1, 2, 5, 7]:
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], "x"))
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], "y"))
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], "z"))

    # A is Bx15x15
    A = torch.stack(p, 1)

    # b is a Bx15x1
    b = torch.stack(
        [
            dst[:, 0:1, 0],
            dst[:, 0:1, 1],
            dst[:, 0:1, 2],
            dst[:, 1:2, 0],
            dst[:, 1:2, 1],
            dst[:, 1:2, 2],
            dst[:, 2:3, 0],
            dst[:, 2:3, 1],
            dst[:, 2:3, 2],
            # dst[:, 3:4, 0], dst[:, 3:4, 1], dst[:, 3:4, 2],
            # dst[:, 4:5, 0], dst[:, 4:5, 1], dst[:, 4:5, 2],
            dst[:, 5:6, 0],
            dst[:, 5:6, 1],
            dst[:, 5:6, 2],
            # dst[:, 6:7, 0], dst[:, 6:7, 1], dst[:, 6:7, 2],
            dst[:, 7:8, 0],
            dst[:, 7:8, 1],
            dst[:, 7:8, 2],
        ],
        1,
    )

    # solve the system Ax = b
    X: torch.Tensor = _torch_solve_cast(A, b)

    # create variable to return
    batch_size: int = src.shape[0]
    M = torch.empty(batch_size, 16, device=src.device, dtype=src.dtype)
    M[..., :15] = X[..., 0]
    M[..., -1].fill_(1)

    return M.view(-1, 4, 4)  # Bx4x4


def _build_perspective_param3d(p: torch.Tensor, q: torch.Tensor, axis: str) -> torch.Tensor:
    ones_tensor = torch.ones_like(p)[..., 0:1]
    zeros_tensor = torch.zeros_like(p)[..., 0:1]

    if axis == "x":
        return torch.cat(
            [
                p[:, 0:1],
                p[:, 1:2],
                p[:, 2:3],
                ones_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                -p[:, 0:1] * q[:, 0:1],
                -p[:, 1:2] * q[:, 0:1],
                -p[:, 2:3] * q[:, 0:1],
            ],
            1,
        )

    if axis == "y":
        return torch.cat(
            [
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                p[:, 0:1],
                p[:, 1:2],
                p[:, 2:3],
                ones_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                -p[:, 0:1] * q[:, 1:2],
                -p[:, 1:2] * q[:, 1:2],
                -p[:, 2:3] * q[:, 1:2],
            ],
            1,
        )

    if axis == "z":
        return torch.cat(
            [
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                zeros_tensor,
                p[:, 0:1],
                p[:, 1:2],
                p[:, 2:3],
                ones_tensor,
                -p[:, 0:1] * q[:, 2:3],
                -p[:, 1:2] * q[:, 2:3],
                -p[:, 2:3] * q[:, 2:3],
            ],
            1,
        )

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")


def warp_perspective3d(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: tuple[int, int, int],
    flags: str = "bilinear",
    border_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Warp a volume by a source→destination 4x4 homography in voxel coordinates.

    ``out = warp_perspective3d(src, M, dsize)`` takes volumes ``(B, C, D, H, W)`` and one ``(B, 4, 4)`` homography per
    volume that maps **source** voxels ``(x, y, z)`` to **destination** voxels, and returns the ``(B, C, d, h, w)``
    destination volumes for ``dsize = (d, h, w)``; ``flags`` is the interpolation mode and ``border_mode`` the padding
    mode. Pass ``align_corners=True``: at the default ``False`` the sampling grid is corner-aligned while
    ``grid_sample`` reads half-voxel coordinates, so even an identity ``M`` resamples the volume
    (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). For images, use :func:`warp_perspective`; for a
    ``(B, 3, 4)`` affine matrix, :func:`warp_affine3d`.

    Each output voxel samples the source at:

    .. math::
        \text{dst}(x, y, z) = \text{src}\left( \pi\left( M^{-1} \cdot (x, y, z, 1)^{T} \right) \right)

    where :math:`\pi` divides by the fourth (homogeneous) coordinate.

    Convention:
        - ``M`` is the source→destination **pixel** homography ``(B, 4, 4)``, with the batch of ``src``; an
          unbatched ``(4, 4)`` ``M`` is accepted for a single volume
        - align_corners: ``False`` by default (differs from the 2D :func:`warp_perspective`,
          whose default is ``True``)
        - border_mode: ``'zeros'`` by default
        - a zero output dimension returns an autograd-connected empty tensor;
          negative output dimensions raise ``ValueError``

    Args:
        src: input image with shape ``(B, C, D, H, W)``.
        M: transformation matrix with shape ``(B, 4, 4)``.
        dsize: size of the output image (depth, height, width).
        flags: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        border_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        the warped input image ``(B, C, d, h, w)``, spatial sizes given by ``dsize``.

    See Also:
        - :func:`warp_perspective`: the 2D warp of images by ``(B, 3, 3)`` homographies.
        - :func:`warp_affine3d`: a ``(B, 3, 4)`` affine matrix, correct at its default ``align_corners=True``.
        - :func:`homography_warp3d`: a destination→source homography, in normalized coordinates by default.
        - :func:`get_perspective_transform3d`: the homography that maps box corners onto box corners.

    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

    if not len(src.shape) == 5:
        raise ValueError(f"Input src must be a BxCxDxHxW torch.Tensor. Got {src.shape}")

    # An unbatched 4x4 matrix has always been accepted here (the historical shape guard used
    # ``or``, so it never fired for one), and it warps correctly. Keep that working by promoting
    # it, rather than letting the tightened guard below turn it into a ``ValueError``.
    if len(M.shape) == 2 and M.shape[-2:] == (4, 4):
        M = M[None]

    if not (len(M.shape) == 3 and M.shape[-2:] == (4, 4)):
        raise ValueError(f"Input M must be a Bx4x4 torch.Tensor. Got {M.shape}")

    if dsize[0] < 0 or dsize[1] < 0 or dsize[2] < 0:
        raise ValueError(f"Output size must be non-negative. Got {dsize}.")
    if dsize[0] == 0 or dsize[1] == 0 or dsize[2] == 0:
        return _empty_warp_output_3d(src, M, dsize, flags, border_mode, align_corners)
    out_dtype = src.dtype
    if _runs_in_float32(src, M):
        src, M = src.float(), M.float()

    # launches the warper
    d, h, w = src.shape[-3:]
    return _transform_warp_impl3d(src, M, (d, h, w), dsize, flags, border_mode, align_corners).to(out_dtype)


def homography_warp(
    patch_src: torch.Tensor,
    src_homo_dst: torch.Tensor,
    dsize: tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    normalized_coordinates: bool = True,
    normalized_homography: bool = True,
) -> torch.Tensor:
    """Warp an image by a destination→source homography in normalized ``[-1, 1]`` coordinates.

    ``out = homography_warp(patch_src, src_homo_dst, dsize)`` takes images ``(N, C, H, W)`` and one ``(N, 3, 3)``
    homography per image that maps **destination** points to **source** points in **normalized** :math:`[-1, 1]`
    coordinates, and returns the ``(N, C, h, w)`` destination images for ``dsize = (h, w)``. In normalized coordinates
    an identity maps the whole source onto the whole output, so a different ``dsize`` resizes the image. For a
    source→destination homography in **pixel** coordinates, as :func:`get_perspective_transform`,
    :func:`~kornia.geometry.homography.find_homography_dlt` and :class:`~kornia.geometry.ransac.RANSAC` return it, use
    :func:`warp_perspective`. Passed here unchanged, such a homography is read as a normalized destination→source one,
    which typically samples outside the source, so the output is mostly or entirely zeros (with the default
    ``padding_mode="zeros"``).

    ``normalized_coordinates=False`` reads the same destination→source homography in pixels, and
    ``normalized_homography=False`` instead hands a pixel source→destination homography to :func:`warp_perspective`
    with this function's ``align_corners=False`` default, which for a pixel homography samples the same positions as
    ``align_corners=True`` (see ``align_corners`` below), up to which pixel ``mode="nearest"`` takes for a sample
    exactly halfway between two pixels.

    Convention:
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for homography direction, normalized
        coordinates and ``align_corners`` sampling rules.

        - ``src_homo_dst`` is the destination→source homography ``(N, 3, 3)`` when
          ``normalized_homography=True`` (default), in normalized :math:`[-1, 1]` coordinates by default
          (``normalized_coordinates=True``). With ``normalized_coordinates=False`` it is in **pixel**
          coordinates, pixel centres at integers: it maps the pixel ``(x, y)`` of the ``dsize`` output to the
          source pixel it samples, so an identity returns the top-left ``dsize`` window of the input
        - with ``normalized_homography=False`` it is consumed as the source→destination **pixel** homography,
          exactly like :func:`warp_perspective`, and ``normalized_coordinates`` is ignored
        - ``src_homo_dst`` has the batch of ``patch_src``: a ``(1, 3, 3)`` homography is not shared by a larger
          batch
        - with ``normalized_coordinates=False`` the pixel grid is warped and normalized in at least float32, also
          under a half default dtype and in a caller's autocast region, and then cast to the homography's dtype
        - align_corners: ``False`` by default (differs from :func:`warp_perspective`). Along a source axis longer
          than one pixel, a pixel homography samples the same positions under both settings, which then differ
          only in ``padding_mode="reflection"`` and, with ``mode="nearest"``, in which pixel a sample exactly
          halfway between two pixels takes
        - negative output dimensions raise ``ValueError``

    Args:
        patch_src: The image or torch.Tensor to warp. Should be from source of shape ``(N, C, H, W)``.
        src_homo_dst: The homography or torch.stack of homographies of shape ``(N, 3, 3)`` —
            destination to source when ``normalized_homography=True`` (default), source to
            destination (pixel) when ``normalized_homography=False``.
        dsize: output size ``(h, w)``.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
        normalized_coordinates: whether ``src_homo_dst`` is in normalized :math:`[-1, 1]` coordinates
            (``True``) or in pixel coordinates (``False``). Ignored when ``normalized_homography=False``.
        normalized_homography: whether ``src_homo_dst`` is a destination→source homography in the coordinates
            ``normalized_coordinates`` selects (``True``, default) or a pixel source→destination homography
            (``False``).

    Return:
        the warped images ``(N, C, h, w)``, sampled at the source locations of the destination pixels.

    See Also:
        - :func:`warp_perspective`: takes the source→destination homography in pixel coordinates.
        - :class:`HomographyWarper`: the module form, which can precompute the sampling grid for repeated warps.
        - :func:`homography_warp3d`: the same warp for ``(N, C, D, H, W)`` volumes by ``(N, 4, 4)`` homographies.
        - :func:`warp_grid`: the warped sampling grid alone, without sampling the image.
        - :func:`~kornia.geometry.conversions.normalize_homography`: convert a pixel source→destination homography to
          normalized coordinates, with this function's ``align_corners`` (it defaults to ``True`` there); invert the
          result for this function.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = homography_warp(input, homography, (32, 32))

    Example:
        >>> img = torch.rand(1, 4, 5, 6)
        >>> H = torch.eye(3)[None]
        >>> out = homography_warp(img, H, (4, 2), align_corners=True, normalized_homography=False)
        >>> print(out.shape)
        torch.Size([1, 4, 4, 2])

    """
    if dsize[0] < 0 or dsize[1] < 0:
        raise ValueError(f"Output size must be non-negative. Got {dsize}.")
    if not src_homo_dst.device == patch_src.device:
        raise TypeError(
            f"Patch and homography must be on the same device. Got patch.device: {patch_src.device} "
            f"src_H_dst.device: {src_homo_dst.device}."
        )
    if normalized_homography:
        height, width = dsize
        if normalized_coordinates:
            grid = create_meshgrid(
                height,
                width,
                normalized_coordinates=True,
                device=patch_src.device,
                dtype=patch_src.dtype,
                align_corners=align_corners,
            )
            warped_grid = warp_grid(grid, src_homo_dst)
        else:
            # A pixel homography lands on source pixels, which grid_sample reads normalized. The pixel arithmetic
            # runs in at least float32, unaffected by autocast, then the grid returns to the homography's dtype.
            warped_grid = _warp_pixel_grid(src_homo_dst, height, width)
            warped_grid = _normalize_warped_pixel_grid(
                warped_grid, patch_src.shape[-2], patch_src.shape[-1], align_corners
            ).to(src_homo_dst.dtype)

        return _torch_grid_sample(
            patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
    return warp_perspective(
        patch_src, src_homo_dst, dsize, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


def _transform_warp_impl3d(
    src: torch.Tensor,
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: tuple[int, int, int],
    dsize_dst: tuple[int, int, int],
    grid_mode: str,
    padding_mode: str,
    align_corners: bool,
) -> torch.Tensor:
    """Compute the transform in normalized coordinates and perform the warping."""
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography3d(dst_pix_trans_src_pix, dsize_src, dsize_dst)

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
    return homography_warp3d(src, src_norm_trans_dst_norm, dsize_dst, grid_mode, padding_mode, align_corners, True)


def homography_warp3d(
    patch_src: torch.Tensor,
    src_homo_dst: torch.Tensor,
    dsize: tuple[int, int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    normalized_coordinates: bool = True,
) -> torch.Tensor:
    """Warp a volume by a destination→source 3D homography in normalized coordinates.

    ``out = homography_warp3d(patch_src, src_homo_dst, dsize)`` takes volumes ``(N, C, D, H, W)`` and one
    ``(N, 4, 4)`` homography per volume that maps **destination** points ``(x, y, z)`` to **source** points in
    **normalized** :math:`[-1, 1]` coordinates, and returns the ``(N, C, d, h, w)`` destination volumes for
    ``dsize = (d, h, w)``. In normalized coordinates, pass ``align_corners=True``: the normalized grid is always
    corner-aligned, so at the default ``False`` even an identity resamples the volume
    (`#4503 <https://github.com/kornia/kornia/issues/4503>`_). For a source→destination homography in voxel
    coordinates, use :func:`warp_perspective3d` with ``align_corners=True``, which at its default ``False`` resamples
    even an identity for the same reason, or pass the inverse of that homography here with
    ``normalized_coordinates=False``. This function has no ``normalized_homography`` switch.

    Convention:
        - input: ``(N, C, D, H, W)``; ``dsize`` is the output size ``(d, h, w)``
        - ``src_homo_dst`` is the destination→source homography ``(N, 4, 4)``, in normalized
          :math:`[-1, 1]` coordinates by default (``normalized_coordinates=True``), and acts on
          ``(x, y, z, 1)`` column vectors: ``x`` indexes ``W``, ``y`` indexes ``H``, ``z`` indexes
          ``D``. That is the order :func:`torch.nn.functional.grid_sample` reads a 5-D grid in, and
          the order :func:`warp_grid3d` takes. kornia's own 3-D grids and pixel coordinates are
          ``(d, x, y)`` (see :func:`~kornia.geometry.grid.create_meshgrid3d`), so the sampling grid
          is reordered here before the homography is applied; nothing else in the module changes
          its order
        - with ``normalized_coordinates=False`` ``src_homo_dst`` is in **voxel** coordinates, voxel
          centres at integers, in the same ``(x, y, z)`` order: it maps the voxel ``(x, y, z)`` of the
          ``dsize`` output to the source voxel it samples, so an identity returns the ``dsize`` window
          of the input that starts at voxel ``(0, 0, 0)``. The warped voxel grid is normalized with the
          source size under ``align_corners`` before ``grid_sample``
        - with ``normalized_coordinates=False`` the voxel grid is warped and normalized in at least
          float32, also under a half default dtype and in a caller's autocast region, and then cast to
          the homography's dtype
        - align_corners: ``False`` by default. With ``normalized_coordinates=False`` and a source axis
          longer than one voxel, a voxel homography samples the same positions under both settings,
          which then differ only in ``padding_mode="reflection"`` and, with ``mode="nearest"``, in which voxel a
          sample exactly halfway between two voxels takes
        - padding_mode: ``'zeros'`` by default
        - negative output dimensions raise ``ValueError``

    Args:
        patch_src: The image or torch.Tensor to warp. Should be from source of shape ``(N, C, D, H, W)``.
        src_homo_dst: The homography or torch.stack of homographies from destination to source of shape
          ``(N, 4, 4)``.
        dsize: output size ``(d, h, w)``.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
        normalized_coordinates: whether ``src_homo_dst`` is in normalized :math:`[-1, 1]` coordinates
          (``True``) or in voxel coordinates (``False``).

    Return:
        the warped volumes ``(N, C, d, h, w)``, sampled at the source locations of the destination voxels.

    See Also:
        - :func:`homography_warp`: the 2D warp of images, which also has a ``normalized_homography`` switch.
        - :func:`warp_perspective3d`: takes the source→destination homography in voxel coordinates.
        - :func:`warp_grid3d`: the warped sampling grid alone, in ``(x, y, z)`` order.
        - :func:`~kornia.geometry.grid.create_meshgrid3d`: kornia's ``(d, x, y)`` grid, which this function reorders.

    Example:
        >>> input = torch.rand(1, 3, 8, 32, 32)
        >>> homography = torch.eye(4).view(1, 4, 4)
        >>> output = homography_warp3d(input, homography, (8, 32, 32))

    """
    if dsize[0] < 0 or dsize[1] < 0 or dsize[2] < 0:
        raise ValueError(f"Output size must be non-negative. Got {dsize}.")
    if not src_homo_dst.device == patch_src.device:
        raise TypeError(
            f"Patch and homography must be on the same device. Got patch.device: {patch_src.device} "
            f"src_H_dst.device: {src_homo_dst.device}."
        )

    out_dtype = patch_src.dtype
    if _runs_in_float32(patch_src, src_homo_dst):
        patch_src, src_homo_dst = patch_src.float(), src_homo_dst.float()
    depth, height, width = dsize
    if normalized_coordinates:
        grid = create_meshgrid3d(
            depth,
            height,
            width,
            normalized_coordinates=True,
            device=patch_src.device,
            dtype=torch.promote_types(patch_src.dtype, src_homo_dst.dtype),
        )
        # ``create_meshgrid3d`` follows kornia's ``(d, x, y)`` convention, which
        # ``normalize_pixel_coordinates3d`` and ``conv_soft_argmax3d`` rely on. ``warp_grid3d`` and
        # ``grid_sample`` both read the last axis as ``(x, y, z)``, so the depth channel moves last.
        # Stacked rather than indexed with a list, which is copied to the device with a blocking transfer.
        grid = torch.stack((grid[..., 1], grid[..., 2], grid[..., 0]), -1)
        warped_grid = warp_grid3d(grid, src_homo_dst)
    else:
        # A voxel homography lands on source voxels, which grid_sample reads normalized. The voxel arithmetic runs
        # in at least float32, unaffected by autocast, then the grid returns to the homography's dtype.
        warped_grid = _warp_pixel_grid3d(src_homo_dst, depth, height, width)
        warped_grid = _normalize_warped_pixel_grid3d(
            warped_grid, patch_src.shape[-3], patch_src.shape[-2], patch_src.shape[-1], align_corners
        ).to(src_homo_dst.dtype)

    return _torch_grid_sample(
        patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    ).to(out_dtype)
