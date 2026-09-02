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
from torch import nn
from typing_extensions import TypedDict

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.geometry.subpix import (
    AdaptiveQuadInterp3d,
    ConvQuadInterp3d,
    IterativeQuadInterp3d,
    NonMaximaSuppression2d,
    nms3d_minmax,
)
from kornia.geometry.transform import ScalePyramid, pyrdown, resize

from .laf import laf_from_center_scale_ori
from .orientation import PassLAF
from .responses import BlobHessian

# Max |sin| among the 11 boundary points sampled by laf_to_boundary_points(n_pts=12):
#   angles = linspace(0, 2π, n_pts - 1) = linspace(0, 2π, 11) → k * 2π/10 for k=0..10
#   max|sin| at k=2 and k=3: sin(2π/5) ≈ 0.9511;  max|cos| at k=0: cos(0) = 1.0
# Used to inline the boundary check in _process_octave for isotropic LAFs (rotmat=eye(2)),
# avoiding CPU→GPU allocation + bmm every octave.
_MAX_ABS_SIN_12: float = math.sin(2 * 2 * math.pi / 10)  # ≈ 0.9511


def _scale_index_to_scale(max_coords: torch.Tensor, sigmas: torch.Tensor, num_levels: int) -> torch.Tensor:
    r"""Auxiliary function for ScaleSpaceDetector.

    Converts scale level index from the subpix module to the actual
    scale, using the sigmas from the ScalePyramid output.

    Args:
        max_coords: torch.Tensor [BxNx3].
        sigmas: torch.Tensor [BxD], D >= 1
        num_levels: number of levels in the scale index.

    Returns:
        torch.Tensor [BxNx3].

    """
    B = max_coords.shape[0]
    base_sigma = sigmas[:, 0].view(B, 1, 1)  # (B, 1, 1) — per-batch base sigma
    max_coords[:, :, 0:1] = base_sigma * torch.pow(2.0, max_coords[:, :, 0:1] / float(num_levels))
    return max_coords


def _resize_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    r"""Resample a full-resolution mask onto ``ref``'s spatial size and dtype, conservatively.

    A mask says where a detection may be. A boolean or integer mask is binary -- any non-zero
    value keeps a position, so a 0/1 and an OpenCV-style 0/255 mask mean the same thing -- and a
    floating-point mask is used as weights (see :func:`_weight_scores`). The resample is a
    min-pool: a level pixel takes the smallest mask value among the source pixels it covers, so a
    zero region suppresses every level pixel it touches and a thin one cannot fall between the
    samples of an interpolation. The result is cast to ``ref``'s dtype, so the weighting that
    follows cannot promote the response map to the mask's dtype; a weight the image dtype cannot
    hold (below ~6e-8 for a float16 image) rounds to zero and suppresses. Weights above one are
    clamped to one -- a weight never promotes -- so a float 0/255 mask, an OpenCV mask that went
    through ``.astype(np.float32)``, means the same as the integer one.
    """
    if mask.is_floating_point():
        m = mask.to(torch.float64 if mask.dtype == torch.float64 else torch.float32).clamp_max(1.0)
    else:
        m = mask.ne(0).to(torch.float32)
    h, w = ref.shape[-2], ref.shape[-1]
    m = _adaptive_min_pool_1d(m, h, dim=-2)
    m = _adaptive_min_pool_1d(m, w, dim=-1)
    return m.to(ref.dtype)


def _adaptive_min_pool_1d(m: torch.Tensor, out_size: int, dim: int) -> torch.Tensor:
    r"""Min-pool ``m`` along ``dim`` onto ``out_size`` samples with :func:`adaptive_max_pool2d`'s windows.

    Output sample ``i`` covers the source samples ``floor(i * n / out_size) .. ceil((i + 1) * n / out_size) - 1``,
    exactly the window of ``torch.nn.functional.adaptive_max_pool2d``, so the result equals
    ``-adaptive_max_pool2d(-m)`` on CPU. It is spelled as a gather so that every device pools the same windows:
    on MPS, ``adaptive_max_pool2d`` returns the wrong shape when the output is larger than the input and pools
    other windows than CPU when the ratio is not an integer, so the resampled mask, and with it the detections,
    would depend on the device.
    """
    dim = dim % m.dim()
    n = m.shape[dim]
    if n == out_size:
        return m
    # Window bounds are shape arithmetic, so they are Python constants under `torch.compile`.
    starts = [(i * n) // out_size for i in range(out_size)]
    ends = [-((-(i + 1) * n) // out_size) for i in range(out_size)]
    width = max(e - s for s, e in zip(starts, ends))
    # (out_size, width) source indices per output sample; the positions past a window's end are
    # clamped onto its last sample, which is inside the window, so they never change the min.
    idx = [[min(s + j, e - 1) for j in range(width)] for s, e in zip(starts, ends)]
    index = torch.tensor(idx, device=m.device, dtype=torch.long).reshape(-1)
    gathered = m.index_select(dim, index)
    shape = list(m.shape)
    shape[dim] = out_size
    shape.insert(dim + 1, width)
    return gathered.reshape(shape).amin(dim=dim + 1)


def _weight_scores(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    r"""Scale detection ``scores`` by ``weights`` toward the worst score, so a weight never promotes.

    A weight in ``(0, 1]`` multiplies a non-negative score and *divides* a negative one: the score
    moves away from the best in both cases, and a down-weighted candidate can never outrank a
    full-weight candidate whose unweighted score was at least as good. A plain multiply pulls a
    negative score toward zero, i.e. up the ranking, which inverted the order of a signed
    response. Positions with a zero or negative weight are excluded from the candidates before
    this runs; their divisor is replaced by one so no ``inf`` is produced there.

    The quotient of a negative score and a small weight can leave the dtype's range, and ``-inf``
    is the sentinel an unfilled slot carries into the ranking, so the arithmetic runs in float32
    for half-precision input and the result is bounded at ``finfo.min`` of the score dtype before
    the cast back: a weighted detection is finite and so always stays ahead of the padding. Wider
    dtypes are unchanged wherever the quotient is representable.
    """
    dtype = scores.dtype
    wide = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    s, w = scores.to(wide), weights.to(wide)
    divisor = torch.where(w > 0, w, torch.ones_like(w))
    out = torch.where(s >= 0, s * w, s / divisor)
    return out.clamp_min(torch.finfo(dtype).min).to(dtype)


def _create_octave_mask(mask: torch.Tensor, octave_resp: torch.Tensor) -> torch.Tensor:
    r"""Resample a mask onto the given octave response map, as ``(B, 1, 1, H, W)``."""
    return _resize_mask(mask, octave_resp).unsqueeze(1)


def _check_mask(mask: torch.Tensor, img: torch.Tensor) -> None:
    r"""Check that ``mask`` is ``(1 or B, 1, H, W)`` for an image ``(B, C, H, W)``.

    The mask is resampled onto every level, so a mask of another spatial size would be stretched
    silently onto the wrong geometry -- a stale mask from before a resize, or a transposed one.
    It must be single-channel: it multiplies a single-channel response per level in
    :class:`MultiResolutionDetector`, and in :class:`ScaleSpaceDetector` it is broadcast over the
    scale levels, where a channel axis would land on the level axis instead.
    """
    KORNIA_CHECK(mask.dim() == 4 and mask.shape[1] == 1, f"mask must be (1 or B, 1, H, W). Got {tuple(mask.shape)}")
    KORNIA_CHECK(mask.device == img.device, f"mask device {mask.device} must match the image device {img.device}")
    KORNIA_CHECK(
        mask.shape[0] in (1, img.shape[0]),
        f"mask batch {mask.shape[0]} must be 1 or match the image batch {img.shape[0]}",
    )
    KORNIA_CHECK(
        mask.shape[-2:] == img.shape[-2:],
        f"mask spatial size {tuple(mask.shape[-2:])} must match the image {tuple(img.shape[-2:])}",
    )


def _zero_unfilled(lafs: torch.Tensor, filled: torch.Tensor) -> torch.Tensor:
    r"""Replace the LAFs of the slots no detection filled with the zero LAF.

    ``filled`` is a ``(B, N)`` boolean mask of the slots a detection actually filled. The
    affine-shape and orientation modules may normalise or propagate invalid padding frames
    differently by dtype, so the mask is re-applied after them. ``where`` rather than a multiply,
    so a module that returns NaN for a zero frame cannot leak it into the padding.

    Both detectors' ``forward`` read the mask off the LAFs that ``detect`` returns: a zero LAF is
    the padding contract, and it is the one signal a subclass overriding ``detect`` also honours.
    The response cannot serve: :class:`ScaleSpaceDetector`'s response function is pluggable and
    may be signed, so an exact zero there is a legitimate maximum.
    """
    return torch.where(filled.view(filled.shape[0], -1, 1, 1), lafs, torch.zeros_like(lafs))


class ScaleSpaceDetector(nn.Module):
    r"""nn.Module for differentiable local feature detection.

    As close as possible to classical local feature detectors
    like Harris, Hessian-Affine or SIFT (DoG).

    It has 5 modules inside: scale pyramid generator, response ("cornerness") function,
    sub-pixel localization, affine shape estimator and patch orientation estimator.
    Each of those modules could be replaced with a learned custom one, as long as
    they respect output shape.

    Args:
        num_features: Number of features to detect. In order to keep everything batchable,
          output would always have num_features output, even for completely homogeneous images.
        mr_size: multiplier for local feature scale compared to the detection scale.
          6.0 is matching OpenCV 12.0 convention for SIFT.
        scale_pyr_module: generates scale pyramid. See :class:`~kornia.geometry.ScalePyramid` for details.
          Default: ScalePyramid(3, 1.6, 15).
        resp_module: calculates ``'cornerness'`` of the pixel.
        subpix_module: performs non-maximum suppression and refines keypoint location to sub-pixel /
          sub-scale accuracy. See :class:`~kornia.geometry.subpix.ConvQuadInterp3d` for details.
        ori_module: for local feature orientation estimation. Default:class:`~kornia.feature.PassLAF`,
           which does nothing. See :class:`~kornia.feature.LAFOrienter` for details.
        aff_module: for local feature affine shape estimation. Default: :class:`~kornia.feature.PassLAF`,
            which does nothing. See :class:`~kornia.feature.LAFAffineShapeEstimator` for details.
        minima_are_also_good: if True, then both response function minima and maxima are detected.
            Useful for symmetric response functions like DoG or Hessian. Default is False.
        compile_modules: selects which sub-modules to wrap with :func:`torch.compile`.
            Pass ``True`` to compile every sub-module, ``False`` (default) for none, or a list
            containing any subset of ``["scale_pyr", "resp", "subpix", "ori", "aff"]``.
            Compiling ``subpix`` gives ~5x GPU speedup for the default
            :class:`~kornia.geometry.subpix.ConvQuadInterp3d` backend by fusing its iteration loop.
            The first call incurs a one-time compilation cost; subsequent calls are fast.

    """

    def __init__(
        self,
        num_features: int = 500,
        mr_size: float = 6.0,
        scale_pyr_module: Optional[nn.Module] = None,
        resp_module: Optional[nn.Module] = None,
        subpix_module: Optional[nn.Module] = None,
        ori_module: Optional[nn.Module] = None,
        aff_module: Optional[nn.Module] = None,
        minima_are_also_good: bool = False,
        scale_space_response: bool = False,
        compile_modules: Union[bool, List[str]] = False,
    ) -> None:
        super().__init__()
        self.mr_size = mr_size
        self.num_features = num_features

        _all_names = {"scale_pyr", "resp", "subpix", "ori", "aff"}
        if compile_modules is True:
            _compile_set = _all_names
        elif compile_modules is False:
            _compile_set = set()
        else:
            _compile_set = set(compile_modules)
            unknown = _compile_set - _all_names
            if unknown:
                raise ValueError(f"Unknown module names in compile_modules: {unknown}. Valid: {_all_names}")

        if _compile_set:
            # Allow torch.compile to keep data-dependent shape ops (torch.where / nonzero)
            # inside the compiled graph as unbacked symbols, avoiding graph breaks and the
            # 0/1-specialization recompilations that would otherwise fire whenever an octave
            # first encounters zero NMS maxima (blurry/extreme-viewpoint images).
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

        def _maybe_compile(mod: nn.Module, name: str) -> nn.Module:
            return torch.compile(mod, dynamic=True) if name in _compile_set else mod

        if scale_pyr_module is None:
            extra_levels = 3 if scale_space_response else 2
            scale_pyr_module = ScalePyramid(3, 1.6, 16, extra_levels=extra_levels)
        self.scale_pyr = _maybe_compile(scale_pyr_module, "scale_pyr")
        if resp_module is None:
            resp_module = BlobHessian()
        self.resp = _maybe_compile(resp_module, "resp")
        if subpix_module is None:
            subpix_module = AdaptiveQuadInterp3d(strict_maxima_bonus=0.0, allow_scale_steps=True)
        # Record before torch.compile wraps the module — isinstance won't match OptimizedModule.
        self._is_iterative_subpix: bool = isinstance(
            subpix_module, (ConvQuadInterp3d, AdaptiveQuadInterp3d, IterativeQuadInterp3d)
        )
        self.subpix = _maybe_compile(subpix_module, "subpix")
        if ori_module is None:
            ori_module = PassLAF()
        self.ori = _maybe_compile(ori_module, "ori")
        if aff_module is None:
            aff_module = PassLAF()
        self.aff = _maybe_compile(aff_module, "aff")
        self.minima_are_also_good = minima_are_also_good
        # scale_space_response should be True if the response function works on scale space
        # like Difference-of-Gaussians
        self.scale_space_response = scale_space_response

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_features={self.num_features}, "
            f"mr_size={self.mr_size}, "
            f"scale_pyr={self.scale_pyr.__repr__()}, "
            f"resp={self.resp.__repr__()}, "
            f"subpix={self.subpix.__repr__()}, "
            f"ori={self.ori.__repr__()}, "
            f"aff={self.aff.__repr__()})"
        )

    def _process_octave(
        self,
        octave: torch.Tensor,
        sigmas_oct: torch.Tensor,
        num_feats: int,
        mask: Optional[torch.Tensor],
        rotmat: torch.Tensor,
        num_levels: int,
        is_iterative_subpix: bool,
        px_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one scale-space octave: response → NMS/subpix → top-K → LAF.

        Returns the top-K responses, their LAFs, and a boolean mask of the slots a detection
        actually filled. The batched top-K below ranks over the whole volume and returns its
        ``fill`` sentinel once the octave runs out of candidates, and the border check rejects
        candidates after the ranking; the mask is what tells those slots apart from a detection,
        since the response alone cannot.
        """
        dev = octave.device
        B, CH, L, H, W = octave.size()

        # Run response function
        if self.scale_space_response:
            oct_resp = self.resp(octave, sigmas_oct.view(-1))  # (B, 1, Ldog, H, W)
        else:
            level_resp = self.resp(octave.permute(0, 2, 1, 3, 4).reshape(B * L, CH, H, W), sigmas_oct.view(-1))
            KORNIA_CHECK(
                level_resp.dim() == 4
                and level_resp.shape[0] == B * L
                and level_resp.shape[1] == 1
                and level_resp.shape[-2:] == (H, W),
                "resp_module must return one response map per image with shape "
                f"({B * L}, 1, {H}, {W}). Got {tuple(level_resp.shape)}",
            )
            oct_resp = level_resp.view(B, L, 1, H, W)
            # Reorder to (B, CH, L, H, W) for scale-space NMS
            oct_resp = oct_resp.permute(0, 2, 1, 3, 4)
        KORNIA_CHECK(
            oct_resp.dim() == 5 and oct_resp.shape[0] == B and oct_resp.shape[1] == 1 and oct_resp.shape[-2:] == (H, W),
            "resp_module must return one scale-space response channel with shape "
            f"({B}, 1, L, {H}, {W}). Got {tuple(oct_resp.shape)}. For a multi-channel image, convert it to "
            "grayscale first or use a response function that reduces the channels to one.",
        )
        # Iterative sub-pixel modules flatten the full response volume internally. The
        # level/channel permutation above is contiguous when CH == 1 (the common path), but
        # not for a response that preserves multiple channels.
        oct_resp = oct_resp.contiguous()
        scale_sigmas = sigmas_oct[:, : oct_resp.shape[2]]

        # Always precompute NMS masks in one fused pass.
        # - For minima_are_also_good: both masks are needed anyway.
        # - Otherwise: max_nms_mask is passed to subpix (skips its internal NMS on GPU)
        #   and drives the sparse top-K below.
        max_nms_mask: torch.Tensor
        min_nms_mask: torch.Tensor
        max_nms_mask, min_nms_mask = nms3d_minmax(oct_resp)

        # The mask is applied to the candidates, not to the response the NMS reads: multiplying the
        # response first would carve a hard edge into it, and every response pixel next to a zeroed
        # neighbour becomes a "maximum" along that edge. The resampled mask is conservative (see
        # `_resize_mask`), so a zero region drops every candidate it touches; a weight scales the
        # score of the candidates it keeps.
        oct_mask: Optional[torch.Tensor] = None
        oct_keep: Optional[torch.Tensor] = None
        if mask is not None:
            resampled = _create_octave_mask(mask, oct_resp)
            oct_keep = resampled > 0
            max_nms_mask = max_nms_mask & oct_keep
            min_nms_mask = min_nms_mask & oct_keep
            # A boolean or integer mask resamples to exactly 0/1, and the zeros are already dropped
            # above, so weighting by it is a no-op over the whole volume; only real weights run.
            if mask.is_floating_point():
                oct_mask = resampled

        if self.minima_are_also_good:
            if is_iterative_subpix:
                coord_max, response_max = self.subpix(oct_resp, precomputed_nms_mask=max_nms_mask)
                coord_min, response_min = self.subpix(-oct_resp, precomputed_nms_mask=min_nms_mask)
            else:
                coord_max, response_max = self.subpix(oct_resp)
                coord_min, response_min = self.subpix(-oct_resp)
        elif is_iterative_subpix:
            coord_max, response_max = self.subpix(oct_resp, precomputed_nms_mask=max_nms_mask)
        else:
            coord_max, response_max = self.subpix(oct_resp)

        # Zero responses at scale border levels so they never reach top-K.
        # (nms3d_minmax already sets the masks False at these positions.)
        response_max[:, :, 0] = 0.0
        response_max[:, :, -1] = 0.0
        if oct_mask is not None:
            response_max = _weight_scores(response_max, oct_mask)

        if self.minima_are_also_good:
            response_min[:, :, 0] = 0.0
            response_min[:, :, -1] = 0.0
            if oct_mask is not None:
                response_min = _weight_scores(response_min, oct_mask)
            take_min_mask = (response_min > response_max) & min_nms_mask
            response_max = torch.where(take_min_mask, response_min, response_max)
            coord_max = torch.where(take_min_mask.unsqueeze(2), coord_min, coord_max)
            # Candidate positions: original max-NMS plus swapped min-NMS positions.
            cand_mask = max_nms_mask | take_min_mask
        else:
            cand_mask = max_nms_mask

        # Sparse top-K: gather the small set of NMS candidates first, then run top-K
        # on that (~few-thousand) set instead of the full CHxLxHxW volume (~millions).
        # nms3d_minmax guarantees cand_mask is False at scale border levels already.
        # Response/candidate tensors are (B, C, L, H, W), while coordinates are
        # (B, C, 3, L, H, W). Move the xyz axis last before flattening so entry i in
        # every tensor describes the same (channel, scale, y, x) candidate. Responses
        # are single-channel by the contract checked above, but custom sub-pixel modules
        # still have to preserve this axis order.
        mask_flat = cand_mask.reshape(B, -1)  # (B, C*L*H*W)
        resp_flat = response_max.reshape(B, -1)  # (B, C*L*H*W)
        coord_flat = coord_max.movedim(2, -1).reshape(B, -1, 3)  # (B, C*L*H*W, 3)

        if B == 1:
            nms_idx = mask_flat[0].nonzero(as_tuple=True)[0]  # (M,)
            resp_cands = resp_flat[0][nms_idx]  # (M,)
            coord_cands = coord_flat[0][nms_idx]  # (M, 3)
            k_eff = min(num_feats, nms_idx.shape[0])
            # Only NMS candidates are gathered here, so every returned slot is one.
            is_cand = torch.ones(1, k_eff, dtype=torch.bool, device=dev)
            if k_eff > 0:
                resp_flat_best, local_idx = torch.topk(resp_cands, k=k_eff)
                max_coords_best = coord_cands[local_idx].unsqueeze(0)  # (1, k_eff, 3)
                resp_flat_best = resp_flat_best.unsqueeze(0)  # (1, k_eff)
            else:
                resp_flat_best = resp_flat.new_zeros(1, 0)
                max_coords_best = coord_flat.new_zeros(1, 0, 3)
        else:
            # Batched fallback: mask non-candidates to -inf so they lose top-K to every finite
            # response. (A finite sentinel such as `finfo.min / 2` is not below every finite
            # response, and outranked a genuine extreme maximum.)
            resp_masked = resp_flat.masked_fill(~mask_flat, float("-inf"))
            k_eff = min(num_feats, resp_masked.size(1))
            resp_flat_best, idxs = torch.topk(resp_masked, k=k_eff, dim=1)
            max_coords_best = torch.gather(coord_flat, 1, idxs.unsqueeze(-1).expand(-1, -1, 3))
            # `topk` cannot rank among the masked-out positions -- they all carry the same
            # `fill` -- so an image with fewer than `num_feats` maxima gets an arbitrary subset
            # of non-candidates back. Carry the candidacy of each selected slot forward.
            is_cand = torch.gather(mask_flat, 1, idxs)

        B, N = resp_flat_best.size()

        max_coords_best = _scale_index_to_scale(max_coords_best, scale_sigmas, num_levels)

        current_lafs = torch.cat(
            [
                self.mr_size * max_coords_best[:, :, 0].view(B, N, 1, 1) * rotmat,
                max_coords_best[:, :, 1:3].view(B, N, 2, 1),
            ],
            3,
        )

        # Inline equivalent of laf_is_inside_image(scale_laf(current_lafs, 0.5), octave[:, 0], 5)
        # for isotropic LAFs (rotmat = eye(2)).  Avoids: scale_laf (torch.cat), and
        # laf_to_boundary_points (linspace/sin/cos allocations + CPU→GPU transfer + bmm).
        # For the axis-aligned isotropic case the 12-pt boundary check reduces to:
        #   max x-extent = max|sin| * half_s;  max y-extent = max|cos| * half_s = half_s
        half_s = current_lafs[:, :, 0, 0] * 0.5
        cx = current_lafs[:, :, 0, 2]
        cy = current_lafs[:, :, 1, 2]
        h, w = octave.shape[3], octave.shape[4]
        # Valid pixel coordinates run 0 .. w-1 / 0 .. h-1, so the upper bound is (size - 1) - border.
        x_max = float(w - 1) - 5
        y_max = float(h - 1) - 5
        good_mask = (
            (cx - half_s * _MAX_ABS_SIN_12 >= 5)
            & (cx + half_s * _MAX_ABS_SIN_12 <= x_max)
            & (cy - half_s >= 5)
            & (cy + half_s <= y_max)
        )
        if oct_keep is not None:
            # The mask was checked at the integer NMS site, and the sub-pixel step has since moved
            # the centre -- across the mask boundary, when the true peak lies just inside the zero
            # region. Re-check the refined centre against the resampled mask, conservatively: every
            # octave pixel the centre touches (its floor and ceil in x and y) has to be allowed, as a
            # zero region already suppresses every candidate within one octave pixel of it.
            keep_flat = oct_keep.reshape(oct_keep.shape[0], -1)  # (1 or B, H*W)
            if keep_flat.shape[0] != B:
                keep_flat = keep_flat.expand(B, -1)
            x0 = cx.floor().long().clamp_(0, w - 1)
            x1 = cx.ceil().long().clamp_(0, w - 1)
            y0 = cy.floor().long().clamp_(0, h - 1)
            y1 = cy.ceil().long().clamp_(0, h - 1)
            for yy in (y0, y1):
                for xx in (x0, x1):
                    good_mask = good_mask & torch.gather(keep_flat, 1, yy * w + xx)
        # A slot is filled by a candidate whose frame lies inside the image and the mask. Everything
        # else -- a batched top-K sentinel, or a candidate the border or mask check rejects --
        # carries `-inf` from here so that it loses the cross-octave ranking in `_detect` to every
        # real detection, however negative; `_detect` zeroes its response and LAF once ranked.
        filled = is_cand & good_mask
        resp_flat_best = resp_flat_best.masked_fill(~filled, float("-inf"))
        current_lafs.mul_(px_size)
        return resp_flat_best, current_lafs, filled

    def _detect(
        self, img: torch.Tensor, num_feats: int, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the detection and also return the boolean mask of the slots a detection filled.

        ``forward`` needs the mask after ``aff`` and ``ori`` have run, and it cannot be recovered
        from the responses, so :meth:`detect` is a two-value view of this.
        """
        if mask is not None:
            _check_mask(mask, img)
        dev = img.device
        dtype: torch.dtype = img.dtype
        sp, sigmas, _ = self.scale_pyr(img)

        # ── Hoist loop invariants ────────────────────────────────────────────
        if isinstance(self.scale_pyr.n_levels, torch.Tensor):
            num_levels = int(self.scale_pyr.n_levels.item())
        elif isinstance(self.scale_pyr.n_levels, int):
            num_levels = self.scale_pyr.n_levels
        else:
            raise TypeError(
                "Expected the scale pyramid module to have `n_levels` as a torch.Tensor or int."
                f"Gotcha {type(self.scale_pyr.n_levels)}"
            )
        rotmat = torch.eye(2, dtype=dtype, device=dev).view(1, 1, 2, 2)
        is_iterative_subpix = self._is_iterative_subpix
        px_size0 = 0.5 if self.scale_pyr.double_image else 1.0
        px_sizes = [px_size0 * (2.0**i) for i in range(len(sp))]

        # ── Process octaves sequentially ────────────────────────────────────
        # All octaves are independent once the scale pyramid is built, but CUDA
        # stream parallelism does not help here: subpix allocates large scatter
        # tables per call, and concurrent CUDA allocations contend for the same
        # device memory allocator lock.  Tested: sequential ≈ parallel on GPU.
        n_oct = len(sp)
        results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
            self._process_octave(
                sp[i], sigmas[i], num_feats, mask, rotmat, num_levels, is_iterative_subpix, px_sizes[i]
            )
            for i in range(n_oct)
        ]

        # Sort and keep best n across all octaves. Sparse per-octave top-K may yield fewer total
        # candidates than num_feats (e.g. small images with very few NMS maxima); the result is
        # padded to preserve the shape contract [B, num_feats, ...]. Every unfilled slot -- the
        # padding, a batched top-K sentinel, a border- or mask-rejected candidate -- carries `-inf`
        # through the ranking so that a genuine detection, however negative, still sorts ahead of
        # it, and is zeroed only afterwards. (`_weight_scores` keeps a weighted score finite.)
        responses = torch.cat([r[0] for r in results], 1)
        lafs = torch.cat([r[1] for r in results], 1)
        filled = torch.cat([r[2] for r in results], 1)
        n_candidates = responses.size(1)
        if n_candidates < num_feats:
            pad = num_feats - n_candidates
            responses = F.pad(responses, (0, pad), value=float("-inf"))
            lafs = F.pad(lafs, (0, 0, 0, 0, 0, pad))
            filled = F.pad(filled, (0, pad))
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = torch.gather(lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 3))
        filled = torch.gather(filled, 1, idxs)
        responses = torch.where(filled, responses, torch.zeros_like(responses))
        return responses, _zero_unfilled(lafs, filled), filled

    def detect(
        self, img: torch.Tensor, num_feats: int, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect local features in an image batch.

        Args:
            img: Input image tensor with shape `(B, C, H, W)`.
            num_feats: Number of features requested from the detector.
            mask: Optional mask with shape `(1 or B, 1, H, W)` saying where a detection may be. A boolean or integer
                mask is binary (any non-zero value keeps a position), a floating-point mask is used as weights on the
                detection scores: a weight in `(0, 1]` scales a score toward the worst (a non-negative score is
                multiplied by it, a negative one divided), so a down-weighted candidate never outranks a full-weight
                one, and a zero or negative weight suppresses. Weights above one are clamped to one, so a float
                0/255 mask means what the integer one means. The weights are cast to the image dtype. The mask is
                resampled onto every octave conservatively, so a zero region suppresses every candidate within one
                octave pixel of it.

        Returns:
            Tuple containing detection scores and local affine frames, shaped `(B, num_feats)` and `(B, num_feats,
            2, 3)`. A slot that no detection filled -- there were fewer candidates than requested, or a candidate's
            frame reached outside the image -- carries a zero response and a zero LAF, and sorts after every real
            detection. The converse does not hold: a signed response function can peak at exactly zero, and that
            slot keeps its frame.
        """
        responses, lafs, _ = self._detect(img, num_feats, mask)
        return responses, lafs

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Three stage local feature detection.

        First the location and scale of interest points are determined by detect function.
        Then affine shape and orientation.

        Args:
            img: image to extract features with shape [BxCxHxW]
            mask: a mask saying where a detection may be, shape [1x1xHxW] or [Bx1xHxW]. A boolean or integer mask
                is binary, a floating-point mask weights the detection scores; see :meth:`detect`.

        Returns:
            Tuple of ``lafs`` with shape [BxNx2x3], the detected local affine frames, and ``responses`` with shape
            [BxN], the response function values for the corresponding lafs. When an image yields fewer maxima than
            ``num_features``, the remaining slots hold a zero response and a zero LAF, whichever affine-shape and
            orientation modules are configured.

        """
        # `detect` is the public extension point used by subclasses. Infer occupancy from its
        # zero-LAF padding contract rather than bypassing an override through `_detect`.
        responses, lafs = self.detect(img, self.num_features, mask)
        filled = lafs.ne(0).any(dim=-1).any(dim=-1)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return _zero_unfilled(lafs, filled), responses


class Detector_config(TypedDict):
    """Configuration for the Scale Space Detector.

    Attributes:
        nms_size: The size of the Non-Maximum Suppression window.
        pyramid_levels: The number of levels in the image pyramid.
    """

    nms_size: int
    pyramid_levels: int
    up_levels: int
    scale_factor_levels: float
    s_mult: float


def get_default_detector_config() -> Detector_config:
    """Return default config."""
    # Return a shallow copy to ensure modifications outside don't affect the module-level config.
    return _DEFAULT_DETECTOR_CONFIG.copy()


class MultiResolutionDetector(nn.Module):
    """Multi-scale feature detector, based on code from KeyNet. Can be used with any response function.

    This is based on the original code from paper
    "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters".
    See :cite:`KeyNet2019` for more details.

    Args:
        model: response function, such as KeyNet or BlobHessian
        num_features: Number of features to detect.
        conf: Dict with initialization parameters. Do not pass it, unless you know what you are doing`.
        ori_module: for local feature orientation estimation. Default: :class:`~kornia.feature.PassLAF`,
           which does nothing. See :class:`~kornia.feature.LAFOrienter` for details.
        aff_module: for local feature affine shape estimation. Default: :class:`~kornia.feature.PassLAF`,
            which does nothing. See :class:`~kornia.feature.LAFAffineShapeEstimator` for details.
        compile_model: wrap the response function and the non-maxima suppression with :func:`torch.compile`.
        score_threshold: minimum response for a position to count as a detection. Must be non-negative:
            non-maxima suppression writes an exact zero at every suppressed position, so a negative
            threshold would admit all of them.

    """

    def __init__(
        self,
        model: nn.Module,
        num_features: int = 2048,
        config: Optional[Detector_config] = None,
        ori_module: Optional[nn.Module] = None,
        aff_module: Optional[nn.Module] = None,
        compile_model: bool = False,
        score_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if config is None:
            config = get_default_detector_config()
        # Load extraction configuration
        self.num_pyramid_levels = config["pyramid_levels"]
        self.num_upscale_levels = config["up_levels"]
        self.scale_factor_levels = config["scale_factor_levels"]
        self.mr_size = config["s_mult"]
        self.nms_size = config["nms_size"]
        if score_threshold < 0:
            # Non-maxima suppression encodes a suppressed position as an exact zero, so a
            # negative threshold admits every suppressed pixel in the image as a "detection"
            # and collides with the zero response that marks an unfilled slot.
            raise ValueError(f"score_threshold must be non-negative. Got {score_threshold}")
        self.score_threshold = score_threshold
        nms = NonMaximaSuppression2d((self.nms_size, self.nms_size))
        self.num_features = num_features

        if compile_model:
            self.model = torch.compile(model, dynamic=True)
            self.nms = torch.compile(nms, dynamic=True)
        else:
            self.model = model
            self.nms = nms

        if ori_module is None:
            self.ori: nn.Module = PassLAF()
        else:
            self.ori = ori_module

        if aff_module is None:
            self.aff: nn.Module = PassLAF()
        else:
            self.aff = aff_module

    def remove_borders(self, score_map: torch.Tensor, borders: int = 15) -> torch.Tensor:
        """Remove the borders of the image to avoid detections on the corners."""
        mask = torch.zeros_like(score_map)
        mask[:, :, borders:-borders, borders:-borders] = 1
        return mask * score_map

    def detect_features_on_single_level(
        self,
        level_img: torch.Tensor,
        num_kp: int,
        factor: Tuple[float, float],
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect keypoints on one image-pyramid level.

        The response function may consume a multi-channel image -- for example, a learned color detector -- but must
        return one response map. A LAF has no response-channel identity, so independent per-channel detections are
        ambiguous and rejected. :class:`ScaleSpaceDetector` follows the same contract.

        Args:
            level_img: Image tensor for a single pyramid level.
            num_kp: Number of keypoints requested from this pyramid level.
            factor: Scale factor mapping coordinates from the current pyramid level back to the original image
                resolution.
            mask: Optional mask with shape :math:`(1, 1, H, W)` at the *original* image resolution, saying where a
                detection may be. It is resampled onto this level conservatively and applied to the non-maxima
                suppression output, so a zero region drops every maximum within one level pixel of it and a
                floating-point weight scales the score. Keyword-only.

        Returns:
            Tuple containing scores and local affine frames detected at the requested pyramid level, with
            ``min(num_kp, H * W)`` slots. When the level holds fewer above-threshold maxima than that, the remaining
            slots are padded with a zero response and a zero LAF.
        """
        resp_map = self.model(level_img)
        KORNIA_CHECK(
            resp_map.dim() == 4 and resp_map.shape[0] == 1 and resp_map.shape[1] == 1,
            f"model must return one response map with shape (1, 1, H, W). Got {tuple(resp_map.shape)}. "
            "For a multi-channel image, convert it to grayscale first or use a response function that "
            "reduces the channels to one map.",
        )
        # A response index is decoded as a level pixel with no offset or stride, so a map of another
        # size -- a valid-convolution net -- would put every keypoint off its peak and have the mask
        # resampled onto the wrong geometry. Nothing can infer the offset from the shape.
        KORNIA_CHECK(
            resp_map.shape[-2:] == level_img.shape[-2:],
            f"model must return a response map with the level's spatial size {tuple(level_img.shape[-2:])}. "
            f"Got {tuple(resp_map.shape[-2:])}; pad the response function so that its output matches its input.",
        )
        det_map = self.nms(self.remove_borders(resp_map))
        # The mask is applied to the maxima, not to the response the NMS reads: a hard edge in the
        # response would turn every pixel beside a zeroed neighbour into a "maximum".
        if mask is not None:
            weights = _resize_mask(mask, det_map)
            # A boolean or integer mask resamples to exactly 0/1: dropping is all it can do. A float
            # mask is gated the same way `ScaleSpaceDetector` gates its candidates, `weight > 0`, which
            # also drops a NaN weight: `s * NaN` sorts first in `topk` and would consume a slot.
            if mask.is_floating_point():
                det_map = _weight_scores(det_map, weights).masked_fill(~(weights > 0), 0.0)
            else:
                det_map = det_map * weights
        w = det_map.shape[-1]
        det_flat = det_map.view(-1)  # (H*W,)

        # Mask out non-maxima (zeroed by NMS) and below-threshold scores, then topk.
        # Using masked_fill + topk instead of nonzero: avoids data-dependent output shapes,
        # supports score_threshold, and is compatible with torch.compile.
        det_masked = det_flat.masked_fill(det_flat <= self.score_threshold, float("-inf"))
        k = min(num_kp, det_flat.numel())
        top_scores, top_flat_idx = torch.topk(det_masked, k=k)

        # `topk` cannot rank among the masked-out positions — they all carry the same `fill` —
        # so once this level runs out of real candidates it returns an arbitrary tie-break subset
        # of them, in practice the lowest flat indices, i.e. the border strip `remove_borders` has
        # just zeroed. Neutralise those slots rather than handing the sentinel back: zero response
        # and zero LAF, which is how `ScaleSpaceDetector.detect` pads a short result.
        valid = top_scores > self.score_threshold
        top_scores = torch.where(valid, top_scores, torch.zeros_like(top_scores))

        # Convert flat indices to (y, x) pixel coordinates and project them to the original
        # resolution. The arithmetic runs in at least float32 -- a half-precision image cannot hold
        # a pixel index times a scale factor exactly -- and only the finished coordinate is cast
        # to the image dtype, which is exact for every integer a half-precision type can hold.
        yx = torch.stack([top_flat_idx // w, top_flat_idx % w], dim=1)  # (k, 2)
        wide = torch.float64 if level_img.dtype == torch.float64 else torch.float32
        fx = torch.tensor([factor[0], factor[1]], device=level_img.device, dtype=wide)
        xy_projected = (yx.view(1, k, 2).flip(2).to(wide) * fx).to(level_img.dtype)
        scale_val = 0.5 * (factor[0] + factor[1]) * self.mr_size
        scale = level_img.new_full((1, k, 1, 1), scale_val)
        lafs = laf_from_center_scale_ori(xy_projected, scale, level_img.new_zeros(1, k, 1))
        lafs = lafs * valid.view(1, k, 1, 1).to(lafs.dtype)
        return top_scores, lafs

    def _detect_level(
        self, level_img: torch.Tensor, num_kp: int, factor: Tuple[float, float], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # `mask` is passed only when there is one, so a subclass that overrides the public method
        # with the historical three-argument signature keeps working for the unmasked call.
        if mask is None:
            return self.detect_features_on_single_level(level_img, num_kp, factor)
        return self.detect_features_on_single_level(level_img, num_kp, factor, mask=mask)

    def detect(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect local features in an image batch.

        Args:
            img: Input image tensor with shape `(1, C, H, W)`.
            mask: Optional mask with shape `(1, 1, H, W)` saying where a detection may be. A boolean or integer
                mask is binary (any non-zero value keeps a position), a floating-point mask is used as weights on
                the detection scores: a weight in `(0, 1]` scales a score toward the worst, so a down-weighted
                maximum never outranks a full-weight one, and a zero or negative weight suppresses. Weights above
                one are clamped to one, so a float 0/255 mask means what the integer one means. The weights are
                cast to the image dtype. The mask is resampled onto every pyramid level conservatively, so a zero
                region suppresses every maximum within one level pixel of it.

        Returns:
            Tuple containing detection scores and local affine frames, shaped `(1, num_features)` and
            `(1, num_features, 2, 3)`. The shape holds even when the image yields fewer above-threshold maxima
            than requested: those slots carry a zero response and a zero LAF, and sort after every real detection.
            LAF centres are pixel coordinates cast to the image dtype, so a half-precision image gives centres at
            that dtype's integer resolution: exact up to 256 in bfloat16 and up to 2048 in float16, and coarser
            beyond.
        """
        KORNIA_CHECK_SHAPE(img, ["1", "C", "H", "W"])
        if mask is not None:
            _check_mask(mask, img)
        # Compute points per level
        num_features_per_level: List[float] = []
        tmp = 0.0
        factor_points = self.scale_factor_levels**2
        levels = self.num_pyramid_levels + self.num_upscale_levels + 1
        for idx_level in range(levels):
            tmp += factor_points ** (-1 * (idx_level - self.num_upscale_levels))
            nf = self.num_features * factor_points ** (-1 * (idx_level - self.num_upscale_levels))
            num_features_per_level.append(nf)
        shares: List[float] = [x / tmp for x in num_features_per_level]
        # Largest-remainder (Hamilton) apportionment: `shares` sums to `self.num_features` (up to
        # float error), but flooring each one independently discards every fractional part, so the
        # floors can sum to well under `self.num_features` -- to zero at `num_features=1`, where the
        # default six shares are 0.508 .. 0.016. The finest level's quota also dominates every other
        # one's, so a small request degenerates to querying a single scale. Hand the shortfall --
        # `self.num_features` minus the sum of floors -- to the levels with the largest fractional
        # remainders, one slot each, so the quotas always sum to exactly `self.num_features` and stay
        # spread across scales instead of collapsing onto the finest level.
        num_features_per_level = [int(x) for x in shares]
        shortfall = self.num_features - sum(num_features_per_level)
        by_remainder = sorted(range(len(shares)), key=lambda i: shares[i] - num_features_per_level[i], reverse=True)
        for idx_level in by_remainder[:shortfall]:
            num_features_per_level[idx_level] += 1

        _, _, h, w = img.shape
        img_up = img
        cur_img = img
        all_responses: List[torch.Tensor] = []
        all_lafs: List[torch.Tensor] = []
        # Extract features from the upper levels
        for idx_level in range(self.num_upscale_levels):
            nf = num_features_per_level[len(num_features_per_level) - self.num_pyramid_levels - 1 - (idx_level + 1)]
            num_points_level = int(nf)

            # Resize input image
            up_factor = self.scale_factor_levels ** (1 + idx_level)
            nh, nw = int(h * up_factor), int(w * up_factor)
            up_factor_kpts = (float(w) / float(nw), float(h) / float(nh))
            img_up = resize(img_up, (nh, nw), interpolation="bilinear", align_corners=False)

            cur_scores, cur_lafs = self._detect_level(img_up, num_points_level, up_factor_kpts, mask)

            all_responses.append(cur_scores.view(1, -1))
            all_lafs.append(cur_lafs)

        # Extract features from the downsampling pyramid
        for idx_level in range(self.num_pyramid_levels + 1):
            if idx_level > 0:
                cur_img = pyrdown(cur_img, factor=self.scale_factor_levels)
                _, _, nh, nw = cur_img.shape
                factor = (float(w) / float(nw), float(h) / float(nh))
            else:
                factor = (1.0, 1.0)

            num_points_level = int(num_features_per_level[idx_level])
            if idx_level > 0 or (self.num_upscale_levels > 0):
                num_points_level = sum(num_features_per_level[: idx_level + 1 + self.num_upscale_levels])

            cur_scores, cur_lafs = self._detect_level(cur_img, num_points_level, factor, mask)
            all_responses.append(cur_scores.view(1, -1))
            all_lafs.append(cur_lafs)
        responses = torch.cat(all_responses, 1)
        lafs = torch.cat(all_lafs, 1)
        # The levels can produce fewer slots than requested — a level is capped at its own pixel
        # count, so a deep enough pyramid level runs out of positions before it fills its quota.
        # Pad up so the returned shape is always `num_features`, the same way
        # `ScaleSpaceDetector.detect` does; the padding is the zero response and zero LAF used
        # everywhere else here.
        if lafs.shape[1] < self.num_features:
            pad = self.num_features - lafs.shape[1]
            responses = F.pad(responses, (0, pad))
            lafs = F.pad(lafs, (0, 0, 0, 0, 0, pad))
        # Rank unconditionally. The levels are concatenated in level order, and a level that under-
        # fills its own quota pads in place, so without this a short result interleaves the real
        # detections with the padding instead of listing them first, unlike every other path here
        # and unlike `ScaleSpaceDetector.detect`. Detections score strictly above the non-negative
        # threshold, so they sort ahead of the zero-response padding.
        responses, idxs = torch.topk(responses, k=self.num_features, dim=1)
        lafs = torch.gather(lafs, 1, idxs[..., None, None].expand(-1, -1, 2, 3))
        return responses, lafs

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Three stage local feature detection.

        First the location and scale of interest points are determined by detect function.
        Then affine shape and orientation.

        Args:
            img: image to extract features with shape [1xCxHxW]. KeyNetDetector does not support batch processing,
                because the number of detections is different on each image.
            mask: a mask saying where a detection may be, shape [1x1xHxW]. A boolean or integer mask is binary, a
                floating-point mask weights the detection scores; see :meth:`detect`.

        Returns:
            Tuple of ``lafs`` with shape [1xNx2x3], the detected local affine frames, and ``responses`` with shape
            [1xN], the response function values for the corresponding lafs. When the image yields fewer
            above-threshold maxima than ``num_features``, the remaining slots hold a zero response and a zero LAF,
            whichever affine-shape and orientation modules are configured.

        """
        KORNIA_CHECK_SHAPE(img, ["1", "C", "H", "W"])
        responses, lafs = self.detect(img, mask)
        # Occupancy comes from `detect`'s zero-LAF padding contract, the same way as in
        # `ScaleSpaceDetector.forward`, so an override of `detect` is honoured as well.
        filled = lafs.ne(0).any(dim=-1).any(dim=-1)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return _zero_unfilled(lafs, filled), responses


_DEFAULT_DETECTOR_CONFIG: Detector_config = {
    # Extraction Parameters
    "nms_size": 15,
    "pyramid_levels": 4,
    "up_levels": 1,
    "scale_factor_levels": math.sqrt(2),
    "s_mult": 22.0,
}
