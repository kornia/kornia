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

from kornia.core.check import KORNIA_CHECK_SHAPE
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
#   angles = linspace(0, 2π, 11) → k * 2π/11 for k=0..10
#   max|sin| at k=3: sin(6π/11) ≈ 0.9898;  max|cos| at k=0: cos(0) = 1.0
# Used to inline the boundary check in _process_octave for isotropic LAFs (rotmat=eye(2)),
# avoiding CPU→GPU allocation + bmm every octave.
_MAX_ABS_SIN_12: float = math.sin(3 * 2 * math.pi / 11)  # ≈ 0.9898


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


def _create_octave_mask(mask: torch.Tensor, octave_shape: List[int]) -> torch.Tensor:
    r"""Downsample a mask based on the given octave shape."""
    mask_shape = octave_shape[-2:]
    mask_octave = F.interpolate(mask, mask_shape, mode="bilinear", align_corners=False)
    return mask_octave.unsqueeze(1)


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one scale-space octave: response → NMS/subpix → top-K → LAF."""
        dev = octave.device
        dtype = octave.dtype
        B, CH, L, H, W = octave.size()

        # Run response function
        if self.scale_space_response:
            oct_resp = self.resp(octave, sigmas_oct.view(-1))  # (B, C, Ldog, H, W)
        else:
            oct_resp = self.resp(octave.permute(0, 2, 1, 3, 4).reshape(B * L, CH, H, W), sigmas_oct.view(-1)).view(
                B, L, CH, H, W
            )
            # Reorder to (B, CH, L, H, W) for scale-space NMS
            oct_resp = oct_resp.permute(0, 2, 1, 3, 4)
        scale_sigmas = sigmas_oct[:, : oct_resp.shape[2]]

        if mask is not None:
            oct_mask: torch.Tensor = _create_octave_mask(mask, oct_resp.shape)
            oct_resp = oct_mask * oct_resp

        # Always precompute NMS masks in one fused pass.
        # - For minima_are_also_good: both masks are needed anyway.
        # - Otherwise: max_nms_mask is passed to subpix (skips its internal NMS on GPU)
        #   and drives the sparse top-K below.
        max_nms_mask: torch.Tensor
        min_nms_mask: torch.Tensor
        max_nms_mask, min_nms_mask = nms3d_minmax(oct_resp)

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

        if self.minima_are_also_good:
            response_min[:, :, 0] = 0.0
            response_min[:, :, -1] = 0.0
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
        mask_flat = cand_mask.view(B, -1)  # (B, L*H*W)
        resp_flat = response_max.view(B, -1)  # (B, L*H*W)
        coord_flat = coord_max.view(B, 3, -1).permute(0, 2, 1)  # (B, L*H*W, 3)

        if B == 1:
            nms_idx = mask_flat[0].nonzero(as_tuple=True)[0]  # (M,)
            resp_cands = resp_flat[0][nms_idx]  # (M,)
            coord_cands = coord_flat[0][nms_idx]  # (M, 3)
            k_eff = min(num_feats, nms_idx.shape[0])
            if k_eff > 0:
                resp_flat_best, local_idx = torch.topk(resp_cands, k=k_eff)
                max_coords_best = coord_cands[local_idx].unsqueeze(0)  # (1, k_eff, 3)
                resp_flat_best = resp_flat_best.unsqueeze(0)  # (1, k_eff)
            else:
                resp_flat_best = resp_flat.new_zeros(1, 0)
                max_coords_best = coord_flat.new_zeros(1, 0, 3)
        else:
            # Batched fallback: mask non-candidates to -inf so they lose top-K.
            fill = torch.finfo(dtype).min / 2
            resp_masked = resp_flat.masked_fill(~mask_flat, fill)
            k_eff = min(num_feats, resp_masked.size(1))
            resp_flat_best, idxs = torch.topk(resp_masked, k=k_eff, dim=1)
            max_coords_best = torch.gather(coord_flat, 1, idxs.unsqueeze(-1).expand(-1, -1, 3))

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
        good_mask = (
            (cx - half_s * _MAX_ABS_SIN_12 >= 5)
            & (cx + half_s * _MAX_ABS_SIN_12 <= w - 5)
            & (cy - half_s >= 5)
            & (cy + half_s <= h - 5)
        )
        resp_flat_best = resp_flat_best * good_mask.to(dev, dtype)
        current_lafs.mul_(px_size)
        return resp_flat_best, current_lafs

    def detect(
        self, img: torch.Tensor, num_feats: int, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        results: List[Tuple[torch.Tensor, torch.Tensor]] = [
            self._process_octave(
                sp[i], sigmas[i], num_feats, mask, rotmat, num_levels, is_iterative_subpix, px_sizes[i]
            )
            for i in range(n_oct)
        ]

        # Sort and keep best n across all octaves.
        # Sparse per-octave top-K may yield fewer total candidates than num_feats
        # (e.g. small images with very few NMS maxima).  topk then pads with zeros
        # to preserve the shape contract [B, num_feats, ...].
        responses = torch.cat([r[0] for r in results], 1)
        lafs = torch.cat([r[1] for r in results], 1)
        n_candidates = responses.size(1)
        if n_candidates < num_feats:
            pad = num_feats - n_candidates
            responses = F.pad(responses, (0, pad))
            lafs = F.pad(lafs, (0, 0, 0, 0, 0, pad))
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = torch.gather(lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 3))
        return responses, lafs

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Three stage local feature detection.

        First the location and scale of interest points are determined by detect function.
        Then affine shape and orientation.

        Args:
            img: image to extract features with shape [BxCxHxW]
            mask: a mask with weights where to apply the response function. The shape must be the same as
              the input image.

        Returns:
            lafs: shape [BxNx2x3]. Detected local affine frames.
            responses: shape [BxNx1]. Response function values for corresponding lafs

        """
        responses, lafs = self.detect(img, self.num_features, mask)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return lafs, responses


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
        self, level_img: torch.Tensor, num_kp: int, factor: Tuple[float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        det_map = self.nms(self.remove_borders(self.model(level_img)))
        _, _, _h, w = det_map.shape
        det_flat = det_map.view(-1)  # (H*W,) — B=1, C=1 guaranteed by MultiResolutionDetector

        # Mask out non-maxima (zeroed by NMS) and below-threshold scores, then topk.
        # Using masked_fill + topk instead of nonzero: avoids data-dependent output shapes,
        # supports score_threshold, and is compatible with torch.compile.
        fill = torch.finfo(det_flat.dtype).min / 2
        det_masked = det_flat.masked_fill(det_flat <= self.score_threshold, fill)
        k = min(num_kp, det_flat.numel())
        top_scores, top_flat_idx = torch.topk(det_masked, k=k)

        # Convert flat indices to (y, x) pixel coordinates.
        yx = torch.stack([top_flat_idx // w, top_flat_idx % w], dim=1)  # (k, 2)

        fx = level_img.new_tensor([factor[0], factor[1]])
        xy_projected = yx.view(1, k, 2).flip(2).float() * fx
        scale_val = 0.5 * (factor[0] + factor[1]) * self.mr_size
        scale = level_img.new_full((1, k, 1, 1), scale_val)
        lafs = laf_from_center_scale_ori(xy_projected, scale, level_img.new_zeros(1, k, 1))
        return top_scores, lafs

    def detect(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute points per level
        num_features_per_level: List[float] = []
        tmp = 0.0
        factor_points = self.scale_factor_levels**2
        levels = self.num_pyramid_levels + self.num_upscale_levels + 1
        for idx_level in range(levels):
            tmp += factor_points ** (-1 * (idx_level - self.num_upscale_levels))
            nf = self.num_features * factor_points ** (-1 * (idx_level - self.num_upscale_levels))
            num_features_per_level.append(nf)
        num_features_per_level = [int(x / tmp) for x in num_features_per_level]

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

            cur_scores, cur_lafs = self.detect_features_on_single_level(img_up, num_points_level, up_factor_kpts)

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

            cur_scores, cur_lafs = self.detect_features_on_single_level(cur_img, num_points_level, factor)
            all_responses.append(cur_scores.view(1, -1))
            all_lafs.append(cur_lafs)
        responses = torch.cat(all_responses, 1)
        lafs = torch.cat(all_lafs, 1)
        if lafs.shape[1] > self.num_features:
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
            mask: a mask with weights where to apply the response function. The shape must be the same as
              the input image.

        Returns:
            lafs: shape [1xNx2x3]. Detected local affine frames.
            responses: shape [1xNx1]. Response function values for corresponding lafs

        """
        KORNIA_CHECK_SHAPE(img, ["1", "C", "H", "W"])
        responses, lafs = self.detect(img, mask)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return lafs, responses


_DEFAULT_DETECTOR_CONFIG: Detector_config = {
    # Extraction Parameters
    "nms_size": 15,
    "pyramid_levels": 4,
    "up_levels": 1,
    "scale_factor_levels": math.sqrt(2),
    "s_mult": 22.0,
}
