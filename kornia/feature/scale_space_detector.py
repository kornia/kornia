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

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import TypedDict

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.subpix import IterativeQuadInterp3d, NonMaximaSuppression2d
from kornia.geometry.transform import ScalePyramid, pyrdown, resize

from .laf import laf_from_center_scale_ori, laf_is_inside_image
from .orientation import PassLAF
from .responses import BlobHessian


def _scale_index_to_scale(
    max_coords: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    r"""Convert a fractional scale-level index to an actual sigma via linear interpolation.

    Args:
        max_coords: shape :math:`(B, N, 3)`.  Column 0 is a float scale-level
            index in :math:`[0, D-1]`; columns 1-2 are ``x`` (width) and ``y``
            (height) pixel coordinates.
        sigmas: shape :math:`(B, D)`.  Per-level sigma values for the current
            octave, as returned by :class:`~kornia.geometry.ScalePyramid`.

    Returns:
        Tensor of shape :math:`(B, N, 3)` with column 0 replaced by the
        interpolated sigma value.
    """
    _, _, _ = max_coords.shape
    D = sigmas.shape[1]

    scale_idx = max_coords[:, :, 0].clamp(0.0, D - 1.0)  # (B, N) float
    idx_lo = scale_idx.long().clamp(0, D - 2)  # (B, N) int
    idx_hi = (idx_lo + 1).clamp(0, D - 1)  # (B, N) int
    t = scale_idx - idx_lo.to(scale_idx.dtype)  # (B, N) fractional part

    sigma_lo = sigmas.gather(1, idx_lo)  # (B, N)
    sigma_hi = sigmas.gather(1, idx_hi)  # (B, N)
    sigma = sigma_lo + t * (sigma_hi - sigma_lo)  # (B, N) linearly interpolated

    return torch.cat([sigma.unsqueeze(2), max_coords[:, :, 1:]], dim=2)


def _create_octave_mask(mask: torch.Tensor, octave_shape: list[int]) -> torch.Tensor:
    r"""Downsample a mask to match the given octave spatial shape."""
    mask_shape = octave_shape[-2:]
    mask_octave = F.interpolate(mask, mask_shape, mode="bilinear", align_corners=False)
    return mask_octave.unsqueeze(1)


class ScaleSpaceDetector(nn.Module):
    r"""Differentiable local feature detector operating in scale space.

    Closely mimics classical detectors such as Harris, Hessian-Affine and
    SIFT (DoG).  The five internal modules — scale pyramid, response function,
    NMS/subpixel refinement, affine shape estimator and orientation estimator —
    can each be swapped for a learned equivalent as long as they respect the
    expected output shapes.

    Args:
        num_features: number of features to detect.  The output always contains
            exactly ``num_features`` entries (padded with low-response detections
            for homogeneous regions).
        mr_size: multiplier for the LAF scale relative to the detection sigma.
            ``6.0`` matches OpenCV's ``12.0`` convention for SIFT.
        scale_pyr_module: scale pyramid generator.
            Default: ``ScalePyramid(3, 1.6, 15)``.
        resp_module: response ("cornerness") function.
            Default: :class:`~kornia.feature.BlobHessian`.
        nms_module: NMS and subpixel refinement module.
            Must return ``(coords [B,C,3,D,H,W], values [B,C,D,H,W])`` like
            :class:`~kornia.geometry.subpix.IterativeQuadInterp3d`.
            Default: :class:`~kornia.geometry.subpix.IterativeQuadInterp3d`.
        ori_module: orientation estimator.
            Default: :class:`~kornia.feature.PassLAF` (identity).
        aff_module: affine shape estimator.
            Default: :class:`~kornia.feature.PassLAF` (identity).
        minima_are_also_good: if ``True`` both maxima and minima of the response
            are returned.  Useful for symmetric responses like DoG or Hessian.
        scale_space_response: set ``True`` when the response function already
            operates on the full scale-space volume (e.g. DoG), ``False`` when
            it is applied independently per level (e.g. Hessian).
    """

    def __init__(
        self,
        num_features: int = 500,
        mr_size: float = 6.0,
        scale_pyr_module: Optional[nn.Module] = None,
        resp_module: Optional[nn.Module] = None,
        nms_module: Optional[nn.Module] = None,
        ori_module: Optional[nn.Module] = None,
        aff_module: Optional[nn.Module] = None,
        minima_are_also_good: bool = False,
        scale_space_response: bool = False,
    ) -> None:
        super().__init__()
        self.mr_size = mr_size
        self.num_features = num_features
        if scale_pyr_module is None:
            scale_pyr_module = ScalePyramid(3, 1.6, 15)
        self.scale_pyr = scale_pyr_module
        if resp_module is None:
            resp_module = BlobHessian()
        self.resp = resp_module
        if nms_module is None:
            nms_module = IterativeQuadInterp3d()
        self.nms = nms_module
        if ori_module is None:
            ori_module = PassLAF()
        self.ori = ori_module
        if aff_module is None:
            aff_module = PassLAF()
        self.aff = aff_module
        self.minima_are_also_good = minima_are_also_good
        self.scale_space_response = scale_space_response

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_features={self.num_features}, "
            f"mr_size={self.mr_size}, "
            f"scale_pyr={self.scale_pyr.__repr__()}, "
            f"resp={self.resp.__repr__()}, "
            f"nms={self.nms.__repr__()}, "
            f"ori={self.ori.__repr__()}, "
            f"aff={self.aff.__repr__()})"
        )

    def detect(
        self, img: torch.Tensor, num_feats: int, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = img.device
        dtype = img.dtype
        sp, sigmas, _ = self.scale_pyr(img)
        all_responses: list[torch.Tensor] = []
        all_lafs: list[torch.Tensor] = []
        px_size = 0.5 if self.scale_pyr.double_image else 1.0

        for oct_idx, octave in enumerate(sp):
            sigmas_oct = sigmas[oct_idx]  # (B, n_levels + extra_levels)
            B, CH, L, H, W = octave.size()

            # --- compute response ---
            if self.scale_space_response:
                # Response function consumes the full scale-space octave,
                # e.g. BlobDoG returns (B, 1, L-1, H, W).
                oct_resp = self.resp(octave, sigmas_oct.view(-1))
            else:
                # Response applied independently per level, then stacked.
                oct_resp = self.resp(
                    octave.permute(0, 2, 1, 3, 4).reshape(B * L, CH, H, W),
                    sigmas_oct.view(-1),
                ).view(B, L, CH, H, W)
                # (B, CH, L, H, W) — channel-first for 3-D NMS
                oct_resp = oct_resp.permute(0, 2, 1, 3, 4)

            if mask is not None:
                oct_resp = _create_octave_mask(mask, list(oct_resp.shape)) * oct_resp

            # --- NMS + subpixel refinement ---
            coord_max, response_max = self.nms(oct_resp)

            if self.minima_are_also_good:
                coord_min, response_min = self.nms(-oct_resp)
                take_min = response_min > response_max
                response_max = torch.where(take_min, response_min, response_max)
                coord_max = torch.where(take_min.unsqueeze(2), coord_min, coord_max)

            # --- flatten + top-k per octave ---
            # coord_max: (B, CH, 3, D, H, W)  — ordering: [scale_idx, x, y]
            # response_max: (B, CH, D, H, W)
            responses_flat = response_max.view(B, -1)  # (B, N_all)
            coords_flat = coord_max.view(B, 3, -1).permute(0, 2, 1)  # (B, N_all, 3)

            if responses_flat.shape[1] > num_feats:
                resp_flat_best, idxs = torch.topk(responses_flat, k=num_feats, dim=1)
                coords_best = coords_flat.gather(1, idxs.unsqueeze(-1).expand(-1, -1, 3))
            else:
                resp_flat_best = responses_flat
                coords_best = coords_flat

            B, N = resp_flat_best.shape

            # --- scale index → actual sigma via interpolation ---
            coords_best = _scale_index_to_scale(coords_best, sigmas_oct)

            # --- build local affine frames ---
            rotmat = torch.eye(2, dtype=dtype, device=dev).view(1, 1, 2, 2)
            current_lafs = torch.cat(
                [
                    self.mr_size * coords_best[:, :, 0].view(B, N, 1, 1) * rotmat,
                    coords_best[:, :, 1:3].view(B, N, 2, 1),
                ],
                dim=3,
            )

            # Zero out LAFs whose support patches touch the image boundary
            good_mask = laf_is_inside_image(current_lafs, octave[:, 0])
            resp_flat_best = resp_flat_best * good_mask.to(dev, dtype)

            # Rescale LAFs to input-image pixel coordinates
            current_lafs = current_lafs * px_size

            all_responses.append(resp_flat_best)
            all_lafs.append(current_lafs)
            px_size *= 2.0

        # --- final top-k across all octaves ---
        responses = torch.cat(all_responses, dim=1)
        lafs = torch.cat(all_lafs, dim=1)
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = lafs.gather(1, idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 3))
        return responses, lafs

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect local affine frames in three stages: location/scale, affine shape, orientation.

        Args:
            img: input image with shape :math:`(B, C, H, W)`.
            mask: optional spatial mask with the same shape as ``img``.
                Non-zero pixels indicate regions where detection is allowed.

        Returns:
            - Local affine frames with shape :math:`(B, N, 2, 3)`.
            - Response values with shape :math:`(B, N, 1)`.
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
    ) -> None:
        super().__init__()
        self.model = model
        if config is None:
            config = get_default_detector_config()
        # Load extraction configuration
        self.num_pyramid_levels = config["pyramid_levels"]
        self.num_upscale_levels = config["up_levels"]
        self.scale_factor_levels = config["scale_factor_levels"]
        self.mr_size = config["s_mult"]
        self.nms_size = config["nms_size"]
        self.nms = NonMaximaSuppression2d((self.nms_size, self.nms_size))
        self.num_features = num_features

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
        self, level_img: torch.Tensor, num_kp: int, factor: tuple[float, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        det_map = self.nms(self.remove_borders(self.model(level_img)))
        device = level_img.device
        dtype = level_img.dtype
        yx = det_map.nonzero()[:, 2:].t()
        scores = det_map[0, 0, yx[0], yx[1]]  # keynet supports only non-batched images

        scores_sorted, indices = torch.sort(scores, descending=True)

        indices = indices[torch.where(scores_sorted > 0.0)]
        yx = yx[:, indices[:num_kp]].t()
        current_kp_num = len(yx)
        xy_projected = yx.view(1, current_kp_num, 2).flip(2) * torch.tensor(factor, device=device, dtype=dtype)
        scale_factor = 0.5 * (factor[0] + factor[1])
        scale = scale_factor * self.mr_size * torch.ones(1, current_kp_num, 1, 1, device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(
            xy_projected, scale, torch.zeros(1, current_kp_num, 1, device=device, dtype=dtype)
        )
        return scores_sorted[:num_kp], lafs

    def detect(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute points per level
        num_features_per_level: list[float] = []
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
        all_responses: list[torch.Tensor] = []
        all_lafs: list[torch.Tensor] = []
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
                nf2 = [num_features_per_level[a] for a in range(0, idx_level + 1 + self.num_upscale_levels)]
                res_points = torch.tensor(nf2).sum().item()
                num_points_level = int(res_points)

            cur_scores, cur_lafs = self.detect_features_on_single_level(cur_img, num_points_level, factor)
            all_responses.append(cur_scores.view(1, -1))
            all_lafs.append(cur_lafs)
        responses = torch.cat(all_responses, 1)
        lafs = torch.cat(all_lafs, 1)
        if lafs.shape[1] > self.num_features:
            responses, idxs = torch.topk(responses, k=self.num_features, dim=1)
            lafs = lafs.gather(1, idxs[..., None, None].expand(-1, -1, 2, 3))
        return responses, lafs

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect local affine frames in three stages: location/scale, affine shape, orientation.

        Args:
            img: input image with shape :math:`(1, C, H, W)`.  KeyNetDetector does not support
                batch processing because the number of detections varies per image.
            mask: optional spatial mask with the same shape as ``img``.

        Returns:
            - Local affine frames with shape :math:`(1, N, 2, 3)`.
            - Response values with shape :math:`(1, N, 1)`.
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
