import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from kornia.core import Module, Tensor, concatenate, sin, stack, tensor, where, zeros
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.feature.sold2.structures import DetectorCfg, HeatMapRefineCfg, JunctionRefineCfg, LineDetectorCfg
from kornia.geometry.bbox import nms
from kornia.utils import dataclass_to_dict, dict_to_dataclass, torch_meshgrid

from .backbones import SOLD2Net

urls: Dict[str, str] = {}
urls["wireframe"] = "https://www.polybox.ethz.ch/index.php/s/blOrW89gqSLoHOk/download"


class SOLD2_detector(Module):
    r"""Module, which detects line segments in an image.

    This is based on the original code from the paper "SOLD²: Self-supervised
    Occlusion-aware Line Detector and Descriptor". See :cite:`SOLD22021` for more details.

    Args:
        config (DetectorCfg): Configuration object containing all parameters. None will load the default parameters,
            which are tuned for images in the range 400~800 px. Using a dataclass ensures type safety and clearer
            parameter management.
        pretrained (bool): If True, download and set pretrained weights to the model.

    Returns:
        The raw junction and line heatmaps, as well as the list of detected line segments (ij coordinates convention).

    Example:
        >>> img = torch.rand(1, 1, 512, 512)
        >>> sold2_detector = SOLD2_detector()
        >>> line_segments = sold2_detector(img)["line_segments"]
    """

    def __init__(self, pretrained: bool = True, config: Optional[DetectorCfg] = None) -> None:
        if isinstance(config, dict):
            warnings.warn(
                "Usage of config as a plain dictionary is deprecated in favor of"
                "`kornia.features.sold2.structures.DetectorCfg`. The support of plain"
                "dictionaries as config will be removed in kornia v0.8.0 (December 2024).",
                category=DeprecationWarning,
                stacklevel=2,
            )
            config = dict_to_dataclass(config, DetectorCfg)
        super().__init__()
        # Initialize some parameters
        self.config = config if config is not None else DetectorCfg()
        self.grid_size = self.config.grid_size
        self.junc_detect_thresh = self.config.detection_thresh
        self.max_num_junctions = self.config.max_num_junctions

        # Load the pre-trained model
        self.model = SOLD2Net(dataclass_to_dict(self.config))

        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls["wireframe"], map_location=torch.device("cpu"))
            state_dict = self.adapt_state_dict(pretrained_dict["model_state_dict"])
            self.model.load_state_dict(state_dict)
        self.eval()

        # Initialize the line detector with a configuration from the dataclass
        self.line_detector = LineSegmentDetectionModule(self.config.line_detector_cfg)

    def adapt_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        del state_dict["w_junc"]
        del state_dict["w_heatmap"]
        del state_dict["w_desc"]
        state_dict["heatmap_decoder.conv_block_lst.2.0.weight"] = state_dict["heatmap_decoder.conv_block_lst.2.weight"]
        state_dict["heatmap_decoder.conv_block_lst.2.0.bias"] = state_dict["heatmap_decoder.conv_block_lst.2.bias"]
        del state_dict["heatmap_decoder.conv_block_lst.2.weight"]
        del state_dict["heatmap_decoder.conv_block_lst.2.bias"]
        return state_dict

    def forward(self, img: Tensor) -> Dict[str, Any]:
        """
        Args:
            img: batched images with shape :math:`(B, 1, H, W)`.

        Return:
            - ``line_segments``: list of N line segments in each of the B images :math:`List[(N, 2, 2)]`.
            - ``junction_heatmap``: raw junction heatmap of shape :math:`(B, H, W)`.
            - ``line_heatmap``: raw line heatmap of shape :math:`(B, H, W)`.
        """
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        outputs = {}

        # Forward pass of the CNN backbone
        net_outputs = self.model(img)
        outputs["junction_heatmap"] = net_outputs["junctions"]
        outputs["line_heatmap"] = net_outputs["heatmap"]

        # Loop through all images
        lines = []
        for junc_prob, heatmap in zip(net_outputs["junctions"], net_outputs["heatmap"]):
            # Get the junctions
            junctions = prob_to_junctions(junc_prob, self.grid_size, self.junc_detect_thresh, self.max_num_junctions)

            # Run the line detector
            line_map, junctions, _ = self.line_detector.detect(junctions, heatmap)
            lines.append(line_map_to_segments(junctions, line_map))
        outputs["line_segments"] = lines

        return outputs


class LineSegmentDetectionModule:
    r"""Module extracting line segments from junctions and line heatmaps.

    Args:
        config (LineDetectorCfg): Configuration dataclass containing all settings required for line segment detection.
            - detect_thresh (float): Probability threshold for mean activation (0. ~ 1.).
            - num_samples (int): Number of sampling locations along the line segments.
            - inlier_thresh (float): Minimum inlier ratio to satisfy (0. ~ 1.) => 0. means no threshold.
            - heatmap_low_thresh (float): Lowest threshold for pixel considered as a candidate in junction recovery.
            - heatmap_high_thresh (float): Higher threshold for NMS in junction recovery.
            - max_local_patch_radius (float): Maximum patch to be considered in local maximum search.
            - lambda_radius (float): Lambda factor in linear local maximum search formulation.
            - use_candidate_suppression (bool): Apply candidate suppression to break long segments into sub-segments.
            - nms_dist_tolerance (float): Distance tolerance for NMS. Decides whether the junctions are on the line.
            - use_heatmap_refinement (bool): Whether to use heatmap refinement methods.
            - heatmap_refine_cfg: Configuration for heatmap refinement methods.
            - use_junction_refinement (bool): Whether to use junction refinement methods.
            - junction_refine_cfg: Configuration for junction refinement methods.

    Example:
        >>> config = LineDetectorCfg(detect_thresh=0.6, use_heatmap_refinement=True)
        >>> module = LineSegmentDetectionModule(config)
        >>> junctions, heatmap = torch.rand(10, 2), torch.rand(256, 256)
        >>> line_map, junctions, _ = module.detect(junctions, heatmap)
    """

    def __init__(self, config: LineDetectorCfg = LineDetectorCfg()) -> None:
        # Load LineDetectorCfg
        self.config = config

        # Line detection parameters
        self.detect_thresh = self.config.detect_thresh
        # self.detect_thresh = detect_thresh

        # Line sampling parameters
        self.num_samples = self.config.num_samples
        self.inlier_thresh = self.config.inlier_thresh
        self.local_patch_radius = self.config.max_local_patch_radius
        self.lambda_radius = self.config.lambda_radius

        # Detecting junctions on the boundary parameters
        self.low_thresh = self.config.heatmap_low_thresh
        self.high_thresh = self.config.heatmap_high_thresh

        # Pre-compute the linspace sampler
        self.torch_sampler = torch.linspace(0, 1, self.num_samples)

        # Long line segment suppression configuration
        self.use_candidate_suppression = self.config.use_candidate_suppression
        self.nms_dist_tolerance = self.config.nms_dist_tolerance

        # Heatmap refinement configuration
        self.use_heatmap_refinement = self.config.use_heatmap_refinement
        self.heatmap_refine_cfg = self.config.heatmap_refine_cfg
        if self.use_heatmap_refinement and self.heatmap_refine_cfg is None:
            raise ValueError("[Error] Missing heatmap refinement config.")

        # Junction refinement configuration
        self.use_junction_refinement = self.config.use_junction_refinement
        self.junction_refine_cfg = self.config.junction_refine_cfg
        if self.use_junction_refinement and self.junction_refine_cfg is None:
            raise ValueError("[Error] Missing junction refinement config.")

    def detect(self, junctions: Tensor, heatmap: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Main function performing line segment detection."""
        KORNIA_CHECK_SHAPE(heatmap, ["H", "W"])
        H, W = heatmap.shape
        device = junctions.device

        # Perform the heatmap refinement
        if self.use_heatmap_refinement and isinstance(self.heatmap_refine_cfg, HeatMapRefineCfg):
            if self.heatmap_refine_cfg.mode == "global":
                heatmap = self.refine_heatmap(
                    heatmap, self.heatmap_refine_cfg.ratio, self.heatmap_refine_cfg.valid_thresh
                )
            elif self.heatmap_refine_cfg.mode == "local":
                heatmap = self.refine_heatmap_local(
                    heatmap,
                    self.heatmap_refine_cfg.num_blocks,
                    self.heatmap_refine_cfg.overlap_ratio,
                    self.heatmap_refine_cfg.ratio,
                    self.heatmap_refine_cfg.valid_thresh,
                )

        # Initialize empty line map
        num_junctions = len(junctions)
        line_map_pred = zeros([num_junctions, num_junctions], device=device, dtype=torch.int32)

        # Stop if there are not enough junctions
        if num_junctions < 2:
            return line_map_pred, junctions, heatmap

        # Generate the candidate map
        candidate_map = torch.triu(
            torch.ones([num_junctions, num_junctions], device=device, dtype=torch.int32), diagonal=1
        )

        # Optionally perform candidate filtering
        if self.use_candidate_suppression:
            candidate_map = self.candidate_suppression(junctions, candidate_map)

        # Fetch the candidates
        candidate_indices = where(candidate_map)
        candidate_index_map = concatenate([candidate_indices[0][..., None], candidate_indices[1][..., None]], -1)

        # Get the corresponding start and end junctions
        candidate_junc_start = junctions[candidate_index_map[:, 0]]
        candidate_junc_end = junctions[candidate_index_map[:, 1]]

        # Get the sampling locations (N x 64)
        sampler = self.torch_sampler.to(device)[None]
        cand_samples_h = candidate_junc_start[:, 0:1] * sampler + candidate_junc_end[:, 0:1] * (1 - sampler)
        cand_samples_w = candidate_junc_start[:, 1:2] * sampler + candidate_junc_end[:, 1:2] * (1 - sampler)

        # Clip to image boundary
        cand_h = torch.clamp(cand_samples_h, min=0, max=H - 1)
        cand_w = torch.clamp(cand_samples_w, min=0, max=W - 1)

        # [Local maximum search]
        # Compute normalized segment lengths
        segments_length = torch.sqrt(
            torch.sum((candidate_junc_start.to(torch.float32) - candidate_junc_end.to(torch.float32)) ** 2, dim=-1)
        )
        normalized_seg_length = segments_length / (((H**2) + (W**2)) ** 0.5)

        # Perform local max search
        num_cand = len(cand_h)
        group_size = 10000
        if num_cand > group_size:
            num_iter = math.ceil(num_cand / group_size)
            sampled_feat_lst = []
            for iter_idx in range(num_iter):
                if not iter_idx == num_iter - 1:
                    cand_h_ = cand_h[iter_idx * group_size : (iter_idx + 1) * group_size, :]
                    cand_w_ = cand_w[iter_idx * group_size : (iter_idx + 1) * group_size, :]
                    normalized_seg_length_ = normalized_seg_length[iter_idx * group_size : (iter_idx + 1) * group_size]
                else:
                    cand_h_ = cand_h[iter_idx * group_size :, :]
                    cand_w_ = cand_w[iter_idx * group_size :, :]
                    normalized_seg_length_ = normalized_seg_length[iter_idx * group_size :]
                sampled_feat_ = self.detect_local_max(heatmap, cand_h_, cand_w_, H, W, normalized_seg_length_, device)
                sampled_feat_lst.append(sampled_feat_)
            sampled_feat = concatenate(sampled_feat_lst, 0)
        else:
            sampled_feat = self.detect_local_max(heatmap, cand_h, cand_w, H, W, normalized_seg_length, device)

        # [Simple threshold detection]
        # detection_results is a mask over all candidates
        detection_results = torch.mean(sampled_feat, dim=-1) > self.detect_thresh

        # [Inlier threshold detection]
        if self.inlier_thresh > 0:
            inlier_ratio = torch.sum(sampled_feat > self.detect_thresh, dim=-1).to(heatmap.dtype) / self.num_samples
            detection_results_inlier = inlier_ratio >= self.inlier_thresh
            detection_results = detection_results * detection_results_inlier

        # Convert detection results back to line_map_pred
        detected_junc_indexes = candidate_index_map[detection_results]
        line_map_pred[detected_junc_indexes[:, 0], detected_junc_indexes[:, 1]] = 1
        line_map_pred[detected_junc_indexes[:, 1], detected_junc_indexes[:, 0]] = 1

        # [Junction refinement]
        if self.use_junction_refinement and len(detected_junc_indexes) > 0:
            junctions, line_map_pred = self.refine_junction_perturb(junctions, line_map_pred, heatmap, H, W, device)

        return line_map_pred, junctions, heatmap

    def refine_heatmap(self, heatmap: Tensor, ratio: float = 0.2, valid_thresh: float = 1e-2) -> Tensor:
        """Global heatmap refinement method."""
        # Grab the top 10% values
        heatmap_values = heatmap[heatmap > valid_thresh]
        sorted_values = torch.sort(heatmap_values, descending=True)[0]
        top10_len = math.ceil(sorted_values.shape[0] * ratio)
        max20 = torch.mean(sorted_values[:top10_len])
        heatmap = torch.clamp(heatmap / max20, min=0.0, max=1.0)
        return heatmap

    def refine_heatmap_local(
        self,
        heatmap: Tensor,
        num_blocks: int = 5,
        overlap_ratio: float = 0.5,
        ratio: float = 0.2,
        valid_thresh: float = 2e-3,
    ) -> Tensor:
        """Local heatmap refinement method."""
        # Get the shape of the heatmap
        H, W = heatmap.shape
        increase_ratio = 1 - overlap_ratio
        h_block = round(H / (1 + (num_blocks - 1) * increase_ratio))
        w_block = round(W / (1 + (num_blocks - 1) * increase_ratio))

        # Iterate through each block
        count_map = zeros(heatmap.shape, dtype=torch.int, device=heatmap.device)
        heatmap_output = zeros(heatmap.shape, dtype=torch.float, device=heatmap.device)
        for h_idx in range(num_blocks):
            for w_idx in range(num_blocks):
                # Fetch the heatmap
                h_start = round(h_idx * h_block * increase_ratio)
                w_start = round(w_idx * w_block * increase_ratio)
                h_end = h_start + h_block if h_idx < num_blocks - 1 else H
                w_end = w_start + w_block if w_idx < num_blocks - 1 else W

                subheatmap = heatmap[h_start:h_end, w_start:w_end]
                if subheatmap.max() > valid_thresh:
                    subheatmap = self.refine_heatmap(subheatmap, ratio, valid_thresh=valid_thresh)

                # Aggregate it to the final heatmap
                heatmap_output[h_start:h_end, w_start:w_end] += subheatmap
                count_map[h_start:h_end, w_start:w_end] += 1
        heatmap_output = torch.clamp(heatmap_output / count_map, max=1.0, min=0.0)

        return heatmap_output

    def candidate_suppression(self, junctions: Tensor, candidate_map: Tensor) -> Tensor:
        """Suppress overlapping long lines in the candidate segments."""
        # Define the distance tolerance
        dist_tolerance = self.nms_dist_tolerance

        # Compute distance between junction pairs
        # (num_junc x 1 x 2) - (1 x num_junc x 2) => num_junc x num_junc map
        line_dist_map = torch.sum((torch.unsqueeze(junctions, dim=1) - junctions[None, ...]) ** 2, dim=-1) ** 0.5

        # Fetch all the "detected lines"
        seg_indexes = where(torch.triu(candidate_map, diagonal=1))
        start_point_idxs = seg_indexes[0]
        end_point_idxs = seg_indexes[1]
        start_points = junctions[start_point_idxs, :]
        end_points = junctions[end_point_idxs, :]

        # Fetch corresponding entries
        line_dists = line_dist_map[start_point_idxs, end_point_idxs]

        # Check whether they are on the line
        dir_vecs = (end_points - start_points) / torch.norm(end_points - start_points, dim=-1)[..., None]
        # Get the orthogonal distance
        cand_vecs = junctions[None, ...] - start_points.unsqueeze(dim=1)
        cand_vecs_norm = torch.norm(cand_vecs, dim=-1)
        # Check whether they are projected directly onto the segment
        proj = torch.einsum("bij,bjk->bik", cand_vecs, dir_vecs[..., None]) / line_dists[..., None, None]
        # proj is num_segs x num_junction x 1
        proj_mask = (proj >= 0) * (proj <= 1)
        cand_angles = torch.acos(
            torch.einsum("bij,bjk->bik", cand_vecs, dir_vecs[..., None]) / cand_vecs_norm[..., None]
        )
        cand_dists = cand_vecs_norm[..., None] * sin(cand_angles)
        junc_dist_mask = cand_dists <= dist_tolerance
        junc_mask = junc_dist_mask * proj_mask

        # Minus starting points
        num_segs = len(start_point_idxs)
        junc_counts = torch.sum(junc_mask, dim=[1, 2])
        junc_counts -= junc_mask[..., 0][torch.arange(0, num_segs), start_point_idxs].to(torch.int)
        junc_counts -= junc_mask[..., 0][torch.arange(0, num_segs), end_point_idxs].to(torch.int)

        # Get the invalid candidate mask
        final_mask = junc_counts > 0
        candidate_map[start_point_idxs[final_mask], end_point_idxs[final_mask]] = 0

        return candidate_map

    def refine_junction_perturb(
        self, junctions: Tensor, line_map: Tensor, heatmap: Tensor, H: int, W: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Refine the line endpoints in a similar way as in LSD."""
        # Fetch refinement parameters
        if not isinstance(self.junction_refine_cfg, JunctionRefineCfg):
            raise TypeError(
                "Expected to have dataclass of type JunctionRefineCfg for junction."
                f"Gotcha {type(self.junction_refine_cfg)}"
            )
        num_perturbs = self.junction_refine_cfg.num_perturbs
        perturb_interval = self.junction_refine_cfg.perturb_interval
        side_perturbs = (num_perturbs - 1) // 2

        # Fetch the 2D perturb mat
        perturb_vec = torch.arange(
            start=-perturb_interval * side_perturbs,
            end=perturb_interval * (side_perturbs + 1),
            step=perturb_interval,
            device=device,
        )

        h1_grid, w1_grid, h2_grid, w2_grid = torch_meshgrid(
            [perturb_vec, perturb_vec, perturb_vec, perturb_vec], indexing="ij"
        )

        perturb_tensor = concatenate(
            [h1_grid[..., None], w1_grid[..., None], h2_grid[..., None], w2_grid[..., None]], -1
        )
        perturb_tensor_flat = perturb_tensor.view(-1, 2, 2)

        # Fetch all the detected lines
        detected_seg_indexes = where(torch.triu(line_map, diagonal=1))
        start_points = junctions[detected_seg_indexes[0]]
        end_points = junctions[detected_seg_indexes[1]]
        line_segments = stack([start_points, end_points], 1)

        line_segment_candidates = line_segments.unsqueeze(dim=1) + perturb_tensor_flat[None]
        # Clip the boundaries
        line_segment_candidates[..., 0] = torch.clamp(line_segment_candidates[..., 0], min=0, max=H - 1)
        line_segment_candidates[..., 1] = torch.clamp(line_segment_candidates[..., 1], min=0, max=W - 1)

        # Iterate through all the segments
        refined_segment_lst = []
        num_segments = len(line_segments)
        for idx in range(num_segments):
            segment = line_segment_candidates[idx]
            # Get the corresponding start and end junctions
            candidate_junc_start = segment[:, 0]
            candidate_junc_end = segment[:, 1]

            # Get the sampling locations (N x 64)
            sampler = self.torch_sampler.to(device)[None]
            cand_samples_h = candidate_junc_start[:, 0:1] * sampler + candidate_junc_end[:, 0:1] * (1 - sampler)
            cand_samples_w = candidate_junc_start[:, 1:2] * sampler + candidate_junc_end[:, 1:2] * (1 - sampler)

            # Clip to image boundary
            cand_h = torch.clamp(cand_samples_h, min=0, max=H - 1)
            cand_w = torch.clamp(cand_samples_w, min=0, max=W - 1)

            # Perform bilinear sampling
            segment_feat = self.detect_bilinear(heatmap, cand_h, cand_w)
            segment_results = torch.mean(segment_feat, dim=-1)
            max_idx = torch.argmax(segment_results)
            refined_segment_lst.append(segment[max_idx][None])

        # Concatenate back to segments
        refined_segments = concatenate(refined_segment_lst, 0)

        # Convert back to junctions and line_map
        junctions_new = concatenate([refined_segments[:, 0, :], refined_segments[:, 1, :]], 0)
        junctions_new = torch.unique(junctions_new, dim=0)
        line_map_new = self.segments_to_line_map(junctions_new, refined_segments)

        return junctions_new, line_map_new

    def segments_to_line_map(self, junctions: Tensor, segments: Tensor) -> Tensor:
        """Convert the list of segments to line map."""
        # Create empty line map
        num_junctions = len(junctions)
        line_map = zeros([num_junctions, num_junctions], device=junctions.device)

        # Get the indices of paired junctions
        _, idx_junc1 = where(torch.all(junctions[None] == segments[:, None, 0], dim=2))
        _, idx_junc2 = where(torch.all(junctions[None] == segments[:, None, 1], dim=2))

        # Assign the labels
        line_map[idx_junc1, idx_junc2] = 1
        line_map[idx_junc2, idx_junc1] = 1

        return line_map

    def detect_bilinear(self, heatmap: Tensor, cand_h: Tensor, cand_w: Tensor) -> Tensor:
        """Detection by bilinear sampling."""
        # Get the floor and ceiling locations
        cand_h_floor = torch.floor(cand_h).to(torch.long)
        cand_h_ceil = torch.ceil(cand_h).to(torch.long)
        cand_w_floor = torch.floor(cand_w).to(torch.long)
        cand_w_ceil = torch.ceil(cand_w).to(torch.long)

        # Perform the bilinear sampling
        cand_samples_feat = (
            heatmap[cand_h_floor, cand_w_floor] * (cand_h_ceil - cand_h) * (cand_w_ceil - cand_w)
            + heatmap[cand_h_floor, cand_w_ceil] * (cand_h_ceil - cand_h) * (cand_w - cand_w_floor)
            + heatmap[cand_h_ceil, cand_w_floor] * (cand_h - cand_h_floor) * (cand_w_ceil - cand_w)
            + heatmap[cand_h_ceil, cand_w_ceil] * (cand_h - cand_h_floor) * (cand_w - cand_w_floor)
        )

        return cand_samples_feat

    def detect_local_max(
        self,
        heatmap: Tensor,
        cand_h: Tensor,
        cand_w: Tensor,
        H: int,
        W: int,
        normalized_seg_length: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Detection by local maximum search."""
        # Compute the distance threshold
        dist_thresh = 0.5 * (2**0.5) + self.lambda_radius * normalized_seg_length
        # Make it N x 64
        dist_thresh = torch.repeat_interleave(dist_thresh[..., None], self.num_samples, dim=-1)

        # Compute the candidate points
        cand_points = concatenate([cand_h[..., None], cand_w[..., None]], -1)
        cand_points_round = torch.round(cand_points)  # N x 64 x 2

        # Construct local patches 9x9 = 81
        patch_mask = zeros([int(2 * self.local_patch_radius + 1), int(2 * self.local_patch_radius + 1)], device=device)
        patch_center = tensor([[self.local_patch_radius, self.local_patch_radius]], device=device, dtype=torch.float32)
        H_patch_points, W_patch_points = where(patch_mask >= 0)
        patch_points = concatenate([H_patch_points[..., None], W_patch_points[..., None]], -1)
        # Fetch the circle region
        patch_center_dist = torch.sqrt(torch.sum((patch_points - patch_center) ** 2, dim=-1))
        patch_points = patch_points[patch_center_dist <= self.local_patch_radius, :]
        # Shift [0, 0] to the center
        patch_points = patch_points - self.local_patch_radius

        # Construct local patch mask
        patch_points_shifted = torch.unsqueeze(cand_points_round, dim=2) + patch_points[None, None]
        patch_dist = torch.sqrt(torch.sum((torch.unsqueeze(cand_points, dim=2) - patch_points_shifted) ** 2, dim=-1))
        patch_dist_mask = patch_dist < dist_thresh[..., None]

        # Get all points => num_points_center x num_patch_points x 2
        points_H = torch.clamp(patch_points_shifted[:, :, :, 0], min=0, max=H - 1).to(torch.long)
        points_W = torch.clamp(patch_points_shifted[:, :, :, 1], min=0, max=W - 1).to(torch.long)
        points = concatenate([points_H[..., None], points_W[..., None]], -1)

        # Sample the feature (N x 64 x 81)
        sampled_feat = heatmap[points[:, :, :, 0], points[:, :, :, 1]]
        # Filtering using the valid mask
        sampled_feat = sampled_feat * patch_dist_mask.to(torch.float32)
        if len(sampled_feat) == 0:
            sampled_feat_lmax = torch.empty(0, self.num_samples)
        else:
            sampled_feat_lmax = torch.max(sampled_feat, dim=-1)[0]

        return sampled_feat_lmax


def line_map_to_segments(junctions: Tensor, line_map: Tensor) -> Tensor:
    """Convert a junction connectivity map to a Nx2x2 tensor of segments."""
    junc_loc1, junc_loc2 = where(torch.triu(line_map))
    segments = stack([junctions[junc_loc1], junctions[junc_loc2]], 1)
    return segments


def prob_to_junctions(prob: Tensor, dist: float, prob_thresh: float = 0.01, top_k: int = 0) -> Tensor:
    """Extract junctions from a probability map, apply NMS, and extract the top k candidates."""
    # Extract the junctions
    junctions = stack(where(prob >= prob_thresh), -1).float()
    if len(junctions) == 0:
        return junctions

    # Perform NMS
    boxes = concatenate([junctions - dist / 2, junctions + dist / 2], 1)
    scores = prob[prob >= prob_thresh]
    remainings = nms(boxes, scores, 0.001)
    junctions = junctions[remainings]

    # Keep only the topk values
    if top_k > 0:
        k = min(len(junctions), top_k)
        junctions = junctions[:k]

    return junctions
