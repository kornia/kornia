from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SOLD2Net
from .sold2_detector import LineSegmentDetectionModule, line_map_to_segments, super_nms

urls: Dict[str, str] = {}
urls["wireframe"] = "https://www.polybox.ethz.ch/index.php/s/blOrW89gqSLoHOk/download"


default_cfg = {
    'backbone_cfg': {
        'input_channel': 1,
        'depth': 4,
        'num_stacks': 2,
        'num_blocks': 1,
        'num_classes': 5,
    },
    'use_descriptor': True,
    'grid_size': 8,
    'keep_border_valid': True,
    'detection_thresh': 0.0153846,  # = 1/65: threshold of junction detection
    'max_num_junctions': 500,  # maximum number of junctions per image
    'line_detector_cfg': {
        'detect_thresh': 0.5,
        'num_samples': 64,
        'inlier_thresh': 0.99,
        'use_candidate_suppression': True,
        'nms_dist_tolerance': 3.,
        'use_heatmap_refinement': True,
        'heatmap_refine_cfg': {
            'mode': "local",
            'ratio': 0.2,
            'valid_thresh': 0.001,
            'num_blocks': 20,
            'overlap_ratio': 0.5,
        },
        'use_junction_refinement': True,
        'junction_refine_cfg': {
            'num_perturbs': 9,
            'perturb_interval': 0.25,
        },
    },
    'line_matcher_cfg': {
        'cross_check': True,
        'num_samples': 5,
        'min_dist_pts': 8,
        'top_k_candidates': 10,
        'grid_size': 4,
    },
}


class SOLD2(nn.Module):
    r"""Module, which detects and describe line segments in an image.

    This is based on the original code from the paper "SOLD²: Self-supervised
    Occlusion-aware Line Detector and Descriptor". See :cite:`SOLD22021` for more details.

    Args:
        config: Dict specifying parameters. None will load the default parameters,
            which are tuned for images in the range 400~800 px.
        pretrained: If True, download and set pretrained weights to the model.

    Returns:
        The raw junction and line heatmaps, the semi-dense descriptor map,
        as well as the list of detected line segments (ij coordinates convention).

    Example:
        >>> images = torch.rand(2, 1, 512, 512)
        >>> sold2 = SOLD2()
        >>> outputs = sold2(images)
        >>> line_seg1 = outputs["line_segments"][0].detach().cpu().numpy()
        >>> line_seg2 = outputs["line_segments"][1].detach().cpu().numpy()
        >>> desc1 = outputs["dense_desc"][0]
        >>> desc2 = outputs["dense_desc"][1]
        >>> matches = sold2.match(line_seg1, line_seg2, desc1[None], desc2[None])
    """

    def __init__(self, pretrained: bool = True, config: Dict = None):
        super().__init__()
        # Initialize some parameters
        self.config = default_cfg if config is None else config
        self.grid_size = self.config["grid_size"]
        self.junc_detect_thresh = self.config.get("detection_thresh", 1 / 65)
        self.max_num_junctions = self.config.get("max_num_junctions", 500)

        # Load the pre-trained model
        self.model = SOLD2Net(self.config)
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls["wireframe"], map_location=lambda storage, loc: storage)
            state_dict = self.adapt_state_dict(pretrained_dict['model_state_dict'])
            self.model.load_state_dict(state_dict)
        self.eval()

        # Initialize the line detector
        self.line_detector_cfg = self.config["line_detector_cfg"]
        self.line_detector = LineSegmentDetectionModule(**self.config["line_detector_cfg"])

        # Initialize the line matcher
        self.line_matcher = WunschLineMatcher(**self.config["line_matcher_cfg"])

    def forward(self, img: torch.Tensor) -> Dict:
        """
        Args:
            img: batched images with shape :math:`(N, 1, H, W)`.

        :return:
            - ``line_segments``: list of line segments in each of the N images :math:`List[(Nlines, 2, 2)]`.
            - ``raw_junc_heatmap``: raw junction heatmap of shape :math:`(N, H, W)`.
            - ``raw_line_heatmap``: raw line heatmap of shape :math:`(N, H, W)`.
            - ``dense_desc``: the semi-dense descriptor map of shape :math:`(N, 128, H/4, W/4)`.

        :rtype: dict
        """
        if ((not len(img.shape) == 4) or (not isinstance(img, torch.Tensor))):
            raise ValueError("The input image should be a 4D torch tensor.")
        device = img.device
        outputs = {}

        # Forward pass of the CNN backbone
        net_outputs = self.model(img)

        # Descriptor map
        outputs["dense_desc"] = net_outputs["descriptors"]

        # Junction NMS
        outputs["raw_junc_heatmap"] = net_outputs["junctions"]
        junc_pred_nms = super_nms(
            net_outputs["junctions"].unsqueeze(-1).detach().cpu().numpy(),
            self.grid_size, self.junc_detect_thresh, self.max_num_junctions)

        # Retrieve the line heatmap
        heatmap = net_outputs["heatmap"][:, 0]
        outputs["raw_line_heatmap"] = heatmap
        heatmap_np = heatmap.detach().cpu().numpy()

        # Loop through all images
        lines = []
        for i, curr_junc_pred_nms in enumerate(junc_pred_nms):
            junctions = np.stack(np.where(curr_junc_pred_nms), axis=-1)
            # Run the line detector
            line_map, junctions, _ = self.line_detector.detect(
                junctions, heatmap_np[i], device=device)
            if isinstance(line_map, torch.Tensor):
                line_map = line_map.cpu().numpy()
            if isinstance(junctions, torch.Tensor):
                junctions = junctions.cpu().numpy()
            lines.append(torch.from_numpy(line_map_to_segments(junctions, line_map)))
        outputs["line_segments"] = lines

        return outputs

    def match(self, line_seg1: np.ndarray, line_seg2: np.ndarray,
              desc1: torch.Tensor, desc2: torch.Tensor) -> np.ndarray:
        """Find the best matches between two sets of line segments and their corresponding descriptors.

        Args:
            line_seg1, line_seg2: list of line segments in two images, with shape [num_lines, 2, 2].
            desc1, desc2: semi-dense descriptor maps of the images, with shape [1, 128, H/4, W/4].
        Returns:
            A np.array of size [num_lines1] indicating the index in line_seg2 of the matched line,
            for each line in line_seg1. -1 means that the line is not matched.
        """
        return self.line_matcher(line_seg1, line_seg2, desc1, desc2)

    def adapt_state_dict(self, state_dict):
        del state_dict["w_junc"]
        del state_dict["w_heatmap"]
        del state_dict["w_desc"]
        state_dict["heatmap_decoder.conv_block_lst.2.0.weight"] = state_dict["heatmap_decoder.conv_block_lst.2.weight"]
        state_dict["heatmap_decoder.conv_block_lst.2.0.bias"] = state_dict["heatmap_decoder.conv_block_lst.2.bias"]
        del state_dict["heatmap_decoder.conv_block_lst.2.weight"]
        del state_dict["heatmap_decoder.conv_block_lst.2.bias"]
        return state_dict


class WunschLineMatcher:
    """Class matching two sets of line segments with the Needleman-Wunsch algorithm."""
    def __init__(self, cross_check=True, num_samples=10, min_dist_pts=8,
                 top_k_candidates=10, grid_size=8, line_score=False):
        self.cross_check = cross_check
        self.num_samples = num_samples
        self.min_dist_pts = min_dist_pts
        self.top_k_candidates = top_k_candidates
        self.grid_size = grid_size
        self.line_score = line_score  # True to compute saliency on a line

    def __call__(self, line_seg1, line_seg2, desc1, desc2):
        """Find the best matches between two sets of line segments and their corresponding descriptors."""
        img_size1 = (desc1.shape[2] * self.grid_size,
                     desc1.shape[3] * self.grid_size)
        img_size2 = (desc2.shape[2] * self.grid_size,
                     desc2.shape[3] * self.grid_size)
        device = desc1.device

        # Default case when an image has no lines
        if len(line_seg1) == 0:
            return np.empty((0), dtype=int)
        if len(line_seg2) == 0:
            return -np.ones(len(line_seg1), dtype=int)

        # Sample points regularly along each line
        line_points1, valid_points1 = self.sample_line_points(line_seg1)
        line_points2, valid_points2 = self.sample_line_points(line_seg2)
        line_points1 = torch.tensor(line_points1.reshape(-1, 2),
                                    dtype=torch.float, device=device)
        line_points2 = torch.tensor(line_points2.reshape(-1, 2),
                                    dtype=torch.float, device=device)

        # Extract the descriptors for each point
        grid1 = keypoints_to_grid(line_points1, img_size1)
        grid2 = keypoints_to_grid(line_points2, img_size2)
        desc1 = F.normalize(F.grid_sample(desc1, grid1, align_corners=False)[0, :, :, 0], dim=0)
        desc2 = F.normalize(F.grid_sample(desc2, grid2, align_corners=False)[0, :, :, 0], dim=0)

        # Precompute the distance between line points for every pair of lines
        # Assign a score of -1 for invalid points
        scores = (desc1.t() @ desc2).cpu().numpy()
        scores[~valid_points1.flatten()] = -1
        scores[:, ~valid_points2.flatten()] = -1
        scores = scores.reshape(len(line_seg1), self.num_samples,
                                len(line_seg2), self.num_samples)
        scores = scores.transpose(0, 2, 1, 3)
        # scores.shape = (n_lines1, n_lines2, num_samples, num_samples)

        # Pre-filter the line candidates and find the best match for each line
        matches = self.filter_and_match_lines(scores)

        # [Optionally] filter matches with mutual nearest neighbor filtering
        if self.cross_check:
            matches2 = self.filter_and_match_lines(
                scores.transpose(1, 0, 3, 2))
            mutual = matches2[matches] == np.arange(len(line_seg1))
            matches[~mutual] = -1

        return matches

    def sample_line_points(self, line_seg):
        """Regularly sample points along each line segments, with a minimal distance between each point.

        Pad the remaining points.
        Inputs:
            line_seg: an Nx2x2 np.array.
        Outputs:
            line_points: an Nxnum_samplesx2 np.array.
            valid_points: a boolean Nxnum_samples np.array.
        """
        num_lines = len(line_seg)
        line_lengths = np.linalg.norm(line_seg[:, 0] - line_seg[:, 1], axis=1)

        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples_lst = np.clip(line_lengths // self.min_dist_pts,
                                  2, self.num_samples)
        line_points = np.empty((num_lines, self.num_samples, 2), dtype=float)
        valid_points = np.empty((num_lines, self.num_samples), dtype=bool)
        for n in np.arange(2, self.num_samples + 1):
            # Consider all lines where we can fit up to n points
            cur_mask = num_samples_lst == n
            cur_line_seg = line_seg[cur_mask]
            line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                        cur_line_seg[:, 1, 0],
                                        n, axis=-1)
            line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                        cur_line_seg[:, 1, 1],
                                        n, axis=-1)
            cur_line_points = np.stack([line_points_x, line_points_y], axis=-1)

            # Pad
            cur_num_lines = len(cur_line_seg)
            cur_valid_points = np.ones((cur_num_lines, self.num_samples),
                                       dtype=bool)
            cur_valid_points[:, n:] = False
            cur_line_points = np.concatenate([
                cur_line_points,
                np.zeros((cur_num_lines, self.num_samples - n, 2), dtype=float)],
                axis=1)

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points

    def filter_and_match_lines(self, scores):
        """Use the scores to keep the top k best lines, compute the Needleman- Wunsch algorithm on each candidate
        pairs, and keep the highest score.

        Inputs:
            scores: a (N, M, n, n) np.array containing the pairwise scores
                    of the elements to match.
        Outputs:
            matches: a (N) np.array containing the indices of the best match
        """
        # Pre-filter the pairs and keep the top k best candidate lines
        line_scores1 = scores.max(3)
        valid_scores1 = line_scores1 != -1
        line_scores1 = ((line_scores1 * valid_scores1).sum(2)
                        / valid_scores1.sum(2))
        line_scores2 = scores.max(2)
        valid_scores2 = line_scores2 != -1
        line_scores2 = ((line_scores2 * valid_scores2).sum(2)
                        / valid_scores2.sum(2))
        line_scores = (line_scores1 + line_scores2) / 2
        topk_lines = np.argsort(line_scores,
                                axis=1)[:, -self.top_k_candidates:]
        # topk_lines.shape = (n_lines1, top_k_candidates)
        top_scores = np.take_along_axis(scores, topk_lines[:, :, None, None],
                                        axis=1)

        # Consider the reversed line segments as well
        top_scores = np.concatenate([top_scores, top_scores[..., ::-1]],
                                    axis=1)

        # Compute the line distance matrix with Needleman-Wunsch algo and
        # retrieve the closest line neighbor
        n_lines1, top2k, n, m = top_scores.shape
        top_scores = top_scores.reshape((n_lines1 * top2k, n, m))
        nw_scores = self.needleman_wunsch(top_scores)
        nw_scores = nw_scores.reshape(n_lines1, top2k)
        matches = np.mod(np.argmax(nw_scores, axis=1), top2k // 2)
        matches = topk_lines[np.arange(n_lines1), matches]
        return matches

    def needleman_wunsch(self, scores):
        """Batched implementation of the Needleman-Wunsch algorithm.

        The cost of the InDel operation is set to 0 by subtracting the gap
        penalty to the scores.
        Inputs:
            scores: a (B, N, M) np.array containing the pairwise scores
                    of the elements to match.
        """
        b, n, m = scores.shape

        # Recalibrate the scores to get a gap score of 0
        gap = 0.1
        nw_scores = scores - gap

        # Run the dynamic programming algorithm
        nw_grid = np.zeros((b, n + 1, m + 1), dtype=float)
        for i in range(n):
            for j in range(m):
                nw_grid[:, i + 1, j + 1] = np.maximum(
                    np.maximum(nw_grid[:, i + 1, j], nw_grid[:, i, j + 1]),
                    nw_grid[:, i, j] + nw_scores[:, i, j])

        return nw_grid[:, -1, -1]


def keypoints_to_grid(keypoints: torch.Tensor, img_size: tuple) -> torch.Tensor:
    """Convert a list of keypoints into a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate.

    Args:
        keypoints: a tensor [N, 2] or batched tensor [B, N, 2] of N keypoints (ij coordinates convention).
        img_size: the original image size (H, W)
    """
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2)
    return grid_points
