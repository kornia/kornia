from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor, concatenate, pad, stack
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.testing import KORNIA_CHECK_SHAPE
from kornia.utils import map_location_to_cpu

from .backbones import SOLD2Net
from .sold2_detector import LineSegmentDetectionModule, line_map_to_segments, prob_to_junctions

urls: Dict[str, str] = {}
urls["wireframe"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/sold2_wireframe.pth"


default_cfg: Dict = {
    'backbone_cfg': {'input_channel': 1, 'depth': 4, 'num_stacks': 2, 'num_blocks': 1, 'num_classes': 5},
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
        'nms_dist_tolerance': 3.0,
        'use_heatmap_refinement': True,
        'heatmap_refine_cfg': {
            'mode': "local",
            'ratio': 0.2,
            'valid_thresh': 0.001,
            'num_blocks': 20,
            'overlap_ratio': 0.5,
        },
        'use_junction_refinement': True,
        'junction_refine_cfg': {'num_perturbs': 9, 'perturb_interval': 0.25},
    },
    'line_matcher_cfg': {
        'cross_check': True,
        'num_samples': 5,
        'min_dist_pts': 8,
        'top_k_candidates': 10,
        'grid_size': 4,
    },
}


class SOLD2(Module):
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
        >>> line_seg1 = outputs["line_segments"][0]
        >>> line_seg2 = outputs["line_segments"][1]
        >>> desc1 = outputs["dense_desc"][0]
        >>> desc2 = outputs["dense_desc"][1]
        >>> matches = sold2.match(line_seg1, line_seg2, desc1[None], desc2[None])
    """

    def __init__(self, pretrained: bool = True, config: Optional[Dict] = None):
        super().__init__()
        # Initialize some parameters
        self.config = default_cfg if config is None else config
        self.grid_size = self.config["grid_size"]
        self.junc_detect_thresh = self.config.get("detection_thresh", 1 / 65)
        self.max_num_junctions = self.config.get("max_num_junctions", 500)

        # Load the pre-trained model
        self.model = SOLD2Net(self.config)
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls["wireframe"], map_location=map_location_to_cpu)
            state_dict = self.adapt_state_dict(pretrained_dict['model_state_dict'])
            self.model.load_state_dict(state_dict)
        self.eval()

        # Initialize the line detector
        self.line_detector_cfg = self.config["line_detector_cfg"]
        self.line_detector = LineSegmentDetectionModule(**self.config["line_detector_cfg"])

        # Initialize the line matcher
        self.line_matcher = WunschLineMatcher(**self.config["line_matcher_cfg"])

    def forward(self, img: Tensor) -> Dict:
        """
        Args:
            img: batched images with shape :math:`(B, 1, H, W)`.

        Return:
            - ``line_segments``: list of N line segments in each of the B images :math:`List[(N, 2, 2)]`.
            - ``junction_heatmap``: raw junction heatmap of shape :math:`(B, H, W)`.
            - ``line_heatmap``: raw line heatmap of shape :math:`(B, H, W)`.
            - ``dense_desc``: the semi-dense descriptor map of shape :math:`(B, 128, H/4, W/4)`.
        """
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        outputs = {}

        # Forward pass of the CNN backbone
        net_outputs = self.model(img)
        outputs["junction_heatmap"] = net_outputs["junctions"]
        outputs["line_heatmap"] = net_outputs["heatmap"]
        outputs["dense_desc"] = net_outputs["descriptors"]

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

    def match(self, line_seg1: Tensor, line_seg2: Tensor, desc1: Tensor, desc2: Tensor) -> Tensor:
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


class WunschLineMatcher(Module):
    """Class matching two sets of line segments with the Needleman-Wunsch algorithm.

    TODO: move it later in kornia.feature.matching
    """

    def __init__(
        self,
        cross_check: bool = True,
        num_samples: int = 10,
        min_dist_pts: int = 8,
        top_k_candidates: int = 10,
        grid_size: int = 8,
        line_score: bool = False,
    ):
        super().__init__()
        self.cross_check = cross_check
        self.num_samples = num_samples
        self.min_dist_pts = min_dist_pts
        self.top_k_candidates = top_k_candidates
        self.grid_size = grid_size
        self.line_score = line_score  # True to compute saliency on a line

    def forward(self, line_seg1: Tensor, line_seg2: Tensor, desc1: Tensor, desc2: Tensor) -> Tensor:
        """Find the best matches between two sets of line segments and their corresponding descriptors."""
        KORNIA_CHECK_SHAPE(line_seg1, ["N", "2", "2"])
        KORNIA_CHECK_SHAPE(line_seg2, ["N", "2", "2"])
        KORNIA_CHECK_SHAPE(desc1, ["B", "D", "H", "H"])
        KORNIA_CHECK_SHAPE(desc2, ["B", "D", "H", "H"])
        device = desc1.device
        img_size1 = (desc1.shape[2] * self.grid_size, desc1.shape[3] * self.grid_size)
        img_size2 = (desc2.shape[2] * self.grid_size, desc2.shape[3] * self.grid_size)

        # Default case when an image has no lines
        if len(line_seg1) == 0:
            return torch.empty(0, dtype=torch.int, device=device)
        if len(line_seg2) == 0:
            return -torch.ones(len(line_seg1), dtype=torch.int, device=device)

        # Sample points regularly along each line
        line_points1, valid_points1 = self.sample_line_points(line_seg1)
        line_points2, valid_points2 = self.sample_line_points(line_seg2)
        line_points1 = line_points1.reshape(-1, 2)
        line_points2 = line_points2.reshape(-1, 2)

        # Extract the descriptors for each point
        grid1 = keypoints_to_grid(line_points1, img_size1)
        grid2 = keypoints_to_grid(line_points2, img_size2)
        desc1 = F.normalize(F.grid_sample(desc1, grid1, align_corners=False)[0, :, :, 0], dim=0)
        desc2 = F.normalize(F.grid_sample(desc2, grid2, align_corners=False)[0, :, :, 0], dim=0)

        # Precompute the distance between line points for every pair of lines
        # Assign a score of -1 for invalid points
        scores = desc1.t() @ desc2
        scores[~valid_points1.flatten()] = -1
        scores[:, ~valid_points2.flatten()] = -1
        scores = scores.reshape(len(line_seg1), self.num_samples, len(line_seg2), self.num_samples)
        scores = scores.permute(0, 2, 1, 3)
        # scores.shape = (n_lines1, n_lines2, num_samples, num_samples)

        # Pre-filter the line candidates and find the best match for each line
        matches = self.filter_and_match_lines(scores)

        # [Optionally] filter matches with mutual nearest neighbor filtering
        if self.cross_check:
            matches2 = self.filter_and_match_lines(scores.permute(1, 0, 3, 2))
            mutual = matches2[matches] == torch.arange(len(line_seg1), device=device)
            matches[~mutual] = -1

        return matches

    def sample_line_points(self, line_seg: Tensor) -> Tuple:
        """Regularly sample points along each line segments, with a minimal distance between each point.

        Pad the remaining points.
        Inputs:
            line_seg: an Nx2x2 Tensor.
        Outputs:
            line_points: an N x num_samples x 2 Tensor.
            valid_points: a boolean N x num_samples Tensor.
        """
        KORNIA_CHECK_SHAPE(line_seg, ["N", "2", "2"])
        num_lines = len(line_seg)
        line_lengths = torch.norm(line_seg[:, 0] - line_seg[:, 1], dim=1)

        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples_lst = torch.clamp(
            torch.div(line_lengths, self.min_dist_pts, rounding_mode='floor'), 2, self.num_samples
        ).int()
        line_points = torch.empty((num_lines, self.num_samples, 2), dtype=torch.float)
        valid_points = torch.empty((num_lines, self.num_samples), dtype=torch.bool)
        for n_samp in range(2, self.num_samples + 1):
            # Consider all lines where we can fit up to n_samp points
            cur_mask = num_samples_lst == n_samp
            cur_line_seg = line_seg[cur_mask]
            line_points_x = batched_linspace(cur_line_seg[:, 0, 0], cur_line_seg[:, 1, 0], n_samp, dim=-1)
            line_points_y = batched_linspace(cur_line_seg[:, 0, 1], cur_line_seg[:, 1, 1], n_samp, dim=-1)
            cur_line_points = stack([line_points_x, line_points_y], -1)

            # Pad
            cur_line_points = pad(cur_line_points, (0, 0, 0, self.num_samples - n_samp))
            cur_valid_points = torch.ones(len(cur_line_seg), self.num_samples, dtype=torch.bool)
            cur_valid_points[:, n_samp:] = False

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points

    def filter_and_match_lines(self, scores: Tensor) -> Tensor:
        """Use the scores to keep the top k best lines, compute the Needleman- Wunsch algorithm on each candidate
        pairs, and keep the highest score.

        Inputs:
            scores: a (N, M, n, n) Tensor containing the pairwise scores
                    of the elements to match.
        Outputs:
            matches: a (N) Tensor containing the indices of the best match
        """
        KORNIA_CHECK_SHAPE(scores, ["M", "N", "n", "n"])

        # Pre-filter the pairs and keep the top k best candidate lines
        line_scores1 = scores.max(3)[0]
        valid_scores1 = line_scores1 != -1
        line_scores1 = (line_scores1 * valid_scores1).sum(2) / valid_scores1.sum(2)
        line_scores2 = scores.max(2)[0]
        valid_scores2 = line_scores2 != -1
        line_scores2 = (line_scores2 * valid_scores2).sum(2) / valid_scores2.sum(2)
        line_scores = (line_scores1 + line_scores2) / 2
        topk_lines = torch.argsort(line_scores, dim=1)[:, -self.top_k_candidates :]
        # topk_lines.shape = (n_lines1, top_k_candidates)

        top_scores = torch.take_along_dim(scores, topk_lines[:, :, None, None], dim=1)

        # Consider the reversed line segments as well
        top_scores = concatenate([top_scores, torch.flip(top_scores, dims=[-1])], 1)

        # Compute the line distance matrix with Needleman-Wunsch algo and
        # retrieve the closest line neighbor
        n_lines1, top2k, n, m = top_scores.shape
        top_scores = top_scores.reshape((n_lines1 * top2k, n, m))
        nw_scores = self.needleman_wunsch(top_scores)
        nw_scores = nw_scores.reshape(n_lines1, top2k)
        matches = torch.remainder(torch.argmax(nw_scores, dim=1), top2k // 2)
        matches = topk_lines[torch.arange(n_lines1), matches]
        return matches

    def needleman_wunsch(self, scores: Tensor) -> Tensor:
        """Batched implementation of the Needleman-Wunsch algorithm.

        The cost of the InDel operation is set to 0 by subtracting the gap
        penalty to the scores.
        Inputs:
            scores: a (B, N, M) Tensor containing the pairwise scores
                    of the elements to match.
        """
        KORNIA_CHECK_SHAPE(scores, ["B", "N", "M"])
        b, n, m = scores.shape

        # Recalibrate the scores to get a gap score of 0
        gap = 0.1
        nw_scores = scores - gap

        # Run the dynamic programming algorithm
        nw_grid = torch.zeros(b, n + 1, m + 1, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                nw_grid[:, i + 1, j + 1] = torch.maximum(
                    torch.maximum(nw_grid[:, i + 1, j], nw_grid[:, i, j + 1]), nw_grid[:, i, j] + nw_scores[:, i, j]
                )

        return nw_grid[:, -1, -1]


def keypoints_to_grid(keypoints: Tensor, img_size: tuple) -> Tensor:
    """Convert a list of keypoints into a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate.

    Args:
        keypoints: a tensor [N, 2] of N keypoints (ij coordinates convention).
        img_size: the original image size (H, W)
    """
    KORNIA_CHECK_SHAPE(keypoints, ["N", "2"])
    n_points = len(keypoints)
    grid_points = normalize_pixel_coordinates(keypoints[:, [1, 0]], img_size[0], img_size[1])
    grid_points = grid_points.view(-1, n_points, 1, 2)
    return grid_points


def batched_linspace(start, end, step, dim):
    """Batch version of torch.normalize (similar to the numpy one)."""
    intervals = ((end - start) / (step - 1)).unsqueeze(dim)
    broadcast_size = [1] * len(intervals.shape)
    broadcast_size[dim] = step
    samples = torch.arange(step, dtype=torch.float, device=start.device).reshape(broadcast_size)
    samples = start.unsqueeze(dim) + samples * intervals
    return samples
