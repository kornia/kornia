from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor, concatenate, pad, stack
from kornia.core.check import KORNIA_CHECK_DM_DESC, KORNIA_CHECK_SHAPE
from kornia.feature.laf import get_laf_center
from kornia.feature.sold2.structures import LineMatcherCfg
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.utils.helpers import is_mps_tensor_safe

from .adalam import get_adalam_default_config, match_adalam


def _cdist(d1: Tensor, d2: Tensor) -> Tensor:
    r"""Manual `torch.cdist` for M1."""
    if (not is_mps_tensor_safe(d1)) and (not is_mps_tensor_safe(d2)):
        return torch.cdist(d1, d2)
    d1_sq = (d1**2).sum(dim=1, keepdim=True)
    d2_sq = (d2**2).sum(dim=1, keepdim=True)
    dm = d1_sq.repeat(1, d2.size(0)) + d2_sq.repeat(1, d1.size(0)).t() - 2.0 * d1 @ d2.t()
    dm = dm.clamp(min=0.0).sqrt()
    return dm


def _get_default_fginn_params() -> Dict[str, Any]:
    config = {"th": 0.85, "mutual": False, "spatial_th": 10.0}
    return config


def _get_lazy_distance_matrix(desc1: Tensor, desc2: Tensor, dm_: Optional[Tensor] = None) -> Tensor:
    """Helper function, which checks validity of provided distance matrix, or calculates L2-distance matrix dm is
    not provided.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.
    """
    if dm_ is None:
        dm = _cdist(desc1, desc2)
    else:
        KORNIA_CHECK_DM_DESC(desc1, desc2, dm_)
        dm = dm_
    return dm


def _no_match(dm: Tensor) -> Tuple[Tensor, Tensor]:
    """Helper function, which output empty tensors.

    Returns:
            - Descriptor distance of matching descriptors, shape of :math:`(0, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(0, 2)`.
    """
    dists = torch.empty(0, 1, device=dm.device, dtype=dm.dtype)
    idxs = torch.empty(0, 2, device=dm.device, dtype=torch.long)
    return dists, idxs


def match_nn(desc1: Tensor, desc2: Tensor, dm: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    r"""Function, which finds nearest neighbors in desc2 for each vector in desc1.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Returns:
        - Descriptor distance of matching descriptors, shape of :math:`(B1, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(B1, 2)`.
    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])
    if (len(desc1) == 0) or (len(desc2) == 0):
        return _no_match(desc1)
    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)
    match_dists, idxs_in_2 = torch.min(distance_matrix, dim=1)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=idxs_in_2.device)
    matches_idxs = concatenate([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_mnn(desc1: Tensor, desc2: Tensor, dm: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Function, which finds mutual nearest neighbors in desc2 for each vector in desc1.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(B3, 2)`,
          where 0 <= B3 <= min(B1, B2)
    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])
    if (len(desc1) == 0) or (len(desc2) == 0):
        return _no_match(desc1)
    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)
    ms = min(distance_matrix.size(0), distance_matrix.size(1))
    match_dists, idxs_in_2 = torch.min(distance_matrix, dim=1)
    match_dists2, idxs_in_1 = torch.min(distance_matrix, dim=0)
    minsize_idxs = torch.arange(ms, device=distance_matrix.device)

    if distance_matrix.size(0) <= distance_matrix.size(1):
        mutual_nns = minsize_idxs == idxs_in_1[idxs_in_2][:ms]
        matches_idxs = concatenate([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], 1)[mutual_nns]
        match_dists = match_dists[mutual_nns]
    else:
        mutual_nns = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
        matches_idxs = concatenate([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], 1)[mutual_nns]
        match_dists = match_dists2[mutual_nns]
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_snn(desc1: Tensor, desc2: Tensor, th: float = 0.8, dm: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Function, which finds nearest neighbors in desc2 for each vector in desc1.

    The method satisfies first to second nearest neighbor distance <= th.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
          where 0 <= B3 <= B1.
    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])

    if desc2.shape[0] < 2:  # We cannot perform snn check, so output empty matches
        return _no_match(desc1)
    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)
    vals, idxs_in_2 = torch.topk(distance_matrix, 2, dim=1, largest=False)
    ratio = vals[:, 0] / vals[:, 1]
    mask = ratio <= th
    match_dists = ratio[mask]
    if len(match_dists) == 0:
        return _no_match(distance_matrix)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=distance_matrix.device)[mask]
    idxs_in_2 = idxs_in_2[:, 0][mask]
    matches_idxs = concatenate([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_smnn(desc1: Tensor, desc2: Tensor, th: float = 0.95, dm: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Function, which finds mutual nearest neighbors in desc2 for each vector in desc1.

    the method satisfies first to second nearest neighbor distance <= th.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2,
          shape of :math:`(B3, 2)` where 0 <= B3 <= B1.
    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])

    if (desc1.shape[0] < 2) or (desc2.shape[0] < 2):
        return _no_match(desc1)
    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)

    dists1, idx1 = match_snn(desc1, desc2, th, distance_matrix)
    dists2, idx2 = match_snn(desc2, desc1, th, distance_matrix.t())

    if len(dists2) > 0 and len(dists1) > 0:
        idx2 = idx2.flip(1)
        if not is_mps_tensor_safe(idx1):
            idxs_dm = torch.cdist(idx1.float(), idx2.float(), p=1.0)
        else:
            idxs1_rep = idx1.to(desc1).repeat_interleave(idx2.size(0), dim=0)
            idxs_dm = (idx2.to(desc2).repeat(idx1.size(0), 1) - idxs1_rep).abs().sum(dim=1)
            idxs_dm = idxs_dm.reshape(idx1.size(0), idx2.size(0))
        mutual_idxs1 = idxs_dm.min(dim=1)[0] < 1e-8
        mutual_idxs2 = idxs_dm.min(dim=0)[0] < 1e-8
        good_idxs1 = idx1[mutual_idxs1.view(-1)]
        good_idxs2 = idx2[mutual_idxs2.view(-1)]
        dists1_good = dists1[mutual_idxs1.view(-1)]
        dists2_good = dists2[mutual_idxs2.view(-1)]
        _, idx_upl1 = torch.sort(good_idxs1[:, 0])
        _, idx_upl2 = torch.sort(good_idxs2[:, 0])
        good_idxs1 = good_idxs1[idx_upl1]
        match_dists = torch.max(dists1_good[idx_upl1], dists2_good[idx_upl2])
        matches_idxs = good_idxs1
        match_dists, matches_idxs = match_dists.view(-1, 1), matches_idxs.view(-1, 2)
    else:
        match_dists, matches_idxs = _no_match(distance_matrix)
    return match_dists, matches_idxs


def match_fginn(
    desc1: Tensor,
    desc2: Tensor,
    lafs1: Tensor,
    lafs2: Tensor,
    th: float = 0.8,
    spatial_th: float = 10.0,
    mutual: bool = False,
    dm: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Function, which finds nearest neighbors in desc2 for each vector in desc1.

    The method satisfies first to second nearest neighbor distance <= th,
    and assures 2nd nearest neighbor is geometrically inconsistent with the 1st one
    (see :cite:`MODS2015` for more details)

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
        lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        th: distance ratio threshold.
        spatial_th: minimal distance in pixels to 2nd nearest neighbor.
        mutual: also perform mutual nearest neighbor check
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
          where 0 <= B3 <= B1.
    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])
    BIG_NUMBER = 1000000.0

    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)
    dtype = distance_matrix.dtype

    if desc2.shape[0] < 2:  # We cannot perform snn check, so output empty matches
        return _no_match(distance_matrix)

    num_candidates = max(2, min(10, desc2.shape[0]))
    vals_cand, idxs_in_2 = torch.topk(distance_matrix, num_candidates, dim=1, largest=False)
    vals = vals_cand[:, 0]
    xy2 = get_laf_center(lafs2).view(-1, 2)
    candidates_xy = xy2[idxs_in_2]
    kdist = torch.norm(candidates_xy - candidates_xy[0:1], p=2, dim=2)
    fginn_vals = vals_cand[:, 1:] + (kdist[:, 1:] < spatial_th).to(dtype) * BIG_NUMBER
    fginn_vals_best, fginn_idxs_best = fginn_vals.min(dim=1)

    # orig_idxs = idxs_in_2.gather(1, fginn_idxs_best.unsqueeze(1))[0]
    # if you need to know fginn indexes - uncomment

    vals_2nd = fginn_vals_best
    idxs_in_2 = idxs_in_2[:, 0]

    ratio = vals / vals_2nd
    mask = ratio <= th
    match_dists = ratio[mask]
    if len(match_dists) == 0:
        return _no_match(distance_matrix)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=distance_matrix.device)[mask]
    idxs_in_2 = idxs_in_2[mask]
    matches_idxs = concatenate([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    match_dists, matches_idxs = match_dists.view(-1, 1), matches_idxs.view(-1, 2)

    if not mutual:  # returning 1-way matches
        return match_dists, matches_idxs
    _, idxs_in_1_mut = torch.min(distance_matrix, dim=0)
    good_mask = matches_idxs[:, 0] == idxs_in_1_mut[matches_idxs[:, 1]]
    return match_dists[good_mask], matches_idxs[good_mask]


class DescriptorMatcher(Module):
    """Module version of matching functions.

    See :func:`~kornia.feature.match_nn`, :func:`~kornia.feature.match_snn`,
        :func:`~kornia.feature.match_mnn` or :func:`~kornia.feature.match_smnn` for more details.

    Args:
        match_mode: type of matching, can be `nn`, `snn`, `mnn`, `smnn`.
        th: threshold on distance ratio, or other quality measure.
    """

    def __init__(self, match_mode: str = "snn", th: float = 0.8) -> None:
        super().__init__()
        _match_mode: str = match_mode.lower()
        self.known_modes = ["nn", "mnn", "snn", "smnn"]
        if _match_mode not in self.known_modes:
            raise NotImplementedError(f"{match_mode} is not supported. Try one of {self.known_modes}")
        self.match_mode = _match_mode
        self.th = th

    def forward(self, desc1: Tensor, desc2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
        """
        if self.match_mode == "nn":
            out = match_nn(desc1, desc2)
        elif self.match_mode == "mnn":
            out = match_mnn(desc1, desc2)
        elif self.match_mode == "snn":
            out = match_snn(desc1, desc2, self.th)
        elif self.match_mode == "smnn":
            out = match_smnn(desc1, desc2, self.th)
        else:
            raise NotImplementedError
        return out


class GeometryAwareDescriptorMatcher(Module):
    """Module version of matching functions.

    See :func:`~kornia.feature.match_nn`, :func:`~kornia.feature.match_snn`,
        :func:`~kornia.feature.match_mnn` or :func:`~kornia.feature.match_smnn` for more details.

    Args:
        match_mode: type of matching, can be `fginn`.
        th: threshold on distance ratio, or other quality measure.
    """

    known_modes: ClassVar[List[str]] = ["fginn", "adalam"]

    def __init__(self, match_mode: str = "fginn", params: Dict[str, Tensor] = {}) -> None:
        super().__init__()
        _match_mode: str = match_mode.lower()
        if _match_mode not in self.known_modes:
            raise NotImplementedError(f"{match_mode} is not supported. Try one of {self.known_modes}")
        self.match_mode = _match_mode
        self.params = params

    def forward(self, desc1: Tensor, desc2: Tensor, lafs1: Tensor, lafs2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
        """
        if self.match_mode == "fginn":
            params = _get_default_fginn_params()
            params.update(self.params)
            out = match_fginn(desc1, desc2, lafs1, lafs2, params["th"], params["spatial_th"], params["mutual"])
        elif self.match_mode == "adalam":
            _params = get_adalam_default_config()
            _params.update(self.params)  # type: ignore[typeddict-item]
            out = match_adalam(desc1, desc2, lafs1, lafs2, config=_params)
        else:
            raise NotImplementedError
        return out


class WunschLineMatcher(Module):
    """Class matching two sets of line segments with the Needleman-Wunsch algorithm."""

    def __init__(self, config: LineMatcherCfg = LineMatcherCfg()) -> None:
        super().__init__()
        # Initialize the parameters
        self.config = config
        self.cross_check = self.config.cross_check
        self.num_samples = self.config.num_samples
        self.min_dist_pts = self.config.min_dist_pts
        self.top_k_candidates = self.config.top_k_candidates
        self.grid_size = self.config.grid_size
        self.line_score = self.config.line_score

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

    def sample_line_points(self, line_seg: Tensor) -> Tuple[Tensor, Tensor]:
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
        dev = line_seg.device
        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples_lst = torch.clamp(
            torch.div(line_lengths, self.min_dist_pts, rounding_mode="floor"), 2, self.num_samples
        ).int()
        line_points = torch.empty((num_lines, self.num_samples, 2), dtype=torch.float, device=dev)
        valid_points = torch.empty((num_lines, self.num_samples), dtype=torch.bool, device=dev)
        for n_samp in range(2, self.num_samples + 1):
            # Consider all lines where we can fit up to n_samp points
            cur_mask = num_samples_lst == n_samp
            cur_line_seg = line_seg[cur_mask]
            line_points_x = batched_linspace(cur_line_seg[:, 0, 0], cur_line_seg[:, 1, 0], n_samp, dim=-1)
            line_points_y = batched_linspace(cur_line_seg[:, 0, 1], cur_line_seg[:, 1, 1], n_samp, dim=-1)
            cur_line_points = stack([line_points_x, line_points_y], -1)

            # Pad
            cur_line_points = pad(cur_line_points, (0, 0, 0, self.num_samples - n_samp))
            cur_valid_points = torch.ones(len(cur_line_seg), self.num_samples, dtype=torch.bool, device=dev)
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
        dev = scores.device
        # Run the dynamic programming algorithm
        nw_grid = torch.zeros(b, n + 1, m + 1, dtype=torch.float, device=dev)
        for i in range(n):
            for j in range(m):
                nw_grid[:, i + 1, j + 1] = torch.maximum(
                    torch.maximum(nw_grid[:, i + 1, j], nw_grid[:, i, j + 1]), nw_grid[:, i, j] + nw_scores[:, i, j]
                )

        return nw_grid[:, -1, -1]


def keypoints_to_grid(keypoints: Tensor, img_size: Tuple[int, int]) -> Tensor:
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


def batched_linspace(start: Tensor, end: Tensor, step: int, dim: int) -> Tensor:
    """Batch version of torch.normalize (similar to the numpy one)."""
    intervals = ((end - start) / (step - 1)).unsqueeze(dim)
    broadcast_size = [1] * len(intervals.shape)
    broadcast_size[dim] = step
    samples = torch.arange(step, dtype=torch.float, device=start.device).reshape(broadcast_size)
    samples = start.unsqueeze(dim) + samples * intervals
    return samples
