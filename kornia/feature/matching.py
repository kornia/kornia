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

from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_DM_DESC, KORNIA_CHECK_SHAPE
from kornia.core.utils import _l2_normalize, is_exporting, is_mps_tensor_safe
from kornia.feature.laf import get_laf_center
from kornia.feature.steerers import DiscreteSteerer

from .adalam import get_adalam_default_config, match_adalam


def _cdist(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    r"""Compute pairwise L2 distances between rows of d1 and d2.

    Uses ``torch.cdist`` on non-MPS devices outside export. Falls back to a manual
    squared-distance implementation for MPS tensors and export. Half-precision
    inputs are computed in float32: in half precision the squared norms round away
    the squared distance between nearby descriptors, and ``torch.cdist`` has no
    float16 kernel on CPU. Distances are returned in the input dtype.
    """
    half = (torch.float16, torch.bfloat16)
    output_dtype = d1.dtype
    if output_dtype in half and d2.dtype == output_dtype:
        d1, d2 = d1.float(), d2.float()
    if (
        not is_exporting()  # `torch.cdist` has no ONNX lowering
        and (not is_mps_tensor_safe(d1))
        and (not is_mps_tensor_safe(d2))
    ):
        distances = torch.cdist(d1, d2)
    else:
        # Autocast would lower the matmul precision again and cancel nearby distances.
        with torch.autocast(device_type="cpu", enabled=False), torch.autocast(device_type="cuda", enabled=False):
            d1_sq = (d1**2).sum(dim=1, keepdim=True)
            d2_sq = (d2**2).sum(dim=1, keepdim=True)
            dm = d1_sq.repeat(1, d2.size(0)) + d2_sq.repeat(1, d1.size(0)).t() - 2.0 * d1 @ d2.t()
            dm = dm.clamp(min=0.0)
            mask = dm > 0.0
            safe_dm = torch.where(mask, dm, torch.ones_like(dm))
            distances = torch.where(mask, safe_dm.sqrt(), torch.zeros_like(dm))
    return distances.to(output_dtype)


def _get_default_fginn_params() -> Dict[str, Any]:
    return {"th": 0.85, "mutual": False, "spatial_th": 10.0}


def _get_lazy_distance_matrix(
    desc1: torch.Tensor, desc2: torch.Tensor, dm_: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Check validity of provided distance matrix, or calculates L2-distance matrix if dm is not provided.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm_: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    """
    if dm_ is None:
        dm = _cdist(desc1, desc2)
    else:
        KORNIA_CHECK_DM_DESC(desc1, desc2, dm_)
        dm = dm_
    return dm


def _no_match(dm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Output empty tensors.

    Returns:
            - Descriptor distance of matching descriptors, shape of :math:`(0, 1)`.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(0, 2)`.

    """
    dists = torch.empty(0, 1, device=dm.device, dtype=dm.dtype)
    idxs = torch.empty(0, 2, device=dm.device, dtype=torch.long)
    return dists, idxs


def match_nn(
    desc1: torch.Tensor, desc2: torch.Tensor, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Find nearest neighbors in desc2 for each vector in desc1.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Returns:
        - Descriptor distance of matching descriptors, shape of :math:`(B1, 1)`.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(B1, 2)`.

    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])
    if (len(desc1) == 0) or (len(desc2) == 0):
        return _no_match(desc1)
    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)
    match_dists, idxs_in_2 = torch.min(distance_matrix, dim=1)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=idxs_in_2.device)
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_mnn(
    desc1: torch.Tensor, desc2: torch.Tensor, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find mutual nearest neighbors in desc2 for each vector in desc1.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(B3, 2)`,
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
        matches_idxs = torch.cat([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], 1)[mutual_nns]
        match_dists = match_dists[mutual_nns]
    else:
        mutual_nns = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
        matches_idxs = torch.cat([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], 1)[mutual_nns]
        match_dists = match_dists2[mutual_nns]
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_snn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find nearest neighbors in desc2 for each vector in desc1.

    The method satisfies first to second nearest neighbor distance <= th.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
          where 0 <= B3 <= B1.

    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])

    if desc1.shape[0] == 0 or desc2.shape[0] < 2:  # We cannot perform snn check, so output empty matches
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
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_smnn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.95, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find mutual nearest neighbors in desc2 for each vector in desc1.

    the method satisfies first to second nearest neighbor distance <= th.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
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
        # Each target occurs at most once in idx2. Join on its integer index instead of
        # comparing every pair of matches in floating point (quadratic memory and inexact in half).
        reverse_lookup = torch.full((desc2.size(0),), -1, dtype=torch.long, device=idx2.device)
        reverse_lookup[idx2[:, 0]] = torch.arange(idx2.size(0), device=idx2.device)
        reverse_positions = reverse_lookup[idx1[:, 1]]
        mutual = (reverse_positions >= 0) & (idx2[reverse_positions, 1] == idx1[:, 0])
        # match_snn already returns source indices in ascending order.
        matches_idxs = idx1[mutual]
        match_dists = torch.maximum(dists1[mutual], dists2[reverse_positions[mutual]])
    else:
        match_dists, matches_idxs = _no_match(distance_matrix)
    return match_dists, matches_idxs


def match_smnn_batched(
    desc1: torch.Tensor,
    desc2: torch.Tensor,
    th: float = 0.95,
    dm: Optional[torch.Tensor] = None,
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Match independent descriptor pairs with a batched symmetric nearest-neighbor ratio test.

    Like :func:`match_smnn`, each match must pass Lowe's first/second L2 distance
    ratio test in both directions and have mutual nearest neighbors. The returned
    quality is the larger of the two ratios. Pairs are processed together with
    batched distance computation and top-k reductions, without a per-pair loop.

    Args:
        desc1: First descriptors, shape :math:`(B, N, D)`.
        desc2: Second descriptors, shape :math:`(B, M, D)`.
        th: Inclusive distance-ratio threshold.
        dm: Optional precomputed L2 distances, shape :math:`(B, N, M)`.
        mask1: Optional boolean mask of valid first descriptors, shape :math:`(B, N)`.
        mask2: Optional boolean mask of valid second descriptors, shape :math:`(B, M)`.
            Masks allow padding variable-length pairs; padding never becomes a neighbor.

    Returns:
        - Ratios, shape :math:`(K, 1)`, with the input device and dtype.
        - Long indices, shape :math:`(K, 3)`: ``(batch_index, index_in_desc1, index_in_desc2)``.
          Indices refer to the original padded tensors, sorted by batch and first-descriptor index.
          A pair with fewer than two valid descriptors on either side contributes no matches.

    Note:
        Memory for the distance matrix scales as :math:`B N M`. Bucket pairs by
        descriptor counts and limit batch size when matching large collections.
        Half-precision descriptors use float32 distance computation before ratios
        are converted back to the input dtype.
        Masked inputs use ``torch.cdist``'s direct Euclidean path to avoid the
        cancellation possible in its matrix-multiplication implementation. This
        can be slower for large masked descriptor sets; unmasked inputs retain
        the default small-input direct path and large-input matrix implementation.
        As in :func:`match_smnn`, a zero second-neighbor distance produces an
        undefined ratio and that ambiguous match is rejected. Ties follow
        :func:`torch.topk` and do not have a guaranteed cross-device ordering.

    Example:
        >>> a = torch.tensor([[[0., 0.], [1., 1.], [2., 2.]]])
        >>> ratios, indices = match_smnn_batched(a, a.flip(1))
        >>> indices
        tensor([[0, 0, 2],
                [0, 1, 1],
                [0, 2, 0]])

    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "N", "D"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "M", "D"])
    batch, n, dim = desc1.shape
    if desc2.shape[0] != batch or desc2.shape[2] != dim:
        raise ValueError("Descriptor batch sizes and dimensions must match")
    if desc1.device != desc2.device or desc1.dtype != desc2.dtype or not desc1.is_floating_point():
        raise ValueError("Descriptors must have the same floating dtype and device")
    m = desc2.shape[1]
    if dm is not None and (dm.shape != (batch, n, m) or dm.device != desc1.device or dm.dtype != desc1.dtype):
        raise ValueError("Distance matrix must have shape (B, N, M) and the descriptor dtype/device")
    for mask, shape in ((mask1, (batch, n)), (mask2, (batch, m))):
        if mask is not None and (mask.shape != shape or mask.dtype != torch.bool or mask.device != desc1.device):
            raise ValueError("Validity masks must be boolean tensors with shape (B, N)/(B, M) on the input device")
    if batch == 0 or n < 2 or m < 2:
        return desc1.new_empty((0, 1)), torch.empty((0, 3), dtype=torch.long, device=desc1.device)

    valid1 = torch.ones((batch, n), dtype=torch.bool, device=desc1.device) if mask1 is None else mask1
    valid2 = torch.ones((batch, m), dtype=torch.bool, device=desc1.device) if mask2 is None else mask2
    masked = mask1 is not None or mask2 is not None
    if dm is None:
        # Exclude padded values from the calculation itself. Masking the resulting
        # distances is too late for NaN/Inf padding, which can poison gradients.
        work1 = desc1.masked_fill(~valid1.unsqueeze(-1), 0.0) if mask1 is not None else desc1
        work2 = desc2.masked_fill(~valid2.unsqueeze(-1), 0.0) if mask2 is not None else desc2
        work1 = work1.float() if work1.dtype in (torch.float16, torch.bfloat16) else work1
        work2 = work2.float() if work2.dtype in (torch.float16, torch.bfloat16) else work2
        if not is_exporting() and not is_mps_tensor_safe(desc1):
            distances = torch.cdist(
                work1,
                work2,
                compute_mode="donot_use_mm_for_euclid_dist" if masked else "use_mm_for_euclid_dist_if_necessary",
            )
        else:
            # MPS/ONNX lack cdist. Accumulate direct differences in descriptor
            # chunks to bound temporary memory without looping over pairs.
            squared = work1.new_zeros((batch, n, m))
            for start in range(0, dim, 16):
                difference = work1[:, :, None, start : start + 16] - work2[:, None, :, start : start + 16]
                squared = squared + (difference * difference).sum(-1)
            positive = squared > 0
            distances = torch.where(positive, torch.where(positive, squared, torch.ones_like(squared)).sqrt(), 0.0)
    else:
        distances = dm
    if mask1 is not None or mask2 is not None:
        distances = distances.masked_fill(~(valid1.unsqueeze(2) & valid2.unsqueeze(1)), float("inf"))
    values1, neighbors1 = distances.topk(2, dim=2, largest=False)
    values2, neighbors2 = distances.transpose(1, 2).topk(2, dim=2, largest=False)
    defined1 = torch.isfinite(values1).all(-1) & (values1[..., 1] > 0)
    defined2 = torch.isfinite(values2).all(-1) & (values2[..., 1] > 0)
    # Substitute finite operands before division: rejecting a 0/0 ratio only
    # after division would still leave NaNs in the backward pass for dm.
    ratios1 = torch.where(defined1, values1[..., 0], 0.0) / torch.where(defined1, values1[..., 1], 1.0)
    ratios2 = torch.where(defined2, values2[..., 0], 0.0) / torch.where(defined2, values2[..., 1], 1.0)
    nearest2 = neighbors1[..., 0]
    reverse_ratio = ratios2.gather(1, nearest2)
    mutual = neighbors2[..., 0].gather(1, nearest2) == torch.arange(n, device=desc1.device)
    enough = (valid1.sum(1) >= 2) & (valid2.sum(1) >= 2)
    keep = (
        valid1
        & enough.unsqueeze(1)
        & defined1
        & defined2.gather(1, nearest2)
        & mutual
        & (ratios1 <= th)
        & (reverse_ratio <= th)
    )
    batch_index, index1 = keep.nonzero(as_tuple=True)
    indices = torch.stack((batch_index, index1, nearest2[batch_index, index1]), dim=1)
    ratios = torch.maximum(ratios1, reverse_ratio)[batch_index, index1].unsqueeze(1).to(desc1.dtype)
    return ratios, indices


def match_fginn(
    desc1: torch.Tensor,
    desc2: torch.Tensor,
    lafs1: torch.Tensor,
    lafs2: torch.Tensor,
    th: float = 0.8,
    spatial_th: float = 10.0,
    mutual: bool = False,
    dm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find nearest neighbors in desc2 for each vector in desc1.

    The method satisfies first to second nearest neighbor distance <= th,
    and assures 2nd nearest neighbor is geometrically inconsistent with the 1st one
    (see :cite:`MODS2015` for more details)

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    .. note::
        The geometric check looks at the ``min(10, B2)`` nearest candidates and penalizes every one of them
        that lies within ``spatial_th`` pixels of the query's own 1st nearest neighbor. When *all* of them do --
        a dense cluster of detections on one structure -- the effective 2nd nearest neighbor distance saturates,
        the ratio collapses towards zero and the match is **accepted**. That is the intended reading: no distinct
        competing structure among the candidates means the 1st nearest neighbor is unambiguous.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`. Accepted for API symmetry with
          :func:`~kornia.feature.match_adalam` but not read by this function -- only
          ``lafs2`` feeds the geometric check.
        lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.
        th: distance ratio threshold.
        spatial_th: minimal distance in pixels to 2nd nearest neighbor.
        mutual: also perform mutual nearest neighbor check.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
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
    candidates_xy = xy2[idxs_in_2]  # (B1, num_candidates, 2)
    # Distance from every candidate to the 1st nearest neighbour *of the same query*.
    # Indexing dim 0 here would take query 0's candidate list and broadcast it over all queries.
    kdist = torch.norm(candidates_xy - candidates_xy[:, 0:1], p=2, dim=2)
    fginn_vals = vals_cand[:, 1:] + (kdist[:, 1:] < spatial_th).to(dtype) * BIG_NUMBER
    vals_2nd, _ = fginn_vals.min(dim=1)
    idxs_in_2 = idxs_in_2[:, 0]

    ratio = vals / vals_2nd
    mask = ratio <= th
    match_dists = ratio[mask]
    if len(match_dists) == 0:
        return _no_match(distance_matrix)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=distance_matrix.device)[mask]
    idxs_in_2 = idxs_in_2[mask]
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    match_dists, matches_idxs = match_dists.view(-1, 1), matches_idxs.view(-1, 2)

    if not mutual:  # returning 1-way matches
        return match_dists, matches_idxs
    _, idxs_in_1_mut = torch.min(distance_matrix, dim=0)
    good_mask = matches_idxs[:, 0] == idxs_in_1_mut[matches_idxs[:, 1]]
    return match_dists[good_mask], matches_idxs[good_mask]


class DescriptorMatcher(nn.Module):
    """nn.Module version of descriptor-only matching functions.

    This matcher only requires descriptors (no LAFs). For geometry-aware matching that uses LAFs,
    see :class:`~kornia.feature.GeometryAwareDescriptorMatcher`.

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

    def forward(self, desc1: torch.Tensor, desc2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward.

        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.

        Returns:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
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


class DescriptorMatcherWithSteerer(nn.Module):
    """Matching that is invariant under rotations, using Steerers.

    Args:
        steerer: An instance of :func:`kornia.feature.steerers.DiscreteSteerer`.
        steerer_order: order of discretisation of rotation angles, e.g. 4 leads to quarter rotations.
        steer_mode: can be `global`, `local`.
            `global` means that the we output matches from the global rotation with most matches.
            `local` means that we output matches from a distance matrix
            where the distance between each descriptor pair is the minimal over rotations.
        match_mode: type of matching, can be `nn`, `snn`, `mnn`, `smnn`.
            WARNING: using steer_mode `global` with match_mode `nn` will lead to bad results
            since `nn` doesn't generate different amount of matches depending on goodness of fit.
        th: threshold on distance ratio, or other quality measure.

    Example:
        The example below is deliberately small so that it runs anywhere. A real workload uses
        full-resolution images and many more keypoints, and is worth moving onto an accelerator --
        uncomment the ``device`` lines to do that.

        >>> import kornia.feature as KF
        >>> # import kornia as K
        >>> # device = K.core.utils.get_cuda_or_mps_device_if_available()
        >>> img1 = torch.randn([1, 3, 128, 128])
        >>> img2 = torch.randn([1, 3, 128, 128])
        >>> # img1, img2 = img1.to(device), img2.to(device)
        >>> dedode = KF.DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-SO2")
        >>> # dedode = dedode.to(device)
        >>> steerer_order = 8  # discretisation order of rotation angles
        >>> steerer = KF.steerers.DiscreteSteerer.create_dedode_default(
        ... generator_type="SO2", steerer_order=steerer_order
        ... )
        >>> # steerer = steerer.to(device)
        >>> matcher = KF.matching.DescriptorMatcherWithSteerer(
        ... steerer=steerer, steerer_order=steerer_order, steer_mode="global", match_mode="smnn", th=0.98
        ... )
        >>> with torch.inference_mode():
        ...     kps1, scores1, descs1 = dedode(img1, n=1_000)
        ...     kps2, scores2, descs2 = dedode(img2, n=1_000)
        ...     kps1, kps2, descs1, descs2 = kps1[0], kps2[0], descs1[0], descs2[0]
        ...     dists, idxs, num_rot = matcher(
        ...         descs1, descs2, normalize=True, subset_size=500,
        ...     )
        >>> idxs.shape == (dists.shape[0], 2), 0 <= num_rot < steerer_order
        (True, True)
        >>> bool((idxs[:, 0] < len(kps1)).all()), bool((idxs[:, 1] < len(kps2)).all())
        (True, True)
        >>> # print(f"{idxs.shape[0]} tentative matches with steered DeDoDe")
        >>> # print(f"at rotation of {num_rot * 360 / steerer_order} degrees")

    """

    def __init__(
        self,
        steerer: DiscreteSteerer,
        steerer_order: int,
        steer_mode: str = "global",
        match_mode: str = "snn",
        th: float = 0.8,
    ) -> None:
        super().__init__()
        self.steerer = steerer
        self.steerer_order = steerer_order

        _steer_mode: str = steer_mode.lower()
        self.known_steer_modes = ["global", "local"]
        if _steer_mode not in self.known_steer_modes:
            raise NotImplementedError(f"{steer_mode} is not supported. Try one of {self.known_steer_modes}")
        self.steer_mode = _steer_mode
        _match_mode: str = match_mode.lower()
        self.known_modes = ["nn", "mnn", "snn", "smnn"]
        if _match_mode not in self.known_modes:
            raise NotImplementedError(f"{match_mode} is not supported. Try one of {self.known_modes}")
        self.match_mode = _match_mode
        self.th = th

    def matching_function(
        self, d1: torch.Tensor, d2: torch.Tensor, dm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch descriptor matching to the configured matching strategy.

        Args:
            d1: Descriptor tensor from the first image with shape `(N1, D)`, where `N1` is descriptor count and `D` is
                descriptor dimension.
            d2: Descriptor tensor from the second image with shape `(N2, D)`, where `N2` is descriptor count and `D` is
                descriptor dimension.
            dm: Optional descriptor-distance matrix used instead of recomputing distances from `d1` and `d2`.

        Returns:
            Tuple containing match distances or scores and index pairs selected by the configured matching strategy.
        """
        if self.match_mode == "nn":
            return match_nn(d1, d2, dm=dm)
        if self.match_mode == "mnn":
            return match_mnn(d1, d2, dm=dm)
        if self.match_mode == "snn":
            return match_snn(d1, d2, self.th, dm=dm)
        if self.match_mode == "smnn":
            return match_smnn(d1, d2, self.th, dm=dm)
        raise NotImplementedError

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        normalize: bool = False,
        subset_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """Run forward.

        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            normalize: bool to decide whether to F.normalize descriptors to unit norm.
            subset_size: If set, the subset size to use for determining optimal
                number of rotations. Smaller subset size leads to faster but less
                accurate matching. Only used when `self.steer_mode` is `"global"`.

        Returns:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
            - Number of global rotations from desc1 to desc2, in terms of `self.steerer_order`
                (will be `None` if `self.steer_mode` is `local`).

        """
        rot1to2 = None

        if normalize:
            desc1 = _l2_normalize(desc1, dim=-1)
            desc2 = _l2_normalize(desc2, dim=-1)

        if self.steer_mode == "global":
            if subset_size is not None:
                subsample1 = torch.randperm(desc1.shape[0])[:subset_size]
                subsample2 = torch.randperm(desc2.shape[0])[:subset_size]
                _, _, rot1to2 = self(
                    desc1[subsample1],
                    desc2[subsample2],
                    normalize=normalize,
                )
                desc1 = self.steerer.steer_descriptions(
                    desc1,
                    steerer_power=rot1to2,
                    normalize=normalize,
                )
                dist, idx = self.matching_function(desc1, desc2, None)
                return dist, idx, rot1to2
            dist, idx = self.matching_function(desc1, desc2, None)
            rot1to2 = 0
            for r in range(1, self.steerer_order):
                desc1 = self.steerer.steer_descriptions(desc1, normalize=normalize)
                dist_new, idx_new = self.matching_function(desc1, desc2, None)
                if idx_new.shape[0] > idx.shape[0]:
                    dist, idx, rot1to2 = dist_new, idx_new, r
        elif self.steer_mode == "local":
            dm = _cdist(desc1, desc2)
            for _ in range(1, self.steerer_order):
                desc1 = self.steerer.steer_descriptions(desc1, normalize=normalize)
                dm_new = _cdist(desc1, desc2)
                dm = torch.minimum(dm, dm_new)
            dist, idx = self.matching_function(desc1, desc2, dm)
        else:
            raise NotImplementedError

        return dist, idx, rot1to2


class GeometryAwareDescriptorMatcher(nn.Module):
    """nn.Module version of geometry-aware matching functions that use LAFs (Local Affine Frames).

    Unlike :class:`~kornia.feature.DescriptorMatcher`, this matcher requires both descriptors and LAFs.
    See :func:`~kornia.feature.match_fginn` or :func:`~kornia.feature.match_adalam` for more details.

    Args:
        match_mode: type of matching, can be `fginn` or `adalam`.
        params: dictionary of parameters for the matching function.

    """

    known_modes: ClassVar[List[str]] = ["fginn", "adalam"]

    def __init__(self, match_mode: str = "fginn", params: Optional[Dict[str, torch.Tensor]] = None) -> None:
        super().__init__()
        _match_mode: str = match_mode.lower()
        if _match_mode not in self.known_modes:
            raise NotImplementedError(f"{match_mode} is not supported. Try one of {self.known_modes}")
        self.match_mode = _match_mode
        self.params = params or {}

    def forward(
        self, desc1: torch.Tensor, desc2: torch.Tensor, lafs1: torch.Tensor, lafs2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward.

        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        Returns:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
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
