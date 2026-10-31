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

from kornia.core._compat import torch_version_ge
from kornia.core.check import KORNIA_CHECK_DM_DESC, KORNIA_CHECK_SHAPE
from kornia.core.utils import _l2_normalize, is_autocast_enabled, is_exporting, is_mps_tensor_safe
from kornia.feature.laf import get_laf_center
from kornia.feature.steerers import DiscreteSteerer

from .adalam import match_adalam

# ``aten::_cdist_backward`` gained its MPS kernel in torch 2.13; kornia supports torch >= 2.5.1.
_MPS_HAS_CDIST_BACKWARD = torch_version_ge(2, 13)


def _torch_cdist_available(x: torch.Tensor) -> bool:
    """Return whether ``torch.cdist`` can compute, and differentiate, the distances of descriptors like ``x``."""
    if is_exporting():  # `torch.cdist` has no ONNX lowering
        return False
    # TorchScript cannot read ``_MPS_HAS_CDIST_BACKWARD``, so a scripted call keeps the fallback on MPS.
    mps_has_backward = False
    if not torch.jit.is_scripting():
        mps_has_backward = _MPS_HAS_CDIST_BACKWARD
    return mps_has_backward or not is_mps_tensor_safe(x)


def _cdist(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    r"""Compute pairwise L2 distances between rows of d1 and d2.

    Uses ``torch.cdist``, except under export, where it has no ONNX lowering, and on
    MPS before torch 2.13, where it has no backward. Both fall back to the
    matrix-multiplication form of the squared distance. Half-precision inputs are
    computed in float32: in half precision the squared norms round away the squared
    distance between nearby descriptors, and ``torch.cdist`` has no float16 kernel on
    CPU. Distances are returned in the input dtype.
    """
    half = (torch.float16, torch.bfloat16)
    output_dtype = d1.dtype
    # a separate `if`: TorchScript skips the body only under a bare `not torch.jit.is_scripting()`
    if not torch.jit.is_scripting():  # noqa: SIM102
        if d1.dtype != d2.dtype:
            # Inside autocast, `torch.cdist` computes a mixed pair of float16, bfloat16 and float32 in float32, and
            # so does the fallback here; autocast does not cast float64. Outside autocast, the CPU `torch.cdist`
            # raises for a mixed pair, while on MPS it computes one of up to 25 rows in the dtype of d1.
            castable = (torch.float16, torch.bfloat16, torch.float32)
            if not (is_autocast_enabled() and d1.dtype in castable and d2.dtype in castable):
                raise RuntimeError(f"The descriptors must have the same dtype, got {d1.dtype} and {d2.dtype}")
            d1, d2 = d1.float(), d2.float()
    if output_dtype in half:
        d1, d2 = d1.float(), d2.float()
    if _torch_cdist_available(d1) and _torch_cdist_available(d2):
        distances = torch.cdist(d1, d2)
    else:
        # Autocast would lower the matmul precision again and cancel nearby distances.
        with torch.autocast(device_type="cpu", enabled=False), torch.autocast(device_type="cuda", enabled=False):
            d1_sq = (d1**2).sum(dim=1, keepdim=True)
            d2_sq = (d2**2).sum(dim=1, keepdim=True)
            dm = d1_sq + d2_sq.t() - 2.0 * d1 @ d2.t()
            dm = dm.clamp(min=0.0)
            mask = dm > 0.0
            safe_dm = torch.where(mask, dm, 1.0)
            distances = torch.where(mask, safe_dm.sqrt(), 0.0)
    return distances.to(output_dtype)


def _get_default_fginn_params() -> Dict[str, Any]:
    return {"th": 0.85, "mutual": False, "spatial_th": 10.0}


def _get_lazy_distance_matrix(
    desc1: torch.Tensor, desc2: torch.Tensor, dm_: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Check validity of provided distance matrix, or calculates L2-distance matrix if dm is not provided.

    Args:
        desc1: Batch of descriptors of a shape ``(B1, D)``.
        desc2: Batch of descriptors of a shape ``(B2, D)``.
        dm_: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of ``(B1, B2)``.

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
            - Descriptor distance of matching descriptors, shape of ``(0, 1)``.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2, shape of ``(0, 2)``.

    """
    dists = torch.empty(0, 1, device=dm.device, dtype=dm.dtype)
    idxs = torch.empty(0, 2, device=dm.device, dtype=torch.long)
    return dists, idxs


def match_nn(
    desc1: torch.Tensor, desc2: torch.Tensor, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Match each descriptor in desc1 to its nearest neighbor in desc2, with no filtering.

    Convention:
        - Every ``match_*`` function returns ``(dists, idxs)``, one row per match. ``idxs`` is an int64
          ``(M, 2)`` tensor of ``(index in desc1, index in desc2)`` rows (:func:`match_smnn_batched` adds the
          batch index as a first column). ``dists`` is ``(M, 1)``, in the dtype of ``dm`` when it is given and in
          the descriptor dtype otherwise, except that :func:`match_adalam` returns float32. Lower values are better,
          and what they are depends on the function:

          - :func:`match_nn`, :func:`match_mnn`: the distance, the entry of ``dm`` when it is given.
          - :func:`match_snn`: the ratio of the nearest to the second-nearest distance, not a distance.
          - :func:`match_smnn`, :func:`match_smnn_batched`: the larger of the two ratios, desc1 to desc2 and back.
          - :func:`match_fginn`: the ratio of the nearest distance to that of a second neighbour, taken among the 10
            nearest, whose ``lafs2`` centre lies at least ``spatial_th`` from the nearest one's, and 0 without one;
            see its Convention block.
          - :func:`match_adalam`: the inverse of the local RANSAC confidence; see its Convention block.

          :class:`~kornia.feature.LightGlueMatcher` returns LightGlue's matching score in the place of ``dists``,
          which is higher for better matches, and :class:`~kornia.feature.LocalFeatureMatcher` reports
          ``1 - dists`` as ``confidence``.
        - The ratio tests of :func:`match_snn`, :func:`match_smnn`, :func:`match_smnn_batched` and
          :func:`match_fginn` keep ``ratio <= th``: a ratio equal to ``th`` passes.
        - Without ``dm``, the L2 distances between the descriptors are used. A given ``dm`` is read as it is:
          :func:`match_nn` and :func:`match_mnn` accept any matrix in which smaller is better and return its
          entries, while the ratio tests divide its entries, so they need non-negative values, and squared
          distances need ``th ** 2``.
        - An empty desc1 or desc2 gives a ``(0, 1)`` and a ``(0, 2)`` tensor (``(0, 3)`` for
          :func:`match_smnn_batched`). The ratio tests need a second neighbour: :func:`match_snn` and
          :func:`match_fginn` return no match unless desc2 has two rows, and :func:`match_smnn`,
          :func:`match_smnn_batched` and :func:`match_adalam`, which scores its putative matches by the same ratio,
          need two rows in both.

    Args:
        desc1: Batch of descriptors of a shape ``(B1, D)``.
        desc2: Batch of descriptors of a shape ``(B2, D)``.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of ``(B1, B2)``.

    Returns:
        - Descriptor distance of matching descriptors, shape of ``(B1, 1)``.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2, shape of ``(B1, 2)``.

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
    """Match the descriptors of desc1 and desc2 that are each other's nearest neighbors (mutual nearest neighbors).

    Convention:
        See the Convention block on :func:`match_nn`. A pair is kept only when each of the two descriptors is the
        nearest neighbour of the other.

    Args:
        desc1: Batch of descriptors of a shape ``(B1, D)``.
        desc2: Batch of descriptors of a shape ``(B2, D)``.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of ``(B1, B2)``.

    Return:
        - Descriptor distance of matching descriptors, shape of ``(B3, 1)``.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2, shape of ``(B3, 2)``,
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


def _first_to_second_nearest_ratio(distance_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return each row's ratio of the smallest to the second-smallest distance, and the column of the smallest.

    :func:`match_snn` calls this on MPS in place of ``torch.topk(distance_matrix, 2, dim=1, largest=False)``, which
    is two to three times slower there. Elsewhere ``torch.topk`` is at least as fast: as fast on CUDA, and two to five
    times faster on an x86 CPU, where it reads the matrix about once and these reductions several times. The ratios
    are those of ``torch.topk``, which orders nan after ``+inf``. Where a ratio is a number, its column is that of the
    smallest distance, the first of tied ones. A row without a finite smallest and a finite, non-zero second-smallest
    distance gets no gradient. A row needs at least two columns.
    """
    inf = float("inf")
    # A row-major copy of the transposed matrix that match_smnn passes; the MPS timings of #6465 were taken with it.
    dm = distance_matrix.contiguous()
    is_nan = torch.isnan(dm)
    remaining = torch.nan_to_num(dm, nan=inf, posinf=inf, neginf=-inf)
    first, first_idx = remaining.min(dim=1)
    idx = first_idx[:, None]
    # Overwrite the smallest with another distance of its row, which keeps the minimum of the others. It is read from
    # `dm`, not from `remaining`: autograd must not save a tensor that is then modified in place.
    other = dm.gather(1, (idx == 0).long())
    remaining.scatter_(1, idx, torch.nan_to_num(other, nan=inf, posinf=inf, neginf=-inf))
    # Ordered after +inf, a nan is the second-smallest distance only when every other distance of the row is nan.
    no_second = is_nan.scatter_(1, idx, True).all(dim=1)
    second = torch.where(no_second, float("nan"), remaining.amin(dim=1))
    # Without such a pair the quotient's gradient is nan or infinite, and amin's backward would spread it over every
    # distance of the row. Those rows keep the quotient as their ratio, but detached.
    defined = torch.isfinite(first) & torch.isfinite(second) & (second != 0)
    ratio = torch.where(defined, first, 0) / torch.where(defined, second, 1)
    return torch.where(defined, ratio, (first / second).detach()), first_idx


def match_snn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match each descriptor in desc1 to its nearest neighbor in desc2 if it passes Lowe's second-nearest ratio test.

    The ratio of the first to the second nearest neighbor distance must be <= th.

    Convention:
        See the Convention block on :func:`match_nn`; ``dists`` is the ratio of the nearest to the second-nearest
        distance.

    Args:
        desc1: Batch of descriptors of a shape ``(B1, D)``.
        desc2: Batch of descriptors of a shape ``(B2, D)``.
        th: distance ratio threshold.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of ``(B1, B2)``.

    Return:
        - Ratio of the nearest to the second-nearest distance of each match, shape of ``(B3, 1)``.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2. Shape: ``(B3, 2)``,
          where 0 <= B3 <= B1.

    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])

    if desc1.shape[0] == 0 or desc2.shape[0] < 2:  # We cannot perform snn check, so output empty matches
        return _no_match(desc1)
    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)
    if is_mps_tensor_safe(distance_matrix):
        ratio, idxs_in_2 = _first_to_second_nearest_ratio(distance_matrix)
    else:
        vals, idxs = torch.topk(distance_matrix, 2, dim=1, largest=False)
        ratio, idxs_in_2 = vals[:, 0] / vals[:, 1], idxs[:, 0]
    mask = ratio <= th
    match_dists = ratio[mask]
    if len(match_dists) == 0:
        return _no_match(distance_matrix)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=distance_matrix.device)[mask]
    idxs_in_2 = idxs_in_2[mask]
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_smnn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.95, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match mutual nearest neighbors of desc1 and desc2 that also pass the second-nearest ratio test both ways.

    The ratio of the first to the second nearest neighbor distance must be <= th in both directions.

    Convention:
        - See the Convention block on :func:`match_nn`. A pair is kept when the two descriptors are mutual nearest
          neighbours and pass the ratio test, ``ratio <= th``, in both directions; ``dists`` is the larger of the
          two ratios.
        - ``th`` defaults to 0.95, where :func:`match_snn` and :func:`match_fginn` default to 0.8.

    Args:
        desc1: Batch of descriptors of a shape ``(B1, D)``.
        desc2: Batch of descriptors of a shape ``(B2, D)``.
        th: distance ratio threshold.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of ``(B1, B2)``.

    Return:
        - The larger of the two distance ratios of each match, shape of ``(B3, 1)``.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
          shape of ``(B3, 2)`` where 0 <= B3 <= B1.

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

    Convention:
        See the Convention blocks on :func:`match_nn` and :func:`match_smnn`. For each batch item it returns the
        pairs and ratios that :func:`match_smnn` returns for the valid descriptors of that item, with the batch
        index as the first column of the indices. For float16 and bfloat16 descriptors these are the results of
        :func:`match_smnn` on float32 copies, with the ratios rounded back, and a few pairs and ratios can differ
        slightly from those of :func:`match_smnn` on the half-precision descriptors.

    Args:
        desc1: First descriptors, shape ``(B, N, D)``.
        desc2: Second descriptors, shape ``(B, M, D)``.
        th: Inclusive distance-ratio threshold.
        dm: Optional precomputed L2 distances, shape ``(B, N, M)``.
        mask1: Optional boolean mask of valid first descriptors, shape ``(B, N)``.
        mask2: Optional boolean mask of valid second descriptors, shape ``(B, M)``.
            Masks allow padding variable-length pairs; padding never becomes a neighbor.

    Returns:
        - Ratios, shape ``(K, 1)``, with the input device and dtype.
        - Long indices, shape ``(K, 3)``: ``(batch_index, index_in_desc1, index_in_desc2)``.
          Indices refer to the original padded tensors, sorted by batch and first-descriptor index.
          A pair with fewer than two valid descriptors on either side contributes no matches.

    Note:
        Memory for the distance matrix scales as :math:`B N M`. Bucket pairs by
        descriptor counts and limit batch size when matching large collections.
        Half-precision descriptors use float32 distance computation before ratios
        are converted back to the input dtype.
        Masked inputs compute the distances directly, without the matrix-multiplication
        form, which avoids its cancellation but can be slower for large masked descriptor sets.
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
        # Without the matrix-multiplication form, which it takes for unmasked pairs with more than
        # 25 descriptors on a side, `torch.cdist` on MPS holds all (B, N, M, D) differences at once.
        mps_direct = is_mps_tensor_safe(desc1) and (masked or max(n, m) <= 25)
        if _torch_cdist_available(desc1) and not mps_direct:
            distances = torch.cdist(
                work1,
                work2,
                compute_mode="donot_use_mm_for_euclid_dist" if masked else "use_mm_for_euclid_dist_if_necessary",
            )
        else:
            # Export, MPS before torch 2.13 and direct distances on MPS: accumulate direct differences
            # in descriptor chunks to bound temporary memory without looping over pairs.
            squared = work1.new_zeros((batch, n, m))
            for start in range(0, dim, 16):
                difference = work1[:, :, None, start : start + 16] - work2[:, None, :, start : start + 16]
                squared = squared + (difference * difference).sum(-1)
            positive = squared > 0
            distances = torch.where(positive, torch.where(positive, squared, 1.0).sqrt(), 0.0)
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
    """Match desc1 to desc2 by a ratio test against the first geometrically inconsistent nearest neighbor (FGINN).

    The ratio test compares the nearest distance with that of the nearest candidate whose ``lafs2`` centre lies at
    least ``spatial_th`` from the first one's; closer candidates are skipped (see :cite:`MODS2015` for more details).

    Convention:
        - See the Convention block on :func:`match_nn`. ``dists`` is a ratio: the nearest distance over the distance
          of the second neighbour, which is the nearest of the ``min(10, B2)`` nearest candidates, the first one
          excluded, whose ``lafs2`` centre lies at least ``spatial_th`` from the centre of the first one. Candidates
          closer than that count as the same structure, not as competitors, whatever their descriptor distance. When
          all of them are -- a dense cluster of detections on one structure -- the ratio is 0 and the match is
          **accepted**: no distinct competing structure among the candidates means the nearest neighbour is
          unambiguous. A descriptor beyond the ``min(10, B2)`` nearest is never the second neighbour, even when every
          candidate is excluded.
        - Only the centres of ``lafs2`` are read, and ``spatial_th`` is in their units: pixels for LAFs in pixels.

    Args:
        desc1: Batch of descriptors of a shape ``(B1, D)``.
        desc2: Batch of descriptors of a shape ``(B2, D)``.
        lafs1: LAFs of a shape ``(1, B1, 2, 3)``. Accepted for API symmetry with
          :func:`~kornia.feature.match_adalam` but not read by this function -- only
          ``lafs2`` feeds the geometric check.
        lafs2: LAFs of a shape ``(1, B2, 2, 3)``.
        th: distance ratio threshold.
        spatial_th: minimal distance between the ``lafs2`` centres of the 1st and the 2nd nearest neighbor, in the
          units of those centres.
        mutual: also perform mutual nearest neighbor check.
        dm: torch.Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of ``(B1, B2)``.

    Return:
        - Ratio of the nearest to the second-neighbour distance of each match, see the Convention block, shape of
          ``(B3, 1)``.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2. Shape: ``(B3, 2)``,
          where 0 <= B3 <= B1.

    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])

    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)

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
    # Candidates closer than spatial_th to the 1st nearest neighbour are masked out, whatever the scale of the
    # distances. When none is left, the 2nd nearest neighbour distance is inf and the ratio is 0. torch.where also
    # promotes an integer dm to the default floating dtype, which can hold inf.
    fginn_vals = torch.where(kdist[:, 1:] < spatial_th, float("inf"), vals_cand[:, 1:])
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

    Convention:
        See the Convention block on :func:`~kornia.feature.match_nn`. ``forward`` returns what the function of
        ``match_mode`` (case-insensitive) returns for the two descriptor sets, with the module's ``th`` for ``snn``
        and ``smnn``. ``th`` defaults to 0.8 for both, while :func:`~kornia.feature.match_smnn` defaults to 0.95,
        and ``nn`` and ``mnn`` do not read it. There is no ``dm``.

    Args:
        match_mode: type of matching, can be `nn`, `snn`, `mnn`, `smnn`.
        th: distance ratio threshold of the `snn` and `smnn` modes.

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
            desc1: Batch of descriptors of a shape ``(B1, D)``.
            desc2: Batch of descriptors of a shape ``(B2, D)``.

        Returns:
            - ``dists`` of the matching function, see :func:`~kornia.feature.match_nn`, shape of ``(B3, 1)``.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
                shape of ``(B3, 2)`` where :math:`0 <= B3 <= B1`.

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
            desc1: Batch of descriptors of a shape ``(B1, D)``.
            desc2: Batch of descriptors of a shape ``(B2, D)``.
            normalize: bool to decide whether to F.normalize descriptors to unit norm.
            subset_size: If set, the subset size to use for determining optimal
                number of rotations. Smaller subset size leads to faster but less
                accurate matching. Only used when `self.steer_mode` is `"global"`.

        Returns:
            - Descriptor distance of matching descriptors, shape of ``(B3, 1)``.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
                shape of ``(B3, 2)`` where :math:`0 <= B3 <= B1`.
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

    Convention:
        See the Convention block on :func:`~kornia.feature.match_nn`. ``forward`` returns what the function of
        ``match_mode`` (case-insensitive) returns:

        - ``"fginn"`` reads the keys ``th``, ``spatial_th`` and ``mutual`` of ``params`` and ignores any other key.
          ``th`` defaults to 0.85, while :func:`~kornia.feature.match_fginn` defaults to 0.8.
        - ``"adalam"`` passes ``params`` as the AdaLAM ``config``, and no ``hw1`` or ``hw2``, so AdaLAM takes the
          extent of each image's keypoints as the image size.

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
            desc1: Batch of descriptors of a shape ``(B1, D)``.
            desc2: Batch of descriptors of a shape ``(B2, D)``.
            lafs1: LAFs of a shape ``(1, B1, 2, 3)``.
            lafs2: LAFs of a shape ``(1, B2, 2, 3)``.

        Returns:
            - ``dists`` of the matching function, see :func:`~kornia.feature.match_nn`, shape of ``(B3, 1)``.
            - Long torch.Tensor indexes of matching descriptors in desc1 and desc2,
                shape of ``(B3, 2)`` where :math:`0 <= B3 <= B1`.

        """
        if self.match_mode == "fginn":
            params = _get_default_fginn_params()
            params.update(self.params)
            out = match_fginn(desc1, desc2, lafs1, lafs2, params["th"], params["spatial_th"], params["mutual"])
        elif self.match_mode == "adalam":
            out = match_adalam(desc1, desc2, lafs1, lafs2, config=self.params)
        else:
            raise NotImplementedError
        return out
