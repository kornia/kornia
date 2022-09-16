from typing import Optional, Tuple

import torch
import torch.nn as nn


from kornia.testing import KORNIA_CHECK_SHAPE, is_mps_tensor_safe


def _cdist(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    r"""Manual `torch.cdist` for M1"""
    if (not is_mps_tensor_safe(d1)) and (not is_mps_tensor_safe(d2)):
        return torch.cdist(d1, d2)
    d1_sq = (d1 ** 2).sum(dim=1, keepdim=True)
    d2_sq = (d2 ** 2).sum(dim=1, keepdim=True)
    dm = d1_sq.repeat(1, d2.size(0)) + d2_sq.repeat(1, d1.size(0)).t() - 2.0 * d1 @ d2.t()
    dm = dm.clamp(min=0.0).sqrt()
    return dm


def match_nn(
    desc1: torch.Tensor, desc2: torch.Tensor, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    if dm is None:
        dm = _cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError
    match_dists, idxs_in_2 = torch.min(dm, dim=1)
    idxs_in1: torch.Tensor = torch.arange(0, idxs_in_2.size(0), device=idxs_in_2.device)
    matches_idxs: torch.Tensor = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_mnn(
    desc1: torch.Tensor, desc2: torch.Tensor, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    if dm is None:
        dm = _cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    ms = min(dm.size(0), dm.size(1))
    match_dists, idxs_in_2 = torch.min(dm, dim=1)
    match_dists2, idxs_in_1 = torch.min(dm, dim=0)
    minsize_idxs = torch.arange(ms, device=dm.device)

    if dm.size(0) <= dm.size(1):
        mutual_nns = minsize_idxs == idxs_in_1[idxs_in_2][:ms]
        matches_idxs = torch.cat([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)[mutual_nns]
        match_dists = match_dists[mutual_nns]
    else:
        mutual_nns = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
        matches_idxs = torch.cat([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], dim=1)[mutual_nns]
        match_dists = match_dists2[mutual_nns]
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_snn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = _cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    vals, idxs_in_2 = torch.topk(dm, 2, dim=1, largest=False)
    ratio = vals[:, 0] / vals[:, 1]
    mask = ratio <= th
    match_dists = ratio[mask]
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=dm.device)[mask]
    idxs_in_2 = idxs_in_2[:, 0][mask]
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_smnn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    if desc1.shape[0] < 2:
        raise AssertionError
    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = _cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    dists1, idx1 = match_snn(desc1, desc2, th, dm)
    dists2, idx2 = match_snn(desc2, desc1, th, dm.t())

    if len(dists2) > 0 and len(dists1) > 0:
        idx2 = idx2.flip(1)
        if not is_mps_tensor_safe(idx1):
            idxs_dm = torch.cdist(idx1.float(), idx2.float(), p=1.0)
        else:
            idxs2_rep = idx2.float().repeat_interleave(idx1.size(0), dim=0)
            idxs_dm = (idx1.float().repeat(idx2.size(0), 1) - idxs2_rep).abs().sum(dim=1)
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
    else:
        matches_idxs, match_dists = torch.empty(0, 2, device=dm.device), torch.empty(0, 1, device=dm.device)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


class DescriptorMatcher(nn.Module):
    """Module version of matching functions.

    See :func:`~kornia.feature.match_nn`, :func:`~kornia.feature.match_snn`,
        :func:`~kornia.feature.match_mnn` or :func:`~kornia.feature.match_smnn` for more details.

    Args:
        match_mode: type of matching, can be `nn`, `snn`, `mnn`, `smnn`.
        th: threshold on distance ratio, or other quality measure.
    """
    known_modes = ['nn', 'mnn', 'snn', 'smnn']

    def __init__(self, match_mode: str = 'snn', th: float = 0.8) -> None:
        super().__init__()
        _match_mode: str = match_mode.lower()
        if _match_mode not in self.known_modes:
            raise NotImplementedError(f"{match_mode} is not supported. Try one of {self.known_modes}")
        self.match_mode = _match_mode
        self.th = th

    def forward(self, desc1: torch.Tensor, desc2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
        """
        if self.match_mode == 'nn':
            out = match_nn(desc1, desc2)
        elif self.match_mode == 'mnn':
            out = match_mnn(desc1, desc2)
        elif self.match_mode == 'snn':
            out = match_snn(desc1, desc2, self.th)
        elif self.match_mode == 'smnn':
            out = match_smnn(desc1, desc2, self.th)
        else:
            raise NotImplementedError
        return out
