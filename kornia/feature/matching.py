import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from typing import Tuple, Optional
TupleTensor = Tuple[torch.Tensor, torch.Tensor]


def match_nn(desc1: torch.Tensor, desc2: torch.Tensor) -> TupleTensor:
    '''Function, which finds nearest neightbors for each vector in desc1.
    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 desriptor distance


    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B1, 2)`, :math:`(B1, 1)`
    '''
    dm = torch.cdist(desc1, desc2)
    match_dists, idxs_in_2 = torch.min(dm, dim=1)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0))
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    return matches_idxs.view(-1, 2), match_dists.view(-1, 1)


def match_mnn(desc1: torch.Tensor, desc2: torch.Tensor) -> TupleTensor:
    '''Function, which finds mutual nearest neightbors for each vector in desc1 and desc2,
    which satisfies first to second nearest neighbor distance <= th check
    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 desriptor distance

    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B3, 2)`, :math:`(B3, 1)` where 0 <= B3 <= min(B1,B2)
    '''
    dm = torch.cdist(desc1, desc2)
    ms = min(dm.size(0), dm.size(1))
    match_dists, idxs_in_2 = torch.min(dm, dim=1)
    match_dists2, idxs_in_1 = torch.min(dm, dim=0)
    minsize_idxs = torch.arange(ms)
    if dm.size(0) <= dm.size(1):
        mutual_nns = minsize_idxs == idxs_in_1[idxs_in_2][:ms]
        matches_idxs = torch.cat([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)[mutual_nns]
        match_dists = match_dists[mutual_nns]
    else:
        mutual_nns = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
        matches_idxs = torch.cat([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], dim=1)[mutual_nns]
        match_dists = match_dists2[mutual_nns]
    return matches_idxs.view(-1, 2), match_dists.view(-1, 1)


def match_snn(desc1: torch.Tensor, desc2: torch.Tensor,
              th: float = 0.8, dm=None) -> TupleTensor:
    '''Function, which finds mutual nearest neightbors for each vector in desc1 and desc2,
    which satisfy first to second nearest neighbor distance <= th check in both directions.
    So, it is intersection of match_mnn(d1,d2), match_snn(d1,d2), match_mnn(d2,d1)

    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 desriptor distance ratio 1st to 2nd nearest neighbor


    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B3, 2)`, :math:`(B3, 1)`, where 0 <= B3 <= min(B1, B2)
    '''
    if dm is None:
        dm = torch.cdist(desc1, desc2)
    vals, idxs_in_2 = torch.topk(dm, 2, dim=1, largest=False)
    ratio = vals[:, 0] / vals[:, 1]
    mask = ratio <= th
    match_dists = ratio[mask]
    idxs_in1 = torch.arange(0, idxs_in_2.size(0))[mask]
    idxs_in_2 = idxs_in_2[:, 0][mask]
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.cpu().view(-1, 1)], dim=1)
    return matches_idxs.view(-1, 2), match_dists.view(-1, 1)


def match_smnn(desc1: torch.Tensor, desc2: torch.Tensor,
               th: float = 0.8) -> TupleTensor:
    '''Function, which finds mutual nearest neightbors for each vector in desc1 and desc2,
    which satisfy first to second nearest neighbor distance <= th check in both directions.
    So, it is intersection of match_mnn(d1,d2), match_snn(d1,d2), match_snn(d2,d1)
    Resulting distance ratio should be maximum over over distance ratio in both directions
    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 desriptor distance ratio 1st to 2nd nearest neighbor


    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B3, 2)`, :math:`(B3, 1)`, where 0 <= B3 <= min(B1, B2)
    '''
    dm = torch.cdist(desc1, desc2)
    idx1, dists1 = match_snn(desc1, desc2, th, dm)
    idx2, dists2 = match_snn(desc2, desc1, th, dm.t())
    if len(dists2) > 0 and len(dists1) > 0:
        idx2 = idx2.flip(1)
        idxs_dm = torch.cdist(idx1.float(), idx2.float())
        mutual_idxs1 = idxs_dm.min(dim=1)[0] < 1e-8
        mutual_idxs2 = idxs_dm.min(dim=0)[0] < 1e-8
        good_idxs1 = idx1[mutual_idxs1.view(-1)]
        good_idxs2 = idx2[mutual_idxs2.view(-1)]
        dists1_good = dists1[mutual_idxs1.view(-1)]
        dists2_good = dists2[mutual_idxs2.view(-1)]
        _, idx_upl1 = torch.sort(good_idxs1[:, 0])
        _, idx_upl2 = torch.sort(good_idxs2[:, 0])
        good_idxs1 = good_idxs1[idx_upl1]
        good_idxs2 = good_idxs2[idx_upl2]
        match_dists = torch.max(dists1_good[idx_upl1], dists2_good[idx_upl2])
        matches_idxs = good_idxs1
    else:
        matches_idxs, match_dists = torch.empty(0, 2), torch.empty(0, 1)
    return matches_idxs.view(-1, 2), match_dists.view(-1, 1)
