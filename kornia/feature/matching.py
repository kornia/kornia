from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK_DM_DESC, KORNIA_CHECK_SHAPE
from kornia.feature.laf import get_laf_center
from kornia.feature.steerers import DiscreteSteerer
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


class DescriptorMatcherWithSteerer(Module):
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
        >>> import kornia as K
        >>> import kornia.feature as KF
        >>> device = K.utils.get_cuda_or_mps_device_if_available()
        >>> img1 = torch.randn([1, 3, 768, 768], device=device)
        >>> img2 = torch.randn([1, 3, 768, 768], device=device)
        >>> dedode = KF.DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-SO2").to(device)
        >>> steerer_order = 8  # discretisation order of rotation angles
        >>> steerer = KF.steerers.DiscreteSteerer.create_dedode_default(
        ... generator_type="SO2", steerer_order=steerer_order
        ... )
        >>> steerer = steerer.to(device)
        >>> matcher = KF.matching.DescriptorMatcherWithSteerer(
        ... steerer=steerer, steerer_order=steerer_order, steer_mode="global", match_mode="smnn", th=0.98
        ... )
        >>> with torch.inference_mode():
        ...     kps1, scores1, descs1 = dedode(img1, n=20_000)
        ...     kps2, scores2, descs2 = dedode(img2, n=20_000)
        ...     kps1, kps2, descs1, descs2 = kps1[0], kps2[0], descs1[0], descs2[0]
        ...     dists, idxs, num_rot = matcher(
        ...         descs1, descs2, normalize=True, subset_size=1000,
        ...     )
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

    def matching_function(self, d1: Tensor, d2: Tensor, dm: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.match_mode == "nn":
            return match_nn(d1, d2, dm=dm)
        elif self.match_mode == "mnn":
            return match_mnn(d1, d2, dm=dm)
        elif self.match_mode == "snn":
            return match_snn(d1, d2, self.th, dm=dm)
        elif self.match_mode == "smnn":
            return match_smnn(d1, d2, self.th, dm=dm)
        else:
            raise NotImplementedError

    def forward(
        self,
        desc1: Tensor,
        desc2: Tensor,
        normalize: bool = False,
        subset_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Optional[int]]:
        """
        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            normalize: bool to decide whether to normalize descriptors to unit norm.
            subset_size: If set, the subset size to use for determining optimal
                number of rotations. Smaller subset size leads to faster but less
                accurate matching. Only used when `self.steer_mode` is `"global"`.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
            - Number of global rotations from desc1 to desc2, in terms of `self.steerer_order`
                (will be `None` if `self.steer_mode` is `local`).
        """
        rot1to2 = None

        if normalize:
            desc1 = torch.nn.functional.normalize(desc1, dim=-1)
            desc2 = torch.nn.functional.normalize(desc2, dim=-1)

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
            for r in range(1, self.steerer_order):
                desc1 = self.steerer.steer_descriptions(desc1, normalize=normalize)
                dm_new = _cdist(desc1, desc2)
                dm = torch.minimum(dm, dm_new)
            dist, idx = self.matching_function(desc1, desc2, dm)
        else:
            raise NotImplementedError

        return dist, idx, rot1to2


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
