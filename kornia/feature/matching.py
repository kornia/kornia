from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from kornia.feature import extract_patches_from_pyramid, get_laf_center


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
    if len(desc1.shape) != 2:
        raise AssertionError
    if len(desc2.shape) != 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
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
    if len(desc1.shape) != 2:
        raise AssertionError
    if len(desc2.shape) != 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
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
    if len(desc1.shape) != 2:
        raise AssertionError
    if len(desc2.shape) != 2:
        raise AssertionError
    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
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
    if len(desc1.shape) != 2:
        raise AssertionError
    if len(desc2.shape) != 2:
        raise AssertionError
    if desc1.shape[0] < 2:
        raise AssertionError
    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    dists1, idx1 = match_snn(desc1, desc2, th, dm)
    dists2, idx2 = match_snn(desc2, desc1, th, dm.t())

    if len(dists2) > 0 and len(dists1) > 0:
        idx2 = idx2.flip(1)
        idxs_dm = torch.cdist(idx1.float(), idx2.float(), p=1.0)
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
    """Module version of matching functions. See :function:`~kornia.feature.match_snn` for details
    Args:
        match_mode: type of matching, can be `nn`, `snn`, `mnn`, `smnn`. Default `snn`.
        th: threshold on distance ratio, or other quality measure. Default 0.8
    """
    known_modes = ['nn', 'mnn', 'snn', 'smnn']

    def __init__(self, match_mode: str = 'snn', th: float = 0.8) -> None:
        super().__init__()
        if match_mode.lower() not in self.known_modes:
            raise NotImplementedError(f"{match_mode} is not supported. Try one of {self.known_modes}")
        self.match_mode = match_mode.lower()
        self.th = th

    def forward(self, desc1: torch.Tensor, desc2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          desc1: Batch of descriptors of a shape :math:`(B1, D)`.
          desc2: Batch of descriptors of a shape :math:`(B2, D)`.

        Return:
          - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
          - Long tensor indexes of matching descriptors in desc1 and desc2,
            shape of :math:`(B3, 2)` where 0 <= B3 <= B1."""
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


class LocalFeatureMatcher(nn.Module):
    r"""Module, which finds correspondences between two images based on local features.
    Args:
        detector: Local feature detector. See :class:`~kornia.feature.ScaleSpaceDetector`
        descriptor: Local patch descriptor, see :class:`~kornia.feature.HardNet`
                    or :class:`~kornia.feature.SIFTDescriptor`
        matcher: Descriptor matcher, see :class:`~kornia.feature.DescriptorMatcher`

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> gftt_hardnet_matcher = LocalFeatureMatcher(kornia.ScaleSpaceDetector(500),
        >>>                                kornia.feature.HardNet(True),
        >>>                                kornia.feature.DescriptorMatcher())
        >>> out = gftt_hardnet_matcher(input)
    """

    def __init__(self, detector, descriptor, matcher):
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.eval()

    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''Function for feature extraction from simple image'''
        lafs0, resps0 = self.detector(image)
        patch_size: int = self.descriptor.patch_size
        patches = extract_patches_from_pyramid(image, lafs0, PS = patch_size)
        B, N, CH, H, W = patches.size()
        descs0 = self.descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
        return {"lafs": lafs0, "responses": resps0, "descriptors": descs0}

    def forward(self, data: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: {
                'image0': (torch.Tensor): (N, 1, H1, W1)
                'image1': (torch.Tensor): (N, 1, H2, W2)
                'mask0'(optional) : (torch.Tensor): (N, H1, W1) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H2, W2)
            }
        Returns:
            out: {
                    "keypoints0": (torch.Tensor): (NC, 2) matching keypoints from image0
                    "keypoints1":  (torch.Tensor): (NC, 2) matching keypoints from image1
                    "confidence": (torch.Tensor): (NC) - confidence score [0, 1]
                    "keypoints0": (torch.Tensor): (1, NC, 2, 3) matching LAFs from image0
                    "keypoints1":  (torch.Tensor): (1, NC, 2, 3) matching LAFs from image1
                    "batch_indexes": (torch.Tensor): (NC) - batch indexes for the keypoints and lafs
            }
        """
        num_image_pairs = data['image0'].shape[0]
        if ('lafs0' not in data.keys()) or ('descriptors0' not in data.keys()):
            # One can supply pre-extracted local features
            feats_dict0 = self.extract_features(data['image0'])
            lafs0, descs0 = feats_dict0['lafs'], feats_dict0['descriptors']
        else:
            lafs0, descs0 = data['lafs0'], data['descriptors0']
        if ('lafs1' not in data.keys()) or ('descriptors1' not in data.keys()):
            feats_dict1 = self.extract_features(data['image1'])
            lafs1, descs1 = feats_dict1['lafs'], feats_dict1['descriptors']
        else:
            lafs1, descs1 = data['lafs1'], data['descriptors1']
        keypoints0 = get_laf_center(lafs0)
        keypoints1 = get_laf_center(lafs1)

        out_keypoints0: List[torch.Tensor] = []
        out_keypoints1: List[torch.Tensor] = []
        out_confidence: List[torch.Tensor] = []
        out_batch_indexes: List[torch.Tensor] = []
        out_lafs0: List[torch.Tensor] = []
        out_lafs1: List[torch.Tensor] = []

        for batch_idx in range(num_image_pairs):
            dists, idxs = self.matcher(descs0[batch_idx], descs1[batch_idx])
            current_keypoints_0 = keypoints0[batch_idx, idxs[:, 0]]
            current_keypoints_1 = keypoints1[batch_idx, idxs[:, 1]]
            current_lafs_0 = lafs0[batch_idx, idxs[:, 0]]
            current_lafs_1 = lafs1[batch_idx, idxs[:, 1]]

            out_confidence.append(1.0 - dists)
            batch_idxs = batch_idx * torch.ones(len(dists),
                                                device=keypoints0.device,
                                                dtype=torch.long)
            out_keypoints0.append(current_keypoints_0)
            out_keypoints1.append(current_keypoints_1)
            out_lafs0.append(current_lafs_0)
            out_lafs1.append(current_lafs_1)
            out_batch_indexes.append(batch_idxs)

        out = {'keypoints0': torch.cat(out_keypoints0, dim=0).view(-1, 2),
               'keypoints1': torch.cat(out_keypoints1, dim=0).view(-1, 2),
               'lafs0': torch.cat(out_lafs0, dim=0).view(1, -1, 2, 3),
               'lafs1': torch.cat(out_lafs1, dim=0).view(1, -1, 2, 3),
               'confidence': torch.cat(out_confidence, dim=0).view(-1),
               'batch_indexes': torch.cat(out_batch_indexes, dim=0).view(-1)}
        return out
