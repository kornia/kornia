import math
from typing import Optional, Tuple, Union

import torch
from typing_extensions import NotRequired, TypedDict

from kornia.core import Tensor, concatenate, tensor, where

from .ransac import ransac
from .utils import dist_matrix, orientation_diff


class AdalamConfig(TypedDict):
    """A config structure for the Adalam model.

    Args:
        area_ratio: Ratio between seed circle area and image area. Higher values produce more seeds with smaller
        neighborhoods
    search_expansion: Expansion factor of the seed circle radius for the purpose of collecting neighborhoods.
        Increases neighborhood radius without changing seed distribution
    ransac_iters: Fixed number of inner GPU-RANSAC iterations
    min_inliers: Minimum number of inliers required to accept inliers coming from a neighborhood
    min_confidence: Threshold used by the confidence-based GPU-RANSAC
    orientation_difference_threshold: Maximum difference in orientations for a point to be accepted in a
        neighborhood. Set to None to disable the use of keypoint orientations
    scale_rate_threshold: Maximum difference (ratio) in scales for a point to be accepted in a neighborhood. Set
        to None to disable the use of keypoint scales
    detected_scale_rate_threshold: Prior on maximum possible scale change detectable in image couples. Affinities
        with higher scale changes are regarded as outliers
    refit: Whether to perform refitting at the end of the RANSACs. Generally improves accuracy at the cost of
        runtime
    force_seed_mnn: Whether to consider only MNN for the purpose of selecting seeds. Generally improves accuracy
        at the cost of runtime
    device: Device to be used for running AdaLAM. Use GPU if available.
    mnn: Default None. You can provide a MNN mask in input to skip MNN computation and still get the improvement.
    """

    area_ratio: NotRequired[int]
    search_expansion: NotRequired[int]
    ransac_iters: NotRequired[int]
    min_inliers: NotRequired[int]
    min_confidence: NotRequired[int]
    orientation_difference_threshold: NotRequired[int]
    scale_rate_threshold: NotRequired[float]
    detected_scale_rate_threshold: NotRequired[int]
    refit: NotRequired[bool]
    force_seed_mnn: NotRequired[bool]
    device: NotRequired[torch.device]
    mnn: NotRequired[Tensor]


def _no_match(dm: Tensor) -> Tuple[Tensor, Tensor]:
    """Helper function, which output empty tensors.

    Returns:
            - Descriptor distance of matching descriptors, shape of :math:`(0, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(0, 2)`.
    """
    dists = torch.empty(0, 1, device=dm.device, dtype=dm.dtype)
    idxs = torch.empty(0, 2, device=dm.device, dtype=torch.long)
    return dists, idxs


def select_seeds(
    dist1: Tensor, R1: Union[float, Tensor], scores1: Tensor, fnn12: Tensor, mnn: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    """Select seed correspondences among the set of available matches.

    dist1: Precomputed distance matrix between keypoints in image I_1
    R1: Base radius of neighborhoods in image I_1
    scores1: Confidence scores on the putative_matches. Usually holds Lowe's ratio scores.
    fnn12: Matches between keypoints of I_1 and I_2.
           The i-th entry of fnn12 is j if and only if keypoint k_i in image I_1 is matched to keypoint k_j in image I_2
    mnn: A mask indicating which putative matches are also mutual nearest neighbors. See documentation on 'force_seed_mnn' in the DEFAULT_CONFIG.
         If None, it disables the mutual nearest neighbor filtering on seed point selection.
         Expected a bool tensor with shape (num_keypoints_in_source_image,)

    Returns:
        Indices of seed points.

        im1seeds: Keypoint index of chosen seeds in image I_1
        im2seeds: Keypoint index of chosen seeds in image I_2
    """  # noqa: E501
    im1neighmap = dist1 < R1**2  # (n1, n1)
    # find out who scores higher than whom
    im1scorescomp = scores1.unsqueeze(1) > scores1.unsqueeze(0)  # (n1, n1)
    # find out who scores higher than all of its neighbors: seed points
    if mnn is not None:
        im1bs = (~torch.any(im1neighmap & im1scorescomp & mnn.unsqueeze(0), dim=1)) & mnn & (scores1 < 0.8**2)  # (n1,)
    else:
        im1bs = (~torch.any(im1neighmap & im1scorescomp, dim=1)) & (scores1 < 0.8**2)

    # collect all seeds in both images and the 1NN of the seeds of the other image
    im1seeds = where(im1bs)[0]  # (n1bs) index format
    im2seeds = fnn12[im1bs]  # (n1bs) index format
    return im1seeds, im2seeds


def extract_neighborhood_sets(
    o1: Optional[Tensor],
    o2: Optional[Tensor],
    s1: Optional[Tensor],
    s2: Optional[Tensor],
    dist1: Tensor,
    im1seeds: Tensor,
    im2seeds: Tensor,
    k1: Tensor,
    k2: Tensor,
    R1: Union[float, Tensor],
    R2: Union[float, Tensor],
    fnn12: Tensor,
    ORIENTATION_THR: float,
    SCALE_RATE_THR: float,
    SEARCH_EXP: float,
    MIN_INLIERS: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Assign keypoints to seed points. This checks both the distance and the agreement of the local transformation
    if available.

    o1: Orientations of keypoints in image I_1
    o2: Orientations of keypoints in image I_2
    s1: Scales of keypoints in image I_1
    s2: Scales of keypoints in image I_2
    dist1: Precomputed distance matrix between keypoints in image I_1
    im1seeds: Keypoint index of chosen seeds in image I_1
    im2seeds: Keypoint index of chosen seeds in image I_2
    k1: Keypoint locations in image I_1
    k2: Keypoint locations in image I_2
    R1: Base radius of neighborhoods in image I_1
    R2: Base radius of neighborhoods in image I_2
    fnn12: Matches between keypoints of I_1 and I_2.
           The i-th entry of fnn12 is j if and only if keypoint k_i in image I_1 is matched to keypoint k_j in image I_2
    ORIENTATION_THR: Maximum deviation of orientation with respect to seed S_i to keep a keypoint in i-th neighborhood
    SCALE_RATE_THR: Maximum deviation of scale with respect to seed S_i to keep a keypoint in i-th neighborhood
    SEARCH_EXP: Expansion rate for both radii R1 and R2 to consider inclusion of neighboring keypoints
    MIN_INLIERS: Minimum number of inliers to keep a seed point. This is used as an early filter here
                 to remove already seeds with not enough samples to ever pass this threshold.

    Returns:
        Local neighborhoods assignments:

        local_neighs_mask: Boolean matrix of size (num_seeds, num_keypoints).
                           Entry (i, j) is True iff keypoint j was assigned to seed i.
        rdims: Number of keypoints included in the neighborhood for each seed
        im1seeds: Keypoint index of chosen seeds in image I_1
        im2seeds: Keypoint index of chosen seeds in image I_2
    """
    dst1 = dist1[im1seeds, :]
    dst2 = dist_matrix(k2[fnn12[im1seeds]], k2[fnn12])

    # initial candidates are matches which are close to the same seed in both images
    local_neighs_mask = (dst1 < (SEARCH_EXP * R1) ** 2) & (dst2 < (SEARCH_EXP * R2) ** 2)

    # If requested, also their orientation delta should be compatible with that of the corresponding seed
    if ORIENTATION_THR is not None and ORIENTATION_THR < 180 and (o1 is not None) and (o2 is not None):
        relo = orientation_diff(o1, o2[fnn12])
        orientation_diffs = torch.abs(orientation_diff(relo.unsqueeze(0), relo[im1seeds].unsqueeze(1)))
        local_neighs_mask = local_neighs_mask & (orientation_diffs < ORIENTATION_THR)

    # If requested, also their scale delta should be compatible with that of the corresponding seed
    if SCALE_RATE_THR is not None and (SCALE_RATE_THR < 10) and (s1 is not None) and (s2 is not None):
        rels = s2[fnn12] / s1
        scale_rates = rels[im1seeds].unsqueeze(1) / rels.unsqueeze(0)
        local_neighs_mask = (
            local_neighs_mask & (scale_rates < SCALE_RATE_THR) & (scale_rates > 1 / SCALE_RATE_THR)
        )  # (ns, n1)

    # count how many keypoints ended up in each neighborhood
    numn1 = torch.sum(local_neighs_mask, dim=1)
    # and only keep the ones that have enough points
    valid_seeds = numn1 >= MIN_INLIERS

    local_neighs_mask = local_neighs_mask[valid_seeds, :]

    rdims = numn1[valid_seeds]

    return local_neighs_mask, rdims, im1seeds[valid_seeds], im2seeds[valid_seeds]


def extract_local_patterns(
    fnn12: Tensor,
    fnn_to_seed_local_consistency_map_corr: Tensor,
    k1: Tensor,
    k2: Tensor,
    im1seeds: Tensor,
    im2seeds: Tensor,
    scores: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Prepare local neighborhoods around each seed for the parallel RANSACs. This involves two steps: 1) Collect
    all selected keypoints and refer them with respect to their seed point 2) Sort keypoints by score for the
    progressive sampling to pick the best samples first.

    fnn12: Matches between keypoints of I_1 and I_2.
           The i-th entry of fnn12 is j if and only if keypoint k_i in image I_1 is matched to keypoint k_j in image I_2
    fnn_to_seed_local_consistency_map_corr: Boolean matrix of size (num_seeds, num_keypoints).
                                            Entry (i, j) is True iff keypoint j was assigned to seed i.
    k1: Keypoint locations in image I_1
    k2: Keypoint locations in image I_2
    im1seeds: Keypoint index of chosen seeds in image I_1
    im2seeds: Keypoint index of chosen seeds in image I_2
    scores: Scores to rank correspondences by confidence.
            Lower scores are assumed to be more confident, consistently with Lowe's ratio scores.
            Note: scores should be between 0 and 1 for this function to work as expected.

    Returns:
        All information required for running the parallel RANSACs.
        Data is formatted so that all inputs for different RANSACs are concatenated
            along the same dimension to support different input sizes.

        im1loc: Keypoint locations in image I_1 for each RANSAC sample.
        im2loc: Keypoint locations in image I_2 for each RANSAC sample.
        ransidx: Integer identifier of the RANSAC problem.
                 This allows to distinguish inputs belonging to the same problem.
        tokp1: Index of the original keypoint in image I_1 for each RANSAC sample.
        tokp2: Index of the original keypoint in image I_2 for each RANSAC sample.
    """
    # first get an indexing representation of the assignments:
    # - ransidx holds the index of the seed for each assignment
    # - tokp1 holds the index of the keypoint in image I_1 for each assignment
    ransidx, tokp1 = where(fnn_to_seed_local_consistency_map_corr)
    # - and of course tokp2 holds the index of the corresponding keypoint in image I_2
    tokp2 = fnn12[tokp1]

    # Now take the locations in the image of each considered keypoint ...
    im1abspattern = k1[tokp1]
    im2abspattern = k2[tokp2]

    # ... and subtract the location of its corresponding seed to get relative coordinates
    im1loc = im1abspattern - k1[im1seeds[ransidx]]
    im2loc = im2abspattern - k2[im2seeds[ransidx]]

    # Finally we need to sort keypoints by scores in a way that assignments to the same seed are close together
    # To achieve this we assume scores lie in (0, 1) and add the integer index of the corresponding seed
    expanded_local_scores = scores[tokp1] + ransidx.type(scores.dtype)

    sorting_perm = torch.argsort(expanded_local_scores)

    return im1loc[sorting_perm], im2loc[sorting_perm], ransidx, tokp1[sorting_perm], tokp2[sorting_perm]


def adalam_core(
    k1: Tensor,
    k2: Tensor,
    fnn12: Tensor,
    scores1: Tensor,
    config: AdalamConfig,
    mnn: Optional[Tensor] = None,
    im1shape: Optional[Tuple[int, int]] = None,
    im2shape: Optional[Tuple[int, int]] = None,
    o1: Optional[Tensor] = None,
    o2: Optional[Tensor] = None,
    s1: Optional[Tensor] = None,
    s2: Optional[Tensor] = None,
    return_dist: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tensor]:
    """Call the core functionality of AdaLAM, i.e. just outlier filtering. No sanity check is performed on the
    inputs.

    Inputs:
        k1: keypoint locations in the source image, in pixel coordinates.
            Expected a float32 tensor with shape (num_keypoints_in_source_image, 2).
        k2: keypoint locations in the destination image, in pixel coordinates.
            Expected a float32 tensor with shape (num_keypoints_in_destination_image, 2).
        fn12: Initial set of putative matches to be filtered.
              The current implementation assumes that these are unfiltered nearest neighbor matches,
              so it requires this to be a list of indices a_i such that the source keypoint i is associated to the
              destination keypoint a_i. For now to use AdaLAM on different inputs a workaround on the input format is
              required. Expected a long tensor with shape (num_keypoints_in_source_image,).
        scores1: Confidence scores on the putative_matches. Usually holds Lowe's ratio scores.
        mnn: A mask indicating which putative matches are also mutual nearest neighbors. See documentation on
             'force_seed_mnn' in the DEFAULT_CONFIG. If None, it disables the mutual nearest neighbor filtering on seed
             point selection. Expected a bool tensor with shape (num_keypoints_in_source_image,)
        im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of wasted
                  runtime. So please provide it. Expected a tuple with (width, height) or (height, width) of source
                  image
        im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost of
                  wasted runtime. So please provide it. Expected a tuple with (width, height) or (height, width) of
                  destination image
        o1/o2: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config is
               set to None. See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
               Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)
        s1/s2: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
               See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
               Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)
        return_dist: if True, inverse confidence value is also outputted. Default is False

    Returns:
        idxs: A long tensor with shape (num_filtered_matches, 2) with indices of corresponding keypoints in k1 and k2.
        dists: inverse confidence ratio.
    """
    AREA_RATIO = config["area_ratio"]
    SEARCH_EXP = config["search_expansion"]
    RANSAC_ITERS = config["ransac_iters"]
    MIN_INLIERS = config["min_inliers"]
    MIN_CONF = config["min_confidence"]
    ORIENTATION_THR = config["orientation_difference_threshold"]
    SCALE_RATE_THR = config["scale_rate_threshold"]
    REFIT = config["refit"]

    if isinstance(im1shape, tuple):
        _im1shape = tensor(im1shape, device=k1.device, dtype=k1.dtype)
    else:
        k1mins = k1.min(dim=0).values
        k1maxs = k1.max(dim=0).values
        _im1shape = k1maxs - k1mins

    if isinstance(im2shape, tuple):
        _im2shape = tensor(im2shape, device=k2.device, dtype=k2.dtype)
    else:
        k2mins = k2.min(dim=0).values
        k2maxs = k2.max(dim=0).values
        _im2shape = k2maxs - k2mins

    # Compute seed selection radii to be invariant to image rescaling
    R1 = torch.sqrt(torch.prod(_im1shape[:2]) / AREA_RATIO / math.pi)
    R2 = torch.sqrt(torch.prod(_im2shape[:2]) / AREA_RATIO / math.pi)

    # Precompute the inner distances of keypoints in image I_1
    dist1 = dist_matrix(k1, k1)

    # Select seeds
    im1seeds, im2seeds = select_seeds(dist1, R1, scores1, fnn12, mnn)

    # Find the neighboring and coherent keyopints consistent with each seed
    local_neighs_mask, rdims, im1seeds, im2seeds = extract_neighborhood_sets(
        o1,
        o2,
        s1,
        s2,
        dist1,
        im1seeds,
        im2seeds,
        k1,
        k2,
        R1,
        R2,
        fnn12,
        ORIENTATION_THR,
        SCALE_RATE_THR,
        SEARCH_EXP,
        MIN_INLIERS,
    )

    if rdims.shape[0] == 0:
        # No seed point survived. Just output ratio-test matches. This should happen very rarely.
        score_mask = scores1 <= 0.95
        absolute_im1idx = where(score_mask)[0]
        if len(absolute_im1idx) > 0:
            absolute_im2idx = fnn12[absolute_im1idx]
            out_scores = scores1[score_mask].reshape(-1, 1)
            idxs = torch.stack([absolute_im1idx, absolute_im2idx], dim=1)
        else:
            idxs, out_scores = _no_match(scores1)
        if return_dist:
            return idxs, out_scores
        else:
            return idxs

    # Format neighborhoods for parallel RANSACs
    im1loc, im2loc, ransidx, tokp1, tokp2 = extract_local_patterns(
        fnn12, local_neighs_mask, k1, k2, im1seeds, im2seeds, scores1
    )
    im1loc = im1loc / (R1 * SEARCH_EXP)
    im2loc = im2loc / (R2 * SEARCH_EXP)

    # Run the parallel confidence-based RANSACs to perform local affine verification
    inlier_idx, _, inl_confidence, inlier_counts = ransac(
        xsamples=im1loc, ysamples=im2loc, rdims=rdims, iters=RANSAC_ITERS, refit=REFIT, config=dict(config)
    )

    conf = inl_confidence[ransidx[inlier_idx]]
    cnt = inlier_counts[ransidx[inlier_idx]].float()
    dist_ratio = 1.0 / conf
    passed_inliers_mask = (conf >= MIN_CONF) & (cnt * (1 - dist_ratio) >= MIN_INLIERS)
    accepted_inliers = inlier_idx[passed_inliers_mask]
    accepted_dist = dist_ratio[passed_inliers_mask]

    absolute_im1idx = tokp1[accepted_inliers]
    absolute_im2idx = tokp2[accepted_inliers]

    final_matches = torch.stack([absolute_im1idx, absolute_im2idx], dim=1)
    if final_matches.shape[0] > 1:
        # https://stackoverflow.com/a/72005790
        final_matches, idxs, counts = torch.unique(final_matches, dim=0, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idxs)
        cum_sum = counts.cumsum(0)
        cum_sum = concatenate((torch.tensor([0], dtype=cum_sum.dtype, device=cum_sum.device), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        accepted_dist = accepted_dist[first_indicies]
    if return_dist:
        return final_matches, accepted_dist.reshape(-1, 1)
    return final_matches
