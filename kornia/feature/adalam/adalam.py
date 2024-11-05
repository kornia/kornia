# Integrated from original AdaLAM repo
# https://github.com/cavalli1234/AdaLAM
# Copyright (c) 2020, Luca Cavalli

from typing import Optional, Tuple, Union

import torch

from kornia.core import Tensor, as_tensor
from kornia.core.check import KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE
from kornia.feature.laf import get_laf_center, get_laf_orientation, get_laf_scale

from .core import AdalamConfig, _no_match, adalam_core
from .utils import dist_matrix


def get_adalam_default_config() -> AdalamConfig:
    return AdalamConfig(
        area_ratio=100,
        search_expansion=4,
        ransac_iters=128,
        min_inliers=6,
        min_confidence=200,
        orientation_difference_threshold=30,
        scale_rate_threshold=1.5,
        detected_scale_rate_threshold=5,
        refit=True,
        force_seed_mnn=True,
        device=torch.device("cpu"),
    )


def match_adalam(
    desc1: Tensor,
    desc2: Tensor,
    lafs1: Tensor,
    lafs2: Tensor,
    config: Optional[AdalamConfig] = None,
    hw1: Optional[Tuple[int, int]] = None,
    hw2: Optional[Tuple[int, int]] = None,
    dm: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Function, which performs descriptor matching, followed by AdaLAM filtering (see :cite:`AdaLAM2020` for more
    details)

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
        lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.
        config: dict with AdaLAM config
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
          where 0 <= B3 <= B1.
    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])
    KORNIA_CHECK_LAF(lafs1)
    KORNIA_CHECK_LAF(lafs2)
    config_ = get_adalam_default_config()
    if config is None:
        config_["device"] = desc1.device
    else:
        config_ = get_adalam_default_config()
        for key, val in config.items():
            if key not in config_.keys():
                print(
                    f"WARNING: custom configuration contains a key which is not recognized ({key}). "
                    f"Known configurations are {list(config_.keys())}."
                )
                continue
            # TypedDict does not support variable names. https://stackoverflow.com/a/59583427/1983544
            config_[key] = val  # type: ignore
    adalam_object = AdalamFilter(config_)
    idxs, quality = adalam_object.match_and_filter(
        get_laf_center(lafs1).reshape(-1, 2),
        get_laf_center(lafs2).reshape(-1, 2),
        desc1,
        desc2,
        hw1,
        hw2,
        get_laf_orientation(lafs1).reshape(-1),
        get_laf_orientation(lafs2).reshape(-1),
        get_laf_scale(lafs1).reshape(-1),
        get_laf_scale(lafs2).reshape(-1),
        return_dist=True,
    )
    return quality, idxs


class AdalamFilter:
    def __init__(self, custom_config: Optional[AdalamConfig] = None) -> None:
        """This class acts as a wrapper to the method AdaLAM for outlier filtering.

        init args:
            custom_config: dictionary overriding the default configuration. Missing parameters are kept as default.
                           See documentation of DEFAULT_CONFIG for specific explanations on the accepted parameters.
        """
        if custom_config is not None:
            self.config = custom_config
        else:
            self.config = get_adalam_default_config()

    def filter_matches(
        self,
        k1: Tensor,
        k2: Tensor,
        putative_matches: Tensor,
        scores: Tensor,
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
            putative_matches: Initial set of putative matches to be filtered.
                              The current implementation assumes that these are unfiltered nearest neighbor matches,
                              so it requires this to be a list of indices a_i such that the source keypoint i is
                              associated to the destination keypoint a_i. For now to use AdaLAM on different inputs a
                              workaround on the input format is required.
                              Expected a long tensor with shape (num_keypoints_in_source_image,).
            scores: Confidence scores on the putative_matches. Usually holds Lowe's ratio scores.
            mnn: A mask indicating which putative matches are also mutual nearest neighbors. See documentation on
                 'force_seed_mnn' in the DEFAULT_CONFIG. If None, it disables the mutual nearest neighbor filtering on
                 seed point selection. Expected a bool tensor with shape (num_keypoints_in_source_image,)
            im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of
                      wasted runtime. So please provide it. Expected a tuple with (width, height) or (height, width)
                      of source image
            im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost
                      of wasted runtime. So please provide it. Expected a tuple with (width, height) or (height, width)
                      of destination image
            o1/o2: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config
                   is set to None. See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
                   Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)
            s1/s2: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
                   See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
                   Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)
            return_dist: if True, inverse confidence value is also outputted.

        Returns:
            Filtered putative matches.
            A long tensor with shape (num_filtered_matches, 2) with indices of corresponding keypoints in k1 and k2.
        """
        with torch.no_grad():
            return adalam_core(
                k1,
                k2,
                fnn12=putative_matches,
                scores1=scores,
                mnn=mnn,
                im1shape=im1shape,
                im2shape=im2shape,
                o1=o1,
                o2=o2,
                s1=s1,
                s2=s2,
                config=self.config,
                return_dist=return_dist,
            )

    def match_and_filter(
        self,
        k1: Tensor,
        k2: Tensor,
        d1: Tensor,
        d2: Tensor,
        im1shape: Optional[Tuple[int, int]] = None,
        im2shape: Optional[Tuple[int, int]] = None,
        o1: Optional[Tensor] = None,
        o2: Optional[Tensor] = None,
        s1: Optional[Tensor] = None,
        s2: Optional[Tensor] = None,
        return_dist: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Standard matching and filtering with AdaLAM. This function:

            - performs some elementary sanity check on the inputs;
            - wraps input arrays into torch tensors and loads to GPU if necessary;
            - extracts nearest neighbors;
            - finds mutual nearest neighbors if required;
            - finally calls AdaLAM filtering.

        Inputs:
            k1: keypoint locations in the source image, in pixel coordinates.
                Expected an array with shape (num_keypoints_in_source_image, 2).
            k2: keypoint locations in the destination image, in pixel coordinates.
                Expected an array with shape (num_keypoints_in_destination_image, 2).
            d1: descriptors in the source image.
                Expected an array with shape (num_keypoints_in_source_image, descriptor_size).
            d2: descriptors in the destination image.
                Expected an array with shape (num_keypoints_in_destination_image, descriptor_size).
            im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of
                      wasted runtime. So please provide it. Expected a tuple with (width, height) or (height, width)
                      of source image
            im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost
                      of wasted runtime. So please provide it. Expected a tuple with (width, height) or (height, width)
                      of destination image
            o1/o2: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config
                   is set to None. See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
                   Expected an array with shape (num_keypoints_in_source/destination_image,)
            s1/s2: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
                   See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
                   Expected an array with shape (num_keypoints_in_source/destination_image,)
            return_dist: if True, inverse confidence value is also outputted.

        Returns:
            Filtered putative matches.
            A long tensor with shape (num_filtered_matches, 2) with indices of corresponding keypoints in k1 and k2.
        """
        if s1 is None or s2 is None:
            if self.config["scale_rate_threshold"] is not None:
                raise AttributeError(
                    "Current configuration considers keypoint scales for filtering, but scales have not been provided.\n"  # noqa: E501
                    "Please either provide scales or set 'scale_rate_threshold' to None to disable scale filtering"
                )
        if o1 is None or o2 is None:
            if self.config["orientation_difference_threshold"] is not None:
                raise AttributeError(
                    "Current configuration considers keypoint orientations for filtering, but orientations have not been provided.\n"  # noqa: E501
                    "Please either provide orientations or set 'orientation_difference_threshold' to None to disable orientations filtering"  # noqa: E501
                )
        _k1 = as_tensor(k1, device=self.config["device"], dtype=torch.float32)
        _k2 = as_tensor(k2, device=self.config["device"], dtype=torch.float32)
        _d1 = as_tensor(d1, device=self.config["device"], dtype=torch.float32)
        _d2 = as_tensor(d2, device=self.config["device"], dtype=torch.float32)
        if o1 is not None:
            _o1 = as_tensor(o1, device=self.config["device"], dtype=torch.float32)
            _o2 = as_tensor(o2, device=self.config["device"], dtype=torch.float32)
        else:
            _o1, _o2 = o1, o2
        if s1 is not None:
            _s1 = as_tensor(s1, device=self.config["device"], dtype=torch.float32)
            _s2 = as_tensor(s2, device=self.config["device"], dtype=torch.float32)
        else:
            _s1, _s2 = s1, s2

        if (len(_d2) <= 1) or (len(_d1) <= 1):
            idxs, dists = _no_match(_d1)
            if return_dist:
                return idxs, dists
            return idxs

        distmat = dist_matrix(_d1, _d2, is_normalized=False)
        dd12, nn12 = torch.topk(distmat, k=2, dim=1, largest=False)  # (n1, 2)

        putative_matches = nn12[:, 0]
        scores = dd12[:, 0] / dd12[:, 1].clamp_min_(1e-3)

        if self.config["force_seed_mnn"]:
            dd21, nn21 = torch.min(distmat, dim=0)  # (n2,)
            mnn = nn21[putative_matches] == torch.arange(_k1.shape[0], device=self.config["device"])
        else:
            mnn = None

        return self.filter_matches(
            _k1, _k2, putative_matches, scores, mnn, im1shape, im2shape, _o1, _o2, _s1, _s2, return_dist
        )
