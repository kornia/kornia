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

# Integrated from original AdaLAM repo
# https://github.com/cavalli1234/AdaLAM
# Copyright (c) 2020, Luca Cavalli

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE
from kornia.feature.laf import get_laf_center, get_laf_orientation, get_laf_scale

from .core import AdalamConfig, _no_match, adalam_core
from .utils import dist_matrix


def get_adalam_default_config() -> AdalamConfig:
    """Return default Adalam Config."""
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
    desc1: torch.Tensor,
    desc2: torch.Tensor,
    lafs1: torch.Tensor,
    lafs2: torch.Tensor,
    config: Optional[AdalamConfig] = None,
    hw1: Optional[Sequence[int]] = None,
    hw2: Optional[Sequence[int]] = None,
    dm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform descriptor matching, followed by AdaLAM filtering.

    See :cite:`AdaLAM2020` for more details.

    Convention:
        - See the Convention block on :func:`~kornia.feature.match_nn`. ``dists`` is the inverse of the confidence of
          the most confident local RANSAC that accepted the match (overlapping neighbourhoods often accept a match
          several times), in float32 whatever the descriptor dtype, so at most ``1 / min_confidence`` (``1 / 200`` by
          default). If no seed survives, for example with very few keypoints, the matches of a ratio test on squared
          distances, ``ratio <= 0.95``, are returned with that ratio instead.
        - The putative matches are the nearest neighbours in the distance matrix ``dm``, scored by the ratio of the
          two smallest distances; with ``force_seed_mnn`` only mutual nearest neighbours can be seeds. ``dm`` holds
          plain distances, like the ``dm`` of :func:`~kornia.feature.match_nn` and the other matchers: AdaLAM
          squares it internally (its thresholds are tuned for squared distances), so ``dm=torch.cdist(desc1, desc2)``
          gives the matches of ``dm=None``, where the squared L2 distances between the descriptors are used.
        - The LAFs give each keypoint's centre, orientation in degrees and scale. Orientations and scales are read
          only through how their change between the two images varies from match to match: adding an angle to the
          orientations of one image or both, negating both, or scaling the LAFs of one image or both alike changes
          nothing.
        - ``hw1`` and ``hw2`` enter only through the area of their image, which sets the neighbourhood radius in that
          image, so ``(H, W)`` and ``(W, H)`` give the same matches. Without them, the extent of each image's
          keypoints is used.
        - ``config`` is merged into the defaults of ``kornia.feature.adalam.get_adalam_default_config()``; an
          unknown key warns and is ignored. The computation runs on the device of ``desc1`` unless ``config`` sets
          ``"device"``.
        - Exact local affine correspondences are retained when enough distinct keypoints support them beyond the
          seed and minimal fitting pair. Unlike the reference implementation, equal tiny residuals alone do not
          imply duplicate keypoints. Their confidence uses a conservative residual-area floor of ``2**-24`` in
          normalized coordinates, so inverse confidence stays finite even for an exact fit.

    Args:
        desc1: Batch of descriptors of a shape ``(B1, D)``.
        desc2: Batch of descriptors of a shape ``(B2, D)``.
        lafs1: LAFs of a shape ``(1, B1, 2, 3)``.
        lafs2: LAFs of a shape ``(1, B2, 2, 3)``.
        config: dict with AdaLAM config
        dm: torch.Tensor containing the non-negative distances (not squared) from each descriptor in desc1
          to each descriptor in desc2, shape of ``(B1, B2)``. Smaller is better.
        hw1: size of the first image, ``(H, W)`` or ``(W, H)``.
        hw2: size of the second image, ``(H, W)`` or ``(W, H)``.


    Return:
        - Inverse local RANSAC confidence of each match, float32, shape of ``(B3, 1)``; see the Convention
          block.
        - Long torch.Tensor indexes of matching descriptors in desc1 and desc2. Shape: ``(B3, 2)``,
          where 0 <= B3 <= B1.

    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])
    KORNIA_CHECK_LAF(lafs1)
    KORNIA_CHECK_LAF(lafs2)
    config_ = get_adalam_default_config()
    config_["device"] = desc1.device
    if config is not None:
        for key, val in config.items():
            if key not in config_:
                warnings.warn(
                    f"Custom configuration contains a key which is not recognized ({key}). "
                    f"Known configurations are {list(config_.keys())}.",
                    stacklevel=2,
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
        dm=dm,
    )
    return quality, idxs


class AdalamFilter:
    """Implement the AdaLAM (Adaptive Locally-Affine Matching) filter for outlier rejection.

    This class wraps the AdaLAM algorithm to filter feature matches based on
    local affine consistency.

    Args:
        custom_config: Optional configuration object for AdaLAM parameters.
    """

    def __init__(self, custom_config: Optional[AdalamConfig] = None) -> None:
        """Wrap the method AdaLAM for outlier filtering.

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
        k1: torch.Tensor,
        k2: torch.Tensor,
        putative_matches: torch.Tensor,
        scores: torch.Tensor,
        mnn: Optional[torch.Tensor] = None,
        im1shape: Optional[Sequence[int]] = None,
        im2shape: Optional[Sequence[int]] = None,
        o1: Optional[torch.Tensor] = None,
        o2: Optional[torch.Tensor] = None,
        s1: Optional[torch.Tensor] = None,
        s2: Optional[torch.Tensor] = None,
        return_dist: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Call the core functionality of AdaLAM, i.e. just outlier filtering.

        No sanity check is performed on the inputs.

        Args:
            k1: keypoint locations in the source image, in pixel coordinates.
                Expected a float32 torch.Tensor with shape (num_keypoints_in_source_image, 2).
            k2: keypoint locations in the destination image, in pixel coordinates.
                Expected a float32 torch.Tensor with shape (num_keypoints_in_destination_image, 2).
            putative_matches: Initial set of putative matches to be filtered.
                              The current implementation assumes that these are unfiltered nearest neighbor matches,
                              so it requires this to be a list of indices a_i such that the source keypoint i is
                              associated to the destination keypoint a_i. For now to use AdaLAM on different inputs a
                              workaround on the input format is required.
                              Expected a long torch.Tensor with shape (num_keypoints_in_source_image,).
            scores: Confidence scores on the putative_matches. Usually holds Lowe's ratio scores.
            mnn: A mask indicating which putative matches are also mutual nearest neighbors. See documentation on
                 'force_seed_mnn' in the DEFAULT_CONFIG. If None, it disables the mutual nearest neighbor filtering on
                 seed point selection. Expected a bool torch.Tensor with shape (num_keypoints_in_source_image,)
            im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of
                      wasted runtime. So please provide it. Expected a tuple or list with (width, height) or
                      (height, width) of source image
            im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost
                      of wasted runtime. So please provide it. Expected a tuple or list with (width, height) or
                      (height, width) of destination image
            o1: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config
                   is set to None. See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
                   Expected a float32 torch.Tensor with shape (num_keypoints_in_source/destination_image,)
            o2: same as o1 but for destination.
            s1: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
                   See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
                   Expected a float32 torch.Tensor with shape (num_keypoints_in_source/destination_image,)
            s2: same as s1 but for destination.
            return_dist: if True, inverse confidence value is also outputted.

        Returns:
            Filtered putative matches.
            A long torch.Tensor with shape (num_filtered_matches, 2) with indices of corresponding
            keypoints in k1 and k2.

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
        k1: torch.Tensor,
        k2: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
        im1shape: Optional[Sequence[int]] = None,
        im2shape: Optional[Sequence[int]] = None,
        o1: Optional[torch.Tensor] = None,
        o2: Optional[torch.Tensor] = None,
        s1: Optional[torch.Tensor] = None,
        s2: Optional[torch.Tensor] = None,
        return_dist: bool = False,
        dm: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Match and filter with AdaLAM.

        This function:

            - performs some elementary sanity check on the inputs;
            - wraps input arrays into torch tensors and loads to GPU if necessary;
            - extracts nearest neighbors;
            - finds mutual nearest neighbors if required;
            - finally calls AdaLAM filtering.

        Args:
            k1: keypoint locations in the source image, in pixel coordinates.
                Expected an array with shape (num_keypoints_in_source_image, 2).
            k2: keypoint locations in the destination image, in pixel coordinates.
                Expected an array with shape (num_keypoints_in_destination_image, 2).
            d1: descriptors in the source image.
                Expected an array with shape (num_keypoints_in_source_image, descriptor_size).
            d2: descriptors in the destination image.
                Expected an array with shape (num_keypoints_in_destination_image, descriptor_size).
            im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of
                      wasted runtime. So please provide it. Expected a tuple or list with (width, height) or
                      (height, width) of source image
            im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost
                      of wasted runtime. So please provide it. Expected a tuple or list with (width, height) or
                      (height, width) of destination image
            o1: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config
                   is set to None. See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
                   Expected an array with shape (num_keypoints_in_source/destination_image,)
            o2: Same as o1 for destination.
            s1: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
                   See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
                   Expected an array with shape (num_keypoints_in_source/destination_image,)
            s2: Same as s1 for destination.
            return_dist: if True, inverse confidence value is also outputted.
            dm: non-negative distances (not squared) from each descriptor in d1 to each descriptor in d2, shape
                (num_keypoints_in_source_image, num_keypoints_in_destination_image); smaller is better. They are
                squared internally, like the L2 distances used when dm is None.

        Returns:
            Filtered putative matches.
            A long torch.Tensor with shape (num_filtered_matches, 2) with indices of corresponding
            keypoints in k1 and k2.

        """
        if (s1 is None or s2 is None) and self.config["scale_rate_threshold"] is not None:
            raise AttributeError(
                "Current configuration considers keypoint scales for filtering, but scales have not been provided.\n"
                "Please either provide scales or set 'scale_rate_threshold' to None to disable scale filtering"
            )
        if (o1 is None or o2 is None) and self.config["orientation_difference_threshold"] is not None:
            raise AttributeError(
                "Current configuration considers keypoint orientations for filtering, but orientations have not been provided.\n"  # noqa: E501
                "Please either provide orientations or set 'orientation_difference_threshold' to None to disable orientations filtering"  # noqa: E501
            )
        _k1 = torch.as_tensor(k1, device=self.config["device"], dtype=torch.float32)
        _k2 = torch.as_tensor(k2, device=self.config["device"], dtype=torch.float32)
        _d1 = torch.as_tensor(d1, device=self.config["device"], dtype=torch.float32)
        _d2 = torch.as_tensor(d2, device=self.config["device"], dtype=torch.float32)
        if o1 is not None:
            _o1 = torch.as_tensor(o1, device=self.config["device"], dtype=torch.float32)
            _o2 = torch.as_tensor(o2, device=self.config["device"], dtype=torch.float32)
        else:
            _o1, _o2 = o1, o2
        if s1 is not None:
            _s1 = torch.as_tensor(s1, device=self.config["device"], dtype=torch.float32)
            _s2 = torch.as_tensor(s2, device=self.config["device"], dtype=torch.float32)
        else:
            _s1, _s2 = s1, s2

        if dm is not None:
            KORNIA_CHECK(
                tuple(dm.shape) == (len(_d1), len(_d2)),
                f"dm must have shape {(len(_d1), len(_d2))}, got {tuple(dm.shape)}",
            )

        if (len(_d2) <= 1) or (len(_d1) <= 1):
            dists, idxs = _no_match(_d1)
            if return_dist:
                return idxs, dists
            return idxs

        if dm is None:
            distmat = dist_matrix(_d1, _d2, is_normalized=False)
        else:
            distmat = torch.as_tensor(dm, device=self.config["device"], dtype=torch.float32).pow(2)
        dd12, nn12 = torch.topk(distmat, k=2, dim=1, largest=False)  # (n1, 2)

        putative_matches = nn12[:, 0]
        scores = dd12[:, 0] / dd12[:, 1].clamp_min_(1e-3)

        if self.config["force_seed_mnn"]:
            _dd21, nn21 = torch.min(distmat, dim=0)  # (n2,)
            mnn = nn21[putative_matches] == torch.arange(_k1.shape[0], device=self.config["device"])
        else:
            mnn = None

        return self.filter_matches(
            _k1, _k2, putative_matches, scores, mnn, im1shape, im2shape, _o1, _o2, _s1, _s2, return_dist
        )
