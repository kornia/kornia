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

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK
from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, LocalFeatureMatcher, LoFTR
from kornia.geometry.linalg import transform_points
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective


class HomographyTracker(nn.Module):
    r"""Perform local-feature-based tracking of the target planar object in the sequence of the frames.

    Args:
        initial_matcher: image matching module, e.g. :class:`~kornia.feature.LocalFeatureMatcher`
                          or :class:`~kornia.feature.LoFTR`. Default: :class:`~kornia.feature.GFTTAffNetHardNet`.
        fast_matcher: fast image matching module, e.g. :class:`~kornia.feature.LocalFeatureMatcher`
                          or :class:`~kornia.feature.LoFTR`. Default: :class:`~kornia.feature.DescriptorMatcher`.
        ransac: homography estimation module. Default: :class:`~kornia.geometry.RANSAC`.
        minimum_inliers_num: threshold for number inliers for matching to be successful. RANSAC needs 4 correspondences,
            so fewer than 4 matches always count as a failed match, whatever this value is.

    """

    def __init__(
        self,
        initial_matcher: Optional[nn.Module] = None,
        fast_matcher: Optional[nn.Module] = None,
        ransac: Optional[nn.Module] = None,
        minimum_inliers_num: int = 30,
    ) -> None:
        super().__init__()
        self.initial_matcher = initial_matcher or (
            LocalFeatureMatcher(GFTTAffNetHardNet(3000), DescriptorMatcher("smnn", 0.95))
        )
        self.fast_matcher = fast_matcher or LoFTR("outdoor")
        self.ransac = ransac or RANSAC("homography", inl_th=5.0, batch_size=4096, max_iter=10, max_lo_iters=10)
        self.minimum_inliers_num = minimum_inliers_num

        # placeholders
        self.target: torch.Tensor
        self.target_initial_representation: Dict[str, Any] = {}
        self.target_fast_representation: Dict[str, Any] = {}
        self.previous_homography: Optional[torch.Tensor] = None

        self.inliers_num: int = 0
        self.keypoints0_num: int = 0
        self.keypoints1_num: int = 0

        self.reset_tracking()

    @property
    def device(self) -> torch.device:
        """Return the device used by the current target image tensor.

        Returns:
            The ``torch.device`` where ``self.target`` is allocated.
        """
        return self.target.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type used by the current target image tensor.

        Returns:
            The ``torch.dtype`` of ``self.target``.
        """
        return self.target.dtype

    @torch.no_grad()
    def set_target(self, target: torch.Tensor) -> None:
        """Register a new target image and refresh cached matcher features.

        Args:
            target: Reference target image tensor used for subsequent matching.

        Returns:
            None.

        The target is extracted once here for :class:`~kornia.feature.LoFTR` and
        :class:`~kornia.feature.LocalFeatureMatcher` matchers, and the cached features are reused for every frame.
        Setting a target starts a new track: the previous homography maps the old target's coordinates, so it is
        cleared. Other matchers get no cached representation.
        """
        self.target = target
        self.reset_tracking()
        self.target_initial_representation = self._extract_target(self.initial_matcher, target)
        self.target_fast_representation = self._extract_target(self.fast_matcher, target)

    @staticmethod
    def _extract_target(matcher: nn.Module, target: torch.Tensor) -> Dict[str, Any]:
        # Matchers that accept pre-extracted features for image0: LoFTR takes `features0` instead of `image0`,
        # LocalFeatureMatcher takes `lafs0`, `responses0` and `descriptors0` next to `image0`.
        if isinstance(matcher, LoFTR):
            return {"features0": matcher.extract_features(target)}
        if isinstance(matcher, LocalFeatureMatcher):
            return {f"{k}0": v for k, v in matcher.extract_features(target).items()}
        return {}

    def _matcher_input(self, representation: Dict[str, Any], frame: torch.Tensor) -> Dict[str, Any]:
        data: Dict[str, Any] = {"image1": frame, **representation}
        if "features0" not in representation:
            data["image0"] = self.target
        return data

    def reset_tracking(self) -> None:
        """Reset temporal tracking state from previously processed frames.

        Returns:
            None.
        """
        self.previous_homography = None

    def no_match(self) -> Tuple[torch.Tensor, bool]:
        """Return a failed-match response and clear current match statistics.

        Returns:
            A tuple ``(H, is_valid)`` where ``H`` is an empty ``3 x 3`` tensor
            on the tracker device and dtype, and ``is_valid`` is ``False``.
        """
        self.inliers_num = 0
        self.keypoints0_num = 0
        self.keypoints1_num = 0
        return torch.empty(3, 3, device=self.device, dtype=self.dtype), False

    def match_initial(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Estimate a homography from the initial target frame to frame ``x``.

        Args:
            x: Current frame tensor to match against the stored target image.

        Returns:
            A tuple ``(H, is_valid)`` where ``H`` is the estimated homography
            matrix and ``is_valid`` indicates whether enough inliers were found.

        The method updates keypoint counters, inlier statistics, and stores the
        estimated homography as ``previous_homography`` on success.
        """
        match_dict: Dict[str, torch.Tensor] = self.initial_matcher(
            self._matcher_input(self.target_initial_representation, x)
        )
        keypoints0 = match_dict["keypoints0"][match_dict["batch_indexes"] == 0]
        keypoints1 = match_dict["keypoints1"][match_dict["batch_indexes"] == 0]

        self.keypoints0_num = len(keypoints0)
        self.keypoints1_num = len(keypoints1)

        if self.keypoints0_num < max(self.minimum_inliers_num, 4):
            return self.no_match()

        H, inliers = self.ransac(keypoints0, keypoints1)
        self.inliers_num = inliers.sum().item()

        if self.inliers_num < self.minimum_inliers_num:
            return self.no_match()
        self.previous_homography = H.clone()

        return H, True

    def track_next_frame(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Track the target in frame ``x`` using the previous homography prior.

        Args:
            x: Current frame tensor to align with the target image.

        Returns:
            A tuple ``(H, is_valid)`` where ``H`` is the updated homography and
            ``is_valid`` indicates whether tracking remained reliable.

        The frame is first prewarped by the inverse of the previous homography,
        then matched with ``fast_matcher`` and verified using RANSAC.
        """
        KORNIA_CHECK(
            self.previous_homography is not None,
            "track_next_frame needs a previous homography; call forward() or match_initial() first.",
        )
        Hwarp = self.previous_homography.clone()[None]  # ty: ignore[possibly-missing-attribute]
        # make a bit of border for safety
        Hwarp[:, 0:2, 0:2] = Hwarp[:, 0:2, 0:2] / 0.8
        Hwarp[:, 0:2, 2] -= 10.0
        Hinv = torch.inverse(Hwarp)
        h, w = self.target.shape[2:]
        frame_warped = warp_perspective(x, Hinv, (h, w))
        match_dict = self.fast_matcher(self._matcher_input(self.target_fast_representation, frame_warped))
        keypoints0 = match_dict["keypoints0"][match_dict["batch_indexes"] == 0]
        keypoints1 = match_dict["keypoints1"][match_dict["batch_indexes"] == 0]
        keypoints1 = transform_points(Hwarp, keypoints1)

        self.keypoints0_num = len(keypoints0)
        self.keypoints1_num = len(keypoints1)

        if self.keypoints0_num < max(self.minimum_inliers_num, 4):
            self.reset_tracking()
            return self.no_match()

        H, inliers = self.ransac(keypoints0, keypoints1)
        self.inliers_num = inliers.sum().item()

        if self.inliers_num < self.minimum_inliers_num:
            self.reset_tracking()
            return self.no_match()

        self.previous_homography = H.clone()
        return H, True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Run one tracking step on frame ``x``.

        Args:
            x: Current frame tensor.

        Returns:
            A tuple ``(H, is_valid)`` from ``track_next_frame`` when previous
            state exists, otherwise from ``match_initial``.
        """
        if self.previous_homography is not None:
            return self.track_next_frame(x)
        return self.match_initial(x)
