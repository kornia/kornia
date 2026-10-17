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

from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK
from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, LocalFeatureMatcher, LoFTR
from kornia.geometry.linalg import transform_points
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective


class HomographyTracker(nn.Module):
    r"""Perform local-feature-based tracking of the target planar object in the sequence of the frames.

    Convention:
        - The returned homography ``H`` is :math:`(3, 3)` and maps target pixel coordinates :math:`(x, y)` to frame
          pixel coordinates, the direction of ``cv2.findHomography(target_points, frame_points)``.
        - The target and every frame are one grayscale image, :math:`(1, 1, H, W)`; a frame can have another size than
          the target. Call :meth:`set_target` before the first frame. The target is not a buffer, so
          :meth:`~torch.nn.Module.to` leaves it where it is: call :meth:`set_target` again after moving the tracker.
        - For a :class:`~kornia.feature.LoFTR`, a :class:`~kornia.feature.LocalFeatureMatcher` or a matcher with an
          ``extract_features`` module, :meth:`set_target` extracts the target's features once. Changes made in place to
          the target or to that matcher's weights afterwards are not noticed; call :meth:`set_target` again.
        - A failed match returns ``(H, False)`` with ``H`` uninitialised (:meth:`no_match`). A failure while tracking
          also drops the previous homography, so the next frame is matched from scratch by ``initial_matcher``.
        - Every homography the tracker returns or stores as ``previous_homography``, :meth:`no_match`'s included, is
          float32 for a ``float16`` or ``bfloat16`` target and has the target's dtype otherwise.

    Args:
        initial_matcher: image matching module for a frame without a previous homography, e.g.
            :class:`~kornia.feature.LocalFeatureMatcher` or :class:`~kornia.feature.LoFTR`. Default:
            ``LocalFeatureMatcher(GFTTAffNetHardNet(3000), DescriptorMatcher("smnn", 0.95))``.
        fast_matcher: image matching module for the next frames, matched after warping them with the previous
            homography, e.g. :class:`~kornia.feature.LocalFeatureMatcher` or :class:`~kornia.feature.LoFTR`. Default:
            ``LoFTR("outdoor")``.
        ransac: homography estimation module. Default: ``RANSAC("homography", inl_th=5.0, batch_size=4096,
            max_iter=10, max_lo_iters=10)``, looser than the ``RANSAC("homography")`` (``inl_th=2.0``) of
            :class:`~kornia.contrib.ImageStitcher`.
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
        # The matcher and the target each cached representation was extracted from. A plain dict, so that the matchers
        # are not registered a second time as submodules.
        self._cache_source: Dict[str, Tuple[nn.Module, torch.Tensor]] = {}
        self.previous_homography: Optional[torch.Tensor] = None
        # The homography the tracker found last and the target whose coordinates it maps. A previous_homography
        # assigned from outside is not this object, and is kept as it is.
        self._prior_source: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

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

    @property
    def _homography_dtype(self) -> torch.dtype:
        # The dtype of every homography returned or kept as the prior: float32 for a half target, as the float32
        # keypoints of a half matcher give it, so that it can be inverted and its sampling grid stays sub-pixel.
        return torch.promote_types(self.dtype, torch.float32)

    @torch.no_grad()
    def set_target(self, target: torch.Tensor) -> None:
        """Register a new target image and refresh cached matcher features.

        Args:
            target: Reference target image tensor used for subsequent matching.

        Returns:
            None.

        The target's features are extracted once here and reused for every frame, for three kinds of matcher:

        - :class:`~kornia.feature.LoFTR`: its :meth:`~kornia.feature.LoFTR.extract_features` output is passed as
          ``features0``, in place of ``image0``.
        - :class:`~kornia.feature.LocalFeatureMatcher`: the ``lafs``, ``responses`` and ``descriptors`` of its
          :meth:`~kornia.feature.LocalFeatureMatcher.extract_features` are passed as ``lafs0``, ``responses0`` and
          ``descriptors0``, next to ``image0``.
        - Any other matcher whose ``extract_features`` attribute is a :class:`torch.nn.Module`: every entry of the dict
          it returns is passed with a ``0`` suffix (``desc`` as ``desc0``), next to ``image0``.

        Other matchers get no cached representation and receive ``image0`` and ``image1`` only. A matcher replaced after
        this call, or a ``target`` reassigned directly, gets the target's features extracted again on its next use.
        Setting a target starts a new track: the previous homography maps the old target's coordinates, so it is
        cleared. A ``target`` reassigned directly starts a new track too, on the next :meth:`forward` or
        :meth:`track_next_frame`.
        """
        self.target = target
        self.reset_tracking()
        self.target_initial_representation = self._extract_target(self.initial_matcher, target)
        self.target_fast_representation = self._extract_target(self.fast_matcher, target)
        self._cache_source = {"initial": (self.initial_matcher, target), "fast": (self.fast_matcher, target)}

    def _target_representation(self, initial: bool) -> Dict[str, Any]:
        # The cached representation of the target for the configured matcher. It is extracted again, once, when the
        # matcher or the target was replaced after set_target: the cache holds the old model's features of the old
        # target. Without set_target nothing is cached, and the matcher gets image0.
        key = "initial" if initial else "fast"
        matcher = self.initial_matcher if initial else self.fast_matcher
        source = self._cache_source.get(key)
        if source is not None and (source[0] is not matcher or source[1] is not self.target):
            with torch.no_grad():
                representation = self._extract_target(matcher, self.target)
            if initial:
                self.target_initial_representation = representation
            else:
                self.target_fast_representation = representation
            self._cache_source[key] = (matcher, self.target)
        return self.target_initial_representation if initial else self.target_fast_representation

    @staticmethod
    def _extract_target(matcher: nn.Module, target: torch.Tensor) -> Dict[str, Any]:
        if isinstance(matcher, LoFTR):
            return {"features": matcher.extract_features(target)}
        if isinstance(matcher, LocalFeatureMatcher):
            return matcher.extract_features(target)
        extract_features = getattr(matcher, "extract_features", None)
        if isinstance(extract_features, nn.Module):
            return extract_features(target)
        return {}

    def _matcher_input(self, matcher: nn.Module, representation: Dict[str, Any], frame: torch.Tensor) -> Dict[str, Any]:
        data: Dict[str, Any] = {"image0": self.target, "image1": frame}
        for k, v in representation.items():
            data[f"{k}0"] = v
        if isinstance(matcher, LoFTR) and "features0" in data:  # LoFTR takes features0 in place of image0, not both
            del data["image0"]
        return data

    def reset_tracking(self) -> None:
        """Reset temporal tracking state from previously processed frames.

        Returns:
            None.
        """
        self.previous_homography = None
        self._prior_source = None

    def _drop_prior_of_another_target(self) -> None:
        # A target assigned directly, without set_target, starts a new track as set_target does: the homography the
        # tracker found maps the old target's coordinates.
        source = self._prior_source
        if source is not None and source[0] is self.previous_homography and source[1] is not self.target:
            self.reset_tracking()

    def no_match(self) -> Tuple[torch.Tensor, bool]:
        """Return a failed-match response and clear current match statistics.

        Returns:
            A tuple ``(H, is_valid)`` where ``H`` is an uninitialised ``3 x 3`` tensor on the tracker device, in
            float32 for a ``float16`` or ``bfloat16`` target and in the target's dtype otherwise, and ``is_valid`` is
            ``False``.
        """
        self.inliers_num = 0
        self.keypoints0_num = 0
        self.keypoints1_num = 0
        return torch.empty(3, 3, device=self.device, dtype=self._homography_dtype), False

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
            self._matcher_input(self.initial_matcher, self._target_representation(True), x)
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
        H = H.to(self._homography_dtype)
        self.previous_homography = H.clone()
        self._prior_source = (self.previous_homography, self.target)

        return H, True

    def track_next_frame(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Track the target in frame ``x`` using the previous homography prior.

        Args:
            x: Current frame tensor to align with the target image.

        Returns:
            A tuple ``(H, is_valid)`` where ``H`` is the updated homography and
            ``is_valid`` indicates whether tracking remained reliable.

        The frame is first prewarped by the inverse of the previous homography, widened by a margin,
        then matched with ``fast_matcher`` and verified using RANSAC. The inverse is computed in the homography dtype
        (float32 for a ``float16`` or ``bfloat16`` target), the frame is warped in that dtype or in its own if that is
        wider, and ``fast_matcher`` gets the warped frame in the dtype of ``x``.
        """
        self._drop_prior_of_another_target()
        KORNIA_CHECK(
            self.previous_homography is not None,
            "track_next_frame needs a previous homography; call forward() or match_initial() first.",
        )
        homography_dtype = self._homography_dtype
        Hwarp = cast(torch.Tensor, self.previous_homography).to(homography_dtype, copy=True)[None]
        # make a bit of border for safety
        Hwarp[:, 0:2, 0:2] = Hwarp[:, 0:2, 0:2] / 0.8
        Hwarp[:, 0:2, 2] -= 10.0
        Hinv = torch.inverse(Hwarp)
        h, w = self.target.shape[2:]
        # grid_sample takes the image and the sampling grid in one dtype, and a half grid is coarser than a pixel on a
        # large frame (bfloat16 keeps 8 mantissa bits): a half frame is warped in float32 and cast back for the matcher.
        warp_dtype = torch.promote_types(x.dtype, homography_dtype)
        frame_warped = warp_perspective(x.to(warp_dtype), Hinv, (h, w)).to(x.dtype)
        match_dict = self.fast_matcher(
            self._matcher_input(self.fast_matcher, self._target_representation(False), frame_warped)
        )
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

        H = H.to(homography_dtype)
        self.previous_homography = H.clone()
        self._prior_source = (self.previous_homography, self.target)
        return H, True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Run one tracking step on frame ``x``.

        Args:
            x: Current frame tensor.

        Returns:
            A tuple ``(H, is_valid)`` from ``track_next_frame`` when previous
            state exists, otherwise from ``match_initial``.
        """
        self._drop_prior_of_another_target()
        if self.previous_homography is not None:
            return self.track_next_frame(x)
        return self.match_initial(x)
