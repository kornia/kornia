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

from unittest.mock import MagicMock

import pytest
import torch

from kornia.core._compat import torch_version_le
from kornia.core.exceptions import BaseError
from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, LocalFeatureMatcher, LoFTR, SIFTFeature
from kornia.geometry import rescale, transform_points
from kornia.tracking import HomographyTracker

from testing.base import BaseTester


def _make_tracker(minimum_inliers_num: int = 5) -> HomographyTracker:
    """Return a HomographyTracker whose heavy sub-modules are replaced with lightweight mocks."""
    initial_matcher = MagicMock()
    fast_matcher = MagicMock()
    ransac = MagicMock()
    return HomographyTracker(
        initial_matcher=initial_matcher,
        fast_matcher=fast_matcher,
        ransac=ransac,
        minimum_inliers_num=minimum_inliers_num,
    )


def _match_dict(n_keypoints: int, device: torch.device, dtype: torch.dtype) -> dict:
    """Produce a fake match dict with n_keypoints matches for batch 0."""
    return {
        "keypoints0": torch.rand(n_keypoints, 2, device=device, dtype=dtype),
        "keypoints1": torch.rand(n_keypoints, 2, device=device, dtype=dtype),
        "batch_indexes": torch.zeros(n_keypoints, dtype=torch.long, device=device),
    }


def _texture(h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.avg_pool2d(torch.rand(1, 1, h, w, generator=g), 5, 1, 2)


class TestHomographyTrackerCaching:
    def _sift_matcher(self) -> LocalFeatureMatcher:
        return LocalFeatureMatcher(SIFTFeature(300), DescriptorMatcher("snn", 0.9))

    def test_target_features_are_extracted_once(self):
        # C-4: the nn.Module guard never matched a method, so the target was re-extracted on every frame.
        initial, fast = self._sift_matcher(), self._sift_matcher()
        shapes = []
        for m in (initial, fast):
            m.local_feature.register_forward_hook(lambda mod, inp, out: shapes.append(tuple(inp[0].shape)))
        tracker = HomographyTracker(initial, fast, minimum_inliers_num=10)
        tracker.set_target(_texture(60, 80))
        assert shapes == [(1, 1, 60, 80), (1, 1, 60, 80)]
        shapes.clear()
        tracker(_texture(70, 90, seed=1))
        assert shapes == [(1, 1, 70, 90)]

    def test_loftr_target_features_are_extracted_once(self):
        loftr = LoFTR(None)
        calls = []
        loftr.backbone.register_forward_hook(lambda mod, inp, out: calls.append(tuple(inp[0].shape)))
        tracker = HomographyTracker(self._sift_matcher(), loftr, minimum_inliers_num=10)
        tracker.set_target(_texture(64, 80))
        assert calls == [(1, 1, 64, 80)]
        calls.clear()
        tracker.previous_homography = torch.eye(3)
        tracker.track_next_frame(_texture(72, 96, seed=1))
        assert calls == [(1, 1, 64, 80)]  # only the pre-warped frame, at the target's size

    def test_track_next_frame_without_a_prior_raises(self):
        # C-5: UnboundLocalError on Hwarp.
        tracker = _make_tracker()
        tracker.set_target(torch.rand(1, 1, 8, 9))
        with pytest.raises(BaseError, match="previous homography"):
            tracker.track_next_frame(torch.rand(1, 1, 8, 9))

    def test_fewer_than_four_matches_is_a_failed_match(self):
        # C-6: minimum_inliers_num < 4 let 3 matches reach RANSAC, which raised.
        tracker = _make_tracker(minimum_inliers_num=0)
        tracker.set_target(torch.rand(1, 1, 8, 9))
        tracker.initial_matcher.return_value = _match_dict(3, torch.device("cpu"), torch.float32)
        _, ok = tracker.match_initial(torch.rand(1, 1, 8, 9))
        assert not ok
        tracker.ransac.assert_not_called()

    def test_fewer_than_four_matches_fails_track_next_frame(self):
        tracker = _make_tracker(minimum_inliers_num=0)
        tracker.set_target(torch.rand(1, 1, 8, 9))
        tracker.previous_homography = torch.eye(3)
        tracker.fast_matcher.return_value = _match_dict(3, torch.device("cpu"), torch.float32)
        _, ok = tracker.track_next_frame(torch.rand(1, 1, 8, 9))
        assert not ok
        tracker.ransac.assert_not_called()
        assert tracker.previous_homography is None

    def test_set_target_starts_a_new_track(self):
        # C-7 (controller ruling): the old prior maps the old target's coordinates.
        tracker = _make_tracker()
        tracker.set_target(torch.rand(1, 1, 8, 9))
        tracker.previous_homography = torch.eye(3)
        tracker.set_target(torch.rand(1, 1, 6, 7))
        assert tracker.previous_homography is None


class TestHomographyTrackerUnit:
    """Fast unit tests for HomographyTracker using mocked sub-modules."""

    def test_reset_tracking_clears_homography(self):
        tracker = _make_tracker()
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.previous_homography = torch.eye(3)
        tracker.reset_tracking()
        assert tracker.previous_homography is None

    def test_set_target_without_extract_features(self):
        tracker = _make_tracker()
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        assert torch.equal(tracker.target, image)
        # Representations stay empty when extract_features not present
        assert tracker.target_initial_representation == {}
        assert tracker.target_fast_representation == {}

    def test_set_target_with_extract_features(self):
        # A matcher that is neither LoFTR nor LocalFeatureMatcher gets no cached representation.
        initial_matcher = MagicMock()
        fast_matcher = MagicMock()
        tracker = HomographyTracker(initial_matcher=initial_matcher, fast_matcher=fast_matcher, ransac=MagicMock())
        tracker.set_target(torch.rand(1, 1, 8, 8))
        assert tracker.target_initial_representation == {}
        assert tracker.target_fast_representation == {}
        initial_matcher.extract_features.assert_not_called()
        fast_matcher.extract_features.assert_not_called()

    def test_no_match_returns_empty_tensor_and_false(self):
        tracker = _make_tracker()
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        H, success = tracker.no_match()
        assert not success
        assert H.shape == (3, 3)
        assert tracker.inliers_num == 0

    def test_match_initial_too_few_keypoints(self):
        # Fewer than minimum_inliers_num keypoints → no_match
        tracker = _make_tracker(minimum_inliers_num=10)
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.initial_matcher.return_value = _match_dict(3, torch.device("cpu"), torch.float32)
        _, success = tracker.match_initial(torch.rand(1, 1, 8, 8))
        assert not success

    def test_match_initial_too_few_inliers(self):
        # Enough keypoints but RANSAC reports few inliers → no_match
        tracker = _make_tracker(minimum_inliers_num=5)
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.initial_matcher.return_value = _match_dict(20, torch.device("cpu"), torch.float32)
        # RANSAC returns homography + inlier mask with only 2 inliers
        inliers = torch.zeros(20, dtype=torch.bool)
        inliers[:2] = True
        tracker.ransac.return_value = (torch.eye(3), inliers)
        _, success = tracker.match_initial(torch.rand(1, 1, 8, 8))
        assert not success

    def test_match_initial_success(self):
        tracker = _make_tracker(minimum_inliers_num=5)
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.initial_matcher.return_value = _match_dict(20, torch.device("cpu"), torch.float32)
        inliers = torch.ones(20, dtype=torch.bool)
        H_expected = torch.eye(3) * 2
        tracker.ransac.return_value = (H_expected, inliers)
        H, success = tracker.match_initial(torch.rand(1, 1, 8, 8))
        assert success
        assert tracker.previous_homography is not None
        assert torch.allclose(H, H_expected)

    def test_forward_routes_to_match_initial_when_no_previous(self):
        tracker = _make_tracker(minimum_inliers_num=5)
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.initial_matcher.return_value = _match_dict(20, torch.device("cpu"), torch.float32)
        inliers = torch.ones(20, dtype=torch.bool)
        tracker.ransac.return_value = (torch.eye(3), inliers)
        assert tracker.previous_homography is None
        _, success = tracker(torch.rand(1, 1, 8, 8))
        assert success

    def test_forward_routes_to_track_next_frame_when_previous_set(self):
        tracker = _make_tracker(minimum_inliers_num=5)
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.previous_homography = torch.eye(3)
        tracker.fast_matcher.return_value = _match_dict(20, torch.device("cpu"), torch.float32)
        inliers = torch.ones(20, dtype=torch.bool)
        tracker.ransac.return_value = (torch.eye(3), inliers)
        _, success = tracker(torch.rand(1, 1, 8, 8))
        assert success

    def test_track_next_frame_too_few_keypoints_resets(self):
        tracker = _make_tracker(minimum_inliers_num=10)
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.previous_homography = torch.eye(3)
        tracker.fast_matcher.return_value = _match_dict(3, torch.device("cpu"), torch.float32)
        _, success = tracker.track_next_frame(torch.rand(1, 1, 8, 8))
        assert not success
        assert tracker.previous_homography is None  # reset_tracking was called

    def test_track_next_frame_too_few_inliers_resets(self):
        tracker = _make_tracker(minimum_inliers_num=5)
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        tracker.previous_homography = torch.eye(3)
        tracker.fast_matcher.return_value = _match_dict(20, torch.device("cpu"), torch.float32)
        inliers = torch.zeros(20, dtype=torch.bool)
        inliers[:2] = True
        tracker.ransac.return_value = (torch.eye(3), inliers)
        _, success = tracker.track_next_frame(torch.rand(1, 1, 8, 8))
        assert not success
        assert tracker.previous_homography is None  # reset_tracking was called


class TestHomographyTracker(BaseTester):
    @pytest.mark.slow
    def test_smoke(self, device):
        tracker = HomographyTracker().to(device)
        assert tracker is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_nomatch(self, device, dtype, data):
        # The fixture is session-scoped and shared; cast into a copy rather than in place.
        data = {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(100), DescriptorMatcher("smnn", 0.95)).to(device, dtype)
        tracker = HomographyTracker(matcher, matcher, minimum_inliers_num=100)
        tracker.set_target(data["image0"])
        torch.random.manual_seed(0)
        _, success = tracker(torch.zeros_like(data["image0"]))
        assert not success

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real(self, device, dtype, data):
        # The fixture is session-scoped and shared; cast into a copy rather than in place.
        data = {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        # This is not unit test, but that is quite good integration test

        data["image0"] = rescale(data["image0"], 0.5, interpolation="bilinear", align_corners=False)
        data["image1"] = rescale(data["image1"], 0.5, interpolation="bilinear", align_corners=False)

        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(1000), DescriptorMatcher("snn", 0.8)).to(device, dtype)
        torch.manual_seed(8)  # issue kornia#2027
        tracker = HomographyTracker(matcher, matcher).to(device, dtype)

        with torch.no_grad():
            tracker.set_target(data["image0"])
            torch.manual_seed(8)  # issue kornia#2027
            homography, success = tracker(data["image1"])
        assert success
        pts_src = data["pts0"].to(device, dtype) / 2.0
        pts_dst = data["pts1"].to(device, dtype) / 2.0
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

        with torch.no_grad():
            torch.manual_seed(6)
            homography, success = tracker(data["image1"])
        assert success
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)
