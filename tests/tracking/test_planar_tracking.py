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
from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, LocalFeatureMatcher, SIFTFeature
from kornia.geometry import rescale, transform_points
from kornia.tracking import HomographyTracker

from testing.base import BaseTester


def _make_tracker(minimum_inliers_num: int = 5) -> HomographyTracker:
    """Return a HomographyTracker whose heavy sub-modules are replaced with lightweight mocks."""
    initial_matcher = MagicMock()
    fast_matcher = MagicMock()
    ransac = MagicMock()
    # Remove extract_features so set_target skips feature pre-extraction
    del initial_matcher.extract_features
    del fast_matcher.extract_features
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
        from torch import nn

        fake_feats = {"desc": torch.rand(1, 4)}

        class _FakeExtract(nn.Module):
            def forward(self, x):
                return fake_feats

        initial_matcher = MagicMock()
        fast_matcher = MagicMock()
        initial_matcher.extract_features = _FakeExtract()
        fast_matcher.extract_features = _FakeExtract()
        tracker = HomographyTracker(
            initial_matcher=initial_matcher,
            fast_matcher=fast_matcher,
            ransac=MagicMock(),
        )
        image = torch.rand(1, 1, 8, 8)
        tracker.set_target(image)
        assert tracker.target_initial_representation == fake_feats
        assert tracker.target_fast_representation == fake_feats

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


@pytest.fixture()
def data_url():
    url = "https://github.com/kornia/data_test/blob/main/loftr_outdoor_and_homography_data.pt?raw=true"
    return url


class TestHomographyTracker(BaseTester):
    @pytest.mark.slow
    def test_smoke(self, device):
        tracker = HomographyTracker().to(device)
        assert tracker is not None

    @pytest.mark.slow
    def test_nomatch(self, device, dtype, data_url):
        data = torch.hub.load_state_dict_from_url(data_url)

        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(100), DescriptorMatcher("smnn", 0.95)).to(device, dtype)
        tracker = HomographyTracker(matcher, matcher, minimum_inliers_num=100)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        tracker.set_target(data["image0"])
        torch.random.manual_seed(0)
        _, success = tracker(torch.zeros_like(data["image0"]))
        assert not success

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    def test_real(self, device, dtype, data_url):
        data = torch.hub.load_state_dict_from_url(data_url)
        # This is not unit test, but that is quite good integration test
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)

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
