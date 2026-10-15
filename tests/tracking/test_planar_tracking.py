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

import copy
from unittest.mock import MagicMock

import pytest
import torch

from kornia.core._compat import torch_version_le
from kornia.core.exceptions import BaseError
from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, LocalFeatureMatcher, LoFTR, SIFTFeature
from kornia.feature.loftr.loftr import default_cfg
from kornia.geometry import rescale, transform_points
from kornia.tracking import HomographyTracker

from testing.base import BaseTester


def _make_tracker(minimum_inliers_num: int = 5) -> HomographyTracker:
    """Return a HomographyTracker whose heavy sub-modules are replaced with lightweight mocks."""
    initial_matcher = MagicMock()
    fast_matcher = MagicMock()
    ransac = MagicMock()
    # Set extract_features to None so isinstance(..., nn.Module) guard skips feature pre-extraction
    initial_matcher.extract_features = None
    fast_matcher.extract_features = None
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


class _CountingExtract(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> dict:
        self.calls += 1
        return {"desc": x.mean((2, 3))}


class _CustomMatcher(torch.nn.Module):
    """A matcher that is neither LoFTR nor LocalFeatureMatcher, with an nn.Module `extract_features`; records inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.extract_features = _CountingExtract()
        self.inputs: list = []

    def forward(self, data: dict) -> dict:
        self.inputs.append(dict(data))
        return _match_dict(20, torch.device("cpu"), torch.float32)


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

    def test_loftr_without_cached_features_gets_image0(self):
        # A target assigned directly, without set_target, has no cached features0, so LoFTR needs image0 (as on main).
        loftr = LoFTR(None)
        keys = []
        loftr.register_forward_pre_hook(lambda mod, args: keys.append(sorted(args[0])))
        tracker = HomographyTracker(loftr, self._sift_matcher(), minimum_inliers_num=10)
        tracker.target = _texture(64, 80)
        tracker.match_initial(_texture(64, 80, seed=1))
        assert keys == [["image0", "image1"]]

    def test_fast_matcher_swapped_to_loftr_after_set_target_gets_its_own_features0(self):
        # The cached representation belongs to the SIFT matcher set before; the LoFTR swapped in extracts the target
        # itself and gets its own features0 (not image0 next to it, which LoFTR rejects).
        tracker = HomographyTracker(self._sift_matcher(), self._sift_matcher(), minimum_inliers_num=10)
        tracker.set_target(_texture(64, 80))
        loftr = LoFTR(None)
        inputs = []
        loftr.register_forward_pre_hook(lambda mod, args: inputs.append(dict(args[0])))
        tracker.fast_matcher = loftr
        tracker.previous_homography = torch.eye(3)
        tracker.track_next_frame(_texture(72, 96, seed=1))
        assert len(inputs) == 1
        assert sorted(inputs[0]) == ["features0", "image1"]
        assert inputs[0]["features0"] is tracker.target_fast_representation["features"]

    @staticmethod
    def _loftr(seed: int) -> LoFTR:
        # Random weights; a low coarse threshold lets matches through, so different weights give different matches.
        config = copy.deepcopy(default_cfg)
        config["match_coarse"]["thr"] = 1e-4
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return LoFTR(None, config)

    def test_loftr_swapped_after_set_target_matches_an_uncached_call(self):
        # CodeRabbit 4277422844: a LoFTR swapped in after set_target was fed the old LoFTR's cached features0.
        old, new = self._loftr(1), self._loftr(2)
        tracker = HomographyTracker(old, self._sift_matcher(), minimum_inliers_num=10**6)
        target, frame = _texture(64, 80), _texture(72, 96, seed=1)
        tracker.set_target(target)
        tracker.initial_matcher = new
        outputs, shapes = [], []
        new.register_forward_hook(lambda mod, args, out: outputs.append(out))
        new.backbone.register_forward_hook(lambda mod, inp, out: shapes.append(tuple(inp[0].shape)))
        tracker.match_initial(frame)  # with gradients enabled: the target is still extracted without a graph
        assert not tracker.target_initial_representation["features"]["coarse"].requires_grad
        with torch.no_grad():
            tracker.match_initial(frame)
            fresh = new({"image0": target, "image1": frame})
        assert len(fresh["keypoints0"]) > 0
        for out in outputs[:2]:
            for k in ("keypoints0", "keypoints1", "confidence", "batch_indexes"):
                assert torch.equal(out[k], fresh[k])
        assert shapes[:3] == [(1, 1, 64, 80), (1, 1, 72, 96), (1, 1, 72, 96)]  # the target once, after the swap

    def test_local_feature_matcher_swapped_after_set_target_matches_an_uncached_call(self):
        tracker = HomographyTracker(self._sift_matcher(), self._sift_matcher(), minimum_inliers_num=10**6)
        target = _texture(64, 80)
        new = LocalFeatureMatcher(SIFTFeature(8), DescriptorMatcher("snn", 0.9))
        inputs, outputs, shapes = [], [], []
        new.register_forward_pre_hook(lambda mod, args: inputs.append(args[0]["image1"]))
        new.register_forward_hook(lambda mod, args, out: outputs.append(out))
        new.local_feature.register_forward_hook(lambda mod, inp, out: shapes.append(tuple(inp[0].shape)))
        with torch.no_grad():
            tracker.set_target(target)
            tracker.fast_matcher = new
            for _ in range(2):
                tracker.previous_homography = torch.eye(3)
                tracker.track_next_frame(_texture(64, 80))
            in_tracker = list(shapes)
            fresh = new({"image0": target, "image1": inputs[0]})
        assert len(fresh["keypoints0"]) > 0
        for k in ("keypoints0", "keypoints1", "batch_indexes"):
            assert torch.equal(outputs[0][k], fresh[k])
        assert in_tracker == [(1, 1, 64, 80)] * 3  # the target once after the swap, then the warped frame twice

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

    def test_target_assigned_directly_starts_a_new_track(self):
        # CodeRabbit 4277848437: after a target was assigned directly, forward tracked it with the old target's prior.
        tracker = _make_tracker()
        tracker.set_target(torch.rand(1, 1, 8, 9))
        tracker.initial_matcher.return_value = _match_dict(20, torch.device("cpu"), torch.float32)
        tracker.fast_matcher.return_value = _match_dict(20, torch.device("cpu"), torch.float32)
        tracker.ransac.return_value = (torch.eye(3), torch.ones(20, dtype=torch.bool))
        assert tracker(torch.rand(1, 1, 8, 9))[1]
        tracker.target = torch.rand(1, 1, 6, 7)
        assert tracker(torch.rand(1, 1, 6, 7))[1]
        assert tracker.initial_matcher.call_count == 2
        tracker.fast_matcher.assert_not_called()
        assert tracker(torch.rand(1, 1, 6, 7))[1]  # the new target's own prior is kept
        tracker.fast_matcher.assert_called_once()
        tracker.target = torch.rand(1, 1, 6, 7)
        with pytest.raises(BaseError, match="previous homography"):
            tracker.track_next_frame(torch.rand(1, 1, 6, 7))
        tracker.fast_matcher.assert_called_once()
        # A homography assigned from outside after a reassignment is the caller's prior for the new target: kept.
        assert tracker(torch.rand(1, 1, 6, 7))[1]  # match_initial finds a prior for the current target
        tracker.target = torch.rand(1, 1, 6, 7)
        tracker.previous_homography = torch.eye(3)
        assert tracker(torch.rand(1, 1, 6, 7))[1]
        assert tracker.initial_matcher.call_count == 3
        assert tracker.fast_matcher.call_count == 2


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

    def test_custom_extract_features_entries_reach_the_matcher(self):
        # A custom matcher whose `extract_features` is an nn.Module gets image0, image1 and every cached entry with a
        # "0" suffix, in match_initial and in track_next_frame; the target is extracted once, in set_target.
        initial, fast = _CustomMatcher(), _CustomMatcher()
        ransac = MagicMock(return_value=(torch.eye(3), torch.ones(20, dtype=torch.bool)))
        tracker = HomographyTracker(initial, fast, ransac, minimum_inliers_num=5)
        target = torch.rand(1, 1, 8, 9)
        tracker.set_target(target)
        assert tracker(torch.rand(1, 1, 8, 9))[1]  # match_initial
        assert tracker(torch.rand(1, 1, 8, 9))[1]  # track_next_frame
        cached_by_matcher = (
            (initial, tracker.target_initial_representation),
            (fast, tracker.target_fast_representation),
        )
        for matcher, cached in cached_by_matcher:
            assert matcher.extract_features.calls == 1
            (data,) = matcher.inputs
            assert sorted(data) == ["desc0", "image0", "image1"]
            assert data["image0"] is target
            assert data["desc0"] is cached["desc"]

    def test_custom_matcher_swapped_or_target_reassigned_after_set_target(self):
        # A replaced matcher, or a target assigned directly, gets the target extracted again, once, by the matcher in
        # use: the cache held the old matcher's features of the old target.
        initial, fast = _CustomMatcher(), _CustomMatcher()
        ransac = MagicMock(return_value=(torch.eye(3), torch.ones(20, dtype=torch.bool)))
        tracker = HomographyTracker(initial, fast, ransac, minimum_inliers_num=5)
        tracker.set_target(torch.rand(1, 1, 8, 9))
        new_fast = _CustomMatcher()
        tracker.fast_matcher = new_fast
        other = torch.rand(1, 1, 8, 9)
        tracker.target = other
        for _ in range(2):
            tracker.match_initial(torch.rand(1, 1, 8, 9))
            tracker.track_next_frame(torch.rand(1, 1, 8, 9))
        assert initial.extract_features.calls == 2  # set_target, then the reassigned target once
        assert new_fast.extract_features.calls == 1
        assert fast.inputs == []
        for matcher in (initial, new_fast):
            first, second = matcher.inputs
            assert torch.equal(first["desc0"], other.mean((2, 3)))
            assert second["desc0"] is first["desc0"]

    def test_extract_features_that_is_not_a_module_is_not_called(self):
        # A matcher that is neither LoFTR nor LocalFeatureMatcher, and whose extract_features is not an nn.Module, gets
        # no cached representation.
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
