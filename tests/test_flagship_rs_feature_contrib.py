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

from __future__ import annotations

import numpy as np
import pytest
import torch

from testing.base import BaseTester

from tests.test_flagship_coverage import load_suite


def build(module, suite):
    args = (2, 64, 80, torch.device("cpu"), torch.float32, False, None, None)
    if suite == "contrib":
        args += (None,)
    selected = frozenset(["SIFTFeature" if suite == "feature" else "connected_components_union_find"])
    return module.build_ops(*args, selected=selected)[0]


def require_rs(name):
    rs = pytest.importorskip("kornia_rs")
    if not hasattr(rs, "imgproc") or not hasattr(rs.imgproc, name):
        pytest.skip("Requires the kornia-rs 0.2 feature/contrib API")
    return rs


@pytest.mark.device_agnostic
class TestFlagshipRsFeatureContrib(BaseTester):
    @pytest.mark.parametrize("suite", ["feature", "contrib"])
    def test_missing_optional_api(self, suite, monkeypatch):
        module = load_suite(suite)
        if suite == "feature":
            monkeypatch.setattr(module, "NUM_DESCRIPTORS", 8)
        monkeypatch.setattr(module, "kornia_rs_fn", lambda name: None, raising=False)
        for row in build(module, suite).values():
            assert "kornia-rs" in row
            assert row["kornia-rs"] is None

    def test_sift_prepared_input_and_persistent_detector(self, monkeypatch):
        module = load_suite("feature")
        monkeypatch.setattr(module, "NUM_DESCRIPTORS", 8)
        instances = []
        calls = []

        class Sift:
            def __init__(self, n_features):
                assert n_features == module.NUM_FEATURES
                instances.append(self)

            def detect_and_compute(self, image):
                assert image.dtype == np.float32
                assert image.flags.c_contiguous
                assert image.ndim == 3
                assert image.shape[-1] == 1
                assert 1 < image.max() <= 255
                assert image.min() >= 0
                calls.append(image)
                return [], np.empty((0, 128), dtype=np.float32)

        monkeypatch.setattr(module, "kornia_rs_fn", lambda name: Sift if name == "Sift" else None, raising=False)
        row = build(module, "feature")["SIFTFeature"]
        assert callable(row.get("kornia-rs"))
        assert len(instances) == 1
        assert not calls
        row["kornia-rs"]()
        row["kornia-rs"]()
        assert len(instances) == 1
        assert calls[0] is calls[2]
        assert calls[1] is calls[3]

    def test_sift_matches_opencv_feature_count(self, monkeypatch):
        require_rs("Sift")
        cv2 = pytest.importorskip("cv2")
        module = load_suite("feature")
        monkeypatch.setattr(module, "NUM_DESCRIPTORS", 8)
        row = build(module, "feature")["SIFTFeature"]
        assert callable(row.get("kornia-rs"))
        detector = cv2.SIFT_create(nfeatures=module.NUM_FEATURES)
        gray = module.blob_images(2, 64, 80)
        for image, (keypoints, descriptors) in zip(gray, row["kornia-rs"]()):
            pixels = (image[0].numpy() * 255).round().astype(np.uint8)
            expected, _ = detector.detectAndCompute(pixels, None)
            assert len(keypoints) == len(expected) > 0
            assert descriptors.shape == (len(keypoints), 128)
            assert descriptors.dtype == np.float32
            assert np.isfinite(descriptors).all()

    def test_connected_components_partition(self, monkeypatch):
        require_rs("connected_components")
        module = load_suite("contrib")
        # The diagonal pair must join under 8-connectivity; the distant point
        # remains separate. Compare partitions, since native label IDs and
        # whether the returned count includes background can differ.
        masks = torch.zeros(2, 1, 64, 80)
        masks[:, :, 2, 2] = 1
        masks[:, :, 3, 3] = 1
        masks[:, :, 50, 50] = 1
        monkeypatch.setattr(module, "blob_masks", lambda b, h, w: masks)
        row = build(module, "contrib")["connected_components_union_find"]
        assert callable(row.get("kornia-rs"))
        expected = row["kornia (eager)"]()
        for i, (_, labels) in enumerate(row["kornia-rs"]()):
            labels = np.asarray(labels).reshape(64, 80)
            foreground = masks[i, 0].numpy().astype(bool)
            assert np.array_equal(labels != 0, foreground)
            actual = torch.from_numpy(labels[foreground])
            reference = expected[i, 0][torch.from_numpy(foreground)]
            self.assert_close(actual[:, None] == actual[None, :], reference[:, None] == reference[None, :])
