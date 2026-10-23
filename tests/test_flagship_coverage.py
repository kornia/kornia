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

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from testing.base import BaseTester

NEW_OPS = {
    "color": {"hsv_to_rgb"},
    "filters": {"spatial_gradient"},
    "geometry": {"remap", "transform_points", "depth_to_3d"},
    "feature": {"match_snn", "match_mnn", "match_smnn", "match_smnn_batched"},
}


def load_suite(suite):
    path = Path(__file__).parents[1] / "benchmarks" / suite / "flagship.py"
    spec = importlib.util.spec_from_file_location(f"flagship_{suite}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def build(suite, module, cv2=None, skc=None):
    args = (2, 16, 20, torch.device("cpu"), torch.float32, False)
    selected = frozenset(NEW_OPS[suite])
    if suite == "color":
        return module.build_ops(*args, cv2, None, skc, None, selected=selected)[0]
    if suite == "filters":
        return module.build_ops(*args, cv2, None, None, None, None, selected_ops=selected)[0]
    return module.build_ops(*args, cv2, None, selected=selected)[0]


class TestFlagshipCoverage(BaseTester):
    @pytest.mark.parametrize("suite", NEW_OPS)
    @pytest.mark.device_agnostic
    def test_without_optional_libraries(self, suite, monkeypatch):
        module = load_suite(suite)
        declared = module.AVAILABLE_OPS if suite == "filters" else module.OPS
        assert NEW_OPS[suite] <= set(declared)
        if suite == "feature":
            monkeypatch.setattr(module, "NUM_DESCRIPTORS", 32)
        ops = build(suite, module)
        assert set(ops) == NEW_OPS[suite]
        for row in ops.values():
            assert row.get("opencv") is None
            result = row["kornia (eager)"]()
            tensors = result if isinstance(result, (list, tuple)) else [result]
            for value in tensors:
                for tensor in value if isinstance(value, tuple) else [value]:
                    assert torch.isfinite(tensor).all()

    @pytest.mark.device_agnostic
    def test_hsv_inverse_native_units(self):
        cv2 = pytest.importorskip("cv2")
        module = load_suite("color")
        row = build("color", module, cv2)["hsv_to_rgb"]
        actual = row["kornia (eager)"]()
        _, expected = module.image_batch(2, 16, 20, torch.device("cpu"), torch.float32)
        self.assert_close(actual, expected)
        for backend in ("opencv", "opencv (stacked)"):
            result = np.asarray(row[backend]()).reshape(2, 16, 20, 3)
            self.assert_close(actual, torch.from_numpy(result).permute(0, 3, 1, 2))

    @pytest.mark.device_agnostic
    def test_hsv_inverse_scikit_image(self):
        skc = pytest.importorskip("skimage.color")
        row = build("color", load_suite("color"), skc=skc)["hsv_to_rgb"]
        expected = torch.from_numpy(np.asarray(row["scikit-image"]())).permute(0, 3, 1, 2)
        self.assert_close(row["kornia (eager)"](), expected)

    @pytest.mark.device_agnostic
    def test_spatial_gradient_opencv(self):
        cv2 = pytest.importorskip("cv2")
        row = build("filters", load_suite("filters"), cv2)["spatial_gradient"]
        # Native OpenCV output is a pair of HWC derivative arrays per image.
        expected = torch.from_numpy(np.asarray(row["opencv"]())).permute(0, 4, 1, 2, 3)
        self.assert_close(row["kornia (eager)"](), expected)

    @pytest.mark.parametrize("op", ["remap", "transform_points"])
    @pytest.mark.device_agnostic
    def test_geometry_opencv(self, op):
        cv2 = pytest.importorskip("cv2")
        row = build("geometry", load_suite("geometry"), cv2)[op]
        actual = row["kornia (eager)"]()
        expected = torch.from_numpy(np.asarray(row["opencv"]()))
        expected = expected.permute(0, 3, 1, 2) if op == "remap" else expected.squeeze(2)
        self.assert_close(actual, expected, rtol=1e-4, atol=2e-5)

    @pytest.mark.device_agnostic
    def test_depth_projection(self):
        module = load_suite("geometry")
        row = build("geometry", module)["depth_to_3d"]
        points = row["kornia (eager)"]()
        assert points.shape == (2, 3, 16, 20)
        assert (points[:, 2] > 0).all()
        y, x = torch.meshgrid(torch.arange(16), torch.arange(20), indexing="ij")
        self.assert_close(points[:, 0] / points[:, 2] * 20 + 9.5, x.float().expand(2, -1, -1))
        self.assert_close(points[:, 1] / points[:, 2] * 20 + 7.5, y.float().expand(2, -1, -1))

    @pytest.mark.parametrize("op", sorted(NEW_OPS["feature"]))
    @pytest.mark.device_agnostic
    def test_matching_opencv(self, op, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        module = load_suite("feature")
        monkeypatch.setattr(module, "NUM_DESCRIPTORS", 32)
        row = build("feature", module, cv2)[op]
        actual = row["kornia (eager)"]()
        if op == "match_smnn_batched":
            _, indices = actual
            actual_pairs = [{tuple(pair.tolist()) for pair in indices[indices[:, 0] == b, 1:]} for b in range(2)]
        else:
            actual_pairs = [{tuple(pair.tolist()) for pair in indices} for _, indices in actual]
        expected = [{(m.queryIdx, m.trainIdx) for m in matches} for matches in row["opencv"]()]
        assert all(actual_pairs)
        assert actual_pairs == expected

    @pytest.mark.device_agnostic
    def test_snn_inclusive_threshold(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        module = load_suite("feature")
        monkeypatch.setattr(
            module,
            "descriptor_pairs",
            lambda b: (torch.zeros(b, 1, 1), torch.tensor([[[4.0], [5.0]]]).expand(b, -1, -1)),
        )
        row = build("feature", module, cv2)["match_snn"]
        assert len(row["kornia (eager)"]()[0][1]) == 1
        assert len(row["opencv"]()[0]) == 1

    @pytest.mark.device_agnostic
    def test_matching_mutuality_and_reverse_ratio(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        module = load_suite("feature")
        # Both first queries pass forward SNN, but only query 0 is mutual. Its
        # reverse ratio is 0.09/0.11 > 0.8, so neither survives symmetric SNN.
        monkeypatch.setattr(
            module,
            "descriptor_pairs",
            lambda b: (
                torch.tensor([[[0.0], [0.2], [10.0]]]).expand(b, -1, -1),
                torch.tensor([[[0.09], [9.0], [20.0]]]).expand(b, -1, -1),
            ),
        )
        expected = {
            "match_snn": {(0, 0), (1, 0), (2, 1)},
            "match_mnn": {(0, 0), (2, 1)},
            "match_smnn": {(2, 1)},
            "match_smnn_batched": {(2, 1)},
        }
        for name, row in build("feature", module, cv2).items():
            result = row["kornia (eager)"]()
            indices = result[1][:, 1:] if name == "match_smnn_batched" else result[0][1]
            assert {tuple(pair.tolist()) for pair in indices} == expected[name]
            assert {(m.queryIdx, m.trainIdx) for m in row["opencv"]()[0]} == expected[name]

    @pytest.mark.parametrize("op", ["dice_loss", "dice_loss_macro", "dice_loss_weighted"])
    def test_dice_variants_backward(self, op, device, dtype):
        module = load_suite("losses")
        ops, failures = module.build_ops(2, 16, 20, device, dtype, False, None, selected=frozenset({op}))
        assert not failures
        (gradient,) = ops[op]["kornia (eager)"]()
        assert gradient.shape == (2, module.NUM_CLASSES, 16, 20)
        assert torch.isfinite(gradient).all()
        assert (gradient != 0).any()

    @pytest.mark.device_agnostic
    def test_hsv_inverse_pil(self):
        pil = pytest.importorskip("PIL.Image")
        module = load_suite("color")
        ops, _ = module.build_ops(
            2,
            16,
            20,
            torch.device("cpu"),
            torch.float32,
            False,
            None,
            None,
            None,
            pil,
            selected=frozenset({"hsv_to_rgb"}),
        )
        row = ops["hsv_to_rgb"]
        expected = (
            torch.from_numpy(np.asarray([np.asarray(im) for im in row["PIL"]()])).permute(0, 3, 1, 2).float() / 255
        )
        # Byte hue quantization can move RGB by approximately three byte levels.
        self.assert_close(row["kornia (eager)"](), expected, atol=4 / 255, rtol=0)
