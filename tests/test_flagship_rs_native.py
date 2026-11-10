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

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from testing.base import BaseTester

from tests.test_flagship_coverage import load_suite


class TestFlagshipRsNative(BaseTester):
    @pytest.mark.device_agnostic
    def test_resolver_layouts(self, monkeypatch):
        load_suite("enhance")
        common = sys.modules["common"]

        def fn():
            return None

        for wheel in (SimpleNamespace(probe=fn), SimpleNamespace(imgproc=SimpleNamespace(probe=fn))):
            monkeypatch.setitem(sys.modules, "kornia_rs", wheel)
            assert common.kornia_rs_fn("probe") is fn
            assert common.kornia_rs_fn("missing") is None
        monkeypatch.setitem(sys.modules, "kornia_rs", None)
        assert common.kornia_rs_fn("probe") is None

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("suite", ["enhance", "morphology", "filters"])
    def test_missing_native_functions(self, suite, monkeypatch):
        module = load_suite(suite)
        monkeypatch.setattr(module, "krs_fn" if suite == "filters" else "kornia_rs_fn", lambda name: None)
        ops = self.build(suite, module, None)
        assert all(row.get("kornia-rs") is None for row in ops.values())

    @staticmethod
    def build(suite, module, cv2):
        args = (2, 32, 40, torch.device("cpu"), torch.float32, False)
        if suite == "enhance":
            libs = dict.fromkeys(("cv2", "tvf", "A", "ske", "pil_enh", "pil_ops", "pil"))
            libs["cv2"] = cv2
            return module.build_ops(*args, libs, selected=frozenset(("normalize", "equalize", "equalize_clahe")))[0]
        if suite == "morphology":
            return module.build_ops(*args, cv2, None, None, None)[0]
        return module.build_ops(*args, cv2, None, None, None, None, selected_ops=frozenset(("canny",)))[0]

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("suite", ["enhance", "morphology", "filters"])
    def test_native_outputs(self, suite):
        pytest.importorskip("kornia_rs")
        cv2 = pytest.importorskip("cv2")
        module = load_suite(suite)
        ops = self.build(suite, module, cv2)
        required = {
            "enhance": ("normalize_mean_std", "equalize_hist", "clahe"),
            "morphology": ("dilate", "erode"),
            "filters": ("canny", "gray_from_rgb"),
        }
        if any(sys.modules["common"].kornia_rs_fn(name) is None for name in required[suite]):
            pytest.skip("installed kornia-rs lacks the requested APIs")
        for name, row in ops.items():
            assert callable(row.get("kornia-rs"))
            actual = row["kornia-rs"]()
            assert len(actual) == 2
            if name == "normalize":
                expected = row["kornia (eager)"]().permute(0, 2, 3, 1)
                self.assert_close(torch.from_numpy(np.asarray(actual)), expected)
            elif name == "canny":
                # RGB->gray rounding can differ by one level: compare Canny using its own gray input.
                imgs, _ = module.image_batch(2, 32, 40, torch.device("cpu"), torch.float32)
                gray = module.krs_fn("gray_from_rgb")
                for image, result in zip(imgs, actual):
                    expected = cv2.Canny(gray(image)[..., 0], 100, 200)
                    np.testing.assert_array_equal(result[..., 0], expected)
            elif name == "equalize_clahe":
                # Identical grid/clip parameters; native interpolation can round one byte differently.
                self.assert_close(
                    torch.from_numpy(np.asarray(actual)).float(),
                    torch.from_numpy(np.asarray(row["opencv"]())).float(),
                    atol=1,
                    rtol=0,
                )
            else:
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(row["opencv"]()))
