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

RS_OPS = {
    "color": ("rgb_to_grayscale", "rgb_to_hsv", "hsv_to_rgb", "rgb_to_lab", "rgb_to_ycbcr", "raw_to_rgb"),
    "geometry": ("warp_perspective", "warp_affine", "rotate", "rotate_shared", "resize"),
}


def build(module, suite):
    args = (2, 16, 20, torch.device("cpu"), torch.float32, False, None, None)
    if suite == "color":
        args += (None, None)
    return module.build_ops(*args, selected=frozenset(RS_OPS[suite]))[0]


def require_rs():
    rs = pytest.importorskip("kornia_rs")
    if not hasattr(rs, "imgproc") or not hasattr(rs.imgproc, "rgb_from_hsv"):
        pytest.skip("Requires the kornia-rs 0.2 color/geometry API")
    return rs


@pytest.mark.device_agnostic
class TestFlagshipRsColorGeometry(BaseTester):
    @pytest.mark.parametrize("suite", RS_OPS)
    def test_missing_optional_api(self, suite, monkeypatch):
        module = load_suite(suite)
        monkeypatch.setattr(module, "kornia_rs_fn", lambda name: None, raising=False)
        for row in build(module, suite).values():
            assert "kornia-rs" in row
            assert row["kornia-rs"] is None
            assert torch.isfinite(row["kornia (eager)"]()).all()

    @pytest.mark.parametrize("op", RS_OPS["color"])
    def test_color_native_units(self, op):
        require_rs()
        row = build(load_suite("color"), "color")[op]
        assert callable(row.get("kornia-rs"))
        native = np.asarray(row["kornia-rs"]())
        actual = torch.from_numpy(native).permute(0, 3, 1, 2).float()
        expected = row["kornia (eager)"]()
        if op == "rgb_to_hsv":
            assert native.dtype == np.float32
            actual[:, 0] *= 2 * torch.pi / 255
            actual[:, 1:] /= 255
        elif op == "rgb_to_lab":
            assert native.dtype == np.float32
        else:
            assert native.dtype == (np.float32 if op == "hsv_to_rgb" else np.uint8)
            actual /= 255
        if op == "rgb_to_ycbcr":
            # The Rust function's native ordering is Y, Cr, Cb despite its name.
            actual = actual[:, [0, 2, 1]]
        if op == "raw_to_rgb":
            # Both demosaic RGGB; their border interpolation rules differ.
            actual, expected = actual[..., 2:-2, 2:-2], expected[..., 2:-2, 2:-2]
        self.assert_close(actual, expected, atol=0.02 if op == "rgb_to_lab" else 2 / 255, rtol=0)

    @pytest.mark.parametrize("op", RS_OPS["geometry"])
    def test_geometry_interior_parity(self, op, monkeypatch):
        require_rs()
        module = load_suite("geometry")

        def smooth_images(b, h, w, device, dtype):
            # A low-frequency image makes byte rounding observable without amplifying
            # tiny coordinate quantization differences into large pixel differences.
            yy, xx = np.mgrid[:h, :w]
            image = np.stack((xx * 4 + 20, yy * 4 + 20, xx + yy + 20), axis=-1).astype(np.uint8)
            images = [image.copy() for _ in range(b)]
            tensor = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2).to(device=device, dtype=dtype) / 255
            return images, tensor

        monkeypatch.setattr(module, "image_batch", smooth_images)
        row = build(module, "geometry")[op]
        assert callable(row.get("kornia-rs"))
        native = np.asarray(row["kornia-rs"]())
        assert native.dtype == np.uint8
        actual = torch.from_numpy(native).permute(0, 3, 1, 2).float() / 255
        expected = row["kornia (eager)"]()
        assert actual.shape == expected.shape
        if op == "resize":
            self.assert_close(actual, expected, atol=1 / 255, rtol=0)
        else:
            # Independent all-ones image identifies positions whose bilinear footprint
            # lies fully inside the source; the Rust border rule is tested separately.
            monkeypatch.setattr(
                module,
                "image_batch",
                lambda b, h, w, device, dtype: (
                    [np.full((h, w, 3), 255, np.uint8) for _ in range(b)],
                    torch.ones(b, 3, h, w, device=device, dtype=dtype),
                ),
            )
            mask_row = build(module, "geometry")[op]
            mask = mask_row["kornia (eager)"]() > 1 - 1e-5
            assert mask.any()
            self.assert_close(actual[mask], expected[mask], atol=1 / 255, rtol=0)

    def test_rotate_uses_same_precomputed_matrix(self):
        require_rs()
        rows = build(load_suite("geometry"), "geometry")
        np.testing.assert_array_equal(rows["rotate"]["kornia-rs"](), rows["rotate_shared"]["kornia-rs"]())

    @pytest.mark.parametrize("op", ["warp_affine", "warp_perspective"])
    def test_native_warp_border_convention(self, op):
        rs = require_rs()
        image = np.full((4, 5, 3), 200, np.uint8)
        matrix = [1.0, 0.0, 0.5, 0.0, 1.0, 0.0]
        if op == "warp_perspective":
            matrix += [0.0, 0.0, 1.0]
        result = getattr(rs.imgproc, op)(image, matrix, (4, 5), "bilinear")
        np.testing.assert_array_equal(result[0, :, 0], [0, 200, 200, 200, 200])
        matrix[2] = -0.5
        result = getattr(rs.imgproc, op)(image, matrix, (4, 5), "bilinear")
        np.testing.assert_array_equal(result, image)
