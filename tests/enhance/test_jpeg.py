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

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestDiffJPEG(BaseTester):
    def test_smoke(self, device, dtype) -> None:
        """This test standard usage."""
        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_keyword_argument(self, device, dtype) -> None:
        """Regression test for #3745: the image must be passable via the ``input=`` keyword."""
        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(input=img, jpeg_quality=jpeg_quality)
        self.assert_close(img_jpeg, kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality))

    def test_smoke_not_div_by_16(self, device, dtype) -> None:
        """This test standard usage."""
        B, H, W = 2, 33, 33
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_multi_batch(self, device, dtype) -> None:
        """Here we test two batch dimensions."""
        B, H, W = 4, 32, 32
        img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(1,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_custom_qt(self, device, dtype) -> None:
        """Here we test if we can handle custom quantization tables."""
        B, H, W = 4, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_non_batch_param(self, device, dtype) -> None:
        """Here we test if we can handle non-batched JPEG parameters (JPEG quality and QT's)."""
        B, H, W = 3, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(1,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(1, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(1, 8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_non_batch_inp(self, device, dtype) -> None:
        """Here we test if we can handle non-batched inputs (input image, JPEG quality, and QT's)."""
        H, W = 32, 32
        img = torch.rand(3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(1,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_quality_zero_is_finite(self, device, dtype) -> None:
        """A quality of 0 is inside the documented [0, 100] range and must not produce NaN (#4205)."""
        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.zeros(B, device=device, dtype=dtype).requires_grad_(True)

        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        assert torch.isfinite(img_jpeg).all()

        img_jpeg.sum().backward()
        assert jpeg_quality.grad is not None
        assert torch.isfinite(jpeg_quality.grad).all()

    def test_quality_zero_matches_quality_one(self, device, dtype) -> None:
        """Quality 0 is clamped to 1, the strongest compression, as libjpeg does (#4205)."""
        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)

        img_zero = kornia.enhance.jpeg_codec_differentiable(img, torch.zeros(B, device=device, dtype=dtype))
        img_one = kornia.enhance.jpeg_codec_differentiable(img, torch.ones(B, device=device, dtype=dtype))
        self.assert_close(img_zero, img_one)

    def test_sub_unit_quality_is_untouched(self, device, dtype) -> None:
        """The guard is exactly the zero point: a quality in (0, 1) was already finite and keeps its own
        (harsher) scale rather than collapsing onto quality 1 (#4205)."""
        B, H, W = 1, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        qt = torch.ones(1, 8, 8, device=device, dtype=dtype)

        img_half = kornia.enhance.jpeg_codec_differentiable(
            img, torch.tensor([0.5], device=device, dtype=dtype), qt, qt
        )
        img_one = kornia.enhance.jpeg_codec_differentiable(img, torch.tensor([1.0], device=device, dtype=dtype), qt, qt)

        assert torch.isfinite(img_half).all()
        assert (img_half - img_one).abs().max() > 1e-2

    def test_quality_zero_only_affects_its_own_batch_entry(self, device, dtype) -> None:
        """The guard is elementwise: a 0 in the batch must not disturb its batch-mates (#4205)."""
        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        mixed = torch.tensor([0.0, 50.0], device=device, dtype=dtype)

        img_mixed = kornia.enhance.jpeg_codec_differentiable(img, mixed)
        img_alone = kornia.enhance.jpeg_codec_differentiable(img, torch.full((B,), 50.0, device=device, dtype=dtype))
        self.assert_close(img_mixed[1], img_alone[1])

    def test_exception(self, device, dtype) -> None:
        """Test exceptions (non-tensor input, wrong JPEG quality shape, wrong img shape, and wrong QT shape.)"""
        B = 2
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        with pytest.raises(TypeError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(1904.0, jpeg_quality)
        assert "Input input type is not a torch.Tensor" in str(errinfo.value)

        from kornia.core.exceptions import TypeCheckError

        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        with pytest.raises(TypeCheckError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(img, None)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        from kornia.core.exceptions import BaseError

        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B, 3, 2, 1), device=device, dtype=dtype)
        with pytest.raises(BaseError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        from kornia.core.exceptions import BaseError

        B, H, W = 4, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B, 7, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        with pytest.raises(BaseError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert (
            "Shape dimension mismatch" in str(errinfo.value)
            or "Expected shape" in str(errinfo.value)
            or "shape must be" in str(errinfo.value)
        )

        from kornia.core.exceptions import BaseError

        B, H, W = 4, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 7), device=device, dtype=dtype)
        with pytest.raises(BaseError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert (
            "Shape dimension mismatch" in str(errinfo.value)
            or "Expected shape" in str(errinfo.value)
            or "shape must be" in str(errinfo.value)
        )

        from kornia.core.exceptions import BaseError

        B, H, W = 4, 32, 32
        img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B * B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B * 2, 8, 8), device=device, dtype=dtype)
        with pytest.raises(BaseError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "Batch dimensions do not match" in str(errinfo.value)

        from kornia.core.exceptions import BaseError

        B, H, W = 4, 32, 32
        img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B * B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B * 2, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        with pytest.raises(BaseError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "Batch dimensions do not match" in str(errinfo.value)

        from kornia.core.exceptions import BaseError

        B, H, W = 4, 32, 32
        img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B * 2,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        with pytest.raises(BaseError) as errinfo:
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "Batch dimensions do not match" in str(errinfo.value)

    def test_cardinality(self, device, dtype) -> None:
        B, H, W = 1, 16, 16
        img = torch.zeros(B, 3, H, W, device=device, dtype=dtype)
        img[..., 4:-4, 4:-4] = 1.0
        jpeg_quality = torch.tensor([2.0], device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        # Numbers generated based on reference implementation
        img_jpeg_ref = torch.tensor(
            [
                [
                    [
                        [
                            -0.000,
                            0.002,
                            0.063,
                            0.060,
                            0.020,
                            0.017,
                            0.078,
                            0.146,
                            0.146,
                            0.078,
                            0.017,
                            0.020,
                            0.060,
                            0.063,
                            0.002,
                            -0.000,
                        ],
                        [
                            0.002,
                            0.015,
                            0.008,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.008,
                            0.015,
                            0.002,
                        ],
                        [
                            0.063,
                            0.009,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.009,
                            0.063,
                        ],
                        [
                            0.060,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.173,
                            0.246,
                            0.178,
                            0.080,
                            0.080,
                            0.178,
                            0.246,
                            0.173,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.060,
                        ],
                        [
                            0.020,
                            -0.000,
                            -0.000,
                            0.173,
                            0.694,
                            0.971,
                            0.960,
                            0.847,
                            0.847,
                            0.960,
                            0.971,
                            0.694,
                            0.173,
                            -0.000,
                            -0.000,
                            0.020,
                        ],
                        [
                            0.017,
                            -0.000,
                            -0.000,
                            0.246,
                            0.971,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.971,
                            0.246,
                            -0.000,
                            -0.000,
                            0.017,
                        ],
                        [
                            0.078,
                            -0.000,
                            -0.000,
                            0.178,
                            0.960,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.960,
                            0.178,
                            -0.000,
                            -0.000,
                            0.078,
                        ],
                        [
                            0.146,
                            -0.000,
                            -0.000,
                            0.080,
                            0.847,
                            1.000,
                            1.000,
                            0.781,
                            0.781,
                            1.000,
                            1.000,
                            0.847,
                            0.080,
                            -0.000,
                            -0.000,
                            0.146,
                        ],
                        [
                            0.146,
                            -0.000,
                            -0.000,
                            0.080,
                            0.847,
                            1.000,
                            1.000,
                            0.781,
                            0.781,
                            1.000,
                            1.000,
                            0.847,
                            0.080,
                            -0.000,
                            -0.000,
                            0.146,
                        ],
                        [
                            0.078,
                            -0.000,
                            -0.000,
                            0.178,
                            0.960,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.960,
                            0.178,
                            -0.000,
                            -0.000,
                            0.078,
                        ],
                        [
                            0.017,
                            -0.000,
                            -0.000,
                            0.246,
                            0.971,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.971,
                            0.246,
                            -0.000,
                            -0.000,
                            0.017,
                        ],
                        [
                            0.020,
                            -0.000,
                            -0.000,
                            0.173,
                            0.694,
                            0.971,
                            0.960,
                            0.847,
                            0.847,
                            0.960,
                            0.971,
                            0.694,
                            0.173,
                            -0.000,
                            -0.000,
                            0.020,
                        ],
                        [
                            0.060,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.173,
                            0.246,
                            0.178,
                            0.080,
                            0.080,
                            0.178,
                            0.246,
                            0.173,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.060,
                        ],
                        [
                            0.063,
                            0.009,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.009,
                            0.063,
                        ],
                        [
                            0.002,
                            0.015,
                            0.008,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.008,
                            0.015,
                            0.002,
                        ],
                        [
                            -0.000,
                            0.002,
                            0.063,
                            0.060,
                            0.020,
                            0.017,
                            0.078,
                            0.146,
                            0.146,
                            0.078,
                            0.017,
                            0.020,
                            0.060,
                            0.063,
                            0.002,
                            -0.000,
                        ],
                    ],
                    [
                        [
                            -0.000,
                            0.002,
                            0.063,
                            0.060,
                            0.020,
                            0.017,
                            0.078,
                            0.146,
                            0.146,
                            0.078,
                            0.017,
                            0.020,
                            0.060,
                            0.063,
                            0.002,
                            -0.000,
                        ],
                        [
                            0.002,
                            0.015,
                            0.008,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.008,
                            0.015,
                            0.002,
                        ],
                        [
                            0.063,
                            0.009,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.009,
                            0.063,
                        ],
                        [
                            0.060,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.173,
                            0.246,
                            0.178,
                            0.080,
                            0.080,
                            0.178,
                            0.246,
                            0.173,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.060,
                        ],
                        [
                            0.020,
                            -0.000,
                            -0.000,
                            0.173,
                            0.694,
                            0.971,
                            0.960,
                            0.847,
                            0.847,
                            0.960,
                            0.971,
                            0.694,
                            0.173,
                            -0.000,
                            -0.000,
                            0.020,
                        ],
                        [
                            0.017,
                            -0.000,
                            -0.000,
                            0.246,
                            0.971,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.971,
                            0.246,
                            -0.000,
                            -0.000,
                            0.017,
                        ],
                        [
                            0.078,
                            -0.000,
                            -0.000,
                            0.178,
                            0.960,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.960,
                            0.178,
                            -0.000,
                            -0.000,
                            0.078,
                        ],
                        [
                            0.146,
                            -0.000,
                            -0.000,
                            0.080,
                            0.847,
                            1.000,
                            1.000,
                            0.781,
                            0.781,
                            1.000,
                            1.000,
                            0.847,
                            0.080,
                            -0.000,
                            -0.000,
                            0.146,
                        ],
                        [
                            0.146,
                            -0.000,
                            -0.000,
                            0.080,
                            0.847,
                            1.000,
                            1.000,
                            0.781,
                            0.781,
                            1.000,
                            1.000,
                            0.847,
                            0.080,
                            -0.000,
                            -0.000,
                            0.146,
                        ],
                        [
                            0.078,
                            -0.000,
                            -0.000,
                            0.178,
                            0.960,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.960,
                            0.178,
                            -0.000,
                            -0.000,
                            0.078,
                        ],
                        [
                            0.017,
                            -0.000,
                            -0.000,
                            0.246,
                            0.971,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.971,
                            0.246,
                            -0.000,
                            -0.000,
                            0.017,
                        ],
                        [
                            0.020,
                            -0.000,
                            -0.000,
                            0.173,
                            0.694,
                            0.971,
                            0.960,
                            0.847,
                            0.847,
                            0.960,
                            0.971,
                            0.694,
                            0.173,
                            -0.000,
                            -0.000,
                            0.020,
                        ],
                        [
                            0.060,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.173,
                            0.246,
                            0.178,
                            0.080,
                            0.080,
                            0.178,
                            0.246,
                            0.173,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.060,
                        ],
                        [
                            0.063,
                            0.009,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.009,
                            0.063,
                        ],
                        [
                            0.002,
                            0.015,
                            0.008,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.008,
                            0.015,
                            0.002,
                        ],
                        [
                            -0.000,
                            0.002,
                            0.063,
                            0.060,
                            0.020,
                            0.017,
                            0.078,
                            0.146,
                            0.146,
                            0.078,
                            0.017,
                            0.020,
                            0.060,
                            0.063,
                            0.002,
                            -0.000,
                        ],
                    ],
                    [
                        [
                            -0.000,
                            0.002,
                            0.063,
                            0.060,
                            0.020,
                            0.017,
                            0.078,
                            0.146,
                            0.146,
                            0.078,
                            0.017,
                            0.020,
                            0.060,
                            0.063,
                            0.002,
                            -0.000,
                        ],
                        [
                            0.002,
                            0.015,
                            0.008,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.008,
                            0.015,
                            0.002,
                        ],
                        [
                            0.063,
                            0.009,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.009,
                            0.063,
                        ],
                        [
                            0.060,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.173,
                            0.246,
                            0.178,
                            0.080,
                            0.080,
                            0.178,
                            0.246,
                            0.173,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.060,
                        ],
                        [
                            0.020,
                            -0.000,
                            -0.000,
                            0.173,
                            0.694,
                            0.971,
                            0.960,
                            0.847,
                            0.847,
                            0.960,
                            0.971,
                            0.694,
                            0.173,
                            -0.000,
                            -0.000,
                            0.020,
                        ],
                        [
                            0.017,
                            -0.000,
                            -0.000,
                            0.246,
                            0.971,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.971,
                            0.246,
                            -0.000,
                            -0.000,
                            0.017,
                        ],
                        [
                            0.078,
                            -0.000,
                            -0.000,
                            0.178,
                            0.960,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.960,
                            0.178,
                            -0.000,
                            -0.000,
                            0.078,
                        ],
                        [
                            0.146,
                            -0.000,
                            -0.000,
                            0.080,
                            0.847,
                            1.000,
                            1.000,
                            0.781,
                            0.781,
                            1.000,
                            1.000,
                            0.847,
                            0.080,
                            -0.000,
                            -0.000,
                            0.146,
                        ],
                        [
                            0.146,
                            -0.000,
                            -0.000,
                            0.080,
                            0.847,
                            1.000,
                            1.000,
                            0.781,
                            0.781,
                            1.000,
                            1.000,
                            0.847,
                            0.080,
                            -0.000,
                            -0.000,
                            0.146,
                        ],
                        [
                            0.078,
                            -0.000,
                            -0.000,
                            0.178,
                            0.960,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.960,
                            0.178,
                            -0.000,
                            -0.000,
                            0.078,
                        ],
                        [
                            0.017,
                            -0.000,
                            -0.000,
                            0.246,
                            0.971,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            1.000,
                            0.971,
                            0.246,
                            -0.000,
                            -0.000,
                            0.017,
                        ],
                        [
                            0.020,
                            -0.000,
                            -0.000,
                            0.173,
                            0.694,
                            0.971,
                            0.960,
                            0.847,
                            0.847,
                            0.960,
                            0.971,
                            0.694,
                            0.173,
                            -0.000,
                            -0.000,
                            0.020,
                        ],
                        [
                            0.060,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.173,
                            0.246,
                            0.178,
                            0.080,
                            0.080,
                            0.178,
                            0.246,
                            0.173,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.060,
                        ],
                        [
                            0.063,
                            0.009,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.009,
                            0.063,
                        ],
                        [
                            0.002,
                            0.015,
                            0.008,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            -0.000,
                            0.008,
                            0.015,
                            0.002,
                        ],
                        [
                            -0.000,
                            0.002,
                            0.063,
                            0.060,
                            0.020,
                            0.017,
                            0.078,
                            0.146,
                            0.146,
                            0.078,
                            0.017,
                            0.020,
                            0.060,
                            0.063,
                            0.002,
                            -0.000,
                        ],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        # We use a slightly higher tolerance since our implementation varies from the reference implementation
        self.assert_close(img_jpeg, img_jpeg_ref, rtol=0.01, atol=0.01)

    def test_module(self, device, dtype) -> None:
        B, H, W = 4, 16, 16
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        diff_jpeg_module = kornia.enhance.JPEGCodecDifferentiable(qt_y, qt_c)
        img_jpeg = diff_jpeg_module(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_module_with_param(self, device, dtype) -> None:
        B, H, W = 4, 16, 16
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=1, high=100, size=(B,), device=device, dtype=dtype)
        qt_y = torch.nn.Parameter(torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype))
        qt_c = torch.nn.Parameter(torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype))
        diff_jpeg_module = kornia.enhance.JPEGCodecDifferentiable(qt_y, qt_c)
        img_jpeg = diff_jpeg_module(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    # @pytest.mark.slow
    def test_gradcheck(self, device) -> None:
        """We test that the gradient matches the gradient of the reference implementation."""
        B, H, W = 1, 16, 16
        img = torch.zeros(B, 3, H, W, device=device, dtype=torch.float)
        img[..., 0, 4:-4, 4:-4] = 1.0
        img[..., 1, 4:-4, 4:-4] = 0.5
        img[..., 2, 4:-4, 4:-4] = 0.5
        img.requires_grad = True
        jpeg_quality = torch.tensor([10.0], device=device, dtype=torch.float, requires_grad=True)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        (img_jpeg - torch.zeros_like(img_jpeg)).abs().sum().backward()
        # Numbers generated based on reference implementation
        img_jpeg_mean_grad_ref = torch.tensor([0.1919], device=device)
        jpeg_quality_grad_ref = torch.tensor([0.0042], device=device)
        # We use a slightly higher tolerance since our implementation varies from the reference implementation
        self.assert_close(img.grad.mean().view(-1), img_jpeg_mean_grad_ref, rtol=0.01, atol=0.01)
        self.assert_close(jpeg_quality.grad, jpeg_quality_grad_ref, rtol=0.01, atol=0.01)

    def test_basis_cache_after_inference_mode(self, device, dtype, monkeypatch) -> None:
        """A first call under ``torch.inference_mode()`` must not break later calls that track gradients.

        The DCT basis is memoized per dtype and device. Caching the inference tensors built by that first call made
        later calls in which autograd tracks the image fail in the forward with "Inference tensors cannot be saved
        for backward".
        """
        monkeypatch.setattr("kornia.enhance.jpeg._DCT8_CACHE", {})
        torch.manual_seed(0)
        img = torch.rand(2, 3, 16, 16).to(device=device, dtype=dtype)
        jpeg_quality = torch.tensor([30.0, 90.0], device=device, dtype=dtype)
        with torch.inference_mode():
            expected = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        img = img.clone().requires_grad_(True)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        img_jpeg.sum().backward()
        self.assert_close(img_jpeg.detach(), expected)
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()

    def test_basis_cache_after_compile_in_inference_mode(self, device, dtype, torch_optimizer, monkeypatch) -> None:
        """A compiled first call under ``torch.inference_mode()`` must not break later calls that track gradients."""
        monkeypatch.setattr("kornia.enhance.jpeg._DCT8_CACHE", {})
        torch.manual_seed(0)
        img = torch.rand(2, 3, 16, 16).to(device=device, dtype=dtype)
        jpeg_quality = torch.tensor([30.0, 90.0], device=device, dtype=dtype)
        op_optimized = torch_optimizer(kornia.enhance.jpeg_codec_differentiable)
        with torch.inference_mode():
            op_optimized(img, jpeg_quality)
        img = img.clone().requires_grad_(True)
        kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality).sum().backward()
        assert img.grad is not None
        assert torch.isfinite(img.grad).all()

    @pytest.mark.device_agnostic
    def test_basis_cache_after_export(self, monkeypatch) -> None:
        """Tracing with ``torch.export`` must not leave fake tensors in the DCT basis cache.

        The trace runs the forward on fake tensors. Caching the basis built there made later eager calls in the same
        process return fake tensors without data: ``.item()`` gave a symbol and ``.numpy()`` raised.
        """
        monkeypatch.setattr("kornia.enhance.jpeg._DCT8_CACHE", {})
        torch.manual_seed(0)
        img = torch.rand(1, 3, 16, 16)
        jpeg_quality = torch.tensor([50.0])
        module = kornia.enhance.JPEGCodecDifferentiable()
        torch.export.export(module, (img, jpeg_quality), strict=False)
        img_jpeg = module(img, jpeg_quality)
        assert type(img_jpeg) is torch.Tensor
        kornia.enhance.jpeg._DCT8_CACHE.clear()
        self.assert_close(img_jpeg, module(img, jpeg_quality))
