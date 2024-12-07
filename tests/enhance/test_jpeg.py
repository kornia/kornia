import pytest
import torch

import kornia

from testing.base import BaseTester


class TestDiffJPEG(BaseTester):
    def test_smoke(self, device, dtype) -> None:
        """Test standard usage."""
        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_smoke_not_div_by_16(self, device, dtype) -> None:
        """Test standard usage."""
        B, H, W = 2, 33, 33
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_multi_batch(self, device, dtype) -> None:
        """Here we test two batch dimensions."""
        B, H, W = 4, 32, 32
        img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=0, high=100, size=(1,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_custom_qt(self, device, dtype) -> None:
        """Here we test if we can handle custom quantization tables."""
        B, H, W = 4, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_non_batch_param(self, device, dtype) -> None:
        """Here we test if we can handle non-batched JPEG parameters (JPEG quality and QT's)."""
        B, H, W = 3, 32, 32
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=0, high=100, size=(1,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(1, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(1, 8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_non_batch_inp(self, device, dtype) -> None:
        """Here we test if we can handle non-batched inputs (input image, JPEG quality, and QT's)."""
        H, W = 32, 32
        img = torch.rand(3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=0, high=100, size=(1,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(8, 8), device=device, dtype=dtype)
        img_jpeg = kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_exception(self, device, dtype) -> None:
        """Test exceptions (non-tensor input, wrong JPEG quality shape, wrong img shape, and wrong QT shape.)"""
        with pytest.raises(TypeError) as errinfo:
            B = 2
            jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(1904.0, jpeg_quality)
        assert "Input input type is not a" in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            B, H, W = 2, 32, 32
            img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(img, None)
        assert "Not a Tensor type. Got" in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            B, H, W = 2, 32, 32
            img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
            jpeg_quality = torch.randint(low=0, high=100, size=(B, 3, 2, 1), device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality)
        assert "shape must be [" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 4, 32, 32
            img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
            jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
            qt_y = torch.randint(low=1, high=255, size=(B, 7, 8), device=device, dtype=dtype)
            qt_c = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "shape must be [" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 4, 32, 32
            img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
            jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
            qt_y = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
            qt_c = torch.randint(low=1, high=255, size=(B, 8, 7), device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "shape must be [" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 4, 32, 32
            img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
            jpeg_quality = torch.randint(low=0, high=100, size=(B * B,), device=device, dtype=dtype)
            qt_y = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
            qt_c = torch.randint(low=1, high=255, size=(B * 2, 8, 8), device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "Batch dimensions do not match." in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 4, 32, 32
            img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
            jpeg_quality = torch.randint(low=0, high=100, size=(B * B,), device=device, dtype=dtype)
            qt_y = torch.randint(low=1, high=255, size=(B * 2, 8, 8), device=device, dtype=dtype)
            qt_c = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "Batch dimensions do not match." in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 4, 32, 32
            img = torch.rand(B, B, 3, H, W, device=device, dtype=dtype)
            jpeg_quality = torch.randint(low=0, high=100, size=(B * 2,), device=device, dtype=dtype)
            qt_y = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
            qt_c = torch.randint(low=1, high=255, size=(B * B, 8, 8), device=device, dtype=dtype)
            kornia.enhance.jpeg_codec_differentiable(img, jpeg_quality, qt_y, qt_c)
        assert "Batch dimensions do not match." in str(errinfo)

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
        jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
        qt_y = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 8), device=device, dtype=dtype)
        diff_jpeg_module = kornia.enhance.JPEGCodecDifferentiable(qt_y, qt_c)
        img_jpeg = diff_jpeg_module(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_module_with_param(self, device, dtype) -> None:
        B, H, W = 4, 16, 16
        img = torch.rand(B, 3, H, W, device=device, dtype=dtype)
        jpeg_quality = torch.randint(low=0, high=100, size=(B,), device=device, dtype=dtype)
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
