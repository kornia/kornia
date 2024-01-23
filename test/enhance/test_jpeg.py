import pytest
import torch

import kornia
from kornia.testing import BaseTester, assert_close


class TestDiffJPEG(BaseTester):
    def test_smoke(self, device, dtype) -> None:
        """This test standard usage."""
        B, H, W = 2, 32, 32
        img = torch.rand(B, 3, H, W)
        jpeg_quality = torch.randint(low=0, high=100, size=(B,))
        img_jpeg = kornia.enhance.diff_jpeg(img, jpeg_quality)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_custom_qt(self, device, dtype) -> None:
        """Here we test if we can handle custom quantization tables."""
        B, H, W = 4, 32, 32
        img = torch.rand(B, 3, H, W)
        jpeg_quality = torch.randint(low=0, high=100, size=(B,))
        qt_y = torch.randint(low=1, high=255, size=(B, 8, 8))
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 8))
        img_jpeg = kornia.enhance.diff_jpeg(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_non_batch_param(self, device, dtype) -> None:
        """Here we test if we can handle non-batched JPEG parameters (JPEG quality and QT's)."""
        B, H, W = 3, 32, 32
        img = torch.rand(B, 3, H, W)
        jpeg_quality = torch.randint(low=0, high=100, size=(1,))
        qt_y = torch.randint(low=1, high=255, size=(1, 8, 8))
        qt_c = torch.randint(low=1, high=255, size=(1, 8, 8))
        img_jpeg = kornia.enhance.diff_jpeg(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_exception(self, device, dtype) -> None:
        """Test exeptions (non-tensor input, wrong JPEG quality shape, wrong img shape, and wrong QT shape.)"""
        with pytest.raises(TypeError) as errinfo:
            B = 2
            jpeg_quality = torch.randint(low=0, high=100, size=(B,))
            kornia.enhance.diff_jpeg(1904.0, jpeg_quality)
        assert "Not a Tensor type. Got" in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            B, H, W = 2, 32, 32
            img = torch.rand(B, 3, H, W)
            kornia.enhance.diff_jpeg(img, None)
        assert "Not a Tensor type. Got" in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            B, H, W = 2, 32, 32
            img = torch.rand(B, 3, H, W)
            jpeg_quality = torch.randint(low=0, high=100, size=(B, 3, 2, 1))
            kornia.enhance.diff_jpeg(img, jpeg_quality)
        assert "shape must be [" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 2, 31, 31
            img = torch.rand(B, 3, H, W)
            jpeg_quality = torch.randint(low=0, high=100, size=(B,))
            kornia.enhance.diff_jpeg(img, jpeg_quality)
        assert "divisible" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 4, 32, 32
            img = torch.rand(B, 3, H, W)
            jpeg_quality = torch.randint(low=0, high=100, size=(B,))
            qt_y = torch.randint(low=1, high=255, size=(B, 7, 8))
            qt_c = torch.randint(low=1, high=255, size=(B, 8, 8))
            kornia.enhance.diff_jpeg(img, jpeg_quality, qt_y, qt_c)
        assert "shape must be [" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            B, H, W = 4, 32, 32
            img = torch.rand(B, 3, H, W)
            jpeg_quality = torch.randint(low=0, high=100, size=(B,))
            qt_y = torch.randint(low=1, high=255, size=(B, 8, 8))
            qt_c = torch.randint(low=1, high=255, size=(B, 8, 7))
            kornia.enhance.diff_jpeg(img, jpeg_quality, qt_y, qt_c)
        assert "shape must be [" in str(errinfo)

    def test_cardinality(self, device, dtype) -> None:
        B, H, W = 1, 16, 16
        img = torch.zeros(B, 3, H, W)
        img[..., 4:-4, 4:-4] = 1.0
        jpeg_quality = torch.tensor([2.0])
        img_jpeg = kornia.enhance.diff_jpeg(img, jpeg_quality)
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
            ], dtype=dtype
        )
        # We use a slightly higher tolerance since our implementation varies from the reference implementation
        assert_close(img_jpeg[..., :4, :4], img_jpeg_ref[..., :4, :4], rtol=0.01, atol=0.01)

    def test_module(self, device, dtype) -> None:
        B, H, W = 4, 16, 16
        img = torch.rand(B, 3, H, W)
        jpeg_quality = torch.randint(low=0, high=100, size=(B,))
        qt_y = torch.randint(low=1, high=255, size=(B, 8, 8))
        qt_c = torch.randint(low=1, high=255, size=(B, 8, 8))
        diff_jpeg_module = kornia.enhance.DiffJPEG()
        img_jpeg = diff_jpeg_module(img, jpeg_quality, qt_y, qt_c)
        assert img_jpeg is not None
        assert img_jpeg.shape == img.shape

    def test_gradcheck(self, device) -> None:
        """Not applicable since diff. JPEG is not continuous, thus PyTorch's gradcheck will always fail."""
        pass
