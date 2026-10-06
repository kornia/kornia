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
import torch.nn.functional as F

import kornia

from testing.base import BaseTester


class TestSSIM3d(BaseTester):
    @pytest.mark.parametrize("image_dtype", [torch.uint8, torch.int16])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_integer_images(self, device, image_dtype, padding):
        # Integer Gaussian weights must not truncate to zero (#5297).
        black = torch.zeros((1, 1, 5, 5, 5), device=device, dtype=image_dtype)
        white = torch.full_like(black, 255)
        actual = kornia.metrics.ssim3d(black, white, 3, max_val=255.0, padding=padding)
        assert actual.dtype == torch.float32
        self.assert_close(actual, torch.full_like(actual, 0.0001), atol=1e-6, rtol=1e-4)

    @pytest.mark.parametrize("image_dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_integer_images_match_the_float32_result(self, device, image_dtype, padding):
        # A textured pair exercises the variances and the covariance, which a pair of constant images cancels.
        # The 12-bit values of the wider types are not all exact in float16.
        max_val = 255.0 if image_dtype == torch.uint8 else 4095.0
        generator = torch.Generator().manual_seed(0)
        img1 = (torch.rand((1, 2, 8, 8, 8), generator=generator) * max_val).round()
        img2 = (img1 + torch.randn((1, 2, 8, 8, 8), generator=generator) * max_val / 6).clamp(0, max_val).round()
        img1, img2 = img1.to(device), img2.to(device)
        expected = kornia.metrics.ssim3d(img1, img2, 5, max_val=max_val, padding=padding)
        actual = kornia.metrics.ssim3d(img1.to(image_dtype), img2.to(image_dtype), 5, max_val=max_val, padding=padding)
        assert actual.dtype == torch.float32
        assert expected.mean() < 0.95
        self.assert_close(actual, expected, rtol=0, atol=0)

    def test_dynamo_dynamic_range(self, device, dtype, torch_optimizer):
        img = torch.full((1, 1) + (8,) * 3, 128.0, device=device, dtype=dtype)
        optimized = torch_optimizer(kornia.metrics.ssim3d)
        actual = optimized(img, img, 3, max_val=255.0)
        assert actual.dtype == dtype
        self.assert_close(actual, torch.ones_like(actual))

    def test_mixed_input_dtypes(self, device, dtype):
        first = torch.full((1, 1) + (8,) * 3, 128.0, device=device, dtype=dtype)
        second = torch.full_like(first, 64.0, dtype=torch.float32)
        actual = kornia.metrics.ssim3d(first, second, 3, max_val=255.0)
        output_dtype = torch.promote_types(dtype, torch.float32)
        assert actual.dtype == output_dtype
        c1 = (0.01 * 255.0) ** 2
        expected = torch.full_like(actual, (2 * 128 * 64 + c1) / (128**2 + 64**2 + c1))
        self.assert_close(actual, expected)

    def test_pixel_range_gradients(self, device, dtype):
        if device.type == "mps":
            pytest.skip("Float64 reference is unsupported on MPS")
        img1 = torch.arange(5**3, device=device, dtype=dtype).reshape((1, 1) + (5,) * 3) % 17 * 15
        img2 = (img1.flip(-1) * 0.8).detach().requires_grad_()
        img1 = img1.detach().requires_grad_()
        ref1, ref2 = img1.detach().double().requires_grad_(), img2.detach().double().requires_grad_()
        actual = kornia.metrics.ssim3d(img1, img2, 3, max_val=255.0)
        reference = kornia.metrics.ssim3d(ref1, ref2, 3, max_val=255.0)
        self.assert_close(actual, reference.to(dtype))
        actual_grads = torch.autograd.grad(actual.sum(), (img1, img2))
        reference_grads = torch.autograd.grad(reference.sum(), (ref1, ref2))
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            assert torch.isfinite(actual_grad).all()
            self.assert_close(actual_grad, reference_grad.to(dtype), rtol=2e-2, atol=1e-5)

    def test_autocast_dynamic_range(self, device, dtype):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Autocast regression covers CPU and CUDA backends")
        img1 = torch.arange(8**3, device=device, dtype=dtype).reshape((1, 1) + (8,) * 3) % 17 * 15
        img2 = img1.flip(-1) * 0.8
        expected = kornia.metrics.ssim3d(img1.double(), img2.double(), 3, max_val=255.0).to(dtype)
        autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            actual = kornia.metrics.ssim3d(img1, img2, 3, max_val=255.0)
        assert actual.dtype == dtype
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("values", [(128.0, 128.0, 255.0), (128.0, 64.0, 255.0), (0.0, 0.0, 0.5)])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_constant_image_dynamic_range(self, device, dtype, values, padding):
        first, second, max_val = values
        shape = (1, 2) + (8,) * 3
        img1 = torch.full(shape, first, device=device, dtype=dtype)
        img2 = torch.full(shape, second, device=device, dtype=dtype)
        actual = kornia.metrics.ssim3d(img1, img2, 3, max_val=max_val, padding=padding)
        # Constant images have zero local variance/covariance in the SSIM formula.
        c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
        expected_value = (2 * first * second + c1) * c2 / ((first**2 + second**2 + c1) * c2 + 1e-12)
        expected = torch.full_like(actual, expected_value)
        assert actual.dtype == dtype
        self.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "shape,padding,window_size,max_value",
        [
            ((1, 1, 3, 3, 3), "same", 5, 1.0),
            ((1, 1, 3, 3, 3), "same", 3, 2.0),
            ((1, 1, 3, 3, 3), "same", 3, 0.5),
            ((1, 1, 3, 3, 3), "valid", 3, 1.0),
            ((2, 4, 3, 3, 3), "same", 3, 1.0),
        ],
    )
    def test_smoke(self, shape, padding, window_size, max_value, device, dtype):
        img_a = (torch.ones(shape, device=device, dtype=dtype) * max_value).clamp(0.0, max_value)
        img_b = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.metrics.ssim3d(img_a, img_b, window_size, max_value, padding=padding)
        expected = torch.ones_like(actual, device=device, dtype=dtype)

        self.assert_close(actual, expected * 0.0001)

        actual = kornia.metrics.ssim3d(img_a, img_a, window_size, max_value, padding=padding)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "shape,padding,window_size,expected",
        [
            ((1, 1, 2, 2, 3), "same", 3, (1, 1, 2, 2, 3)),
            ((1, 1, 3, 3, 3), "same", 5, (1, 1, 3, 3, 3)),
            ((1, 1, 3, 3, 3), "valid", 3, (1, 1, 1, 1, 1)),
            ((2, 4, 3, 3, 3), "same", 3, (2, 4, 3, 3, 3)),
        ],
    )
    def test_cardinality(self, shape, padding, window_size, expected, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.metrics.ssim3d(img, img, window_size, padding=padding)

        assert actual.shape == expected

    def test_exception(self, device, dtype):
        img = torch.rand(1, 1, 3, 3, 3, device=device, dtype=dtype)

        # Check if both are tensors
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.metrics.ssim3d(1.0, img, 3)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.metrics.ssim3d(img, 1.0, 3)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        # Check both shapes
        from kornia.core.exceptions import ShapeError

        img_wrong_shape = torch.rand(3, 3, device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            kornia.metrics.ssim3d(img, img_wrong_shape, 3)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            kornia.metrics.ssim3d(img_wrong_shape, img, 3)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        # Check if same shape
        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(Exception) as errinfo:
            kornia.metrics.ssim3d(img, img_b, 3)
        assert "img1 and img2 shapes must be the same. Got:" in str(errinfo)

    def test_unit(self, device, dtype):
        img_a = torch.tensor(
            [
                [
                    [
                        [[0.7, 1.0, 0.5], [1.0, 0.3, 1.0], [0.2, 1.0, 0.1]],
                        [[0.2, 1.0, 0.1], [1.0, 0.3, 1.0], [0.7, 1.0, 0.5]],
                        [[1.0, 0.3, 1.0], [0.7, 1.0, 0.5], [0.2, 1.0, 0.1]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        img_b = torch.ones(1, 1, 3, 3, 3, device=device, dtype=dtype) * 0.5

        actual = kornia.metrics.ssim3d(img_a, img_b, 3, padding="same")

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0093, 0.0080, 0.0075], [0.0075, 0.0068, 0.0063], [0.0067, 0.0060, 0.0056]],
                        [[0.0077, 0.0070, 0.0065], [0.0077, 0.0069, 0.0064], [0.0075, 0.0066, 0.0062]],
                        [[0.0075, 0.0069, 0.0064], [0.0078, 0.0070, 0.0065], [0.0077, 0.0067, 0.0064]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "shape,padding,window_size,max_value",
        [
            ((1, 1, 3, 3, 3), "same", 5, 1.0),
            ((1, 1, 3, 3, 3), "same", 3, 2.0),
            ((1, 1, 3, 3, 3), "same", 3, 0.5),
            ((1, 1, 3, 3, 3), "valid", 3, 1.0),
        ],
    )
    def test_module(self, shape, padding, window_size, max_value, device, dtype):
        img_a = torch.rand(shape, device=device, dtype=dtype).clamp(0.0, max_value)
        img_b = torch.rand(shape, device=device, dtype=dtype).clamp(0.0, max_value)

        ops = kornia.metrics.ssim3d
        mod = kornia.metrics.SSIM3D(window_size, max_value, padding=padding)

        ops_out = ops(img_a, img_b, window_size, max_value, padding=padding)
        mod_out = mod(img_a, img_b)

        self.assert_close(ops_out, mod_out)

    def test_gradcheck(self, device):
        img = torch.rand(1, 1, 3, 3, 3, device=device, dtype=torch.float64)

        op = kornia.metrics.ssim3d

        self.gradcheck(op, (img, img, 3), nondet_tol=1e-8)


class TestConventionsSSIM3D(BaseTester):
    """Pins for the map shape of :func:`ssim3d` and its border, window and ``padding`` warts."""

    @staticmethod
    def _pair(device, dtype):
        # a two-sample, two-channel batch with D != H != W; y is a noisy copy of x
        g = torch.Generator().manual_seed(0)
        x = torch.rand(2, 2, 6, 7, 9, generator=g)
        y = (x + 0.3 * torch.randn(2, 2, 6, 7, 9, generator=g)).clamp(0, 1)
        return x.to(device, dtype), y.to(device, dtype)

    def test_convention_ssim3d_map_shape_and_valid_crop(self, device, dtype):
        # ssim3d scores every voxel of (B, C, D, H, W); 'valid' crops (window_size - 1) / 2 voxels from both ends of
        # each of D, H and W, and equals the 'same' map with that border cut off.
        x, y = self._pair(device, dtype)
        same = kornia.metrics.ssim3d(x, y, 5)
        valid = kornia.metrics.ssim3d(x, y, 5, padding="valid")
        assert same.shape == (2, 2, 6, 7, 9)
        assert valid.shape == (2, 2, 2, 3, 5)
        self.assert_close(valid, same[..., 2:-2, 2:-2, 2:-2])
        # Relabelling check: swapping D and W in both volumes swaps them in the map (the window is isotropic).
        swapped = kornia.metrics.ssim3d(x.transpose(-3, -1), y.transpose(-3, -1), 5)
        self.assert_close(swapped, same.transpose(-3, -1))

    def test_wart_ssim3d_same_border_is_replicate_5534(self, device, dtype):
        """The 'same' border of ssim3d is replicate, where ssim reflects (#5534)."""
        # The 'same' map equals the 'valid' map of the replicate-padded volumes and differs from that of the
        # reflect-padded ones at the border.
        x, y = self._pair(device, dtype)
        work = torch.promote_types(dtype, torch.float32)  # ssim3d computes half-precision volumes in float32

        def padded(mode):
            xp = F.pad(x.to(work), (2, 2, 2, 2, 2, 2), mode=mode)
            yp = F.pad(y.to(work), (2, 2, 2, 2, 2, 2), mode=mode)
            return kornia.metrics.ssim3d(xp, yp, 5, padding="valid").to(dtype)

        same = kornia.metrics.ssim3d(x, y, 5)
        self.assert_close(same, padded("replicate"))
        assert (same - padded("reflect")).abs().max() > 0.1

    def test_wart_ssim3d_window_is_built_in_float32_5534(self, device, dtype):
        """ssim3d builds its Gaussian window in float32 whatever the input dtype (#5534)."""
        # A float64 ssim3d therefore equals a float64 SSIM computed with the float32-rounded window, and misses the one
        # computed with a float64 window by far more than float64 roundoff. The comparison runs on the 'valid' voxels.
        if dtype != torch.float64:
            pytest.skip("the float32 rounding of the window shows only in a float64 result")
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1, 1, 6, 7, 9, generator=g, dtype=torch.float64).to(device)
        y = torch.rand(1, 1, 6, 7, 9, generator=g, dtype=torch.float64).to(device)

        def reference(window_dtype):
            window = kornia.filters.get_gaussian_kernel3d((5, 5, 5), (1.5, 1.5, 1.5), dtype=window_dtype)
            window = window.to(device, torch.float64)[None]

            def blur(t):
                return F.conv3d(t, window)

            mu_x, mu_y = blur(x), blur(y)
            var_x, var_y = blur(x * x) - mu_x**2, blur(y * y) - mu_y**2
            cov = blur(x * y) - mu_x * mu_y
            c1, c2 = 0.01**2, 0.03**2
            return (2 * mu_x * mu_y + c1) * (2 * cov + c2) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2) + 1e-12)

        actual = kornia.metrics.ssim3d(x, y, 5, padding="valid")
        assert (actual - reference(torch.float32)).abs().max() < 1e-13
        assert (actual - reference(torch.float64)).abs().max() > 1e-9

    def test_wart_ssim3d_padding_is_not_validated_5537(self, device, dtype):
        """Every padding other than 'valid', including 'VALID' and 'bogus', silently gives the 'same' map (#5537)."""
        x, y = self._pair(device, dtype)
        same = kornia.metrics.ssim3d(x, y, 5)
        for padding in ("VALID", "bogus"):
            out = kornia.metrics.ssim3d(x, y, 5, padding=padding)
            assert out.shape == same.shape
            self.assert_close(out, same, rtol=0.0, atol=0.0)
