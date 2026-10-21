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


class TestHalfPrecisionAccuracy(BaseTester):
    @pytest.fixture(autouse=True)
    def half_only(self, dtype):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision numerical regressions")

    @pytest.mark.parametrize("space", ["lab", "luv"])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_color(self, device, dtype, space, inverse):
        # Fixed representable RGB samples include black, white, and colors near neutral.
        rgb = torch.tensor(
            [[0, 1, 0.5, 0.25, 0.75], [0, 1, 0.5, 0.75, 0.25], [0, 1, 0.5, 0.5, 0.5]],
            device=device,
            dtype=dtype,
        ).reshape(3, 1, 5)
        forward = getattr(kornia.color, "rgb_to_" + space)
        op = getattr(kornia.color, space + "_to_rgb") if inverse else forward
        data = forward(rgb.float()).to(dtype) if inverse else rgb
        reference = op(data.to(device="cpu", dtype=torch.float64)).to(device=device, dtype=dtype)
        actual = op(data)
        assert actual.dtype == dtype
        self.assert_close(actual, reference, rtol=torch.finfo(dtype).eps, atol=2e-4)

    @pytest.mark.parametrize("op", [kornia.color.normals_to_rgb255, kornia.color.rgb255_to_normals])
    def test_normals(self, device, dtype, op):
        data = torch.tensor([[-0.75, 0, 0.75], [0.5, 0, -0.5], [0.25, 0, -0.25]], device=device, dtype=dtype).reshape(
            3, 1, 3
        )
        if op is kornia.color.rgb255_to_normals:
            data = ((data.float() + 1) * 127.5).to(dtype)
        reference = op(data.to(device="cpu", dtype=torch.float64)).to(device=device, dtype=dtype)
        actual = op(data)
        assert torch.isfinite(actual).all()
        self.assert_close(actual, reference, rtol=torch.finfo(dtype).eps / 2, atol=1e-5)

    @pytest.mark.parametrize("sigma", [1.3, 11.0])
    def test_gaussian(self, device, dtype, sigma):
        reference = kornia.filters.gaussian(11, sigma, mean=3.0, dtype=torch.float64).to(device=device, dtype=dtype)
        actual = kornia.filters.gaussian(11, sigma, mean=3.0, device=device, dtype=dtype)
        self.assert_close(actual, reference, rtol=0, atol=0)

    @pytest.mark.parametrize("separable", [True, False])
    def test_blur(self, device, dtype, separable):
        data = (torch.arange(3 * 8 * 8, device=device).reshape(1, 3, 8, 8) % 17).to(dtype) / 16
        reference = kornia.filters.gaussian_blur2d(
            data.to(device="cpu", dtype=torch.float64), 5, (1.25, 1.5), separable=separable
        ).to(device=device, dtype=dtype)
        actual = kornia.filters.gaussian_blur2d(data, 5, (1.25, 1.5), separable=separable)
        self.assert_close(actual, reference, rtol=0, atol=0)

    def test_canny_large_step(self, device, dtype):
        # Squaring the unnormalized Sobel gradient of a 255-level step overflows float16.
        data = torch.zeros(1, 1, 8, 8, device=device, dtype=dtype)
        data[..., 4:] = 255
        actual = kornia.filters.canny(data, kernel_size=1, low_threshold=100, high_threshold=200)
        reference = kornia.filters.canny(
            data.to(device="cpu", dtype=torch.float64), kernel_size=1, low_threshold=100, high_threshold=200
        )
        for result, expected in zip(actual, reference):
            assert torch.isfinite(result).all()
            self.assert_close(result, expected.to(device=device, dtype=dtype), rtol=0, atol=0)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_rank_deficient_zca(self, device, dtype, dim):
        data = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], device=device, dtype=dtype)
        if dim == 0:
            data = data.t()
        # Each centered sample lies on (1, -1, 1, -1); covariance has one eigenvalue of 2.
        expected = (
            data.to(device="cpu", dtype=torch.float64)
            - data.to(device="cpu", dtype=torch.float64).mean(dim, keepdim=True)
        ) / (2 + 1e-6) ** 0.5
        model = kornia.enhance.ZCAWhitening(dim=dim).fit(data)
        self.assert_close(model(data), expected.to(device=device, dtype=dtype))
        self.assert_close(kornia.enhance.zca_whiten(data, dim=dim), expected.to(device=device, dtype=dtype))
        assert model.transform_matrix.dtype == torch.float32
        assert model.mean_vector.dtype == torch.float32

    @pytest.mark.parametrize(
        "op",
        [
            kornia.filters.gaussian_blur2d,
            kornia.enhance.zca_whiten,
            kornia.color.rgb_to_lab,
            kornia.color.lab_to_rgb,
            kornia.color.rgb_to_luv,
            kornia.color.luv_to_rgb,
            kornia.filters.canny,
        ],
    )
    def test_enclosing_autocast(self, device, dtype, op):
        if device.type != "cpu":
            pytest.skip("CPU autocast regression")
        data = (torch.arange(192, device=device).reshape(4, 3, 4, 4) % 13).to(dtype) / 16
        args = (3, (1.25, 1.5)) if op is kornia.filters.gaussian_blur2d else ()
        expected = op(data, *args)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = op(data, *args)
        if isinstance(actual, tuple):
            for result, reference in zip(actual, expected):
                self.assert_close(result, reference, rtol=0, atol=0)
        else:
            self.assert_close(actual, expected, rtol=0, atol=0)

    def test_inverse_enclosing_autocast(self, device, dtype):
        if device.type != "cpu":
            pytest.skip("CPU autocast regression")
        data = torch.tensor([[0, 2], [1, 3], [-1, 4], [0.5, 1]], device=device, dtype=dtype)
        model = kornia.enhance.ZCAWhitening(compute_inv=True).fit(data)
        whitened = model(data)
        expected = model.inverse_transform(whitened)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.inverse_transform(whitened)
        self.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("name", ["rgb_to_lab", "lab_to_rgb", "rgb_to_luv", "luv_to_rgb", "blur", "zca"])
    def test_backward(self, device, dtype, name):
        data = ((torch.arange(3 * 6 * 6, device=device).reshape(1, 3, 6, 6) % 17).to(dtype) / 16).requires_grad_()
        if name == "blur":
            op = kornia.filters.GaussianBlur2d(3, (1.25, 1.5))
        elif name == "zca":
            # Distinct covariance eigenvalues keep the SVD derivative well-defined.
            data = torch.tensor([[0, 2], [1, 3], [-1, 4], [0.5, 1]], device=device, dtype=dtype, requires_grad=True)
            op = kornia.enhance.zca_whiten
        else:
            op = getattr(kornia.color, name)
        wide = data.detach().float().requires_grad_()
        gradient = torch.autograd.grad(op(data).sum(), data)[0]
        reference = torch.autograd.grad(op(wide).sum(), wide)[0]
        assert torch.isfinite(gradient).all()
        assert gradient.dtype == dtype
        self.assert_close(gradient, reference.to(dtype))

    @pytest.mark.parametrize("name", ["rgb_to_lab", "lab_to_rgb", "rgb_to_luv", "luv_to_rgb", "blur", "zca"])
    def test_dynamo(self, device, dtype, name, torch_optimizer):
        data = (torch.arange(4 * 3 * 6 * 6, device=device).reshape(4, 3, 6, 6) % 17).to(dtype) / 16
        if name == "blur":
            op = kornia.filters.GaussianBlur2d(3, (1.25, 1.5))
        elif name == "zca":
            data = data[:, :, :1, :1]
            op = kornia.enhance.zca_whiten
        else:
            op = getattr(kornia.color, name)
        self.assert_close(torch_optimizer(op)(data), op(data))
