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

from kornia.filters import UnsharpMask, gaussian_blur2d, unsharp_mask

from testing.base import BaseTester, supports_reflect_padding


class Testunsharp(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, (5, 3)])
    @pytest.mark.parametrize("sigma", [(3.0, 1.0), (0.5, 0.1)])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_smoke(self, shape, kernel_size, sigma, params_as_tensor, device, dtype):
        if params_as_tensor is True:
            sigma = torch.tensor([sigma], device=device, dtype=dtype).repeat(shape[0], 1)

        data = torch.ones(shape, device=device, dtype=dtype)
        actual = unsharp_mask(data, kernel_size, sigma, "replicate")
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_cardinality(self, shape, device, dtype):
        kernel_size = (5, 7)
        sigma = (1.5, 2.1)

        data = torch.ones(shape, device=device, dtype=dtype)
        actual = unsharp_mask(data, kernel_size, sigma, "replicate")
        assert actual.shape == shape

    @pytest.mark.skip(reason="nothing to test")
    def test_exception(self): ...

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        data = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        sigma = (1.5, 2.1)
        actual = unsharp_mask(data, kernel_size, sigma, "replicate")
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        # test parameters
        shape = (1, 3, 5, 5)
        kernel_size = (3, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        data = torch.rand(shape, device=device, dtype=torch.float64)
        self.gradcheck(unsharp_mask, (data, kernel_size, sigma, "replicate"))

    def test_module(self, device, dtype):
        params = [(3, 3), (1.5, 1.5)]
        op = unsharp_mask
        op_module = UnsharpMask(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("sigma", [(3.0, 1.0), (0.5, 0.1)])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_dynamo(self, sigma, params_as_tensor, device, dtype, torch_optimizer):
        if params_as_tensor is True:
            sigma = torch.tensor([sigma], device=device, dtype=dtype)

        data = torch.ones(1, 3, 10, 10, device=device, dtype=dtype)
        op = UnsharpMask(3, sigma)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


class TestConventionsUnsharpMask(BaseTester):
    """Pins for the formula and the range of :func:`unsharp_mask`."""

    @pytest.mark.parametrize("border_type", ["reflect", "constant"])
    def test_convention_unsharp_mask_is_twice_the_image_minus_its_gaussian_blur(self, border_type, device, dtype):
        # out = 2 * x - gaussian_blur2d(x, kernel_size, sigma, border_type) = x + 1 * (x - blur): the gain is fixed at 1
        # (scikit-image's amount=1) and kernel_size, sigma and border_type are gaussian_blur2d's.
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        image = torch.rand(2, 3, 9, 13).to(device=device, dtype=dtype)
        expected = 2 * image - gaussian_blur2d(image, (3, 5), (1.0, 2.0), border_type)
        self.assert_close(unsharp_mask(image, (3, 5), (1.0, 2.0), border_type), expected)
        self.assert_close(UnsharpMask((3, 5), (1.0, 2.0), border_type)(image), expected)

    def test_convention_unsharp_mask_output_is_not_clamped(self, device, dtype):
        # The output is not clipped to the input range: a 0 | 1 step overshoots on both sides of the edge
        # (scikit-image clips by default, preserve_range=False).
        # Snippet used to generate expected:
        #   step = torch.zeros(1, 1, 7, 10); step[..., 5:] = 1; print(unsharp_mask(step, (5, 5), (1.5, 1.5))[0, 0, 3])
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        step = torch.zeros(1, 1, 7, 10, device=device, dtype=dtype)
        step[..., 5:] = 1.0
        out = unsharp_mask(step, (5, 5), (1.5, 1.5))
        expected = [0.0, 0.0, 0.0, -0.120078, -0.353959, 1.353959, 1.120078, 1.0, 1.0, 1.0]
        self.assert_close(out[0, 0, 3], torch.tensor(expected, device=device, dtype=dtype))
        assert out.min() < -0.3
        assert out.max() > 1.3
