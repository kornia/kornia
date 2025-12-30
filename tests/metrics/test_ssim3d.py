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


class TestSSIM3d(BaseTester):
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
