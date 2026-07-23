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


class TestConnectedComponents(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == (1, 1, 3, 4)

    @pytest.mark.parametrize("shape", [(1, 3, 4), (2, 1, 3, 4)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == shape

    def test_exception(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)

        with pytest.raises(TypeError) as errinf:
            assert kornia.contrib.connected_components(img, 1.0)
        assert "Input num_iterations must be a positive integer." in str(errinf)

        with pytest.raises(TypeError) as errinf:
            assert kornia.contrib.connected_components("not a tensor", 0)
        assert "Input imagetype is not a torch.Tensor" in str(errinf)

        with pytest.raises(TypeError) as errinf:
            assert kornia.contrib.connected_components(img, 0)
        assert "Input num_iterations must be a positive integer." in str(errinf)

        with pytest.raises(ValueError) as errinf:
            img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
            assert kornia.contrib.connected_components(img, 2)
        assert "Input image shape must be (*,1,H,W). Got:" in str(errinf)

    def test_value(self, device, dtype):
        img = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 15.0, 15.0, 0.0, 0.0, 12.0],
                        [0.0, 15.0, 15.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 35.0, 35.0, 0.0],
                        [0.0, 0.0, 0.0, 35.0, 35.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        out = kornia.contrib.connected_components(img, num_iterations=10)
        self.assert_close(out, expected)
        
    def test_float16_precision_no_collision(self, device):
        """Regression test for a silent correctness bug: label IDs used to be
        generated using the input image's own dtype via torch.arange(...,
        dtype=image.dtype). float16 can only represent integers exactly up to
        2**11 = 2048, so on any image larger than that, two genuinely
        disconnected components could receive colliding IDs and end up with
        the same final label, with no error or warning."""
        W, H = 8, 3000  # H * W = 24000 pixels, well past float16's exact-integer limit
        img = torch.zeros(1, 1, H, W, device=device)
        img[0, 0, 2051, 0] = 1.0  # component A
        img[0, 0, 2053, 0] = 1.0  # component B, 2 rows away -> not connected

        out = kornia.contrib.connected_components(img.to(torch.float16), num_iterations=150)

        label_a = out[0, 0, 2051, 0]
        label_b = out[0, 0, 2053, 0]
        assert label_a != label_b, "two disconnected components must not share a label"

    def test_gradcheck(self, device):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.contrib.connected_components, (img,))

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.contrib.connected_components
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))


def test_compute_padding():
    assert kornia.contrib.compute_padding((6, 6), (2, 2)) == (0, 0, 0, 0)
    assert kornia.contrib.compute_padding((7, 7), (2, 2)) == (0, 1, 0, 1)
    assert kornia.contrib.compute_padding((8, 7), (4, 4)) == (0, 0, 0, 1)
