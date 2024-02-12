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
        assert "Input imagetype is not a Tensor. Got:" in str(errinf)

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
                        [0.0, 14.0, 14.0, 0.0, 0.0, 11.0],
                        [0.0, 14.0, 14.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 34.0, 34.0, 0.0],
                        [0.0, 0.0, 0.0, 34.0, 34.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        out = kornia.contrib.connected_components(img, num_iterations=10)
        self.assert_close(out, expected)

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
