import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestMedianBlur:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)
        assert kornia.filters.median_blur(inp, (3, 3)).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 6, 4, 4, device=device, dtype=dtype)
        assert kornia.filters.median_blur(inp, (3, 3)).shape == (2, 6, 4, 4)

    def test_kernel_3x3(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 7.0, 5.0, 0.0],
                    [0.0, 3.0, 1.0, 1.0, 0.0],
                    [0.0, 6.0, 9.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [36.0, 7.0, 25.0, 0.0, 0.0],
                    [3.0, 14.0, 1.0, 0.0, 0.0],
                    [65.0, 59.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        kernel_size = (3, 3)
        actual = kornia.filters.median_blur(inp, kernel_size)
        assert_close(actual[0, 0, 2, 2], torch.tensor(3.0).to(actual))
        assert_close(actual[0, 1, 1, 1], torch.tensor(14.0).to(actual))

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        actual = kornia.filters.median_blur(inp, kernel_size)
        assert_close(actual, actual)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.median_blur, (img, (5, 3)), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        kernel_size = (3, 5)
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.median_blur
        op_script = torch.jit.script(op)
        actual = op_script(img, kernel_size)
        expected = op(img, kernel_size)
        assert_close(actual, expected)

    def test_module(self, device, dtype):
        kernel_size = (3, 5)
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.median_blur
        op_module = kornia.filters.MedianBlur((3, 5))
        actual = op_module(img)
        expected = op(img, kernel_size)
        assert_close(actual, expected)
