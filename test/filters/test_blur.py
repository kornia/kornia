from __future__ import annotations

import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestBoxBlur:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)
        blur = kornia.filters.BoxBlur((3, 3))
        assert blur(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 6, 4, 4, device=device, dtype=dtype)
        blur = kornia.filters.BoxBlur((3, 3))
        assert blur(inp).shape == (2, 6, 4, 4)

    def test_kernel_3x3(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        kernel_size = (3, 3)
        actual = kornia.filters.box_blur(inp, kernel_size)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(actual.sum(), torch.tensor(35.0).to(actual), rtol=tol_val, atol=tol_val)

    # TODO(dmytro): normalized does not make any effect
    def test_kernel_3x3_nonormalize(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        kernel_size = (3, 3)
        actual = kornia.filters.box_blur(inp, kernel_size, normalized=False)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(actual.sum(), torch.tensor(35.0).to(actual), rtol=tol_val, atol=tol_val)

    def test_kernel_5x5(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        kernel_size = (5, 5)
        expected = inp.sum((1, 2, 3)) / torch.mul(*kernel_size)

        actual = kornia.filters.box_blur(inp, kernel_size)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(actual[:, 0, 2, 2], expected, rtol=tol_val, atol=tol_val)

    def test_kernel_5x5_batch(self, device, dtype):
        batch_size = 3
        inp = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1, 1)

        kernel_size = (5, 5)
        expected = inp.sum((1, 2, 3)) / torch.mul(*kernel_size)

        actual = kornia.filters.box_blur(inp, kernel_size)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(actual[:, 0, 2, 2], expected, rtol=tol_val, atol=tol_val)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        actual = kornia.filters.box_blur(inp, kernel_size)
        assert_close(actual, actual)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.box_blur, (img, (3, 3)), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filters.box_blur
        op_script = torch.jit.script(op)

        kernel_size = (3, 3)
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        actual = op_script(img, kernel_size)
        expected = op(img, kernel_size)
        assert_close(actual, expected)

    def test_module(self, device, dtype):
        op = kornia.filters.box_blur
        op_module = kornia.filters.BoxBlur

        kernel_size = (3, 3)
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        actual = op_module(kernel_size)(img)
        expected = op(img, kernel_size)
        assert_close(actual, expected)
