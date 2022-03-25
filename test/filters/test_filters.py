import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestFilter2D:
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_smoke(self, padding, device, dtype):
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        _, height, width = kernel.shape
        input = torch.ones(1, 1, 7, 8, device=device, dtype=dtype)
        b, c, h, w = input.shape
        if padding == 'same':
            out = kornia.filters.filter2d(input, kernel, padding=padding)
            assert out.shape == (b, c, h, w)
        else:
            out = kornia.filters.filter2d(input, kernel, padding=padding)
            assert out.shape == (b, c, h - height + 1, w - width + 1)

    @pytest.mark.parametrize("batch_size", [2, 3, 6, 8])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_batch(self, batch_size, padding, device, dtype):
        B: int = batch_size
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        _, height, width = kernel.shape
        input = torch.ones(B, 3, 7, 8, device=device, dtype=dtype)
        b, c, h, w = input.shape
        if padding == 'same':
            out = kornia.filters.filter2d(input, kernel, padding=padding)
            assert out.shape == (b, c, h, w)
        else:
            out = kornia.filters.filter2d(input, kernel, padding=padding)
            assert out.shape == (b, c, h - height + 1, w - width + 1)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_mean_filter(self, padding, device, dtype):
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected_same = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 5.0, 5.0, 5.0, 0.0],
                        [0.0, 5.0, 5.0, 5.0, 0.0],
                        [0.0, 5.0, 5.0, 5.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_valid = torch.tensor(
            [[[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]]], device=device, dtype=dtype
        )

        actual = kornia.filters.filter2d(input, kernel, padding=padding)
        if padding == 'same':
            assert_close(actual, expected_same)
        else:
            assert_close(actual, expected_valid)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_mean_filter_2batch_2ch(self, padding, device, dtype):
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, -1, -1)

        expected_same = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 5.0, 5.0, 5.0, 0.0],
                        [0.0, 5.0, 5.0, 5.0, 0.0],
                        [0.0, 5.0, 5.0, 5.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, -1, -1)

        expected_valid = torch.tensor(
            [[[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]]], device=device, dtype=dtype
        ).expand(2, 2, -1, -1)

        actual = kornia.filters.filter2d(input, kernel, padding=padding)
        if padding == 'same':
            assert_close(actual, expected_same)
        else:
            assert_close(actual, expected_valid)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_normalized_mean_filter(self, padding, device, dtype):
        kernel = torch.ones(1, 3, 3).to(device)
        input = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, -1, -1)

        nv: float = 5.0 / 9  # normalization value
        expected_same = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, nv, nv, nv, 0.0],
                        [0.0, nv, nv, nv, 0.0],
                        [0.0, nv, nv, nv, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, -1, -1)

        expected_valid = torch.tensor(
            [[[[nv, nv, nv], [nv, nv, nv], [nv, nv, nv]]]], device=device, dtype=dtype
        ).expand(2, 2, -1, -1)

        actual = kornia.filters.filter2d(input, kernel, normalized=True, padding=padding)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        if padding == 'same':
            assert_close(actual, expected_same, rtol=tol_val, atol=tol_val)
        else:
            assert_close(actual, expected_valid, rtol=tol_val, atol=tol_val)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_even_sized_filter(self, padding, device, dtype):
        kernel = torch.ones(1, 2, 2, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_same = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 5.0, 5.0, 0.0, 0.0],
                        [0.0, 5.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_valid = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 5.0, 0.0], [0.0, 5.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )

        actual = kornia.filters.filter2d(input, kernel, padding=padding)
        if padding == 'same':
            assert_close(actual, expected_same)
        else:
            assert_close(actual, expected_valid)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_mix_sized_filter_padding_same(self, padding, device, dtype):
        kernel = torch.ones(1, 5, 6, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_same = torch.tensor(
            [
                [
                    [
                        [2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 0.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 0.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 0.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0, 0.0]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        actual = kornia.filters.filter2d(input, kernel, padding='same', border_type='constant')
        assert_close(actual, expected_same)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_noncontiguous(self, padding, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        kernel = torch.ones(1, 2, 2, device=device, dtype=dtype)

        actual = kornia.filters.filter2d(inp, kernel, padding=padding)
        assert_close(actual, actual)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_separable(self, padding, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 9, 9, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        kernel_x = torch.ones(1, 3, device=device, dtype=dtype)
        kernel_y = torch.ones(1, 3, device=device, dtype=dtype)
        kernel = kernel_y.t() @ kernel_x
        out = kornia.filters.filter2d(inp, kernel[None], padding=padding)
        out_sep = kornia.filters.filter2d_separable(inp, kernel_x, kernel_y, padding=padding)
        assert_close(out, out_sep)

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3, device=device)
        input = torch.ones(1, 1, 7, 8, device=device)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        kernel = utils.tensor_to_gradcheck_var(kernel)  # to var
        assert gradcheck(kornia.filters.filter2d, (input, kernel), raise_exception=True)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_jit(self, padding, device, dtype):
        op = kornia.filters.filter2d
        op_script = torch.jit.script(op)

        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        input = torch.ones(1, 1, 7, 8, device=device, dtype=dtype)
        expected = op(input, kernel, padding=padding)
        actual = op_script(input, kernel, padding=padding)
        assert_close(actual, expected)


class TestFilter3D:
    def test_smoke(self, device, dtype):
        kernel = torch.rand(1, 3, 3, 3).to(device)
        input = torch.ones(1, 1, 6, 7, 8).to(device)
        assert kornia.filters.filter3d(input, kernel).shape == input.shape

    @pytest.mark.parametrize("batch_size", [2, 3, 6, 8])
    def test_batch(self, batch_size, device, dtype):
        B: int = batch_size
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        input = torch.ones(B, 3, 6, 7, 8, device=device, dtype=dtype)
        assert kornia.filters.filter3d(input, kernel).shape == input.shape

    def test_mean_filter(self, device, dtype):
        kernel = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
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
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        actual = kornia.filters.filter3d(input, kernel)
        assert_close(actual, expected)

    def test_mean_filter_2batch_2ch(self, device, dtype):
        kernel = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        input = input.expand(2, 2, -1, -1, -1)

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected = expected.expand(2, 2, -1, -1, -1)

        actual = kornia.filters.filter3d(input, kernel)
        assert_close(actual, expected)

    def test_normalized_mean_filter(self, device, dtype):
        kernel = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        input = input.expand(2, 2, -1, -1, -1)

        nv = 5.0 / 27  # normalization value
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, nv, nv, nv, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected = expected.expand(2, 2, -1, -1, -1)

        actual = kornia.filters.filter3d(input, kernel, normalized=True)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(actual, expected, rtol=tol_val, atol=tol_val)

    def test_even_sized_filter(self, device, dtype):
        kernel = torch.ones(1, 2, 2, 2, device=device, dtype=dtype)
        input = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
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
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        actual = kornia.filters.filter3d(input, kernel)
        assert_close(actual, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1, -1)
        kernel = torch.ones(1, 2, 2, 2, device=device, dtype=dtype)

        actual = kornia.filters.filter3d(inp, kernel)
        expected = actual
        assert_close(actual, expected)

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3, 3, device=device)
        input = torch.ones(1, 1, 6, 7, 8, device=device)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        kernel = utils.tensor_to_gradcheck_var(kernel)  # to var
        assert gradcheck(kornia.filters.filter3d, (input, kernel), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filters.filter3d
        op_script = torch.jit.script(op)

        kernel = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)
        input = torch.ones(1, 1, 2, 7, 8, device=device, dtype=dtype)
        expected = op(input, kernel)
        actual = op_script(input, kernel)
        assert_close(actual, expected)
