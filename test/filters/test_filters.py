import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestFilter2D:
    def test_smoke(self, device, dtype):
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        input = torch.ones(1, 1, 7, 8, device=device, dtype=dtype)
        assert kornia.filter2d(input, kernel).shape == input.shape

    @pytest.mark.parametrize("batch_size", [2, 3, 6, 8])
    def test_batch(self, batch_size, device, dtype):
        B: int = batch_size
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        input = torch.ones(B, 3, 7, 8, device=device, dtype=dtype)
        assert kornia.filter2d(input, kernel).shape == input.shape

    def test_mean_filter(self, device, dtype):
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
        expected = torch.tensor(
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

        actual = kornia.filter2d(input, kernel)
        assert_close(actual, expected)

    def test_mean_filter_2batch_2ch(self, device, dtype):
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

        expected = torch.tensor(
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

        actual = kornia.filter2d(input, kernel)
        assert_close(actual, expected)

    def test_normalized_mean_filter(self, device, dtype):
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
        expected = torch.tensor(
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

        actual = kornia.filter2d(input, kernel, normalized=True)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(actual, expected, rtol=tol_val, atol=tol_val)

    def test_even_sized_filter(self, device, dtype):
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

        expected = torch.tensor(
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

        actual = kornia.filter2d(input, kernel)
        assert_close(actual, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        kernel = torch.ones(1, 2, 2, device=device, dtype=dtype)

        actual = kornia.filter2d(inp, kernel)
        expected = actual
        assert_close(actual, actual)

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3, device=device)
        input = torch.ones(1, 1, 7, 8, device=device)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        kernel = utils.tensor_to_gradcheck_var(kernel)  # to var
        assert gradcheck(kornia.filter2d, (input, kernel), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filter2d
        op_script = torch.jit.script(op)

        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        input = torch.ones(1, 1, 7, 8, device=device, dtype=dtype)
        expected = op(input, kernel)
        actual = op_script(input, kernel)
        assert_close(actual, expected)


class TestFilter3D:
    def test_smoke(self, device, dtype):
        kernel = torch.rand(1, 3, 3, 3).to(device)
        input = torch.ones(1, 1, 6, 7, 8).to(device)
        assert kornia.filter3d(input, kernel).shape == input.shape

    @pytest.mark.parametrize("batch_size", [2, 3, 6, 8])
    def test_batch(self, batch_size, device, dtype):
        B: int = batch_size
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        input = torch.ones(B, 3, 6, 7, 8, device=device, dtype=dtype)
        assert kornia.filter3d(input, kernel).shape == input.shape

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

        actual = kornia.filter3d(input, kernel)
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

        actual = kornia.filter3d(input, kernel)
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

        actual = kornia.filter3d(input, kernel, normalized=True)

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

        actual = kornia.filter3d(input, kernel)
        assert_close(actual, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1, -1)
        kernel = torch.ones(1, 2, 2, 2, device=device, dtype=dtype)

        actual = kornia.filter3d(inp, kernel)
        expected = actual
        assert_close(actual, expected)

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3, 3, device=device)
        input = torch.ones(1, 1, 6, 7, 8, device=device)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        kernel = utils.tensor_to_gradcheck_var(kernel)  # to var
        assert gradcheck(kornia.filter3d, (input, kernel), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.filter3d
        op_script = torch.jit.script(op)

        kernel = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)
        input = torch.ones(1, 1, 2, 7, 8, device=device, dtype=dtype)
        expected = op(input, kernel)
        actual = op_script(input, kernel)
        assert_close(actual, expected)
