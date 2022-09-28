from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestCanny:
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)

        canny = kornia.filters.Canny()
        magnitude, edges = canny(inp)

        assert magnitude.shape == (1, 1, 4, 4)
        assert edges.shape == (1, 1, 4, 4)

    def test_shape_batch(self, device, dtype):
        batch_size = 2
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)

        canny = kornia.filters.Canny()
        magnitude, edges = canny(inp)

        assert magnitude.shape == (batch_size, 1, 4, 4)
        assert edges.shape == (batch_size, 1, 4, 4)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        magnitude, edges = kornia.filters.canny(inp)

        assert inp.is_contiguous() is False

        assert magnitude.is_contiguous()
        assert edges.is_contiguous()

        assert magnitude.shape == (batch_size, 1, 5, 5)
        assert edges.shape == (batch_size, 1, 5, 5)

    def test_magnitude(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.2458, 0.9672, 1.2458, 0.0],
                        [0.0, 0.9672, 0.0, 0.9672, 0.0],
                        [0.0, 1.2458, 0.9672, 1.2458, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = kornia.filters.canny(inp)

        tol_val: float = utils._get_precision(device, dtype)
        assert_close(magnitude, expected_magnitude, rtol=tol_val, atol=tol_val)
        assert_close(edges, expected_edges, rtol=tol_val, atol=tol_val)

    def test_magnitude_hyst(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.5, 0.4, 0.5, 0.45, 0.1],
                        [0.3, 0.2, 0.3, 0.0, 0.3],
                        [0.5, 1.0, 1.0, 0.6, 0.75],
                        [0.2, 0.4, 0.6, 0.0, 0.5],
                        [0.1, 0.35, 0.35, 0.26, 0.1],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.4858, 0.5594, 0.6878, 0.6977, 0.5602],
                        [0.1129, 0.0000, 0.0000, 0.4531, 0.0000],
                        [0.6115, 0.5859, 0.6110, 0.6766, 0.5160],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [1.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = kornia.filters.canny(inp, hysteresis=True)

        tol_val: float = utils._get_precision(device, dtype)
        assert_close(magnitude, expected_magnitude, rtol=tol_val, atol=tol_val)
        assert_close(edges, expected_edges, rtol=tol_val, atol=tol_val)

    def test_magnitude_hyst_false(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.5, 0.4, 0.5, 0.45, 0.1],
                        [0.3, 0.2, 0.3, 0.0, 0.3],
                        [0.5, 1.0, 1.0, 0.6, 0.75],
                        [0.2, 0.4, 0.6, 0.0, 0.5],
                        [0.1, 0.35, 0.35, 0.26, 0.1],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.4858, 0.5594, 0.6878, 0.6977, 0.5602],
                        [0.1129, 0.0000, 0.0000, 0.4531, 0.0000],
                        [0.6115, 0.5859, 0.6110, 0.6766, 0.5160],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [0.5000, 0.0000, 0.0000, 1.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = kornia.filters.canny(inp, hysteresis=False)

        tol_val: float = utils._get_precision(device, dtype)
        assert_close(magnitude, expected_magnitude, rtol=tol_val, atol=tol_val)
        assert_close(edges, expected_edges, rtol=tol_val, atol=tol_val)

    def test_magnitude_threshold(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.5, 0.4, 0.5, 0.45, 0.1],
                        [0.3, 0.2, 0.3, 0.0, 0.3],
                        [0.5, 1.0, 1.0, 0.6, 0.75],
                        [0.2, 0.4, 0.6, 0.0, 0.5],
                        [0.1, 0.35, 0.35, 0.26, 0.1],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.4858, 0.5594, 0.6878, 0.6977, 0.5602],
                        [0.1129, 0.0000, 0.0000, 0.4531, 0.0000],
                        [0.6115, 0.5859, 0.6110, 0.6766, 0.5160],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = kornia.filters.canny(inp, low_threshold=0.3, high_threshold=0.9)

        tol_val: float = utils._get_precision(device, dtype)
        assert_close(magnitude, expected_magnitude, rtol=tol_val, atol=tol_val)
        assert_close(edges, expected_edges, rtol=tol_val, atol=tol_val)

    def test_gradcheck(self, device, dtype):
        if "cuda" in str(device):
            pytest.skip("RuntimeError: Backward is not reentrant, i.e., running backward,")
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.canny, img, raise_exception=True)

    def test_jit(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.sobel
        op_script = torch.jit.script(op)
        expected_magnitude, expected_edges = op(img)
        actual_magnitude, actual_edges = op_script(img)
        assert_close(actual_magnitude, expected_magnitude)
        assert_close(actual_edges, expected_edges)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = kornia.filters.canny
        op_module = kornia.filters.Canny()
        expected_magnitude, expected_edges = op(img)
        actual_magnitude, actual_edges = op_module(img)
        assert_close(actual_magnitude, expected_magnitude)
        assert_close(actual_edges, expected_edges)
