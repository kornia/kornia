import pytest
import torch

from kornia.filters import Canny, canny
from kornia.testing import BaseTester, tensor_to_gradcheck_var
from kornia.utils._compat import torch_version


class TestCanny(BaseTester):
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('kernel_size', [3, (5, 7)])
    @pytest.mark.parametrize('sigma', [(1.5, 1.0), (2.5, 0.5)])
    @pytest.mark.parametrize('hysteresis', [False, True])
    @pytest.mark.parametrize('low_threshold,high_threshold', [(0.1, 0.2), (0.3, 0.5)])
    def test_smoke(self, batch_size, kernel_size, sigma, hysteresis, low_threshold, high_threshold, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)

        op = Canny(low_threshold, high_threshold, kernel_size, sigma, hysteresis)
        actual = op(inp)
        assert len(actual) == 2
        assert actual[0].shape == (batch_size, 1, 4, 4)
        assert actual[1].shape == (batch_size, 1, 4, 4)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)

        op = Canny()
        magnitude, edges = op(inp)

        assert magnitude.shape == (batch_size, 1, 4, 4)
        assert edges.shape == (batch_size, 1, 4, 4)

    def test_exception(self, device, dtype):
        with pytest.raises(Exception) as errinfo:
            Canny(0.3, 0.2)
        assert 'low_threshold should be smaller than the high_threshold' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            Canny(-2, 0.3)
        assert 'Invalid low threshold.' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            Canny(0.1, 3)
        assert 'Invalid high threshold.' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            canny(1)
        assert 'Not a Tensor type' in str(errinfo)

        inp = torch.zeros(3, 4, 4, device=device, dtype=dtype)
        with pytest.raises(Exception) as errinfo:
            canny(inp)
        assert 'shape must be [[\'B\', \'C\', \'H\', \'W\']]' in str(errinfo)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(batch_size, 3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        magnitude, edges = canny(inp)

        assert magnitude.is_contiguous()
        assert edges.is_contiguous()

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

        magnitude, edges = canny(inp)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

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

        magnitude, edges = canny(inp, hysteresis=True)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

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

        magnitude, edges = canny(inp, hysteresis=False)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

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

        magnitude, edges = canny(inp, low_threshold=0.3, high_threshold=0.9)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        if "cuda" in str(device):
            pytest.skip("RuntimeError: Backward is not reentrant, i.e., running backward,")
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = tensor_to_gradcheck_var(img)
        self.gradcheck(canny, img)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = canny
        op_module = Canny()
        expected_magnitude, expected_edges = op(img)
        actual_magnitude, actual_edges = op_module(img)
        self.assert_close(actual_magnitude, expected_magnitude)
        self.assert_close(actual_edges, expected_edges)

    @pytest.mark.parametrize('kernel_size', [5, (5, 7)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.skipif(torch_version() in {'2.0.0', '2.0.1'}, reason='Not working on 2.0')
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        inpt = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = Canny(kernel_size=kernel_size)
        op_optimized = torch_optimizer(op)

        expected_magnitude, expected_edges = op(inpt)
        actual_magnitude, actual_edges = op_optimized(inpt)

        self.assert_close(actual_magnitude, expected_magnitude)
        self.assert_close(actual_edges, expected_edges)
