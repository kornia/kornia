import pytest
import torch

from kornia.filters import Sobel, SpatialGradient, SpatialGradient3d, sobel, spatial_gradient, spatial_gradient3d
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestSpatialGradient(BaseTester):
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('mode', ['sobel', 'diff'])
    @pytest.mark.parametrize('order', [1, 2])
    @pytest.mark.parametrize('normalized', [True, False])
    def test_smoke(self, batch_size, mode, order, normalized, device, dtype):
        inpt = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)
        actual = SpatialGradient(mode, order, normalized)(inpt)
        assert isinstance(actual, torch.Tensor)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)
        assert SpatialGradient()(inp).shape == (batch_size, 3, 2, 4, 4)

    def test_exception(self):
        with pytest.raises(TypeError) as errinfo:
            spatial_gradient(1)
        assert 'Not a Tensor type' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            spatial_gradient(torch.zeros(1))
        assert 'shape must be [[\'B\', \'C\', \'H\', \'W\']]' in str(errinfo)

    def test_edges(self, device, dtype):
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

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 1.0, 0.0, -1.0, 0.0],
                            [1.0, 3.0, 0.0, -3.0, -1.0],
                            [2.0, 4.0, 0.0, -4.0, -2.0],
                            [1.0, 3.0, 0.0, -3.0, -1.0],
                            [0.0, 1.0, 0.0, -1.0, 0.0],
                        ],
                        [
                            [0.0, 1.0, 2.0, 1.0, 0.0],
                            [1.0, 3.0, 4.0, 3.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0],
                            [-1.0, -3.0, -4.0, -3.0, -1],
                            [0.0, -1.0, -2.0, -1.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        edges = spatial_gradient(inp, normalized=False)
        self.assert_close(edges, expected)

    def test_edges_norm(self, device, dtype):
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

        expected = (
            torch.tensor(
                [
                    [
                        [
                            [
                                [0.0, 1.0, 0.0, -1.0, 0.0],
                                [1.0, 3.0, 0.0, -3.0, -1.0],
                                [2.0, 4.0, 0.0, -4.0, -2.0],
                                [1.0, 3.0, 0.0, -3.0, -1.0],
                                [0.0, 1.0, 0.0, -1.0, 0.0],
                            ],
                            [
                                [0.0, 1.0, 2.0, 1.0, 0.0],
                                [1.0, 3.0, 4.0, 3.0, 1.0],
                                [0.0, 0.0, 0.0, 0.0, 0],
                                [-1.0, -3.0, -4.0, -3.0, -1],
                                [0.0, -1.0, -2.0, -1.0, 0.0],
                            ],
                        ]
                    ]
                ],
                device=device,
                dtype=dtype,
            )
            / 8.0
        )

        edges = spatial_gradient(inp, normalized=True)
        self.assert_close(edges, expected)

    def test_edges_sep(self, device, dtype):
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

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, -1.0, 0.0],
                            [1.0, 1.0, 0.0, -1.0, -1.0],
                            [0.0, 1.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, -1.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        edges = spatial_gradient(inp, 'diff', normalized=False)
        self.assert_close(edges, expected)

    def test_edges_sep_norm(self, device, dtype):
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

        expected = (
            torch.tensor(
                [
                    [
                        [
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, -1.0, 0.0],
                                [1.0, 1.0, 0.0, -1.0, -1.0],
                                [0.0, 1.0, 0.0, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ],
                            [
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 1.0, 1.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, -1.0, -1.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0, 0.0, 0.0],
                            ],
                        ]
                    ]
                ],
                device=device,
                dtype=dtype,
            )
            / 2.0
        )

        edges = spatial_gradient(inp, 'diff', normalized=True)
        self.assert_close(edges, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = spatial_gradient(inp)

        assert inp.is_contiguous() is False
        assert actual.is_contiguous()
        assert actual.shape == (3, 3, 2, 5, 5)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(spatial_gradient, (img,))

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = spatial_gradient
        op_module = SpatialGradient()
        expected = op(img)
        actual = op_module(img)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize('mode', ['sobel', 'diff'])
    @pytest.mark.parametrize('order', [1, 2])
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_dynamo(self, batch_size, order, mode, device, dtype, torch_optimizer):
        inpt = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        if order == 1 and dtype == torch.float64:
            # TODO: FIX order 1 spatial gradient with fp64 on dynamo
            pytest.xfail(reason='Order 1 on spatial gradient may be wrong computed for float64 on dynamo')
        op = SpatialGradient(mode, order)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))


class TestSpatialGradient3d(BaseTester):
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('mode', ['diff'])  # TODO: add support to 'sobel'
    @pytest.mark.parametrize('order', [1, 2])
    def test_smoke(self, batch_size, mode, order, device, dtype):
        inpt = torch.ones(batch_size, 3, 2, 7, 4, device=device, dtype=dtype)
        actual = SpatialGradient3d(mode, order)(inpt)
        assert isinstance(actual, torch.Tensor)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 2, 4, 5, 6, device=device, dtype=dtype)
        sobel = SpatialGradient3d()
        assert sobel(inp).shape == (batch_size, 2, 3, 4, 5, 6)

    def test_exception(self):
        with pytest.raises(TypeError) as errinfo:
            spatial_gradient3d(1)
        assert 'Not a Tensor type' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            spatial_gradient3d(torch.zeros(1))
        assert 'shape must be [[\'B\', \'C\', \'D\', \'H\', \'W\']]' in str(errinfo)

    def test_edges(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
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
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                                [0.5000, 0.5000, 0.0000, -0.5000, -0.5000],
                                [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.5000, 0.0000, -0.5000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                        ],
                        [
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                            [
                                [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                                [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, -0.5000, -0.5000, -0.5000, 0.0000],
                                [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                            ],
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                        ],
                        [
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                                [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
                                [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                            [
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                                [0.0000, -0.5000, 0.0000, -0.5000, 0.0000],
                                [0.0000, 0.0000, -0.5000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            ],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        edges = spatial_gradient3d(inp)
        self.assert_close(edges, expected)

    def test_gradcheck(self, device):
        img = torch.rand(1, 1, 1, 3, 4, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        fast_mode = 'cpu' in str(device)  # disable fast mode on gpu
        self.gradcheck(spatial_gradient3d, (img,), fast_mode=fast_mode)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 1, 4, 5, device=device, dtype=dtype)
        op = spatial_gradient3d
        op_module = SpatialGradient3d()
        expected = op(img)
        actual = op_module(img)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize('mode', ['diff'])
    @pytest.mark.parametrize('order', [1, 2])
    def test_dynamo(self, mode, order, device, dtype, torch_optimizer):
        inpt = torch.ones(1, 3, 1, 10, 10, device=device, dtype=dtype)
        op = SpatialGradient3d(mode, order)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))


class TestSobel(BaseTester):
    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('normalized', [True, False])
    def test_smoke(self, batch_size, normalized, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 7, device=device, dtype=dtype)
        actual = Sobel()(inp)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == (batch_size, 3, 4, 7)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 7, device=device, dtype=dtype)
        assert Sobel()(inp).shape == (batch_size, 3, 4, 7)

    def test_exception(self):
        with pytest.raises(TypeError) as errinfo:
            sobel(1)
        assert 'Not a Tensor type' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            sobel(torch.zeros(1))
        assert 'shape must be [[\'B\', \'C\', \'H\', \'W\']]' in str(errinfo)

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

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 1.4142, 2.0, 1.4142, 0.0],
                        [1.4142, 4.2426, 4.00, 4.2426, 1.4142],
                        [2.0, 4.0000, 0.00, 4.0000, 2.0],
                        [1.4142, 4.2426, 4.00, 4.2426, 1.4142],
                        [0.0, 1.4142, 2.0, 1.4142, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        edges = sobel(inp, normalized=False, eps=0.0)
        self.assert_close(edges, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        op = Sobel()
        actual = op(inp)

        assert inp.is_contiguous() is False
        assert actual.is_contiguous()
        assert actual.shape == (3, 3, 5, 5)

    @pytest.mark.parametrize('normalized', [True, False])
    def test_gradcheck(self, normalized, device):
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(sobel, (img, normalized))

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = sobel
        op_module = Sobel()
        expected = op(img)
        actual = op_module(img)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        if dtype == torch.float64:
            # TODO: investigate sobel for float64 with dynamo
            pytest.xfail(reason='The sobel results can be different after dynamo on fp64')
        inpt = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = Sobel()
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))
