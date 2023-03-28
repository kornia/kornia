import pytest
import torch

from kornia.filters import DexiNed, filter2d, filter2d_separable, filter3d
from kornia.testing import BaseTester, assert_close, tensor_to_gradcheck_var
from kornia.utils._compat import torch_version_le


class TestFilter2D(BaseTester):
    @pytest.mark.parametrize("border_type", ['constant', 'reflect', 'replicate', 'circular'])
    @pytest.mark.parametrize("normalized", [True, False])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_smoke(self, border_type, normalized, padding, device, dtype):
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        _, height, width = kernel.shape
        sample = torch.ones(1, 1, 7, 8, device=device, dtype=dtype)
        b, c, h, w = sample.shape

        actual = filter2d(sample, kernel, border_type, normalized, padding)
        assert isinstance(actual, torch.Tensor)
        assert actual.shape in {(b, c, h, w), (b, c, h - height + 1, w - width + 1)}

    @pytest.mark.parametrize("batch_size", [2, 3, 6, 8])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_cardinality(self, batch_size, padding, device, dtype):
        B: int = batch_size
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        _, height, width = kernel.shape
        sample = torch.ones(B, 3, 7, 8, device=device, dtype=dtype)
        b, c, h, w = sample.shape
        out = filter2d(sample, kernel, padding=padding)
        if padding == 'same':
            assert out.shape == (b, c, h, w)
        else:
            assert out.shape == (b, c, h - height + 1, w - width + 1)

    def test_conv(self, device, dtype):
        inp = torch.zeros(1, 1, 5, 5, device=device, dtype=dtype)
        inp[..., 2, 2] = 1.0
        kernel = torch.arange(1, 10).reshape(3, 3).to(device, dtype)[None]
        corr_expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 9.0, 8.0, 7.0, 0.0],
                        [0.0, 6.0, 5.0, 4.0, 0.0],
                        [0.0, 3.0, 2.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        conv_expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 2.0, 3.0, 0.0],
                        [0.0, 4.0, 5.0, 6.0, 0.0],
                        [0.0, 7.0, 8.0, 9.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        out_corr = filter2d(inp, kernel, behaviour='corr')
        assert_close(out_corr, corr_expected)
        out_conv = filter2d(inp, kernel, behaviour='conv')
        assert_close(out_conv, conv_expected)

    def test_exception(self):
        k = torch.ones(1, 1, 1)
        inpt = torch.ones(1, 1, 1, 1)
        with pytest.raises(TypeError) as errinfo:
            filter2d(1, k)
        assert 'Not a Tensor type.' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            filter2d(inpt, 1)
        assert 'Not a Tensor type.' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            filter2d(torch.ones(1), k)
        assert 'shape must be [[\'B\', \'C\', \'H\', \'W\']]' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            filter2d(inpt, torch.ones(1))
        assert 'shape must be [[\'B\', \'H\', \'W\']]' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            filter2d(inpt, k, border_type='a')
        assert 'Invalid border, gotcha a. Ex' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            filter2d(inpt, k, padding='a')
        assert 'Invalid padding mode, gotcha a. Ex' in str(errinfo)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_mean_filter(self, padding, device, dtype):
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        sample = torch.tensor(
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

        actual = filter2d(sample, kernel, padding=padding)

        if padding == 'same':
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

            self.assert_close(actual, expected_same)
        else:
            expected_valid = torch.tensor(
                [[[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]]], device=device, dtype=dtype
            )

            self.assert_close(actual, expected_valid)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_mean_filter_2batch_2ch(self, padding, device, dtype):
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        sample = torch.tensor(
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

        actual = filter2d(sample, kernel, padding=padding)

        if padding == 'same':
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

            self.assert_close(actual, expected_same)
        else:
            expected_valid = torch.tensor(
                [[[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]]], device=device, dtype=dtype
            ).expand(2, 2, -1, -1)
            self.assert_close(actual, expected_valid)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_normalized_mean_filter(self, padding, device, dtype):
        kernel = torch.ones(1, 3, 3, device=device, dtype=dtype)
        sample = torch.tensor(
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
        actual = filter2d(sample, kernel, normalized=True, padding=padding)

        if padding == 'same':
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

            self.assert_close(actual, expected_same)
        else:
            expected_valid = torch.tensor(
                [[[[nv, nv, nv], [nv, nv, nv], [nv, nv, nv]]]], device=device, dtype=dtype
            ).expand(2, 2, -1, -1)

            self.assert_close(actual, expected_valid)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_even_sized_filter(self, padding, device, dtype):
        kernel = torch.ones(1, 2, 2, device=device, dtype=dtype)
        sample = torch.tensor(
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

        actual = filter2d(sample, kernel, padding=padding)

        if padding == 'same':
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

            self.assert_close(actual, expected_same)
        else:
            expected_valid = torch.tensor(
                [[[[0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 5.0, 0.0], [0.0, 5.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
                device=device,
                dtype=dtype,
            )

            self.assert_close(actual, expected_valid)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_mix_sized_filter_padding_same(self, padding, device, dtype):
        kernel = torch.ones(1, 5, 6, device=device, dtype=dtype)
        sample_ = torch.tensor(
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
                        [2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        actual = filter2d(sample_, kernel, padding='same', border_type='constant')
        self.assert_close(actual, expected_same)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_noncontiguous(self, padding, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        kernel = torch.ones(1, 2, 2, device=device, dtype=dtype)

        actual = filter2d(inp, kernel, padding=padding)
        assert actual.is_contiguous()

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_separable(self, padding, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 9, 9, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        kernel_x = torch.ones(1, 3, device=device, dtype=dtype)
        kernel_y = torch.ones(1, 3, device=device, dtype=dtype)
        kernel = kernel_y.t() @ kernel_x
        out = filter2d(inp, kernel[None], padding=padding)
        out_sep = filter2d_separable(inp, kernel_x, kernel_y, padding=padding)
        self.assert_close(out, out_sep)

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3, device=device)
        sample = torch.ones(1, 1, 7, 8, device=device)

        # evaluate function gradient
        sample = tensor_to_gradcheck_var(sample)  # to var
        kernel = tensor_to_gradcheck_var(kernel)  # to var
        self.gradcheck(filter2d, (sample, kernel), nondet_tol=1e-8)

    @pytest.mark.skip(reason='filter2d do not have a module')
    def test_module(self):
        ...

    @pytest.mark.parametrize("normalized", [True, False])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_dynamo(self, normalized, padding, device, dtype, torch_optimizer):
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        inpt = torch.ones(2, 3, 10, 10, device=device, dtype=dtype)
        op = filter2d
        op_optimized = torch_optimizer(op)

        expected = op(inpt, kernel, padding=padding, normalized=normalized)
        actual = op_optimized(inpt, kernel, padding=padding, normalized=normalized)

        self.assert_close(actual, expected)


class TestFilter3D(BaseTester):
    @pytest.mark.parametrize("border_type", ['constant', 'reflect', 'replicate', 'circular'])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_smoke(self, border_type, normalized, device, dtype):
        if torch_version_le(1, 9, 1) and border_type == 'reflect':
            pytest.skip(reason='Reflect border is not implemented for 3D on torch < 1.9.1')

        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        inpt = torch.ones(1, 1, 6, 7, 8, device=device, dtype=dtype)
        actual = filter3d(inpt, kernel, border_type, normalized)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == inpt.shape

    @pytest.mark.parametrize("batch_size", [2, 3, 6, 8])
    def test_cardinality(self, batch_size, device, dtype):
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        inpt = torch.ones(batch_size, 3, 6, 7, 8, device=device, dtype=dtype)
        assert filter3d(inpt, kernel).shape == inpt.shape

    def test_exception(self):
        k = torch.ones(1, 1, 1, 1)
        inpt = torch.ones(1, 1, 1, 1, 1)
        with pytest.raises(TypeError) as errinfo:
            filter3d(1, k)
        assert 'Not a Tensor type.' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            filter3d(inpt, 1)
        assert 'Not a Tensor type.' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            filter3d(torch.ones(1), k)
        assert 'shape must be [[\'B\', \'C\', \'D\', \'H\', \'W\']]' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            filter3d(inpt, torch.ones(1))
        assert 'shape must be [[\'B\', \'D\', \'H\', \'W\']]' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            filter3d(inpt, k, border_type='a')
        assert 'Invalid border, gotcha a. Ex' in str(errinfo)

    def test_mean_filter(self, device, dtype):
        kernel = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        sample = torch.tensor(
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

        actual = filter3d(sample, kernel)
        self.assert_close(actual, expected)

    def test_mean_filter_2batch_2ch(self, device, dtype):
        kernel = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        sample = torch.tensor(
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
        sample = sample.expand(2, 2, -1, -1, -1)

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

        actual = filter3d(sample, kernel)
        self.assert_close(actual, expected)

    def test_normalized_mean_filter(self, device, dtype):
        kernel = torch.ones(1, 3, 3, 3, device=device, dtype=dtype)
        sample = torch.tensor(
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
        sample = sample.expand(2, 2, -1, -1, -1)

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

        actual = filter3d(sample, kernel, normalized=True)

        self.assert_close(actual, expected)

    def test_even_sized_filter(self, device, dtype):
        kernel = torch.ones(1, 2, 2, 2, device=device, dtype=dtype)
        sample = torch.tensor(
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

        actual = filter3d(sample, kernel)
        self.assert_close(actual, expected)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1, -1)
        kernel = torch.ones(1, 2, 2, 2, device=device, dtype=dtype)

        actual = filter3d(inp, kernel)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3, 3, device=device)
        sample = torch.ones(1, 1, 6, 7, 8, device=device)

        # evaluate function gradient
        sample = tensor_to_gradcheck_var(sample)  # to var
        kernel = tensor_to_gradcheck_var(kernel)  # to var
        self.gradcheck(filter3d, (sample, kernel), nondet_tol=1e-8)

    @pytest.mark.skip(reason='filter3d do not have a module')
    def test_module(self):
        ...

    @pytest.mark.parametrize("normalized", [True, False])
    def test_dynamo(self, normalized, device, dtype, torch_optimizer):
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        inpt = torch.ones(2, 3, 4, 10, 10, device=device, dtype=dtype)
        op = filter3d
        op_optimized = torch_optimizer(op)

        expected = op(inpt, kernel, normalized=normalized)
        actual = op_optimized(inpt, kernel, normalized=normalized)

        self.assert_close(actual, expected)


class TestDexiNed:
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        net = DexiNed(pretrained=False).to(device, dtype)
        out = net(img)
        assert len(out) == 7
        assert out[-1].shape == (2, 1, 32, 32)

    @pytest.mark.parametrize("data", ["dexined"], indirect=True)
    def test_inference(self, device, dtype, data):
        model = DexiNed(pretrained=False)
        model.load_state_dict(data, strict=True)
        model = model.to(device, dtype)
        model.eval()

        img = torch.tensor([[[[0.0, 255.0, 0.0], [0.0, 255.0, 0.0], [0.0, 255.0, 0.0]]]], device=device, dtype=dtype)
        img = img.repeat(1, 3, 1, 1)

        expect = torch.tensor(
            [[[[-0.3709, 0.0519, -0.2839], [0.0627, 0.6587, -0.1276], [-0.1840, -0.3917, -0.8240]]]],
            device=device,
            dtype=dtype,
        )

        out = model(img)[-1]
        assert_close(out, expect, atol=3e-4, rtol=3e-4)

    @pytest.mark.skip(reason='DexiNed do not compile with dynamo.')
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: update the dexined to be possible to use with dynamo
        inpt = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        op = DexiNed(pretrained=True).to(device, dtype)
        op_optimized = torch_optimizer(op)

        expected = op(inpt)
        actual = op_optimized(inpt)

        assert_close(actual, expected)
