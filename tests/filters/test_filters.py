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
import torch.nn.functional as F

from kornia.core._compat import torch_version_le
from kornia.core.exceptions import BaseError
from kornia.filters import (
    convolve2d,
    convolve3d,
    correlate2d,
    correlate3d,
    fft_conv,
    filter2d,
    filter2d_separable,
    filter3d,
    gaussian,
    get_binary_kernel2d,
    get_box_kernel1d,
    get_box_kernel2d,
    get_diff_kernel2d,
    get_gaussian_discrete_kernel1d,
    get_gaussian_erf_kernel1d,
    get_gaussian_kernel1d,
    get_gaussian_kernel2d,
    get_gaussian_kernel3d,
    get_hanning_kernel1d,
    get_hanning_kernel2d,
    get_laplacian_kernel1d,
    get_laplacian_kernel2d,
    get_motion_kernel2d,
    get_motion_kernel3d,
    get_sobel_kernel2d,
    get_spatial_gradient_kernel2d,
    get_spatial_gradient_kernel3d,
    laplacian_1d,
)

from testing.base import (
    BaseTester,
    _probe_zeros,
    _supports_kernel_probe,
    supports_nearest_3d_grid_sample,
    supports_reflect_padding,
)


class TestFilter2D(BaseTester):
    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
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
        if padding == "same":
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
        out_corr = filter2d(inp, kernel, behaviour="corr")
        self.assert_close(out_corr, corr_expected)
        out_conv = filter2d(inp, kernel, behaviour="conv")
        self.assert_close(out_conv, conv_expected)

    def test_exception(self):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        k = torch.ones(1, 1, 1)
        data = torch.ones(1, 1, 1, 1)
        with pytest.raises(TypeCheckError) as errinfo:
            filter2d(1, k)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            filter2d(data, 1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            filter2d(torch.ones(1), k)
        assert "Shape dimension mismatch" in str(errinfo.value)
        assert "['B', 'C', 'H', 'W']" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            filter2d(data, torch.ones(1))
        assert "Shape dimension mismatch" in str(errinfo.value)
        assert "['B', 'H', 'W']" in str(errinfo.value)

        with pytest.raises(Exception) as errinfo:
            filter2d(data, k, border_type="a")
        assert "Invalid border, a. Ex" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            filter2d(data, k, padding="a")
        assert "Invalid padding mode, a. Ex" in str(errinfo)

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

        if padding == "same":
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

        if padding == "same":
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

        if padding == "same":
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

        if padding == "same":
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

        actual = filter2d(sample_, kernel, padding="same", border_type="constant")
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
        kernel = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        sample = torch.ones(1, 1, 7, 8, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(filter2d, (sample, kernel), nondet_tol=1e-8)

    @pytest.mark.skip(reason="filter2d do not have a module")
    def test_module(self): ...

    @pytest.mark.parametrize("normalized", [True, False])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_dynamo(self, normalized, padding, device, dtype, torch_optimizer):
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 10, 10, device=device, dtype=dtype)
        op = filter2d
        op_optimized = torch_optimizer(op)

        expected = op(data, kernel, padding=padding, normalized=normalized)
        actual = op_optimized(data, kernel, padding=padding, normalized=normalized)

        self.assert_close(actual, expected)


class TestFilter3D(BaseTester):
    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_smoke(self, border_type, normalized, device, dtype):
        if torch_version_le(1, 9, 1) and border_type == "reflect":
            pytest.skip(reason="Reflect border is not implemented for 3D on torch < 1.9.1")

        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        data = torch.ones(1, 1, 6, 7, 8, device=device, dtype=dtype)
        actual = filter3d(data, kernel, border_type, normalized)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == data.shape

    @pytest.mark.parametrize("batch_size", [2, 3, 6, 8])
    def test_cardinality(self, batch_size, device, dtype):
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        data = torch.ones(batch_size, 3, 6, 7, 8, device=device, dtype=dtype)
        assert filter3d(data, kernel).shape == data.shape

    def test_exception(self):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        k = torch.ones(1, 1, 1, 1)
        data = torch.ones(1, 1, 1, 1, 1)
        with pytest.raises(TypeCheckError) as errinfo:
            filter3d(1, k)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            filter3d(data, 1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            filter3d(torch.ones(1), k)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            filter3d(data, torch.ones(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        with pytest.raises(Exception) as errinfo:
            filter3d(data, k, border_type="a")
        assert "Invalid border, gotcha a. Ex" in str(errinfo)

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

    @pytest.mark.parametrize("kernel_batch", [1, 2])
    @pytest.mark.parametrize("normalized", [True, False])
    @pytest.mark.parametrize("behaviour", ["corr", "conv"])
    def test_noncontiguous_kernel(self, kernel_batch, normalized, behaviour, device, dtype):
        data = torch.arange(840, device=device, dtype=dtype).reshape(2, 2, 7, 6, 5) / 840
        kernel = (torch.arange(30 * kernel_batch, device=device, dtype=dtype) % 7 - 3).reshape(kernel_batch, 2, 3, 5)
        kernel = kernel.permute(0, 3, 2, 1)
        assert not kernel.is_contiguous()

        weights = kernel.flip((-3, -2, -1)) if behaviour == "conv" else kernel
        if normalized:
            weights = weights / weights.abs().sum(dim=(-3, -2, -1), keepdim=True)
        expected = torch.cat(
            [
                torch.nn.functional.conv3d(
                    torch.nn.functional.pad(data[i : i + 1], (0, 1, 1, 1, 2, 2), mode="replicate"),
                    weights[0 if kernel_batch == 1 else i][None, None].expand(2, 1, -1, -1, -1),
                    groups=2,
                )
                for i in range(2)
            ]
        )
        actual = filter3d(data, kernel, normalized=normalized, behaviour=behaviour)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("normalized", [True, False])
    @pytest.mark.parametrize("behaviour", ["corr", "conv"])
    @pytest.mark.parametrize("noncontiguous", [True, False])
    def test_gradcheck(self, normalized, behaviour, noncontiguous, device):
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=torch.float64)
        if noncontiguous:
            kernel = kernel.permute(0, 3, 2, 1)
        sample = torch.ones(1, 1, 6, 7, 8, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(
            lambda data, kernel: filter3d(data, kernel, normalized=normalized, behaviour=behaviour),
            (sample, kernel),
            nondet_tol=1e-8,
        )

    @pytest.mark.skip(reason="filter3d do not have a module")
    def test_module(self): ...

    @pytest.mark.parametrize("normalized", [True, False])
    def test_dynamo(self, normalized, device, dtype, torch_optimizer):
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 4, 10, 10, device=device, dtype=dtype)
        op = filter3d
        op_optimized = torch_optimizer(op)

        expected = op(data, kernel, normalized=normalized)
        actual = op_optimized(data, kernel, normalized=normalized)

        self.assert_close(actual, expected)


class TestFilter2D_fftconv(BaseTester):
    @pytest.mark.parametrize("padding", ["same", "valid"])
    @pytest.mark.parametrize("behaviour", ["corr", "conv"])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_matches_spatial_filter(self, padding, behaviour, normalized, device, dtype):
        sample = torch.arange(1, 169, device=device, dtype=dtype).reshape(2, 2, 6, 7) / 128
        kernel = torch.tensor([[[1, 2, 3], [3, 2, 1]], [[2, 1, 2], [1, 3, 1]]], device=device, dtype=dtype)

        actual = fft_conv(sample, kernel, normalized=normalized, padding=padding, behaviour=behaviour)
        expected = filter2d(sample, kernel, normalized=normalized, padding=padding, behaviour=behaviour)

        assert actual.dtype == dtype
        assert actual.device == sample.device
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("normalized", [True, False])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_smoke(self, border_type, normalized, padding, device, dtype):
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        _, height, width = kernel.shape
        sample = torch.ones(1, 1, 7, 8, device=device, dtype=dtype)
        b, c, h, w = sample.shape

        actual = fft_conv(sample, kernel, border_type, normalized, padding)
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
        out = fft_conv(sample, kernel, padding=padding)
        if padding == "same":
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
        out_corr = fft_conv(inp, kernel, behaviour="corr")
        self.assert_close(out_corr, corr_expected, atol=1e-6, rtol=1e-6)
        out_conv = fft_conv(inp, kernel, behaviour="conv")
        self.assert_close(out_conv, conv_expected, atol=1e-6, rtol=1e-6)

    def test_exception(self):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        k = torch.ones(1, 1, 1)
        data = torch.ones(1, 1, 1, 1)
        with pytest.raises(TypeCheckError) as errinfo:
            fft_conv(1, k)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            fft_conv(data, 1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            fft_conv(torch.ones(1), k)
        assert "Shape dimension mismatch" in str(errinfo.value)
        assert "['B', 'C', 'H', 'W']" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            fft_conv(data, torch.ones(1))
        assert "Shape dimension mismatch" in str(errinfo.value)
        assert "['B', 'H', 'W']" in str(errinfo.value)

        with pytest.raises(Exception) as errinfo:
            fft_conv(data, k, border_type="a")
        assert "Invalid border, a. Ex" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            fft_conv(data, k, padding="a")
        assert "Invalid padding mode, a. Ex" in str(errinfo)

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

        actual = fft_conv(sample, kernel, padding=padding)

        if padding == "same":
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

        actual = fft_conv(sample, kernel, padding=padding)

        if padding == "same":
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
        actual = fft_conv(sample, kernel, normalized=True, padding=padding)

        if padding == "same":
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

        actual = fft_conv(sample, kernel, padding=padding)

        if padding == "same":
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

        actual = fft_conv(sample_, kernel, padding="same", border_type="constant")
        self.assert_close(actual, expected_same)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_noncontiguous(self, padding, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        kernel = torch.ones(1, 2, 2, device=device, dtype=dtype)

        actual = fft_conv(inp, kernel, padding=padding)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        kernel = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        sample = torch.ones(1, 1, 7, 8, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(fft_conv, (sample, kernel), nondet_tol=1e-8)

    @pytest.mark.skip(reason="filter2d do not have a module")
    def test_module(self): ...

    @pytest.mark.parametrize("normalized", [True, False])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_dynamo(self, normalized, padding, device, dtype, torch_optimizer):
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 10, 10, device=device, dtype=dtype)
        op = fft_conv
        op_optimized = torch_optimizer(op)

        expected = op(data, kernel, padding=padding, normalized=normalized)
        actual = op_optimized(data, kernel, padding=padding, normalized=normalized)

        self.assert_close(actual, expected)


class TestCorrelateConvolveExports:
    """The four correlate/convolve aliases are public in every sense kornia uses (#5161)."""

    NAMES = ("correlate2d", "convolve2d", "correlate3d", "convolve3d")

    def test_from_import_names_the_defining_objects(self):
        import importlib

        from kornia.filters import convolve2d, convolve3d, correlate2d, correlate3d

        defining = importlib.import_module("kornia.filters.filter")
        imported = {
            "correlate2d": correlate2d,
            "convolve2d": convolve2d,
            "correlate3d": correlate3d,
            "convolve3d": convolve3d,
        }
        assert set(imported) == set(self.NAMES)
        for name, func in imported.items():
            assert func is getattr(defining, name), name
            assert func.__module__ == "kornia.filters.filter", name

    @pytest.mark.parametrize("name", NAMES)
    def test_listed_in_all(self, name):
        import kornia.filters as KF

        # `from kornia.filters import *` binds exactly the names in `__all__`.
        assert name in KF.__all__

    def test_all_is_well_formed(self):
        import kornia.filters as KF

        assert len(KF.__all__) == len(set(KF.__all__))
        assert all(hasattr(KF, name) for name in KF.__all__)


class TestCorrelateConvolveShapes(BaseTester):
    """The output shapes the four docstrings state, for odd and even kernels."""

    @pytest.mark.parametrize("func", [correlate2d, convolve2d])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (3, 3), (2, 2), (4, 3), (2, 5)])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_cardinality_2d(self, func, kernel_size, padding, device, dtype):
        b, c, h, w = 2, 3, 7, 8
        kh, kw = kernel_size
        inp = torch.ones(b, c, h, w, device=device, dtype=dtype)
        kernel = torch.ones(1, kh, kw, device=device, dtype=dtype)
        expected = (b, c, h, w) if padding == "same" else (b, c, h - kh + 1, w - kw + 1)
        # the shape does not depend on the border; "constant" also runs in half precision on old torch
        assert func(inp, kernel, border_type="constant", padding=padding).shape == expected

    @pytest.mark.parametrize("func", [correlate3d, convolve3d])
    @pytest.mark.parametrize("kernel_size", [(1, 1, 1), (3, 3, 3), (2, 2, 2), (4, 3, 2)])
    def test_cardinality_3d(self, func, kernel_size, device, dtype):
        inp = torch.ones(2, 3, 5, 7, 8, device=device, dtype=dtype)
        kernel = torch.ones(1, *kernel_size, device=device, dtype=dtype)
        # the shape does not depend on the border; "constant" also runs in half precision on old torch
        assert func(inp, kernel, border_type="constant").shape == inp.shape


class TestCorrelate2d(BaseTester):
    def test_equivalent_to_filter2d_corr(self, device, dtype):
        inp = torch.rand(1, 1, 7, 8, device=device, dtype=dtype)
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        expected = filter2d(inp, kernel, behaviour="corr")
        result = correlate2d(inp, kernel)
        self.assert_close(result, expected)

    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_smoke(self, border_type, padding, device, dtype):
        inp = torch.ones(1, 1, 7, 8, device=device, dtype=dtype)
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        out = correlate2d(inp, kernel, border_type=border_type, padding=padding)
        assert isinstance(out, torch.Tensor)


class TestConvolve2d(BaseTester):
    def test_equivalent_to_filter2d_conv(self, device, dtype):
        inp = torch.rand(1, 1, 7, 8, device=device, dtype=dtype)
        kernel = torch.rand(1, 3, 3, device=device, dtype=dtype)
        expected = filter2d(inp, kernel, behaviour="conv")
        result = convolve2d(inp, kernel)
        self.assert_close(result, expected)

    def test_differs_from_correlate_asymmetric_kernel(self, device, dtype):
        inp = torch.rand(1, 1, 7, 8, device=device, dtype=dtype)
        # Asymmetric kernel so corr != conv
        kernel = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype)
        corr = correlate2d(inp, kernel)
        conv = convolve2d(inp, kernel)
        assert not torch.allclose(corr, conv)


class TestCorrelate3d(BaseTester):
    def test_equivalent_to_filter3d_corr(self, device, dtype):
        inp = torch.rand(1, 1, 5, 7, 8, device=device, dtype=dtype)
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        expected = filter3d(inp, kernel, behaviour="corr")
        result = correlate3d(inp, kernel)
        self.assert_close(result, expected)


class TestConvolve3d(BaseTester):
    def test_equivalent_to_filter3d_conv(self, device, dtype):
        inp = torch.rand(1, 1, 5, 7, 8, device=device, dtype=dtype)
        kernel = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        expected = filter3d(inp, kernel, behaviour="conv")
        result = convolve3d(inp, kernel)
        self.assert_close(result, expected)


# ----------------------------------------------------------------------------------------------------------------------
# Convention and wart pins: the filtering API and the kernel builders.


def _replicate_padding_3d_op(device_type: str, dtype: torch.dtype) -> None:
    F.pad(_probe_zeros(device_type, dtype, 1, 1, 2, 2, 2), (1, 1, 1, 1, 1, 1), mode="replicate")


def _nearest_2d_grid_sample_op(device_type: str, dtype: torch.dtype) -> None:
    grid = _probe_zeros(device_type, dtype, 1, 1, 1, 2)
    F.grid_sample(_probe_zeros(device_type, dtype, 1, 1, 2, 2), grid, mode="nearest", align_corners=True)


def _supports_replicate_padding_3d(device: torch.device, dtype: torch.dtype) -> bool:
    return _supports_kernel_probe(_replicate_padding_3d_op, device.type, dtype)


def _supports_nearest_2d_grid_sample(device: torch.device, dtype: torch.dtype) -> bool:
    return _supports_kernel_probe(_nearest_2d_grid_sample_op, device.type, dtype)


def _rand(*shape: int, device, dtype, seed: int = 0) -> torch.Tensor:
    """Draw on CPU from a private generator, then move, so the values do not depend on the device."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=generator).to(device=device, dtype=dtype)


def _delta(shape: tuple[int, ...], index: tuple[int, ...], device, dtype) -> torch.Tensor:
    out = torch.zeros(shape, device=device, dtype=dtype)
    out[index] = 1.0
    return out


def _peaks(out: torch.Tensor) -> list[list[int]]:
    """Positions of the entries above 0.5 in ``out[0, 0]`` (an FFT result carries roundoff elsewhere)."""
    return (out[0, 0].float().abs() > 0.5).nonzero().tolist()


_FILTER2D_FNS = {"filter2d": filter2d, "fft_conv": fft_conv}


def _fft_guard(name: str, device: torch.device, dtype: torch.dtype) -> None:
    """Skip fft_conv where the device has no FFT for the dtype (fft_conv computes a CPU half input in float32)."""
    if name != "fft_conv" or device.type == "cpu":
        return
    try:
        torch.fft.rfftn(torch.zeros(3, 5, device=device, dtype=dtype))
    except (RuntimeError, NotImplementedError):
        pytest.skip("this device has no FFT for this dtype, which fft_conv needs")


class TestConventionsFilter2d(BaseTester):
    @pytest.mark.parametrize("name", ["filter2d", "fft_conv"])
    def test_convention_filter2d_is_correlation(self, name, device, dtype):
        _fft_guard(name, device, dtype)
        fn = _FILTER2D_FNS[name]
        image = _delta((1, 1, 6, 9), (0, 0, 2, 5), device, dtype)
        kernel = _delta((1, 3, 3), (0, 0, 2), device, dtype)  # one weight on the up-right neighbour
        # correlation: out[y, x] = image[y - 1, x + 1], so the delta lands one row down and one column left
        assert _peaks(fn(image, kernel, border_type="constant")) == [[3, 4]]
        # behaviour='conv' flips the kernel first
        assert _peaks(fn(image, kernel, border_type="constant", behaviour="conv")) == [[1, 6]]
        # relabel: transposing the image and the kernel transposes the output
        transposed = fn(image.transpose(-2, -1), kernel.transpose(-2, -1), border_type="constant")
        assert _peaks(transposed) == [[4, 3]]

    def test_convention_filter3d_is_correlation(self, device, dtype):
        volume = _delta((1, 1, 5, 6, 7), (0, 0, 1, 4, 5), device, dtype)  # D != H != W
        kernel = _delta((1, 3, 3, 3), (0, 0, 2, 1), device, dtype)
        # correlation: out[z, y, x] = volume[z - 1, y + 1, x]
        assert _peaks(filter3d(volume, kernel, border_type="constant")) == [[2, 3, 5]]
        assert _peaks(filter3d(volume, kernel, border_type="constant", behaviour="conv")) == [[0, 5, 5]]
        # relabel: swapping D and W in the volume and the kernel reverses the output coordinates
        swapped = filter3d(volume.permute(0, 1, 4, 3, 2), kernel.permute(0, 3, 2, 1), border_type="constant")
        assert _peaks(swapped) == [[5, 3, 2]]

    @pytest.mark.parametrize("name", ["filter2d", "filter2d_separable", "fft_conv"])
    def test_convention_filter2d_reflect_excludes_the_edge_pixel(self, name, device, dtype):
        _fft_guard(name, device, dtype)
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        ys, xs = torch.meshgrid(torch.arange(5), torch.arange(7), indexing="ij")
        ramp = (xs + 10 * ys).float()[None, None]  # value = x + 10 y on a 5 x 7 image
        # One weight on the up-left neighbour: out[y, x] = padded[y - 1, x - 1]. torch's reflect mirrors about the edge
        # pixel without repeating it (scipy 'mirror', OpenCV BORDER_REFLECT_101), so row and column -1 read row and
        # column 1; scipy's 'reflect' would read row and column 0.
        rows, cols = [1, 0, 1, 2, 3], [1, 0, 1, 2, 3, 4, 5]
        expected = ramp[:, :, rows][:, :, :, cols].to(device=device, dtype=dtype)
        image = ramp.to(device=device, dtype=dtype)
        if name == "filter2d_separable":
            tap = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
            default, reflect = filter2d_separable(image, tap, tap), filter2d_separable(image, tap, tap, "reflect")
        else:
            kernel = _delta((1, 3, 3), (0, 0, 0), device, dtype)
            fn = _FILTER2D_FNS[name]
            default, reflect = fn(image, kernel), fn(image, kernel, border_type="reflect")
        self.assert_close(default, expected)
        self.assert_close(reflect, expected)

    def test_convention_filter3d_default_border_is_replicate(self, device, dtype):
        if not _supports_replicate_padding_3d(device, dtype):
            pytest.skip("replication_pad3d is unavailable for this device/dtype")
        zs, ys, xs = torch.meshgrid(torch.arange(3), torch.arange(4), torch.arange(5), indexing="ij")
        ramp = (xs + 10 * ys + 100 * zs).float()[None, None]  # at most 234, exact in bfloat16
        # One weight at the kernel's corner: out[z, y, x] = padded[z - 1, y - 1, x - 1]. filter3d defaults to
        # 'replicate' (filter2d to 'reflect'), so index -1 reads the edge sample 0 on every axis.
        expected = ramp[:, :, [0, 0, 1]][:, :, :, [0, 0, 1, 2]][:, :, :, :, [0, 0, 1, 2, 3]]
        volume = ramp.to(device=device, dtype=dtype)
        kernel = _delta((1, 3, 3, 3), (0, 0, 0, 0), device, dtype)
        self.assert_close(filter3d(volume, kernel), expected.to(device=device, dtype=dtype))

    @pytest.mark.parametrize("name", ["filter2d", "filter2d_separable", "fft_conv"])
    def test_convention_filter2d_even_kernel_centre(self, name, device, dtype):
        _fft_guard(name, device, dtype)

        # A single weight at the kernel's first tap. The anchor of a (kH, kW) kernel is ((kH - 1) // 2, (kW - 1) // 2),
        # the anchor of torch's F.conv2d(padding='same'): a (4, 2) kernel moves the delta by (+1, 0). An anchor at
        # k // 2 (the OpenCV and scipy default, and kornia.morphology's) would move it by (+2, +1).
        def run(image, kh, kw):
            if name == "filter2d_separable":
                kernel_x = _delta((1, kw), (0, 0), device, dtype)
                kernel_y = _delta((1, kh), (0, 0), device, dtype)
                return filter2d_separable(image, kernel_x, kernel_y, border_type="constant")
            return _FILTER2D_FNS[name](image, _delta((1, kh, kw), (0, 0, 0), device, dtype), border_type="constant")

        image = _delta((1, 1, 6, 9), (0, 0, 2, 5), device, dtype)
        assert _peaks(run(image, 4, 2)) == [[3, 5]]
        # relabel: the transposed image with the transposed (2, 4) kernel gives the transposed position
        assert _peaks(run(image.transpose(-2, -1), 2, 4)) == [[5, 3]]
        if name == "filter2d_separable":
            return  # no behaviour argument
        # behaviour='conv' flips the kernel, putting the weight at its last tap (3, 1), and keeps the anchor (1, 0):
        # the delta moves by (-2, -1). An anchor at k // 2 would move it by (-1, 0).
        fn = _FILTER2D_FNS[name]
        conv = fn(image, _delta((1, 4, 2), (0, 0, 0), device, dtype), border_type="constant", behaviour="conv")
        assert _peaks(conv) == [[0, 4]]
        kernel_t = _delta((1, 2, 4), (0, 0, 0), device, dtype)
        conv_t = fn(image.transpose(-2, -1), kernel_t, border_type="constant", behaviour="conv")
        assert _peaks(conv_t) == [[4, 0]]

    @pytest.mark.parametrize("name", ["filter2d", "fft_conv"])
    def test_convention_filter2d_conv_equals_scipy_convolve_for_an_even_kernel(self, name, device, dtype):
        _fft_guard(name, device, dtype)
        # scipy 1.17.1, numpy 2.0.0:
        #   image = np.random.default_rng(0).integers(0, 3, (5, 7)).astype(float)
        #   ndi.convolve(image, [[1, 2, 3, 4], [5, 6, 7, 8]], mode="constant", cval=0.0)  # default origin
        image = [[2, 1, 1, 0, 0, 0, 0], [0, 0, 2, 1, 2, 1, 1], [2, 2, 1, 1, 1, 2, 0], [2, 2, 0, 1, 2, 1, 0]]
        image += [[2, 2, 2, 0, 0, 2, 0]]
        expected = [[27, 34, 25, 24, 13, 13, 7], [21, 34, 44, 51, 44, 39, 23], [41, 56, 46, 39, 39, 33, 20]]
        expected += [[38, 53, 46, 34, 32, 29, 16], [36, 42, 30, 26, 12, 14, 16]]
        image_t = torch.tensor(image, device=device, dtype=dtype)[None, None]
        expected_t = torch.tensor(expected, device=device, dtype=dtype)[None, None]
        kernel = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], device=device, dtype=dtype)
        fn = _FILTER2D_FNS[name]
        self.assert_close(fn(image_t, kernel, "constant", behaviour="conv"), expected_t)
        # relabel: transposing the image and the kernel transposes the output
        conv_t = fn(image_t.transpose(-2, -1), kernel.transpose(-2, -1), "constant", behaviour="conv")
        self.assert_close(conv_t, expected_t.transpose(-2, -1))

    def test_convention_filter3d_even_kernel_centre(self, device, dtype):
        volume = _delta((1, 1, 5, 6, 7), (0, 0, 1, 4, 3), device, dtype)
        kernel = _delta((1, 4, 2, 6), (0, 0, 0, 0), device, dtype)
        # anchor ((4 - 1) // 2, (2 - 1) // 2, (6 - 1) // 2) = (1, 0, 2) on (D, H, W)
        assert _peaks(filter3d(volume, kernel, border_type="constant")) == [[2, 4, 5]]
        swapped = filter3d(volume.permute(0, 1, 4, 3, 2), kernel.permute(0, 3, 2, 1), border_type="constant")
        assert _peaks(swapped) == [[5, 4, 2]]

    @pytest.mark.parametrize("name", ["filter2d", "filter2d_separable", "fft_conv"])
    def test_convention_filter2d_valid_output_is_the_same_output_cropped(self, name, device, dtype):
        _fft_guard(name, device, dtype)
        image = _rand(2, 1, 6, 9, device=device, dtype=dtype)
        kernel_y = _rand(1, 4, device=device, dtype=dtype, seed=1)
        kernel_x = _rand(1, 5, device=device, dtype=dtype, seed=2)
        outs = {}
        for padding in ("same", "valid"):
            if name == "filter2d_separable":
                outs[padding] = filter2d_separable(image, kernel_x, kernel_y, "constant", padding=padding)
            else:
                kernel = kernel_y[:, :, None] * kernel_x[:, None, :]
                outs[padding] = _FILTER2D_FNS[name](image, kernel, "constant", padding=padding)
        # 'valid' returns (H - kH + 1, W - kW + 1), the 'same' output from the anchor ((4 - 1) // 2, (5 - 1) // 2) on
        assert outs["valid"].shape == (2, 1, 3, 5)
        self.assert_close(outs["valid"], outs["same"][..., 1:4, 2:7])

    def test_convention_filter2d_separable_equals_filter2d_with_the_outer_product(self, device, dtype):
        image = _rand(2, 3, 6, 9, device=device, dtype=dtype)
        kernel_x = _rand(2, 4, device=device, dtype=dtype, seed=1)  # per-sample, even, along W
        kernel_y = _rand(2, 3, device=device, dtype=dtype, seed=2)  # per-sample, along H
        out = filter2d_separable(image, kernel_x, kernel_y, border_type="constant")
        outer = kernel_y[:, :, None] * kernel_x[:, None, :]  # (B, kH, kW)
        # the separable path rounds its intermediate row pass, a few ulp apart from one pass in half precision
        half = dtype in (torch.float16, torch.bfloat16)
        self.assert_close(out, filter2d(image, outer, border_type="constant"), low_tolerance=half)

    def test_convention_filter2d_kernel_batch_is_per_sample_not_per_channel(self, device, dtype):
        # B = C = 3, so a per-sample and a per-channel reading of three kernels would differ
        image = _rand(3, 3, 5, 7, device=device, dtype=dtype)
        kernels = _rand(3, 3, 3, device=device, dtype=dtype, seed=1)
        out = filter2d(image, kernels, border_type="constant")
        shared = filter2d(image, kernels[:1], border_type="constant")
        for i in range(3):
            for c in range(3):
                plane = image[i : i + 1, c : c + 1]
                self.assert_close(out[i : i + 1, c : c + 1], filter2d(plane, kernels[i : i + 1], "constant"))
                self.assert_close(shared[i : i + 1, c : c + 1], filter2d(plane, kernels[:1], "constant"))
        # one sample with C kernels is not read as one kernel per channel
        with pytest.raises((RuntimeError, BaseError)):
            filter2d(image[:1], kernels, border_type="constant")

    @pytest.mark.parametrize("name", ["filter2d", "filter2d_separable", "fft_conv", "filter3d"])
    def test_convention_filter2d_normalized_divides_by_the_absolute_sum(self, name, device, dtype):
        _fft_guard(name, device, dtype)
        taps = torch.tensor([[1.0, 2.0, -4.0]], device=device, dtype=dtype)  # sum -1, absolute sum 7
        if name == "filter2d_separable":
            image = _rand(1, 2, 5, 7, device=device, dtype=dtype)
            taps_y = torch.tensor([[3.0, -1.0]], device=device, dtype=dtype)  # sum 2, absolute sum 4

            def call(*kernels, **kwargs):
                return filter2d_separable(image, *kernels, "constant", **kwargs)

            kernels, divided = (taps, taps_y), (taps / 7, taps_y / 4)
        else:
            fn = filter3d if name == "filter3d" else _FILTER2D_FNS[name]
            image = _rand(1, 2, *((3,) if name == "filter3d" else ()), 5, 7, device=device, dtype=dtype)
            # two rows (two depth slices in 3-D) with absolute sums 7 and 4: the whole kernel's is 11
            rows = torch.tensor([[1.0, 2.0, -4.0], [3.0, 0.0, 1.0]], device=device, dtype=dtype)
            kernel = rows[None, :, None] if name == "filter3d" else rows[None]

            def call(*kernels, **kwargs):
                return fn(image, *kernels, "constant", **kwargs)

            kernels, divided = (kernel,), (kernel / 11,)
        self.assert_close(call(*kernels, normalized=True), call(*divided))
        # the default is normalized=False
        self.assert_close(call(*kernels), call(*kernels, normalized=False))

    @pytest.mark.parametrize("name", ["filter2d", "fft_conv", "filter3d"])
    def test_convention_filter2d_casts_the_kernel_to_the_input_dtype_and_device(self, name, device, dtype):
        _fft_guard(name, device, dtype)
        fn = filter3d if name == "filter3d" else _FILTER2D_FNS[name]
        spatial = (3, 5, 7) if name == "filter3d" else (5, 7)
        image = _rand(1, 2, *spatial, device=device, dtype=dtype)
        kernel = torch.rand(1, *((3,) * len(spatial)), dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        kernel.requires_grad_(True)  # a float64 CPU leaf
        out = fn(image, kernel, "constant")
        assert out.dtype == dtype
        assert out.device == image.device
        self.assert_close(out, fn(image, kernel.detach().to(device=device, dtype=dtype), "constant"))
        # the cast keeps the kernel on the autograd graph
        out.float().sum().backward()
        assert kernel.grad is not None
        assert kernel.grad.dtype == torch.float64

    @pytest.mark.parametrize("border_type, too_narrow, wide_enough", [("reflect", 2, 3), ("circular", 1, 2)])
    def test_convention_filter2d_reflect_and_circular_need_the_axis_longer_than_the_pad(
        self, border_type, too_narrow, wide_enough, device, dtype
    ):
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        # A kernel 4 wide pads 1 column before and 2 after. reflect needs every pad shorter than the axis, circular
        # at most as long; a narrower axis raises.
        kernel = _rand(1, 1, 4, device=device, dtype=dtype)
        out = filter2d(_rand(1, 1, 3, wide_enough, device=device, dtype=dtype), kernel, border_type)
        assert out.shape == (1, 1, 3, wide_enough)
        with pytest.raises((RuntimeError, BaseError)):
            filter2d(_rand(1, 1, 3, too_narrow, device=device, dtype=dtype), kernel, border_type)
        # constant padding runs on any size
        assert filter2d(_rand(1, 1, 3, 1, device=device, dtype=dtype), kernel, "constant").shape == (1, 1, 3, 1)

    @pytest.mark.parametrize("name", ["filter2d", "filter2d_separable", "filter3d"])
    def test_wart_kernel_batch_dividing_the_input_batch_is_applied_cyclically_5154(self, name, device, dtype):
        """Two kernels for four samples are not rejected: sample i is filtered with kernel i % 2 (#5154)."""
        if name == "filter2d_separable":
            image = _rand(4, 1, 5, 7, device=device, dtype=dtype)
            kernel_x = _rand(2, 3, device=device, dtype=dtype, seed=1)
            kernel_y = _rand(2, 3, device=device, dtype=dtype, seed=2)

            def run(x, lo, hi):
                return filter2d_separable(x, kernel_x[lo:hi], kernel_y[lo:hi], "constant")

        else:
            fn = filter3d if name == "filter3d" else filter2d
            image = _rand(4, 1, *((3,) if name == "filter3d" else ()), 5, 7, device=device, dtype=dtype)
            kernels = _rand(2, *((3,) if name == "filter3d" else ()), 3, 3, device=device, dtype=dtype, seed=1)

            def run(x, lo, hi):
                return fn(x, kernels[lo:hi], "constant")

        out = run(image, 0, 2)
        assert out.shape == image.shape
        for i in range(4):
            self.assert_close(out[i : i + 1], run(image[i : i + 1], i % 2, i % 2 + 1))

    def test_wart_fft_conv_broadcasts_one_sample_over_the_kernel_batch_5154(self, device, dtype):
        """fft_conv returns a batch of 4 for one sample and 4 kernels, where filter2d raises (#5154)."""
        _fft_guard("fft_conv", device, dtype)
        image = _rand(1, 1, 5, 7, device=device, dtype=dtype)
        kernels = _rand(4, 3, 3, device=device, dtype=dtype, seed=1)
        out = fft_conv(image, kernels, border_type="constant")
        assert out.shape == (4, 1, 5, 7)
        for j in range(4):
            self.assert_close(out[j : j + 1], fft_conv(image, kernels[j : j + 1], border_type="constant"))

    @pytest.mark.parametrize("name", ["filter2d", "filter2d_separable", "filter3d", "fft_conv"])
    def test_wart_integer_input_truncates_a_fractional_kernel_to_zero_5155(self, name, device, dtype):
        """An integer input casts the kernel to its dtype: a 1/9 box kernel turns a constant 100 image to 0 (#5155)."""
        if name != "fft_conv" and device.type != "cpu":
            pytest.skip("integer convolution raises on this device, so there is no truncated result to pin (#5155)")
        spatial = (3, 5, 7) if name == "filter3d" else (5, 7)
        image = torch.full((1, 1, *spatial), 100, dtype=torch.uint8, device=device)
        if name == "filter2d_separable":
            third = torch.full((1, 3), 1 / 3, device=device, dtype=dtype)
            out = filter2d_separable(image, third, third, "constant")
        elif name == "filter3d":
            out = filter3d(image, torch.full((1, 3, 3, 3), 1 / 27, device=device, dtype=dtype), "constant")
        else:
            out = _FILTER2D_FNS[name](image, torch.full((1, 3, 3), 1 / 9, device=device, dtype=dtype), "constant")
        # the interior of a box-filtered constant image is 100 in floating point
        assert out[..., 2, 3].flatten()[0].item() == 0

    @pytest.mark.parametrize("name", ["filter2d", "fft_conv"])
    def test_wart_uppercase_padding_returns_the_valid_size_5156(self, name, device, dtype):
        """padding='SAME' passes the case-insensitive check but misses the 'same' branch: valid size (#5156)."""
        _fft_guard(name, device, dtype)
        image = _rand(1, 1, 5, 7, device=device, dtype=dtype)
        kernel = _rand(1, 3, 3, device=device, dtype=dtype, seed=1)
        assert _FILTER2D_FNS[name](image, kernel, "constant", padding="SAME").shape == (1, 1, 3, 5)

    @pytest.mark.parametrize(
        "case",
        ["filter2d_border_type", "fft_conv_border_type", "filter3d_border_type", "kernel2d_mode", "kernel3d_mode"],
    )
    def test_wart_uppercase_border_type_or_mode_passes_validation_then_raises_5156(self, case, device, dtype):
        """'REFLECT', 'Replicate', 'Sobel' and 'Diff' pass the case-insensitive check, then fail to dispatch (#5156)."""
        image = _rand(1, 1, 5, 7, device=device, dtype=dtype)
        kernel = _rand(1, 3, 3, device=device, dtype=dtype, seed=1)
        calls = {
            "filter2d_border_type": lambda: filter2d(image, kernel, border_type="REFLECT"),
            "fft_conv_border_type": lambda: fft_conv(image, kernel, border_type="REFLECT"),
            "filter3d_border_type": lambda: filter3d(image[:, :, None], kernel[:, None], border_type="Replicate"),
            "kernel2d_mode": lambda: get_spatial_gradient_kernel2d("Sobel", 1, device=device, dtype=dtype),
            "kernel3d_mode": lambda: get_spatial_gradient_kernel3d("Diff", 1, device=device, dtype=dtype),
        }
        # the error comes from past the check (torch's F.pad or the kernel dispatch), not from kornia's validation
        with pytest.raises(Exception) as error:
            calls[case]()
        assert not isinstance(error.value, BaseError)

    @pytest.mark.parametrize("behaviour", ["corr", "conv"])
    def test_convention_filter3d_normalized_accepts_a_non_contiguous_kernel_5159(self, behaviour, device, dtype):
        """filter3d(normalized=True) gives a permuted kernel the result of its contiguous copy (#5159)."""
        volume = _rand(1, 1, 5, 6, 7, device=device, dtype=dtype)
        kernel = _rand(1, 3, 4, 5, device=device, dtype=dtype, seed=1).permute(0, 3, 2, 1)  # (1, 5, 4, 3)
        assert not kernel.is_contiguous()
        expected = filter3d(volume, kernel.contiguous(), "constant", normalized=True, behaviour=behaviour)
        out = filter3d(volume, kernel, "constant", normalized=True, behaviour=behaviour)
        assert torch.equal(out, expected)

    def test_wart_fft_conv_valid_padding_with_a_kernel_larger_than_the_input_5285(self, device, dtype):
        """With padding='valid', a 7 x 3 kernel on a 5 x 6 image gives fft_conv a 4 x 4 output (#5285)."""
        _fft_guard("fft_conv", device, dtype)
        image = _rand(1, 1, 5, 6, device=device, dtype=dtype)
        kernel = _rand(1, 7, 3, device=device, dtype=dtype, seed=1)
        with pytest.raises((RuntimeError, BaseError)):
            filter2d(image, kernel, "constant", padding="valid")
        assert fft_conv(image, kernel, "constant", padding="valid").shape == (1, 1, 4, 4)


# (name, factory(device, dtype), shape) for non-square sizes, so every axis order is visible
_KERNEL_SHAPES = [
    ("get_box_kernel1d", lambda d, t: get_box_kernel1d(4, device=d, dtype=t), (1, 4)),
    ("get_box_kernel2d", lambda d, t: get_box_kernel2d((3, 4), device=d, dtype=t), (1, 3, 4)),
    ("gaussian", lambda d, t: gaussian(5, 1.0, device=d, dtype=t), (1, 5)),
    ("get_gaussian_kernel1d", lambda d, t: get_gaussian_kernel1d(5, 1.0, device=d, dtype=t), (1, 5)),
    ("get_gaussian_erf_kernel1d", lambda d, t: get_gaussian_erf_kernel1d(5, 1.0, device=d, dtype=t), (1, 5)),
    ("get_gaussian_discrete_kernel1d", lambda d, t: get_gaussian_discrete_kernel1d(5, 1.0, device=d, dtype=t), (1, 5)),
    ("get_gaussian_kernel2d", lambda d, t: get_gaussian_kernel2d((3, 5), (1.0, 1.0), device=d, dtype=t), (1, 3, 5)),
    (
        "get_gaussian_kernel3d",
        lambda d, t: get_gaussian_kernel3d((3, 5, 7), (1.0, 1.0, 1.0), device=d, dtype=t),
        (1, 3, 5, 7),
    ),
    ("get_laplacian_kernel1d", lambda d, t: get_laplacian_kernel1d(5, device=d, dtype=t), (5,)),
    ("get_laplacian_kernel2d", lambda d, t: get_laplacian_kernel2d((3, 5), device=d, dtype=t), (3, 5)),
    ("get_hanning_kernel1d", lambda d, t: get_hanning_kernel1d(5, device=d, dtype=t), (5,)),
    ("get_hanning_kernel2d", lambda d, t: get_hanning_kernel2d((3, 5), device=d, dtype=t), (3, 5)),
    ("get_binary_kernel2d", lambda d, t: get_binary_kernel2d((3, 5), device=d, dtype=t), (15, 1, 3, 5)),
    ("get_sobel_kernel2d", lambda d, t: get_sobel_kernel2d(device=d, dtype=t), (2, 3, 3)),
    ("get_diff_kernel2d", lambda d, t: get_diff_kernel2d(device=d, dtype=t), (2, 3, 3)),
    ("sobel_order2", lambda d, t: get_spatial_gradient_kernel2d("sobel", 2, device=d, dtype=t), (3, 5, 5)),
    ("diff_order2", lambda d, t: get_spatial_gradient_kernel2d("diff", 2, device=d, dtype=t), (3, 3, 3)),
    ("diff3d_order1", lambda d, t: get_spatial_gradient_kernel3d("diff", 1, device=d, dtype=t), (3, 1, 3, 3, 3)),
    ("diff3d_order2", lambda d, t: get_spatial_gradient_kernel3d("diff", 2, device=d, dtype=t), (6, 1, 3, 3, 3)),
    (
        "get_motion_kernel2d",
        lambda d, t: get_motion_kernel2d(5, torch.tensor(30.0, device=d, dtype=t)),
        (1, 5, 5),
    ),
    (
        "get_motion_kernel3d",
        lambda d, t: get_motion_kernel3d(5, torch.tensor([[0.0, 30.0, 0.0]], device=d, dtype=t)),
        (1, 5, 5, 5),
    ),
]

# (name, builder(size, device, dtype), rejected sizes, accepted sizes, the rule kornia states in its message)
_KERNEL_SIZE_RULES = [
    (
        "get_gaussian_kernel1d",
        lambda k, d, t: get_gaussian_kernel1d(k, 1.0, device=d, dtype=t),
        [0, 2, 4],
        [1, 3],
        "an odd integer bigger than 0",
    ),
    (
        "get_gaussian_kernel1d_force_even",
        lambda k, d, t: get_gaussian_kernel1d(k, 1.0, True, device=d, dtype=t),
        [0],
        [1, 2, 4],
        "an even or odd integer bigger than 0",
    ),
    (
        "get_gaussian_erf_kernel1d",
        lambda k, d, t: get_gaussian_erf_kernel1d(k, 1.0, device=d, dtype=t),
        [0, 4],
        [3],
        "an odd integer bigger than 0",
    ),
    (
        "get_gaussian_discrete_kernel1d",
        lambda k, d, t: get_gaussian_discrete_kernel1d(k, 1.0, device=d, dtype=t),
        [0, 4],
        [3],
        "an odd integer bigger than 0",
    ),
    (
        "get_gaussian_kernel2d",
        lambda k, d, t: get_gaussian_kernel2d((3, k), (1.0, 1.0), device=d, dtype=t),
        [0, 4],
        [1, 3],
        "an odd integer bigger than 0",
    ),
    (
        "get_gaussian_kernel3d",
        lambda k, d, t: get_gaussian_kernel3d((3, 5, k), (1.0, 1.0, 1.0), device=d, dtype=t),
        [0, 4],
        [1, 3],
        "an odd integer bigger than 0",
    ),
    (
        "get_laplacian_kernel1d",
        lambda k, d, t: get_laplacian_kernel1d(k, device=d, dtype=t),
        [0, 4],
        [3],  # 1 is rejected too, as the all-zero kernel, with its own message (#5175)
        "an odd integer bigger than 0",
    ),
    (
        "get_laplacian_kernel2d",
        lambda k, d, t: get_laplacian_kernel2d((3, k), device=d, dtype=t),
        [0, 4],
        [1, 3],
        "an odd integer bigger than 0",
    ),
    (
        "get_hanning_kernel1d",
        lambda k, d, t: get_hanning_kernel1d(k, device=d, dtype=t),
        [1, 2],
        [3, 4],
        "an even or odd integer bigger than 2",
    ),
    (
        "get_hanning_kernel2d",
        lambda k, d, t: get_hanning_kernel2d((3, k), device=d, dtype=t),
        [1, 2],
        [3, 4],
        "an even or odd integer bigger than 2",
    ),
    # the motion builders take their device and dtype from a tensor angle; a float angle builds on the CPU
    (
        "get_motion_kernel2d",
        lambda k, d, t: get_motion_kernel2d(k, 0.0),
        [1, 4],
        [3, 5],
        "an odd integer bigger than 2",
    ),
    (
        "get_motion_kernel3d",
        lambda k, d, t: get_motion_kernel3d(k, (0.0, 0.0, 0.0)),
        [1, 4],
        [3, 5],
        "an odd integer bigger than 2",
    ),
]

_UNIT_SUM_KERNELS = [
    ("gaussian", lambda d, t: gaussian(5, 1.5, device=d, dtype=t)),
    ("get_gaussian_kernel1d", lambda d, t: get_gaussian_kernel1d(7, 1.5, device=d, dtype=t)),
    ("get_gaussian_erf_kernel1d", lambda d, t: get_gaussian_erf_kernel1d(7, 1.5, device=d, dtype=t)),
    ("get_gaussian_discrete_kernel1d", lambda d, t: get_gaussian_discrete_kernel1d(7, 1.5, device=d, dtype=t)),
    ("get_gaussian_kernel2d", lambda d, t: get_gaussian_kernel2d((3, 5), (1.0, 1.5), device=d, dtype=t)),
    ("get_gaussian_kernel3d", lambda d, t: get_gaussian_kernel3d((3, 5, 7), (0.7, 1.0, 1.5), device=d, dtype=t)),
    ("get_box_kernel1d", lambda d, t: get_box_kernel1d(4, device=d, dtype=t)),
    ("get_box_kernel2d", lambda d, t: get_box_kernel2d((3, 4), device=d, dtype=t)),
    (
        "get_motion_kernel2d",
        lambda d, t: get_motion_kernel2d(
            5, torch.tensor(30.0, device=d, dtype=t), torch.tensor(0.5, device=d, dtype=t)
        ),
    ),
    (
        "get_motion_kernel3d",
        lambda d, t: get_motion_kernel3d(
            5, torch.tensor([[10.0, 30.0, 20.0]], device=d, dtype=t), torch.tensor([0.5], device=d, dtype=t)
        ),
    ),
]


def _kernel_guard(name: str, device: torch.device, dtype: torch.dtype) -> None:
    """Skip a builder whose kernel cannot be built on this device/dtype."""
    if name == "get_motion_kernel2d" and not _supports_nearest_2d_grid_sample(device, dtype):
        pytest.skip("2D grid_sample (nearest) is unavailable for this device/dtype")
    if name == "get_motion_kernel3d" and not supports_nearest_3d_grid_sample(device, dtype):
        pytest.skip("3D grid_sample (nearest) is unavailable for this device/dtype")


def _grid(*sizes: int, centre: tuple[int, ...], device, dtype) -> tuple[torch.Tensor, ...]:
    """Integer coordinate grids (ij order) shifted so that ``centre`` is the origin."""
    axes = [torch.arange(n) - c for n, c in zip(sizes, centre)]
    return tuple(g.to(device=device, dtype=dtype) for g in torch.meshgrid(*axes, indexing="ij"))


def _correlate_at(kernel: torch.Tensor, field: torch.Tensor, centre: tuple[int, ...]) -> torch.Tensor:
    """Correlate a stack of kernels ``(N, *k)`` with ``field`` at one point, as filter2d / filter3d do."""
    window = tuple(slice(c - k // 2, c + k // 2 + 1) for c, k in zip(centre, kernel.shape[1:]))
    dims = tuple(range(-len(centre), 0))
    return (kernel * field[window]).sum(dims)


class TestConventionsKernels(BaseTester):
    @pytest.mark.parametrize("name, factory, shape", _KERNEL_SHAPES, ids=[case[0] for case in _KERNEL_SHAPES])
    def test_convention_kernel_builder_output_shapes(self, name, factory, shape, device, dtype):
        _kernel_guard(name, device, dtype)
        kernel = factory(device, dtype)
        assert kernel.shape == shape
        assert kernel.dtype == dtype
        assert kernel.device.type == device.type

    @pytest.mark.parametrize(
        "name, build, rejected, accepted, rule", _KERNEL_SIZE_RULES, ids=[case[0] for case in _KERNEL_SIZE_RULES]
    )
    def test_convention_kernel_builder_size_rules(self, name, build, rejected, accepted, rule, device, dtype):
        for size in rejected:
            with pytest.raises(BaseError, match=f"Kernel size must be {rule}\\."):
                build(size, device, dtype)
        for size in accepted:
            build(size, device, dtype)

    @pytest.mark.parametrize("name, factory", _UNIT_SUM_KERNELS, ids=[case[0] for case in _UNIT_SUM_KERNELS])
    def test_convention_smoothing_kernels_sum_to_one(self, name, factory, device, dtype):
        _kernel_guard(name, device, dtype)
        kernel = factory(device, dtype)
        sums = kernel.flatten(1).sum(-1)
        self.assert_close(sums, torch.ones_like(sums))

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("get_gaussian_kernel1d", [0.00962006, 0.2054237, 0.56991249, 0.2054237, 0.00962006]),
            ("get_gaussian_erf_kernel1d", [0.01589041, 0.22154163, 0.52513592, 0.22154163, 0.01589041]),
            ("get_gaussian_discrete_kernel1d", [0.01881815, 0.15514675, 0.6520702, 0.15514675, 0.01881815]),
        ],
    )
    def test_convention_gaussian_kernel1d_variants_sampled_erf_discrete(self, name, expected, device, dtype):
        # kernel_size 5, sigma 0.7, normalised to sum 1 (numpy, scipy 1.17.1, opencv 5.0.0; n = arange(-2, 3)):
        #   sampled:  cv2.getGaussianKernel(5, 0.7), i.e. exp(-n**2 / (2 * 0.7**2))
        #   erf:      the pixel-integrated Gaussian, scipy.special.ndtr((n + 0.5) / 0.7) - ndtr((n - 0.5) / 0.7)
        #   discrete: Lindeberg's discrete Gaussian, scipy.special.ive(abs(n), 0.7**2)
        _kernel_guard(name, device, dtype)
        builders = {
            "get_gaussian_kernel1d": get_gaussian_kernel1d,
            "get_gaussian_erf_kernel1d": get_gaussian_erf_kernel1d,
            "get_gaussian_discrete_kernel1d": get_gaussian_discrete_kernel1d,
        }
        kernel = builders[name](5, 0.7, device=device, dtype=dtype)
        self.assert_close(kernel, torch.tensor([expected], device=device, dtype=dtype))

    def test_convention_gaussian_even_window_centres_at_mean_minus_half(self, device, dtype):
        # The default mean window_size // 2 = 2 puts the centre of a 4-sample window at 1.5: symmetric
        default = gaussian(4, 1.0, device=device, dtype=dtype)
        self.assert_close(default, default.flip(-1))
        self.assert_close(get_gaussian_kernel1d(4, 1.0, force_even=True, device=device, dtype=dtype), default)
        # an explicit mean=1 on an even window centres at 0.5, so samples 0 and 1 weigh the same
        shifted = gaussian(4, 1.0, mean=1.0, device=device, dtype=dtype)
        self.assert_close(shifted[:, 0], shifted[:, 1])
        assert shifted[0, 1] > shifted[0, 2]
        # an odd window centres at the mean itself
        odd = gaussian(5, 1.0, mean=1.0, device=device, dtype=dtype)
        assert int(odd[0].float().argmax()) == 1
        self.assert_close(odd[:, 0], odd[:, 2])

    def test_convention_gaussian_kernel2d_sizes_and_sigmas_are_y_then_x(self, device, dtype):
        kernel = get_gaussian_kernel2d((7, 9), (1.0, 2.0), device=device, dtype=dtype)
        assert kernel.shape == (1, 7, 9)
        along_y = get_gaussian_kernel1d(7, 1.0, device=device, dtype=dtype)[0]
        along_x = get_gaussian_kernel1d(9, 2.0, device=device, dtype=dtype)[0]
        self.assert_close(kernel[0], along_y[:, None] * along_x[None, :])
        # relabel: swapping both tuples transposes the kernel
        swapped = get_gaussian_kernel2d((9, 7), (2.0, 1.0), device=device, dtype=dtype)
        self.assert_close(swapped[0], kernel[0].T)

    def test_convention_gaussian_kernel3d_sizes_and_sigmas_are_z_y_x(self, device, dtype):
        kernel = get_gaussian_kernel3d((3, 5, 7), (0.7, 1.0, 1.5), device=device, dtype=dtype)
        assert kernel.shape == (1, 3, 5, 7)
        along_z = get_gaussian_kernel1d(3, 0.7, device=device, dtype=dtype)[0]
        along_y = get_gaussian_kernel1d(5, 1.0, device=device, dtype=dtype)[0]
        along_x = get_gaussian_kernel1d(7, 1.5, device=device, dtype=dtype)[0]
        self.assert_close(kernel[0], along_z[:, None, None] * along_y[None, :, None] * along_x[None, None, :])

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_convention_gaussian_kernel_batched_sigma_gives_one_kernel_per_row(self, ndim, device, dtype):
        builder = {1: get_gaussian_kernel1d, 2: get_gaussian_kernel2d, 3: get_gaussian_kernel3d}[ndim]
        sizes = (3, 5, 7)[:ndim]
        size_arg = sizes[0] if ndim == 1 else sizes
        sigma = torch.tensor([[0.8, 1.2, 1.6][:ndim], [1.5, 0.6, 1.1][:ndim]], device=device, dtype=dtype)  # (B, ndim)
        kernels = builder(size_arg, sigma)
        assert kernels.shape == (2, *sizes)
        for b in range(2):
            self.assert_close(kernels[b : b + 1], builder(size_arg, sigma[b : b + 1]))

    def test_convention_laplacian_kernels_are_ones_with_a_balancing_negative_centre(self, device, dtype):
        def ones_with_centre(shape, centre):
            out = torch.ones(shape, device=device, dtype=dtype)
            out[tuple(s // 2 for s in shape)] = centre
            return out

        self.assert_close(get_laplacian_kernel1d(3, device=device, dtype=dtype), ones_with_centre((3,), -2.0))
        self.assert_close(get_laplacian_kernel1d(5, device=device, dtype=dtype), ones_with_centre((5,), -4.0))
        kernel = get_laplacian_kernel2d(3, device=device, dtype=dtype)  # the 8-neighbour stencil
        self.assert_close(kernel, ones_with_centre((3, 3), -8.0))
        self.assert_close(get_laplacian_kernel2d((3, 5), device=device, dtype=dtype), ones_with_centre((3, 5), -14.0))
        # the negative centre makes the response positive on a convex field: 3 per unit d2/dx2 or d2/dy2
        ys, xs = _grid(9, 11, centre=(3, 6), device=device, dtype=dtype)
        for field in (xs * xs / 2, ys * ys / 2):
            self.assert_close(
                _correlate_at(kernel[None], field, (3, 6)), torch.tensor([3.0], device=device, dtype=dtype)
            )

    def test_convention_laplacian_1d_puts_an_even_kernels_negative_tap_at_k_half(self, device, dtype):
        # laplacian_1d does not validate the size: an even one is accepted and its centre tap sits at k // 2
        self.assert_close(
            laplacian_1d(4, device=device, dtype=dtype), torch.tensor([1.0, 1.0, -3.0, 1.0], device=device, dtype=dtype)
        )
        expected = torch.tensor([1.0, 1.0, 1.0, -5.0, 1.0, 1.0], device=device, dtype=dtype)
        self.assert_close(laplacian_1d(6, device=device, dtype=dtype), expected)

    def test_convention_binary_kernel2d_is_a_row_major_one_hot_stack(self, device, dtype):
        kernel = get_binary_kernel2d((3, 5), device=device, dtype=dtype)
        # channel i is one-hot at (i // 5, i % 5)
        self.assert_close(kernel, torch.eye(15, device=device, dtype=dtype).view(15, 1, 3, 5))

    @pytest.mark.parametrize(
        "name, scale", [("get_sobel_kernel2d", 8.0), ("sobel", 8.0), ("get_diff_kernel2d", 2.0), ("diff", 2.0)]
    )
    def test_convention_spatial_gradient_kernel2d_stacks_dx_then_dy(self, name, scale, device, dtype):
        builders = {"get_sobel_kernel2d": get_sobel_kernel2d, "get_diff_kernel2d": get_diff_kernel2d}
        if name in builders:
            kernel = builders[name](device=device, dtype=dtype)
        else:
            kernel = get_spatial_gradient_kernel2d(name, 1, device=device, dtype=dtype)
        ys, xs = _grid(9, 11, centre=(3, 6), device=device, dtype=dtype)
        field = 2 * xs + 3 * ys  # d/dx = 2, d/dy = 3
        # channel 0 estimates d/dx and channel 1 d/dy, positive for values increasing with column / row, in raw
        # units: Sobel 8 and diff 2 per unit slope
        expected = torch.tensor([2 * scale, 3 * scale], device=device, dtype=dtype)
        self.assert_close(_correlate_at(kernel, field, (3, 6)), expected)
        # relabel: transposing the field swaps the channels
        self.assert_close(_correlate_at(kernel, field.T, (6, 3)), expected.flip(0))

    @pytest.mark.parametrize("mode, scales", [("sobel", (64.0, 64.0, 64.0)), ("diff", (1.0, 4.0, 1.0))])
    def test_convention_spatial_gradient_kernel2d_order2_stacks_dxx_dxy_dyy(self, mode, scales, device, dtype):
        kernel = get_spatial_gradient_kernel2d(mode, 2, device=device, dtype=dtype)
        ys, xs = _grid(9, 11, centre=(3, 6), device=device, dtype=dtype)
        # unit d2/dx2, d2/dxdy and d2/dy2 fields; row f of the response holds the three channels' answers to field f
        fields = (xs * xs / 2, xs * ys, ys * ys / 2)
        response = torch.stack([_correlate_at(kernel, field, (3, 6)) for field in fields])
        self.assert_close(response, torch.diag(torch.tensor(scales, device=device, dtype=dtype)))

    def test_convention_spatial_gradient_kernel3d_is_diff_only_in_derivative_units(self, device, dtype):
        zs, ys, xs = _grid(7, 9, 11, centre=(3, 4, 6), device=device, dtype=dtype)  # D != H != W
        first = get_spatial_gradient_kernel3d("diff", 1, device=device, dtype=dtype)
        assert first.shape == (3, 1, 3, 3, 3)
        # (d/dx, d/dy, d/dz) in unit slope
        response = _correlate_at(first[:, 0], xs + 2 * ys + 3 * zs, (3, 4, 6))
        self.assert_close(response, torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype))
        second = get_spatial_gradient_kernel3d("diff", 2, device=device, dtype=dtype)
        assert second.shape == (6, 1, 3, 3, 3)
        # (dxx, dyy, dzz, dxy, dyz, dxz), each in unit curvature
        fields = (xs * xs / 2, ys * ys / 2, zs * zs / 2, xs * ys, ys * zs, xs * zs)
        response = torch.stack([_correlate_at(second[:, 0], field, (3, 4, 6)) for field in fields])
        self.assert_close(response, torch.eye(6, device=device, dtype=dtype))
        with pytest.raises(NotImplementedError):
            get_spatial_gradient_kernel3d("sobel", 1, device=device, dtype=dtype)

    def test_convention_motion_kernel2d_angle_is_degrees_counterclockwise(self, device, dtype):
        _kernel_guard("get_motion_kernel2d", device, dtype)

        def heaviest(angle: float) -> tuple[int, int]:
            angle_t = torch.tensor(angle, device=device, dtype=dtype)
            kernel = get_motion_kernel2d(5, angle_t, torch.tensor(1.0, device=device, dtype=dtype))[0]
            return divmod(int(kernel.float().flatten().argmax()), 5)

        # direction=+1 weighs the unrotated line's left end most
        assert heaviest(0.0) == (2, 0)
        # +90 degrees turns it counter-clockwise as displayed (row 0 at the top): the left end goes to the bottom
        assert heaviest(90.0) == (4, 2)
        assert heaviest(-90.0) == (0, 2)
        assert heaviest(30.0) == (3, 0)

    def test_convention_motion_kernel2d_direction_weights_the_back_end(self, device, dtype):
        _kernel_guard("get_motion_kernel2d", device, dtype)

        def kernel(direction: float) -> torch.Tensor:
            angle = torch.tensor(0.0, device=device, dtype=dtype)
            return get_motion_kernel2d(5, angle, torch.tensor(direction, device=device, dtype=dtype))[0]

        # direction=+1 is a linear ramp down from the left end whose far tap is 0: only k - 1 taps weigh
        forward = torch.zeros(5, 5, device=device, dtype=dtype)
        forward[2] = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], device=device, dtype=dtype)
        self.assert_close(kernel(1.0), forward)
        self.assert_close(kernel(-1.0), forward.flip(-1))
        uniform = torch.zeros(5, 5, device=device, dtype=dtype)
        uniform[2] = 0.2
        self.assert_close(kernel(0.0), uniform)
        # in between, the ramp's slope is linear in direction: 0.5 weighs the taps 0.3 down to 0.1
        half = torch.zeros(5, 5, device=device, dtype=dtype)
        half[2] = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1], device=device, dtype=dtype)
        self.assert_close(kernel(0.5), half)
        self.assert_close(kernel(-0.5), half.flip(-1))
        # direction is clamped to [-1, 1], at both ends
        self.assert_close(kernel(5.0), forward)
        self.assert_close(kernel(-5.0), forward.flip(-1))

    def test_convention_motion_kernel2d_batched_angle_needs_a_matching_direction(self, device, dtype):
        _kernel_guard("get_motion_kernel2d", device, dtype)
        angles = torch.tensor([0.0, 30.0, 90.0], device=device, dtype=dtype)
        directions = torch.tensor([1.0, -0.5, 0.0], device=device, dtype=dtype)
        kernels = get_motion_kernel2d(5, angles, directions)
        assert kernels.shape == (3, 5, 5)
        assert kernels.dtype == dtype
        for b in range(3):
            self.assert_close(kernels[b : b + 1], get_motion_kernel2d(5, angles[b], directions[b]))
        with pytest.raises(BaseError, match="direction and angle must have the same length"):
            get_motion_kernel2d(5, angles, 1.0)

    def test_convention_motion_kernel2d_nearest_drops_off_axis_taps(self, device, dtype):
        _kernel_guard("get_motion_kernel2d", device, dtype)
        angle = torch.tensor(45.0, device=device, dtype=dtype)
        direction = torch.tensor(0.0, device=device, dtype=dtype)
        nearest = get_motion_kernel2d(5, angle, direction, mode="nearest")[0]
        # at 45 degrees nearest resampling keeps 3 of the 5 taps of the uniform line, each 1/3
        kept = nearest[nearest > 0]
        assert kept.numel() == 3
        self.assert_close(kept, torch.full((3,), 1 / 3, device=device, dtype=dtype))
        # bilinear resampling spreads the weight off the line instead
        bilinear = get_motion_kernel2d(5, angle, direction, mode="bilinear")[0]
        assert int((bilinear > 0).sum()) > 5

    def test_convention_motion_kernel3d_angle_is_an_axis_angle_vector(self, device, dtype):
        _kernel_guard("get_motion_kernel3d", device, dtype)

        def kernel(angles: tuple[float, float, float]) -> torch.Tensor:
            angle_t = torch.tensor([angles], device=device, dtype=dtype)
            return get_motion_kernel3d(5, angle_t, torch.tensor([1.0], device=device, dtype=dtype))[0]

        def heaviest(angles: tuple[float, float, float]) -> tuple[int, int, int]:
            index = int(kernel(angles).float().flatten().argmax())
            return (index // 25, index // 5 % 5, index % 5)  # (d, h, w)

        # direction=+1 weighs the unrotated line's -x end most
        assert heaviest((0.0, 0.0, 0.0)) == (2, 2, 0)
        # positive pitch (about y) sends that end to +z
        assert heaviest((0.0, 90.0, 0.0)) == (4, 2, 2)
        # positive roll (about z) sends it to -y: clockwise as displayed, opposite to get_motion_kernel2d's angle
        assert heaviest((0.0, 0.0, 90.0)) == (2, 0, 2)
        # the line lies along x, so yaw (about x) alone leaves the kernel unchanged
        self.assert_close(kernel((90.0, 0.0, 0.0)), kernel((0.0, 0.0, 0.0)))

        # Rodrigues' formula for (90, 90, 0): rotate through 90 * sqrt(2) degrees about (1, 1, 0) / sqrt(2).
        # The heavy-end direction (-1, 0, 0) becomes (-0.19715, -0.80285, 0.56264), unlike either Euler
        # composition (0, 0, 1) or (0, -1, 0). Nearest resampling keeps the three central line taps, whose weights
        # on the size-5 line are 0.1, 0.2 and 0.3, renormalised below in (depth, row, column) order.
        expected = torch.zeros(5, 5, 5, device=device, dtype=dtype)
        expected[1, 3, 2] = 1 / 6
        expected[2, 2, 2] = 1 / 3
        expected[3, 1, 2] = 1 / 2
        self.assert_close(kernel((90.0, 90.0, 0.0)), expected)

    def test_wart_gaussian_discrete_kernel1d_tap_count_is_not_kernel_size_5158(self, device, dtype):
        """get_gaussian_discrete_kernel1d gives 3 taps for kernel_size=1 and k + 1 for an even force_even k (#5158)."""
        assert get_gaussian_discrete_kernel1d(1, 1.0, device=device, dtype=dtype).shape == (1, 3)
        even = get_gaussian_discrete_kernel1d(4, 1.0, force_even=True, device=device, dtype=dtype)
        assert even.shape == (1, 5)

    def test_wart_gaussian_erf_kernel1d_even_kernel_peaks_at_k_half_5158(self, device, dtype):
        """get_gaussian_erf_kernel1d(force_even=True) samples about k // 2, so an even kernel is off-centre (#5158)."""
        kernel = get_gaussian_erf_kernel1d(4, 1.0, force_even=True, device=device, dtype=dtype)[0]
        # the sampled get_gaussian_kernel1d with the same arguments is symmetric about (k - 1) / 2 = 1.5
        assert int(kernel.float().argmax()) == 2
        assert kernel[2] > kernel[1]

    @pytest.mark.parametrize(
        "sigma, expected",
        [
            (1.0, [0.05088223571, 0.2118383232, 0.4745588821, 0.2118383232, 0.05088223571]),
            (7.0, [0.1958895744, 0.2020423257, 0.2041361999, 0.2020423257, 0.1958895744]),
            (20.0, [0.1994995631, 0.2002500302, 0.2005008133, 0.2002500302, 0.1994995631]),
        ],
    )
    def test_convention_gaussian_discrete_kernel1d_is_finite_for_a_large_sigma_5227(
        self, sigma, expected, device, dtype
    ):
        """get_gaussian_discrete_kernel1d scales its Bessel terms by exp(-sigma**2), so it does not overflow (#5227)."""
        # Unscaled terms overflow to an all-NaN kernel from sigma about 6.8 in float32 and bfloat16 and about 19 in
        # float64, and in float16 for every sigma > 0 once kernel_size is 5 or more. Reference (scipy 1.17.1):
        # scipy.special.ive(abs(n), sigma**2) for n = arange(-2, 3), normalized.
        kernel = get_gaussian_discrete_kernel1d(5, sigma, device=device, dtype=dtype)
        self.assert_close(kernel, torch.tensor([expected], device=device, dtype=dtype))

    def test_convention_gaussian_discrete_kernel1d_is_unimodal_for_a_large_sigma_5227(self, device, dtype):
        """The discrete kernel follows the discrete Gaussian at sigma 14, where float64 used to drift (#5227)."""
        kernel = get_gaussian_discrete_kernel1d(85, 14.0, device=device, dtype=dtype)[0]
        # scipy.special.ive(abs(n), 14.0**2) normalized (scipy 1.17.1) at taps 38, 39, 42 (n = -4, -3, 0). A Miller
        # recurrence started too low for sigma**2 makes tap 39 dip below tap 38 in float64.
        expected = torch.tensor([0.02743744108, 0.02793304819, 0.02858345732], device=device, dtype=dtype)
        self.assert_close(kernel[[38, 39, 42]], expected)
        assert kernel[39] > kernel[38]

    @pytest.mark.parametrize(
        "builder", [get_gaussian_kernel1d, get_gaussian_erf_kernel1d, get_gaussian_discrete_kernel1d]
    )
    def test_wart_gaussian_kernel1d_rejects_a_python_int_sigma_5157(self, builder, device, dtype):
        """The 1d Gaussian builders raise for sigma=1, where the 2d builder accepts sigma=(1, 1) (#5157)."""
        with pytest.raises((BaseError, AttributeError)):
            builder(5, 1, device=device, dtype=dtype)
        assert builder(5, 1.0, device=device, dtype=dtype).shape == (1, 5)
        assert get_gaussian_kernel2d((5, 5), (1, 1), device=device, dtype=dtype).shape == (1, 5, 5)

    @pytest.mark.parametrize("case", ["box_int32", "gaussian_uint8", "laplacian_uint8", "gradient3d_int32"])
    def test_wart_kernel_builders_truncate_or_wrap_in_an_integer_dtype_5155(self, case, device):
        """An integer dtype truncates fractional taps to 0, and uint8 wraps the negative ones (#5155)."""
        if case == "box_int32":
            assert bool((get_box_kernel1d(3, device=device, dtype=torch.int32) == 0).all())
        elif case == "gaussian_uint8":
            # the offsets -2 and -1 wrap to 254 and 255, so the taps before the centre vanish
            kernel = get_gaussian_kernel1d(5, 1.5, device=device, dtype=torch.uint8)[0]
            assert kernel[:2].tolist() == [0, 0]
            assert bool((kernel[2:] > 0).all())
        elif case == "laplacian_uint8":
            assert get_laplacian_kernel1d(5, device=device, dtype=torch.uint8).tolist() == [1, 1, 252, 1, 1]
        else:
            # the first-order taps are +-0.5
            assert bool((get_spatial_gradient_kernel3d("diff", 1, device=device, dtype=torch.int32) == 0).all())

    def test_wart_motion_kernel2d_nearest_ties_change_with_a_full_turn_5181(self):
        """At a sampling-tie angle roundoff picks the tap: 30 and -330 degrees build other kernels (#5181)."""
        # a float angle builds the kernel on the CPU in float32, whatever the test device
        difference = get_motion_kernel2d(5, 30.0, 1.0) - get_motion_kernel2d(5, -330.0, 1.0)
        assert float(difference.abs().max()) > 0.1

    def test_wart_motion_kernel2d_on_mps_differs_from_cpu_for_some_angles_5181(self, device, dtype):
        """A tensor angle builds get_motion_kernel2d on its own device, and MPS gives other kernels (#5181)."""
        if device.type != "mps":
            pytest.skip("#5181 compares the kernel built on MPS with the one built on the CPU")
        _kernel_guard("get_motion_kernel2d", torch.device("cpu"), dtype)
        # every whole degree, including 120 and 210
        angles = torch.arange(0.0, 360.0, 1.0, dtype=dtype)
        directions = torch.full_like(angles, 0.3)
        on_cpu = get_motion_kernel2d(7, angles, directions)
        on_mps = get_motion_kernel2d(7, angles.to(device), directions.to(device)).cpu()
        # a sampling tie rounded the other way moves a tap weight (about 0.18), far beyond half-precision roundoff
        differing = (on_mps.float() - on_cpu.float()).abs().flatten(1).amax(1) > 0.05
        if not bool(differing.any()):
            pytest.skip("this MPS backend rounds the sampling ties like the CPU (#5181)")
        # a Python-float angle builds the kernel on the CPU in float32, equal to the CPU tensor-angle kernel
        on_cpu_f32 = get_motion_kernel2d(7, angles.float(), directions.float())
        for index in differing.nonzero().flatten().tolist():
            from_float = get_motion_kernel2d(7, float(angles[index]), float(directions[index]))
            assert from_float.device.type == "cpu"
            self.assert_close(from_float[0], on_cpu_f32[index])

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_wart_box_kernel_is_a_stride_zero_view_5160(self, ndim, device, dtype):
        """get_box_kernel1d/2d return an expanded view of one scalar, so writing one tap rewrites all (#5160)."""
        if ndim == 1:
            kernel = get_box_kernel1d(3, device=device, dtype=dtype)
        else:
            kernel = get_box_kernel2d((3, 4), device=device, dtype=dtype)
        kernel[(0,) * kernel.dim()] = 0.0
        assert bool((kernel == 0).all())
