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

import importlib

import pytest
import torch
import torch.nn.functional as F

from kornia.core._compat import torch_version
from kornia.filters import Sobel, SpatialGradient, SpatialGradient3d, sobel, spatial_gradient, spatial_gradient3d
from kornia.filters.kernels import get_spatial_gradient_kernel2d, normalize_kernel2d

from testing.base import BaseTester, supports_replicate_padding

sobel_module = importlib.import_module("kornia.filters.sobel")


class TestSpatialGradient(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("mode", ["sobel", "diff"])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_smoke(self, batch_size, mode, order, normalized, device, dtype):
        data = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)
        actual = SpatialGradient(mode, order, normalized)(data)
        assert isinstance(actual, torch.Tensor)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)
        assert SpatialGradient()(inp).shape == (batch_size, 3, 2, 4, 4)

    def test_exception(self):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            spatial_gradient(1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            spatial_gradient(torch.zeros(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

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

    @pytest.mark.parametrize("normalized", [True, False])
    def test_sobel_preserves_nonfinite_kernel_behavior(self, normalized, device, dtype):
        inp = torch.zeros(1, 1, 3, 3, device=device, dtype=dtype)
        inp[..., 0, 0] = float("nan")
        inp[..., 1, 1] = float("inf")

        kernel = get_spatial_gradient_kernel2d("sobel", 1, device=device, dtype=dtype)
        if normalized:
            kernel = normalize_kernel2d(kernel)
        expected = F.conv2d(F.pad(inp, [1, 1, 1, 1], "replicate"), kernel[:, None])
        actual = spatial_gradient(inp, normalized=normalized).reshape_as(expected)

        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        assert torch.equal(torch.isinf(actual), torch.isinf(expected))
        self.assert_close(torch.nan_to_num(actual), torch.nan_to_num(expected))

    @pytest.mark.parametrize("normalized", [True, False])
    def test_convention_upper_case_sobel_takes_the_fixed_kernel_path_5156(self, normalized, monkeypatch, device, dtype):
        """mode='Sobel' at order=1 uses the fixed kernel of 'sobel' and does not call the kernel builder (#5156)."""

        def should_not_run(*args, **kwargs):
            raise AssertionError("order=1 Sobel on a floating input must use the fixed kernel")

        monkeypatch.setattr(sobel_module, "get_spatial_gradient_kernel2d", should_not_run)
        img = torch.rand(1, 2, 5, 7, device=device, dtype=dtype)
        expected = spatial_gradient(img, "sobel", 1, normalized)
        assert torch.equal(spatial_gradient(img, "Sobel", 1, normalized), expected)

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

        edges = spatial_gradient(inp, "diff", normalized=False)
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

        edges = spatial_gradient(inp, "diff", normalized=True)
        self.assert_close(edges, expected)

    @pytest.mark.parametrize("mode", ["sobel", "diff"])
    def test_second_order_quadratics(self, mode, device, dtype):
        coords = torch.arange(7, device=device, dtype=dtype)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        # one image per quadratic surface: x^2, x*y, y^2
        inp = torch.stack([x * x, x * y, y * y])[:, None]
        # the unnormalized (dxx, dxy, dyy) response of each surface is constant away from the border
        if mode == "sobel":
            expected = torch.tensor(
                [[128.0, 0.0, 0.0], [0.0, 64.0, 0.0], [0.0, 0.0, 128.0]], device=device, dtype=dtype
            )
        else:
            expected = torch.tensor([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 2.0]], device=device, dtype=dtype)

        actual = spatial_gradient(inp, mode, order=2, normalized=False)[:, 0, :, 2:-2, 2:-2]
        self.assert_close(actual, expected[..., None, None].expand_as(actual))

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = spatial_gradient(inp)

        assert inp.is_contiguous() is False
        assert actual.is_contiguous()
        assert actual.shape == (3, 3, 2, 5, 5)

    @pytest.mark.parametrize("mode", ["sobel", "diff"])
    def test_second_order_normalized_quadratics(self, mode, device, dtype):
        # With normalized=True every second order channel estimates the derivative itself, so on a quadratic
        # surface the channels are the exact second derivatives and dxx * dyy - dxy**2 is the exact determinant
        # of the Hessian. Before, dxy had a scale of its own.
        coords = torch.arange(9, device=device, dtype=dtype)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        surfaces = torch.stack([x * x, x * y, y * y, x * x + y * y, (x + y) ** 2 / 2])[:, None]
        out = spatial_gradient(surfaces, mode, order=2, normalized=True)[..., 3:-3, 3:-3]

        expected = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [1.0, 1.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out, expected[:, None, :, None, None].expand_as(out))

        det = out[:, :, 0] * out[:, :, 2] - out[:, :, 1] ** 2
        expected_det = torch.tensor([0.0, -1.0, 0.0, 4.0, 0.0], device=device, dtype=dtype)
        self.assert_close(det, expected_det[:, None, None, None].expand_as(det))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(spatial_gradient, (img,))

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = spatial_gradient
        op_module = SpatialGradient()
        expected = op(img)
        actual = op_module(img)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("mode", ["sobel", "diff"])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.xfail(torch_version() in {"2.0.1"}, reason="random failing")
    def test_dynamo(self, batch_size, order, mode, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        if order == 1 and dtype == torch.float64:
            # TODO: FIX order 1 spatial gradient with fp64 on dynamo
            pytest.xfail(reason="Order 1 on spatial gradient may be wrong computed for float64 on dynamo")
        op = SpatialGradient(mode, order)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


class TestSpatialGradient3d(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("mode", ["diff"])  # TODO: add support to 'sobel'
    @pytest.mark.parametrize("order", [1, 2])
    def test_smoke(self, batch_size, mode, order, device, dtype):
        data = torch.ones(batch_size, 3, 2, 7, 4, device=device, dtype=dtype)
        actual = SpatialGradient3d(mode, order)(data)
        assert isinstance(actual, torch.Tensor)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 2, 4, 5, 6, device=device, dtype=dtype)
        sobel = SpatialGradient3d()
        assert sobel(inp).shape == (batch_size, 2, 3, 4, 5, 6)

    def test_exception(self):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            spatial_gradient3d(1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            spatial_gradient3d(torch.zeros(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

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

    def test_second_order_quadratics(self, device, dtype):
        coords = torch.arange(5, device=device, dtype=dtype)
        z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
        # one volume per quadratic surface, in the order of the output channels (dxx, dyy, dzz, dxy, dyz, dxz)
        inp = torch.stack([x * x, y * y, z * z, x * y, y * z, x * z])[:, None]
        expected = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], device=device, dtype=dtype))

        actual = spatial_gradient3d(inp, "diff", order=2)[:, 0, :, 1:-1, 1:-1, 1:-1]
        self.assert_close(actual, expected[..., None, None, None].expand_as(actual))

    def test_convention_upper_case_diff_takes_the_slicing_path_5156(self, monkeypatch, device, dtype):
        """mode='Diff' at order=1 takes the slicing path of 'diff' and does not call the kernel builder (#5156)."""

        def should_not_run(*args, **kwargs):
            raise AssertionError("order=1 diff must use the slicing path")

        monkeypatch.setattr(sobel_module, "get_spatial_gradient_kernel3d", should_not_run)
        img = torch.rand(1, 2, 3, 5, 7, device=device, dtype=dtype)
        expected = spatial_gradient3d(img, "diff", 1)
        assert torch.equal(spatial_gradient3d(img, "Diff", 1), expected)

    def test_gradcheck(self, device):
        img = torch.rand(1, 1, 1, 3, 4, device=device, dtype=torch.float64)
        fast_mode = "cpu" in str(device)  # disable fast mode on gpu
        self.gradcheck(spatial_gradient3d, (img,), fast_mode=fast_mode)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 1, 4, 5, device=device, dtype=dtype)
        op = spatial_gradient3d
        op_module = SpatialGradient3d()
        expected = op(img)
        actual = op_module(img)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("mode", ["diff"])
    @pytest.mark.parametrize("order", [1, 2])
    def test_dynamo(self, mode, order, device, dtype, torch_optimizer):
        data = torch.ones(1, 3, 1, 10, 10, device=device, dtype=dtype)
        op = SpatialGradient3d(mode, order)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


class TestSobel(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_smoke(self, batch_size, normalized, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 7, device=device, dtype=dtype)
        actual = Sobel()(inp)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == (batch_size, 3, 4, 7)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 7, device=device, dtype=dtype)
        assert Sobel()(inp).shape == (batch_size, 3, 4, 7)

    def test_exception(self):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            sobel(1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            sobel(torch.zeros(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

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

    @pytest.mark.parametrize("normalized", [True, False])
    def test_gradcheck(self, normalized, device):
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(sobel, (img, normalized))

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = sobel
        op_module = Sobel()
        expected = op(img)
        actual = op_module(img)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        if dtype == torch.float64:
            # TODO: investigate sobel for float64 with dynamo
            pytest.xfail(reason="The sobel results can be different after dynamo on fp64")
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = Sobel()
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


def _supports_replicate_padding_3d(device, dtype):
    # torch 2.5.1 has no float16 CPU replication_pad3d, like the 2-D kernel that supports_replicate_padding probes
    try:
        F.pad(torch.zeros(1, 1, 2, 2, 2, device=device, dtype=dtype), (1, 1, 1, 1, 1, 1), mode="replicate")
    except RuntimeError as err:
        if "not implemented for" in str(err):
            return False
        raise
    return True


class TestConventionsSpatialGradient(BaseTester):
    @staticmethod
    def _require_replicate_padding(device, dtype, three_d=False):
        if not supports_replicate_padding(device, dtype) or (
            three_d and not _supports_replicate_padding_3d(device, dtype)
        ):
            pytest.skip("torch has no replicate padding kernel for this device and dtype")

    @staticmethod
    def _grid(h, w, device, dtype):
        return torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij"
        )

    def test_convention_spatial_gradient_normalized_ramp(self, device, dtype):
        self._require_replicate_padding(device, dtype)
        # H != W; an x-ramp of slope 3 and a y-ramp of slope 2, read at an interior pixel off the diagonal
        ys, xs = self._grid(6, 9, device, dtype)
        x_ramp = (3 * xs + 1)[None, None]
        y_ramp = (2 * ys + 1)[None, None]
        raw_scale = {"sobel": 8.0, "diff": 2.0}
        for mode in ("sobel", "diff"):
            # the default (normalized=True) is in derivative units; channel 0 = d/dx along W, 1 = d/dy along H (y down)
            gx = spatial_gradient(x_ramp, mode)
            gy = spatial_gradient(y_ramp, mode)
            assert gx.shape == (1, 1, 2, 6, 9)
            self.assert_close(gx[0, 0, :, 2, 5], torch.tensor([3.0, 0.0], device=device, dtype=dtype))
            self.assert_close(gy[0, 0, :, 2, 5], torch.tensor([0.0, 2.0], device=device, dtype=dtype))
            # normalized=False returns the raw kernel response: 8x for Sobel (as cv2.Sobel), 2x for diff
            raw = spatial_gradient(x_ramp, mode, normalized=False)[0, 0, :, 2, 5]
            self.assert_close(raw, torch.tensor([3.0 * raw_scale[mode], 0.0], device=device, dtype=dtype))
            # relabel: transposing the image swaps the two channels and transposes each
            gxt = spatial_gradient(x_ramp.transpose(-1, -2), mode)
            self.assert_close(gxt[:, :, [1, 0]], gx.transpose(-1, -2))

    def test_convention_spatial_gradient_second_order_channels(self, device, dtype):
        self._require_replicate_padding(device, dtype)
        # quadratics centred off the probe pixel (3, 4) of a 7x10 image; channels are (dxx, dxy, dyy)
        ys, xs = self._grid(7, 10, device, dtype)
        x, y = xs - 4, ys - 4
        surfaces = torch.stack([x * x / 2, x * y, y * y / 2, x * x / 2 + 3 * x * y - y * y])[:, None]
        exact = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 3.0, -2.0]])
        raw_scale = {"sobel": torch.tensor([64.0, 64.0, 64.0]), "diff": torch.tensor([1.0, 4.0, 1.0])}
        for mode in ("sobel", "diff"):
            out = spatial_gradient(surfaces, mode, order=2)
            self.assert_close(out[:, 0, :, 3, 4], exact.to(device, dtype))
            raw = spatial_gradient(surfaces, mode, order=2, normalized=False)[:, 0, :, 3, 4]
            self.assert_close(raw, (exact * raw_scale[mode]).to(device, dtype))
            # relabel: transposing swaps dxx and dyy, dxy stays
            out_t = spatial_gradient(surfaces.transpose(-1, -2), mode, order=2)
            self.assert_close(out_t[:, :, [2, 1, 0]], out.transpose(-1, -2))

    def test_convention_spatial_gradient_border_is_replicate(self, device, dtype):
        self._require_replicate_padding(device, dtype, three_d=True)
        # the border is replicated (not the reflect default of filter2d / laplacian), so a ramp's derivative at the
        # first and last column is half its slope; reflect would give 0 there
        ys, xs = self._grid(6, 9, device, dtype)
        x_ramp = (3 * xs + 1)[None, None]
        y_ramp = (2 * ys + 1)[None, None]
        for mode in ("sobel", "diff"):
            gx = spatial_gradient(x_ramp, mode)[0, 0, 0, 2]
            self.assert_close(gx[[0, 4, 8]], torch.tensor([1.5, 3.0, 1.5], device=device, dtype=dtype))
            gy = spatial_gradient(y_ramp, mode)[0, 0, 1, :, 4]
            self.assert_close(gy[[0, 3, 5]], torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype))
        x_ramp3d = (3 * torch.arange(7, device=device, dtype=dtype) + 1).expand(1, 1, 5, 6, 7)
        g3 = spatial_gradient3d(x_ramp3d)[0, 0, 0, 2, 3]
        self.assert_close(g3[[0, 3, 6]], torch.tensor([1.5, 3.0, 1.5], device=device, dtype=dtype))
        z_ramp3d = (5 * torch.arange(5, device=device, dtype=dtype) + 1).view(1, 1, 5, 1, 1).expand(1, 1, 5, 6, 7)
        gz = spatial_gradient3d(z_ramp3d)[0, 0, 2, :, 3, 4]
        self.assert_close(gz[[0, 2, 4]], torch.tensor([2.5, 5.0, 2.5], device=device, dtype=dtype))

    def test_convention_spatial_gradient3d_channel_order(self, device, dtype):
        self._require_replicate_padding(device, dtype, three_d=True)
        # D != H != W (5, 6, 7); first order (dx, dy, dz) along (W, H, D), always in derivative units
        zs, ys, xs = torch.meshgrid(
            torch.arange(5, device=device, dtype=dtype),
            torch.arange(6, device=device, dtype=dtype),
            torch.arange(7, device=device, dtype=dtype),
            indexing="ij",
        )
        ramps = torch.stack([3 * xs + 1, 2 * ys + 1, 5 * zs + 1])[:, None]
        first = spatial_gradient3d(ramps)
        assert first.shape == (3, 1, 3, 5, 6, 7)
        expected = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]], device=device, dtype=dtype)
        self.assert_close(first[:, 0, :, 2, 3, 4], expected)
        # relabel: swapping D and W swaps dx and dz
        first_t = spatial_gradient3d(ramps.transpose(-1, -3))
        self.assert_close(first_t[:, :, [2, 1, 0]], first.transpose(-1, -3))
        # second order: pure terms first, (dxx, dyy, dzz, dxy, dyz, dxz), unlike the 2-D (dxx, dxy, dyy)
        x, y, z = xs - 4, ys - 2, zs - 1
        quadratics = torch.stack([x * x / 2, y * y / 2, z * z / 2, x * y, y * z, x * z])[:, None]
        second = spatial_gradient3d(quadratics, order=2)[:, 0, :, 2, 3, 4]
        self.assert_close(second, torch.eye(6, device=device, dtype=dtype))

    def test_convention_spatial_gradient3d_sobel_mode_raises(self, device, dtype):
        # 'diff' is the default and the only implemented 3-D mode
        volume = torch.rand(1, 1, 5, 6, 7, device=device, dtype=dtype)
        with pytest.raises(NotImplementedError):
            spatial_gradient3d(volume, mode="sobel")
        with pytest.raises(NotImplementedError):
            SpatialGradient3d(mode="sobel")

    def test_convention_sobel_magnitude_of_normalized_gradient_eps_inside_root(self, device, dtype):
        self._require_replicate_padding(device, dtype)
        # sobel = sqrt(gx^2 + gy^2 + eps) of the normalized gradient: slope (3, 2) -> sqrt(13), not 8 * sqrt(13)
        ys, xs = self._grid(6, 9, device, dtype)
        plane = (3 * xs + 2 * ys + 1)[None, None]
        interior = (..., slice(1, -1), slice(1, -1))
        self.assert_close(sobel(plane)[interior], torch.full((1, 1, 4, 7), 13.0**0.5, device=device, dtype=dtype))
        self.assert_close(
            sobel(plane, normalized=False)[interior],
            torch.full((1, 1, 4, 7), 8 * 13.0**0.5, device=device, dtype=dtype),
        )
        # eps sits inside the root: a flat image returns sqrt(eps), not 0
        flat = torch.full((1, 1, 6, 9), 0.3, device=device, dtype=dtype)
        self.assert_close(sobel(flat), torch.full_like(flat, 1e-3))
        self.assert_close(sobel(flat, eps=0.0), torch.zeros_like(flat))

    def test_convention_spatial_gradient_mode_is_case_insensitive_5156(self, device, dtype):
        """mode is case-insensitive: 'Sobel' and 'DIFF' give the results of 'sobel' and 'diff'."""
        self._require_replicate_padding(device, dtype, three_d=True)
        generator = torch.Generator().manual_seed(0)
        img = torch.rand(1, 1, 6, 9, generator=generator).to(device=device, dtype=dtype)
        for order in (1, 2):
            sobel_out = spatial_gradient(img, "sobel", order)
            diff_out = spatial_gradient(img, "diff", order)
            assert not torch.allclose(sobel_out, diff_out)  # the spelling decides which operator runs
            self.assert_close(spatial_gradient(img, "Sobel", order), sobel_out)
            self.assert_close(spatial_gradient(img, "DIFF", order), diff_out)
        volume = torch.rand(1, 1, 5, 6, 7, generator=generator).to(device=device, dtype=dtype)
        for order in (1, 2):
            self.assert_close(spatial_gradient3d(volume, "Diff", order), spatial_gradient3d(volume, "diff", order))
        self.assert_close(SpatialGradient3d("Diff")(volume), spatial_gradient3d(volume))
