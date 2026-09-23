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

from typing import Tuple

import pytest
import torch

from kornia import enhance
from kornia.core._compat import torch_version_ge
from kornia.geometry import rotate

from testing.base import BaseTester


def _sync(device) -> None:
    # MPS dispatches asynchronously, so a kernel error raised by the forward under test would
    # otherwise surface inside an unrelated later test.
    if device.type == "mps":
        torch.mps.synchronize()


class TestEqualization(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 1, 10, 20
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        res = enhance.equalize_clahe(img)
        assert isinstance(res, torch.Tensor)
        assert res.shape == img.shape
        assert res.device == img.device
        assert res.dtype == img.dtype

    @pytest.mark.parametrize(
        ("B", "C", "grid_size"),
        [
            (None, 1, (2, 2)),
            (None, 3, (2, 2)),
            (1, 1, (2, 2)),
            (1, 3, (2, 2)),
            (4, 1, (2, 2)),
            (4, 3, (2, 2)),
            (2, 2, (8, 8)),
            (None, 1, (2, 4)),
            (1, 3, (4, 2)),
            (4, 1, (1, 3)),
            (2, 2, (3, 1)),
        ],
    )
    def test_cardinality(self, B, C, grid_size, device, dtype):
        H, W = 10, 20
        if B is None:
            img = torch.rand(C, H, W, device=device, dtype=dtype)
        else:
            img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        res = enhance.equalize_clahe(img, grid_size=grid_size)
        assert res.shape == img.shape

    @pytest.mark.parametrize("clip, grid", [(0.0, None), (None, (2, 2)), (2.0, (2, 2))])
    def test_optional_params(self, clip, grid, device, dtype):
        C, H, W = 1, 10, 20
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        if clip is None:
            res = enhance.equalize_clahe(img, grid_size=grid)
        elif grid is None:
            res = enhance.equalize_clahe(img, clip_limit=clip)
        else:
            res = enhance.equalize_clahe(img, clip, grid)
        assert isinstance(res, torch.Tensor)
        assert res.shape == img.shape

    @pytest.mark.parametrize(
        "B, clip, grid, exception_type, expected_error_msg",
        [
            (0, 1.0, (2, 2), ValueError, "Invalid input tensor, it is empty."),  # from perform_keep_shape_image
            (1, 1, (2, 2), TypeError, "Input clip_limit type is not float. Got"),
            (1, 2.0, 2, TypeError, "Input grid_size type is not Tuple. Got"),
            (1, 2.0, (2, 2, 2), TypeError, "Input grid_size is not a Tuple with 2 elements. Got 3"),
            (1, 2.0, (2, 2.0), TypeError, "Input grid_size type is not valid, must be a Tuple[int, int]"),
            (1, 2.0, (2, 0), ValueError, "Input grid_size elements must be positive. Got"),
        ],
    )
    def test_exception(self, B, clip, grid, exception_type, expected_error_msg):
        C, H, W = 1, 10, 20
        img = torch.rand(B, C, H, W)
        with pytest.raises(exception_type) as errinfo:
            enhance.equalize_clahe(img, clip, grid)
        assert expected_error_msg in str(errinfo)

    @pytest.mark.parametrize("dims", [(1, 1, 1, 1, 1), (1, 1)])
    def test_exception_tensor_dims(self, dims):
        img = torch.rand(dims)
        with pytest.raises(ValueError):
            enhance.equalize_clahe(img)

    def test_exception_tensor_type(self):
        with pytest.raises(TypeError):
            enhance.equalize_clahe([1, 2, 3])

    def test_clahe_preserves_out_of_range_fast_path(self):
        x = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
        x.view(-1)[0] = 1.0000001

        y = enhance.equalize_clahe(x, 40.0, (1, 1))

        assert y.sum().item() == pytest.approx(32.87843322753906)
        assert y.max().item() == pytest.approx(1.0)

    @pytest.mark.parametrize("grid_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_single_tile_on_the_differentiable_path(self, grid_size, device, dtype):
        # A (1, 1) grid is global equalization with the clip limit applied, and the slow path raised an
        # IndexError on it because one tile lost its tile axis to squeeze().
        img = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        out = enhance.equalize_clahe(img, 40.0, grid_size, slow_and_differentiable=True)
        assert out.shape == img.shape
        assert torch.isfinite(out).all()

    def test_histogram_skips_out_of_range(self, device, dtype):
        # torch.histc on CPU counts only values inside [min, max] and _tiles_histc must too. 1.0000001 and
        # -1e-7 are inside the window equalize_clahe admits (see RandomClahe's warning and #4564), and MPS's
        # torch.histc counts them, so this also pins CPU/MPS parity.
        from kornia.enhance.equalization import _tiles_histc

        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("1.0000001 and -1e-7 round to 1 and 0 in half precision")
        tiles = torch.tensor([[0.5, 1.0000001, -1e-7, 0.25]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(_tiles_histc(tiles, 4), expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        # The tile histograms keep a static shape, so equalize_clahe is one dynamo graph. A data-dependent
        # op such as bincount splits it into six; torch.compile without fullgraph=True would not notice.
        img = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)

        def op(x):
            return enhance.equalize_clahe(x, 40.0, (2, 2))

        torch._dynamo.reset()
        explanation = torch._dynamo.explain(op)(img)
        assert explanation.graph_break_count == 0, explanation.break_reasons
        self.assert_close(torch_optimizer(op)(img), op(img))

    @pytest.mark.parametrize("grid_size", [(2, 2), (2, 3), (3, 2)])
    def test_gradcheck(self, device, grid_size):
        torch.random.manual_seed(4)
        bs, channels, height, width = 1, 1, 11, 11
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=torch.float64)

        def grad_rot(data, a, b, c):
            rot = rotate(data, torch.tensor(30.0, dtype=data.dtype, device=device))
            return enhance.equalize_clahe(rot, a, b, c)

        self.gradcheck(grad_rot, (inputs, 40.0, grid_size, True), nondet_tol=1e-4)

    @pytest.mark.skip(reason="args and kwargs in decorator")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 10, 20
        inp = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        op = enhance.equalize_clahe
        op_script = torch.jit.script(op)
        self.assert_close(op(inp), op_script(inp))

    def test_module(self):
        # equalize_clahe is only a function
        pass

    @pytest.mark.parametrize("scale, shift", [(2.0, 0.0), (1.0, -1.0)])
    def test_out_of_range_input_names_the_range(self, scale, shift, device, dtype):
        # kornia#4564: the tile-LUT gather used to fail with a raw
        # "index ... is out of bounds for dimension 5 with size 256". MPS range-checks it too
        # (kornia#4600): from torch 2.13 the assert is asynchronous there, so the message arrives at
        # the sync.
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        torch.manual_seed(0)
        x = torch.rand(2, 3, 32, 40, device=device, dtype=dtype) * scale + shift
        with pytest.raises(RuntimeError, match=r"equalize_clahe expects input values in \[0, 1\]"):
            _sync(enhance.equalize_clahe(x).device)

    def test_input_the_lookup_can_index_is_still_accepted(self, device, dtype):
        # The check covers exactly the domain the gather can index, so a hair above 1 keeps working.
        x = torch.rand(2, 3, 32, 40, device=device, dtype=dtype) * 1.0001
        assert enhance.equalize_clahe(x).shape == x.shape

    def test_dynamo_fullgraph(self, device, dtype):
        # The range check must not introduce a graph break.
        x = torch.rand(2, 3, 32, 40, device=device, dtype=dtype)
        torch._dynamo.reset()
        compiled = torch.compile(enhance.equalize_clahe, fullgraph=True, backend="eager")
        self.assert_close(compiled(x), enhance.equalize_clahe(x))

    def test_dynamo_fullgraph_out_of_range_input_names_the_range(self, device, dtype):
        # The compiled graph has to carry the same check: on MPS kornia#4600 is only fixed while
        # compiling if the asynchronous assert is traced, because a host read cannot be.
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        if device.type == "mps" and not torch_version_ge(2, 13):
            pytest.skip("no MPS kernel for _assert_async before torch 2.13, so the check is skipped here")
        torch.manual_seed(0)
        x = torch.rand(2, 3, 32, 40, device=device, dtype=dtype) * 2.0
        torch._dynamo.reset()
        compiled = torch.compile(enhance.equalize_clahe, fullgraph=True, backend="eager")
        with pytest.raises(RuntimeError, match=r"equalize_clahe expects input values in \[0, 1\]"):
            _sync(compiled(x).device)

    @pytest.fixture()
    def img(self, device, dtype):
        height, width = 20, 20
        # TODO: test with a more realistic pattern
        return torch.arange(width, device=device).div(float(width - 1))[None].expand(height, width)[None][None]

    def test_he(self, img):
        # should be similar to enhance.equalize but slower. Similar because the lut is computed in a different way.
        clip_limit: float = 0.0
        grid_size: Tuple = (1, 1)
        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)
        # NOTE: for next versions we need to improve the computation of the LUT
        # and test with a better image
        self.assert_close(
            res[..., 0, :],
            torch.tensor(
                [
                    [
                        [
                            0.0471,
                            0.0980,
                            0.1490,
                            0.2000,
                            0.2471,
                            0.2980,
                            0.3490,
                            0.3490,
                            0.4471,
                            0.4471,
                            0.5490,
                            0.5490,
                            0.6471,
                            0.6471,
                            0.6980,
                            0.7490,
                            0.8000,
                            0.8471,
                            0.8980,
                            1.0000,
                        ]
                    ]
                ],
                dtype=res.dtype,
                device=res.device,
            ),
            low_tolerance=True,
        )

    def test_ahe(self, img):
        clip_limit: float = 0.0
        grid_size: Tuple = (8, 8)
        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)
        # NOTE: for next versions we need to improve the computation of the LUT
        # and test with a better image
        self.assert_close(
            res[..., 0, :],
            torch.tensor(
                [
                    [
                        [
                            0.2471,
                            0.4980,
                            0.7490,
                            0.6667,
                            0.4980,
                            0.4980,
                            0.7490,
                            0.4993,
                            0.4980,
                            0.2471,
                            0.7490,
                            0.4993,
                            0.4980,
                            0.2471,
                            0.4980,
                            0.4993,
                            0.3333,
                            0.2471,
                            0.4980,
                            1.0000,
                        ]
                    ]
                ],
                dtype=res.dtype,
                device=res.device,
            ),
            low_tolerance=True,
        )

    def test_clahe(self, img):
        clip_limit: float = 2.0
        grid_size: Tuple = (8, 8)
        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)
        res_diff = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size, slow_and_differentiable=True)
        # NOTE: for next versions we need to improve the computation of the LUT
        # and test with a better image
        expected = torch.tensor(
            [
                [
                    [
                        0.1216,
                        0.8745,
                        0.9373,
                        0.9163,
                        0.8745,
                        0.8745,
                        0.9373,
                        0.8745,
                        0.8745,
                        0.8118,
                        0.9373,
                        0.8745,
                        0.8745,
                        0.8118,
                        0.8745,
                        0.8745,
                        0.8327,
                        0.8118,
                        0.8745,
                        1.0000,
                    ]
                ]
            ],
            dtype=res.dtype,
            device=res.device,
        )
        exp_diff = torch.tensor(
            [
                [
                    [
                        0.1250,
                        0.8752,
                        0.9042,
                        0.9167,
                        0.8401,
                        0.8852,
                        0.9302,
                        0.9120,
                        0.8750,
                        0.8370,
                        0.9620,
                        0.9077,
                        0.8750,
                        0.8754,
                        0.9204,
                        0.9167,
                        0.8370,
                        0.8806,
                        0.9096,
                        1.0000,
                    ]
                ]
            ],
            dtype=res.dtype,
            device=res.device,
        )
        self.assert_close(res[..., 0, :], expected, low_tolerance=True)
        self.assert_close(res_diff[..., 0, :], exp_diff, low_tolerance=True)

    def test_clahe_non_square_grid(self, device, dtype):
        # Pixel values are 0 and powers of two, exact in every dtype. With 4 x 4 tiles every interpolation weight
        # is a multiple of 1/3, so 9 * 255 * output is an integer. The expected integers come from an exact
        # rational evaluation of CLAHE pixel by pixel, independent of this implementation's tile indexing; the
        # reference is an exact-Fraction restatement of _compute_tiles/_compute_luts/_compute_equalized_tiles
        # (tile size ceil(n/g) rounded up to even, trailing reflect pad, floor(v*256) histogram, clip and
        # redistribute, floor(cumsum*255/P), axis blend weight (T-1-k)/(T-1)); it is posted in full on #4628.
        codes = torch.tensor(
            [
                [1, 2, 6, 7, 1, 7, 1, 5, 4, 8, 4, 6],
                [2, 7, 7, 8, 0, 4, 7, 2, 5, 0, 2, 1],
                [3, 3, 5, 5, 7, 2, 4, 8, 5, 5, 4, 4],
                [6, 5, 0, 4, 5, 3, 2, 8, 1, 3, 7, 7],
                [2, 1, 0, 7, 2, 8, 8, 3, 2, 2, 4, 4],
                [8, 2, 0, 3, 8, 4, 6, 0, 2, 5, 0, 7],
                [8, 8, 2, 3, 8, 6, 6, 3, 1, 4, 8, 4],
                [2, 7, 8, 7, 5, 7, 7, 5, 0, 0, 6, 0],
            ],
            device=device,
        )
        levels = torch.tensor([0.0] + [2.0**-k for k in range(7, -1, -1)], device=device, dtype=dtype)
        img = levels[codes][None, None]
        expected = torch.tensor(
            [
                [423, 567, 1575, 1815, 423, 1719, 423, 1431, 1143, 2295, 1143, 1719],
                [567, 1863, 1863, 2295, 279, 1143, 1719, 759, 1431, 279, 855, 423],
                [855, 855, 1287, 1335, 1767, 711, 1143, 2295, 1431, 1431, 1143, 1143],
                [1623, 1431, 327, 1255, 1367, 903, 663, 2295, 455, 1143, 1911, 1911],
                [855, 519, 375, 1751, 695, 2295, 2295, 967, 839, 951, 1335, 1335],
                [2295, 999, 423, 1191, 2295, 999, 1431, 327, 855, 1719, 423, 2007],
                [2295, 2295, 999, 1191, 2295, 1431, 1431, 951, 519, 1431, 2295, 1431],
                [999, 1719, 2295, 1719, 1335, 1719, 1719, 1335, 375, 423, 1863, 423],
            ],
            dtype=torch.float64,
        )
        expected = expected.div(9 * 255).to(dtype).to(device)[None, None]
        # Both orientations: the two border regions are indexed by the grid size of different axes.
        self.assert_close(enhance.equalize_clahe(img, 40.0, (2, 3)), expected)
        self.assert_close(enhance.equalize_clahe(img.transpose(-2, -1), 40.0, (3, 2)), expected.transpose(-2, -1))

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    @pytest.mark.parametrize("grid_size", [(1, 2), (2, 3), (3, 4), (2, 6), (6, 2)])
    def test_clahe_non_square_grid_transpose(self, grid_size, slow_and_differentiable, device, dtype):
        # Transposing the image and the grid transposes the output. Every grid tiles 12 x 24 without padding.
        # Clipping must preserve transpose equivariance for both the fast and slow differentiable paths.
        torch.manual_seed(2531)
        img = torch.rand(1, 2, 12, 24, device=device, dtype=dtype)
        out = enhance.equalize_clahe(img, 40.0, grid_size, slow_and_differentiable)
        out_t = enhance.equalize_clahe(img.transpose(-2, -1), 40.0, grid_size[::-1], slow_and_differentiable)
        self.assert_close(out_t, out.transpose(-2, -1))

    def test_clahe_transpose_equivariance_4632(self):
        torch.manual_seed(28)
        img = torch.rand(1, 1, 12, 12, dtype=torch.float32)

        output = enhance.equalize_clahe(
            img,
            clip_limit=40.0,
            grid_size=(2, 2),
            slow_and_differentiable=True,
        )
        output_transposed = enhance.equalize_clahe(
            img.transpose(-2, -1),
            clip_limit=40.0,
            grid_size=(2, 2),
            slow_and_differentiable=True,
        ).transpose(-2, -1)

        self.assert_close(output, output_transposed, low_tolerance=True)
