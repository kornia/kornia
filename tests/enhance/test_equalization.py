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

import math
from typing import Tuple

import pytest
import torch

from kornia import enhance
from kornia.core._compat import torch_version_ge
from kornia.core.check import are_checks_enabled, disable_checks, enable_checks
from kornia.enhance.equalization import _compute_tiles
from kornia.geometry import rotate

from testing.base import BaseTester


def _clahe_interpolation_reference(img, luts, tile_size):
    # A tile spans i*T .. i*T+T-1, so its centre is i*T+(T-1)/2. Evaluate the blend between
    # these centres pixel by pixel, independently of production's half-tile indexing.
    values = img.detach().cpu().double()
    tables = luts.detach().cpu().double()
    expected = torch.zeros_like(values)
    gh, gw = tables.shape[1:3]
    for y in range(values.shape[-2]):
        fy = (y + 0.5) / tile_size[0] - 0.5
        y0 = math.floor(fy)
        ay = fy - y0
        for x in range(values.shape[-1]):
            fx = (x + 0.5) / tile_size[1] - 0.5
            x0 = math.floor(fx)
            ax = fx - x0
            # Preserve the input dtype's lookup quantization; this reference tests only interpolation.
            index = (img[..., y, x] * 255).long().cpu().unsqueeze(-1)
            for row, wy in [(y0, 1 - ay), (y0 + 1, ay)]:
                for col, wx in [(x0, 1 - ax), (x0 + 1, ax)]:
                    lut = tables[:, min(max(row, 0), gh - 1), min(max(col, 0), gw - 1)]
                    expected[..., y, x] += wy * wx * lut.gather(-1, index).squeeze(-1) / 255
    return expected.to(img)


def _sync(device) -> None:
    # MPS dispatches asynchronously, so a kernel error raised by the forward under test would
    # otherwise surface inside an unrelated later test.
    if device.type == "mps":
        torch.mps.synchronize()


class TestEqualization(BaseTester):
    @pytest.mark.parametrize("grid", [(2, 3), (1, 3), (2, 1), (1, 1)])
    def test_clahe_interpolation_coordinates_5698(self, grid, device, dtype):
        # Unequal axis lengths distinguish 1/4 from 1/8; constant LUTs isolate blending from histograms.
        gh, gw = grid
        img = torch.zeros(1, 1, gh * 4, gw * 8, device=device, dtype=dtype)
        rows = torch.arange(gh, device=device, dtype=dtype)[:, None]
        cols = torch.arange(gw, device=device, dtype=dtype)[None, :]
        luts = (16 + 96 * rows + 48 * cols)[None, :, :, None, None].expand(1, gh, gw, 1, 256)
        tiles = enhance.equalization._compute_interpolation_tiles(img, (4, 8))
        out = enhance.equalization._compute_equalized_tiles(tiles, luts)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape_as(img)
        # Every pixel: internal regions, both opposing borders of each axis, and all four corners.
        self.assert_close(out, _clahe_interpolation_reference(img, luts, (4, 8)))
        if gh > 1:
            expected = torch.full_like(out[..., 2:6, 0], 96 / (4 * 255))
            expected[..., -1] /= 2  # The last interval reaches the clamped border halfway through.
            self.assert_close(out[..., 3:7, 0] - out[..., 2:6, 0], expected)
        if gw > 1:
            start = (gw - 2) * 8 + 4
            expected = torch.full_like(out[..., 0, start : start + 8], 48 / (8 * 255))
            expected[..., -1] /= 2
            self.assert_close(out[..., 0, start + 1 : start + 9] - out[..., 0, start : start + 8], expected)

    @pytest.mark.parametrize("dim", [-1, -2])
    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    @pytest.mark.parametrize("clip", [2.0, 40.0])
    def test_clahe_flip_symmetry_5698(self, dim, slow_and_differentiable, clip, device, dtype):
        # Unequal even tile sizes, without padding: reflection must reflect the tile centres and LUTs too.
        # Dyadic values are exact in every dtype and avoid changing bins through input rounding.
        y = torch.arange(12, device=device, dtype=dtype)[:, None]
        x = torch.arange(24, device=device, dtype=dtype)[None, :]
        img = ((2 * y + x) / 64)[None, None]
        out = enhance.equalize_clahe(img, clip, (3, 3), slow_and_differentiable)
        flipped = enhance.equalize_clahe(img.flip(dim), clip, (3, 3), slow_and_differentiable)
        self.assert_close(flipped, out.flip(dim))

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    @pytest.mark.parametrize("size", [(12, 24), (11, 23)])
    def test_clahe_interpolation_paths_5698(self, size, slow_and_differentiable, device, dtype):
        # Smooth dyadic inputs avoid an input quantization discrepancy; the second size needs padding.
        y = torch.arange(size[0], device=device, dtype=dtype)[:, None]
        x = torch.arange(size[1], device=device, dtype=dtype)[None, :]
        img = ((2 * y + x) / 64)[None, None]
        tiles, _ = _compute_tiles(img, (3, 3), True)
        luts = enhance.equalization._compute_luts(tiles, clip=2.0, diff=slow_and_differentiable)
        expected = _clahe_interpolation_reference(img, luts, tiles.shape[-2:])
        out = enhance.equalize_clahe(img, 2.0, (3, 3), slow_and_differentiable)
        self.assert_close(out, expected)
        assert out.shape == img.shape
        assert out.dtype == dtype
        assert out.device == img.device
        assert torch.isfinite(out).all()
        assert ((out >= 0) & (out <= 1)).all()

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

    @pytest.mark.parametrize(
        ("size", "grid"),
        [
            ((8, 8), (8, 8)),  # kornia#4783: an image as large as the grid
            ((8, 16), (8, 8)),  # only the vertical axis is too small
            ((16, 8), (8, 8)),  # only the horizontal axis is too small
            ((4, 4), (8, 8)),  # already rejected before, message now names the limit
            ((4, 10), (4, 6)),  # a non-square grid: the limit is named per axis, in (H, W) order
        ],
    )
    def test_exception_image_too_small_for_grid_4783(self, size, grid):
        # kornia#4783: reflect padding needs the pad below the axis it reflects, so an image that
        # only matches the grid used to reach F.pad and fail with a raw padding error.
        img = torch.rand(1, 1, *size)
        with pytest.raises(ValueError) as errinfo:
            enhance.equalize_clahe(img, grid_size=grid)
        assert "Cannot compute tiles" in str(errinfo.value)
        assert f"Got image size {size} and grid size {grid}" in str(errinfo.value)
        assert f"smallest image this grid admits is ({grid[0] + 1}, {grid[1] + 1})" in str(errinfo.value)

    @pytest.mark.parametrize(
        ("size", "grid", "smallest"),
        [((4, 10), (8, 6), (5, 4)), ((10, 3), (5, 6), (3, 4))],
    )
    def test_compute_tiles_odd_tiles_names_its_own_limit_4783(self, size, grid, smallest):
        # equalize_clahe rejects an axis no larger than its grid before it tiles; _compute_tiles called with odd
        # tiles only needs an axis above half its grid size, and the limit the message names must be accepted.
        with pytest.raises(ValueError) as errinfo:
            _compute_tiles(torch.rand(1, 1, *size), grid, even_tile_size=False)
        assert f"smallest image this grid admits is {smallest}" in str(errinfo.value)
        tiles, _ = _compute_tiles(torch.rand(1, 1, *smallest), grid, even_tile_size=False)
        assert tiles.shape[1:3] == grid

    @pytest.mark.parametrize("grid", [(2, 2), (4, 4), (8, 8)])
    def test_smallest_image_the_grid_admits_is_accepted_4783(self, grid, device, dtype):
        # The size the message names must work, so the bound it reports is exact.
        img = torch.rand(1, 1, grid[0] + 1, grid[1] + 1, device=device, dtype=dtype)
        out = enhance.equalize_clahe(img, grid_size=grid)
        assert out.shape == img.shape

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

        assert y.sum().item() == pytest.approx(8539 / 255)
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
        # Like torch.histc on CPU, _tiles_histc counts only values inside [0, 1]. 1.0000001 and -1e-7 are inside the
        # window equalize_clahe admits (see RandomClahe's warning and #4564), and MPS's torch.histc counts them, so
        # this also pins CPU/MPS parity. With 4 bins a value is counted in (x * 3).long(): 0.5 in 1 and 0.25 in 0.
        from kornia.enhance.equalization import _tiles_histc

        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("1.0000001 and -1e-7 round to 1 and 0 in half precision")
        tiles = torch.tensor([[0.5, 1.0000001, -1e-7, 0.25]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
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

    def test_out_of_range_input_names_the_range_with_checks_disabled(self, device, dtype):
        # The range check keeps the gather in bounds, so it runs with checks disabled too (``python -O``,
        # ``KORNIA_CHECKS=0``): compiled on the CPU, the gather with an index out of range terminates the process.
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        torch.manual_seed(0)
        x = torch.rand(2, 3, 32, 40, device=device, dtype=dtype) * 2.0
        were_checks_enabled = are_checks_enabled()
        disable_checks()
        try:
            with pytest.raises(RuntimeError, match=r"equalize_clahe expects input values in \[0, 1\]"):
                _sync(enhance.equalize_clahe(x).device)
        finally:
            if were_checks_enabled:
                enable_checks()

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

    def test_dynamo_grad_out_of_range_input_names_the_range(self, device, dtype):
        # The range check keeps the gather in bounds, so a compiled torch.func.grad keeps it, although a compiled
        # functorch transform skips the value checks that guard no index.
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        if device.type == "mps" and not torch_version_ge(2, 13):
            pytest.skip("no MPS kernel for _assert_async before torch 2.13, so the check is skipped here")
        if not torch_version_ge(2, 14):
            pytest.skip("Dynamo in torch 2.9.1 and older cannot compile torch.func.grad over equalize_clahe")
        x = torch.linspace(0, 2, 2 * 3 * 32 * 40, device=device, dtype=dtype).reshape(2, 3, 32, 40)
        torch._dynamo.reset()
        gradient = torch.func.grad(lambda image: enhance.equalize_clahe(image).sum())
        compiled = torch.compile(gradient, fullgraph=True, backend="eager")
        with pytest.raises(RuntimeError, match=r"equalize_clahe expects input values in \[0, 1\]"):
            _sync(compiled(x).device)

    @pytest.fixture
    def img(self, device, dtype):
        height, width = 20, 20
        # TODO: test with a more realistic pattern
        return torch.arange(width, device=device).div(float(width - 1))[None].expand(height, width)[None][None]

    def test_he(self, img):
        # With no clip limit and one tile this is global equalization with OpenCV's rounded LUT: the 20-step ramp maps
        # to 20 evenly spaced levels, round(12.75 * (i + 1)) with ties to even, as an exact rational evaluation of
        # CLAHE gives. Counting each pixel in floor(256 x) instead of the entry floor(255 x) it reads merged four pairs
        # of neighbouring steps (#6181).
        res = enhance.equalize_clahe(img, clip_limit=0.0, grid_size=(1, 1))
        levels = [13, 26, 38, 51, 64, 76, 89, 102, 115, 128, 140, 153, 166, 178, 191, 204, 217, 230, 242, 255]
        expected = torch.tensor(levels, dtype=res.dtype, device=res.device) / 255
        self.assert_close(res[0, 0, 0], expected, rtol=0, atol=0)

    def test_ahe(self, img):
        clip_limit: float = 0.0
        grid_size: Tuple = (8, 8)
        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)

        tiles, padded = _compute_tiles(img, grid_size, even_tile_size=False)
        luts = enhance.equalization._compute_luts(tiles, clip=clip_limit, diff=False)
        expected = _clahe_interpolation_reference(padded, luts, tiles.shape[-2:])
        expected = expected[..., : img.shape[-2], : img.shape[-1]]

        self.assert_close(res, expected)

    def test_clahe(self, img):
        clip_limit: float = 2.0
        grid_size: Tuple = (8, 8)

        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)
        res_diff = enhance.equalize_clahe(
            img,
            clip_limit=clip_limit,
            grid_size=grid_size,
            slow_and_differentiable=True,
        )

        tiles, padded = _compute_tiles(img, grid_size, even_tile_size=False)

        luts = enhance.equalization._compute_luts(tiles, clip=clip_limit, diff=False)
        expected = _clahe_interpolation_reference(padded, luts, tiles.shape[-2:])
        expected = expected[..., : img.shape[-2], : img.shape[-1]]
        self.assert_close(res, expected)

        luts_diff = enhance.equalization._compute_luts(tiles, clip=clip_limit, diff=True)
        expected_diff = _clahe_interpolation_reference(padded, luts_diff, tiles.shape[-2:])
        expected_diff = expected_diff[..., : img.shape[-2], : img.shape[-1]]
        self.assert_close(res_diff, expected_diff)

        # The references above take their lookup tables from _compute_luts, so they cannot see a wrong table: pin
        # the first row as well. Each 3 x 3 tile holds 3 columns of one value each, so with the clip limit of 1
        # count per bin the 6 clipped counts are spread over every 42nd bin, as in OpenCV, and the gradient keeps
        # its range; piling them into the first 6 bins mapped every pixel but the first above 0.77. No other test
        # pins the values of the differentiable path's soft histogram.
        expected_row = torch.tensor(
            [0.2235, 0.3333, 0.332, 0.2967, 0.4431, 0.4458, 0.4078, 0.5569, 0.5556, 0.5176]
            + [0.6667, 0.6654, 0.6301, 0.7765, 0.7791, 0.7412, 0.8902, 0.9255, 0.9268, 1.0],
            dtype=res.dtype,
            device=res.device,
        )
        expected_row_diff = torch.tensor(
            [0.2222, 0.3335, 0.2841, 0.3087, 0.3545, 0.4586, 0.388, 0.4, 0.6001, 0.479]
            + [0.6712, 0.6155, 0.6411, 0.6371, 0.7207, 0.7493, 0.6889, 0.8889, 0.8387, 1.0],
            dtype=res.dtype,
            device=res.device,
        )
        self.assert_close(res[0, 0, 0], expected_row, rtol=0, atol=1e-4)
        self.assert_close(res_diff[0, 0, 0], expected_row_diff, low_tolerance=True)

    @pytest.mark.parametrize("count, expected_level", [(100, 100), (128, 128)])
    @pytest.mark.parametrize("clip_as_tensor", [False, True])
    def test_clahe_lut_rounding(self, count, expected_level, clip_as_tensor, device, dtype):
        # With clipping disabled, the first CDF value is count * 255 / 256:
        # 99.609375 rounds to 100, and the tie 127.5 rounds to 128 (#5659).
        img = torch.ones(256, device=device, dtype=dtype)
        img[:count] = 0
        img = img.view(1, 1, 16, 16)
        if clip_as_tensor:
            clip = torch.tensor([0.0], device=device, dtype=dtype)
            out = enhance.equalization._equalize_clahe(img, clip, (1, 1), False)
        else:
            out = enhance.equalize_clahe(img, 0.0, (1, 1))
        expected = torch.full_like(img, 255).flatten()
        expected[:count] = expected_level
        self.assert_close(out, expected.view_as(img).div(255), rtol=0, atol=0)
        assert out.dtype == dtype
        assert out.device == device

    def test_clahe_lut_rounding_ties_to_even(self, device, dtype):
        # Four bins with counts [1, 2, 1, 2] give CDF values [0.5, 1.5, 2, 3]. A value is counted in (x * 3).long().
        tiles = torch.tensor([0.0, 0.34, 0.34, 0.67, 1.0, 1.0], device=device, dtype=dtype)
        luts = enhance.equalization._compute_luts(tiles.view(1, 1, 1, 1, 2, 3), num_bins=4, clip=0.0)
        expected = torch.tensor([0.0, 2.0, 2.0, 3.0], device=device, dtype=dtype)
        self.assert_close(luts.flatten(), expected, rtol=0, atol=0)

    def test_clahe_differentiable_lut_is_not_rounded(self, device, dtype):
        tiles = torch.ones(256, device=device, dtype=dtype)
        tiles[:100] = 0
        luts = enhance.equalization._compute_luts(tiles.view(1, 1, 1, 1, 16, 16), clip=0.0, diff=True)
        assert ((luts > luts.floor()) & (luts < luts.ceil())).any()

    @pytest.mark.parametrize("clip_as_tensor", [False, True])
    def test_clahe_residual_spread(self, clip_as_tensor, device, dtype):
        # One 2 x 4 tile of 8 pixels, 4 at level 0 and 4 at level 100, clip limit 40: the per-bin limit is
        # max(40 * 8 // 256, 1) = 1, so 6 counts are clipped. OpenCV spreads a residual below the bin count over
        # every (256 // 6) = 42nd bin from bin 0 (bins 0, 42, 84, 126, 168, 210), which leaves 2 counts at or
        # below bin 0 and 5 at or below bin 100: the levels map to round(2 * 255 / 8) = 64 and
        # round(5 * 255 / 8) = 159. Piling the residual into bins 0..5 instead gave 8 counts at bin 100 and
        # mapped it to 255.
        img = torch.tensor([[0.0] * 4, [100.0] * 4], device=device, dtype=dtype).div(255.0)[None, None]
        clip = torch.tensor([40.0], device=device, dtype=dtype) if clip_as_tensor else 40.0
        out = enhance.equalization._equalize_clahe(img, clip, (1, 1), False)
        expected = torch.tensor([[64.0] * 4, [159.0] * 4], device=device, dtype=dtype).div(255.0)[None, None]
        self.assert_close(out, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("clip_as_tensor", [False, True])
    def test_clahe_residual_spread_after_the_whole_share(self, clip_as_tensor, device, dtype):
        # One 32 x 32 tile of 1024 pixels: 342 in bin 0, 341 in bin 100 and 341 in bin 200. Clip limit 40 caps every
        # bin at 40 * 1024 // 256 = 160, so 544 counts are clipped. Every bin takes the whole share 544 // 256 = 2, and
        # the residual 32 goes, as in OpenCV, to every 256 // 32 = 8th bin from bin 0. The LUT is
        # round(cumsum * 255 / 1024): bin 100 maps to round(535 * 255 / 1024) = 133 and bin 200 to
        # round(908 * 255 / 1024) = 226, where writing the residual into bins 0..31 would give 138 and 228.
        # The counts stay in float32 for half-precision tiles too, so this is exact in every dtype.
        values = torch.cat([torch.zeros(342), torch.full((341,), 100 / 255), torch.full((341,), 200 / 255)])
        tiles = values.to(device, dtype).view(1, 1, 1, 1, 32, 32)
        clip = torch.tensor([40.0], device=device, dtype=dtype) if clip_as_tensor else 40.0
        luts = enhance.equalization._compute_luts(tiles, clip=clip)
        hist = torch.full((256,), 2.0, dtype=torch.float64)
        hist[::8] += 1
        hist[[0, 100, 200]] += 160
        expected = (hist.cumsum(0) * 255 / 1024).round()
        assert expected[100] == 133
        assert expected[200] == 226
        self.assert_close(luts.view(256), expected.to(device, dtype))

    def test_clahe_non_square_grid(self, device, dtype):
        # Pixel values are 0 and powers of two, exact in every dtype. With 4 x 4 tiles every interpolation weight
        # is a multiple of 1/8, so 64 * 255 * output is an integer. The expected integers come from an exact
        # rational evaluation of CLAHE pixel by pixel, independent of this implementation's tile indexing; the
        # reference is an exact-Fraction restatement of _compute_tiles/_compute_luts/_compute_equalized_tiles
        # (tile size ceil(n/g) rounded up to even, trailing reflect pad, floor(v*255) histogram, clip and
        # redistribute, round(cumsum*255/P), axis blend weight (T-k-0.5)/T); the reference posted on #4628 is
        # evaluated here with round-to-even LUTs and the clipped residual spread over every (256 // residual)-th
        # bin from bin 0, as OpenCV does.
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
                [3072, 5120, 12096, 13888, 3712, 13376, 4096, 10816, 8552, 16320, 8192, 11200],
                [5120, 14272, 14144, 16320, 2048, 9032, 13376, 5760, 10560, 2048, 5120, 4096],
                [7296, 7296, 10040, 10264, 13536, 5696, 8703, 16320, 10339, 10145, 8192, 8192],
                [11072, 9432, 2384, 8129, 9832, 6544, 5056, 16320, 3808, 6176, 13888, 13888],
                [5760, 3712, 2608, 13088, 4800, 16320, 16320, 5984, 5184, 5568, 8192, 8192],
                [16320, 6016, 2832, 7056, 16320, 6729, 10416, 2384, 5056, 9191, 2944, 13376],
                [16320, 16320, 5760, 7040, 16320, 10048, 10304, 5504, 3328, 7936, 16320, 8192],
                [6144, 12224, 16320, 12608, 8192, 13120, 13248, 8552, 2688, 2944, 11200, 3072],
            ],
            dtype=torch.float64,
        )
        expected = expected.div(64 * 255).to(dtype).to(device)[None, None]
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

    # The fast path counts, clips and accumulates each tile's histogram in float32 (float64 for float64 images) and
    # casts only the finished LUT back, so a half-precision tile gets the LUT a float32 tile with the same 8-bit levels
    # gets. Counting in float16 overflowed the clipped total of a 256 x 256 tile to inf, the tile the default 8 x 8 grid
    # gives on a 2048 x 2048 image, and rounded counts above 2048; bfloat16 rounded counts above 256.
    @pytest.mark.parametrize("clip_as_tensor", [False, True])
    @pytest.mark.parametrize(
        ("size", "grid", "tile"), [((512, 512), (2, 2), (256, 256)), ((197, 100), (1, 1), (197, 100))]
    )
    def test_clahe_luts_count_outside_half_precision(self, size, grid, tile, clip_as_tensor, device, dtype):
        torch.manual_seed(5374)
        img = torch.rand(1, 2, *size, device=device, dtype=dtype)
        tiles, _ = _compute_tiles(img, grid)
        assert tuple(tiles.shape[-2:]) == tile
        clip = torch.tensor([40.0], device=device) if clip_as_tensor else 40.0
        luts = enhance.equalization._compute_luts(tiles, clip=clip)
        count_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        # A pixel is counted in the level it reads, (x * 255).long() in the image dtype; float16 can round 255 x to
        # another level than float32 does, so the reference tiles hold those levels.
        levels = (tiles * 255).long().to(count_dtype) / 255
        expected = enhance.equalization._compute_luts(levels, clip=clip)
        assert luts.dtype == dtype
        assert torch.equal(luts.to(count_dtype), expected)

    # One 65536-pixel tile: in float16 the default clip limit turned every output pixel into NaN, and a clip limit whose
    # per-bin cap clip * 65536 // 256 exceeds 65504 raised an overflow error on CPU (and aborted the process on MPS).
    @pytest.mark.parametrize("clip_limit", [40.0, 300.0])
    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_clahe_half_precision_tile_of_65536_pixels(self, slow_and_differentiable, clip_limit, device, dtype):
        torch.manual_seed(5374)
        img = torch.rand(1, 1, 256, 256, device=device, dtype=dtype)
        out = enhance.equalize_clahe(img, clip_limit, (1, 1), slow_and_differentiable)
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
        expected = enhance.equalize_clahe(img.cpu().double(), clip_limit, (1, 1), slow_and_differentiable)
        # What remains in half precision is the pixel lookup and the blend, which stay in the input dtype as they do for
        # equalize: at most two gray levels for this uniform image.
        atol = 2.5 / 255 if dtype in (torch.float16, torch.bfloat16) else None
        self.assert_close(out.cpu().double(), expected, rtol=0 if atol else None, atol=atol)

    @pytest.mark.parametrize(
        ("size", "grid", "expected_tile", "expected_padded"),
        [
            ((13, 15), (2, 3), (7, 5), (14, 15)),
            ((100, 150), (8, 8), (13, 19), (104, 152)),
        ],
    )
    def test_clahe_ceil_tile_sizes_for_non_divisible_images_5812(self, size, grid, expected_tile, expected_padded):
        img = torch.rand(1, 1, *size)
        tiles, padded = _compute_tiles(img, grid)

        assert tuple(tiles.shape[-2:]) == expected_tile
        assert tuple(padded.shape[-2:]) == expected_padded

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_clahe_odd_tile_sizes(self, slow_and_differentiable, device):
        # Regression test: odd 9 x 11 tiles must not be expanded to 10 x 12.
        torch.manual_seed(5812)
        img = torch.randint(0, 256, (1, 1, 72, 88), dtype=torch.uint8).float().div(255.0).to(device)
        grid_size = (8, 8)
        clip_limit = 2.0

        out = enhance.equalize_clahe(
            img,
            clip_limit=clip_limit,
            grid_size=grid_size,
            slow_and_differentiable=slow_and_differentiable,
        )

        tiles, padded = _compute_tiles(img, grid_size, even_tile_size=False)
        luts = enhance.equalization._compute_luts(tiles, clip=clip_limit, diff=slow_and_differentiable)
        expected = _clahe_interpolation_reference(padded, luts, tiles.shape[-2:])
        expected = expected[..., : img.shape[-2], : img.shape[-1]]
        self.assert_close(out, expected)

        out_t = enhance.equalize_clahe(
            img.transpose(-2, -1),
            clip_limit=clip_limit,
            grid_size=grid_size,
            slow_and_differentiable=slow_and_differentiable,
        ).transpose(-2, -1)
        self.assert_close(out, out_t)

    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    @pytest.mark.parametrize(
        ("size", "grid", "tile"),
        [
            ((36, 44), (4, 4), (9, 11)),
            ((27, 40), (3, 4), (9, 10)),
            ((13, 15), (2, 3), (7, 5)),
            ((25, 23), (2, 2), (13, 12)),
            ((16, 25), (2, 4), (8, 7)),
        ],
    )
    def test_clahe_tiles_are_ceil_n_over_g_5812(self, size, grid, tile, slow_and_differentiable, device, dtype):
        # Tiles are ceil(n / g) pixels, odd or even, on each axis: odd, mixed, and three grids that do not divide
        # the image and pad it, two of them along one axis only, where OpenCV would also pad the axis the grid
        # divides. Two samples and two channels, each blended from its own lookup tables.
        torch.manual_seed(5812)
        img = torch.rand(2, 2, *size, device=device, dtype=dtype)
        tiles, padded = _compute_tiles(img, grid)
        assert tuple(tiles.shape[-2:]) == tile
        luts = enhance.equalization._compute_luts(tiles, clip=2.0, diff=slow_and_differentiable)
        expected = _clahe_interpolation_reference(padded, luts, tile)[..., : size[0], : size[1]]
        out = enhance.equalize_clahe(img, 2.0, grid, slow_and_differentiable)
        assert out.dtype == dtype
        self.assert_close(out, expected)

    @pytest.mark.parametrize("dim", [-1, -2])
    @pytest.mark.parametrize("slow_and_differentiable", [False, True])
    def test_clahe_flip_symmetry_odd_tiles_5812(self, dim, slow_and_differentiable, device, dtype):
        # A grid that divides the image does not pad it, so odd tiles keep the flip symmetry of even ones.
        y = torch.arange(15, device=device, dtype=dtype)[:, None]
        x = torch.arange(27, device=device, dtype=dtype)[None, :]
        img = ((2 * y + x) / 64)[None, None]
        out = enhance.equalize_clahe(img, 2.0, (3, 3), slow_and_differentiable)
        flipped = enhance.equalize_clahe(img.flip(dim), 2.0, (3, 3), slow_and_differentiable)
        self.assert_close(flipped, out.flip(dim))

    def test_clahe_odd_tiles_gradcheck_5812(self, device):
        torch.manual_seed(4)
        img = torch.rand(1, 1, 9, 15, device=device, dtype=torch.float64)
        self.gradcheck(enhance.equalize_clahe, (img, 40.0, (3, 3), True), nondet_tol=1e-4)

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

    @pytest.mark.parametrize("shape", [(0, 3, 16, 24), (0, 3, 16, 16), (2, 0, 3, 16, 24)])
    def test_clahe_empty_batch(self, shape, device, dtype):
        # An empty batch reaches the function and comes back empty, with the same shape (#6266).
        x = torch.rand(*shape, device=device, dtype=dtype)
        out = enhance.equalize_clahe(x, slow_and_differentiable=True)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        out = enhance.equalize_clahe(x, slow_and_differentiable=False)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
