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
from torch.autograd import gradcheck

import kornia
from kornia.core.exceptions import ShapeError

from testing.base import BaseTester

# ---------------------------------------------------------------------------------------------
# Reference model
#
# BT.470-5 Table 2, items 2.5 and 2.6 (the M/PAL system kornia documents) defines the YUV
# model by three relations, not by a matrix:
#
#     Y = 0.299 R + 0.587 G + 0.114 B
#     U = 0.492 (B - Y)
#     V = 0.877 (R - Y)
#
# https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf
#
# ``kornia.color.rgb_to_yuv`` hardcodes the *matrix* form of those relations, rounded to three
# decimals, so the helpers below are an independent implementation rather than a restatement of
# the library code, and the tolerances against them are set by that rounding.
# ---------------------------------------------------------------------------------------------

_Y_FROM_RGB = (0.299, 0.587, 0.114)
_U_SCALE = 0.492
_V_SCALE = 0.877


def _rgb_to_yuv_reference(rgb: torch.Tensor) -> torch.Tensor:
    """BT.470-5 M/PAL RGB -> YUV, from the defining relations. Expects ``(*, 3, H, W)`` float64."""
    r, g, b = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    y = _Y_FROM_RGB[0] * r + _Y_FROM_RGB[1] * g + _Y_FROM_RGB[2] * b
    u = _U_SCALE * (b - y)
    v = _V_SCALE * (r - y)
    return torch.stack([y, u, v], dim=-3)


def _yuv_to_rgb_reference(yuv: torch.Tensor) -> torch.Tensor:
    """BT.470-5 M/PAL YUV -> RGB, inverting the relations above. Expects ``(*, 3, H, W)`` float64."""
    y, u, v = yuv[..., 0, :, :], yuv[..., 1, :, :], yuv[..., 2, :, :]
    r = y + v / _V_SCALE
    b = y + u / _U_SCALE
    g = (y - _Y_FROM_RGB[0] * r - _Y_FROM_RGB[2] * b) / _Y_FROM_RGB[1]
    return torch.stack([r, g, b], dim=-3)


# YUV of the six reference colours under the relations above. Generated with (cpu, float64):
#   for rgb in ((1,1,1), (0,0,0), (.5,.5,.5), (1,0,0), (0,1,0), (0,0,1)):
#       y = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
#       print(rgb, (y, 0.492*(rgb[2]-y), 0.877*(rgb[0]-y)))
# Every value below is exact in decimal; none of them is a float64 residue.
_REFERENCE_COLORS = {
    "white": ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
    "black": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "gray": ((0.5, 0.5, 0.5), (0.5, 0.0, 0.0)),
    "red": ((1.0, 0.0, 0.0), (0.299, -0.147108, 0.614777)),
    "green": ((0.0, 1.0, 0.0), (0.587, -0.288804, -0.514799)),
    "blue": ((0.0, 0.0, 1.0), (0.114, 0.435912, -0.099978)),
}

# kornia's forward kernel is the BT.470-5 relations rounded to three decimals. Summing the
# per-coefficient rounding over the RGB unit cube bounds the disagreement by
#   U: |0.147108-0.147| + |0.288804-0.289| + |0.435912-0.436| = 3.92e-4
#   V: |0.614777-0.615| + |0.514799-0.515| + |0.099978-0.100| = 4.46e-4
# (Y uses the standard's own three constants, so it is exact.) 5e-4 clears both with margin.
_FORWARD_ATOL = 5.0e-4

# The inverse kernel is *separately* rounded rather than inverted, and one literal is rounded
# badly -- ``2.029`` where the published value is ``2.03211``. Summing |kornia - exact| times the
# documented YUV domain (Y in [0, 1], U in [-0.436, 0.436], V in [-0.615, 0.615]) row by row
# bounds the disagreement with the exact inverse of the relations at
#   R: |1.14 - 1.14025086| * 0.615                                   = 1.543e-4
#   G: |0.396 - 0.39473137| * 0.436 + |0.581 - 0.58080921| * 0.615   = 6.705e-4
#   B: |2.029 - 2.03252033| * 0.436                                  = 1.535e-3
# so B alone sets the tolerance. That excess is kornia#4044, pinned by
# TestYuvToRgb.test_convention_* / test_wart_* below; 1.7e-3 is what the current constants need.
_INVERSE_ATOL = 1.7e-3

# Per-dtype tolerances for every RGB <-> YUV round trip in this file. These are *not* float
# precision: ``K_yuv2rgb @ K_rgb2yuv`` is off the identity by up to 1.356e-3 (B, at rgb=(1,1,0))
# and ``K_rgb2yuv @ K_yuv2rgb`` by up to 6.472e-4 over the documented YUV domain, so a round trip
# cannot do better than that at any precision. See kornia#4044 -- when it lands these should fall
# to the dtype noise floor. float16/bfloat16 are widened to their own measured floors instead.
_ROUND_TRIP_TOL = {
    torch.float64: (1e-5, 1.5e-3),
    torch.float32: (1e-4, 1.5e-3),
    torch.float16: (1e-3, 2.5e-3),
    torch.bfloat16: (8e-3, 1.5e-2),
}


def _round_trip_tol(dtype):
    return _ROUND_TRIP_TOL.get(dtype, (1e-4, 1.5e-3))


# rgb = (1, 1, 0) is the analytic worst case of the #4044 round trip, and its expected B is
# exactly 0.0 -- so ``rtol`` contributes nothing there and the whole budget is ``atol``. The
# shared _ROUND_TRIP_TOL is sized for the seeded random input asserted beside it; this cell sits
# *at* the analytic maximum, where the measured error is 1.35604e-3 (cpu, float32 and float64
# alike -- the dtype contributes ~4e-8) and would consume 90% of that 1.5e-3. Its own 1.8e-3
# keeps it ~25% clear, so a regression that widened the defect stays distinguishable from one
# that merely reached the edge.
#
# That is looser than the three tightest assertions in the file, which do sit at ~10% free
# (TestYuvToRgb::test_unit[blue] 1.5346e-3, TestYuv422ToRgb::test_unit_upsampling 1.5322e-3,
# TestYuvToRgb::test_unit_matches_the_reference_relations 1.5097e-3, all against _INVERSE_ATOL's
# 1.7e-3). Those are deliberately not padded: they are one transform applied to exact decimal
# constants, so the derived 1.535e-3 bound above *is* the value they measure, and widening them
# would stop pinning it. Backend noise is not budgeted in anywhere -- see _cudnn_tf32_disabled.
_WORST_CORNER_ATOL = 1.8e-3


def _skip_without_real_float64(device) -> None:
    # The two #4044 constant pins below hardcode float64 and assert raw deviations at 1e-9, so
    # they need a backend that actually computes in float64. MPS has none at all, and XLA
    # executes a float64 request as float32 (the ``tpu`` fixture in conftest is
    # ``xm.xla_device()``), which moves the pinned values by 3.9e-8 (round-trip error) and
    # 4.4e-8 (U -> B coefficient) -- ~40x the tolerance -- for a pure precision reason.
    if "xla" in device.type or device.type == "mps":
        pytest.skip(f"{device.type} does not compute in float64")


def _unit_atol(base: float, dtype: torch.dtype) -> float:
    """Widen an analytic tolerance by the dtype's own representation error.

    Call sites pass the result as ``atol`` with ``rtol=0.0``. Every bound in this file is an
    *absolute* disagreement between two sets of constants, so handing the same number to
    ``rtol`` as well would quietly double the budget wherever ``|expected| ~ 1``, and the
    derivations above would describe something tighter than what is actually asserted.
    """
    return base + {torch.float64: 0.0, torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 1.2e-2}.get(dtype, 0.0)


@pytest.fixture(autouse=True)
def _cudnn_tf32_disabled(request):
    """Compute in real float32 on CUDA by default, so every tolerance here means what it derives.

    Off cpu, ``_apply_linear_transformation`` takes the ``F.conv2d`` branch, and cuDNN keeps
    PyTorch's ``allow_tf32 = True`` default -- conftest's ``--tf32`` flag drives only
    ``set_float32_matmul_precision``, which does not reach cuDNN. TF32 rounds both conv inputs
    and every kernel literal to 10 mantissa bits, so one forward transform carries up to
    ``2 * 2**-11 * sum|k_i * x_i|`` = 1.2e-3 of backend noise on the V row alone -- twice
    _FORWARD_ATOL, and the same order as the 1.356e-3 constants defect this file exists to pin.
    Every tolerance here is derived from the *constants*, not from a backend's kernel precision,
    so an ordinary run takes TF32 out of the picture rather than budgeting it into each assertion.

    The flag follows ``--tf32`` rather than being forced off, so this module does not become the
    one place in the suite that ignores the option: a ``--tf32`` run gets TF32 in cuDNN as well
    as in matmul, and measures the backend instead of the constants. The repo-wide fix is for
    ``pytest_sessionstart`` to set ``cudnn.allow_tf32`` from the same option -- until then this
    is the only module whose derived bounds depend on it.
    """
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = bool(request.config.getoption("--tf32"))
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


# The three generators below deliberately build on cpu/float64 and leave the move to the
# fixture's device and dtype to the call site: the reference model is evaluated in float64, and
# MPS has no float64 at all.


def _seeded_rand(*shape, seed):
    """Deterministic uniform noise in ``[0, 1)``, cpu/float64, identical on every machine."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=generator, dtype=torch.float64)


def _seeded_yuv(*shape, seed):
    """Deterministic YUV noise inside the documented domain, ``(*, 3, H, W)``, cpu/float64."""
    yuv = _seeded_rand(*shape, seed=seed)
    yuv[..., 1, :, :] = (yuv[..., 1, :, :] * 2.0 - 1.0) * 0.436
    yuv[..., 2, :, :] = (yuv[..., 2, :, :] * 2.0 - 1.0) * 0.615
    return yuv


def _color_image(names, height, width):
    """Tile ``names`` (row-major, one per pixel) into a ``(3, H, W)`` cpu/float64 image."""
    rgb = torch.empty(3, height, width, dtype=torch.float64)
    for index, name in enumerate(names):
        row, col = divmod(index, width)
        rgb[:, row, col] = torch.tensor(_REFERENCE_COLORS[name][0], dtype=torch.float64)
    return rgb


class TestRgbToYuv(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(3, 4, 5, device=device, dtype=dtype)
        out = kornia.color.rgb_to_yuv(img)
        assert isinstance(out, torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        out = kornia.color.rgb_to_yuv(img)
        assert out.shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.rgb_to_yuv([0.0])

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv(torch.ones(1, 1, device=device, dtype=dtype))

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv(torch.ones(2, 1, 1, device=device, dtype=dtype))

    @pytest.mark.parametrize("name", list(_REFERENCE_COLORS))
    def test_unit(self, device, dtype, name):
        rgb_values, yuv_values = _REFERENCE_COLORS[name]
        rgb = torch.tensor(rgb_values, device=device, dtype=dtype).view(3, 1, 1)
        expected = torch.tensor(yuv_values, device=device, dtype=dtype).view(3, 1, 1)

        self.assert_close(kornia.color.rgb_to_yuv(rgb), expected, atol=_unit_atol(_FORWARD_ATOL, dtype), rtol=0.0)

    def test_unit_invariants(self, device, dtype):
        rgb = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # red
                [0.0, 1.0, 0.0],  # green
                [0.0, 0.0, 1.0],  # blue
                [1.0, 1.0, 1.0],  # white
                [0.0, 0.0, 0.0],  # black
            ],
            device=device,
            dtype=dtype,
        ).view(5, 3, 1, 1)

        yuv = kornia.color.rgb_to_yuv(rgb)

        # shape preserved
        assert yuv.shape == rgb.shape

        Y = yuv[:, 0, 0, 0]

        # basic luminance ordering invariants
        assert Y[3] > Y[4]  # white > black
        assert Y[1] > Y[2]  # green generally brighter than blue

        # neutral colors have near-zero chroma. The tolerance is dtype-aware: the fixed 1e-4
        # this test used to assert is below the representation error of float16/bfloat16, so it
        # failed on every half-precision run of the suite.
        chroma_atol = _unit_atol(1e-4, dtype)
        self.assert_close(yuv[3, 1:], torch.zeros_like(yuv[3, 1:]), atol=chroma_atol, rtol=0.0)
        self.assert_close(yuv[4], torch.zeros_like(yuv[4]), atol=chroma_atol, rtol=0.0)

    def test_round_trip_rgb_yuv_rgb(self, device, dtype):
        # Seeded: with unseeded ``torch.rand`` this test failed on roughly one run in eight,
        # because the systematic error of kornia#4044 (1.356e-3) sits *outside* the 1e-3 the
        # test used to assert. The tolerance now states the defect instead of racing it, and
        # the worst-case corner is checked explicitly rather than waited for.
        rtol, atol = _round_trip_tol(dtype)

        rgb = _seeded_rand(3, 4, 5, seed=4045).to(device=device, dtype=dtype)
        self.assert_close(kornia.color.yuv_to_rgb(kornia.color.rgb_to_yuv(rgb)), rgb, atol=atol, rtol=rtol)

        # rgb = (1, 1, 0) maximises the B-channel round-trip error over the whole unit cube. Its
        # expected B is exactly zero, so this cell gets its own atol -- see _WORST_CORNER_ATOL.
        worst = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=dtype).view(3, 1, 1)
        self.assert_close(
            kornia.color.yuv_to_rgb(kornia.color.rgb_to_yuv(worst)),
            worst,
            atol=max(atol, _WORST_CORNER_ATOL),
            rtol=rtol,
        )

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_yuv, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img), op_optimized(img))

    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        module = kornia.color.RgbToYuv().to(device, dtype)
        self.assert_close(module(img), kornia.color.rgb_to_yuv(img))


class TestRgbToYuv420(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(3, 4, 6, device=device, dtype=dtype)
        out = kornia.color.rgb_to_yuv420(img)
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2), (3, 3, 3, 4, 4)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-2] //= 2
        shapeuv[-1] //= 2
        out = kornia.color.rgb_to_yuv420(img)
        assert out[0].shape == tuple(shapey)
        assert out[1].shape == tuple(shapeuv)

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.rgb_to_yuv420([0.0])

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv420(torch.ones(1, 1, device=device, dtype=dtype))

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv420(torch.ones(2, 1, 1, device=device, dtype=dtype))

        # Odd W and odd H are both rejected: chroma is subsampled 2x2, so neither can be halved.
        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv420(torch.ones(3, 2, 1, device=device, dtype=dtype))

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv420(torch.ones(3, 1, 2, device=device, dtype=dtype))

    @pytest.mark.parametrize("name", list(_REFERENCE_COLORS))
    def test_unit(self, device, dtype, name):
        # A flat 2x2 patch: luma is the colour's Y at full resolution, chroma is its (U, V)
        # in the single 2x2 cell. This pins the transform itself, not the subsampling.
        rgb_values, yuv_values = _REFERENCE_COLORS[name]
        rgb = torch.tensor(rgb_values, device=device, dtype=dtype).view(3, 1, 1).expand(3, 2, 2).contiguous()

        y, uv = kornia.color.rgb_to_yuv420(rgb)

        atol = _unit_atol(_FORWARD_ATOL, dtype)
        expected_y = torch.full((1, 2, 2), yuv_values[0], device=device, dtype=dtype)
        expected_uv = torch.tensor(yuv_values[1:], device=device, dtype=dtype).view(2, 1, 1)
        self.assert_close(y, expected_y, atol=atol, rtol=0.0)
        self.assert_close(uv, expected_uv, atol=atol, rtol=0.0)

    def test_unit_subsampling(self, device, dtype):
        # Four different colours inside each 2x2 cell, and two cells that must not be mixed, so
        # the chroma of a cell pins the whole box average rather than any one of its pixels. The
        # expected chroma is averaged with an explicit reshape, which does not share the
        # library's ``unfold`` grouping, so a wrong axis, stride or cell boundary shows up here.
        # Both cells are chosen to have non-zero chroma -- with ``blue, white / red, green`` the
        # second cell box-averages to (0, 0), where a scale, sign flip or U/V swap confined to
        # that cell would map 0 to 0 and pass.
        rgb = _color_image(["red", "green", "blue", "white", "black", "gray", "red", "gray"], 2, 4)
        reference = _rgb_to_yuv_reference(rgb)
        expected_y = reference[:1]
        expected_uv = reference[1:].reshape(2, 1, 2, 2, 2).mean((-3, -1))

        y, uv = kornia.color.rgb_to_yuv420(rgb.to(device=device, dtype=dtype))

        atol = _unit_atol(_FORWARD_ATOL, dtype)
        self.assert_close(y, expected_y.to(device=device, dtype=dtype), atol=atol, rtol=0.0)
        self.assert_close(uv, expected_uv.to(device=device, dtype=dtype), atol=atol, rtol=0.0)
        # The two cells really do differ, so the assertion above has something to discriminate.
        assert not torch.allclose(uv[:, :, 0], uv[:, :, 1])

    def test_forth_and_back(self, device, dtype):
        # 2x2-constant input, so the chroma subsample-then-upsample is lossless and what is
        # measured is the colour transform. Tolerance is bounded by kornia#4044.
        rtol, atol = _round_trip_tol(dtype)
        data = (
            _seeded_rand(3, 4, 5, seed=420)
            .to(device=device, dtype=dtype)
            .repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
        )

        y, uv = kornia.color.rgb_to_yuv420(data)
        self.assert_close(kornia.color.yuv420_to_rgb(y, uv), data, atol=atol, rtol=rtol)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_yuv420, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv420
        op_jit = torch.jit.script(op)
        self.assert_close(op(img)[0], op_jit(img)[0])
        self.assert_close(op(img)[1], op_jit(img)[1])

    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv420
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img)[0], op_optimized(img)[0])
        self.assert_close(op(img)[1], op_optimized(img)[1])

    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        module = kornia.color.RgbToYuv420().to(device, dtype)
        expected = kornia.color.rgb_to_yuv420(img)
        self.assert_close(module(img)[0], expected[0])
        self.assert_close(module(img)[1], expected[1])


class TestRgbToYuv422(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(3, 4, 6, device=device, dtype=dtype)
        out = kornia.color.rgb_to_yuv422(img)
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2), (3, 3, 3, 4, 4)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-1] //= 2
        out = kornia.color.rgb_to_yuv422(img)
        assert out[0].shape == tuple(shapey)
        assert out[1].shape == tuple(shapeuv)

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.rgb_to_yuv422([0.0])

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv422(torch.ones(1, 1, device=device, dtype=dtype))

        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv422(torch.ones(2, 1, 1, device=device, dtype=dtype))

        # Odd W is rejected: 4:2:2 halves the chroma width.
        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv422(torch.ones(3, 2, 1, device=device, dtype=dtype))

        # Odd H is rejected too, even though 4:2:2 subsamples width only. Pinned because the
        # guard is shared verbatim with rgb_to_yuv420 and may well be over-strict here: if it
        # is ever relaxed to the width test alone, that is a behavior change, not a cleanup.
        with pytest.raises(ShapeError):
            kornia.color.rgb_to_yuv422(torch.ones(3, 3, 4, device=device, dtype=dtype))

    @pytest.mark.parametrize("name", list(_REFERENCE_COLORS))
    def test_unit(self, device, dtype, name):
        rgb_values, yuv_values = _REFERENCE_COLORS[name]
        rgb = torch.tensor(rgb_values, device=device, dtype=dtype).view(3, 1, 1).expand(3, 2, 2).contiguous()

        y, uv = kornia.color.rgb_to_yuv422(rgb)

        atol = _unit_atol(_FORWARD_ATOL, dtype)
        expected_y = torch.full((1, 2, 2), yuv_values[0], device=device, dtype=dtype)
        expected_uv = torch.tensor(yuv_values[1:], device=device, dtype=dtype).view(2, 1, 1).expand(2, 2, 1)
        self.assert_close(y, expected_y, atol=atol, rtol=0.0)
        self.assert_close(uv, expected_uv, atol=atol, rtol=0.0)

    def test_unit_subsampling(self, device, dtype):
        # 4:2:2 pairs pixels *horizontally only*; rows stay independent. The expected chroma is
        # averaged over the last axis with an explicit reshape, so subsampling the wrong axis
        # (the failure this file could not previously see) fails here.
        rgb = _color_image(["red", "green", "blue", "white", "black", "red", "green", "blue"], 2, 4)
        reference = _rgb_to_yuv_reference(rgb)
        expected_y = reference[:1]
        expected_uv = reference[1:].reshape(2, 2, 2, 2).mean(-1)

        y, uv = kornia.color.rgb_to_yuv422(rgb.to(device=device, dtype=dtype))

        atol = _unit_atol(_FORWARD_ATOL, dtype)
        self.assert_close(y, expected_y.to(device=device, dtype=dtype), atol=atol, rtol=0.0)
        self.assert_close(uv, expected_uv.to(device=device, dtype=dtype), atol=atol, rtol=0.0)
        # Rows differ, and the chroma plane is neither its own transpose nor its own 180-degree
        # rotation (the palindrome ``red, green, blue, white / white, blue, green, red`` is
        # both), so an axis swap in the subsample cannot pass by accident.
        assert not torch.allclose(uv[:, 0], uv[:, 1])

    def test_forth_and_back(self, device, dtype):
        # 1x2-constant input, so the horizontal chroma subsample is lossless.
        rtol, atol = _round_trip_tol(dtype)
        data = _seeded_rand(3, 4, 5, seed=422).to(device=device, dtype=dtype).repeat_interleave(2, dim=-1)

        y, uv = kornia.color.rgb_to_yuv422(data)
        self.assert_close(kornia.color.yuv422_to_rgb(y, uv), data, atol=atol, rtol=rtol)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_yuv422, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv422
        op_jit = torch.jit.script(op)
        self.assert_close(op(img)[0], op_jit(img)[0])
        self.assert_close(op(img)[1], op_jit(img)[1])

    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv422
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img)[0], op_optimized(img)[0])
        self.assert_close(op(img)[1], op_optimized(img)[1])

    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        module = kornia.color.RgbToYuv422().to(device, dtype)
        expected = kornia.color.rgb_to_yuv422(img)
        self.assert_close(module(img)[0], expected[0])
        self.assert_close(module(img)[1], expected[1])


class TestYuvToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(3, 4, 5, device=device, dtype=dtype)
        assert isinstance(kornia.color.yuv_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.yuv_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.yuv_to_rgb([0.0])

        with pytest.raises(ShapeError):
            kornia.color.yuv_to_rgb(torch.ones(1, 1, device=device, dtype=dtype))

        with pytest.raises(ShapeError):
            kornia.color.yuv_to_rgb(torch.ones(2, 1, 1, device=device, dtype=dtype))

    @pytest.mark.parametrize("name", list(_REFERENCE_COLORS))
    def test_unit(self, device, dtype, name):
        # The YUV of each reference colour must map back to that colour. ``_INVERSE_ATOL`` is
        # 1.7e-3 rather than the 5e-4 the forward direction manages, because of kornia#4044.
        rgb_values, yuv_values = _REFERENCE_COLORS[name]
        yuv = torch.tensor(yuv_values, device=device, dtype=dtype).view(3, 1, 1)
        expected = torch.tensor(rgb_values, device=device, dtype=dtype).view(3, 1, 1)

        self.assert_close(kornia.color.yuv_to_rgb(yuv), expected, atol=_unit_atol(_INVERSE_ATOL, dtype), rtol=0.0)

    def test_unit_matches_the_reference_relations(self, device, dtype):
        yuv = _seeded_yuv(2, 3, 4, 5, seed=1140)
        expected = _yuv_to_rgb_reference(yuv)

        self.assert_close(
            kornia.color.yuv_to_rgb(yuv.to(device=device, dtype=dtype)),
            expected.to(device=device, dtype=dtype),
            atol=_unit_atol(_INVERSE_ATOL, dtype),
            rtol=0.0,
        )

    @pytest.mark.xfail(
        reason="kornia#4044: yuv_to_rgb's matrix is a separately rounded copy of the published "
        "BT.470-5 inverse, not the inverse of rgb_to_yuv's matrix",
        strict=True,
    )
    def test_convention_yuv_to_rgb_inverts_rgb_to_yuv_4044(self, device):
        # Intended behavior: yuv_to_rgb undoes rgb_to_yuv to the precision of the input dtype.
        # It does not. ``K_yuv2rgb @ K_rgb2yuv - I`` is
        #   [[ 0.000100, -0.000100,  0.000000],
        #    [-0.000103,  0.000659, -0.000556],
        #    [ 0.000737,  0.000619, -0.001356]]
        # so the round trip loses up to 1.356e-3 of blue at rgb = (1, 1, 0), independently of
        # dtype -- float64 fails exactly as float32 does, because the error is in the constants.
        # float64 is hardcoded and the dtype fixture dropped so that the mark cannot flip for a
        # precision reason; 1e-6 sits far above the float64 noise floor of a 3x3 matmul and far
        # below the 1.356e-3 being caught. Marked xfail(strict=True) so fixing #4044 makes this
        # XPASS and forces the mark out. Companion wart: test_wart_yuv_to_rgb_kernel_is_not_the_
        # inverse_4044.
        _skip_without_real_float64(device)

        rgb = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=torch.float64).view(3, 1, 1)
        rgb_back = kornia.color.yuv_to_rgb(kornia.color.rgb_to_yuv(rgb))

        assert (rgb_back - rgb).abs().max().item() < 1e-6, "kornia#4044: yuv_to_rgb is not the inverse of rgb_to_yuv"

    def test_wart_yuv_to_rgb_kernel_is_not_the_inverse_4044(self, device):
        # Wart pin for kornia#4044, companion to the strict xfail above: assert the CURRENT
        # error, so a partial fix cannot land unnoticed. Two cells:
        #   (0) the round trip at rgb = (1, 1, 0) still loses 1.356e-3 of blue -- flips as soon
        #       as any of the four inverse literals moves;
        #   (1) the single worst literal is still 2.029 where the exact inverse of kornia's own
        #       forward kernel wants 2.03199968 -- flips on the minimal fix (2.029 -> 2.032).
        # If either fails, #4044 was (partly) fixed -- flip/remove the strict xfail above. NOT a
        # contract that yuv_to_rgb must keep its current constants.
        # Snippet used to generate expected (torch only, cpu float64):
        #   K    = [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]]
        #   Kinv = [[1.0, 0.0, 1.14], [1.0, -0.396, -0.581], [1.0, 2.029, 0.0]]
        #   M = torch.tensor(Kinv) @ torch.tensor(K) - torch.eye(3)
        #   M[2, 0] + M[2, 1]                        # B error at rgb=(1,1,0)  ->  0.001356
        #   torch.linalg.inv(torch.tensor(K))[2, 1]  # the U -> B coefficient  ->  2.03199968...
        # atol 1e-9 on the error magnitude sits six orders below the 1.356e-3 being pinned and
        # well above the float64 noise of a 3x3 matmul.
        _skip_without_real_float64(device)

        rgb = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=torch.float64).view(3, 1, 1)
        rgb_back = kornia.color.yuv_to_rgb(kornia.color.rgb_to_yuv(rgb))
        assert abs((rgb_back - rgb)[2].item() - 0.001356) < 1e-9

        # The U -> B coefficient the library ships, read back through a unit U impulse.
        impulse = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64).view(3, 1, 1)
        u_to_b = kornia.color.yuv_to_rgb(impulse)[2].item()
        assert abs(u_to_b - 2.029) < 1e-9

    def test_forth_and_back(self, device, dtype):
        rtol, atol = _round_trip_tol(dtype)
        yuv = _seeded_yuv(3, 4, 5, seed=1141).to(device=device, dtype=dtype)

        self.assert_close(kornia.color.rgb_to_yuv(kornia.color.yuv_to_rgb(yuv)), yuv, atol=atol, rtol=rtol)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.yuv_to_rgb, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.yuv_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.color.yuv_to_rgb
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img), op_optimized(img))

    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        module = kornia.color.YuvToRgb().to(device, dtype)
        self.assert_close(module(img), kornia.color.yuv_to_rgb(img))


class TestYuv420ToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        imgy = torch.rand(1, 4, 6, device=device, dtype=dtype)
        imguv = torch.rand(2, 2, 3, device=device, dtype=dtype)
        assert isinstance(kornia.color.yuv420_to_rgb(imgy, imguv), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2)])
    def test_cardinality(self, device, dtype, shape):
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-2] //= 2
        shapeuv[-1] //= 2

        imgy = torch.ones(shapey, device=device, dtype=dtype)
        imguv = torch.ones(shapeuv, device=device, dtype=dtype)
        assert kornia.color.yuv420_to_rgb(imgy, imguv).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.yuv420_to_rgb([0.0], [0.0])

        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 1, device=device, dtype=dtype)
            imguv = torch.ones(1, 1, device=device, dtype=dtype)
            kornia.color.yuv420_to_rgb(imgy, imguv)

        # Luma H and W must both be even.
        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 3, 4, device=device, dtype=dtype)
            imguv = torch.ones(2, 1, 2, device=device, dtype=dtype)
            kornia.color.yuv420_to_rgb(imgy, imguv)

        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 4, 3, device=device, dtype=dtype)
            imguv = torch.ones(2, 2, 1, device=device, dtype=dtype)
            kornia.color.yuv420_to_rgb(imgy, imguv)

        # Chroma must be exactly half the luma in both axes.
        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 4, 4, device=device, dtype=dtype)
            imguv = torch.ones(2, 4, 2, device=device, dtype=dtype)
            kornia.color.yuv420_to_rgb(imgy, imguv)

        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 4, 4, device=device, dtype=dtype)
            imguv = torch.ones(2, 2, 4, device=device, dtype=dtype)
            kornia.color.yuv420_to_rgb(imgy, imguv)

        # Luma must be single-channel. (1, 2, 2) / (2, 1, 1) is the accepted shape; the same
        # sizes with a 2-channel luma are rejected by the channel slot of the shape spec, which
        # the rank case above does not reach.
        with pytest.raises(ShapeError):
            imgy = torch.ones(2, 2, 2, device=device, dtype=dtype)
            imguv = torch.ones(2, 1, 1, device=device, dtype=dtype)
            kornia.color.yuv420_to_rgb(imgy, imguv)

    @pytest.mark.parametrize("name", list(_REFERENCE_COLORS))
    def test_unit(self, device, dtype, name):
        rgb_values, yuv_values = _REFERENCE_COLORS[name]
        y = torch.full((1, 2, 2), yuv_values[0], device=device, dtype=dtype)
        uv = torch.tensor(yuv_values[1:], device=device, dtype=dtype).view(2, 1, 1)
        expected = torch.tensor(rgb_values, device=device, dtype=dtype).view(3, 1, 1).expand(3, 2, 2)

        self.assert_close(kornia.color.yuv420_to_rgb(y, uv), expected, atol=_unit_atol(_INVERSE_ATOL, dtype), rtol=0.0)

    def test_unit_upsampling(self, device, dtype):
        # Four 2x2 cells, each with its own chroma, over a luma that varies per pixel. The
        # expected upsample is built by integer-division indexing rather than by
        # ``repeat_interleave``, so an upsample with the wrong factor or the wrong axis (4x1
        # instead of 2x2, say) does not agree with it.
        y = _seeded_rand(1, 4, 4, seed=4200)
        uv = _seeded_yuv(3, 2, 2, seed=4201)[1:]

        index = torch.arange(4) // 2
        uv_full = uv[:, index][:, :, index]
        expected = _yuv_to_rgb_reference(torch.cat([y, uv_full], dim=-3))

        out = kornia.color.yuv420_to_rgb(y.to(device=device, dtype=dtype), uv.to(device=device, dtype=dtype))

        expected = expected.to(device=device, dtype=dtype)
        self.assert_close(out, expected, atol=_unit_atol(_INVERSE_ATOL, dtype), rtol=0.0)
        # The four cells carry different chroma, so a mis-shaped upsample cannot pass silently.
        assert not torch.allclose(uv[:, 0, 0], uv[:, 1, 1])

    def test_forth_and_back(self, device, dtype):
        rtol, atol = _round_trip_tol(dtype)
        datay = _seeded_yuv(3, 4, 6, seed=4202)[:1].to(device=device, dtype=dtype)
        datauv = _seeded_yuv(3, 2, 3, seed=4203)[1:].to(device=device, dtype=dtype)

        out_y, out_uv = kornia.color.rgb_to_yuv420(kornia.color.yuv420_to_rgb(datay, datauv))
        self.assert_close(out_y, datay, atol=atol, rtol=rtol)
        self.assert_close(out_uv, datauv, atol=atol, rtol=rtol)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        imgy = torch.rand(2, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        imguv = torch.rand(2, 2, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.yuv420_to_rgb, (imgy, imguv), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        imgy = torch.ones(2, 1, 4, 4, device=device, dtype=dtype)
        imguv = torch.ones(2, 2, 2, 2, device=device, dtype=dtype)
        op = kornia.color.yuv420_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(imgy, imguv), op_jit(imgy, imguv))

    def test_dynamo(self, device, dtype, torch_optimizer):
        imgy = torch.rand(2, 1, 4, 4, device=device, dtype=dtype)
        imguv = torch.rand(2, 2, 2, 2, device=device, dtype=dtype)
        op = kornia.color.yuv420_to_rgb
        op_optimized = torch_optimizer(op)
        self.assert_close(op(imgy, imguv), op_optimized(imgy, imguv))

    def test_module(self, device, dtype):
        imgy = torch.ones(2, 1, 4, 4, device=device, dtype=dtype)
        imguv = torch.ones(2, 2, 2, 2, device=device, dtype=dtype)
        module = kornia.color.Yuv420ToRgb().to(device, dtype)
        self.assert_close(module(imgy, imguv), kornia.color.yuv420_to_rgb(imgy, imguv))


class TestYuv422ToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        imgy = torch.rand(1, 4, 6, device=device, dtype=dtype)
        imguv = torch.rand(2, 4, 3, device=device, dtype=dtype)
        assert isinstance(kornia.color.yuv422_to_rgb(imgy, imguv), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2)])
    def test_cardinality(self, device, dtype, shape):
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-1] //= 2

        imgy = torch.ones(shapey, device=device, dtype=dtype)
        imguv = torch.ones(shapeuv, device=device, dtype=dtype)
        assert kornia.color.yuv422_to_rgb(imgy, imguv).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises((TypeError, AttributeError)):
            kornia.color.yuv422_to_rgb([0.0], [0.0])

        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 1, device=device, dtype=dtype)
            imguv = torch.ones(1, 1, device=device, dtype=dtype)
            kornia.color.yuv422_to_rgb(imgy, imguv)

        # Luma W must be even.
        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 4, 3, device=device, dtype=dtype)
            imguv = torch.ones(2, 4, 1, device=device, dtype=dtype)
            kornia.color.yuv422_to_rgb(imgy, imguv)

        # Odd luma H is rejected too, even though 4:2:2 subsamples width only. Pinned because
        # the guard is shared verbatim with yuv420_to_rgb and may well be over-strict here.
        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 3, 4, device=device, dtype=dtype)
            imguv = torch.ones(2, 3, 2, device=device, dtype=dtype)
            kornia.color.yuv422_to_rgb(imgy, imguv)

        # Chroma must be exactly half the luma width.
        with pytest.raises(ShapeError):
            imgy = torch.ones(1, 4, 4, device=device, dtype=dtype)
            imguv = torch.ones(2, 4, 4, device=device, dtype=dtype)
            kornia.color.yuv422_to_rgb(imgy, imguv)

        # Luma must be single-channel: rejected by the channel slot of the shape spec, which the
        # rank case above does not reach. Note this is *not* a width-relation case -- 2/1 == 2
        # holds; the height relation it looks like it covers is unchecked, see kornia#4050.
        with pytest.raises(ShapeError):
            imgy = torch.ones(2, 2, 2, device=device, dtype=dtype)
            imguv = torch.ones(2, 1, 1, device=device, dtype=dtype)
            kornia.color.yuv422_to_rgb(imgy, imguv)

    def test_wart_chroma_height_is_unchecked_4050(self, device, dtype):
        # Wart pin for kornia#4050. yuv422_to_rgb validates only the *width* ratio, so a chroma
        # plane whose height does not match the luma reaches ``torch.cat`` and dies there with a
        # bare RuntimeError instead of the ShapeError every other malformed input gets -- the
        # 4:2:0 twin checks both axes. ShapeError does not derive from RuntimeError, so adding
        # the missing guard flips this test: delete it then, and move the case up into
        # test_exception alongside the other ShapeError raise-sites.
        imgy = torch.ones(1, 2, 2, device=device, dtype=dtype)
        imguv = torch.ones(2, 1, 1, device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            kornia.color.yuv422_to_rgb(imgy, imguv)

    @pytest.mark.parametrize("name", list(_REFERENCE_COLORS))
    def test_unit(self, device, dtype, name):
        rgb_values, yuv_values = _REFERENCE_COLORS[name]
        y = torch.full((1, 2, 2), yuv_values[0], device=device, dtype=dtype)
        uv = torch.tensor(yuv_values[1:], device=device, dtype=dtype).view(2, 1, 1).expand(2, 2, 1).contiguous()
        expected = torch.tensor(rgb_values, device=device, dtype=dtype).view(3, 1, 1).expand(3, 2, 2)

        self.assert_close(kornia.color.yuv422_to_rgb(y, uv), expected, atol=_unit_atol(_INVERSE_ATOL, dtype), rtol=0.0)

    def test_unit_upsampling(self, device, dtype):
        # 4:2:2 duplicates chroma horizontally only; rows must stay independent.
        y = _seeded_rand(1, 4, 4, seed=4220)
        uv = _seeded_yuv(3, 4, 2, seed=4221)[1:]

        index = torch.arange(4) // 2
        uv_full = uv[:, :, index]
        expected = _yuv_to_rgb_reference(torch.cat([y, uv_full], dim=-3))

        out = kornia.color.yuv422_to_rgb(y.to(device=device, dtype=dtype), uv.to(device=device, dtype=dtype))

        expected = expected.to(device=device, dtype=dtype)
        self.assert_close(out, expected, atol=_unit_atol(_INVERSE_ATOL, dtype), rtol=0.0)
        # Rows carry different chroma, so a 2x2 upsample would not agree with the reference.
        assert not torch.allclose(uv[:, 0], uv[:, 1])

    def test_forth_and_back(self, device, dtype):
        rtol, atol = _round_trip_tol(dtype)
        datay = _seeded_yuv(3, 4, 6, seed=4222)[:1].to(device=device, dtype=dtype)
        datauv = _seeded_yuv(3, 4, 3, seed=4223)[1:].to(device=device, dtype=dtype)

        out_y, out_uv = kornia.color.rgb_to_yuv422(kornia.color.yuv422_to_rgb(datay, datauv))
        self.assert_close(out_y, datay, atol=atol, rtol=rtol)
        self.assert_close(out_uv, datauv, atol=atol, rtol=rtol)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        imgy = torch.rand(2, 1, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        imguv = torch.rand(2, 2, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.yuv422_to_rgb, (imgy, imguv), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        imgy = torch.ones(2, 1, 4, 4, device=device, dtype=dtype)
        imguv = torch.ones(2, 2, 4, 2, device=device, dtype=dtype)
        op = kornia.color.yuv422_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(imgy, imguv), op_jit(imgy, imguv))

    def test_dynamo(self, device, dtype, torch_optimizer):
        imgy = torch.rand(2, 1, 4, 4, device=device, dtype=dtype)
        imguv = torch.rand(2, 2, 4, 2, device=device, dtype=dtype)
        op = kornia.color.yuv422_to_rgb
        op_optimized = torch_optimizer(op)
        self.assert_close(op(imgy, imguv), op_optimized(imgy, imguv))

    def test_module(self, device, dtype):
        imgy = torch.ones(2, 1, 4, 4, device=device, dtype=dtype)
        imguv = torch.ones(2, 2, 4, 2, device=device, dtype=dtype)
        module = kornia.color.Yuv422ToRgb().to(device, dtype)
        self.assert_close(module(imgy, imguv), kornia.color.yuv422_to_rgb(imgy, imguv))
