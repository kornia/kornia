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

import kornia

from testing import assert_capture_matches_eager, assert_degenerate_path_parity, unrepresentable_sizes

from tests.testing import _historical as hist


class TestUnrepresentableSizes:
    def test_bfloat16_includes_the_known_traps(self):
        sizes = unrepresentable_sizes(torch.bfloat16)
        # 258/300/1000/2050/3000: ``n - 1`` rounds (wave-5 and wave-8 divisor bugs in kornia#4006).
        # 257: ``n`` itself rounds to 256 (wave-9 size bug). Both operands matter.
        for n in (257, 258, 300, 1000, 2050, 3000):
            assert n in sizes

    def test_bfloat16_excludes_exact_neighbourhoods(self):
        sizes = unrepresentable_sizes(torch.bfloat16)
        # Every integer up to 256 is exact in bfloat16, so ``n`` and ``n - 1`` are both exact.
        assert all(n not in sizes for n in range(2, 257))

    def test_float16_boundary(self):
        sizes = unrepresentable_sizes(torch.float16)
        assert all(n not in sizes for n in range(2, 2049))
        for n in (2049, 2050, 3000, 3001):
            assert n in sizes

    def test_float32_is_empty_in_range(self):
        assert unrepresentable_sizes(torch.float32) == []
        assert unrepresentable_sizes(torch.float64) == []

    def test_is_sorted_and_bounded(self):
        sizes = unrepresentable_sizes(torch.bfloat16, lo=250, hi=270)
        assert sizes == sorted(sizes)
        assert sizes[0] >= 250 and sizes[-1] <= 270
        assert sizes == list(range(257, 271))  # every odd n is inexact; every even n has an odd n - 1

    def test_rejects_non_floating_dtype(self):
        with pytest.raises(TypeError, match="floating"):
            unrepresentable_sizes(torch.int64)


def _image_hw(size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor]:
    return (torch.zeros(1, 1, size, 4, device=device, dtype=dtype),)


def _meshgrid_of(fn):
    def run(image):
        return fn(image.shape[-2], image.shape[-1], device=image.device, dtype=image.dtype)

    return run


def _ntp_default_dtype(fn):
    def run(image):
        return fn(image.shape[-2], image.shape[-1], device=image.device)

    return run


class TestAssertCaptureMatchesEager:
    def test_current_meshgrid_passes_under_trace(self, device, dtype):
        assert_capture_matches_eager(
            _meshgrid_of(kornia.geometry.create_meshgrid),
            _image_hw,
            sizes=[1, 2, *unrepresentable_sizes(torch.bfloat16)[:6], 2050, 3000],
            device=device,
            dtype=dtype,
            capture="trace",
        )

    @pytest.mark.parametrize(
        ("wave_fn", "expected_size"),
        [(hist.create_meshgrid_9ed891c5, 257), (hist.create_meshgrid_32ab0eeb, 258)],
        ids=["9ed891c5", "32ab0eeb"],
    )
    def test_historical_meshgrid_bodies_fail_under_trace(self, wave_fn, expected_size, device):
        # These bodies are byte-equal to eager at every size that is exact in bfloat16, which is
        # why 1/2/4 let them through three review rounds. The sweep is what catches them -- and the
        # two bodies first differ at DIFFERENT sizes, which is why ``expected_size`` is pinned per
        # body rather than matched loosely: 9ed891c5 rounds the size itself, so at 257 its divisor
        # is 255 instead of 256 and it already differs; 32ab0eeb rounds only the divisor, and 256 is
        # exact in bfloat16, so 257 passes vacuously there and 258 is the first size that catches
        # it. A single hand-picked size catches one body or the other, never both.
        with pytest.raises(AssertionError, match=rf"size {expected_size}, output"):
            assert_capture_matches_eager(
                _meshgrid_of(wave_fn),
                _image_hw,
                sizes=unrepresentable_sizes(torch.bfloat16)[:6],
                device=device,
                dtype=torch.bfloat16,
                capture="trace",
            )

    def test_historical_meshgrid_bodies_pass_at_the_vacuous_sizes(self, device):
        # Documents the trap: the old test sizes cannot distinguish the buggy body from the fix.
        assert_capture_matches_eager(
            _meshgrid_of(hist.create_meshgrid_32ab0eeb),
            _image_hw,
            sizes=[1, 2, 4, 257],
            device=device,
            dtype=torch.bfloat16,
            capture="trace",
        )

    @pytest.mark.parametrize("default_dtype", [torch.float16, torch.bfloat16], ids=["float16", "bfloat16"])
    def test_historical_normal_transform_pixel_fails_under_half_default_dtype(self, default_dtype, device):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        try:
            with pytest.raises(AssertionError, match="size"):
                assert_capture_matches_eager(
                    _ntp_default_dtype(hist.normal_transform_pixel_1522441d),
                    lambda size, device, dtype: (torch.zeros(1, 1, size, 5, device=device),),
                    sizes=unrepresentable_sizes(default_dtype)[:4],
                    device=device,
                    dtype=default_dtype,
                    capture="trace",
                )
            assert_capture_matches_eager(
                _ntp_default_dtype(kornia.geometry.conversions.normal_transform_pixel),
                lambda size, device, dtype: (torch.zeros(1, 1, size, 5, device=device),),
                sizes=unrepresentable_sizes(default_dtype)[:4],
                device=device,
                dtype=default_dtype,
                capture="trace",
            )
        finally:
            torch.set_default_dtype(previous)

    def test_accepts_tuple_outputs(self, device):
        def two_outputs(image):
            a = kornia.geometry.create_meshgrid(
                image.shape[-2], image.shape[-1], device=image.device, dtype=image.dtype
            )
            return a, a + 1

        assert_capture_matches_eager(two_outputs, _image_hw, sizes=[3], device=device, dtype=torch.float32)

    def test_reports_index_and_difference_for_tuple_outputs(self, device):
        # Output 0 is the fixed library function and agrees under trace; output 1 is the wave-8 body
        # at 258, the size where it first diverges. The helper must name the offending index, not
        # merely the size.
        def good_then_bad(image):
            h, w = image.shape[-2], image.shape[-1]
            good = kornia.geometry.create_meshgrid(h, w, device=image.device, dtype=image.dtype)
            bad = hist.create_meshgrid_32ab0eeb(h, w, device=image.device, dtype=image.dtype)
            return good, bad

        with pytest.raises(AssertionError, match=r"size 258, output 1.*max abs diff"):
            assert_capture_matches_eager(good_then_bad, _image_hw, sizes=[258], device=device, dtype=torch.bfloat16)

    def test_rejects_an_empty_sweep(self, device):
        # unrepresentable_sizes is [] for float32/float64, so the documented-looking call
        # ``sizes=unrepresentable_sizes(dtype)`` would otherwise be a green test that checked
        # nothing under the default --dtype=float32 fixture.
        with pytest.raises(ValueError, match="at least one size"):
            assert_capture_matches_eager(
                _meshgrid_of(kornia.geometry.create_meshgrid),
                _image_hw,
                sizes=unrepresentable_sizes(torch.float32),
                device=device,
                dtype=torch.float32,
            )

    def test_compile_capture_runs_or_skips(self, device, torch_optimizer):
        # ``torch_optimizer`` is the conftest fixture; the test name carries ``compile`` so it is
        # deselected unless KORNIA_TEST_OPTIMIZER is set, exactly like the rest of the suite.
        del torch_optimizer
        assert_capture_matches_eager(
            _meshgrid_of(kornia.geometry.create_meshgrid),
            _image_hw,
            sizes=[1, 2, 258, 300],
            device=device,
            dtype=torch.float32,
            capture="compile",
        )


class TestAssertDegeneratePathParity:
    @staticmethod
    def _warp_kwargs(device, dsize):
        src = torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float32)
        m = torch.eye(2, 3, device=device, dtype=torch.float32)[None]
        return {"src": src, "M": m, "dsize": dsize}

    def test_current_warp_affine_validates_the_same_on_the_empty_path(self, device):
        if torch.device(device).type == "mps":
            # Not a parity gap to weaken around: even the baseline (non-empty src, valid M,
            # dsize=(0, 4)) crashes. kornia's empty-destination fast path still delegates to
            # ``F.grid_sample`` with a zero-sized grid for validation/autograd, and MPS's
            # grid_sampler asserts "Placeholder tensor is empty!" whenever the grid is
            # zero-shaped, even though src is not. Reproduces with a bare
            # ``F.grid_sample(torch.zeros(1, 1, 4, 4, device="mps"), torch.zeros(1, 0, 4, 2,
            # device="mps"), align_corners=True, mode="bilinear", padding_mode="zeros")``, so this
            # is a real, reportable MPS gap in warp_affine's degenerate path -- see
            # task-4-report.md.
            pytest.skip("warp_affine's empty-dsize path crashes on MPS: F.grid_sample asserts on a zero-sized grid")
        full = self._warp_kwargs(device, (4, 4))
        empty = self._warp_kwargs(device, (0, 4))
        bad = [
            ("M", torch.eye(2, 3, device=device, dtype=torch.int64)[None]),
            ("M", torch.eye(3, 3, device=device, dtype=torch.float32)[None]),
            # NOT an integral ``src`` (torch.zeros(1, 1, 4, 4, dtype=torch.int64)): on the full
            # path it reaches ``F.grid_sample``, which raises NotImplementedError
            # ("grid_sampler_2d_cpu_kernel_impl" not implemented for 'Long'); on the degenerate
            # (dsize with a zero dimension) path, ``_empty_warp_output_2d`` raises its own
            # RuntimeError ("Expected a floating point src, got torch.int64.") before ever
            # reaching grid_sample. Real discrepancy -- see task-4-report.md.
        ]
        if torch.device(device).type != "mps":
            # MPS has no float64 tensors at all (construction itself raises TypeError), so this
            # dtype-mismatch case can only be exercised on cpu/cuda.
            bad.append(("M", torch.zeros(1, 2, 3, device=device, dtype=torch.float64)))
        assert_degenerate_path_parity(kornia.geometry.transform.warp_affine, full, empty, bad)

    def test_detects_a_laxer_degenerate_path(self):
        def op(x, dsize):
            if dsize[0] == 0:
                return x.new_zeros(0)
            if x.dtype != torch.float32:
                raise TypeError("float32 only")
            return x

        with pytest.raises(AssertionError, match=r"x=.*TypeError.*vs.*None"):
            assert_degenerate_path_parity(
                op,
                {"x": torch.zeros(2), "dsize": (2,)},
                {"x": torch.zeros(2), "dsize": (0,)},
                [("x", torch.zeros(2, dtype=torch.int64))],
            )

    def test_detects_a_stricter_degenerate_path(self):
        def op(x, dsize):
            if dsize[0] == 0 and x.dtype != torch.float32:
                raise TypeError("float32 only on the empty path")
            return x

        with pytest.raises(AssertionError, match=r"None.*vs.*TypeError"):
            assert_degenerate_path_parity(
                op,
                {"x": torch.zeros(2), "dsize": (2,)},
                {"x": torch.zeros(2), "dsize": (0,)},
                [("x", torch.zeros(2, dtype=torch.int64))],
            )
