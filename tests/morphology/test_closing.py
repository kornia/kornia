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

from kornia.morphology import closing, dilation, erosion
from kornia.morphology import morphology as morphology_module

from testing.base import BaseTester, assert_close, supports_replicate_padding
from testing.parametrized_tester import parametrized_test


@parametrized_test(
    smoke_inputs=lambda device, dtype: (
        torch.rand(1, 3, 4, 4, device=device, dtype=dtype),
        torch.ones((3, 3), device=device, dtype=dtype),
    ),
    cardinality_tests=[
        {
            "inputs": lambda device, dtype: (
                torch.ones((1, 3, 4, 4), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([1, 3, 4, 4]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((2, 3, 2, 4), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([2, 3, 2, 4]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((3, 3, 4, 1), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([3, 3, 4, 1]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((3, 2, 5, 5), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([3, 2, 5, 5]),
        },
    ],
    gradcheck_inputs=lambda device: (
        torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=torch.float64),
        torch.rand(3, 3, requires_grad=True, device=device, dtype=torch.float64),
    ),
)
class TestClosing(BaseTester):
    def setup_method(self) -> None:
        self.func = closing

    def test_kernel(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.7, 1.0, 0.8], [0.7, 0.7, 0.8], [0.7, 0.9, 0.8]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(closing(tensor, kernel), expected, atol=1e-4, rtol=1e-4)

    def test_structural_element(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        structural_element = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype
        )
        expected = torch.tensor([[0.7, 1.0, 0.8], [0.7, 0.7, 0.8], [0.7, 0.9, 0.8]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(
            closing(tensor, torch.ones_like(structural_element), structuring_element=structural_element),
            expected,
        )

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert closing([0.0], kernel)

        with pytest.raises(TypeError):
            assert closing(tensor, [0.0])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert closing(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert closing(tensor, test)

    def test_jit(self, device, dtype):
        op = closing
        op_script = torch.jit.script(op)

        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(tensor, kernel)
        expected = op(tensor, kernel)

        assert_close(actual, expected)

    def test_closing_custom_origin_is_extensive_and_idempotent(self, device, dtype):
        # closing = erosion(dilation(x)) must stay extensive (closing(x) >= x) and idempotent
        # (closing(closing(x)) == closing(x)) under a custom origin too, not just the default
        # centred one. `dilation`'s origin bug broke both for origin=[0, 0]. Mirrors
        # TestOpening.test_opening_custom_origin_is_anti_extensive_and_idempotent. The `>=` on
        # the rand fixture never needs a tolerance (selection only, no interpolation), and
        # repeating `closing` on its own (already-closed) output is likewise exact, so
        # `torch.equal` is fine.
        # Generated with:
        #   torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0))
        # A local `torch.Generator` avoids touching the process-global (and any device) RNG state.
        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        closed = closing(tensor, kernel, origin=[0, 0])
        assert (closed >= tensor).all()
        assert torch.equal(closing(closed, kernel, origin=[0, 0]), closed)

    def test_convention_closing_is_a_morphological_closing(self, device, dtype, monkeypatch):
        # `closing` is `erosion(dilation(x))` with the SAME kernel in both halves, so it is a
        # morphological closing: extensive and idempotent for an asymmetric kernel. That holds EXACTLY
        # for the two kernels below at the default origin (and, for ones(3, 3) at origin=[0, 0], in
        # the custom-origin test above); the last block of this test pins the kernel where it does NOT
        # hold exactly, so the docstring's `max_val` qualification is on record rather than assumed.
        # scikit-image mirrors the footprint inside `closing`, which makes it kornia's closing by the
        # FLIPPED kernel: on the 7x10 rand(seed 0) frame `sm.closing(x, A, mode="ignore")` is bit-equal
        # to `closing(x, A.flip((0, 1)))` and differs from `closing(x, A)` on 5 border pixels of
        # column 0 (the interiors agree); the same flipped-kernel equality holds for the L kernel and, at
        # the default origin, for the even kernels [[1, 0]], ones(2, 2), [[1, 1, 0, 1]] and [[0, 1, 1, 1]]
        # (`torch.rand(1, 1, 7, 10, dtype=torch.float64)` with `torch.Generator().manual_seed(s)`, s = 0..2).
        # scipy has no ignore mode: `grey_closing(..., mode="constant",
        # cval=inf)` pads its dilation half with +inf as well, so it is extensive but differs from
        # kornia's at the border (0.40 for A on `np.random.default_rng(0).random((7, 9))`) and returns inf on
        # the whole first column for `[[1, 0, 0]]`; it equals `blk` below only because that frame's border is
        # already 0. OpenCV's `MORPH_CLOSE` composes without a flip and is not a closing for an asymmetric
        # kernel at all, nor for an even-sized one at its default anchor (`ones(2, 2)`: extensive on 0 of
        # 100 random 6x8 frames, where the odd symmetric `[[1, 0, 1]]` is on all 100).
        # `max`/`min` only select an already-present value of the input, so the idempotence compares
        # with `torch.equal` and the `>=` never needs a tolerance.
        # Generated with (scipy 1.17.1, scikit-image 0.26.0, opencv-python-headless 5.0.0, numpy 2.0.0):
        #   blk = np.zeros((9, 11), np.float32); blk[3:6, 3:7] = 1.0; A = np.array([[0, 1, 1]], bool)
        #   ndi.grey_closing(blk, footprint=A, mode="constant", cval=np.inf) == blk  -> True
        #   sm.closing(blk, A, mode="ignore") == blk                                 -> True
        #   cv2.morphologyEx(blk, cv2.MORPH_CLOSE, A.astype(np.uint8)) == blk        -> False
        #   torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)) for the invariants.
        # A local `torch.Generator` avoids touching the process-global (and any device) RNG state.
        asymmetric = torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=dtype)
        block = torch.zeros(1, 1, 9, 11, device=device, dtype=dtype)
        block[..., 3:6, 3:7] = 1.0
        assert torch.equal(closing(block, asymmetric), block)

        l_kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        closed = closing(tensor, l_kernel)
        assert (closed >= tensor).all()
        assert torch.equal(closing(closed, l_kernel), closed)

        # The docstring names `[[0, 1, 1]]` as exact too, so the invariants are executed for it on the
        # same frame rather than only the block equality above. Measured with kornia in this worktree
        # (torch 2.14.0, CPU) over 20 seeds of rand(1, 1, 7, 10) in float32 and float64: the worst
        # anti-extensivity, opening-idempotence, extensivity and closing-idempotence violation is 0.
        asymmetric_closed = closing(tensor, asymmetric)
        assert (asymmetric_closed >= tensor).all()
        assert torch.equal(closing(asymmetric_closed, asymmetric), asymmetric_closed)

        # The invariants are exact only up to the `max_val` sentinel, so the docstring qualifies them.
        # `[[1, 0, 0]]` at the default origin reads `x(p + 1)`, so its window leaves the image on the
        # right; `dilation` can then emit `x - max_val`, and the next stage's `+ max_val` returns `x`
        # quantised to `max_val`'s spacing, which can push `closing(x)` just BELOW `x`.
        # Generated with kornia in this worktree (torch 2.14.0, CPU, same rand(1, 1, 7, 10) seed 0):
        #   max(x - closing(x)).clamp(min=0):  float32 4.7034e-04, float64 0, float16 9.7998e-01,
        #   bfloat16 9.8047e-01   (float32 ULP of max_val=1e4 is 9.7656e-4; float16's is 8, bf16's 64)
        # The float64 0 is an artefact of this frame, which is float32 `rand` cast up and so has no bits
        # below float64's ULP of `max_val`: `torch.rand(..., dtype=torch.float64)` seed 0 falls short by
        # 7.8804e-13 (worst of 20 seeds 8.96e-13), under that ULP of 1.8190e-12 but not zero.
        # Over 20 seeds in all four dtypes the shortfall stays in columns W-2 and W-1, the column whose
        # dilation window is empty and the one before it, so everywhere else the extensivity is exact.
        # That confinement is what can still fail in half precision, where one ULP of `max_val` (8 in float16,
        # 64 in bfloat16) is wider than the data. `torch.finfo(dtype).eps * 8192` is that ULP for the default
        # `max_val=1e4`, which lies in [8192, 16384); over 1000 frames the miss stays within half of it.
        # On this non-negative frame idempotence survives this kernel exactly; it is the opening half that
        # loses it (see tests/morphology/test_opening.py). Tracked in #4734.
        one_ulp = torch.finfo(dtype).eps * 8192.0
        side_kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        side_closed = closing(tensor, side_kernel)
        shortfall = (tensor - side_closed).clamp(min=0)
        assert shortfall.max() < one_ulp
        assert not bool(shortfall[..., :-2].any())
        assert torch.equal(closing(side_closed, side_kernel), side_closed)
        genuine = tensor
        if dtype == torch.float64:
            genuine = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0), dtype=dtype).to(device)
            genuine_shortfall = (genuine - closing(genuine, side_kernel)).clamp(min=0).max()
            assert 0.0 < genuine_shortfall < one_ulp
        # On negative data the erosion half's empty window (column 0, `max_val + m`) carries the data into the
        # next closing's round trip, so the geodesic closing loses idempotence too, by less than that ULP.
        # Measured with kornia in this worktree (torch 2.14.0 and 2.5.1, CPU and MPS) on the negated frame:
        # float32 3.5977e-04 in column 1, float16 0.99707 and bfloat16 0.99609 in columns 1-2, genuine float64
        # 3.6238e-13.
        negative = -genuine
        negative_closed = closing(negative, side_kernel)
        drift = (closing(negative_closed, side_kernel) - negative_closed).abs()
        assert 0.0 < drift.max() < one_ulp

        # That is the geodesic story only. `dilation` reads `x(p + 1)` and `erosion` reads `y(p - 1)`, so
        # under `replicate` the first column of the closing is `x(1)`, not `x(0)`, and the closing is not
        # extensive at all (idempotence survives); under `circular` the two shifts cancel and the closing
        # is exactly `x`. Measured with kornia in this worktree (torch 2.14.0, CPU, float32) over
        # rand(1, 1, 7, 10) with `torch.Generator().manual_seed(s)`, s = 0..19: worst replicate extensivity
        # miss 0.914, worst replicate idempotence miss 0, worst circular deviation from `x` 0.
        # torch 2.5.1 has no float16 CPU `replication_pad2d`, so the replicate lines follow the probe.
        if supports_replicate_padding(device, dtype):
            dip = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype)[None, None]
            replicated = closing(dip, side_kernel, border_type="replicate")
            assert replicated.flatten().tolist() == [0.0, 0.0, 1.0]
            assert not bool((replicated >= dip).all())
            assert torch.equal(closing(replicated, side_kernel, border_type="replicate"), replicated)
        assert torch.equal(closing(tensor, side_kernel, border_type="circular"), tensor)

        # The composition itself: both halves get every option. `max_val=0.1` is inside the data range, so a
        # half that fell back to the default `1e4` would show.
        options = {"border_type": "constant", "border_value": 0.5, "origin": [0, 0], "max_val": 0.1}
        halves = erosion(dilation(tensor, l_kernel, **options), l_kernel, **options)
        assert torch.equal(closing(tensor, l_kernel, **options), halves)

        # ... and each half receives the caller's `engine`, which its result alone need not reveal.
        seen = []
        resolve = morphology_module._resolve_engine

        def record(engine, *args):
            seen.append(engine)
            return resolve(engine, *args)

        monkeypatch.setattr(morphology_module, "_resolve_engine", record)
        closing(tensor, l_kernel, engine="unfold")
        assert seen == ["unfold", "unfold"]
