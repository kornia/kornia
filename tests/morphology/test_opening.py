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

from kornia.morphology import opening

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
class TestOpening(BaseTester):
    def setup_method(self) -> None:
        self.func = opening

    def test_kernel(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.5, 0.5, 0.3], [0.5, 0.3, 0.3], [0.4, 0.4, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(opening(tensor, kernel), expected, atol=1e-4, rtol=1e-4)

    def test_structural_element(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        structural_element = torch.tensor(
            [[-1.0, 0.0, -10.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype
        )
        expected = torch.tensor([[0.5, 0.5, 0.3], [0.5, 0.3, 0.3], [0.4, 0.4, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(
            opening(tensor, torch.ones_like(structural_element), structuring_element=structural_element),
            expected,
        )

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert opening([0.0], kernel)

        with pytest.raises(TypeError):
            assert opening(tensor, [0.0])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert opening(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert opening(tensor, test)

    def test_jit(self, device, dtype):
        op = opening
        op_script = torch.jit.script(op)

        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(tensor, kernel)
        expected = op(tensor, kernel)

        assert_close(actual, expected)

    def test_opening_custom_origin_is_anti_extensive_and_idempotent(self, device, dtype):
        # opening = dilation(erosion(x)) must stay anti-extensive (opening(x) <= x) and
        # idempotent (opening(opening(x)) == opening(x)) under a custom origin too, not just the
        # default centred one. `dilation`'s origin bug broke both for origin=[0, 0]. `block`
        # already equals its own opening under `ones(3, 3)`, so the equality check also exercises
        # idempotency; 0/1 fixtures compare exactly with `torch.equal`, the `<=` on the rand
        # fixture never needs a tolerance (selection only, no interpolation), and repeating
        # `opening` on its own (already-open) output is likewise exact.
        # Generated with:
        #   block = torch.zeros(1, 1, 7, 10); block[..., 2:5, 3:7] = 1
        #   torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0))
        # A local `torch.Generator` avoids touching the process-global (and any device) RNG state.
        block = torch.zeros(1, 1, 7, 10, device=device, dtype=dtype)
        block[..., 2:5, 3:7] = 1.0
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        assert torch.equal(opening(block, kernel, origin=[0, 0]), block)

        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        opened = opening(tensor, kernel, origin=[0, 0])
        assert (opened <= tensor).all()
        assert torch.equal(opening(opened, kernel, origin=[0, 0]), opened)

    def test_convention_opening_is_a_morphological_opening(self, device, dtype):
        # `opening` is `dilation(erosion(x))` with the SAME kernel in both halves. Because `dilation`
        # reflects the kernel and `erosion` does not, the composition is a morphological opening:
        # anti-extensive and idempotent for an asymmetric kernel, and it leaves a block that is a union
        # of translates of the kernel untouched. That holds EXACTLY for the two kernels below at the
        # default origin (and, for ones(3, 3) at origin=[0, 0], in the custom-origin test above); the
        # last block of this test pins the kernel where it does NOT hold exactly, so the docstring's
        # `max_val` qualification is on record rather than assumed.
        # OpenCV composes without a flip, so its `MORPH_OPEN` is not an opening for an asymmetric
        # kernel, nor for an even-sized one at its default anchor (`ones(2, 2)`: anti-extensive on 0 of
        # 100 random 6x8 frames, where the odd symmetric `[[1, 0, 1]]` is on all 100).
        # scikit-image mirrors the footprint inside `opening`, and on the 7x10 rand(seed 0)
        # frame `sm.opening(x, A, mode="ignore")` is bit-equal to `opening(x, A)`; so is the L kernel's.
        # For the even kernels [[1, 0]], ones(2, 2), [[1, 1, 0, 1]] and [[0, 1, 1, 1]] it is bit-equal to
        # `opening(x, K, origin=[(k_h - 1) // 2, (k_w - 1) // 2])` (rand(1, 1, 7, 10) seeds 0-2, float64),
        # where its erosion half anchors. For the three multi-cell ones it equals neither `opening(x, K)`
        # nor the flipped-kernel opening at the default origin (ones(2, 2): 0.805 apart on seed 0); for
        # `[[1, 0]]` the flipped-kernel opening and the earlier-origin one are both the identity, so it
        # equals that one too. scipy has no ignore
        # mode: `grey_opening(..., mode="constant", cval=-inf)` pads its erosion half with -inf as well,
        # so it is anti-extensive but differs from kornia's at the border (0.61 for A, 0.37 for
        # ones(3, 3) on `np.random.default_rng(0).random((7, 9))`) and returns -inf on the whole last column
        # for `[[1, 0, 0]]`; it equals `blk` below only because that frame's border is already 0.
        # 0/1 fixtures are exact in every dtype and `max`/`min` only ever select an already-present
        # value, so both the block equality and the idempotence compare with `torch.equal`.
        # Generated with (scipy 1.17.1, scikit-image 0.26.0, opencv-python-headless 5.0.0, numpy 2.0.0):
        #   blk = np.zeros((9, 11), np.float32); blk[3:6, 3:7] = 1.0; A = np.array([[0, 1, 1]], bool)
        #   ndi.grey_opening(blk, footprint=A, mode="constant", cval=-np.inf) == blk   -> True
        #   sm.opening(blk, A, mode="ignore") == blk                                   -> True
        #   cv2.morphologyEx(blk, cv2.MORPH_OPEN, A.astype(np.uint8)) == blk           -> False
        #   torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)) for the invariants.
        # A local `torch.Generator` avoids touching the process-global (and any device) RNG state.
        asymmetric = torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=dtype)
        block = torch.zeros(1, 1, 9, 11, device=device, dtype=dtype)
        block[..., 3:6, 3:7] = 1.0
        assert torch.equal(opening(block, asymmetric), block)

        l_kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        opened = opening(tensor, l_kernel)
        assert (opened <= tensor).all()
        assert torch.equal(opening(opened, l_kernel), opened)

        # The docstring names `[[0, 1, 1]]` as exact too, so the invariants are executed for it on the
        # same frame rather than only the block equality above. Measured with kornia in this worktree
        # (torch 2.14.0, CPU) over 20 seeds of rand(1, 1, 7, 10) in float32 and float64: the worst
        # anti-extensivity, opening-idempotence, extensivity and closing-idempotence violation is 0.
        asymmetric_opened = opening(tensor, asymmetric)
        assert (asymmetric_opened <= tensor).all()
        assert torch.equal(opening(asymmetric_opened, asymmetric), asymmetric_opened)

        # The invariants are exact only up to the `max_val` sentinel, so the docstring qualifies them.
        # `[[1, 0, 0]]` at the default origin reads `x(p + 1)`, so its window leaves the image on the
        # right; `dilation` can then emit `x - max_val`, and the next stage's `+ max_val` returns `x`
        # quantised to `max_val`'s spacing. Idempotence then misses by a fraction of that spacing --
        # exactly, not approximately, zero is what a sentinel-free implementation would give.
        # Generated with kornia in this worktree (torch 2.14.0, CPU, same rand(1, 1, 7, 10) seed 0):
        #   max|opening(opening(x)) - opening(x)|:  float32 2.0671e-04, float64 0, float16 9.2969e-01,
        #   bfloat16 9.2969e-01   (float32 ULP of max_val=1e4 is 9.7656e-4; float16's is 8, bf16's 64)
        # The float64 0 is an artefact of this frame, which is float32 `rand` cast up and so has no bits
        # below float64's ULP of `max_val`: `torch.rand(..., dtype=torch.float64)` seed 0 misses by
        # 6.6624e-13 (worst of 20 seeds 9.07e-13), under that ULP of 1.8190e-12 but not zero.
        # Over 20 seeds in all four dtypes the miss stays in columns W-3 and W-2, the two before the column
        # whose dilation window is empty, so everywhere else the idempotence is exact. That confinement is
        # what can still fail in half precision, where the bound `2 * eps * 1e4` (19.5 in float16, 156 in
        # bfloat16) is wider than the data.
        # anti-extensivity survives this kernel exactly; extensivity is the half that fails for
        # `closing` (see tests/morphology/test_closing.py). Tracked in #4734.
        side_kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        side_opened = opening(tensor, side_kernel)
        assert (side_opened <= tensor).all()
        deviation = (opening(side_opened, side_kernel) - side_opened).abs()
        assert deviation.max() <= 2.0 * torch.finfo(dtype).eps * 1e4
        assert not bool(deviation[..., :-3].any())
        if dtype == torch.float64:
            genuine = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0), dtype=dtype).to(device)
            genuine_opened = opening(genuine, side_kernel)
            genuine_miss = (opening(genuine_opened, side_kernel) - genuine_opened).abs().max()
            assert 0.0 < genuine_miss < torch.finfo(dtype).eps * 8192.0

        # That is the geodesic story only. `erosion` reads `x(p - 1)` and `dilation` reads `y(p + 1)`, so
        # under `replicate` the last column of the opening is `x(W - 2)`, not `x(W - 1)`, and the opening
        # is not anti-extensive at all (idempotence survives); under `circular` the two shifts cancel and
        # the opening is exactly `x`. Measured with kornia in this worktree (torch 2.14.0, CPU, float32)
        # over rand(1, 1, 7, 10) with `torch.Generator().manual_seed(s)`, s = 0..19: worst replicate
        # anti-extensivity miss 0.862, worst replicate idempotence miss 0, worst circular deviation from `x` 0.
        # torch 2.5.1 has no float16 CPU `replication_pad2d`, so the replicate lines follow the probe.
        if supports_replicate_padding(device, dtype):
            bump = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype)[None, None]
            replicated = opening(bump, side_kernel, border_type="replicate")
            assert replicated.flatten().tolist() == [0.0, 1.0, 1.0]
            assert not bool((replicated <= bump).all())
            assert torch.equal(opening(replicated, side_kernel, border_type="replicate"), replicated)
        assert torch.equal(opening(tensor, side_kernel, border_type="circular"), tensor)
