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

from kornia.morphology import closing

from testing.base import BaseTester, assert_close
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

    def test_convention_closing_is_a_morphological_closing(self, device, dtype):
        # `closing` is `erosion(dilation(x))` with the SAME kernel in both halves, so it is a
        # morphological closing: extensive and idempotent for an asymmetric kernel. That holds EXACTLY
        # for the two kernels below at the default origin (and, for ones(3, 3) at origin=[0, 0], in
        # the custom-origin test above); the last block of this test pins the kernel where it does NOT
        # hold exactly, so the docstring's `max_val` qualification is on record rather than assumed.
        # scipy's `grey_closing` uses the same convention. scikit-image mirrors the footprint inside
        # `closing`, so its result is a closing too, but it is NOT bit-equal to kornia's: on the 7x10
        # rand(seed 0) frame `sm.closing(x, A, mode="ignore")` differs from `closing(x, A)` on 5
        # border pixels of column 0 (the interiors agree). OpenCV's `MORPH_CLOSE` composes without a
        # flip and is not a closing for an asymmetric kernel at all.
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
        # idempotence survives this kernel exactly; it is the opening half that loses it
        # (see tests/morphology/test_opening.py). Tracked in #4734.
        side_kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        side_closed = closing(tensor, side_kernel)
        shortfall = (tensor - side_closed).clamp(min=0).max()
        assert shortfall <= 2.0 * torch.finfo(dtype).eps * 1e4
        assert torch.equal(closing(side_closed, side_kernel), side_closed)
