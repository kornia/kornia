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

    def test_convention_closing_is_a_morphological_closing(self, device, dtype):
        # `closing` is `erosion(dilation(x))` with the same kernel in both halves; as only `dilation`
        # reflects, it is extensive and idempotent for an asymmetric kernel too, and leaves a block that is
        # already closed untouched.
        # Generated with scipy 1.17.1 / scikit-image 0.26.0 / opencv-python-headless 5.0.0 / numpy 2.0.0:
        #   blk = np.zeros((9, 11), np.float32); blk[3:6, 3:7] = 1.0; A = np.array([[0, 1, 1]], bool)
        #   ndi.grey_closing(blk, footprint=A, mode="constant", cval=np.inf) == blk  -> True
        #   sm.closing(blk, A, mode="ignore") == blk                                 -> True
        #   cv2.morphologyEx(blk, cv2.MORPH_CLOSE, A.astype(np.uint8)) == blk        -> False
        asymmetric = torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=dtype)
        block = torch.zeros(1, 1, 9, 11, device=device, dtype=dtype)
        block[..., 3:6, 3:7] = 1.0
        assert torch.equal(closing(block, asymmetric), block)

        l_kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        for kernel in (l_kernel, asymmetric):
            closed = closing(tensor, kernel)
            assert (closed >= tensor).all()
            assert torch.equal(closing(closed, kernel), closed)

        # `[[1, 0, 0]]` reads `x(p + 1)` in the dilation half and `y(p - 1)` in the erosion half. Under
        # `replicate` the first column becomes `x(1)`, so the closing is not extensive (idempotence
        # survives); under `circular` the two shifts cancel and the closing is exactly `x`.
        side_kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        if supports_replicate_padding(device, dtype):
            dip = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype)[None, None]
            replicated = closing(dip, side_kernel, border_type="replicate")
            assert replicated.flatten().tolist() == [0.0, 0.0, 1.0]
            assert not bool((replicated >= dip).all())
            assert torch.equal(closing(replicated, side_kernel, border_type="replicate"), replicated)
        assert torch.equal(closing(tensor, side_kernel, border_type="circular"), tensor)

    def test_closing_handles_empty_geodesic_windows_4734(self, device, dtype):
        side_kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        tensor = torch.rand(
            1, 1, 7, 10, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        ).to(device=device, dtype=dtype)

        closed = closing(tensor, side_kernel)

        expected = torch.cat((torch.full_like(tensor[..., :1], float("inf")), tensor[..., 1:]), dim=-1)
        assert torch.equal(closed, expected)
        assert (closed >= tensor).all()
        assert torch.equal(closing(closed, side_kernel), closed)
