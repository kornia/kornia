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

from testing.base import BaseTester


class TestVflip(BaseTester):
    def smoke_test(self, device, dtype):
        f = kornia.geometry.transform.Vflip()
        repr = "Vflip()"
        assert str(f) == repr

    def test_vflip(self, device, dtype):
        f = kornia.geometry.transform.Vflip()
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
        )  # 3 x 3

        self.assert_close(f(input), expected)

    def test_batch_vflip(self, device, dtype):
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.geometry.transform.Vflip()
        expected = torch.tensor(
            [[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        self.assert_close(f(input), expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.geometry.transform.vflip(data)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input,))

        # Create new inputs
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 5.0, 0.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor(
            [[[5.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.Vflip(), (input,))

    def test_convention_rank1_raises(self, device, dtype):
        # vflip flips dim -2; a rank-1 tensor has no such dim and raises IndexError, unlike
        # hflip which only needs dim -1 (see TestHflip.test_convention_rank1_supported).
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
        with pytest.raises(IndexError):
            kornia.geometry.transform.vflip(x)


class TestHflip(BaseTester):
    def smoke_test(self, device, dtype):
        f = kornia.geometry.transform.Hflip()
        repr = "Hflip()"
        assert str(f) == repr

    def test_hflip(self, device, dtype):
        f = kornia.geometry.transform.Hflip()
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], device=device, dtype=dtype
        )  # 3 x 3

        self.assert_close(f(input), expected)

    def test_batch_hflip(self, device, dtype):
        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.geometry.transform.Hflip()
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        self.assert_close(f(input), expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.geometry.transform.hflip(data)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input,))

        # Create new inputs
        input = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 5.0, 5.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.Hflip(), (input,))

    def test_convention_rank1_supported(self, device, dtype):
        # hflip's documented rank floor is 1 (operates on dim -1): even a bare rank-1 tensor
        # (no H/W structure at all) works, unlike vflip/rot180 which need a second-to-last
        # axis (see TestVflip/TestRot180.test_convention_rank1_raises). Rank-0 behavior is not
        # part of the documented contract and is intentionally not pinned here.
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
        # Snippet used to generate expected (requires only this module): hflip reverses the
        # last dimension, so for a rank-1 tensor that is simply the element order reversed.
        # x.flip(-1) -> tensor([3., 2., 1., 0.])
        expected = torch.tensor([3.0, 2.0, 1.0, 0.0], device=device, dtype=dtype)
        self.assert_close(kornia.geometry.transform.hflip(x), expected)


class TestRot180(BaseTester):
    def smoke_test(self, device, dtype):
        f = kornia.geometry.transform.Rot180()
        repr = "Rot180()"
        assert str(f) == repr

    def test_rot180(self, device, dtype):
        f = kornia.geometry.transform.Rot180()
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
        )  # 3 x 3

        self.assert_close(f(input), expected)

    def test_batch_rot180(self, device, dtype):
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        f = kornia.geometry.transform.Rot180()
        expected = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        expected = expected.repeat(2, 1, 1)  # 2 x 3 x 3

        self.assert_close(f(input), expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.geometry.transform.rot180(data)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input,))

        # Create new inputs
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 5.0, 0.0]], device=device, dtype=dtype)  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor(
            [[[0.0, 5.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device, dtype=dtype
        )  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=torch.float64
        )  # 3 x 3

        self.gradcheck(kornia.geometry.transform.Rot180(), (input,))

    def test_convention_rank1_raises(self, device, dtype):
        # rot180 flips dims -2 and -1; a rank-1 tensor has no dim -2 and raises IndexError,
        # unlike hflip which only needs dim -1 (see TestHflip.test_convention_rank1_supported).
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
        with pytest.raises(IndexError):
            kornia.geometry.transform.rot180(x)
