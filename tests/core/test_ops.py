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

import numpy as np
import pytest
import torch

from kornia.core.check import are_checks_enabled, disable_checks, enable_checks
from kornia.core.exceptions import BaseError, TypeCheckError
from kornia.core.ops import eye_like, vec_like

from testing.base import BaseTester

# Every `n` below is built by a factory, so no tensor is created at collection time.
ACCEPTED_INTEGERS = {
    "int": lambda: 3,
    "numpy-int64": lambda: np.int64(3),
    "numpy-int32": lambda: np.int32(3),
    "tensor-0d-int64": lambda: torch.tensor(3),
    "tensor-0d-int32": lambda: torch.tensor(3, dtype=torch.int32),
    # Accepted before the integer check existed (torch reads them as a size) and kept: `operator.index` takes them.
    "tensor-one-element-int": lambda: torch.tensor([3]),
    "numpy-0d-array": lambda: np.array(3),
}

REJECTED_NON_INTEGERS = {
    "float": lambda: 2.0,
    "float-fractional": lambda: 3.5,
    "bool-true": lambda: True,
    "bool-false": lambda: False,
    "numpy-float64": lambda: np.float64(2.0),
    "numpy-bool": lambda: np.bool_(True),
    "tensor-0d-float": lambda: torch.tensor(2.0),
    "tensor-0d-bool": lambda: torch.tensor(True),
    "tensor-vector": lambda: torch.tensor([3, 3]),
    "str": lambda: "3",
    "none": lambda: None,
}

NON_POSITIVE_INTEGERS = {
    "zero": lambda: 0,
    "negative": lambda: -1,
    "numpy-zero": lambda: np.int64(0),
    "tensor-0d-zero": lambda: torch.tensor(0),
}


class _BatchLikeMixin(BaseTester):
    """Checks shared by `eye_like` and `vec_like`: both take `(n, input, shared_memory)` and return `(B, n, *)`."""

    op = None  # the function under test, set by the subclass

    def item_shape(self, n):
        """Return the shape of one batch item for a given `n`."""
        raise NotImplementedError

    def expected(self, batch, n, device, dtype):
        """Return the expected output for a batch of `batch` items."""
        raise NotImplementedError

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("input_shape", [(4,), (2, 3), (3, 1, 5, 5)])
    def test_values(self, n, input_shape, device, dtype):
        x = torch.zeros(*input_shape, device=device, dtype=dtype)
        out = self.op(n, x)
        assert out.shape == (input_shape[0], *self.item_shape(n))
        assert out.device == x.device
        assert out.dtype == x.dtype
        assert torch.equal(out, self.expected(input_shape[0], n, device, dtype))

    @pytest.mark.parametrize("shared_memory", [False, True])
    def test_shared_memory_does_not_change_values(self, shared_memory, device, dtype):
        x = torch.zeros(3, 2, device=device, dtype=dtype)
        out = self.op(3, x, shared_memory=shared_memory)
        assert out.shape == (3, *self.item_shape(3))
        assert torch.equal(out, self.expected(3, 3, device, dtype))

    def test_shared_memory_true_expands_one_copy(self, device, dtype):
        x = torch.zeros(4, 2, device=device, dtype=dtype)
        out = self.op(3, x, shared_memory=True)
        assert out.stride(0) == 0
        assert all(out[i].data_ptr() == out[0].data_ptr() for i in range(1, 4))

    def test_shared_memory_false_owns_each_batch_item(self, device, dtype):
        x = torch.zeros(4, 2, device=device, dtype=dtype)
        out = self.op(3, x)
        assert out.is_contiguous()
        assert len({out[i].data_ptr() for i in range(4)}) == 4
        # writing into one batch item must not leak into another
        before = out[1].clone()
        out[0].fill_(7)
        assert torch.equal(out[1], before)

    @pytest.mark.parametrize("make_n", ACCEPTED_INTEGERS.values(), ids=ACCEPTED_INTEGERS.keys())
    def test_integer_n_of_any_integer_type(self, make_n, device, dtype):
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        assert torch.equal(self.op(make_n(), x), self.op(3, x))

    def test_integer_tensor_n_on_the_device(self, device, dtype):
        # The other integer tensors above live on the CPU: this one is read as a size from the accelerator.
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        n = torch.tensor(3, device=device)
        assert torch.equal(self.op(n, x), self.op(3, x))

    def test_meta_integer_tensor_n_raises_torchs_error(self, device, dtype):
        # A meta tensor holds no value to read as a size. It is an integer, so it is not a `TypeCheckError`:
        # torch raises its own `RuntimeError`, as it did before the integer check existed.
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match="meta tensors"):
            self.op(torch.tensor(3, device="meta"), x)

    @pytest.mark.parametrize("make_n", REJECTED_NON_INTEGERS.values(), ids=REJECTED_NON_INTEGERS.keys())
    def test_non_integer_n_raises_type_check_error_naming_n(self, make_n, device, dtype):
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        n = make_n()
        with pytest.raises(TypeCheckError, match=r"n must be an integer\. Got: ") as excinfo:
            self.op(n, x)
        assert isinstance(excinfo.value, BaseError)
        assert excinfo.value.actual_type is type(n)
        assert excinfo.value.expected_type is int

    def test_non_integer_n_message_shows_the_value_and_type(self, device, dtype):
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        with pytest.raises(TypeCheckError, match=r"n must be an integer\. Got: 2\.0 \(float\)"):
            self.op(2.0, x)
        with pytest.raises(TypeCheckError, match=r"n must be an integer\. Got: True \(bool\)"):
            self.op(True, x)

    def test_non_integer_n_is_left_to_torch_when_checks_are_disabled(self, device, dtype):
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        was_enabled = are_checks_enabled()
        disable_checks()
        try:
            with pytest.raises(TypeError) as excinfo:
                self.op(2.0, x)
        finally:
            if was_enabled:
                enable_checks()
        assert not isinstance(excinfo.value, BaseError)

    @pytest.mark.parametrize("make_n", NON_POSITIVE_INTEGERS.values(), ids=NON_POSITIVE_INTEGERS.keys())
    def test_non_positive_n_raises(self, make_n, device, dtype):
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        with pytest.raises(BaseError, match=r"n must be positive\. Got: "):
            self.op(make_n(), x)

    def test_input_without_a_batch_dimension_raises(self, device, dtype):
        x = torch.tensor(1.0, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="at least 1 dimension"):
            self.op(3, x)

    def test_dynamo(self, device, dtype, torch_optimizer):
        x = torch.zeros(2, 4, device=device, dtype=dtype)
        for shared_memory in (False, True):

            def run(t, shared_memory=shared_memory):
                return self.op(3, t, shared_memory=shared_memory)

            self.assert_close(torch_optimizer(run)(x), run(x))

    @pytest.mark.parametrize(
        "make_n", [lambda: 2.0, lambda: True, lambda: "3", lambda: None], ids=["float", "bool", "str", "none"]
    )
    def test_dynamo_non_integer_n_raises_type_check_error(self, make_n, device, dtype, torch_optimizer):
        # `n` is an argument of the compiled function, so the check runs while dynamo traces it. On torch 2.5.1 a
        # builtin that raises inside a `try` aborts the trace with `InternalTorchDynamoError`, not a kornia error.
        x = torch.zeros(2, 4, device=device, dtype=dtype)
        compiled = torch_optimizer(self.op)
        with pytest.raises(TypeCheckError, match=r"n must be an integer\. Got: "):
            compiled(make_n(), x)

    def test_dynamo_symbolic_n(self, device, dtype, torch_optimizer):
        # `n` read off a tensor shape changes between calls. Dynamo specialises it (it never hands the check a
        # `torch.SymInt`), so this pins only that a changing `n` compiles and gives the right result; see
        # `test_symbolic_n_under_make_fx` for a real `SymInt`.
        def run(t):
            return self.op(t.shape[-1], t)

        compiled = torch_optimizer(run)
        for size in (3, 4, 5):
            x = torch.zeros(2, size, device=device, dtype=dtype)
            self.assert_close(compiled(x), run(x))

    def test_symbolic_n_under_make_fx(self, device, dtype):
        # `make_fx(tracing_mode="symbolic")` is what hands the check a real `torch.SymInt`
        from torch.fx.experimental.proxy_tensor import make_fx

        seen = []

        def run(t):
            seen.append(type(t.shape[-1]))
            return self.op(t.shape[-1], t)

        make_fx(run, tracing_mode="symbolic")(torch.zeros(2, 4, device=device, dtype=dtype))
        assert seen == [torch.SymInt]

    def test_scripts(self, device, dtype):
        x = torch.zeros(2, 2, device=device, dtype=dtype)
        scripted = torch.jit.script(self.op)
        assert torch.equal(scripted(3, x), self.op(3, x))
        assert torch.equal(scripted(3, x, True), self.op(3, x, shared_memory=True))
        with pytest.raises(Exception, match=r"n must be positive\. Got: 0"):
            scripted(0, x)


class TestEyeLike(_BatchLikeMixin):
    op = staticmethod(eye_like)

    def item_shape(self, n):
        return (n, n)

    def expected(self, batch, n, device, dtype):
        return torch.eye(n, device=device, dtype=dtype).expand(batch, n, n)

    def test_shared_memory_stride(self, device, dtype):
        x = torch.zeros(4, 2, device=device, dtype=dtype)
        assert eye_like(3, x, shared_memory=True).stride() == (0, 3, 1)
        assert eye_like(3, x, shared_memory=False).stride() == (9, 3, 1)


class TestVecLike(_BatchLikeMixin):
    op = staticmethod(vec_like)

    def item_shape(self, n):
        return (n, 1)

    def expected(self, batch, n, device, dtype):
        return torch.zeros(batch, n, 1, device=device, dtype=dtype)

    def test_shared_memory_stride(self, device, dtype):
        x = torch.zeros(4, 2, device=device, dtype=dtype)
        assert vec_like(3, x, shared_memory=True).stride() == (0, 1, 1)
        assert vec_like(3, x, shared_memory=False).stride() == (3, 1, 1)
