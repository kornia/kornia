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

import copy
import operator
import pickle

import numpy as np
import pytest
import torch

from kornia.core.exceptions import TypeCheckError
from kornia.core.tensor_wrapper import TensorWrapper, _unwrap, _wrap

from testing.base import BaseTester, dynamo_is_available


class TestTensorWrapper(BaseTester):
    def test_smoke(self, device, dtype):
        data = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        tensor = _wrap(data, TensorWrapper)
        assert isinstance(tensor, TensorWrapper)
        assert isinstance(tensor.data, torch.Tensor)
        assert tensor.shape == (1, 2, 3, 4)
        assert tensor.device == device
        assert tensor.dtype == dtype
        self.assert_close(data, tensor.unwrap())

    def test_init_validation(self):
        """Test that TensorWrapper validates input type."""
        with pytest.raises(TypeCheckError, match="Tensor"):
            TensorWrapper([1, 2, 3])  # type: ignore

        with pytest.raises(TypeCheckError, match="Tensor"):
            TensorWrapper("not a tensor")  # type: ignore

    def test_repr(self, device, dtype):
        """Test string representation."""
        data = torch.rand(2, 3, device=device, dtype=dtype)
        tensor = TensorWrapper(data)
        repr_str = repr(tensor)
        assert "TensorWrapper" in repr_str
        assert str(data) in repr_str or repr(data) in repr_str

    def test_data_property(self, device, dtype):
        """Test data property access."""
        data = torch.rand(2, 3, device=device, dtype=dtype)
        tensor = TensorWrapper(data)
        assert tensor.data is data
        assert isinstance(tensor.data, torch.Tensor)

    def test_serialization(self, device, dtype, tmp_path):
        data = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        tensor: TensorWrapper = _wrap(data, TensorWrapper)

        file_path = tmp_path / "tensor.pt"
        torch.save(tensor, file_path)
        assert file_path.is_file()

        loaded_tensor: TensorWrapper = torch.load(file_path, weights_only=False)
        assert isinstance(loaded_tensor, TensorWrapper)

        self.assert_close(loaded_tensor.unwrap(), tensor.unwrap())
        # Check that used_attrs and used_calls are preserved
        assert hasattr(loaded_tensor, "used_attrs")
        assert hasattr(loaded_tensor, "used_calls")

    def test_wrap_list(self, device, dtype):
        data_list = [
            torch.rand(2, device=device, dtype=dtype),
            torch.rand(3, device=device, dtype=dtype),
            TensorWrapper(torch.rand(3, device=device, dtype=dtype)),
            1,
            0.5,
        ]
        tensor_list = _wrap(data_list, TensorWrapper)
        assert isinstance(tensor_list, list)
        assert len(tensor_list) == 5

        tensor_list_data = _unwrap(tensor_list)
        assert len(tensor_list_data) == 5

        self.assert_close(tensor_list_data[0], data_list[0])
        self.assert_close(tensor_list_data[1], data_list[1])
        self.assert_close(tensor_list_data[2], data_list[2])
        assert tensor_list_data[3] == data_list[3]
        assert tensor_list_data[4] == data_list[4]

        for i in range(len(tensor_list_data[:3])):
            self.assert_close(tensor_list[i].unwrap(), data_list[i])

    def test_wrap_tuple(self, device, dtype):
        """Test wrapping tuples."""
        data_tuple = (
            torch.rand(2, device=device, dtype=dtype),
            torch.rand(3, device=device, dtype=dtype),
        )
        tensor_tuple = _wrap(data_tuple, TensorWrapper)
        assert isinstance(tensor_tuple, tuple)
        assert len(tensor_tuple) == 2
        assert isinstance(tensor_tuple[0], TensorWrapper)
        assert isinstance(tensor_tuple[1], TensorWrapper)

    def test_accessors(self, device, dtype):
        data = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        x = _wrap(data, TensorWrapper)
        self.assert_close(x[1].unwrap(), torch.tensor(1.0, device=device, dtype=dtype))
        y = x[0]
        self.assert_close(y.unwrap(), torch.tensor(0.0, device=device, dtype=dtype))
        x[1] = 0.0
        self.assert_close(x.unwrap(), torch.zeros_like(data))

    def test_unary_ops(self, device, dtype):
        data = torch.rand(2, device=device, dtype=dtype)
        x = TensorWrapper(data)
        x1 = TensorWrapper(data)
        x2 = TensorWrapper(data)

        # Methods like x.add() return unwrapped tensors, operators return wrapped tensors
        self.assert_close(x.add(x), (x + x).unwrap())
        self.assert_close(x.add(1), (1 + x).unwrap())
        self.assert_close(x.add(1), (x + 1).unwrap())
        self.assert_close(x.mul(x), (x * x).unwrap())
        self.assert_close(x.mul(1), (1 * x).unwrap())
        self.assert_close(x.mul(1), (x * 1).unwrap())
        self.assert_close(x.sub(x), (x - x).unwrap())
        self.assert_close(x.sub(1), (x - 1).unwrap())
        self.assert_close(x.div(x), (x / x).unwrap())
        self.assert_close(x.true_divide(x), (x / x).unwrap())
        self.assert_close(x.floor_divide(x), (x // x).unwrap())
        self.assert_close(x1.ge(x2), (x1 >= x2).unwrap())
        self.assert_close(x1.gt(x2), (x1 > x2).unwrap())
        self.assert_close(x1.lt(x2), (x1 < x2).unwrap())
        self.assert_close(x1.le(x2), (x1 <= x2).unwrap())
        self.assert_close(x1.eq(x2), (x1 == x2).unwrap())
        self.assert_close(x1.ne(x2), (x1 != x2).unwrap())
        self.assert_close((-x).unwrap(), -data)

    def test_bool_int_conversion(self, device):
        """Test boolean and integer conversion."""
        # Test __bool__
        data_true = torch.tensor([1.0], device=device)
        tensor_true = TensorWrapper(data_true)
        assert bool(tensor_true) is True

        data_false = torch.tensor([0.0], device=device)
        tensor_false = TensorWrapper(data_false)
        assert bool(tensor_false) is False

        # Test __int__
        data_int = torch.tensor([42], device=device, dtype=torch.int32)
        tensor_int = TensorWrapper(data_int)
        assert int(tensor_int) == 42
        assert isinstance(int(tensor_int), int)

    def test_right_side_operations(self, device, dtype):
        """Test right-side operations (radd, rsub, rmul)."""
        data = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
        x = TensorWrapper(data)
        scalar = 5.0

        # Test __radd__: scalar + tensor
        result_radd = scalar + x
        expected_radd = scalar + data
        self.assert_close(result_radd.unwrap(), expected_radd)

        # Test __rsub__: scalar - tensor
        result_rsub = scalar - x
        expected_rsub = scalar - data
        self.assert_close(result_rsub.unwrap(), expected_rsub)

        # Test __rmul__: scalar * tensor
        result_rmul = scalar * x
        expected_rmul = scalar * data
        self.assert_close(result_rmul.unwrap(), expected_rmul)

    def test_used_attrs_tracking(self, device, dtype):
        """Test that attribute usage is tracked."""
        data = torch.rand(2, 3, device=device, dtype=dtype)
        tensor = TensorWrapper(data)

        # Initially empty
        assert len(tensor.used_attrs) == 0

        # Access some attributes
        _ = tensor.shape
        _ = tensor.device
        _ = tensor.dtype

        # Check that they're tracked
        assert "shape" in tensor.used_attrs
        assert "device" in tensor.used_attrs
        assert "dtype" in tensor.used_attrs

    def test_used_calls_tracking(self, device, dtype):
        """Test that function calls are tracked."""
        data = torch.rand(2, 3, device=device, dtype=dtype)
        tensor = TensorWrapper(data)

        # Initially empty
        assert len(tensor.used_calls) == 0

        # Call some operations
        _ = tensor + tensor
        _ = tensor * 2
        _ = torch.sum(tensor)

        # Check that they're tracked
        assert len(tensor.used_calls) > 0
        assert torch.add in tensor.used_calls or torch.mul in tensor.used_calls

    def test_setattr_tracking(self, device, dtype):
        """Test that setattr doesn't track internal attributes."""
        data = torch.rand(2, 3, device=device, dtype=dtype)
        tensor = TensorWrapper(data)

        # Setting a new attribute on the underlying tensor should be tracked
        tensor.some_new_attr = 42
        assert "some_new_attr" in tensor.used_attrs

    def test_callable(self, device, dtype):
        data = torch.ones(2, device=device, dtype=dtype)
        x = TensorWrapper(data)
        y = (x * x).sum(-1, True)
        # sum() method returns unwrapped tensor, so compare directly
        expected = torch.ones_like(y) * 2
        self.assert_close(y, expected)

    def test_len(self, device, dtype):
        """Test __len__ method."""
        data = torch.rand(5, device=device, dtype=dtype)
        tensor = TensorWrapper(data)
        assert len(tensor) == 5
        assert len(tensor) == len(data)

    def test_nested_wrapping(self, device, dtype):
        """Test that wrapping already-wrapped tensors works correctly."""
        data = torch.rand(2, 3, device=device, dtype=dtype)
        wrapped_once = TensorWrapper(data)
        wrapped_twice = _wrap(wrapped_once, TensorWrapper)

        # Should still unwrap to original data
        self.assert_close(_unwrap(wrapped_twice), data)

    def test_string_bytes_not_wrapped(self):
        """Test that strings and bytes are not treated as sequences in __torch_function__."""
        # This tests the fix for not treating strings/bytes as sequences
        data = torch.rand(2, 3)
        tensor = TensorWrapper(data)

        # Should not raise errors when strings/bytes are in args
        result = torch.cat([tensor, tensor])
        assert isinstance(result, TensorWrapper)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    def test_exception(self):
        """Test exception handling for invalid inputs."""
        with pytest.raises(TypeCheckError):
            TensorWrapper([1, 2, 3])  # type: ignore

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass

    def test_gradcheck(self, device):
        # No entry is an integer, where ``% 1.0`` jumps.
        x = torch.tensor([[0.5, 1.25, 2.5], [1.75, 3.5, 0.75]], device=device)

        def fn(t):
            w = TensorWrapper(t)
            return (2 / w + w**2 - abs(-w) % 1.0 + (w @ w.T).sum()).unwrap()

        self.gradcheck(fn, (x,))

        def inplace_fn(t):
            w = TensorWrapper(t * 1)
            w *= t
            w += 1
            w **= 2
            return (2 / w).unwrap()

        self.gradcheck(inplace_fn, (x,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        x = torch.tensor([[0.5, 1.5, 2.0], [1.0, 3.0, 0.75]], device=device, dtype=dtype)
        compiled = torch_optimizer(_build_and_update)
        self.assert_close(compiled(x), _build_and_update(x))


def _build_and_update(t):
    w = TensorWrapper(t.clone())
    w += 1
    w *= 2
    return (2 / w + w**2 - abs(w) % 1.5).unwrap()


_PICKLE_FLOAT_CASES_WITH_IDS = [
    ("add", True, lambda w, x: w + x),
    ("radd", True, lambda w, x: 2 + w),
    ("sub", True, lambda w, x: w - x),
    ("rsub", True, lambda w, x: 2 - w),
    ("mul", True, lambda w, x: w * x),
    ("rmul", True, lambda w, x: 2 * w),
    ("truediv", True, lambda w, x: w / x),
    ("rtruediv", True, lambda w, x: 2 / w),
    ("floordiv", True, lambda w, x: w // x),
    ("rfloordiv", True, lambda w, x: 2 // w),
    ("mod", True, lambda w, x: w % x),
    ("rmod", True, lambda w, x: 2 % w),
    ("pow", True, lambda w, x: w**2),
    ("pow_tensor", True, lambda w, x: w**x),
    ("rpow", True, lambda w, x: 2**w),
    ("matmul", True, lambda w, x: w @ x.T),
    ("rmatmul", True, lambda w, x: w.__rmatmul__(x.T)),
    ("lt", True, lambda w, x: w < x),
    ("le", True, lambda w, x: w <= x),
    ("gt", True, lambda w, x: w > x),
    ("ge", True, lambda w, x: w >= x),
    ("eq", True, lambda w, x: w == x),
    ("ne", True, lambda w, x: w != x),
    ("neg", True, lambda w, x: -w),
    ("pos", True, lambda w, x: +w),
    ("abs", True, lambda w, x: abs(w)),
    ("iadd", True, operator.iadd),
    ("isub", True, operator.isub),
    ("imul", True, operator.imul),
    ("itruediv", True, operator.itruediv),
    ("ifloordiv", True, operator.ifloordiv),
    ("imod", True, operator.imod),
    ("ipow", True, operator.ipow),
    ("tensor_pow", False, lambda w, x: x**w),
    ("torch_unique", False, lambda w, x: torch.unique(w)),
]
_PICKLE_FLOAT_CASES = [(own, fn) for _, own, fn in _PICKLE_FLOAT_CASES_WITH_IDS]
_PICKLE_INT_CASES_WITH_IDS = [
    ("and", True, lambda w, x: w & x),
    ("rand", True, lambda w, x: 1 & w),
    ("or", True, lambda w, x: w | x),
    ("ror", True, lambda w, x: 1 | w),
    ("xor", True, lambda w, x: w ^ x),
    ("rxor", True, lambda w, x: 1 ^ w),
    ("lshift", True, lambda w, x: w << x),
    ("rlshift", True, lambda w, x: 1 << w),
    ("rshift", True, lambda w, x: w >> x),
    ("rrshift", True, lambda w, x: 64 >> w),
    ("invert", True, lambda w, x: ~w),
    ("iand", True, operator.iand),
    ("ior", True, operator.ior),
    ("ixor", True, operator.ixor),
    ("ilshift", True, operator.ilshift),
    ("irshift", True, operator.irshift),
]
_PICKLE_INT_CASES = [(own, fn) for _, own, fn in _PICKLE_INT_CASES_WITH_IDS]


class TestTensorWrapperProtocol(BaseTester):
    """Attribute ownership, copying and the operator protocol (#5189, #5190)."""

    def test_uninitialized_instance_raises_attribute_error(self):
        # An instance made by ``__new__`` alone has no slots set, and Dynamo looks up ``__dict__`` on one while it
        # traces the constructor: every lookup must raise AttributeError instead of recursing through __getattr__.
        blank = TensorWrapper.__new__(TensorWrapper)
        for name in ("anything", "shape", "data", "_data", "used_attrs", "used_calls", "__dict__", "__deepcopy__"):
            with pytest.raises(AttributeError):
                getattr(blank, name)

    def test_owned_names_are_not_forwarded(self, device, dtype):
        wrapper = TensorWrapper(torch.zeros(2, device=device, dtype=dtype))
        for name in ("__dict__", "__copy__", "__deepcopy__", "__getnewargs__", "__getnewargs_ex__"):
            assert not hasattr(wrapper, name)
        assert wrapper.used_attrs == set()

    @pytest.mark.skipif(not dynamo_is_available(), reason="no Dynamo on this torch/python pair")
    @pytest.mark.parametrize(
        "fn, fullgraph",
        [
            (lambda t: TensorWrapper(t).data, True),
            (lambda t: TensorWrapper(t).unwrap(), True),
            (lambda t: (TensorWrapper(t) + 1).unwrap(), True),
            (_build_and_update, True),
            (lambda t: TensorWrapper(t).sum().unwrap(), True),
            # A forwarded method on a wrapper built from an intermediate tensor (#5241).
            (lambda t: TensorWrapper(t.clone()).sum().unwrap(), True),
            (lambda t: TensorWrapper(t * 2).mean(dim=-1).unwrap(), True),
            (lambda t: TensorWrapper(t + 1).reshape(-1).unwrap(), True),
            # torch 2.5.1's Dynamo breaks the graph here, cleanly: it does not send unary ``-`` or ``~``, a
            # comparison, ``@`` or a binary or in-place operator with a tensor operand to a user class, and it cannot
            # trace a torch function on a wrapper or ``__getitem__``.
            (lambda t: torch.add(TensorWrapper(t), 1).unwrap(), False),
            (lambda t: TensorWrapper(t)[0].unwrap(), False),
            (lambda t: (TensorWrapper(t) @ TensorWrapper(t.T)).unwrap(), False),
        ],
        ids=[
            "data",
            "unwrap",
            "add",
            "inplace_and_reflected",
            "method",
            "sum_intermediate",
            "mean_intermediate",
            "reshape_intermediate",
            "torch_add",
            "getitem",
            "matmul",
        ],
    )
    def test_eager_backend_traces_a_function_that_builds_a_wrapper(self, device, dtype, fn, fullgraph):
        # The eager backend keeps this in the ordinary jobs; test_dynamo covers the optimizer backends.
        torch._dynamo.reset()
        x = torch.tensor([[0.5, 1.5, 2.0], [1.0, 3.0, 0.75]], device=device, dtype=dtype)
        out = torch.compile(fn, backend="eager", fullgraph=fullgraph)(x)
        assert type(out) is torch.Tensor
        self.assert_close(out, fn(x))

    def test_deepcopy_copy_and_pickle_keep_the_class(self, device, dtype, tmp_path):
        data = torch.tensor([[0.3, -1.2, 2.5]], device=device, dtype=dtype, requires_grad=True)
        wrapper = TensorWrapper(data)
        _ = wrapper.shape
        _ = wrapper + 1

        deep = copy.deepcopy(wrapper)
        assert type(deep) is TensorWrapper
        self.assert_close(deep.data, data, rtol=0, atol=0)
        assert deep.data.data_ptr() != data.data_ptr()
        assert deep.used_attrs == {"shape"}
        assert deep.used_attrs is not wrapper.used_attrs

        shallow = copy.copy(wrapper)
        assert type(shallow) is TensorWrapper
        assert shallow.data is data
        # The copy tracks its own usage: both sets are copied, neither is shared with the original.
        assert shallow.used_attrs == wrapper.used_attrs
        assert shallow.used_attrs is not wrapper.used_attrs
        assert shallow.used_calls == wrapper.used_calls == {torch.add}
        assert shallow.used_calls is not wrapper.used_calls

        restored = pickle.loads(pickle.dumps(wrapper))  # noqa: S301
        torch.save(wrapper, tmp_path / "wrapper.pt")
        loaded = torch.load(tmp_path / "wrapper.pt", weights_only=False)
        for copied in (deep, restored, loaded):
            assert type(copied) is TensorWrapper
            self.assert_close(copied.data, data, rtol=0, atol=0)
            assert copied.data.requires_grad
            assert copied.data.is_leaf
            assert copied.data.device == data.device

    def test_array_protocols_still_reach_the_tensor(self):
        # Only the wrapper's own names stop at __getattr__: the DLPack and array-interface dunders are still
        # forwarded, so converting a wrapper keeps going through the wrapped tensor's buffer.
        data = torch.tensor([[0.3, -1.2, 2.5]])
        wrapper = TensorWrapper(data)
        self.assert_close(torch.from_dlpack(wrapper), data, rtol=0, atol=0)
        array = np.asarray(wrapper)
        assert array.dtype == np.float32
        assert array.tolist() == data.tolist()

    def test_pow_defers_to_the_other_operand_as_the_tensor_does(self, device, dtype):
        # For an operand torch.pow cannot take, Tensor.__pow__ returns NotImplemented, so Python tries the other
        # operand's __rpow__; ``w ** x`` does the same, like the other forward operators (``w % x``, ``w @ x``).
        class Exponent:
            def __rpow__(self, base):
                return "Exponent.__rpow__"

        data = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        wrapper = TensorWrapper(data.clone())
        assert data ** Exponent() == "Exponent.__rpow__"
        assert wrapper ** Exponent() == "Exponent.__rpow__"
        with pytest.raises(TypeError, match="unsupported operand"):
            _ = wrapper ** "2"

    @pytest.mark.filterwarnings("ignore:__array_wrap__ must accept:DeprecationWarning")
    def test_pow_with_a_numpy_operand_matches_the_tensor(self):
        data = torch.tensor([[1.0, 2.0, 3.0]])
        for exponent in (np.array(2.0), np.array([[2.0, 0.5, 3.0]])):
            expected = data**exponent
            out = TensorWrapper(data.clone()) ** exponent
            # NumPy hands the result back through the forwarded ``__array_wrap__``, so it is wrapped like the
            # result of any other operator.
            assert type(out) is TensorWrapper
            self.assert_close(out.data, expected, rtol=0, atol=0)

    def test_result_class_with_a_wrapper_subclass_on_the_right(self, device, dtype):
        # Arithmetic takes the left wrapper's class; a comparison takes the right operand's subclass, because Python
        # tries a subclass's reflected comparison first.
        class Sub(TensorWrapper):
            pass

        a = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=dtype)
        b = torch.tensor([2.0, 2.0, 3.0], device=device, dtype=dtype)
        for op in (operator.add, operator.mul):
            assert type(op(TensorWrapper(a), Sub(b))) is TensorWrapper
        for op in (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge):
            out = op(TensorWrapper(a), Sub(b))
            assert type(out) is Sub
            assert torch.equal(out.data, op(a, b))

    @pytest.mark.parametrize(
        "op",
        [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod, operator.pow],
    )
    def test_arithmetic_operators_match_the_tensor(self, device, dtype, op):
        a = torch.tensor([[1.0, 2.0, 4.0], [0.5, 3.0, 5.0]], device=device, dtype=dtype)
        b = torch.tensor([[2.0, 0.5, 3.0], [1.5, 2.0, 0.25]], device=device, dtype=dtype)
        wa, wb = TensorWrapper(a), TensorWrapper(b)
        for lhs, rhs, expected in [
            (wa, wb, op(a, b)),
            (wa, b, op(a, b)),
            (a, wb, op(a, b)),
            (wa, 3, op(a, 3)),
            (3, wa, op(3, a)),
        ]:
            out = op(lhs, rhs)
            assert type(out) is TensorWrapper
            self.assert_close(out.data, expected, rtol=0, atol=0)

    def test_matmul_unary_and_conversions_match_the_tensor(self, device, dtype):
        a = torch.tensor([[1.0, -2.0, 4.0], [0.5, 3.0, -5.0]], device=device, dtype=dtype)
        wa = TensorWrapper(a)
        m = torch.tensor([[1.0, 2.0], [0.5, -1.0]], device=device, dtype=dtype)
        for out, expected in [
            (wa @ wa.T, a @ a.T),
            (wa @ TensorWrapper(a.T), a @ a.T),
            (a.T @ wa, a.T @ a),
            (wa.__rmatmul__(m), m @ a),
            (abs(wa), abs(a)),
            (+wa, +a),
            (-wa, -a),
        ]:
            assert type(out) is TensorWrapper
            self.assert_close(out.data, expected, rtol=0, atol=0)

        element = TensorWrapper(a[1, 1])
        assert float(element) == float(a[1, 1])
        assert complex(element) == complex(a[1, 1])
        assert int(element) == int(a[1, 1])
        index = TensorWrapper(torch.tensor(2, device=device))
        assert operator.index(index) == 2
        assert [10, 20, 30][index] == 30

    @pytest.mark.parametrize(
        "op",
        [operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift],
    )
    def test_bitwise_operators_match_the_tensor(self, device, op):
        a = torch.tensor([12, 5, 3], device=device)
        b = torch.tensor([10, 1, 2], device=device)
        wa, wb = TensorWrapper(a), TensorWrapper(b)
        for lhs, rhs, expected in [(wa, wb, op(a, b)), (wa, 1, op(a, 1)), (1, wa, op(1, a))]:
            out = op(lhs, rhs)
            assert type(out) is TensorWrapper
            self.assert_close(out.data, expected, rtol=0, atol=0)

        mask = TensorWrapper(a > 4)
        out = ~mask & (wa < 13)
        assert type(out) is TensorWrapper
        assert out.data.tolist() == (~(a > 4) & (a < 13)).tolist()

    @pytest.mark.parametrize(
        "op, iop",
        [
            (operator.add, operator.iadd),
            (operator.sub, operator.isub),
            (operator.mul, operator.imul),
            (operator.truediv, operator.itruediv),
            (operator.floordiv, operator.ifloordiv),
            (operator.mod, operator.imod),
            (operator.pow, operator.ipow),
        ],
    )
    def test_inplace_operators_update_the_wrapped_tensor(self, device, dtype, op, iop):
        a = torch.tensor([[1.0, 2.0, 4.0], [0.5, 3.0, 5.0]], device=device, dtype=dtype)
        for other in (3, torch.full_like(a, 2.0), TensorWrapper(torch.full_like(a, 2.0))):
            data = a.clone()
            wrapper = TensorWrapper(data)
            alias = wrapper
            out = iop(wrapper, other)
            assert out is alias
            assert out.data is data
            self.assert_close(alias.data, op(a, _unwrap(other)), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "op, iop",
        [
            (operator.and_, operator.iand),
            (operator.or_, operator.ior),
            (operator.xor, operator.ixor),
            (operator.lshift, operator.ilshift),
            (operator.rshift, operator.irshift),
        ],
    )
    def test_inplace_bitwise_operators_update_the_wrapped_tensor(self, device, op, iop):
        a = torch.tensor([12, 5, 3], device=device)
        data = a.clone()
        wrapper = TensorWrapper(data)
        alias = wrapper
        out = iop(wrapper, 1)
        assert out is alias
        assert out.data is data
        assert alias.data.tolist() == op(a, 1).tolist()

    def test_inplace_matmul_rebinds_as_on_the_tensor(self, device, dtype):
        # A tensor has no ``__imatmul__``, so ``t @= m`` rebinds ``t`` to ``t @ m``; the wrapper does the same.
        a = torch.tensor([[1.0, 2.0], [0.5, 3.0]], device=device, dtype=dtype)
        m = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype)
        tensor = a.clone()
        tensor_alias = tensor
        tensor @= m
        assert tensor is not tensor_alias
        wrapper = TensorWrapper(a.clone())
        alias = wrapper
        wrapper @= m
        assert wrapper is not alias
        assert type(wrapper) is TensorWrapper
        self.assert_close(wrapper.data, a @ m, rtol=0, atol=0)
        self.assert_close(alias.data, a, rtol=0, atol=0)

    def test_inplace_operators_raise_where_the_tensor_raises(self, device, dtype):
        # Wrapping does not copy, so an in-place operator writes into the caller's tensor, and it raises where the
        # tensor's in-place operator raises: on a leaf that requires grad, and for a dtype or shape change.
        leaf = torch.ones(2, 3, device=device, dtype=dtype, requires_grad=True)
        integer = torch.tensor([1, 2, 3], device=device)
        for target in (leaf, TensorWrapper(leaf)):
            with pytest.raises(RuntimeError, match="leaf Variable"):
                target += 1
        for target in (integer.clone(), TensorWrapper(integer.clone())):
            with pytest.raises(RuntimeError, match="can't be cast"):
                target /= 2
        for target in (leaf.detach().clone(), TensorWrapper(leaf.detach().clone())):
            with pytest.raises(RuntimeError):
                target += torch.ones(4, 2, 3, device=device, dtype=dtype)

        caller = leaf.detach().clone()
        wrapper = TensorWrapper(caller)
        wrapper += 1
        self.assert_close(caller, torch.full_like(caller, 2.0), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "own, fn",
        _PICKLE_FLOAT_CASES,
        ids=[label for label, _, _ in _PICKLE_FLOAT_CASES_WITH_IDS],
    )
    def test_pickle_and_torch_save_after_each_operator(self, device, dtype, tmp_path, own, fn):
        w = TensorWrapper(torch.tensor([[1.0, 2.0, 4.0], [0.5, 3.0, 5.0]], device=device, dtype=dtype))
        x = torch.tensor([[2.0, 0.5, 3.0], [1.5, 2.0, 0.25]], device=device, dtype=dtype)
        fn(w, x)
        self._check_round_trips(w, own, tmp_path)

    @pytest.mark.parametrize(
        "own, fn",
        _PICKLE_INT_CASES,
        ids=[label for label, _, _ in _PICKLE_INT_CASES_WITH_IDS],
    )
    def test_pickle_and_torch_save_after_each_integer_operator(self, device, tmp_path, own, fn):
        w = TensorWrapper(torch.tensor([12, 5, 3], device=device))
        x = torch.tensor([10, 1, 2], device=device)
        fn(w, x)
        self._check_round_trips(w, own, tmp_path)

    def _check_round_trips(self, w, own, tmp_path):
        assert len(w.used_calls) > 0
        torch.save(w, tmp_path / "wrapper.pt")
        for restored in (
            pickle.loads(pickle.dumps(w)),  # noqa: S301
            torch.load(tmp_path / "wrapper.pt", weights_only=False),
        ):
            assert type(restored) is TensorWrapper
            self.assert_close(restored.data, w.data, rtol=0, atol=0)
            assert restored.used_attrs == w.used_attrs
            # The wrapper's own operators record picklable functions; a torch function that pickle cannot store
            # (``torch.unique``, or the ``Tensor.__pow__`` that ``t ** w`` dispatches) is left out of the state.
            if own:
                assert restored.used_calls == w.used_calls
            else:
                assert restored.used_calls < w.used_calls
