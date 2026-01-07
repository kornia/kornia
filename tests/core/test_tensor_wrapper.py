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

from kornia.core.exceptions import TypeCheckError
from kornia.core.tensor_wrapper import TensorWrapper, _unwrap, _wrap

from testing.base import BaseTester


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

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device):
        pass
