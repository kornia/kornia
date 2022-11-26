import pytest
import torch

from kornia.core import Tensor
from kornia.core.tensor_wrapper import TensorWrapper, unwrap, wrap
from kornia.testing import BaseTester


class TestTensorWrapper(BaseTester):
    def test_smoke(self, device, dtype):
        data = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        tensor = wrap(data, TensorWrapper)
        assert isinstance(tensor, TensorWrapper)
        assert isinstance(tensor.data, Tensor)
        assert tensor.shape == (1, 2, 3, 4)
        assert tensor.device == device
        assert tensor.dtype == dtype
        self.assert_close(data, tensor.data)
        self.assert_close(data, unwrap(tensor))

    def test_wrap_list(self, device, dtype):
        data_list = [torch.rand(2, device=device, dtype=dtype), torch.rand(3, device=device, dtype=dtype)]
        tensor_list = wrap(data_list, TensorWrapper)
        assert isinstance(tensor_list, list)
        assert len(tensor_list) == 2
        tensor_list_data = unwrap(tensor_list)
        assert len(tensor_list_data) == 2
        self.assert_close(tensor_list_data[0], data_list[0])
        self.assert_close(tensor_list_data[1], data_list[1])

    # TODO: implement me
    def test_accessors(self, device, dtype):
        data = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        x = wrap(data, TensorWrapper)
        self.assert_close(x[1], torch.tensor(1.0, device=device, dtype=dtype))
        y = x[0]
        self.assert_close(y, torch.tensor(0.0, device=device, dtype=dtype))
        x[1] = 0.0
        self.assert_close(x, torch.zeros_like(x))

    def test_unary_ops(self, device, dtype):
        data = torch.rand(2, device=device, dtype=dtype)
        x = TensorWrapper(data)

        self.assert_close(x.add(x), x + x)
        self.assert_close(x.mul(x), x * x)
        self.assert_close(x.sub(x), x - x)
        self.assert_close(x.div(x), x / x)
        self.assert_close(x.true_divide(x), x / x)
        self.assert_close(x.floor_divide(x), x // x)
        self.assert_close(x.ge(x), x >= x)
        self.assert_close(x.gt(x), x > x)
        self.assert_close(x.lt(x), x < x)
        self.assert_close(x.le(x), x <= x)
        self.assert_close(x.eq(x), x == x)
        self.assert_close(x.ne(x), x != x)

    def test_callable(self, device, dtype):
        data = torch.ones(2, device=device, dtype=dtype)
        x = TensorWrapper(data)
        y = (x * x).sum(-1, True)
        self.assert_close(y, torch.ones_like(y) * 2)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device):
        pass
