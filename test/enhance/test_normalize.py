import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import BaseTester


class TestNormalize:
    @staticmethod
    def test_smoke(device, dtype):
        mean = [0.5]
        std = [0.1]
        repr = "Normalize(mean=[0.5], std=[0.1])"
        assert str(kornia.enhance.Normalize(mean, std)) == repr

    @staticmethod
    def test_normalize(device, dtype):

        # prepare input data
        data = torch.ones(1, 2, 2, device=device, dtype=dtype)
        mean = torch.tensor([0.5], device=device, dtype=dtype)
        std = torch.tensor([2.0], device=device, dtype=dtype)

        # expected output
        expected = torch.tensor([0.25], device=device, dtype=dtype).repeat(1, 2, 2).view_as(data)

        f = kornia.enhance.Normalize(mean, std)
        assert_allclose(f(data), expected)

    @staticmethod
    def test_broadcast_normalize(device, dtype):

        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([2.0], device=device, dtype=dtype)
        std = torch.tensor([0.5], device=device, dtype=dtype)

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.enhance.Normalize(mean, std)
        assert_allclose(f(data), expected)

    @staticmethod
    def test_float_input(device, dtype):

        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: float = 2.0
        std: float = 0.5

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.enhance.Normalize(mean, std)
        assert_allclose(f(data), expected)

    @staticmethod
    def test_batch_normalize(device, dtype):

        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        # expected output
        expected = torch.tensor([1.25, 1, 0.5], device=device, dtype=dtype).repeat(2, 1, 1).view_as(data)

        f = kornia.enhance.Normalize(mean, std)
        assert_allclose(f(data), expected)

    @pytest.mark.skip(reason="union type not supported")
    @staticmethod
    def test_jit(device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.normalize
        op_script = torch.jit.script(op)

        assert_allclose(op(*inputs), op_script(*inputs))

    @staticmethod
    def test_gradcheck(device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        data = utils.tensor_to_gradcheck_var(data)  # to var
        mean = utils.tensor_to_gradcheck_var(mean)  # to var
        std = utils.tensor_to_gradcheck_var(std)  # to var

        assert gradcheck(kornia.enhance.Normalize(mean, std), (data,), raise_exception=True)

    @staticmethod
    def test_single_value(device, dtype):
        # prepare input data
        mean = torch.tensor(2, device=device, dtype=dtype)
        std = torch.tensor(3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 256, 313, device=device, dtype=dtype)

        # expected output
        expected = (data - mean) / std

        assert_allclose(kornia.normalize(data, mean, std), expected)

    @staticmethod
    def test_module(device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.normalize
        op_module = kornia.enhance.Normalize(mean, std)

        assert_allclose(op(*inputs), op_module(data))


class TestDenormalize:
    @staticmethod
    def test_smoke(device, dtype):
        mean = [0.5]
        std = [0.1]
        repr = "Denormalize(mean=[0.5], std=[0.1])"
        assert str(kornia.enhance.Denormalize(mean, std)) == repr

    @staticmethod
    def test_denormalize(device, dtype):

        # prepare input data
        data = torch.ones(1, 2, 2, device=device, dtype=dtype)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([2.5], device=device, dtype=dtype).repeat(1, 2, 2).view_as(data)

        f = kornia.enhance.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    @staticmethod
    def test_broadcast_denormalize(device, dtype):

        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([2.0], device=device, dtype=dtype)
        std = torch.tensor([0.5], device=device, dtype=dtype)

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.enhance.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    @staticmethod
    def test_float_input(device, dtype):

        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: float = 2.0
        std: float = 0.5

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.enhance.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    @staticmethod
    def test_batch_denormalize(device, dtype):

        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        # expected output
        expected = torch.tensor([6.5, 7, 8], device=device, dtype=dtype).repeat(2, 1, 1).view_as(data)

        f = kornia.enhance.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    @pytest.mark.skip(reason="union type not supported")
    @staticmethod
    def test_jit(device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.denormalize
        op_script = torch.jit.script(op)

        assert_allclose(op(*inputs), op_script(*inputs))

    @staticmethod
    def test_gradcheck(device, dtype):

        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype)

        data = utils.tensor_to_gradcheck_var(data)  # to var
        mean = utils.tensor_to_gradcheck_var(mean)  # to var
        std = utils.tensor_to_gradcheck_var(std)  # to var

        assert gradcheck(kornia.enhance.Denormalize(mean, std), (data,), raise_exception=True)

    @staticmethod
    def test_single_value(device, dtype):

        # prepare input data
        mean = torch.tensor(2, device=device, dtype=dtype)
        std = torch.tensor(3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 256, 313, device=device, dtype=dtype)

        # expected output
        expected = (data * std) + mean

        assert_allclose(kornia.denormalize(data, mean, std), expected)

    @staticmethod
    def test_module(device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.denormalize
        op_module = kornia.enhance.Denormalize(mean, std)

        assert_allclose(op(*inputs), op_module(data))


class TestNormalizeMinMax(BaseTester):
    def test_smoke(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        assert kornia.normalize_min_max(x) is not None
        assert kornia.enhance.normalize_min_max(x) is not None

    def test_exception(self, device, dtype):
        x = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError):
            assert kornia.normalize_min_max(0.0)

        with pytest.raises(TypeError):
            assert kornia.normalize_min_max(x, '', '')

        with pytest.raises(TypeError):
            assert kornia.normalize_min_max(x, 2.0, '')

    @pytest.mark.parametrize("input_shape", [(1, 2, 3, 4), (2, 1, 4, 3), (1, 3, 2, 1)])
    def test_cardinality(self, device, dtype, input_shape):
        x = torch.rand(input_shape, device=device, dtype=dtype)
        assert kornia.normalize_min_max(x).shape == input_shape

    @pytest.mark.parametrize("min_val, max_val", [(1.0, 2.0), (2.0, 3.0), (5.0, 20.0), (40.0, 1000.0)])
    @staticmethod
    def test_range(device, dtype, min_val, max_val):
        x = torch.rand(1, 2, 4, 5, device=device, dtype=dtype)
        out = kornia.normalize_min_max(x, min_val=min_val, max_val=max_val)
        assert_allclose(out.min().item(), min_val)
        assert_allclose(out.max().item(), max_val)

    @staticmethod
    def test_values(device, dtype):
        x = torch.tensor([[[[0.0, 1.0, 3.0], [-1.0, 4.0, 3.0], [9.0, 5.0, 2.0]]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[-0.8, -0.6, -0.2], [-1.0, 0.0, -0.2], [1.0, 0.2, -0.4]]]], device=device, dtype=dtype
        )

        actual = kornia.normalize_min_max(x, min_val=-1.0, max_val=1.0)
        assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        op = kornia.normalize_min_max
        op_jit = torch.jit.script(op)
        assert_allclose(op(x), op_jit(x))

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.normalize_min_max, (x,), raise_exception=True)

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.nn
    def test_module(self, device, dtype):
        pass
