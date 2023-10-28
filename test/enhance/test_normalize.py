import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import BaseTester


class TestNormalize(BaseTester):
    def test_smoke(self, device, dtype):
        mean = [0.5]
        std = [0.1]
        repr = "Normalize(mean=tensor([0.5000]), std=tensor([0.1000]))"
        assert str(kornia.enhance.Normalize(mean, std)) == repr

    def test_normalize(self, device, dtype):
        # prepare input data
        data = torch.ones(1, 2, 2, device=device, dtype=dtype)
        mean = torch.tensor([0.5], device=device, dtype=dtype)
        std = torch.tensor([2.0], device=device, dtype=dtype)

        # expected output
        expected = torch.tensor([0.25], device=device, dtype=dtype).repeat(1, 2, 2).view_as(data)

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_broadcast_normalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([2.0], device=device, dtype=dtype)
        std = torch.tensor([0.5], device=device, dtype=dtype)

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_int_input(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: int = 2
        std: int = 1

        # expected output
        expected = torch.ones_like(data)

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_float_input(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: float = 2.0
        std: float = 0.5

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_batch_normalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        # expected output
        expected = torch.tensor([1.25, 1, 0.5], device=device, dtype=dtype).repeat(2, 1, 1).view_as(data)

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    @pytest.mark.skip(reason="union type not supported")
    def test_jit(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.normalize
        op_script = torch.jit.script(op)

        self.assert_close(op(*inputs), op_script(*inputs))

    def test_gradcheck(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        data = utils.tensor_to_gradcheck_var(data)  # to var
        mean = utils.tensor_to_gradcheck_var(mean)  # to var
        std = utils.tensor_to_gradcheck_var(std)  # to var

        assert gradcheck(kornia.enhance.Normalize(mean, std), (data,), raise_exception=True, fast_mode=True)

    def test_single_value(self, device, dtype):
        # prepare input data
        mean = torch.tensor(2, device=device, dtype=dtype)
        std = torch.tensor(3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 256, 313, device=device, dtype=dtype)

        # expected output
        expected = (data - mean) / std

        self.assert_close(kornia.enhance.normalize(data, mean, std), expected)

    def test_module(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.normalize
        op_module = kornia.enhance.Normalize(mean, std)

        self.assert_close(op(*inputs), op_module(data))

    @pytest.mark.parametrize(
        "mean, std", [((1.0, 1.0, 1.0), (0.5, 0.5, 0.5)), (1.0, 0.5), (torch.tensor([1.0]), torch.tensor([0.5]))]
    )
    def test_random_normalize_different_parameter_types(self, mean, std):
        f = kornia.enhance.Normalize(mean=mean, std=std)
        data = torch.ones(2, 3, 256, 313)
        if isinstance(mean, float):
            expected = (data - torch.as_tensor(mean)) / torch.as_tensor(std)
        else:
            expected = (data - torch.as_tensor(mean[0])) / torch.as_tensor(std[0])
        self.assert_close(f(data), expected)

    @pytest.mark.parametrize("mean, std", [((1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5, 0.5)), ((1.0, 1.0), (0.5, 0.5))])
    def test_random_normalize_invalid_parameter_shape(self, mean, std):
        f = kornia.enhance.Normalize(mean=mean, std=std)
        inputs = torch.arange(0.0, 16.0, step=1).reshape(1, 4, 4).unsqueeze(0)
        with pytest.raises(ValueError):
            f(inputs)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass


class TestDenormalize(BaseTester):
    def test_smoke(self, device, dtype):
        mean = [0.5]
        std = [0.1]
        repr = "Denormalize(mean=[0.5], std=[0.1])"
        assert str(kornia.enhance.Denormalize(mean, std)) == repr

    def test_denormalize(self, device, dtype):
        # prepare input data
        data = torch.ones(1, 2, 2, device=device, dtype=dtype)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([2.5], device=device, dtype=dtype).repeat(1, 2, 2).view_as(data)

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    def test_broadcast_denormalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([2.0], device=device, dtype=dtype)
        std = torch.tensor([0.5], device=device, dtype=dtype)

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    def test_float_input(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: float = 2.0
        std: float = 0.5

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    def test_batch_denormalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        # expected output
        expected = torch.tensor([6.5, 7, 8], device=device, dtype=dtype).repeat(2, 1, 1).view_as(data)

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    @pytest.mark.skip(reason="union type not supported")
    def test_jit(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.denormalize
        op_script = torch.jit.script(op)

        self.assert_close(op(*inputs), op_script(*inputs))

    def test_gradcheck(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype)

        data = utils.tensor_to_gradcheck_var(data)  # to var
        mean = utils.tensor_to_gradcheck_var(mean)  # to var
        std = utils.tensor_to_gradcheck_var(std)  # to var

        assert gradcheck(kornia.enhance.Denormalize(mean, std), (data,), raise_exception=True, fast_mode=True)

    def test_single_value(self, device, dtype):
        # prepare input data
        mean = torch.tensor(2, device=device, dtype=dtype)
        std = torch.tensor(3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 256, 313, device=device, dtype=dtype)

        # expected output
        expected = (data * std) + mean

        self.assert_close(kornia.enhance.denormalize(data, mean, std), expected)

    def test_module(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.denormalize
        op_module = kornia.enhance.Denormalize(mean, std)

        self.assert_close(op(*inputs), op_module(data))

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass


class TestNormalizeMinMax(BaseTester):
    def test_smoke(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        assert kornia.enhance.normalize_min_max(x) is not None
        assert kornia.enhance.normalize_min_max(x) is not None

    def test_exception(self, device, dtype):
        x = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError):
            assert kornia.enhance.normalize_min_max(0.0)

        with pytest.raises(TypeError):
            assert kornia.enhance.normalize_min_max(x, "", "")

        with pytest.raises(TypeError):
            assert kornia.enhance.normalize_min_max(x, 2.0, "")

    @pytest.mark.parametrize("input_shape", [(1, 2, 3, 4), (2, 1, 4, 3), (1, 3, 2, 1)])
    def test_cardinality(self, device, dtype, input_shape):
        x = torch.rand(input_shape, device=device, dtype=dtype)
        assert kornia.enhance.normalize_min_max(x).shape == input_shape

    @pytest.mark.parametrize("min_val, max_val", [(1.0, 2.0), (2.0, 3.0), (5.0, 20.0), (40.0, 1000.0)])
    def test_range(self, device, dtype, min_val, max_val):
        x = torch.rand(1, 2, 4, 5, device=device, dtype=dtype)
        out = kornia.enhance.normalize_min_max(x, min_val=min_val, max_val=max_val)
        self.assert_close(out.min(), torch.tensor(min_val, device=device, dtype=dtype), low_tolerance=True)
        self.assert_close(out.max(), torch.tensor(max_val, device=device, dtype=dtype), low_tolerance=True)

    def test_values(self, device, dtype):
        x = torch.tensor([[[[0.0, 1.0, 3.0], [-1.0, 4.0, 3.0], [9.0, 5.0, 2.0]]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[-0.8, -0.6, -0.2], [-1.0, 0.0, -0.2], [1.0, 0.2, -0.4]]]], device=device, dtype=dtype
        )

        actual = kornia.enhance.normalize_min_max(x, min_val=-1.0, max_val=1.0)
        self.assert_close(actual, expected, low_tolerance=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        op = kornia.enhance.normalize_min_max
        op_jit = torch.jit.script(op)
        self.assert_close(op(x), op_jit(x))

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.enhance.normalize_min_max, (x,), raise_exception=True, fast_mode=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass
