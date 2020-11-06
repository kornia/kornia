import pytest
import torch
import kornia.morphology as m
import kornia.testing as utils  # test utils
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestTopHat(utils.BaseTester):

    def test_smoke(self, device, dtype):
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        assert m.se_to_mask(kernel) is not None

    def test_batch(self, device, dtype):
        input = torch.rand(3, 2, 6, 10, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = m.top_hat(input, kernel)
        assert input.shape == test.shape == (3, 2, 6, 10)

    def test_value(self, device, dtype):
        input = torch.tensor([[0.5, 1., 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]],
                             device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=device, dtype=dtype)
        expected = torch.tensor([[0., 0.5, 0.], [0.2, 0., 0.5], [0., 0.5, 0.]],
                                device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        assert_allclose(m.top_hat(input, kernel), expected)

    def test_exception(self, device, dtype):
        input = torch.ones(1, 1, 3, 4, device=device, dtype=dtype).double()
        kernel = torch.ones(3, 3, device=device, dtype=dtype).double()

        with pytest.raises(TypeError):
            assert m.top_hat([0.], kernel)

        with pytest.raises(TypeError):
            assert m.top_hat(input, [0.])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4)
            assert m.top_hat(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4)
            assert m.top_hat(input, test)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=dtype).double()
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=dtype).double()
        assert gradcheck(m.top_hat, (input, kernel), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        op = m.top_hat
        op_script = torch.jit.script(op)

        input = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(input, kernel)
        expected = op(input, kernel)

        assert_allclose(actual, expected)
