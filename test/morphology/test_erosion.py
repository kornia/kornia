import pytest
import torch
import kornia.morphology as morph
import kornia.testing as utils  # test utils
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestErode(utils.BaseTester):

    def test_smoke(self, device, dtype):
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        assert morph.se_to_mask(kernel) is not None

    def test_batch(self, device, dtype):
        input = torch.rand(3, 2, 6, 10, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        test = morph.erosion(input, kernel)
        assert input.shape == test.shape == (3, 2, 6, 10)

    def test_value(self, device, dtype):
        input = torch.tensor([[0.5, 1., 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]],
                             device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=device, dtype=dtype)
        expected = torch.tensor([[0.5, 0.3, 0.3], [0.3, 0.3, 0.2], [0.4, 0.2, 0.2]],
                                device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        assert_allclose(morph.erosion(input, kernel), expected)

    def test_exception(self, device, dtype):
        input = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert morph.erosion([0.], kernel)

        with pytest.raises(TypeError):
            assert morph.erosion(input, [0.])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert morph.erosion(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert morph.erosion(input, test)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=dtype)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=dtype)
        assert gradcheck(morph.erosion, (input, kernel), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        op = morph.erosion
        op_script = torch.jit.script(op)

        input = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(input, kernel)
        expected = op(input, kernel)

        assert_allclose(actual, expected)
