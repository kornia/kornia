import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils
from kornia.testing import BaseTester


class TestSepia(BaseTester):
    def test_smoke(self, device, dtype):
        input_tensor = torch.ones((2, 3, 1, 1), device=device, dtype=dtype)
        expected_tensor = torch.tensor(
            [[[[1.0]], [[0.8905]], [[0.6936]]], [[[1.0]], [[0.8905]], [[0.6936]]]], device=device, dtype=dtype
        )

        actual = kornia.color.sepia(input_tensor, rescale=True)
        assert actual.shape[:] == (2, 3, 1, 1)
        self.assert_close(actual, expected_tensor, rtol=1e-4, atol=1e-4)

        input_tensor = torch.ones((3, 1, 1), device=device, dtype=dtype)
        expected_tensor = torch.tensor([[[1.0]], [[0.8905]], [[0.6936]]], device=device, dtype=dtype)

        actual = kornia.color.sepia(input_tensor, rescale=True)
        assert actual.shape[:] == (3, 1, 1)
        self.assert_close(actual, expected_tensor, rtol=1e-4, atol=1e-4)

    def test_sepia_calc_without_rescale(self, device, dtype):
        input_tensor = torch.tensor([[[0.1, 0, 1]], [[0.1, 0, 1]], [[0.1, 0, 1]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[0.1351, 0.0, 1.3510]], [[0.1203, 0.0, 1.2030]], [[0.0937, 0.0, 0.9370]]], device=device, dtype=dtype
        )

        actual = kornia.color.sepia(input_tensor, rescale=False)
        self.assert_close(actual, expected)

    def test_sepia_uint(self):
        input_tensor = torch.tensor([[[10, 0, 255]], [[10, 0, 255]], [[10, 0, 255]]], dtype=torch.uint8)

        expected = torch.tensor([[[112, 0, 168]], [[224, 0, 208]], [[76, 0, 18]]], dtype=torch.uint8)

        actual = kornia.color.sepia(input_tensor, rescale=False)
        self.assert_close(actual, expected)

    def test_exception(self, device, dtype):
        inp = torch.randint(1, 10, (3, 1, 1), dtype=torch.int32, device=device)

        with pytest.raises(TypeError):
            kornia.color.sepia(inp)

        with pytest.raises(ValueError):
            kornia.color.sepia(torch.rand(size=(4, 1, 1), dtype=dtype, device=device))

    @pytest.mark.parametrize("batch_shape", [(1, 3, 8, 15), (2, 3, 11, 7), (3, 8, 15)])
    def test_cardinality(self, batch_shape, device, dtype):
        input_tensor = torch.rand(batch_shape, device=device, dtype=dtype)
        actual = kornia.color.sepia(input_tensor)
        assert actual.shape == batch_shape

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = kornia.color.sepia(inp)

        assert inp.is_contiguous() is False
        assert actual.is_contiguous()
        self.assert_close(actual, actual)

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (1, 3, 5, 5)

        # evaluate function gradient
        input_tensor = torch.rand(batch_shape, device=device, dtype=dtype)
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)
        assert gradcheck(kornia.color.sepia, (input_tensor,), raise_exception=True)

    def test_jit(self, device, dtype):
        op = kornia.color.sepia
        op_script = torch.jit.script(op)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img), op_script(img))

    def test_module(self, device, dtype):
        op = kornia.color.sepia
        op_module = kornia.color.Sepia()

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img), op_module(img))
