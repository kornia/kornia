import pytest
import torch
from torch.autograd import gradcheck

import kornia.color as color
import kornia.color.covolutional as covolutional_color
import kornia.testing as utils


class TestSepia(utils.BaseTester):
    def test_smoke(self, device, dtype):
        input_tensor = torch.tensor(
            [[[0.1, 1.0], [0.2, 0.1]], [[0.1, 0.8], [0.2, 0.5]], [[0.1, 0.3], [0.2, 0.8]]], device=device, dtype=dtype
        )

        # With rescale
        expected_tensor = torch.tensor(
            [[[0.1269, 1.0], [0.2537, 0.5400]], [[0.1269, 1.0], [0.2537, 0.5403]], [[0.1269, 1.0], [0.2538, 0.5403]]],
            device=device,
            dtype=dtype,
        )
        actual = covolutional_color.sepia(input_tensor, rescale=True)

        assert actual.shape[:] == (3, 2, 2)
        self.assert_close(actual, expected_tensor, rtol=1e-2, atol=1e-2)

        # Without rescale
        expected_tensor = torch.tensor(
            [
                [[0.1351, 1.0649], [0.2702, 0.5750]],
                [[0.1203, 0.9482], [0.2406, 0.5123]],
                [[0.0937, 0.7385], [0.1874, 0.3990]],
            ],
            device=device,
            dtype=dtype,
        )

        actual = covolutional_color.sepia(input_tensor, rescale=False)
        assert actual.shape[:] == (3, 2, 2)
        self.assert_close(actual, expected_tensor, rtol=1e-2, atol=1e-2)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            covolutional_color.sepia(torch.rand(size=(4, 1, 1), dtype=dtype, device=device))

    @pytest.mark.parametrize("batch_shape", [(1, 3, 8, 15), (2, 3, 11, 7), (3, 8, 15)])
    def test_cardinality(self, batch_shape, device, dtype):
        input_tensor = torch.rand(batch_shape, device=device, dtype=dtype)
        actual = covolutional_color.sepia(input_tensor)
        assert actual.shape == batch_shape

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = covolutional_color.sepia(inp)

        assert inp.is_contiguous() is False
        assert actual.is_contiguous()
        self.assert_close(actual, actual)

    def test_unit(self, device, dtype):
        data = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.sepia(data), covolutional_color.sepia(data))

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (1, 3, 5, 5)

        # evaluate function gradient
        input_tensor = torch.rand(batch_shape, device=device, dtype=dtype)
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)
        assert gradcheck(covolutional_color.sepia, (input_tensor,), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass
