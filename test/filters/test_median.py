import pytest
import torch

from kornia.filters import MedianBlur, median_blur
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestMedianBlur(BaseTester):
    def test_smoke(self, device, dtype):
        inp = torch.zeros(1, 3, 4, 4, device=device, dtype=dtype)
        actual = median_blur(inp, 3)
        assert isinstance(actual, torch.Tensor)

    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('kernel_size', [3, (5, 7)])
    def test_cardinality(self, batch_size, kernel_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)
        actual = median_blur(inp, kernel_size)
        assert actual.shape == (batch_size, 3, 4, 4)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError) as errinfo:
            median_blur(1, 1)
        assert 'Not a Tensor type.' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            median_blur(torch.ones(1, 1, device=device, dtype=dtype), 1)
        assert 'shape must be [[\'B\', \'C\', \'H\', \'W\']].' in str(errinfo)

    def test_kernel_3x3(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 7.0, 5.0, 0.0],
                    [0.0, 3.0, 1.0, 1.0, 0.0],
                    [0.0, 6.0, 9.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [36.0, 7.0, 25.0, 0.0, 0.0],
                    [3.0, 14.0, 1.0, 0.0, 0.0],
                    [65.0, 59.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        kernel_size = (3, 3)
        actual = median_blur(inp, kernel_size)
        self.assert_close(actual[0, 0, 2, 2], torch.tensor(3.0, device=device, dtype=dtype))
        self.assert_close(actual[0, 1, 1, 1], torch.tensor(14.0, device=device, dtype=dtype))

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        actual = median_blur(inp, kernel_size)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(median_blur, (img, (5, 3)))

    def test_module(self, device, dtype):
        kernel_size = (3, 5)
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = median_blur
        op_module = MedianBlur((3, 5))
        actual = op_module(img)
        expected = op(img, kernel_size)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize('kernel_size', [5, (5, 7)])
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        inpt = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MedianBlur(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))
