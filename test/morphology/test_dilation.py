import pytest
import torch
<<<<<<< HEAD
from kornia.morphology.morphology import dilation
=======
>>>>>>> upstream/master
from torch.autograd import gradcheck
from torch.testing import assert_allclose

from kornia.morphology.morphology import dilation

<<<<<<< HEAD
class TestDilate:
=======
>>>>>>> upstream/master

class TestDilate:
    def test_smoke(self, device, dtype):
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        assert kernel is not None

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 5, 5)])
    @pytest.mark.parametrize("kernel", [(3, 3), (5, 5), (3, 5), (5, 3)])
    def test_cardinality(self, device, dtype, shape, kernel):
        img = torch.ones(shape, device=device, dtype=dtype)
        krnl = torch.ones(kernel, device=device, dtype=dtype)
        assert dilation(img, krnl).shape == shape

    def test_value(self, device, dtype):
<<<<<<< HEAD
        tensor = torch.tensor([[0.5, 1., 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]],
                              device=device, dtype=dtype)[None, None, :, :]
        kernel = torch.tensor([[-1., 0., -1.], [0., 0., 0.], [-1., 0., -1.]], device=device, dtype=dtype)
        expected = torch.tensor([[1., 1., 1.], [0.7, 1., 0.8], [0.9, 0.9, 0.9]],
                                device=device, dtype=dtype)[None, None, :, :]
=======
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 1.0, 1.0], [0.7, 1.0, 0.8], [0.9, 0.9, 0.9]], device=device, dtype=dtype)[
            None, None, :, :
        ]
>>>>>>> upstream/master
        assert_allclose(dilation(tensor, kernel), expected)

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert dilation([0.0], kernel)

        with pytest.raises(TypeError):
<<<<<<< HEAD
            assert dilation(tensor, [0.])
=======
            assert dilation(tensor, [0.0])
>>>>>>> upstream/master

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert dilation(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert dilation(tensor, test)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        tensor = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=torch.float64)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=torch.float64)
        assert gradcheck(dilation, (tensor, kernel), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = dilation
        op_script = torch.jit.script(op)

        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(tensor, kernel)
        expected = op(tensor, kernel)

        assert_allclose(actual, expected)
