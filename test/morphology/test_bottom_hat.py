import pytest
import torch
<<<<<<< HEAD
from kornia.morphology.morphology import bottom_hat
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestBlackHat:

=======
<<<<<<< HEAD:test/morphology/test_opening.py
from kornia.morphology.morphology import opening
=======
>>>>>>> upstream/master:test/morphology/test_bottom_hat.py
from torch.autograd import gradcheck
from torch.testing import assert_allclose

from kornia.morphology.morphology import bottom_hat

<<<<<<< HEAD:test/morphology/test_opening.py
class TestOpening:
=======
>>>>>>> upstream/master:test/morphology/test_bottom_hat.py

class TestBlackHat:
>>>>>>> upstream/master
    def test_smoke(self, device, dtype):
        kernel = torch.rand(3, 3, device=device, dtype=dtype)
        assert kernel is not None

<<<<<<< HEAD
    @pytest.mark.parametrize(
        "shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 5, 5)])
    @pytest.mark.parametrize(
        "kernel", [(3, 3), (5, 5)])
    def test_cardinality(self, device, dtype, shape, kernel):
        img = torch.ones(shape, device=device, dtype=dtype)
        krnl = torch.ones(kernel, device=device, dtype=dtype)
        assert bottom_hat(img, krnl).shape == shape

    def test_value(self, device, dtype):
        input = torch.tensor([[0.5, 1., 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]],
                             device=device, dtype=dtype)[None, None, :, :]
        kernel = torch.tensor([[-1., 0., -1.], [0., 0., 0.], [-1., 0., -1.]], device=device, dtype=dtype)
        expected = torch.tensor([[0.2, 0., 0.5], [0., 0.4, 0.], [0.3, 0., 0.6]],
                                device=device, dtype=dtype)[None, None, :, :]
        assert_allclose(bottom_hat(input, kernel), expected)

    def test_exception(self, device, dtype):
        input = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert bottom_hat([0.], kernel)

        with pytest.raises(TypeError):
            assert bottom_hat(input, [0.])
=======
    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 5, 5)])
    @pytest.mark.parametrize("kernel", [(3, 3), (5, 5)])
    def test_cardinality(self, device, dtype, shape, kernel):
        img = torch.ones(shape, device=device, dtype=dtype)
        krnl = torch.ones(kernel, device=device, dtype=dtype)
<<<<<<< HEAD:test/morphology/test_opening.py
        assert opening(img, krnl).shape == shape

    def test_value(self, device, dtype):
        tensor = torch.tensor([[0.5, 1., 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]],
                              device=device, dtype=dtype)[None, None, :, :]
        kernel = torch.tensor([[-1., 0., -1.], [0., 0., 0.], [-1., 0., -1.]], device=device, dtype=dtype)
        expected = torch.tensor([[0.5, 0.5, 0.3], [0.5, 0.3, 0.3], [0.4, 0.4, 0.2]],
                                device=device, dtype=dtype)[None, None, :, :]
        assert_allclose(opening(tensor, kernel), expected)
=======
        assert bottom_hat(img, krnl).shape == shape

    def test_value(self, device, dtype):
        input = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.2, 0.0, 0.5], [0.0, 0.4, 0.0], [0.3, 0.0, 0.6]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_allclose(bottom_hat(input, kernel), expected)
>>>>>>> upstream/master:test/morphology/test_bottom_hat.py

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
<<<<<<< HEAD:test/morphology/test_opening.py
            assert opening([0.], kernel)

        with pytest.raises(TypeError):
            assert opening(tensor, [0.])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert opening(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert opening(tensor, test)
=======
            assert bottom_hat([0.0], kernel)

        with pytest.raises(TypeError):
            assert bottom_hat(input, [0.0])
>>>>>>> upstream/master

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert bottom_hat(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert bottom_hat(input, test)
<<<<<<< HEAD

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        input = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=torch.float64)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=torch.float64)
=======
>>>>>>> upstream/master:test/morphology/test_bottom_hat.py

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        tensor = torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=torch.float64)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=torch.float64)
<<<<<<< HEAD:test/morphology/test_opening.py
        assert gradcheck(opening, (tensor, kernel), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = opening
=======
>>>>>>> upstream/master
        assert gradcheck(bottom_hat, (input, kernel), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = bottom_hat
<<<<<<< HEAD
        op_script = torch.jit.script(op)

        input = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(input, kernel)
        expected = op(input, kernel)
=======
>>>>>>> upstream/master:test/morphology/test_bottom_hat.py
        op_script = torch.jit.script(op)

        tensor = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(tensor, kernel)
        expected = op(tensor, kernel)
>>>>>>> upstream/master

        assert_allclose(actual, expected)
