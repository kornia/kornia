import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


@pytest.mark.parametrize("batch_size", [0, 1, 5])
@pytest.mark.parametrize("ksize", [3, 11])
@pytest.mark.parametrize("angle", [0., 360.])
@pytest.mark.parametrize("direction", [-1., 1.])
def test_get_motion_kernel2d(batch_size, ksize, angle, direction):
    if batch_size != 0:
        angle = torch.tensor([angle] * batch_size)
        direction = torch.tensor([direction] * batch_size)
    else:
        batch_size = 1
    kernel = kornia.get_motion_kernel2d(ksize, angle, direction)
    assert kernel.shape == (batch_size, ksize, ksize)
    assert_allclose(kernel.sum(), batch_size)


class TestMotionBlur:
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_motion_blur(self, batch_shape, device, dtype):
        ksize = 5
        angle = 200.
        direction = 0.3

        input = torch.rand(batch_shape, device=device, dtype=dtype)
        motion = kornia.filters.MotionBlur(ksize, angle, direction)
        assert motion(input).shape == batch_shape

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = 3
        angle = 200.
        direction = 0.3
        actual = kornia.filters.motion_blur(inp, kernel_size, angle, direction)
        expected = actual
        assert_allclose(actual, actual)

    def test_gradcheck(self, device, dtype):
        batch_shape = (1, 3, 4, 5)
        ksize = 9
        angle = 34.
        direction = -0.2

        input = torch.rand(batch_shape, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)
        assert gradcheck(
            kornia.motion_blur,
            (input, ksize, angle, direction, "replicate"),
            raise_exception=True,
        )

    @pytest.mark.skip("angle can be Union")
    def test_jit(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        ksize = 5
        angle = 65.
        direction = .1
        op = kornia.filters.motion_blur
        op_script = torch.jit.script(op)
        actual = op_script(img, ksize, angle, direction)
        expected = op(img, ksize, angle, direction)
        assert_allclose(actual, expected)
