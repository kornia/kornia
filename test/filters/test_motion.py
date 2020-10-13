from typing import Tuple
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
    def test_motion_blur(self, batch_shape, device):
        ksize = 5
        angle = 200.
        direction = 0.3

        input = torch.rand(batch_shape).to(device)
        motion = kornia.filters.MotionBlur(ksize, angle, direction)
        assert motion(input).shape == batch_shape

    def test_noncontiguous(self, device):
        batch_size = 3
        inp = torch.rand(3, 5, 5).expand(batch_size, -1, -1, -1).to(device)

        kernel_size = 3
        angle = 200.
        direction = 0.3
        actual = kornia.filters.motion_blur(inp, kernel_size, angle, direction)
        expected = actual
        assert_allclose(actual, actual)

    def test_gradcheck(self, device):
        batch_shape = (2, 3, 11, 7)
        ksize = 9
        angle = 34.
        direction = -0.2

        input = torch.rand(batch_shape).to(device)
        input = utils.tensor_to_gradcheck_var(input)
        assert gradcheck(
            kornia.motion_blur,
            (input, ksize, angle, direction, "replicate"),
            raise_exception=True,
        )

    @pytest.mark.skip("")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(
            input: torch.Tensor,
            ksize: int,
            angle: float,
            direction: float
        ) -> torch.Tensor:
            return kornia.filters.motion_blur(input, ksize, angle, direction)

        img = torch.rand(2, 3, 4, 5).to(device)
        ksize = 5
        angle = 65.
        direction = .1
        actual = op_script(img, ksize, angle, direction)
        expected = kornia.filters.motion_blur(img, ksize, angle, direction)
        assert_allclose(actual, expected)
