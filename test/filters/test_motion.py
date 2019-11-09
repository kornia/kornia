from typing import Tuple
import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


@pytest.mark.parametrize("ksize", [3, 11])
@pytest.mark.parametrize("angle", [0., 360.])
@pytest.mark.parametrize("direction", [-1., 1.])
def test_get_motion_kernel2d(ksize, angle, direction):
    kernel = kornia.get_motion_kernel2d((ksize, ksize), angle, direction)
    assert kernel.shape == (ksize, ksize)
    assert kernel.sum().item() == pytest.approx(1.0)


class TestMotionBlur:
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("device_type", ("cpu", "cuda"))
    def test_motion_blur(self, batch_shape, device_type):
        ksize = 5
        angle = 200.
        direction = 0.3

        input = torch.rand(batch_shape).to(torch.device(device_type))
        motion = kornia.filters.MotionBlur((ksize, ksize), angle, direction)
        assert motion(input).shape == batch_shape

    def test_gradcheck(self):
        batch_shape = (2, 3, 11, 7)
        ksize = 9
        angle = 34.
        direction = -0.2

        input = torch.rand(batch_shape)
        input = utils.tensor_to_gradcheck_var(input)
        assert gradcheck(
            kornia.motion_blur,
            (input, (ksize, ksize), angle, direction, "replicate"),
            raise_exception=True,
        )

    @pytest.mark.skip("")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(
            input: torch.Tensor,
            kernel_size: Tuple[int, int],
            angle: float,
            direction: float
        ) -> torch.Tensor:
            return kornia.filters.motion_blur(input, kernel_size, angle, direction)

        img = torch.rand(2, 3, 4, 5)
        kernel_size = (5, 5)
        angle = 65.
        direction = .1
        actual = op_script(img, kernel_size, angle, direction)
        expected = kornia.filters.motion_blur(img, kernel_size, angle, direction)
        assert_allclose(actual, expected)
