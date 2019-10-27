import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import math
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

class TestDilation:
    def test_smoke_none(self):
        assert kornia.morphology.dilation(
            torch.zeros([1,6,6]), torch.ones([3,3])
        ).shape == (1, 6, 6)

    @pytest.mark.parametrize('inp_shape', [(3,3), (3,4), (5,5), (4,2)])
    @pytest.mark.parametrize('st_shape', [(1,1), (3,3)])
    def test_zero_structuring_elem(self, inp_shape, st_shape):
        # If the structuring element is all zeros, then the dilated image is all zeros
        img = (torch.rand(inp_shape) > 0.5).float().unsqueeze(0)
        structuring_elem = torch.zeros(st_shape).float()
        assert_allclose(kornia.morphology.dilation(img, structuring_elem), torch.zeros(inp_shape))

    @pytest.mark.parametrize('inp_shape', [(3,3), (3,4), (5,5), (4,2)])
    def test_1_times_1_structuring_element(self, inp_shape):
        # If the structuring element is 1x1, then the dilated image is same as input image
        structuring_elem = torch.ones([1,1]).float()
        img = (torch.rand(inp_shape) > 0.5).float().unsqueeze(0)
        assert_allclose(kornia.morphology.dilation(img, structuring_elem), img)

    def test_impulse_response(self):
        # Image with only one "1" should give back the original structuring element
        structuring_elem = torch.tensor([[1,1,1], [1,0,1], [1,1,1]]).float()
        img = [[0]*5 for i in range(5)]
        img[2][2] = 1
        img = torch.tensor(img).unsqueeze(0).float()
        expected = [[0]*5 for i in range(5)]
        expected[1][1:4] = [1]*3
        expected[2][1] = 1
        expected[2][3] = 1
        expected[3][1:4] = [1]*3
        expected = torch.tensor(expected).unsqueeze(0).float()
        assert_allclose(kornia.morphology.dilation(img, structuring_elem), expected)

    @pytest.mark.parametrize('inp_shape', [(3,3), (3,4), (5,5), (4,2)])
    @pytest.mark.parametrize('st_shape', [(1,1), (3,3)])
    def test_all_ones_or_zeros_input(self, inp_shape, st_shape):
        # If the input is all zeros, then the dilated image is also all zeros
        input = torch.zeros(inp_shape).unsqueeze(0)
        structuring_elem = (torch.rand(st_shape) > 0.5).float()
        assert_allclose(kornia.morphology.dilation(input, structuring_elem), input)

    # TODO add more tests
