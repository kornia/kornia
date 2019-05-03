import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import utils
from common import device_type


class TestRotate:
    def test_smoke(self):
        angle = 0.0
        angle_t = torch.tensor([angle])
        assert str(tgm.Rotate(angle=angle_t)) == 'Rotate(angle=0.0, center=None)'

    def test_angle90(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]])
        expected = torch.tensor([[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]])
        # prepare transformation
        angle = torch.tensor([90.])
        transform = tgm.Rotate(angle)
        assert_allclose(transform(inp), expected)
        
    def test_angle90_batch2(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]],[[
            [0., 0.],
            [5., 3.],
            [6., 4.],
            [0., 0.],
        ]]])
        # prepare transformation
        angle = torch.tensor([90., -90.])
        transform = tgm.Rotate(angle)
        assert_allclose(transform(inp), expected)

    def test_angle90_batch2_broadcast(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]],[[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]]])
        # prepare transformation
        angle = torch.tensor([90.])
        transform = tgm.Rotate(angle)
        assert_allclose(transform(inp), expected)

    def test_gradcheck(self):
        # test parameters
        angle = torch.tensor([90.])
        angle = utils.tensor_to_gradcheck_var(angle)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(tgm.rotate, (input, angle,), raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    def test_jit(self):
        angle = torch.tensor([90.])
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        rot = tgm.Rotate(angle)
        rot_traced = torch.jit.trace(tgm.Rotate(angle), img)
        assert_allclose(rot(img), rot_traced(img))
