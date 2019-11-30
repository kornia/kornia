import pytest

import kornia as kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRotate:
    def test_angle90(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).to(device)
        expected = torch.tensor([[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]]).to(device)
        # prepare transformation
        angle = torch.tensor([90.]).to(device)
        transform = kornia.Rotate(angle)
        assert_allclose(transform(inp), expected)

    def test_angle90_batch2(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1).to(device)
        expected = torch.tensor([[[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]], [[
            [0., 0.],
            [5., 3.],
            [6., 4.],
            [0., 0.],
        ]]]).to(device)
        # prepare transformation
        angle = torch.tensor([90., -90.]).to(device)
        transform = kornia.Rotate(angle)
        assert_allclose(transform(inp), expected)

    def test_angle90_batch2_broadcast(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1).to(device)
        expected = torch.tensor([[[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]], [[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]]]).to(device)
        # prepare transformation
        angle = torch.tensor([90.]).to(device)
        transform = kornia.Rotate(angle)
        assert_allclose(transform(inp), expected)

    def test_gradcheck(self, device):
        # test parameters
        angle = torch.tensor([90.]).to(device)
        angle = utils.tensor_to_gradcheck_var(
            angle, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.rotate, (input, angle,), raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        angle = torch.tensor([90.]).to(device)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        rot = kornia.Rotate(angle)
        rot_traced = torch.jit.trace(kornia.Rotate(angle), img)
        assert_allclose(rot(img), rot_traced(img))


class TestTranslate:
    def test_dxdy(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).to(device)
        expected = torch.tensor([[
            [0., 1.],
            [0., 3.],
            [0., 5.],
            [0., 7.],
        ]]).to(device)
        # prepare transformation
        translation = torch.tensor([[1., 0.]]).to(device)
        transform = kornia.Translate(translation)
        assert_allclose(transform(inp), expected)

    def test_dxdy_batch(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1).to(device)
        expected = torch.tensor([[[
            [0., 1.],
            [0., 3.],
            [0., 5.],
            [0., 7.],
        ]], [[
            [0., 0.],
            [0., 1.],
            [0., 3.],
            [0., 5.],
        ]]]).to(device)
        # prepare transformation
        translation = torch.tensor([[1., 0.], [1., 1.]]).to(device)
        transform = kornia.Translate(translation)
        assert_allclose(transform(inp), expected)

    def test_dxdy_batch_broadcast(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1).to(device)
        expected = torch.tensor([[[
            [0., 1.],
            [0., 3.],
            [0., 5.],
            [0., 7.],
        ]], [[
            [0., 1.],
            [0., 3.],
            [0., 5.],
            [0., 7.],
        ]]]).to(device)
        # prepare transformation
        translation = torch.tensor([[1., 0.]]).to(device)
        transform = kornia.Translate(translation)
        assert_allclose(transform(inp), expected)

    def test_gradcheck(self, device):
        # test parameters
        translation = torch.tensor([[1., 0.]]).to(device)
        translation = utils.tensor_to_gradcheck_var(
            translation, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.translate, (input, translation,),
                         raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        translation = torch.tensor([[1., 0.]]).to(device)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        trans = kornia.Translate(translation)
        trans_traced = torch.jit.trace(kornia.Translate(translation), img)
        assert_allclose(trans(img), trans_traced(img))


class TestScale:
    def test_scale_factor_2(self, device):
        # prepare input data
        inp = torch.tensor([[
            [0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.]
        ]]).to(device)
        # prepare transformation
        scale_factor = torch.tensor([2.]).to(device)
        transform = kornia.Scale(scale_factor)
        assert_allclose(transform(inp).sum().item(), 12.25)

    def test_scale_factor_05(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).to(device)
        expected = torch.tensor([[
            [0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.]
        ]]).to(device)
        # prepare transformation
        scale_factor = torch.tensor([0.5]).to(device)
        transform = kornia.Scale(scale_factor)
        assert_allclose(transform(inp), expected)

    def test_scale_factor_05_batch2(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).repeat(2, 1, 1, 1).to(device)
        expected = torch.tensor([[
            [0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.]
        ]]).to(device)
        # prepare transformation
        scale_factor = torch.tensor([0.5, 0.5]).to(device)
        transform = kornia.Scale(scale_factor)
        assert_allclose(transform(inp), expected)

    def test_scale_factor_05_batch2_broadcast(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).repeat(2, 1, 1, 1).to(device)
        expected = torch.tensor([[
            [0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.]
        ]]).to(device)
        # prepare transformation
        scale_factor = torch.tensor([0.5]).to(device)
        transform = kornia.Scale(scale_factor)
        assert_allclose(transform(inp), expected)

    def test_gradcheck(self, device):
        # test parameters
        scale_factor = torch.tensor([0.5]).to(device)
        scale_factor = utils.tensor_to_gradcheck_var(
            scale_factor, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.scale, (input, scale_factor,),
                         raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        scale_factor = torch.tensor([0.5]).to(device)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        trans = kornia.Scale(scale_factor)
        trans_traced = torch.jit.trace(kornia.Scale(scale_factor), img)
        assert_allclose(trans(img), trans_traced(img))


class TestShear:
    def test_shear_x(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).to(device)
        expected = torch.tensor([[
            [1., 1., 1., 1.],
            [.5, 1., 1., 1.],
            [0., 1., 1., 1.],
            [0., .5, 1., 1.]
        ]]).to(device)

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0]]).to(device)
        transform = kornia.Shear(shear)
        assert_allclose(transform(inp), expected)

    def test_shear_y(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).to(device)
        expected = torch.tensor([[
            [1., .5, 0., 0.],
            [1., 1., 1., .5],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).to(device)

        # prepare transformation
        shear = torch.tensor([[0.0, 0.5]]).to(device)
        transform = kornia.Shear(shear)
        assert_allclose(transform(inp), expected)

    def test_shear_batch2(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).repeat(2, 1, 1, 1).to(device)

        expected = torch.tensor([[[
            [1., 1., 1., 1.],
            [.5, 1., 1., 1.],
            [0., 1., 1., 1.],
            [0., .5, 1., 1.]
        ]], [[
            [1., .5, 0., 0.],
            [1., 1., 1., .5],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]]).to(device)

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0], [0.0, 0.5]]).to(device)
        transform = kornia.Shear(shear)
        assert_allclose(transform(inp), expected)

    def test_shear_batch2_broadcast(self, device):
        # prepare input data
        inp = torch.tensor([[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]).repeat(2, 1, 1, 1).to(device)

        expected = torch.tensor([[[
            [1., 1., 1., 1.],
            [.5, 1., 1., 1.],
            [0., 1., 1., 1.],
            [0., .5, 1., 1.]
        ]]]).to(device)

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0]]).to(device)
        transform = kornia.Shear(shear)
        assert_allclose(transform(inp), expected)

    def test_gradcheck(self, device):
        # test parameters
        shear = torch.tensor([[0.5, 0.0]]).to(device)
        shear = utils.tensor_to_gradcheck_var(
            shear, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.shear, (input, shear,), raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        shear = torch.tensor([[0.5, 0.0]]).to(device)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        trans = kornia.Shear(shear)
        trans_traced = torch.jit.trace(kornia.Shear(shear), img)
        assert_allclose(trans(img), trans_traced(img))
