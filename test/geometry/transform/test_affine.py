import pytest

import kornia as kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestResize:
    def test_smoke(self, device):
        inp = torch.rand(1, 3, 3, 4).to(device)
        out = kornia.resize(inp, (3, 4))
        assert_allclose(inp, out)

    def test_upsize(self, device):
        inp = torch.rand(1, 3, 3, 4).to(device)
        out = kornia.resize(inp, (6, 8))
        assert out.shape == (1, 3, 6, 8)

    def test_downsize(self, device):
        inp = torch.rand(1, 3, 5, 2).to(device)
        out = kornia.resize(inp, (3, 1))
        assert out.shape == (1, 3, 3, 1)

    def test_one_param(self, device):
        inp = torch.rand(1, 3, 5, 2).to(device)
        out = kornia.resize(inp, 10)
        assert out.shape == (1, 3, 25, 10)

    def test_gradcheck(self, device):
        # test parameters
        new_size = 4
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.Resize(new_size), (input, ), raise_exception=True)


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


class TestAffine2d:

    def test_compose_affine_matrix_3x3(self, device):
        """ To get parameters:
        import torchvision as tv
        from PIL import Image
        from torch import Tensor as T
        import math
        import random
        img_size = (96,96)
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        tfm = tv.transforms.RandomAffine(degrees=(-25.0,25.0),
                                        scale=(0.6, 1.4) ,
                                        translate=(0, 0.1),
                                        shear=(-25., 25., -20., 20.))
        angle, translations, scale, shear = tfm.get_params(tfm.degrees, tfm.translate,
                                                        tfm.scale, tfm.shear, img_size)
        print (angle, translations, scale, shear)
        output_size = img_size
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)

        matrix = tv.transforms.functional._get_inverse_affine_matrix(center, angle, translations, scale, shear)
        matrix = np.array(matrix).reshape(2,3)
        print (matrix)
        """
        from torch import Tensor as T
        import math
        batch_size, ch, height, width = 1, 1, 96, 96
        angle, translations = 6.971339922894188, (0.0, -4.0)
        scale, shear = 0.7785685905190581, [11.823560708200617, 7.06797949691645]
        matrix_expected = T([[1.27536969, 4.26828945e-01, -3.23493180e+01],
                             [2.18297196e-03, 1.29424165e+00, -9.19962753e+00]])
        center = T([float(width), float(height)]).view(1, 2) / 2. + 0.5
        center = center.expand(batch_size, -1)
        matrix_kornia = kornia.get_affine_matrix2d(
            T(translations).view(-1, 2),
            center,
            T([scale]).view(-1),
            T([angle]).view(-1),
            T([math.radians(shear[0])]).view(-1, 1),
            T([math.radians(shear[1])]).view(-1, 1))
        matrix_kornia = matrix_kornia.inverse()[0, :2].detach().cpu()
        assert_allclose(matrix_kornia, matrix_expected)
