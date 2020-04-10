import pytest

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.augmentation.functional as F
import kornia.testing as utils  # test utils
from test.common import device
import kornia.augmentation.random as pg


class TestPerspective:

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 3, 4).to(device)
        batch_prob = torch.rand(1) < 0.5
        start_points = torch.rand(1, 4, 2).to(device)
        end_points = torch.rand(1, 4, 2).to(device)

        params = dict(batch_prob=batch_prob, start_points=start_points, end_points=end_points)
        out_data = F.apply_perspective(x_data, params, return_transform=False)

        assert out_data.shape == x_data.shape

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 3, 4).to(device)
        batch_prob = torch.rand(1) < 0.5
        start_points = torch.rand(1, 4, 2).to(device)
        end_points = torch.rand(1, 4, 2).to(device)

        params = dict(batch_prob=batch_prob, start_points=start_points, end_points=end_points)
        out_data = F.apply_perspective(x_data, params, return_transform=True)

        assert isinstance(out_data, tuple)
        assert len(out_data) == 2
        assert out_data[0].shape == x_data.shape
        assert out_data[1].shape == (1, 3, 3)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var

        batch_prob = torch.rand(1) < 0.5

        start_points = torch.rand(1, 4, 2).to(device)
        start_points = utils.tensor_to_gradcheck_var(start_points)  # to var

        end_points = torch.rand(1, 4, 2).to(device)
        end_points = utils.tensor_to_gradcheck_var(end_points)  # to var

        params = dict(batch_prob=batch_prob, start_points=start_points, end_points=end_points)
        assert gradcheck(F.apply_perspective, (input, params,), raise_exception=True)


class TestRandomPerspective:

    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)

        out_perspective = kornia.augmentation.functional.random_perspective(
            x_data, 0.5, 0.5, return_transform=False)

        assert out_perspective.shape == x_data.shape

    def test_smoke_no_transform_batch(self, device):
        x_data = torch.rand(2, 2, 8, 9).to(device)

        out_perspective = kornia.augmentation.functional.random_perspective(
            x_data, 0.5, 0.5, return_transform=False)

        assert out_perspective.shape == x_data.shape

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 4, 5).to(device)

        out_perspective = kornia.augmentation.functional.random_perspective(
            x_data, 0.5, 0.5, return_transform=True)

        assert isinstance(out_perspective, tuple)
        assert len(out_perspective) == 2
        assert out_perspective[0].shape == x_data.shape
        assert out_perspective[1].shape == (1, 3, 3)

    def test_no_transform_module(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)
        out_perspective = kornia.augmentation.RandomPerspective()(x_data)
        assert out_perspective.shape == x_data.shape

    def test_transform_module(self, device):
        x_data = torch.rand(1, 2, 4, 5).to(device)

        out_perspective = kornia.augmentation.RandomPerspective(
            return_transform=True)(x_data)

        assert isinstance(out_perspective, tuple)
        assert len(out_perspective) == 2
        assert out_perspective[0].shape == x_data.shape
        assert out_perspective[1].shape == (1, 3, 3)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(F.random_perspective, (input, 0., 1.), raise_exception=True)


class TestRandomAffine:

    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)
        out = F.random_affine(x_data, 0.)
        assert out.shape == x_data.shape

    def test_smoke_no_transform_batch(self, device):
        x_data = torch.rand(2, 2, 8, 9).to(device)
        out = F.random_affine(x_data, 0.)
        assert out.shape == x_data.shape

    def test_batch_multi_params(self, device):
        x_data = torch.rand(2, 2, 8, 9).to(device)
        out = F.random_affine(x_data, 0., (0., 0.))
        assert out.shape == x_data.shape

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 4, 5).to(device)
        out = F.random_affine(x_data, 0., return_transform=True)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == x_data.shape
        assert out[1].shape == (1, 3, 3)

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
        matrix_kornia = F._compose_affine_matrix_3x3(
            T(translations).view(-1, 2),
            center,
            T([scale]).view(-1),
            T([angle]).view(-1),
            T([math.radians(shear[0])]).view(-1, 1),
            T([math.radians(shear[1])]).view(-1, 1))
        matrix_kornia = matrix_kornia.inverse()[0, :2].detach().cpu()
        assert_allclose(matrix_kornia, matrix_expected)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.RandomAffine(0.), (input, ), raise_exception=True)
