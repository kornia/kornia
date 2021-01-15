import pytest

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.augmentation.functional as F
import kornia.testing as utils  # test utils


class TestPerspective:

    def test_smoke(self, device):
        x_data = torch.rand(1, 2, 3, 4).to(device)
        batch_prob = torch.rand(1, device=device) < 0.5
        start_points = torch.rand(1, 4, 2).to(device)
        end_points = torch.rand(1, 4, 2).to(device)

        params = dict(
            batch_prob=batch_prob, start_points=start_points,
            end_points=end_points)
        flags = dict(
            interpolation=torch.tensor(1),
            align_corners=torch.tensor(False)
        )
        out_data = F.apply_perspective(x_data, params, flags)

        assert out_data.shape == x_data.shape

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var

        start_points = torch.rand(1, 4, 2).to(device)
        start_points = utils.tensor_to_gradcheck_var(start_points)  # to var

        end_points = torch.rand(1, 4, 2).to(device)
        end_points = utils.tensor_to_gradcheck_var(end_points)  # to var

        params = dict(
            start_points=start_points,
            end_points=end_points
        )
        flags = dict(
            interpolation=torch.tensor(1),
            align_corners=torch.tensor(False)
        )
        assert gradcheck(F.apply_perspective, (input, params, flags,), raise_exception=True)


class TestRandomPerspective:

    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)

        out_perspective = kornia.augmentation.RandomPerspective(
            0.5, p=0.5, return_transform=False)(x_data)

        assert out_perspective.shape == x_data.shape

    def test_smoke_no_transform_batch(self, device):
        x_data = torch.rand(2, 2, 8, 9).to(device)

        out_perspective = kornia.augmentation.RandomPerspective(
            0.5, p=0.5, return_transform=False)(x_data)

        assert out_perspective.shape == x_data.shape

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 4, 5).to(device)

        out_perspective = kornia.augmentation.RandomPerspective(
            0.5, p=0.5, return_transform=True)(x_data)

        assert isinstance(out_perspective, tuple)
        assert len(out_perspective) == 2
        assert out_perspective[0].shape == x_data.shape
        assert out_perspective[1].shape == (1, 3, 3)

    def test_no_transform_module(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)
        out_perspective = kornia.augmentation.RandomPerspective()(x_data)
        assert out_perspective.shape == x_data.shape

    def test_transform_module_should_return_identity(self, device):
        torch.manual_seed(0)
        x_data = torch.rand(1, 2, 4, 5).to(device)

        out_perspective = kornia.augmentation.RandomPerspective(p=0.0,
                                                                return_transform=True)(x_data)
        assert isinstance(out_perspective, tuple)
        assert len(out_perspective) == 2
        assert out_perspective[0].shape == x_data.shape
        assert out_perspective[1].shape == (1, 3, 3)
        assert_allclose(out_perspective[0], x_data)
        assert_allclose(out_perspective[1], torch.eye(3, device=device))

    def test_transform_module_should_return_expected_transform(self, device):
        torch.manual_seed(0)
        x_data = torch.rand(1, 2, 4, 5).to(device)

        expected_output = torch.tensor([[[[0.0000, 0.0000, 0.0000, 0.0197, 0.0429],
                                          [0.0000, 0.5632, 0.5322, 0.3677, 0.1430],
                                          [0.0000, 0.3083, 0.4032, 0.1761, 0.0000],
                                          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
                                         [[0.0000, 0.0000, 0.0000, 0.1189, 0.0586],
                                          [0.0000, 0.7087, 0.5420, 0.3995, 0.0863],
                                          [0.0000, 0.2695, 0.5981, 0.5888, 0.0000],
                                          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]],
                                       device=device, dtype=x_data.dtype)

        expected_transform = torch.tensor([[[1.0523, 0.3493, 0.3046],
                                            [-0.1066, 1.0426, 0.5846],
                                            [0.0351, 0.1213, 1.0000]]],
                                          device=device, dtype=x_data.dtype)

        out_perspective = kornia.augmentation.RandomPerspective(p=.99999999,  # step one the random state
                                                                return_transform=True)(x_data)

        assert isinstance(out_perspective, tuple)
        assert len(out_perspective) == 2
        assert out_perspective[0].shape == x_data.shape
        assert out_perspective[1].shape == (1, 3, 3)
        assert_allclose(out_perspective[0], expected_output, atol=1e-4, rtol=1e-4)
        assert_allclose(out_perspective[1], expected_transform, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        # TODO: turned off with p=0
        assert gradcheck(kornia.augmentation.RandomPerspective(p=0.), (input,), raise_exception=True)


class TestRandomAffine:

    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)
        out = kornia.augmentation.RandomAffine(0.)(x_data)
        assert out.shape == x_data.shape

    def test_smoke_no_transform_batch(self, device):
        x_data = torch.rand(2, 2, 8, 9).to(device)
        out = kornia.augmentation.RandomAffine(0.)(x_data)
        assert out.shape == x_data.shape

    @pytest.mark.parametrize("degrees", [45., (-45., 45.), torch.tensor([45., 45.])])
    @pytest.mark.parametrize("translate", [(0.1, 0.1), torch.tensor([0.1, 0.1])])
    @pytest.mark.parametrize("scale", [
        (0.8, 1.2), (0.8, 1.2, 0.9, 1.1), torch.tensor([0.8, 1.2]), torch.tensor([0.8, 1.2, 0.7, 1.3])])
    @pytest.mark.parametrize("shear", [
        5., (-5., 5.), (-5., 5., -3., 3.), torch.tensor(5.),
        torch.tensor([-5., 5.]), torch.tensor([-5., 5., -3., 3.])
    ])
    def test_batch_multi_params(self, degrees, translate, scale, shear, device, dtype):
        x_data = torch.rand(2, 2, 8, 9).to(device)
        out = kornia.augmentation.RandomAffine(
            degrees=degrees, translate=translate, scale=scale, shear=shear)(x_data)
        assert out.shape == x_data.shape

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 4, 5).to(device)
        out = kornia.augmentation.RandomAffine(0., return_transform=True)(x_data)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == x_data.shape
        assert out[1].shape == (1, 3, 3)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        # TODO: turned off with p=0
        assert gradcheck(kornia.augmentation.RandomAffine(10, p=0.), (input, ), raise_exception=True)
