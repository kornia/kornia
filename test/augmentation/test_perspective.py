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
            end_points=end_points, interpolation=torch.tensor(1),
            align_corners=torch.tensor(False))
        out_data = F.apply_perspective(x_data, params)

        assert out_data.shape == x_data.shape

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var

        batch_prob = torch.rand(1) < 0.5

        start_points = torch.rand(1, 4, 2).to(device)
        start_points = utils.tensor_to_gradcheck_var(start_points)  # to var

        end_points = torch.rand(1, 4, 2).to(device)
        end_points = utils.tensor_to_gradcheck_var(end_points)  # to var

        params = dict(
            batch_prob=batch_prob,
            start_points=start_points,
            end_points=end_points,
            interpolation=torch.tensor(1),
            align_corners=torch.tensor(False)
        )
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

        expected_output = torch.tensor([[[[0.0000000000, 0.0000000000, 0.0000000000, 0.0197417457, 0.0429493971],
                                          [0.0000000000, 0.5632190704, 0.5321710110, 0.3676981330, 0.1430126727],
                                          [0.0000000000, 0.3082636893, 0.4031507671, 0.1760708243, 0.0000000000],
                                          [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]],

                                         [[0.0000000000, 0.0000000000, 0.0000000000, 0.1189093292, 0.0585946590],
                                          [0.0000000000, 0.7087295055, 0.5419756770, 0.3995491862, 0.0863459259],
                                          [0.0000000000, 0.2694899142, 0.5981453061, 0.5887590051, 0.0000000000],
                                          [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]]]],
                                       device=device, dtype=x_data.dtype)

        expected_transform = torch.tensor([[[1.0522739887, 0.3492536247, 0.3045728207],
                                            [-0.1066057906, 1.0426188707, 0.5845923424],
                                            [0.0350575559, 0.1213315651, 1.0000000000]]],
                                          device=device, dtype=x_data.dtype)

        out_perspective = kornia.augmentation.RandomPerspective(p=1.0,
                                                                return_transform=True)(x_data)

        assert isinstance(out_perspective, tuple)
        assert len(out_perspective) == 2
        assert out_perspective[0].shape == x_data.shape
        assert out_perspective[1].shape == (1, 3, 3)
        assert_allclose(out_perspective[0], expected_output)
        assert_allclose(out_perspective[1], expected_transform)

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

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.RandomAffine(0.), (input, ), raise_exception=True)
