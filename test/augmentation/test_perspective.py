import pytest

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from test.common import device


class TestPerspective:

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 3, 4).to(device)
        start_points = torch.rand(1, 4, 2).to(device)
        end_points = torch.rand(1, 4, 2).to(device)

        out_perspective = kornia.augmentation.functional.perspective(
            x_data, start_points, end_points, return_transform=False)

        assert out_perspective.shape == x_data.shape

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 3, 4).to(device)
        start_points = torch.rand(1, 4, 2).to(device)
        end_points = torch.rand(1, 4, 2).to(device)

        out_perspective = kornia.augmentation.functional.perspective(
            x_data, start_points, end_points, return_transform=True)

        assert isinstance(out_perspective, tuple)
        assert len(out_perspective) == 2
        assert out_perspective[0].shape == x_data.shape
        assert out_perspective[1].shape == (1, 3, 3)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var

        start_points = torch.rand(1, 4, 2).to(device)
        start_points = utils.tensor_to_gradcheck_var(start_points)  # to var

        end_points = torch.rand(1, 4, 2).to(device)
        end_points = utils.tensor_to_gradcheck_var(end_points)  # to var
        assert gradcheck(kornia.augmentation.functional.perspective,
                         (input, start_points, end_points), raise_exception=True)


class TestRandomPerspective:

    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)

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

    @pytest.mark.skip("Disabled since it crashes by now")
    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.random_perspective, (input, 0.), raise_exception=True)
