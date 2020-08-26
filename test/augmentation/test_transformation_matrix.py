import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
import kornia.augmentation.functional as F
from kornia.constants import pi
from kornia.augmentation import ColorJitter


class TestHorizontalFlipFn:

    def test_random_hflip(self, device):
        flip_param_0 = {'batch_prob': torch.tensor(False)}
        flip_param_1 = {'batch_prob': torch.tensor(True)}

        input = torch.tensor([[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 1., 2.]])  # 3 x 4
        input.to(device)

        expected_transform = torch.tensor([[-1., 0., 3.],
                                           [0., 1., 0.],
                                           [0., 0., 1.]])  # 3 x 3

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3

        assert (F.compute_hflip_transformation(input, params=flip_param_0) == identity).all()
        assert (F.compute_hflip_transformation(input, params=flip_param_1) == expected_transform).all()

    def test_batch_random_hflip(self, device):
        batch_size = 5
        flip_param_0 = {'batch_prob': torch.tensor([False] * 5)}
        flip_param_1 = {'batch_prob': torch.tensor([True] * 5)}

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input.to(device)

        expected_transform = torch.tensor([[[-1., 0., 2.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3

        input = input.repeat(batch_size, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(batch_size, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(batch_size, 1, 1)  # 5 x 3 x 3

        assert (F.compute_hflip_transformation(input, params=flip_param_0) == identity).all()
        assert (F.compute_hflip_transformation(input, params=flip_param_1) == expected_transform).all()


class TestVerticalFlipFn:

    def test_random_vflip(self, device):

        flip_param_0 = {'batch_prob': torch.tensor(False)}
        flip_param_1 = {'batch_prob': torch.tensor(True)}

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3
        input.to(device)

        expected_transform = torch.tensor([[1., 0., 0.],
                                           [0., -1., 2.],
                                           [0., 0., 1.]])  # 3 x 3

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3

        assert (F.compute_vflip_transformation(input, params=flip_param_0) == identity).all()
        assert (F.compute_vflip_transformation(input, params=flip_param_1) == expected_transform).all()

    def test_batch_random_vflip(self, device):
        batch_size = 5
        flip_param_0 = {'batch_prob': torch.tensor([False] * 5)}
        flip_param_1 = {'batch_prob': torch.tensor([True] * 5)}

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input.to(device)

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 2.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3

        input = input.repeat(batch_size, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(batch_size, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(batch_size, 1, 1)  # 5 x 3 x 3

        assert (F.compute_vflip_transformation(input, params=flip_param_0) == identity).all()
        assert (F.compute_vflip_transformation(input, params=flip_param_1) == expected_transform).all()


class TestIntensityTransformation:

    def test_intensity_transformation(self):

        input = torch.rand(3, 5, 5)  # 3 x 5 x 5

        expected_transform = torch.eye(3).unsqueeze(0)  # 3 x 3

        assert_allclose(F.compute_intensity_transformation(input, {}), expected_transform, atol=1e-4, rtol=1e-5)

    def test_intensity_transformation_batch(self):
        batch_size = 2

        input = torch.rand(batch_size, 3, 5, 5)  # 2 x 3 x 5 x 5

        expected_transform = torch.eye(3).unsqueeze(0).expand((batch_size, 3, 3))  # 2 x 3 x 3

        assert_allclose(F.compute_intensity_transformation(input, {}), expected_transform, atol=1e-4, rtol=1e-5)


class TestPerspective:

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 3, 4).to(device)
        batch_prob = torch.rand(1) < 0.5
        start_points = torch.rand(1, 4, 2).to(device)
        end_points = torch.rand(1, 4, 2).to(device)

        params = dict(batch_prob=batch_prob, start_points=start_points, end_points=end_points)
        out_data = F.compute_perspective_transformation(x_data, params)

        assert out_data.shape == (1, 3, 3)
