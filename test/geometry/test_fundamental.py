import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import TEST_DEVICES

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestSymmetricalEpipolarDistance:

    def test_smoke(self):
        pts1 = torch.rand(1, 4, 3)
        pts2 = torch.rand(1, 4, 3)
        Fm = utils.create_random_fundamental_matrix(1)
        assert kornia.geometry.fundamental.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self):
        batch_size = 5
        pts1 = torch.rand(batch_size, 1, 3)
        pts2 = torch.rand(batch_size, 1, 3)
        Fm = utils.create_random_fundamental_matrix(batch_size)
        assert kornia.geometry.fundamental.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (5, 1)

    def test_gradcheck(self):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points_1 = torch.rand(batch_size, num_points, num_dims)
        points_2 = torch.rand(batch_size, num_points, num_dims)
        Fm = utils.create_random_fundamental_matrix(batch_size)
        # evaluate function gradient
        points_1 = utils.tensor_to_gradcheck_var(points_1)  # to var
        points_2 = utils.tensor_to_gradcheck_var(points_2)  # to var
        Fm = utils.tensor_to_gradcheck_var(Fm)  # to var
        assert gradcheck(kornia.geometry.fundamental.symmetrical_epipolar_distance, (points_1, points_2, Fm),
                         raise_exception=True)


class TestSampsonEpipolarDistance:

    def test_smoke(self):
        pts1 = torch.rand(1, 4, 3)
        pts2 = torch.rand(1, 4, 3)
        Fm = utils.create_random_fundamental_matrix(1)
        assert kornia.geometry.fundamental.sampson_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self):
        batch_size = 5
        pts1 = torch.rand(batch_size, 1, 3)
        pts2 = torch.rand(batch_size, 1, 3)
        Fm = utils.create_random_fundamental_matrix(batch_size)
        assert kornia.geometry.fundamental.sampson_epipolar_distance(pts1, pts2, Fm).shape == (5, 1)

    def test_gradcheck(self):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points_1 = torch.rand(batch_size, num_points, num_dims)
        points_2 = torch.rand(batch_size, num_points, num_dims)
        Fm = utils.create_random_fundamental_matrix(batch_size)
        # evaluate function gradient
        points_1 = utils.tensor_to_gradcheck_var(points_1)  # to var
        points_2 = utils.tensor_to_gradcheck_var(points_2)  # to var
        Fm = utils.tensor_to_gradcheck_var(Fm)  # to var
        assert gradcheck(kornia.geometry.fundamental.sampson_epipolar_distance, (points_1, points_2, Fm),
                         raise_exception=True)
