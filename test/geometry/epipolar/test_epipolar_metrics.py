import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.geometry.epipolar as epi
import kornia.testing as utils


class TestSymmetricalEpipolarDistance:

    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = utils.create_random_fundamental_matrix(batch_size).type_as(points2)
        assert gradcheck(epi.symmetrical_epipolar_distance, (points1, points2, Fm),
                         raise_exception=True)


class TestSampsonEpipolarDistance:

    def test_smoke(self, device, dtype):
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.sampson_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.sampson_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = utils.create_random_fundamental_matrix(batch_size).type_as(points2)
        assert gradcheck(epi.sampson_epipolar_distance, (points1, points2, Fm),
                         raise_exception=True)
