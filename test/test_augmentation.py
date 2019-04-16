import pytest
import numpy as np

import torch
import torch.nn as nn
import torchgeometry.augmentation as taug
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import torchvision

import utils
from common import device_type


class TestAugmentationSequential:
    def test_smoke(self):
        height, width, channels = 4, 5, 3
        img = np.ones((height, width, channels))

        transforms = nn.Sequential(
            taug.ToTensor(),
            taug.Grayscale(),
        )
        assert transforms(img).shape == (1, height, width)

    def test_smoke_batch(self):
        batch_size, height, width, channels = 2, 4, 5, 3
        img = np.ones((batch_size, height, width, channels))

        transforms = nn.Sequential(
            taug.ToTensor(),
            taug.Grayscale(),
        )
        assert transforms(img).shape == (batch_size, 1, height, width)


class TestAugmentationCompose:
    def test_smoke(self):
        height, width, channels = 4, 5, 3
        img = np.ones((height, width, channels))

        transforms = torchvision.transforms.Compose([
            taug.ToTensor(),
            taug.Grayscale(),
        ])
        assert transforms(img).shape == (1, height, width)

    def test_smoke_batch(self):
        batch_size, height, width, channels = 2, 4, 5, 3
        img = np.ones((batch_size, height, width, channels))

        transforms = torchvision.transforms.Compose([
            taug.ToTensor(),
            taug.Grayscale(),
        ])
        assert transforms(img).shape == (batch_size, 1, height, width)


class TestToTensor:
    def test_smoke(self):
        assert str(taug.ToTensor()) == 'ToTensor()'

    def test_rgb_to_tensor(self):
        height, width, channels = 4, 5, 3
        img = np.ones((height, width, channels))
        assert taug.ToTensor()(img).shape == (channels, height, width)

    def test_rgb_to_tensor_batch(self):
        batch_size, height, width, channels = 2, 4, 5, 3
        img = np.ones((batch_size, height, width, channels))
        assert taug.ToTensor()(img).shape == \
            (batch_size, channels, height, width)

    def test_mono_to_tensor(self):
        height, width = 4, 5
        img = np.ones((height, width))
        assert taug.ToTensor()(img).shape == (1, height, width)

    def test_gray_to_tensor(self):
        height, width, channels = 4, 5, 1
        img = np.ones((height, width, channels))
        assert taug.ToTensor()(img).shape == (1, height, width)


class TestGrayscale:
    def test_smoke(self):
        assert str(taug.Grayscale()) == 'Grayscale()'

    def test_rgb_to_grayscale(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert taug.Grayscale()(img).shape == (1, height, width)

    def test_rgb_to_grayscale_batch(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert taug.Grayscale()(img).shape == (batch_size, 1, height, width)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(taug.Grayscale(), (img,), raise_exception=True)


class TestNormalize:
    def test_smoke(self):
        mean = [0.5]
        std = [0.1]
        repr = 'Normalize(mean=[0.5], std=[0.1])'
        assert str(taug.Normalize(mean, std)) == repr

    def test_normalize(self):

        # prepare input data
        data = torch.ones(1, 2, 2)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([0.25]).repeat(1, 2, 2).view_as(data)

        f = taug.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_batch_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0])
        std = torch.tensor([2., 2., 2.])

        # expected output
        expected = torch.tensor([1.25, 1., 0.5]).repeat(2, 1, 1).view_as(data)

        f = taug.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1).double()
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0]).double()
        std = torch.tensor([2., 2., 2.]).double()

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(taug.Normalize(mean, std), (data,),
                         raise_exception=True)


class TestRotate:
    def test_smoke(self):
        angle = 0.0
        angle_t = torch.Tensor([angle])
        repr = 'Rotate(angle=0.0, center=None)'
        assert str(taug.Rotate(angle=angle_t)) == repr

    def test_angle90(self):
        # prepare input data
        inp = torch.FloatTensor([[
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]])
        expected = torch.FloatTensor([[
            [0, 0],
            [4, 6],
            [3, 5],
            [0, 0],
        ]])
        # prepare transformation
        angle_t = torch.Tensor([90])
        transform = taug.Rotate(angle_t)
        assert_allclose(transform(inp), expected)

    def test_angle90_batch2(self):
        # prepare input data
        inp = torch.FloatTensor([[
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.FloatTensor([[[
            [0, 0],
            [4, 6],
            [3, 5],
            [0, 0],
        ]], [[
            [0, 0],
            [5, 3],
            [6, 4],
            [0, 0],
        ]]])
        # prepare transformation
        angle_t = torch.Tensor([90, -90])
        transform = taug.Rotate(angle_t)
        assert_allclose(transform(inp), expected)

    def test_random_rotate_value(self):
        transform = taug.RandomRotation(degrees=90)
        inp = torch.rand(1, 3, 4)
        assert transform(inp).shape == (1, 3, 4)

    def test_random_rotate_minmax_value(self):
        transform = taug.RandomRotation(degrees=(-45, 0))
        inp = torch.rand(1, 3, 4)
        assert transform(inp).shape == (1, 3, 4)

    def test_random_rotate_minmax_value_batch(self):
        transform = taug.RandomRotation(degrees=(-45, 0))
        inp = torch.rand(2, 3, 3, 4)
        assert transform(inp).shape == (2, 3, 3, 4)

    def test_compose_transforms(self):
        c, h, w = 3, 4, 2  # channels, height, width
        inp = torch.rand(c, h, w)
        angle = torch.Tensor([90])
        center = torch.Tensor([[(w - 1) / 2, (h - 1) / 2]])

        # compose the transforms
        compose_matrix = nn.Sequential(
            taug.RotationMatrix(angle, center),
        )
        matrix = compose_matrix(taug.identity_matrix())

        transforms = nn.Sequential(
            taug.Rotate(angle, center),
        )

        # apply transforms
        out_affine = taug.affine(inp, matrix[..., :2, :3])
        out_rotate = taug.rotate(inp, angle, center)
        assert_allclose(out_affine, out_rotate)
        assert_allclose(out_affine, transforms(inp))
