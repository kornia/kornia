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

class TestTranslation:
    def test_dxdy(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]])
        expected = torch.tensor([[
            [0., 1.],
            [0., 3.],
            [0., 5.],
            [0., 7.],
        ]])
        # prepare transformation
        translation_t = torch.tensor([[1., 0.]])
        transform = taug.Translate(translation_t)
        assert_allclose(transform(inp), expected)

    def test_dxdy_batch(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[[
            [0., 1.],
            [0., 3.],
            [0., 5.],
            [0., 7.],
        ]],[[
            [0., 0.],
            [0., 1.],
            [0., 3.],
            [0., 5.],
        ]]])
        # prepare transformation
        translation_t = torch.tensor([[1., 0.], [1., 1.]])
        transform = taug.Translate(translation_t)
        assert_allclose(transform(inp), expected)

    def test_compose_transforms(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]])

        # compose the transforms
        translation = torch.tensor([[1., 1.]])
        random_translation = taug.RandomTranslationMatrix(
            torch.tensor([-1, 1]))
        compose_matrix = nn.Sequential(
            taug.TranslationMatrix(translation),
            random_translation,
        )
        matrix = compose_matrix(taug.identity_matrix())

        # rotation with obtained random angle
        translation_composed = translation + random_translation.translation
        transforms = nn.Sequential(
            taug.Translate(translation_composed),
        )

        # apply transforms
        out_affine = taug.affine(inp, matrix[..., :2, :3])
        assert_allclose(out_affine, transforms(inp))


class TestRotation:
    def test_smoke(self):
        angle = 0.0
        angle_t = torch.tensor([angle])
        assert str(taug.Rotate(angle=angle_t)) == 'Rotate(angle=0.0, center=None)'

    def test_angle90(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]])
        expected = torch.tensor([[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]])
        # prepare transformation
        angle_t = torch.tensor([90.])
        transform = taug.Rotate(angle_t)
        assert_allclose(transform(inp), expected)
        
    def test_angle90_batch2(self):
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[[
            [0., 0.],
            [4., 6.],
            [3., 5.],
            [0., 0.],
        ]],[[
            [0., 0.],
            [5., 3.],
            [6., 4.],
            [0., 0.],
        ]]])
        # prepare transformation
        angle_t = torch.tensor([90., -90.])
        transform = taug.Rotate(angle_t)
        assert_allclose(transform(inp), expected)

    def test_compose_transforms(self):
        h, w = 4, 2  # height, width
        center = torch.tensor([[(w - 1) / 2, (h - 1) / 2]])

        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]])

        # compose the transforms
        angle = torch.tensor([90.])
        random_rotation = taug.RandomRotationMatrix(angle, center)
        compose_matrix = nn.Sequential(
            taug.RotationMatrix(angle, center),
            random_rotation,
        )
        matrix = compose_matrix(taug.identity_matrix())

        # rotation with obtained random angle
        angle_composed = angle + random_rotation.angle
        transforms = nn.Sequential(
            taug.Rotate(angle_composed, center),
        )

        # apply transforms
        out_affine = taug.affine(inp, matrix[..., :2, :3])
        out_rotate = taug.rotate(inp, angle_composed, center)
        assert_allclose(out_affine, out_rotate)
        assert_allclose(out_affine, transforms(inp))


class TestComposeTransforms:
    def test_rotation_translation(self):
        h, w = 4, 2  # height, width
        center = torch.tensor([[(w - 1) / 2, (h - 1) / 2]])
        # prepare input data
        inp = torch.tensor([[
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ]])

        # compose the transforms
        angle = torch.tensor([90.])
        translation = torch.tensor([[1., 1.]])

        compose_matrix = nn.Sequential(
            taug.RotationMatrix(angle, center),
            taug.TranslationMatrix(translation),
            taug.TranslationMatrix(-translation),
            taug.RotationMatrix(-angle, center),
        )
        matrix = compose_matrix(taug.identity_matrix())

        # apply transforms
        out_affine = taug.affine(inp, matrix[..., :2, :3])
        assert_allclose(out_affine, inp)
