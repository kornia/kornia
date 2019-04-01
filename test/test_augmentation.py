import pytest
import numpy as np

import torch
import torch.nn as nn
import torchgeometry.augmentation as taug
from torch.autograd import gradcheck 

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
