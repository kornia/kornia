import pytest
import numpy as np

import torch
import torch.nn as nn
import torchgeometry.augmentation as taug
from torch.autograd import gradcheck 

import torchvision

import utils
from common import device_type


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

    def test_rgb_to_tensor_sequential(self):
        height, width, channels = 4, 5, 3 
        img = np.ones((height, width, channels))

        transforms = nn.Sequential(
            taug.ToTensor(),
        )
        assert transforms(img).shape == (channels, height, width)

    def test_rgb_to_tensor_compose(self):
        height, width, channels = 4, 5, 3 
        img = np.ones((height, width, channels))

        transforms = torchvision.transforms.Compose([
            taug.ToTensor(),
        ])
        assert transforms(img).shape == (channels, height, width)
