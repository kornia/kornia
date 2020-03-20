"""
Temporary devloping file to test kornia augmentation functions aginst torchvision.

Note:
1. In torchvision, the conversion from (0, 1) to uint8 then to (0, 1) has info losses.
"""
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import math

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
import kornia.augmentation.functional as F
import kornia.augmentation.param_gen as pg
# TODO: Should we have a class to store those constants?
from kornia.geometry import pi

from test.common import device

from torchvision.transforms import transforms
import torchvision.transforms.functional as tvF
import PIL


to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def tensor_pre_transform_wrapper(input: torch.Tensor):
    """ A wrapper that tried to reproduce the actual output from:
        - transforms.ToPILImage()
        - transforms.ToTensor()
        For each image, simply (img * 255).int() // 255
    """
    return (input * 255).int().float() / 255


class TestHorizontalFlip:
    def test_hflip(self, device):
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        flip_param_1 = {'batch_prob': torch.tensor(True)}
        out_tensor = F._apply_hflip(tensor_pre_transform_wrapper(in_tensor), flip_param_1)
        out_pil = tvF.hflip(in_pil)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-4, rtol=1e-5)


class TestVerticalFlip:
    def test_vflip(self, device):
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        flip_param_1 = {'batch_prob': torch.tensor(True)}
        out_tensor = F._apply_vflip(tensor_pre_transform_wrapper(in_tensor), flip_param_1)
        out_pil = tvF.vflip(in_pil)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-4, rtol=1e-5)


class TestColorJitter:
    def test_contrast_adjustment(self, device):
        # PIL implementation
        # https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageEnhance.py#L57
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        factor = 0.375
        out_tensor = F.adjust_contrast(tensor_pre_transform_wrapper(in_tensor), torch.tensor(factor))
        out_pil = tvF.adjust_contrast(in_pil, factor)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-4, rtol=1e-5)

    def test_brightness_adjustment(self, device):
        # PIL implementation
        # https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageEnhance.py#L74
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        factor = 0.375
        out_tensor = F.adjust_brightness(tensor_pre_transform_wrapper(in_tensor), torch.tensor(factor))
        out_pil = tvF.adjust_brightness(in_pil, factor)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-4, rtol=1e-5)

    def test_saturation_adjustment(self, device):
        # PIL implementation
        # https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageEnhance.py#L39
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        factor = 0.375
        out_tensor = F.adjust_saturation(tensor_pre_transform_wrapper(in_tensor), torch.tensor(factor))
        out_pil = tvF.adjust_saturation(in_pil, factor)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-4, rtol=1e-5)

    def test_hue_adjustment(self, device):
        # Pytorch implementation
        # https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#adjust_hue
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        factor = 0.375
        # TODO: HUE factor range:
        # In Kornia, we used (-pi, pi), while PIL used (-0.5, 0.5)
        # the range need to be re-normalized from (-0.5, 0.5) to (-pi, pi)
        # TODO: align implementation with pytorch
        # Kornia: h_out: torch.Tensor = torch.fmod(h + hue_factor, divisor)
        # Torchvision: np_h += np.uint8(hue_factor * 255)
        out_tensor = F.adjust_hue(tensor_pre_transform_wrapper(in_tensor), torch.tensor(factor / 0.5) * pi)
        out_pil = tvF.adjust_hue(in_pil, factor)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-2, rtol=1e-5)


class TestAffineTransformation:
    def test_rotate(self, device):
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        degrees = 10
        out_tensor = F._apply_rotation(
            tensor_pre_transform_wrapper(in_tensor), {'degrees': torch.tensor(-degrees)}, False)
        out_pil = tvF.rotate(in_pil, angle=degrees)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-4, rtol=1e-5)

    def test_affine_rotate(self, device):
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        # TODO: Enable non-tuple degree params in _get_random_affine_params
        params = pg._get_random_affine_params(
            batch_size=1, height=4, width=5, degrees=[10.0, 10.0], translate=None, scales=None, shears=None)
        out_tensor = F._apply_affine(tensor_pre_transform_wrapper(in_tensor.unsqueeze(dim=0)), {'transform': params}, False)
        out_pil = tvF.affine(in_pil, angle=10.0, translate=[0, 0], scale=0.0, shear=0)
        assert_allclose(out_tensor.squeeze(), to_tensor(out_pil), atol=1e-4, rtol=1e-5)
