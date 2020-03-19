"""
Temporary devloping file to test kornia augmentation functions aginst torchvision.

Note:
1. In torchvision, the conversion from (0, 1) to uint8 then to (0, 1) has info losses.
"""
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
import kornia.augmentation.functional as F
from kornia.augmentation import ColorJitter

from test.common import device

from torchvision.transforms import transforms
import torchvision.transforms.functional as tvF
import PIL


to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


class TestHorizontalFlip:
    def test_hflip(self, device):
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        flip_param_1 = {'batch_prob': torch.tensor(True)}
        out_tensor = F._apply_hflip(in_tensor, flip_param_1)
        out_pil = tvF.hflip(in_pil)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-2, rtol=1e-3)


class TestVerticalFlip:
    def test_vflip(self, device):
        in_tensor = torch.rand((3, 4, 5))
        in_pil = to_pil(in_tensor)
        flip_param_1 = {'batch_prob': torch.tensor(True)}
        out_tensor = F._apply_vflip(in_tensor, flip_param_1)
        out_pil = tvF.vflip(in_pil)
        assert_allclose(out_tensor, to_tensor(out_pil), atol=1e-2, rtol=1e-3)