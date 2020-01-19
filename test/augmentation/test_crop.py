import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from test.common import device


class TestCenterCrop:

    def test_no_transform(self, device):
        inp = torch.rand(1, 2, 4, 4).to(device)
        out = kornia.augmentation.CenterCrop(2)(inp)
        assert out.shape == (1, 2, 2, 2)

    def test_transform(self, device):
        inp = torch.rand(1, 2, 5, 4).to(device)
        out = kornia.augmentation.CenterCrop(2, return_transform=True)(inp)
        assert len(out) == 2
        assert out[0].shape == (1, 2, 2, 2)
        assert out[1].shape == (1, 3, 3)

    def test_no_transform_tuple(self, device):
        inp = torch.rand(1, 2, 5, 4).to(device)
        out = kornia.augmentation.CenterCrop((3, 4))(inp)
        assert out.shape == (1, 2, 3, 4)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.CenterCrop(3), (input,), raise_exception=True)
