from typing import Union, Tuple

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.augmentation import (
    RandomMixUp,
    RandomCutMix
)


class TestRandomMixUp:

    def smoke_test(self, device):
        f = RandomMixUp()
        repr = "RandomMixUp(p=1.0, max_lambda=tensor(1.0), same_on_batch=False)"
        assert str(f) == repr

    def test_random_mixup_p1(self, device):
        torch.manual_seed(0)
        f = RandomMixUp()

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)
        lam = torch.tensor([0.1320, 0.3074]).to(device)

        expected = torch.stack([torch.ones(1, 3, 4) * (1 - lam[0]), torch.ones(1, 3, 4) * lam[1]]).to(device)

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, 0] == label).all()
        assert (out_label[:, 1] == torch.tensor([0, 1])).all()
        assert_allclose(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_p0(self, device):
        torch.manual_seed(0)
        f = RandomMixUp(p=0.)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)
        lam = torch.tensor([0., 0.]).to(device)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, 0] == label).all()
        assert (out_label[:, 1] == torch.tensor([0, 1])).all()
        assert_allclose(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_lam0(self, device):
        torch.manual_seed(0)
        f = RandomMixUp(max_lambda=0.)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)
        lam = torch.tensor([0., 0.]).to(device)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, 0] == label).all()
        assert (out_label[:, 1] == torch.tensor([0, 1])).all()
        assert_allclose(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_same_on_batch(self, device):
        torch.manual_seed(0)
        f = RandomMixUp(same_on_batch=True)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)
        lam = torch.tensor([0.0885, 0.0885]).to(device)

        expected = torch.stack([torch.ones(1, 3, 4) * (1 - lam[0]), torch.ones(1, 3, 4) * lam[1]]).to(device)

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, 0] == label).all()
        assert (out_label[:, 1] == torch.tensor([0, 1])).all()
        assert_allclose(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)


class TestRandomCutMix:

    def smoke_test(self, device):
        f = RandomCutMix(width=3, height=3)
        repr = "RandomCutMix(p=1, num_mix=1, beta=tensor(0.),, width=3, height=3, same_on_batch=False"
        assert str(f) == repr

    def test_random_mixup_p1(self, device):
        torch.manual_seed(76)
        f = RandomCutMix(width=4, height=3, p=1.)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)
        lam = torch.tensor([0.1320, 0.3074]).to(device)

        expected = torch.tensor([[[[0., 0., 0., 1.],
                                   [0., 0., 0., 1.],
                                   [1., 1., 1., 1.]]],
                                 [[[1., 1., 1., 0.],
                                   [1., 1., 1., 0.],
                                   [0., 0., 0., 0.]]]])

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[0, :, 0] == label).all()
        assert (out_label[0, :, 1] == torch.tensor([0, 1])).all()
        assert (out_label[0, :, 2] == torch.tensor([0.5, 0.5])).all()

    def test_random_mixup_p0(self, device):
        torch.manual_seed(76)
        f = RandomCutMix(p=0., width=4, height=3)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[0, :, 0] == label).all()
        assert (out_label[0, :, 1] == torch.tensor([0, 1])).all()
        assert (out_label[0, :, 2] == torch.tensor([0., 0.])).all()

    def test_random_mixup_beta0(self, device):
        torch.manual_seed(76)
        f = RandomCutMix(beta=0., width=4, height=3)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[0, :, 0] == label).all()
        assert (out_label[0, :, 1] == torch.tensor([0, 1])).all()
        assert (out_label[0, :, 2] == torch.tensor([0., 0.])).all()

    def test_random_mixup_num2(self, device):
        torch.manual_seed(76)
        f = RandomCutMix(width=4, height=3, num_mix=5)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)

        expected = torch.tensor([[[[0., 0., 0., 1.],
                                   [0., 0., 0., 1.],
                                   [1., 1., 1., 1.]]],
                                 [[[1., 1., 0., 0.],
                                   [1., 1., 0., 0.],
                                   [0., 0., 0., 0.]]]])

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, :, 0] == label).all()
        assert (out_label[:, :, 1] == torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])).all()
        assert_allclose(out_label[:, :, 2], torch.tensor([[0., 0.], [0., 0.], [0., 0.0833], [0., 0.], [0.5, 0.3333]]),
                        rtol=1e-4, atol=1e-4)

    def test_random_mixup_same_on_batch(self, device):
        torch.manual_seed(0)
        f = RandomCutMix(same_on_batch=True, width=4, height=3)

        input = torch.stack([torch.ones(1, 3, 4), torch.zeros(1, 3, 4)]).to(device)
        label = torch.tensor([1, 0]).to(device)
        lam = torch.tensor([0.0885, 0.0885]).to(device)

        expected = torch.tensor([[[[0., 0., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.]]],
                                 [[[1., 1., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.]]]])

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[0, :, 0] == label).all()
        assert (out_label[0, :, 1] == torch.tensor([0, 1])).all()
        assert_allclose(out_label[0, :, 2], torch.tensor([0.1667, 0.1667]), rtol=1e-4, atol=1e-4)
