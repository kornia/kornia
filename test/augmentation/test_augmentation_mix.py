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

    def test_smoke(self, device, dtype):
        f = RandomMixUp()
        repr = "RandomMixUp(lambda_val=tensor([0., 1.]), p=1.0, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr

    def test_random_mixup_p1(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUp(p=1.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0.1320, 0.3074], device=device, dtype=dtype)

        expected = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype) * (1 - lam[0]),
            torch.ones(1, 3, 4, device=device, dtype=dtype) * lam[1]]
        )

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, 0] == label).all()
        assert (out_label[:, 1] == torch.tensor([0, 1], device=device, dtype=dtype)).all()
        assert_allclose(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_p0(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUp(p=0.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0., 0.], device=device, dtype=dtype)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label == label).all()

    def test_random_mixup_lam0(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUp(lambda_val=(0., 0.), p=1.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0., 0.], device=device, dtype=dtype)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, 0] == label).all()
        assert (out_label[:, 1] == torch.tensor([0, 1], device=device)).all()
        assert_allclose(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_same_on_batch(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUp(same_on_batch=True, p=1.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0.0885, 0.0885], device=device, dtype=dtype)

        expected = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype) * (1 - lam[0]),
            torch.ones(1, 3, 4, device=device, dtype=dtype) * lam[1]
        ])

        out_image, out_label = f(input, label)
        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, 0] == label).all()
        assert (out_label[:, 1] == torch.tensor([0, 1], device=device, dtype=dtype)).all()
        assert_allclose(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)


class TestRandomCutMix:

    def test_smoke(self, device, dtype):
        f = RandomCutMix(width=3, height=3)
        repr = "RandomCutMix(num_mix=1, beta=1.0, cut_size=tensor([0., 1.]), , p=1.0, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr

    def test_random_mixup_p1(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMix(width=4, height=3, p=1.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0.1320, 0.3074], device=device, dtype=dtype)

        expected = torch.tensor([[[[0., 0., 0., 1.],
                                   [0., 0., 0., 1.],
                                   [1., 1., 1., 1.]]],
                                 [[[1., 1., 1., 0.],
                                   [1., 1., 1., 0.],
                                   [0., 0., 0., 0.]]]], device=device, dtype=dtype)

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[0, :, 0] == label).all()
        assert (out_label[0, :, 1] == torch.tensor([0, 1], device=device, dtype=dtype)).all()
        assert (out_label[0, :, 2] == torch.tensor([0.5, 0.5], device=device, dtype=dtype)).all()

    def test_random_mixup_p0(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMix(p=0., width=4, height=3)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label == label).all()

    def test_random_mixup_beta0(self, device, dtype):
        torch.manual_seed(76)
        # beta 0 => resample 0.5 area
        f = RandomCutMix(beta=0., width=4, height=3, p=1.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)

        expected = torch.tensor([[[[0., 0., 1., 1.],
                                   [0., 0., 1., 1.],
                                   [1., 1., 1., 1.]]],
                                 [[[1., 1., 0., 0.],
                                   [1., 1., 0., 0.],
                                   [0., 0., 0., 0.]]]], device=device, dtype=dtype)

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[0, :, 0] == label).all()
        assert (out_label[0, :, 1] == torch.tensor([0, 1], device=device, dtype=dtype)).all()
        # cut area = 4 / 12
        assert_allclose(out_label[0, :, 2], torch.tensor([0.33333, 0.33333], device=device, dtype=dtype))

    def test_random_mixup_num2(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMix(width=4, height=3, num_mix=5, p=1.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)

        expected = torch.tensor([[[[0., 0., 0., 1.],
                                   [0., 0., 0., 1.],
                                   [1., 1., 1., 1.]]],
                                 [[[1., 1., 0., 0.],
                                   [1., 1., 0., 0.],
                                   [0., 0., 0., 0.]]]], device=device, dtype=dtype)

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[:, :, 0] == label).all()
        assert (out_label[:, :, 1] == torch.tensor(
            [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], device=device)).all()
        assert_allclose(out_label[:, :, 2], torch.tensor(
            [[0.0833, 0.3333], [0., 0.1667], [0.5, 0.0833], [0.0833, 0.], [0.5, 0.3333]],
            device=device, dtype=dtype), rtol=1e-4, atol=1e-4)

    def test_random_mixup_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        f = RandomCutMix(same_on_batch=True, width=4, height=3, p=1.)

        input = torch.stack([
            torch.ones(1, 3, 4, device=device, dtype=dtype),
            torch.zeros(1, 3, 4, device=device, dtype=dtype)
        ])
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0.0885, 0.0885], device=device, dtype=dtype)

        expected = torch.tensor([[[[0., 0., 0., 1.],
                                   [0., 0., 0., 1.],
                                   [1., 1., 1., 1.]]],
                                 [[[1., 1., 1., 0.],
                                   [1., 1., 1., 0.],
                                   [0., 0., 0., 0.]]]], device=device, dtype=dtype)

        out_image, out_label = f(input, label)

        assert_allclose(out_image, expected, rtol=1e-4, atol=1e-4)
        assert (out_label[0, :, 0] == label).all()
        assert (out_label[0, :, 1] == torch.tensor([0, 1], device=device, dtype=dtype)).all()
        assert_allclose(out_label[0, :, 2],
                        torch.tensor([0.5000, 0.5000], device=device, dtype=dtype), rtol=1e-4, atol=1e-4)
