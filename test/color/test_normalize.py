import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestNormalize:
    def test_smoke(self):
        mean = [0.5]
        std = [0.1]
        repr = "Normalize(mean=[0.5], std=[0.1])"
        assert str(kornia.color.Normalize(mean, std)) == repr

    def test_normalize(self, device):

        # prepare input data
        data = torch.ones(1, 2, 2).to(device)
        mean = torch.tensor([0.5]).to(device)
        std = torch.tensor([2.0]).to(device)

        # expected output
        expected = torch.tensor([0.25]).repeat(1, 2, 2).view_as(data).to(device)

        f = kornia.color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_broadcast_normalize(self, device):

        # prepare input data
        data = torch.ones(2, 3, 1, 1).to(device)
        data += 2

        mean = torch.tensor([2.0]).to(device)
        std = torch.tensor([0.5]).to(device)

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_float_input(self, device):

        data = torch.ones(2, 3, 1, 1).to(device)
        data += 2

        mean = 2.0
        std = 0.5

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_batch_normalize(self, device):

        # prepare input data
        data = torch.ones(2, 3, 1, 1).to(device)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0]).to(device).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0]).to(device).repeat(2, 1)

        # expected output
        expected = torch.tensor([1.25, 1, 0.5]).to(device).repeat(2, 1, 1).view_as(data)

        f = kornia.color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            return kornia.normalize(data, mean, std)

            data = torch.ones(2, 3, 1, 1).to(device)
            data += 2

            mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1).to(device)
            std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1).to(device)

            actual = op_script(data, mean, std)
            expected = kornia.normalize(data, mean, std)
            assert_allclose(actual, expected)

    def test_gradcheck(self, device):

        # prepare input data
        data = torch.ones(2, 3, 1, 1).to(device)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0])
        std = torch.tensor([2.0, 2.0, 2.0])

        data = utils.tensor_to_gradcheck_var(data)  # to var
        mean = utils.tensor_to_gradcheck_var(mean)  # to var
        std = utils.tensor_to_gradcheck_var(std)  # to var

        assert gradcheck(kornia.color.Normalize(mean, std), (data,), raise_exception=True)

    def test_single_value(self, device):
        # prepare input data
        mean = torch.tensor(2).to(device)
        std = torch.tensor(3).to(device)
        data = torch.ones(2, 3, 256, 313).to(device)

        # expected output
        expected = (data - mean) / std

        assert_allclose(kornia.normalize(data, mean, std), expected)


class TestDenormalize:
    def test_smoke(self):
        mean = [0.5]
        std = [0.1]
        repr = "Denormalize(mean=[0.5], std=[0.1])"
        assert str(kornia.color.Denormalize(mean, std)) == repr

    def test_denormalize(self):

        # prepare input data
        data = torch.ones(1, 2, 2)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([2.5]).repeat(1, 2, 2).view_as(data)

        f = kornia.color.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    def test_broadcast_denormalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([2.0])
        std = torch.tensor([0.5])

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.color.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    def test_float_input(self):

        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = 2.0
        std = 0.5

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.color.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    def test_batch_denormalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

        # expected output
        expected = torch.tensor([6.5, 7, 8]).repeat(2, 1, 1).view_as(data)

        f = kornia.color.Denormalize(mean, std)
        assert_allclose(f(data), expected)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            return kornia.denormalize(data, mean, std)

            data = torch.ones(2, 3, 1, 1)
            data += 2

            mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
            std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

            actual = op_script(data, mean, std)
            expected = kornia.denormalize(data, mean, std)
            assert_allclose(actual, expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0]).double()
        std = torch.tensor([2.0, 2.0, 2.0]).double()

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.Denormalize(mean, std), (data,), raise_exception=True)

    def test_single_value(self):

        # prepare input data
        mean = torch.tensor(2)
        std = torch.tensor(3)
        data = torch.ones(2, 3, 256, 313).float()

        # expected output
        expected = (data * std) + mean

        assert_allclose(kornia.denormalize(data, mean, std), expected)
