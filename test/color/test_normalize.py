import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from common import device_type

import kornia.color as color
import utils


class TestNormalize:
    def test_smoke(self):
        mean = [0.5]
        std = [0.1]
        repr = 'Normalize(mean=[0.5], std=[0.1])'
        assert str(color.Normalize(mean, std)) == repr

    def test_normalize(self):

        # prepare input data
        data = torch.ones(1, 2, 2)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([0.25]).repeat(1, 2, 2).view_as(data)

        f = color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_broadcast_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0])
        std = torch.tensor([2.0, 2.0, 2.0])

        # expected output
        expected = torch.tensor([1.25, 1, 0.5]).repeat(2, 1, 1).view_as(data)

        f = color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_batch_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

        # expected output
        expected = torch.tensor([1.25, 1, 0.5]).repeat(2, 1, 1).view_as(data)

        f = color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor, mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:

            return color.normalize(data, mean, std)

            data = torch.ones(2, 3, 1, 1)
            data += 2

            mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
            std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

            actual = op_script(data, mean, std)
            expected = image.normalize(data, mean, std)
            assert_allclose(actual, expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0]).double()
        std = torch.tensor([2., 2., 2.]).double()

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(color.Normalize(mean, std), (data,),
                         raise_exception=True)
