from typing import Union, Tuple

import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.constants import pi
from kornia.augmentation import (
    RandomDepthicalFlip3D,
    RandomHorizontalFlip3D,
    RandomVerticalFlip3D,
    RandomAffine3D,
    RandomRotation3D,
    RandomCrop3D,
    CenterCrop3D,
    RandomEqualize3D
)


class TestRandomHorizontalFlip3D:

    def test_smoke(self):
        f = RandomHorizontalFlip3D(0.5)
        repr = "RandomHorizontalFlip3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=0.5)"
        assert str(f) == repr

    def test_random_hflip(self, device):

        f = RandomHorizontalFlip3D(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip3D(p=0., return_transform=True)
        f2 = RandomHorizontalFlip3D(p=1.)
        f3 = RandomHorizontalFlip3D(p=0.)

        input = torch.tensor([[[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 1., 2.]],
                              [[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 1., 2.]]])  # 2 x 3 x 4

        input = input.to(device)

        expected = torch.tensor([[[0., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [2., 1., 0., 0.]],
                                 [[0., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [2., 1., 0., 0.]]])  # 2 x 3 x 4

        expected = expected.to(device)

        expected_transform = torch.tensor([[-1., 0., 0., 3.],
                                           [0., 1., 0., 0.],
                                           [0., 0., 1., 0.],
                                           [0., 0., 0., 1.]])  # 4 x 4

        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])  # 4 x 4
        identity = identity.to(device)

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()
        assert (f2(input) == expected).all()
        assert (f3(input) == input).all()

    def test_batch_random_hflip(self, device):

        f = RandomHorizontalFlip3D(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip3D(p=0.0, return_transform=True)

        input = torch.tensor([[[[[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 1., 1.]]]]])  # 1 x 1 x 1 x 3 x 3
        input = input.to(device)

        expected = torch.tensor([[[[[0., 0., 0.],
                                    [0., 0., 0.],
                                    [1., 1., 0.]]]]])  # 1 x 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[[-1., 0., 0., 2.],
                                            [0., 1., 0., 0.],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]]])  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]]])  # 1 x 4 x 4
        identity = identity.to(device)

        input = input.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 4 x 4
        identity = identity.repeat(5, 1, 1)  # 5 x 4 x 4

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()

    def test_same_on_batch(self, device):
        f = RandomHorizontalFlip3D(p=0.5, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device):

        f = nn.Sequential(
            RandomHorizontalFlip3D(p=1.0, return_transform=True),
            RandomHorizontalFlip3D(p=1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomHorizontalFlip3D(p=1.0, return_transform=True),
            RandomHorizontalFlip3D(p=1.0),
        )

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected_transform = torch.tensor([[[-1., 0., 0., 2.],
                                            [0., 1., 0., 0.],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]]])  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform
        expected_transform_1 = expected_transform_1.to(device)

        assert(f(input)[0] == input).all()
        assert(f(input)[1] == expected_transform_1).all()
        assert(f1(input)[0] == input).all()
        assert(f1(input)[1] == expected_transform).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

            return kornia.random_hflip(data)

        input = torch.tensor([[[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 1., 1.]]])  # 1 x 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 5., 5.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self, device):
        input = torch.rand((1, 3, 3)).to(device)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomHorizontalFlip3D(p=1.), (input, ), raise_exception=True)


class TestRandomVerticalFlip3D:

    def test_smoke(self):
        f = RandomVerticalFlip3D(0.5)
        repr = "RandomVerticalFlip3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=0.5)"
        assert str(f) == repr

    def test_random_vflip(self, device):

        f = RandomVerticalFlip3D(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip3D(p=0., return_transform=True)
        f2 = RandomVerticalFlip3D(p=1.)
        f3 = RandomVerticalFlip3D(p=0.)

        input = torch.tensor([[[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 1., 1.]],
                              [[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 1., 1.]]])  # 2 x 3 x 3
        input = input.to(device)

        expected = torch.tensor([[[0., 1., 1.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]],
                                 [[0., 1., 1.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]])  # 2 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[1., 0., 0., 0.],
                                           [0., -1., 0., 2.],
                                           [0., 0., 1., 0.],
                                           [0., 0., 0., 1.]])  # 4 x 4
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])  # 4 x 4
        identity = identity.to(device)

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)
        assert_allclose(f2(input), expected)
        assert_allclose(f3(input), input)

    def test_batch_random_vflip(self, device):

        f = RandomVerticalFlip3D(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip3D(p=0.0, return_transform=True)

        input = torch.tensor([[[[[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 1., 1.]]]]])  # 1 x 1 x 1 x 3 x 3
        input = input.to(device)

        expected = torch.tensor([[[[[0., 1., 1.],
                                    [0., 0., 0.],
                                    [0., 0., 0.]]]]])  # 1 x 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[[1., 0., 0., 0.],
                                            [0., -1., 0., 2.],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]]])  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]]])  # 1 x 4 x 4
        identity = identity.to(device)

        input = input.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 4 x 4
        identity = identity.repeat(5, 1, 1)  # 5 x 4 x 4

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)

    def test_same_on_batch(self, device):
        f = RandomVerticalFlip3D(p=0.5, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device):

        f = nn.Sequential(
            RandomVerticalFlip3D(p=1.0, return_transform=True),
            RandomVerticalFlip3D(p=1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomVerticalFlip3D(p=1.0, return_transform=True),
            RandomVerticalFlip3D(p=1.0),
        )

        input = torch.tensor([[[[[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 1., 1.]]]]])  # 1 x 1 x 1 x 4 x 4
        input = input.to(device)

        expected_transform = torch.tensor([[[1., 0., 0., 0.],
                                            [0., -1., 0., 2.],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]]])  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        assert_allclose(f(input)[0], input.squeeze())
        assert_allclose(f(input)[1], expected_transform_1)
        assert_allclose(f1(input)[0], input.squeeze())
        assert_allclose(f1(input)[1], expected_transform)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return kornia.random_vflip(data)

        input = torch.tensor([[[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 1., 1.]]])  # 4 x 4

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[[0., 0., 0.],
                               [5., 5., 0.],
                               [0., 0., 0.]]])  # 1 x 4 x 4

        input = input.repeat(2, 1, 1)  # 2 x 4 x 4

        expected = torch.tensor([[[[0., 0., 0.],
                                   [5., 5., 0.],
                                   [0., 0., 0.]]]])  # 1 x 4 x 4

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self, device):
        input = torch.rand((1, 3, 3)).to(device)  # 4 x 4
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomVerticalFlip3D(p=1.), (input, ), raise_exception=True)


class TestRandomDepthicalFlip3D:

    def test_smoke(self):
        f = RandomDepthicalFlip3D(0.5)
        repr = "RandomDepthicalFlip3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=0.5)"
        assert str(f) == repr

    def test_random_dflip(self, device):

        f = RandomDepthicalFlip3D(p=1.0, return_transform=True)
        f1 = RandomDepthicalFlip3D(p=0., return_transform=True)
        f2 = RandomDepthicalFlip3D(p=1.)
        f3 = RandomDepthicalFlip3D(p=0.)

        input = torch.tensor([[[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 1.]],
                              [[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 2.]]])  # 2 x 3 x 4

        input = input.to(device)

        expected = torch.tensor([[[0., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 0., 0., 2.]],
                                 [[0., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 0., 0., 1.]]])  # 2 x 3 x 4
        expected = expected.to(device)

        expected_transform = torch.tensor([[1., 0., 0., 0.],
                                           [0., 1., 0., 0.],
                                           [0., 0., -1., 1.],
                                           [0., 0., 0., 1.]])  # 4 x 4
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])  # 4 x 4
        identity = identity.to(device)

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)
        assert_allclose(f2(input), expected)
        assert_allclose(f3(input), input)

    def test_batch_random_dflip(self, device):

        f = RandomDepthicalFlip3D(p=1.0, return_transform=True)
        f1 = RandomDepthicalFlip3D(p=0.0, return_transform=True)

        input = torch.tensor([[[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 1.]],
                              [[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 2.]]])  # 2 x 3 x 4

        input = input.to(device)

        expected = torch.tensor([[[0., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 0., 0., 2.]],
                                 [[0., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 0., 0., 1.]]])  # 2 x 3 x 4
        expected = expected.to(device)

        expected_transform = torch.tensor([[[1., 0., 0., 0.],
                                            [0., 1., 0., 0.],
                                            [0., 0., -1., 1.],
                                            [0., 0., 0., 1.]]])  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]]])  # 1 x 4 x 4
        identity = identity.to(device)

        input = input.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 4 x 4
        identity = identity.repeat(5, 1, 1)  # 5 x 4 x 4

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)

    def test_same_on_batch(self, device):
        f = RandomDepthicalFlip3D(p=0.5, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 2, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device):

        f = nn.Sequential(
            RandomDepthicalFlip3D(p=1.0, return_transform=True),
            RandomDepthicalFlip3D(p=1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomDepthicalFlip3D(p=1.0, return_transform=True),
            RandomDepthicalFlip3D(p=1.0),
        )

        input = torch.tensor([[[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 1.]],
                              [[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 2.]]])  # 2 x 3 x 4
        input = input.to(device)

        expected_transform = torch.tensor([[[1., 0., 0., 0.],
                                            [0., 1., 0., 0.],
                                            [0., 0., -1., 1.],
                                            [0., 0., 0., 1.]]])  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        assert_allclose(f(input)[0], input.squeeze())
        assert_allclose(f(input)[1], expected_transform_1)
        assert_allclose(f1(input)[0], input.squeeze())
        assert_allclose(f1(input)[1], expected_transform)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return kornia.random_vflip(data)

        input = torch.tensor([[[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 1., 1.]]])  # 4 x 4

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[[0., 0., 0.],
                               [5., 5., 0.],
                               [0., 0., 0.]]])  # 1 x 4 x 4

        input = input.repeat(2, 1, 1)  # 2 x 4 x 4

        expected = torch.tensor([[[[0., 0., 0.],
                                   [5., 5., 0.],
                                   [0., 0., 0.]]]])  # 1 x 4 x 4

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self, device):
        input = torch.rand((1, 3, 3)).to(device)  # 4 x 4
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomDepthicalFlip3D(p=1.), (input, ), raise_exception=True)


class TestRandomRotation3D:

    torch.manual_seed(0)  # for random reproductibility

    def test_smoke(self):
        f = RandomRotation3D(degrees=45.5)
        repr = """RandomRotation3D(degrees=tensor([[-45.5000,  45.5000],
        [-45.5000,  45.5000],
        [-45.5000,  45.5000]]), resample=BILINEAR, align_corners=False, p=0.5, """\
        """p_batch=1.0, same_on_batch=False, return_transform=False)"""
        assert str(f) == repr

    def test_random_rotation(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation3D(degrees=45.0, return_transform=True)
        f1 = RandomRotation3D(degrees=45.0)

        input = torch.tensor([[[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]],
                              [[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]],
                              [[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]]], device=device, dtype=dtype)  # 3 x 4 x 4

        expected = torch.tensor([[[[[0.2771, 0.0000, 0.0036, 0.0000],
                                    [0.5751, 0.0183, 0.7505, 0.4702],
                                    [0.0262, 0.2591, 0.5776, 0.4764],
                                    [0.0000, 0.0093, 0.0000, 0.0393]],
                                   [[0.0000, 0.0000, 0.0583, 0.0222],
                                    [0.1665, 0.0000, 1.0424, 1.0224],
                                    [0.1296, 0.4846, 1.4200, 1.2287],
                                    [0.0078, 0.3851, 0.3965, 0.3612]],
                                   [[0.0000, 0.7704, 0.6704, 0.0000],
                                    [0.0000, 0.0332, 0.2414, 0.0524],
                                    [0.0000, 0.3349, 1.4545, 1.3689],
                                    [0.0000, 0.0312, 0.5874, 0.8702]]]]], device=device, dtype=dtype)

        expected_transform = torch.tensor([[[0.5784, 0.7149, -0.3929, -0.0471],
                                            [-0.3657, 0.6577, 0.6585, 0.4035],
                                            [0.7292, -0.2372, 0.6419, -0.3799],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)

        expected_2 = torch.tensor([[[[[1., 0., 0., 2.],
                                      [0., 0., 0., 0.],
                                      [0., 1., 2., 0.],
                                      [0., 0., 1., 2.]],
                                     [[1., 0., 0., 2.],
                                      [0., 0., 0., 0.],
                                      [0., 1., 2., 0.],
                                      [0., 0., 1., 2.]],
                                     [[1., 0., 0., 2.],
                                      [0., 0., 0., 0.],
                                      [0., 1., 2., 0.],
                                      [0., 0., 1., 2.]]]]], device=device, dtype=dtype)

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
        assert_allclose(f1(input), expected_2, rtol=1e-6, atol=1e-4)

    def test_batch_random_rotation(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation3D(degrees=45.0, return_transform=True)

        input = torch.tensor([[[[1., 0., 0., 2.],
                                [0., 0., 0., 0.],
                                [0., 1., 2., 0.],
                                [0., 0., 1., 2.]],
                               [[1., 0., 0., 2.],
                                [0., 0., 0., 0.],
                                [0., 1., 2., 0.],
                                [0., 0., 1., 2.]],
                               [[1., 0., 0., 2.],
                                [0., 0., 0., 0.],
                                [0., 1., 2., 0.],
                                [0., 0., 1., 2.]]]], device=device, dtype=dtype)  # 1 x 1 x 4 x 4

        expected = torch.tensor([[[[[0.0000, 0.5106, 0.1146, 0.0000],
                                    [0.0000, 0.1261, 0.0000, 0.4723],
                                    [0.1714, 0.9931, 0.5442, 0.4684],
                                    [0.0193, 0.5802, 0.4195, 0.0000]],
                                   [[0.0000, 0.2386, 0.0000, 0.0000],
                                    [0.0187, 0.3527, 0.0000, 0.6119],
                                    [0.1294, 1.2251, 0.9130, 0.0942],
                                    [0.0962, 1.0769, 0.8448, 0.0000]],
                                   [[0.0000, 0.0202, 0.0000, 0.0000],
                                    [0.1092, 0.5845, 0.1038, 0.4598],
                                    [0.0000, 1.1218, 1.0796, 0.0000],
                                    [0.0780, 0.9513, 1.1278, 0.0000]]]],
                                 [[[[1.0000, 0.0000, 0.0000, 2.0000],
                                    [0.0000, 0.0000, 0.0000, 0.0000],
                                    [0.0000, 1.0000, 2.0000, 0.0000],
                                    [0.0000, 0.0000, 1.0000, 2.0000]],
                                   [[1.0000, 0.0000, 0.0000, 2.0000],
                                    [0.0000, 0.0000, 0.0000, 0.0000],
                                    [0.0000, 1.0000, 2.0000, 0.0000],
                                    [0.0000, 0.0000, 1.0000, 2.0000]],
                                   [[1.0000, 0.0000, 0.0000, 2.0000],
                                    [0.0000, 0.0000, 0.0000, 0.0000],
                                    [0.0000, 1.0000, 2.0000, 0.0000],
                                    [0.0000, 0.0000, 1.0000, 2.0000]]]]], device=device, dtype=dtype)

        expected_transform = torch.tensor([[[0.7894, -0.6122, 0.0449, 1.1892],
                                            [0.5923, 0.7405, -0.3176, -0.1816],
                                            [0.1612, 0.2773, 0.9472, -0.6049],
                                            [0.0000, 0.0000, 0.0000, 1.0000]],
                                           [[1.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 1.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 1.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)

        input = input.repeat(2, 1, 1, 1, 1)  # 5 x 4 x 4 x 3

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation3D(degrees=40, same_on_batch=True)
        input = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = nn.Sequential(
            RandomRotation3D(torch.tensor([-45.0, 90]), return_transform=True),
            RandomRotation3D(10.4, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomRotation3D(torch.tensor([-45.0, 90]), return_transform=True),
            RandomRotation3D(10.4),
        )

        input = torch.tensor([[[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]],
                              [[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]],
                              [[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]]], device=device, dtype=dtype)  # 3 x 4 x 4

        expected = torch.tensor([[[[[0.2752, 0.0000, 0.0000, 0.0000],
                                    [0.5767, 0.0059, 0.6440, 0.4307],
                                    [0.0000, 0.2793, 0.6638, 0.5716],
                                    [0.0000, 0.0049, 0.0000, 0.0685]],
                                   [[0.0000, 0.0000, 0.1806, 0.0000],
                                    [0.2138, 0.0000, 0.9061, 0.7966],
                                    [0.0657, 0.5395, 1.4299, 1.2912],
                                    [0.0000, 0.3600, 0.3088, 0.3655]],
                                   [[0.0000, 0.6515, 0.8861, 0.0000],
                                    [0.0000, 0.0000, 0.2278, 0.0000],
                                    [0.0027, 0.4403, 1.5462, 1.3480],
                                    [0.0000, 0.1182, 0.6297, 0.8623]]]]], device=device, dtype=dtype)

        expected_transform = torch.tensor([[[0.6306, 0.6496, -0.4247, 0.0044],
                                            [-0.3843, 0.7367, 0.5563, 0.4151],
                                            [0.6743, -0.1876, 0.7142, -0.4443],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)

        expected_transform_2 = torch.tensor([[[0.9611, 0.0495, -0.2717, 0.2557],
                                              [0.1255, 0.7980, 0.5894, -0.4747],
                                              [0.2460, -0.6006, 0.7608, 0.7710],
                                              [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)

        out, mat = f(input)
        _, mat_2 = f1(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
        assert_allclose(mat_2, expected_transform_2, rtol=1e-6, atol=1e-4)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):

        torch.manual_seed(0)  # for random reproductibility

        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return kornia.random_rotation(data, degrees=45.0)

        input = torch.tensor([[[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]],
                              [[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]],
                              [[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]]])  # 3 x 4 x 4

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[[0., 0., 0.],
                               [5., 5., 0.],
                               [0., 0., 0.]],
                              [[0., 0., 0.],
                               [5., 5., 0.],
                               [0., 0., 0.]],
                              [[0., 0., 0.],
                               [5., 5., 0.],
                               [0., 0., 0.]]])  # 3 x 3 x 3

        expected = torch.tensor([[[0.0000, 0.2584, 0.0000],
                                  [2.9552, 5.0000, 0.2584],
                                  [1.6841, 0.4373, 0.0000]]])

        actual = op_trace(input)

        assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)

    def test_gradcheck(self, device):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3, 3)).to(device)  # 3 x 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomRotation3D(degrees=(15.0, 15.0), p=1.), (input, ), raise_exception=True)


class TestRandomCrop3D:
    def test_smoke(self):
        f = RandomCrop3D(size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, p=1.)
        repr = "RandomCrop3D(crop_size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, "\
            "padding_mode=constant, resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, "\
            "return_transform=False)"
        assert str(f) == repr

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_no_padding(self, batch_size, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[[[
            [0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]
        ]]]], device=device, dtype=dtype).repeat(batch_size, 1, 5, 1, 1)
        f = RandomCrop3D(size=(2, 3, 4), padding=None, align_corners=True, p=1.)
        out = f(inp)
        if batch_size == 1:
            expected = torch.tensor([[[[
                [11, 12, 13, 14],
                [16, 17, 18, 19],
                [21, 22, 23, 24]
            ]]]], device=device, dtype=dtype).repeat(batch_size, 1, 2, 1, 1)
        if batch_size == 2:
            expected = torch.tensor([
                [[[[6.0000, 7.0000, 8.0000, 9.0000],
                   [11.0000, 12.0000, 13.0000, 14.0000],
                   [16.0000, 17.0000, 18.0000, 19.0000]],
                  [[6.0000, 7.0000, 8.0000, 9.0000],
                   [11.0000, 12.0000, 13.0000, 14.0000],
                   [16.0000, 17.0000, 18.0000, 19.0000]]]],
                [[[[11.0000, 12.0000, 13.0000, 14.0000],
                   [16.0000, 17.0000, 18.0000, 19.0000],
                   [21.0000, 22.0000, 23.0000, 24.0000]],
                  [[11.0000, 12.0000, 13.0000, 14.0000],
                   [16.0000, 17.0000, 18.0000, 19.0000],
                   [21.0000, 22.0000, 23.0000, 24.0000]]]]], device=device, dtype=dtype)

        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomCrop3D(size=(2, 3, 4), padding=None, align_corners=True, p=1., same_on_batch=True)
        input = torch.eye(6).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 5, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    @pytest.mark.parametrize("padding", [1, (1, 1, 1), (1, 1, 1, 1, 1, 1)])
    def test_padding_batch(self, padding, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]], device=device, dtype=dtype).repeat(batch_size, 1, 3, 1, 1)
        expected = torch.tensor([[[
            [[0., 1., 2., 10.],
             [3., 4., 5., 10.],
             [6., 7., 8., 10.]],
            [[0., 1., 2., 10.],
             [3., 4., 5., 10.],
             [6., 7., 8., 10.]],
        ]], [[
            [[3., 4., 5., 10.],
             [6., 7., 8., 10.],
             [10, 10, 10, 10.]],
            [[10, 10, 10, 10.],
             [10, 10, 10, 10.],
             [10, 10, 10, 10.]],
        ]]], device=device, dtype=dtype)
        f = RandomCrop3D(size=(2, 3, 4), fill=10., padding=padding, align_corners=True, p=1.)
        out = f(inp)

        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[
            [0., 1., 2.],
        ]], device=device, dtype=dtype)
        expected = torch.tensor([[[
            [[9., 9., 9., 9.],
             [9., 9., 9., 9.],
             [9., 9., 9., 9.]],
            [[0., 1., 2., 9.],
             [9., 9., 9., 9.],
             [9., 9., 9., 9.]],
        ]]], device=device, dtype=dtype)
        rc = RandomCrop3D(size=(2, 3, 4), pad_if_needed=True, fill=9, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomCrop3D(size=(3, 3, 3), p=1.), (inp, ), raise_exception=True)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.).forward
        op_script = torch.jit.script(op)
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        actual = op_script(img)
        expected = kornia.center_crop3d(img)
        assert_allclose(actual, expected)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit_trace(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.).forward
        op_script = torch.jit.script(op)
        # 1. Trace op
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        op_trace = torch.jit.trace(op_script, (img,))

        # 2. Generate new input
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        # 3. Evaluate
        actual = op_trace(img)
        expected = op(img)
        assert_allclose(actual, expected)


class TestCenterCrop3D:

    def test_no_transform(self, device, dtype):
        inp = torch.rand(1, 2, 4, 4, 4, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop3D(2)(inp)
        assert out.shape == (1, 2, 2, 2, 2)

    def test_transform(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, 8, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop3D(2, return_transform=True)(inp)
        assert len(out) == 2
        assert out[0].shape == (1, 2, 2, 2, 2)
        assert out[1].shape == (1, 4, 4)

    def test_no_transform_tuple(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, 8, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop3D((3, 4, 5))(inp)
        assert out.shape == (1, 2, 3, 4, 5)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.CenterCrop3D(3), (input,), raise_exception=True)


class TestRandomEqualize3D:
    def test_smoke(self, device, dtype):
        f = RandomEqualize3D(p=0.5)
        repr = "RandomEqualize3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_equalize(self, device, dtype):
        f = RandomEqualize3D(p=1.0, return_transform=True)
        f1 = RandomEqualize3D(p=0., return_transform=True)
        f2 = RandomEqualize3D(p=1.)
        f3 = RandomEqualize3D(p=0.)

        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width, device=device, dtype=dtype).squeeze(dim=0)

        row_expected = torch.tensor([
            0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000
        ], device=device, dtype=dtype)
        expected = self.build_input(channels, depth, height, width, bs=1, row=row_expected,
                                    device=device, dtype=dtype)

        identity = kornia.eye_like(4, expected)

        assert_allclose(f(inputs3d)[0], expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f(inputs3d)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs3d)[0], inputs3d, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs3d)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f2(inputs3d), expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f3(inputs3d), inputs3d, rtol=1e-4, atol=1e-4)

    def test_batch_random_equalize(self, device, dtype):
        f = RandomEqualize3D(p=1.0, return_transform=True)
        f1 = RandomEqualize3D(p=0., return_transform=True)
        f2 = RandomEqualize3D(p=1.)
        f3 = RandomEqualize3D(p=0.)

        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor([
            0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000
        ])
        expected = self.build_input(channels, depth, height, width, bs, row=row_expected,
                                    device=device, dtype=dtype)

        identity = kornia.eye_like(4, expected)  # 2 x 4 x 4

        assert_allclose(f(inputs3d)[0], expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f(inputs3d)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs3d)[0], inputs3d, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs3d)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f2(inputs3d), expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f3(inputs3d), inputs3d, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomEqualize3D(p=0.5, same_on_batch=True)
        input = torch.eye(4, device=device, dtype=dtype)
        input = input.unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 2, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility

        inputs3d = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3 x 3
        inputs3d = utils.tensor_to_gradcheck_var(inputs3d)  # to var
        assert gradcheck(RandomEqualize3D(p=0.5), (inputs3d,), raise_exception=True)

    @staticmethod
    def build_input(channels, depth, height, width, bs=1, row=None, device='cpu', dtype=torch.float32):
        if row is None:
            row = torch.arange(width, device=device, dtype=dtype) / float(width)

        channel = torch.stack([row] * height)
        image = torch.stack([channel] * channels)
        image3d = torch.stack([image] * depth).transpose(0, 1)
        batch = torch.stack([image3d] * bs)

        return batch.to(device, dtype)
