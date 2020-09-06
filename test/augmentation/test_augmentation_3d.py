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
    RandomRotation3D
)


class TestRandomHorizontalFlip3D:

    def smoke_test(self, device):
        f = RandomHorizontalFlip3D(0.5)
        repr = "RandomHorizontalFlip3D(p=0.5, return_transform=False)"
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
            RandomHorizontalFlip3D(1.0, return_transform=True),
            RandomHorizontalFlip3D(1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomHorizontalFlip3D(1.0, return_transform=True),
            RandomHorizontalFlip3D(1.0),
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

    def smoke_test(self, device):
        f = RandomVerticalFlip3D(0.5)
        repr = "RandomVerticalFlip3D(p=0.5, return_transform=False)"
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
            RandomVerticalFlip3D(1.0, return_transform=True),
            RandomVerticalFlip3D(1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomVerticalFlip3D(1.0, return_transform=True),
            RandomVerticalFlip3D(1.0),
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

    def smoke_test(self, device):
        f = RandomDepthicalFlip3D(0.5)
        repr = "RandomDepthicalFlip3D(p=0.5, return_transform=False)"
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
            RandomDepthicalFlip3D(1.0, return_transform=True),
            RandomDepthicalFlip3D(1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomDepthicalFlip3D(1.0, return_transform=True),
            RandomDepthicalFlip3D(1.0),
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


@pytest.mark.skip(reason="need to recompute values")
class TestRandomRotation3D:

    torch.manual_seed(0)  # for random reproductibility

    def smoke_test(self, device):
        f = RandomRotation3D(degrees=45.5)
        repr = "RandomRotation3D(degrees=45.5, return_transform=False)"
        assert str(f) == repr

    def test_random_rotation(self, device):
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
                               [0., 0., 1., 2.]]])  # 3 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[[9.9412e-01, 0.0000e+00, 8.5407e-03, 1.9535e+00],
                                    [1.7328e-05, 3.8945e-03, 1.5617e-02, 1.0553e-04],
                                    [0.0000e+00, 9.8295e-01, 1.9789e+00, 4.9531e-02],
                                    [0.0000e+00, 0.0000e+00, 9.7034e-01, 1.9548e+00]],
                                   [[9.6646e-01, 0.0000e+00, 4.3866e-02, 1.9559e+00],
                                    [1.1586e-02, 0.0000e+00, 1.0260e-04, 0.0000e+00],
                                    [4.4472e-03, 9.9659e-01, 1.9833e+00, 3.4181e-05],
                                    [0.0000e+00, 7.8456e-03, 9.9959e-01, 1.9956e+00]],
                                   [[9.3772e-01, 0.0000e+00, 7.8179e-02, 1.8983e+00],
                                    [2.2707e-02, 0.0000e+00, 9.6477e-04, 2.2624e-02],
                                    [2.1575e-02, 9.9975e-01, 1.9243e+00, 0.0000e+00],
                                    [3.1300e-04, 3.2790e-02, 1.0249e+00, 1.9474e+00]]]]])
        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.7168, 0.5830, 0.3825, -1.1651],
                                            [-0.5853, 0.8012, -0.1242, 1.0699],
                                            [-0.3789, -0.1349, 0.9155, 0.7079],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)

        expected_2 = torch.tensor([[[[[0.5337, 0.0176, 0.0066, 0.2952],
                                      [0.1152, 0.1210, 0.7456, 1.4092],
                                      [0.0000, 0.0000, 0.3873, 0.5115],
                                      [0.0000, 0.0000, 0.0000, 0.0000]],
                                     [[0.1647, 1.5763, 0.5870, 0.0000],
                                      [0.0000, 0.2845, 0.0000, 0.0000],
                                      [0.0000, 0.5602, 0.6448, 0.2305],
                                      [0.3809, 1.3558, 1.5374, 1.5583]],
                                     [[0.0000, 0.0000, 0.0000, 0.0000],
                                      [0.4242, 0.0000, 0.0000, 0.0000],
                                      [1.4400, 0.1584, 0.0000, 0.0000],
                                      [0.0946, 0.0000, 0.0000, 0.0000]]]]])
        expected_2 = expected_2.to(device)

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
        assert_allclose(f1(input), expected_2, rtol=1e-6, atol=1e-4)

    def test_batch_random_rotation(self, device):

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
                                [0., 0., 1., 2.]]]])  # 1 x 1 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[[9.9412e-01, 0.0000e+00, 8.5407e-03, 1.9535e+00],
                                    [1.7328e-05, 3.8945e-03, 1.5617e-02, 1.0553e-04],
                                    [0.0000e+00, 9.8295e-01, 1.9789e+00, 4.9531e-02],
                                    [0.0000e+00, 0.0000e+00, 9.7034e-01, 1.9548e+00]],
                                   [[9.6646e-01, 0.0000e+00, 4.3866e-02, 1.9559e+00],
                                    [1.1586e-02, 0.0000e+00, 1.0260e-04, 0.0000e+00],
                                    [4.4472e-03, 9.9659e-01, 1.9833e+00, 3.4181e-05],
                                    [0.0000e+00, 7.8456e-03, 9.9959e-01, 1.9956e+00]],
                                   [[9.3772e-01, 0.0000e+00, 7.8179e-02, 1.8983e+00],
                                    [2.2707e-02, 0.0000e+00, 9.6477e-04, 2.2624e-02],
                                    [2.1575e-02, 9.9975e-01, 1.9243e+00, 0.0000e+00],
                                    [3.1300e-04, 3.2790e-02, 1.0249e+00, 1.9474e+00]]]],

                                 [[[[5.9268e-01, 4.6201e-01, 0.0000e+00, 1.3414e-01],
                                    [5.0854e-02, 0.0000e+00, 1.2416e-02, 1.1548e+00],
                                    [6.1057e-01, 9.5876e-01, 1.9510e-01, 0.0000e+00],
                                    [7.4132e-01, 8.6556e-01, 1.9031e-01, 0.0000e+00]],
                                   [[0.0000e+00, 1.0143e-01, 2.2231e-01, 0.0000e+00],
                                    [0.0000e+00, 1.8495e-01, 5.0593e-01, 4.9479e-01],
                                    [4.9456e-02, 5.0849e-01, 1.5325e+00, 7.8474e-01],
                                    [0.0000e+00, 4.7385e-01, 1.3469e+00, 1.1854e+00]],
                                   [[0.0000e+00, 0.0000e+00, 0.0000e+00, 9.7841e-02],
                                    [0.0000e+00, 0.0000e+00, 1.2882e-01, 8.0175e-01],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 7.2928e-01],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 2.1100e-01]]]]])
        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.7559, 0.2793, -0.5921, 0.7132],
                                            [-0.2756, 0.9561, 0.0991, 0.1928],
                                            [0.5938, 0.0883, 0.7998, -0.4259],
                                            [0.0000, 0.0000, 0.0000, 1.0000]],

                                           [[0.8194, -0.3079, -0.4836, 1.3678],
                                            [0.0754, 0.8941, -0.4415, 0.7456],
                                            [0.5683, 0.3253, 0.7558, -0.6899],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)

        input = input.repeat(2, 1, 1, 1, 1)  # 5 x 4 x 4 x 3

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)

    def test_same_on_batch(self, device):
        f = RandomRotation3D(degrees=40, same_on_batch=True)
        input = torch.eye(6).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device):

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
                               [0., 0., 1., 2.]]])  # 3 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[[5.8413e-01, 3.1238e-01, 2.7060e-02, 4.1555e-01],
                                    [4.3787e-02, 8.6122e-02, 8.3644e-02, 6.6170e-01],
                                    [3.7727e-01, 8.5232e-01, 5.1960e-01, 9.1474e-02],
                                    [2.7198e-01, 6.9109e-01, 5.7987e-01, 1.7646e-01]],

                                   [[8.8109e-02, 1.6755e-01, 1.0494e-01, 3.5867e-02],
                                    [6.1397e-02, 2.6093e-01, 4.2951e-01, 3.5508e-01],
                                    [7.9497e-02, 5.5609e-01, 1.3745e+00, 6.2975e-01],
                                    [8.9196e-03, 3.6598e-01, 1.1553e+00, 1.2267e+00]],

                                   [[6.1910e-03, 1.4123e-02, 5.0222e-02, 6.4776e-03],
                                    [2.9561e-05, 1.1410e-01, 5.5537e-01, 2.2792e-01],
                                    [0.0000e+00, 5.5169e-02, 5.4580e-01, 4.4281e-01],
                                    [0.0000e+00, 2.9420e-03, 1.7743e-01, 3.0618e-01]]]]])

        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.4690, 0.4978, 0.7295, -1.3100],
                                            [-0.2616, 0.8673, -0.4236, 1.0961],
                                            [-0.8435, 0.0078, 0.5370, 1.5263],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)

        expected_transform_2 = torch.tensor([[[0.2207, 0.0051, 0.9753, -0.6914],
                                              [0.4092, 0.9072, -0.0974, -0.1240],
                                              [-0.8853, 0.4206, 0.1981, 1.4573],
                                              [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform_2 = expected_transform_2.to(device)

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
        assert gradcheck(RandomRotation3D(degrees=(15.0, 15.0)), (input, ), raise_exception=True)
