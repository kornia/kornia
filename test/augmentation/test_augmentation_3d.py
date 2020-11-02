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

<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
=======
>>>>>>> Repr functions and smoke tests fixed (#710)
=======
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
>>>>>>> [Fix] fixes windows issues with augmentation smoke tests (#766)
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

<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
=======
>>>>>>> Repr functions and smoke tests fixed (#710)
=======
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
>>>>>>> [Fix] fixes windows issues with augmentation smoke tests (#766)
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

<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
=======
>>>>>>> Repr functions and smoke tests fixed (#710)
=======
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
>>>>>>> [Fix] fixes windows issues with augmentation smoke tests (#766)
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

<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomRotation3D(degrees=45.5)
        repr = """RandomRotation3D(degrees=tensor([[-45.5000, 45.5000],
        [-45.5000, 45.5000],
        [-45.5000, 45.5000]]), resample=BILINEAR, align_corners=False, p=0.5, """\
=======
=======
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
>>>>>>> [Fix] fixes windows issues with augmentation smoke tests (#766)
    def test_smoke(self):
        f = RandomRotation3D(degrees=45.5)
<<<<<<< refs/remotes/kornia/master
        repr = """RandomRotation3D(degrees=tensor([[-45.5000,  45.5000],
        [-45.5000,  45.5000],
        [-45.5000,  45.5000]]), resample=BILINEAR, align_corners=False, p=0.5, """\
>>>>>>> Repr functions and smoke tests fixed (#710)
=======
        repr = """RandomRotation3D(degrees=tensor([[-45.5000, 45.5000],
        [-45.5000, 45.5000],
        [-45.5000, 45.5000]]), resample=BILINEAR, align_corners=False, p=0.5, """\
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
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
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
                               [0., 0., 1., 2.]]], device=device, dtype=dtype)  # 3 x 4 x 4
=======
                               [0., 0., 1., 2.]]])  # 3 x 4 x 4
        input = input.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)

        expected = torch.tensor([[[[[0.0000, 0.0000, 0.6810, 0.5250],
                                    [0.5052, 0.0000, 0.0000, 0.0613],
                                    [0.1159, 0.1072, 0.5324, 0.0870],
                                    [0.0000, 0.0000, 0.1927, 0.0000]],
                                   [[0.0000, 0.1683, 0.6963, 0.1131],
                                    [0.0566, 0.0000, 0.5215, 0.2796],
                                    [0.0694, 0.6039, 1.4519, 1.1240],
                                    [0.0000, 0.1325, 0.1542, 0.2510]],
                                   [[0.0000, 0.2054, 0.0000, 0.0000],
                                    [0.0026, 0.6088, 0.7358, 0.2319],
<<<<<<< refs/remotes/kornia/master
                                    [0.1261, 1.0830, 1.3687, 1.4940],
                                    [0.0000, 0.0416, 0.2012, 0.3124]]]]], device=device, dtype=dtype)
=======
                                    [0.1262, 1.0830, 1.3687, 1.4940],
                                    [0.0000, 0.0416, 0.2012, 0.3124]]]]])
        expected = expected.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)

        expected_transform = torch.tensor([[[0.6523, 0.3666, -0.6635, 0.6352],
                                            [-0.6185, 0.7634, -0.1862, 1.4689],
                                            [0.4382, 0.5318, 0.7247, -1.1797],
<<<<<<< refs/remotes/kornia/master
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)
=======
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)

        expected_2 = torch.tensor([[[[[0.0000, 0.4771, 0.0243, 0.0000],
                                      [0.0000, 0.1652, 0.0000, 0.6771],
                                      [0.1668, 1.1430, 0.7131, 0.2692],
                                      [0.0285, 0.7100, 0.6012, 0.0000]],
                                     [[0.0000, 0.3068, 0.0000, 0.0000],
                                      [0.0000, 0.3175, 0.0000, 0.6602],
                                      [0.1330, 1.1962, 0.9750, 0.0000],
                                      [0.0648, 0.9818, 0.9785, 0.0000]],
                                     [[0.0000, 0.1136, 0.0000, 0.0000],
                                      [0.0518, 0.4617, 0.0000, 0.4928],
                                      [0.0407, 1.0954, 1.1413, 0.0000],
                                      [0.0587, 0.8768, 1.1815, 0.0000]]]]])
        expected_2 = expected_2.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
=======
=======
<<<<<<< master
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
                               [0., 0., 1., 2.]]], device=device, dtype=dtype)  # 3 x 4 x 4
=======
                               [0., 0., 1., 2.]]])  # 3 x 4 x 4
        input = input.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)

        expected = torch.tensor([[[[[0.0000, 0.0000, 0.6810, 0.5250],
                                    [0.5052, 0.0000, 0.0000, 0.0613],
                                    [0.1159, 0.1072, 0.5324, 0.0870],
                                    [0.0000, 0.0000, 0.1927, 0.0000]],
                                   [[0.0000, 0.1683, 0.6963, 0.1131],
                                    [0.0566, 0.0000, 0.5215, 0.2796],
                                    [0.0694, 0.6039, 1.4519, 1.1240],
                                    [0.0000, 0.1325, 0.1542, 0.2510]],
                                   [[0.0000, 0.2054, 0.0000, 0.0000],
                                    [0.0026, 0.6088, 0.7358, 0.2319],
<<<<<<< master
                                    [0.1261, 1.0830, 1.3687, 1.4940],
                                    [0.0000, 0.0416, 0.2012, 0.3124]]]]], device=device, dtype=dtype)
=======
                                    [0.1262, 1.0830, 1.3687, 1.4940],
                                    [0.0000, 0.0416, 0.2012, 0.3124]]]]])
        expected = expected.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)

        expected_transform = torch.tensor([[[0.6523, 0.3666, -0.6635, 0.6352],
                                            [-0.6185, 0.7634, -0.1862, 1.4689],
                                            [0.4382, 0.5318, 0.7247, -1.1797],
<<<<<<< master
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)
=======
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)

        expected_2 = torch.tensor([[[[[0.0000, 0.4771, 0.0243, 0.0000],
                                      [0.0000, 0.1652, 0.0000, 0.6771],
                                      [0.1668, 1.1430, 0.7131, 0.2692],
                                      [0.0285, 0.7100, 0.6012, 0.0000]],
                                     [[0.0000, 0.3068, 0.0000, 0.0000],
                                      [0.0000, 0.3175, 0.0000, 0.6602],
                                      [0.1330, 1.1962, 0.9750, 0.0000],
                                      [0.0648, 0.9818, 0.9785, 0.0000]],
                                     [[0.0000, 0.1136, 0.0000, 0.0000],
                                      [0.0518, 0.4617, 0.0000, 0.4928],
                                      [0.0407, 1.0954, 1.1413, 0.0000],
                                      [0.0587, 0.8768, 1.1815, 0.0000]]]]])
        expected_2 = expected_2.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)

<<<<<<< refs/remotes/kornia/master
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
>>>>>>> Added random param gen tests. Added device awareness for parameter generators. (#757)

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
<<<<<<< refs/remotes/kornia/master
=======
        assert_allclose(f1(input), expected_2, rtol=1e-6, atol=1e-4)
=======
        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)

        torch.manual_seed(0)  # for random reproductibility
        assert_allclose(f1(input), expected, rtol=1e-6, atol=1e-4)
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)

    def test_batch_random_rotation(self, device, dtype):
>>>>>>> Added random param gen tests. Added device awareness for parameter generators. (#757)

<<<<<<< refs/remotes/kornia/master
        torch.manual_seed(0)  # for random reproductibility
        assert_allclose(f1(input), expected, rtol=1e-6, atol=1e-4)

    def test_batch_random_rotation(self, device, dtype):

=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
        torch.manual_seed(24)  # for random reproductibility

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
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
<<<<<<< master
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
                                [0., 0., 1., 2.]]]], device=device, dtype=dtype)  # 1 x 1 x 4 x 4

        expected = torch.tensor([[[[[1.0000, 0.0000, 0.0000, 2.0000],
=======
                                [0., 0., 1., 2.]]]], device=device, dtype=dtype)  # 1 x 1 x 4 x 4

<<<<<<< refs/remotes/kornia/master
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
>>>>>>> Added random param gen tests. Added device awareness for parameter generators. (#757)
=======
        expected = torch.tensor([[[[[1.0000, 0.0000, 0.0000, 2.0000],
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
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
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
                                    [0.0000, 0.0000, 1.0000, 2.0000]]]],
                                 [[[[0.0000, 0.0726, 0.0000, 0.0000],
                                    [0.1038, 1.0134, 0.5566, 0.1519],
                                    [0.0000, 1.0849, 1.1068, 0.0000],
                                    [0.1242, 1.1065, 0.9681, 0.0000]],
                                   [[0.0000, 0.0047, 0.0166, 0.0000],
                                    [0.0579, 0.4459, 0.0000, 0.4728],
                                    [0.1864, 1.3349, 0.7530, 0.3251],
                                    [0.1431, 1.2481, 0.4471, 0.0000]],
                                   [[0.0000, 0.4840, 0.2314, 0.0000],
                                    [0.0000, 0.0328, 0.0000, 0.1434],
                                    [0.1899, 0.5580, 0.0000, 0.9170],
                                    [0.0000, 0.2042, 0.1571, 0.0855]]]]], device=device, dtype=dtype)

        expected_transform = torch.tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
<<<<<<< refs/remotes/kornia/master
                                            [0.0000, 1.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 1.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 1.0000]],
                                           [[0.7522, -0.6326, -0.1841, 1.5047],
                                            [0.6029, 0.5482, 0.5796, -0.8063],
                                            [-0.2657, -0.5470, 0.7938, 1.4252],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)
=======
                                [0., 0., 1., 2.]]]])  # 1 x 1 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[[7.5651e-01, 6.4340e-02, 0.0000e+00, 0.0000e+00],
                                    [8.9954e-02, 2.0099e-01, 8.2089e-01, 4.4695e-01],
                                    [0.0000e+00, 4.8303e-01, 8.0751e-01, 1.1574e+00],
                                    [0.0000e+00, 6.5891e-02, 1.7392e-01, 2.9013e-01]],
                                   [[4.0104e-01, 0.0000e+00, 1.9018e-02, 7.2668e-04],
                                    [3.5247e-01, 0.0000e+00, 6.5445e-01, 3.7179e-01],
                                    [5.4804e-02, 7.0015e-01, 1.7578e+00, 1.2048e+00],
                                    [3.7536e-02, 3.8235e-01, 9.0737e-01, 1.1033e+00]],
                                   [[1.2648e-02, 0.0000e+00, 9.4951e-01, 4.6696e-01],
                                    [1.2791e-01, 0.0000e+00, 8.2977e-02, 0.0000e+00],
                                    [1.3047e-02, 2.8671e-01, 8.3294e-01, 1.7991e-01],
                                    [0.0000e+00, 5.8076e-02, 6.7866e-01, 1.5130e+00]]]],

                                 [[[[1.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 1.0000e+00, 2.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00]],
                                   [[1.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 1.0000e+00, 2.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00]],
                                   [[1.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 1.0000e+00, 2.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00]]]]])
        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.8017, 0.4358, -0.4090, 0.0527],
                                            [-0.0877, 0.7627, 0.6408, -0.1533],
                                            [0.5912, -0.4779, 0.6497, 0.1803],
=======
                                    [0.0000, 0.0000, 1.0000, 2.0000]]]]], device=device, dtype=dtype)

        expected_transform = torch.tensor([[[0.7894, -0.6122, 0.0449, 1.1892],
                                            [0.5923, 0.7405, -0.3176, -0.1816],
                                            [0.1612, 0.2773, 0.9472, -0.6049],
>>>>>>> Added random param gen tests. Added device awareness for parameter generators. (#757)
                                            [0.0000, 0.0000, 0.0000, 1.0000]],
                                           [[1.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 1.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 1.0000, 0.0000],
<<<<<<< refs/remotes/kornia/master
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
=======
=======
                                            [0.0000, 1.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 1.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 1.0000]],
                                           [[0.7522, -0.6326, -0.1841, 1.5047],
                                            [0.6029, 0.5482, 0.5796, -0.8063],
                                            [-0.2657, -0.5470, 0.7938, 1.4252],
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)
<<<<<<< refs/remotes/kornia/master
>>>>>>> Added random param gen tests. Added device awareness for parameter generators. (#757)
=======
=======
                                [0., 0., 1., 2.]]]])  # 1 x 1 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[[7.5651e-01, 6.4340e-02, 0.0000e+00, 0.0000e+00],
                                    [8.9954e-02, 2.0099e-01, 8.2089e-01, 4.4695e-01],
                                    [0.0000e+00, 4.8303e-01, 8.0751e-01, 1.1574e+00],
                                    [0.0000e+00, 6.5891e-02, 1.7392e-01, 2.9013e-01]],
                                   [[4.0104e-01, 0.0000e+00, 1.9018e-02, 7.2668e-04],
                                    [3.5247e-01, 0.0000e+00, 6.5445e-01, 3.7179e-01],
                                    [5.4804e-02, 7.0015e-01, 1.7578e+00, 1.2048e+00],
                                    [3.7536e-02, 3.8235e-01, 9.0737e-01, 1.1033e+00]],
                                   [[1.2648e-02, 0.0000e+00, 9.4951e-01, 4.6696e-01],
                                    [1.2791e-01, 0.0000e+00, 8.2977e-02, 0.0000e+00],
                                    [1.3047e-02, 2.8671e-01, 8.3294e-01, 1.7991e-01],
                                    [0.0000e+00, 5.8076e-02, 6.7866e-01, 1.5130e+00]]]],

                                 [[[[1.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 1.0000e+00, 2.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00]],
                                   [[1.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 1.0000e+00, 2.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00]],
                                   [[1.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 1.0000e+00, 2.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00]]]]])
        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.8017, 0.4358, -0.4090, 0.0527],
                                            [-0.0877, 0.7627, 0.6408, -0.1533],
                                            [0.5912, -0.4779, 0.6497, 0.1803],
                                            [0.0000, 0.0000, 0.0000, 1.0000]],

                                           [[1.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 1.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 1.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)

        input = input.repeat(2, 1, 1, 1, 1)  # 5 x 4 x 4 x 3

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation3D(degrees=40, same_on_batch=True)
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
        input = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 6, 1, 1)
=======
        input = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1, 1)
>>>>>>> Added random param gen tests. Added device awareness for parameter generators. (#757)
=======
        input = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 6, 1, 1)
>>>>>>> RandomRotation3D cuda fix (#810)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        torch.manual_seed(24)  # for random reproductibility

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
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
                               [0., 0., 1., 2.]]], device=device, dtype=dtype)  # 3 x 4 x 4

        expected = torch.tensor([[[[[0.3431, 0.1239, 0.0000, 1.0348],
                                    [0.0000, 0.2035, 0.1139, 0.1770],
                                    [0.0789, 0.9057, 1.7780, 0.0000],
                                    [0.0000, 0.2286, 1.2498, 1.2643]],
                                   [[0.5460, 0.2131, 0.0000, 1.1453],
                                    [0.0000, 0.0899, 0.0000, 0.4293],
                                    [0.0797, 1.0193, 1.6677, 0.0000],
                                    [0.0000, 0.2458, 1.2765, 1.0920]],
                                   [[0.6322, 0.2614, 0.0000, 0.9207],
                                    [0.0000, 0.0037, 0.0000, 0.6551],
                                    [0.0689, 0.9251, 1.3442, 0.0000],
                                    [0.0000, 0.2449, 0.9856, 0.6862]]]]], device=device, dtype=dtype)

        expected_transform = torch.tensor([[[0.9857, -0.1686, -0.0019, 0.2762],
                                            [0.1668, 0.9739, 0.1538, -0.3650],
                                            [-0.0241, -0.1520, 0.9881, 0.2760],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)

        expected_transform_2 = torch.tensor([[[0.2348, -0.1615, 0.9585, 0.4316],
                                              [0.1719, 0.9775, 0.1226, -0.3467],
                                              [-0.9567, 0.1360, 0.2573, 1.9738],
                                              [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)
=======
                               [0., 0., 1., 2.]]])  # 3 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[[4.5604e-02, 4.6441e-03, 7.1205e-01, 7.3379e-01],
                                    [2.7580e-01, 3.3129e-02, 1.1986e-01, 5.7227e-01],
                                    [1.4329e-01, 4.0604e-02, 4.4003e-03, 1.3030e-01],
                                    [8.0267e-05, 9.0396e-03, 2.8991e-02, 4.0690e-03]],

                                   [[9.8822e-03, 4.4220e-02, 1.2963e-01, 3.9873e-02],
                                    [4.3757e-02, 3.4982e-01, 5.1378e-01, 8.6131e-02],
                                    [6.6809e-02, 5.5708e-01, 1.0904e+00, 4.6732e-01],
                                    [2.9877e-02, 1.9682e-01, 3.5764e-01, 1.0877e-01]],

                                   [[2.3905e-02, 2.1605e-01, 2.5145e-02, 3.3507e-04],
                                    [1.1453e-01, 1.1965e+00, 1.2008e+00, 3.6272e-01],
                                    [1.3368e-01, 5.4211e-01, 1.2059e+00, 1.0104e+00],
                                    [8.6762e-03, 6.7149e-02, 2.0946e-01, 2.7900e-01]]]]])

        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.8369, 0.0343, -0.5463, 0.7395],
                                            [-0.5104, 0.4091, -0.7563, 2.4083],
                                            [0.1976, 0.9118, 0.3599, -1.0240],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)

        expected_transform_2 = torch.tensor([[[0.9869, -0.1351, 0.0879, 0.1343],
                                              [0.1598, 0.7501, -0.6417, 0.7769],
                                              [0.0208, 0.6474, 0.7619, -0.7641],
                                              [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform_2 = expected_transform_2.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
=======
=======
<<<<<<< master
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
                               [0., 0., 1., 2.]]], device=device, dtype=dtype)  # 3 x 4 x 4

        expected = torch.tensor([[[[[0.3431, 0.1239, 0.0000, 1.0348],
                                    [0.0000, 0.2035, 0.1139, 0.1770],
                                    [0.0789, 0.9057, 1.7780, 0.0000],
                                    [0.0000, 0.2286, 1.2498, 1.2643]],
                                   [[0.5460, 0.2131, 0.0000, 1.1453],
                                    [0.0000, 0.0899, 0.0000, 0.4293],
                                    [0.0797, 1.0193, 1.6677, 0.0000],
                                    [0.0000, 0.2458, 1.2765, 1.0920]],
                                   [[0.6322, 0.2614, 0.0000, 0.9207],
                                    [0.0000, 0.0037, 0.0000, 0.6551],
                                    [0.0689, 0.9251, 1.3442, 0.0000],
                                    [0.0000, 0.2449, 0.9856, 0.6862]]]]], device=device, dtype=dtype)

        expected_transform = torch.tensor([[[0.9857, -0.1686, -0.0019, 0.2762],
                                            [0.1668, 0.9739, 0.1538, -0.3650],
                                            [-0.0241, -0.1520, 0.9881, 0.2760],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)

        expected_transform_2 = torch.tensor([[[0.2348, -0.1615, 0.9585, 0.4316],
                                              [0.1719, 0.9775, 0.1226, -0.3467],
                                              [-0.9567, 0.1360, 0.2573, 1.9738],
                                              [0.0000, 0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)
<<<<<<< refs/remotes/kornia/master
>>>>>>> Added random param gen tests. Added device awareness for parameter generators. (#757)
=======
=======
                               [0., 0., 1., 2.]]])  # 3 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[[4.5604e-02, 4.6441e-03, 7.1205e-01, 7.3379e-01],
                                    [2.7580e-01, 3.3129e-02, 1.1986e-01, 5.7227e-01],
                                    [1.4329e-01, 4.0604e-02, 4.4003e-03, 1.3030e-01],
                                    [8.0267e-05, 9.0396e-03, 2.8991e-02, 4.0690e-03]],

                                   [[9.8822e-03, 4.4220e-02, 1.2963e-01, 3.9873e-02],
                                    [4.3757e-02, 3.4982e-01, 5.1378e-01, 8.6131e-02],
                                    [6.6809e-02, 5.5708e-01, 1.0904e+00, 4.6732e-01],
                                    [2.9877e-02, 1.9682e-01, 3.5764e-01, 1.0877e-01]],

                                   [[2.3905e-02, 2.1605e-01, 2.5145e-02, 3.3507e-04],
                                    [1.1453e-01, 1.1965e+00, 1.2008e+00, 3.6272e-01],
                                    [1.3368e-01, 5.4211e-01, 1.2059e+00, 1.0104e+00],
                                    [8.6762e-03, 6.7149e-02, 2.0946e-01, 2.7900e-01]]]]])

        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.8369, 0.0343, -0.5463, 0.7395],
                                            [-0.5104, 0.4091, -0.7563, 2.4083],
                                            [0.1976, 0.9118, 0.3599, -1.0240],
                                            [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform = expected_transform.to(device)

        expected_transform_2 = torch.tensor([[[0.9869, -0.1351, 0.0879, 0.1343],
                                              [0.1598, 0.7501, -0.6417, 0.7769],
                                              [0.0208, 0.6474, 0.7619, -0.7641],
                                              [0.0000, 0.0000, 0.0000, 1.0000]]])
        expected_transform_2 = expected_transform_2.to(device)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)

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
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master


class TestRandomCrop3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomCrop3D(size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, p=1.)
        repr = "RandomCrop3D(crop_size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, "\
            "padding_mode=constant, resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, "\
            "return_transform=False)"
        assert str(f) == repr

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_no_padding(self, batch_size, device, dtype):
        torch.manual_seed(42)
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
        torch.manual_seed(42)
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
            [[3., 4., 5., 10.],
             [6., 7., 8., 10.],
             [10, 10, 10, 10.]],
        ]]], device=device, dtype=dtype)
        f = RandomCrop3D(size=(2, 3, 4), fill=10., padding=padding, align_corners=True, p=1.)
        out = f(inp)

        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(42)
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
=======
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)
=======
<<<<<<< master
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)


class TestRandomCrop3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomCrop3D(size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, p=1.)
        repr = "RandomCrop3D(crop_size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, "\
            "padding_mode=constant, resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, "\
            "return_transform=False)"
        assert str(f) == repr

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_no_padding(self, batch_size, device, dtype):
        torch.manual_seed(42)
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
        torch.manual_seed(42)
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
            [[3., 4., 5., 10.],
             [6., 7., 8., 10.],
             [10, 10, 10, 10.]],
        ]]], device=device, dtype=dtype)
        f = RandomCrop3D(size=(2, 3, 4), fill=10., padding=padding, align_corners=True, p=1.)
        out = f(inp)

        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(42)
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
=======
>>>>>>> [Fix] gpu tests for crop3d and flip (#727)


class TestRandomCrop3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
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
                [10, 11, 12, 13],
                [15, 16, 17, 18],
                [20, 21, 22, 23]
            ]]]], device=device, dtype=dtype).repeat(batch_size, 1, 2, 1, 1)
        if batch_size == 2:
            expected = torch.tensor([[[
                [[0., 1., 2., 3.],
                 [5., 6., 7., 8.],
                 [10, 11, 12, 13]],
                [[0., 1., 2., 3.],
                 [5., 6., 7., 8.],
                 [10, 11, 12, 13]],
            ]], [[
                [[1., 2., 3., 4.],
                 [6., 7., 8., 9.],
                 [11, 12, 13, 14]],
                [[1., 2., 3., 4.],
                 [6., 7., 8., 9.],
                 [11, 12, 13, 14]],
            ]]], device=device, dtype=dtype)

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
            [[10, 10, 10, 10],
             [10, 0., 1., 2.],
             [10, 3., 4., 5.]],
            [[10, 10, 10, 10],
             [10, 0., 1., 2.],
             [10, 3., 4., 5.]],
        ]], [[
            [[10, 10, 10, 10],
             [0., 1., 2., 10],
             [3., 4., 5., 10]],
            [[10, 10, 10, 10],
             [0., 1., 2., 10],
             [3., 4., 5., 10]],
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
            [[9., 0., 1., 2.],
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
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
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
