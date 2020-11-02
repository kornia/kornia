from typing import Union, Tuple

import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.augmentation import (
    RandomMotionBlur,
    RandomMotionBlur3D,
)


class TestRandomMotionBlur:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device):
        f = RandomMotionBlur(kernel_size=(3, 5), angle=(10, 30), direction=0.5)
        repr = "RandomMotionBlur(kernel_size=(3, 5), angle=tensor([10., 30.]), direction=tensor([-0.5000,  0.5000]), "\
            "border_type='constant', p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    @pytest.mark.parametrize("same_on_batch", [True, False])
    @pytest.mark.parametrize("return_transform", [True, False])
    @pytest.mark.parametrize("p", [0., 1.])
    def test_random_motion_blur(self, same_on_batch, return_transform, p, device, dtype):
        f = RandomMotionBlur(kernel_size=(3, 5), angle=(10, 30), direction=0.5,
                             same_on_batch=same_on_batch, return_transform=return_transform, p=p)
        input = torch.randn(1, 3, 5, 6).repeat(2, 1, 1, 1)

        output = f(input)

        if return_transform:
            assert len(output) == 2, f"must return a length 2 tuple if return_transform is True. Got {len(output)}."
            identity = kornia.eye_like(3, input)
            output, mat = output
            assert_allclose(mat, identity, rtol=1e-4, atol=1e-4)

        if same_on_batch:
            assert_allclose(output[0], output[1], rtol=1e-4, atol=1e-4)
        elif p == 0:
            assert_allclose(output, input, rtol=1e-4, atol=1e-4)
        else:
            assert not torch.allclose(output[0], output[1], rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 11, 7)).to(device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        # TODO: Gradcheck for param random gen failed. Suspect get_motion_kernel2d issue.
        params = {
            'batch_prob': torch.tensor([True]),
            'ksize_factor': torch.tensor([31]),
            'angle_factor': torch.tensor([30.]),
            'direction_factor': torch.tensor([-0.5]),
            'border_type': torch.tensor([0]),
        }
        assert gradcheck(RandomMotionBlur(
            kernel_size=3, angle=(10, 30), direction=(-0.5, 0.5), p=1.0), (inp, params), raise_exception=True)


class TestRandomMotionBlur3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5)
        repr = "RandomMotionBlur3D(kernel_size=(3, 5), angle=tensor([[10., 30.],"\
            "\n        [10., 30.],\n        [10., 30.]]), direction=tensor([-0.5000,  0.5000]), " \
            "border_type='constant', p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    @pytest.mark.parametrize("same_on_batch", [True, False])
    @pytest.mark.parametrize("return_transform", [True, False])
    @pytest.mark.parametrize("p", [0., 1.])
    def test_random_motion_blur(self, same_on_batch, return_transform, p, device, dtype):
        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5,
                               same_on_batch=same_on_batch, return_transform=return_transform, p=p)
        input = torch.randn(1, 3, 5, 6, 7).repeat(2, 1, 1, 1, 1)

        output = f(input)

        if return_transform:
            assert len(output) == 2, f"must return a length 2 tuple if return_transform is True. Got {len(output)}."
            identity = kornia.eye_like(4, input)
            output, mat = output
            assert_allclose(mat, identity, rtol=1e-4, atol=1e-4)

        if same_on_batch:
            assert_allclose(output[0], output[1], rtol=1e-4, atol=1e-4)
        elif p == 0:
            assert_allclose(output, input, rtol=1e-4, atol=1e-4)
        else:
            assert not torch.allclose(output[0], output[1], rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 11, 7)).to(device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        params = {
            'batch_prob': torch.tensor([True]),
            'ksize_factor': torch.tensor([31]),
            'angle_factor': torch.tensor([[30., 30., 30.]]),
            'direction_factor': torch.tensor([-0.5]),
            'border_type': torch.tensor([0]),
        }
        assert gradcheck(RandomMotionBlur3D(
            kernel_size=3, angle=(10, 30), direction=(-0.5, 0.5), p=1.0), (inp, params), raise_exception=True)
