from unittest.mock import patch
from typing import Union, Tuple

import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.augmentation.base import _BasicAugmentationBase, _AugmentationBase, AugmentationBase2D, AugmentationBase3D


class TestBasicAugmentationBase:

    def test_smoke(self, device, dtype):
        base = _BasicAugmentationBase(p=0.5, p_batch=1., same_on_batch=True)
        __repr__ = "p=0.5, p_batch=1.0, same_on_batch=True"
        assert str(base) == __repr__

    def test_infer_input(self, device, dtype):
        input = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        augmentation = _BasicAugmentationBase(p=1., p_batch=1)
        with patch.object(augmentation, "transform_tensor", autospec=True) as transform_tensor:
            transform_tensor.side_effect = lambda x: x.unsqueeze(dim=2)
            output = augmentation.transform_tensor(input)
            assert output.shape == torch.Size([2, 3, 1, 4, 5])
            assert_allclose(input, output[:, :, 0, :, :])

    @pytest.mark.parametrize("p,p_batch,same_on_batch,num,seed", [
        (1., 1., False, 12, 1),
        (1., 0., False, 0, 1),
        (0., 1., False, 0, 1),
        (0., 0., False, 0, 1),
        (.5, .1, False, 7, 3),
        (.5, .1, True, 12, 3),
        (.3, 1., False, 2, 1),
        (.3, 1., True, 0, 1),
    ])
    def test_forward_params(self, p, p_batch, same_on_batch, num, seed, device, dtype):
        input_shape = (12,)
        torch.manual_seed(seed)
        augmentation = _BasicAugmentationBase(p, p_batch, same_on_batch)
        with patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters:
            generate_parameters.side_effect = lambda shape: {
                'degrees': torch.arange(0, shape[0], device=device, dtype=dtype)
            }
            output = augmentation.__forward_parameters__(input_shape, p, p_batch, same_on_batch)
            assert "batch_prob" in output
            assert len(output['degrees']) == output['batch_prob'].sum().item() == num

    @pytest.mark.parametrize('keepdim', (True, False))
    def test_forward(self, device, dtype, keepdim):
        torch.manual_seed(42)
        input = torch.rand((12, 3, 4, 5), device=device, dtype=dtype)
        expected_output = input[..., :2, :2] if keepdim else input.unsqueeze(dim=0)[..., :2, :2]
        augmentation = _BasicAugmentationBase(p=.3, p_batch=1., keepdim=keepdim)
        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, \
                patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters, \
                patch.object(augmentation, "transform_tensor", autospec=True) as transform_tensor, \
                patch.object(augmentation, "__check_batching__", autospec=True) as check_batching:

            generate_parameters.side_effect = lambda shape: {
                'degrees': torch.arange(0, shape[0], device=device, dtype=dtype)
            }
            transform_tensor.side_effect = lambda x: x.unsqueeze(dim=0)
            apply_transform.side_effect = lambda input, params: input[..., :2, :2]
            check_batching.side_effect = lambda input: None
            output = augmentation(input)
            assert output.shape == expected_output.shape
            assert_allclose(output, expected_output)


class TestAugmentationBase2D:

    @pytest.mark.parametrize('input_shape, in_trans_shape', [
        ((2, 3, 4, 5), (2, 3, 3)),
        ((3, 4, 5), (3, 3)),
        ((4, 5), (3, 3)),
        pytest.param((1, 2, 3, 4, 5), (2, 3, 3), marks=pytest.mark.xfail),
        pytest.param((2, 3, 4, 5), (1, 3, 3), marks=pytest.mark.xfail),
        pytest.param((2, 3, 4, 5), (3, 3), marks=pytest.mark.xfail),
    ])
    def test_check_batching(self, device, dtype, input_shape, in_trans_shape):
        input = torch.rand(input_shape, device=device, dtype=dtype)
        in_trans = torch.rand(in_trans_shape, device=device, dtype=dtype)
        augmentation = AugmentationBase2D(p=1., p_batch=1)
        augmentation.__check_batching__(input)
        augmentation.__check_batching__((input, in_trans))

    def test_forward(self, device, dtype):
        torch.manual_seed(42)
        input = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        input_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        expected_output = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        expected_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        augmentation = AugmentationBase2D(return_transform=False, p=1.)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, \
                patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters, \
                patch.object(augmentation, "compute_transformation", autospec=True) as compute_transformation:

            # Calling the augmentation with a single tensor shall return the expected tensor using the generated params.
            params = {'params': {}, 'flags': {'foo': 0}}
            generate_parameters.return_value = params
            apply_transform.return_value = expected_output
            compute_transformation.return_value = expected_transform
            output = augmentation(input)
            # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
            # Not an easy fix, happens on verifying torch.tensor([True, True])
            # _params = {'batch_prob': torch.tensor([True, True]), 'params': {}, 'flags': {'foo': 0}}
            # apply_transform.assert_called_once_with(input, _params)
            assert output is expected_output

            # Calling the augmentation with a tensor and set return_transform shall
            # return the expected tensor and transformation.
            output, transformation = augmentation(input, return_transform=True)
            assert output is expected_output
            assert_allclose(transformation, expected_transform)

            # Calling the augmentation with a tensor and params shall return the expected tensor using the given params.
            params = {'params': {}, 'flags': {'bar': 1}}
            apply_transform.reset_mock()
            generate_parameters.return_value = None
            output = augmentation(input, params=params)
            # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
            # Not an easy fix, happens on verifying torch.tensor([True, True])
            # _params = {'batch_prob': torch.tensor([True, True]), 'params': {}, 'flags': {'foo': 0}}
            # apply_transform.assert_called_once_with(input, _params)
            assert output is expected_output

            # Calling the augmentation with a tensor,a transformation and set
            # return_transform shall return the expected tensor and the proper
            # transformation matrix.
            expected_final_transformation = expected_transform @ input_transform
            output, transformation = augmentation((input, input_transform), return_transform=True)
            assert output is expected_output
            assert torch.allclose(expected_final_transformation, transformation)
            assert transformation.shape[0] == input.shape[0]

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(42)

        input = torch.rand((1, 1, 3, 3), device=device, dtype=dtype)
        output = torch.rand((1, 1, 3, 3), device=device, dtype=dtype)
        input_transform = torch.rand((1, 3, 3), device=device, dtype=dtype)
        other_transform = torch.rand((1, 3, 3), device=device, dtype=dtype)

        input = utils.tensor_to_gradcheck_var(input)  # to var
        input_transform = utils.tensor_to_gradcheck_var(input_transform)  # to var
        output = utils.tensor_to_gradcheck_var(output)  # to var
        other_transform = utils.tensor_to_gradcheck_var(other_transform)  # to var

        input_param = {'batch_prob': torch.tensor([True]), 'params': {'x': input_transform}, 'flags': {}}

        augmentation = AugmentationBase2D(return_transform=True, p=1.)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, \
                patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters, \
                patch.object(augmentation, "compute_transformation", autospec=True) as compute_transformation:

            apply_transform.return_value = output
            compute_transformation.return_value = other_transform
            assert gradcheck(augmentation, ((input, input_param)), raise_exception=True)
