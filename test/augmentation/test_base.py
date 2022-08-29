from unittest.mock import patch

import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.augmentation import RandomGaussianBlur
from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.augmentation.base import _BasicAugmentationBase
from kornia.testing import assert_close


class TestBasicAugmentationBase:
    def test_smoke(self):
        base = _BasicAugmentationBase(p=0.5, p_batch=1.0, same_on_batch=True)
        __repr__ = "_BasicAugmentationBase(p=0.5, p_batch=1.0, same_on_batch=True)"
        assert str(base) == __repr__

    def test_infer_input(self, device, dtype):
        input = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        augmentation = _BasicAugmentationBase(p=1.0, p_batch=1)
        with patch.object(augmentation, "transform_tensor", autospec=True) as transform_tensor:
            transform_tensor.side_effect = lambda x: x.unsqueeze(dim=2)
            output = augmentation.transform_tensor(input)
            assert output.shape == torch.Size([2, 3, 1, 4, 5])
            assert_close(input, output[:, :, 0, :, :])

    @pytest.mark.parametrize(
        "p,p_batch,same_on_batch,num,seed",
        [
            (1.0, 1.0, False, 12, 1),
            (1.0, 0.0, False, 0, 1),
            (0.0, 1.0, False, 0, 1),
            (0.0, 0.0, False, 0, 1),
            (0.5, 0.1, False, 7, 3),
            (0.5, 0.1, True, 12, 3),
            (0.3, 1.0, False, 2, 1),
            (0.3, 1.0, True, 0, 1),
        ],
    )
    def test_forward_params(self, p, p_batch, same_on_batch, num, seed, device, dtype):
        input_shape = (12,)
        torch.manual_seed(seed)
        augmentation = _BasicAugmentationBase(p, p_batch, same_on_batch)
        with patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters:
            generate_parameters.side_effect = lambda shape: {
                'degrees': torch.arange(0, shape[0], device=device, dtype=dtype)
            }
            output = augmentation.forward_parameters(input_shape)
            assert "batch_prob" in output
            assert len(output['degrees']) == output['batch_prob'].sum().item() == num

    @pytest.mark.parametrize('keepdim', (True, False))
    def test_forward(self, device, dtype, keepdim):
        torch.manual_seed(42)
        input = torch.rand((12, 3, 4, 5), device=device, dtype=dtype)
        expected_output = input[..., :2, :2] if keepdim else input.unsqueeze(dim=0)[..., :2, :2]
        augmentation = _BasicAugmentationBase(p=0.3, p_batch=1.0, keepdim=keepdim)
        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, patch.object(
            augmentation, "generate_parameters", autospec=True
        ) as generate_parameters, patch.object(
            augmentation, "transform_tensor", autospec=True
        ) as transform_tensor, patch.object(
            augmentation, "transform_output_tensor", autospec=True
        ) as transform_output_tensor:

            generate_parameters.side_effect = lambda shape: {
                'degrees': torch.arange(0, shape[0], device=device, dtype=dtype)
            }
            transform_tensor.side_effect = lambda x: x.unsqueeze(dim=0)
            transform_output_tensor.side_effect = lambda x, y: x.squeeze()
            apply_transform.side_effect = lambda input, params, flags: input[..., :2, :2]
            # check_batching.side_effect = lambda input: None
            output = augmentation(input)
            assert output.shape == expected_output.shape
            assert_close(output, expected_output)

    @pytest.mark.parametrize("seed, expected_batch_prob", [[0, [True, True]], [1, [False, True]]])
    def test_autocast(self, seed, expected_batch_prob, device, dtype):
        # seed=0: triggers all data to be augmented
        # seed=1: triggers part of the dta to be augmented
        torch.manual_seed(seed)

        aug = RandomGaussianBlur((3, 3), (0.1, 3), p=0.5)
        x = torch.rand(2, 3, 100, 100, dtype=torch.float32).to(device)

        # Check that the seed behavior is correct
        params = aug.forward_parameters(x.shape)
        expected_batch_prob = torch.tensor(expected_batch_prob, device=params["batch_prob"].device)
        assert torch.all(params["batch_prob"] == expected_batch_prob)

        with torch.autocast(device.type):
            res = aug(x)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class TestAugmentationBase2D:
    @pytest.mark.parametrize(
        'input_shape, in_trans_shape',
        [
            ((2, 3, 4, 5), (2, 3, 3)),
            ((3, 4, 5), (3, 3)),
            ((4, 5), (3, 3)),
            pytest.param((1, 2, 3, 4, 5), (2, 3, 3), marks=pytest.mark.xfail),
            pytest.param((2, 3, 4, 5), (1, 3, 3), marks=pytest.mark.xfail),
            pytest.param((2, 3, 4, 5), (3, 3), marks=pytest.mark.xfail),
        ],
    )
    def test_check_batching(self, device, dtype, input_shape, in_trans_shape):
        input = torch.rand(input_shape, device=device, dtype=dtype)
        in_trans = torch.rand(in_trans_shape, device=device, dtype=dtype)
        augmentation = AugmentationBase2D(p=1.0, p_batch=1)
        augmentation.__check_batching__(input)
        augmentation.__check_batching__((input, in_trans))

    def test_forward(self, device, dtype):
        torch.manual_seed(42)
        input = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        # input_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        expected_output = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        expected_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        augmentation = AugmentationBase2D(p=1.0)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, patch.object(
            augmentation, "generate_parameters", autospec=True
        ) as generate_parameters, patch.object(
            augmentation, "compute_transformation", autospec=True
        ) as compute_transformation:

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
            output = augmentation(input)
            assert output is expected_output

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
            # expected_final_transformation = expected_transform @ input_transform
            # output = augmentation((input, input_transform))
            # assert output is expected_output

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

        input_param = {'batch_prob': torch.tensor([True]), 'x': input_transform, 'y': {}}

        augmentation = AugmentationBase2D(p=1.0)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, patch.object(
            augmentation, "compute_transformation", autospec=True
        ) as compute_transformation:

            apply_transform.return_value = output
            compute_transformation.return_value = other_transform
            assert gradcheck(augmentation, ((input, input_param)), raise_exception=True)
