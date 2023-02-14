from unittest.mock import patch

import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.augmentation._2d.geometric.affine import RandomAffine
from kornia.augmentation._2d.intensity.gaussian_blur import RandomGaussianBlur
from kornia.augmentation._3d.geometric.affine import RandomAffine3D
from kornia.augmentation._3d.intensity.motion_blur import RandomMotionBlur3D
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
            assert len(output['degrees']) == output['batch_prob'].round().sum().item() == num

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


class TestAugmentationBase2D:
    def test_forward(self, device, dtype):
        torch.manual_seed(42)
        input = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        # input_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        expected_output = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        augmentation = AugmentationBase2D(p=1.0)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, patch.object(
            augmentation, "generate_parameters", autospec=True
        ) as generate_parameters:
            # Calling the augmentation with a single tensor shall return the expected tensor using the generated params.
            params = {'params': {}, 'flags': {'foo': 0}}
            generate_parameters.return_value = params
            apply_transform.return_value = expected_output
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

        input_param = {'batch_prob': torch.tensor([1.]), 'x': input_transform, 'y': {}}

        augmentation = AugmentationBase2D(p=1.0)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform:
            apply_transform.return_value = output
            assert gradcheck(augmentation, ((input, input_param)), raise_exception=True, fast_mode=True)


class TestGeometricAugmentationBase2D:
    @pytest.mark.parametrize("batch_prob", [[1., 1.], [0., 1.], [0., 0.]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        # Uses some subclass of `GeometricAugmentationBase2D` which perform some op which can mismatch the dtype
        # Will cover AugmentationBase2D and RigidAffineAugmentationBase2D too
        aug = RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0)
        x = torch.rand(len(batch_prob), 5, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, '__batch_prob_generator__', return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class TestIntensityAugmentationBase2D:
    @pytest.mark.parametrize("batch_prob", [[1., 1.], [0., 1.], [0., 0.]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        # Uses some subclass of `IntensityAugmentationBase2D` which perform some op which can mismatch the dtype
        # Will cover AugmentationBase2D and RigidAffineAugmentationBase2D too
        aug = RandomGaussianBlur((3, 3), (0.1, 3), p=1)
        x = torch.rand(len(batch_prob), 5, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, '__batch_prob_generator__', return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class TestIntensityAugmentationBase3D:
    @pytest.mark.parametrize("batch_prob", [[1., 1.], [0., 1.], [0., 0.]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        # Uses some subclass of `IntensityAugmentationBase3D` which perform some op which can mismatch the dtype
        # Will cover RigidAffineAugmentationBase3D and AugmentationBase3D too
        aug = RandomMotionBlur3D(3, 35.0, 0.5, p=1)
        x = torch.rand(len(batch_prob), 1, 3, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, '__batch_prob_generator__', return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class TestGeometricAugmentationBase3D:
    @pytest.mark.parametrize("batch_prob", [[1., 1.], [0., 1.], [0., 0.]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        # Uses some subclass of `GeometricAugmentationBase3D` which perform some op which can mismatch the dtype
        # Will cover RigidAffineAugmentationBase3D and AugmentationBase3D too
        aug = RandomAffine3D((15.0, 20.0, 20.0), p=1)
        x = torch.rand(len(batch_prob), 1, 3, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, '__batch_prob_generator__', return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"
