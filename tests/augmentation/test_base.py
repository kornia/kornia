# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import contextlib
import copy
import io
import pickle
import threading
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.augmentation._2d.geometric.affine import RandomAffine
from kornia.augmentation._2d.geometric.horizontal_flip import RandomHorizontalFlip
from kornia.augmentation._2d.intensity.gaussian_blur import RandomGaussianBlur
from kornia.augmentation._2d.intensity.invert import RandomInvert
from kornia.augmentation._2d.mix.mixup import RandomMixUpV2
from kornia.augmentation._3d.geometric.affine import RandomAffine3D
from kornia.augmentation._3d.geometric.depthical_flip import RandomDepthicalFlip3D
from kornia.augmentation._3d.geometric.horizontal_flip import RandomHorizontalFlip3D
from kornia.augmentation._3d.intensity.motion_blur import RandomMotionBlur3D
from kornia.augmentation.base import _BasicAugmentationBase
from kornia.augmentation.random_generator.base import UniformDistribution
from kornia.core import ImageModule
from kornia.core._compat import torch_version_lt
from kornia.core.exceptions import InputRankError, ShapeError
from kornia.filters.dissolving import StableDiffusionDissolving, _DissolvingWraper_HF
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from testing.base import (
    DYNAMO_UNAVAILABLE_REASON,
    BaseTester,
    dynamo_is_available,
    supports_bilinear_2d_grid_sample_backward,
    supports_reflect_padding,
)


class TestBasicAugmentationBase(BaseTester):
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
            self.assert_close(input, output[:, :, 0, :, :])

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
                "degrees": torch.arange(0, shape[0], device=device, dtype=dtype)
            }
            output = augmentation.forward_parameters(input_shape)
            assert "batch_prob" in output
            # generate_parameters is now called with the full batch shape (ONNX-friendly contract).
            assert len(output["degrees"]) == input_shape[0]
            assert output["batch_prob"].sum().item() == num

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_forward(self, device, dtype, keepdim):
        torch.manual_seed(42)
        input = torch.rand((12, 3, 4, 5), device=device, dtype=dtype)
        expected_output = input[..., :2, :2] if keepdim else input.unsqueeze(dim=0)[..., :2, :2]
        augmentation = _BasicAugmentationBase(p=0.3, p_batch=1.0, keepdim=keepdim)
        with (
            patch.object(augmentation, "apply_transform", autospec=True) as apply_transform,
            patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters,
            patch.object(augmentation, "transform_tensor", autospec=True) as transform_tensor,
            patch.object(augmentation, "transform_output_tensor", autospec=True) as transform_output_tensor,
        ):
            generate_parameters.side_effect = lambda shape: {
                "degrees": torch.arange(0, shape[0], device=device, dtype=dtype)
            }
            transform_tensor.side_effect = lambda x: x.unsqueeze(dim=0)
            transform_output_tensor.side_effect = lambda x, y: x.squeeze()
            apply_transform.side_effect = lambda input, params, flags: input[..., :2, :2]
            # check_batching.side_effect = lambda input: None
            output = augmentation(input)
            assert output.shape == expected_output.shape
            self.assert_close(output, expected_output)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_deterministic_p_skips_bernoulli(self, p):
        """When p is 0 or 1 the outcome is deterministic — no Bernoulli sampler should be created."""
        base = _BasicAugmentationBase(p=p, p_batch=0.5)
        assert not isinstance(getattr(base, "_p_gen", None), torch.distributions.Bernoulli)

    @pytest.mark.parametrize("p_batch", [0.0, 1.0])
    def test_deterministic_p_batch_skips_bernoulli(self, p_batch):
        """When p_batch is 0 or 1 the outcome is deterministic — no Bernoulli sampler should be created."""
        base = _BasicAugmentationBase(p=0.5, p_batch=p_batch)
        assert not isinstance(getattr(base, "_p_batch_gen", None), torch.distributions.Bernoulli)


class TestAugmentationBase2D(BaseTester):
    def test_forward(self, device, dtype):
        torch.manual_seed(42)
        input = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        # input_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        expected_output = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        augmentation = AugmentationBase2D(p=1.0)

        with (
            patch.object(augmentation, "apply_transform", autospec=True) as apply_transform,
            patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters,
        ):
            # Calling the augmentation with a single tensor shall return the expected tensor using the generated params.
            params = {"params": {}, "flags": {"foo": 0}}
            generate_parameters.return_value = params
            apply_transform.return_value = expected_output
            output = augmentation(input)
            # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
            # Not an easy fix, happens on verifying torch.tensor([True, True])
            # _params = {'batch_prob': torch.tensor([True, True]), 'params': {}, 'flags': {'foo': 0}}
            # apply_transform.assert_called_once_with(input, _params)
            # Identity check relaxed to value equality: the where-blend always materialises
            # a fresh tensor, so output is never the same object as apply_transform's return.
            assert torch.equal(output, expected_output)

            # Calling the augmentation with a tensor and set return_transform shall
            # return the expected tensor and transformation.
            output = augmentation(input)
            # Identity check relaxed to value equality: the where-blend always materialises
            # a fresh tensor, so output is never the same object as apply_transform's return.
            assert torch.equal(output, expected_output)

            # Calling the augmentation with a tensor and params shall return the expected tensor using the given params.
            params = {"params": {}, "flags": {"bar": 1}}
            apply_transform.reset_mock()
            generate_parameters.return_value = None
            output = augmentation(input, params=params)
            # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
            # Not an easy fix, happens on verifying torch.tensor([True, True])
            # _params = {'batch_prob': torch.tensor([True, True]), 'params': {}, 'flags': {'foo': 0}}
            # apply_transform.assert_called_once_with(input, _params)
            # Identity check relaxed to value equality: the where-blend always materialises
            # a fresh tensor, so output is never the same object as apply_transform's return.
            assert torch.equal(output, expected_output)

            # Calling the augmentation with a tensor,a transformation and set
            # return_transform shall return the expected tensor and the proper
            # transformation matrix.
            # expected_final_transformation = expected_transform @ input_transform
            # output = augmentation((input, input_transform))
            # assert output is expected_output

    def test_gradcheck(self, device):
        torch.manual_seed(42)

        input = torch.rand((1, 1, 3, 3), device=device, dtype=torch.float64)
        output = torch.rand((1, 1, 3, 3), device=device, dtype=torch.float64)
        input_transform = torch.rand((1, 3, 3), device=device, dtype=torch.float64)

        input_param = {"batch_prob": torch.tensor([True]), "x": input_transform, "y": {}}

        augmentation = AugmentationBase2D(p=1.0)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform:
            apply_transform.return_value = output
            self.gradcheck(augmentation, ((input, input_param)))


class TestAugmentationPartialTo(BaseTester):
    @pytest.mark.parametrize("generator_only", [False, True])
    def test_invalid_dtype_preserves_samplers(self, device, dtype, generator_only):
        aug = K.RandomAffine(30.0, p=1.0).to(device=device, dtype=dtype)
        generator = aug._param_generator
        module = generator if generator_only else aug
        sampler = generator.degree_sampler
        with pytest.raises(TypeError, match="only accepts floating point or complex dtypes"):
            module.to(torch.int64)
        assert module.device == device
        assert module.dtype == dtype
        assert generator.degree_sampler is sampler
        assert generator.dtype == dtype
        assert generator((4, 3, 8, 9))["angle"].is_floating_point()

    @pytest.mark.parametrize("generator_only", [False, True])
    def test_move_builds_samplers_once(self, device, generator_only):
        aug = K.RandomAffine(30.0, p=1.0)
        generator = aug._param_generator
        module = generator if generator_only else aug
        # Use the unindexed device spelling to exercise CUDA's canonicalization to cuda:0.
        with patch.object(generator, "make_samplers", wraps=generator.make_samplers) as make_samplers:
            module.to(device.type, dtype=torch.float64 if device.type != "mps" else torch.float16)
        assert make_samplers.call_count == 1

    @pytest.mark.parametrize("move", ["to", "convenience", "container"])
    def test_generator_moves_buffers_and_samplers(self, device, move):
        generator = K.RandomRotation((10.0, 20.0))._param_generator
        target_dtype = torch.float16 if device.type == "mps" else torch.float64
        if move == "to":
            assert generator.to(device=device, dtype=target_dtype) is generator
        elif move == "convenience":
            if device.type in ("cpu", "cuda"):
                getattr(generator, device.type)()
            else:
                generator.to(device)
            generator.half() if target_dtype == torch.float16 else generator.double()
        else:
            torch.nn.Sequential(generator).to(device=device, dtype=target_dtype)
        assert generator.device == device
        assert generator.dtype == target_dtype
        assert generator.degrees.device == device
        assert generator.degrees.dtype == target_dtype
        assert generator.sampler_dict["degrees"].low.device == device
        assert generator.sampler_dict["degrees"].low.dtype == target_dtype

    @pytest.mark.parametrize("generator_only", [False, True])
    def test_dtype_only_to_preserves_device(self, device, dtype, generator_only):
        aug = K.RandomAffine(30.0, p=1.0)
        module = aug._param_generator if generator_only else aug
        module.to(device=device, dtype=torch.float32)
        module.to(dtype=dtype)
        assert module.device == device
        assert module.dtype == dtype
        generator = module if generator_only else module._param_generator
        assert generator.degree_sampler.low.device == device
        assert generator.degree_sampler.low.dtype == dtype

    @pytest.mark.parametrize("generator_only", [False, True])
    def test_device_only_to_preserves_dtype(self, device, dtype, generator_only):
        aug = K.RandomAffine(30.0, p=1.0)
        module = aug._param_generator if generator_only else aug
        module.to(dtype=dtype)
        module.to(device=device)
        assert module.device == device
        assert module.dtype == dtype
        generator = module if generator_only else module._param_generator
        assert generator.degree_sampler.low.device == device
        assert generator.degree_sampler.low.dtype == dtype


class TestGeometricAugmentationBase2D:
    @pytest.mark.parametrize("batch_prob", [[True, True], [False, True], [False, False]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        # Uses some subclass of `GeometricAugmentationBase2D` which perform some op which can mismatch the dtype
        # Will cover AugmentationBase2D and RigidAffineAugmentationBase2D too
        aug = RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0)
        x = torch.rand(len(batch_prob), 5, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, "__batch_prob_generator__", return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class TestIntensityAugmentationBase2D:
    @pytest.mark.parametrize("batch_prob", [[True, True], [False, True], [False, False]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        # Uses some subclass of `IntensityAugmentationBase2D` which perform some op which can mismatch the dtype
        # Will cover AugmentationBase2D and RigidAffineAugmentationBase2D too
        aug = RandomGaussianBlur((3, 3), (0.1, 3), p=1)
        x = torch.rand(len(batch_prob), 5, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, "__batch_prob_generator__", return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class TestIntensityAugmentationBase3D:
    @pytest.mark.parametrize("batch_prob", [[True, True], [False, True], [False, False]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")
        if device.type == "cpu" and torch_version_lt(2, 3, 0):
            pytest.skip("3D CPU autocast requires bfloat16 eye support from PyTorch 2.3")

        # Uses some subclass of `IntensityAugmentationBase3D` which perform some op which can mismatch the dtype
        # Will cover RigidAffineAugmentationBase3D and AugmentationBase3D too
        aug = RandomMotionBlur3D(3, 35.0, 0.5, p=1)
        x = torch.rand(len(batch_prob), 1, 3, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, "__batch_prob_generator__", return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class TestGeometricAugmentationBase3D:
    @pytest.mark.parametrize("batch_prob", [[True, True], [False, True], [False, False]])
    def test_autocast(self, batch_prob, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")
        if device.type == "cpu" and torch_version_lt(2, 3, 0):
            pytest.skip("3D CPU autocast requires bfloat16 eye support from PyTorch 2.3")

        # Uses some subclass of `GeometricAugmentationBase3D` which perform some op which can mismatch the dtype
        # Will cover RigidAffineAugmentationBase3D and AugmentationBase3D too
        aug = RandomAffine3D((15.0, 20.0, 20.0), p=1)
        x = torch.rand(len(batch_prob), 1, 3, 10, 7, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)
        with patch.object(aug, "__batch_prob_generator__", return_value=to_apply):
            params = aug.forward_parameters(x.shape)

        with torch.autocast(device.type):
            res = aug(x, params)

        assert res.dtype == dtype, "The output dtype should match the input dtype"


class _RandomDepthicalFlip3DWithMask(RandomDepthicalFlip3D):
    def apply_transform_mask(self, input, params, flags, transform=None):
        return self.apply_transform(input, params, flags, transform)

    def apply_non_transform_mask(self, input, params, flags, transform=None):
        return input


class TestTransformTensorByShape(BaseTester):
    # Each case gives the output shape for match_channel=False and for True; None means the channel check raises.
    @pytest.mark.parametrize("match_channel", [False, True])
    @pytest.mark.parametrize(
        "reference, input_shape, expected",
        [
            ((2, 3, 5, 6), (5, 6), ((1, 1, 5, 6), None)),
            ((1, 3, 5, 6), (5, 6), ((1, 1, 5, 6), None)),
            ((2, 3, 5, 6), (2, 5, 6), ((2, 1, 5, 6), None)),
            ((2, 3, 5, 6), (3, 5, 6), ((1, 3, 5, 6), (1, 3, 5, 6))),
            ((2, 3, 5, 6), (2, 3, 5, 6), ((2, 3, 5, 6), (2, 3, 5, 6))),
            ((2, 3, 4, 5, 6), (4, 5, 6), ((1, 1, 4, 5, 6), None)),
            ((1, 3, 4, 5, 6), (4, 5, 6), ((1, 1, 4, 5, 6), None)),
            ((2, 3, 4, 5, 6), (2, 4, 5, 6), ((2, 1, 4, 5, 6), None)),
            ((2, 3, 4, 5, 6), (3, 4, 5, 6), ((1, 3, 4, 5, 6), (1, 3, 4, 5, 6))),
            ((2, 3, 4, 5, 6), (2, 3, 4, 5, 6), ((2, 3, 4, 5, 6), (2, 3, 4, 5, 6))),
        ],
    )
    def test_shape_6354(self, device, dtype, reference, input_shape, expected, match_channel):
        augmentation = RandomHorizontalFlip(p=1.0) if len(reference) == 4 else RandomHorizontalFlip3D(p=1.0)
        input = torch.rand(input_shape, device=device, dtype=dtype)
        shape = torch.tensor(reference, device=device)
        expected = expected[match_channel]
        if expected is None:
            with pytest.raises(ValueError, match="C dimension"):
                augmentation.transform_tensor(input, shape=shape, match_channel=match_channel)
        else:
            assert augmentation.transform_tensor(input, shape=shape, match_channel=match_channel).shape == expected

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_3d_bdhw_mask_follows_the_volume_6354(self, device, dtype, keepdim):
        volume = torch.arange(2 * 4 * 5 * 6, device=device).to(dtype).reshape(2, 1, 4, 5, 6)
        augmentation = _RandomDepthicalFlip3DWithMask(p=1.0, keepdim=keepdim)
        output_volume = augmentation(volume)
        output_mask = augmentation.transform_masks(
            volume[:, 0], augmentation._params, augmentation.flags, transform=augmentation.transform_matrix
        )
        expected = output_volume[:, 0] if keepdim else output_volume
        self.assert_close(output_mask, expected, rtol=0, atol=0)


class TestDeviceAgnosticAugmentationParameters(BaseTester):
    """Augmentations keep RNG/params on CPU while accepting accelerator inputs.

    CPU cases provide smoke coverage; the cross-device regression is exercised on CUDA and MPS.
    """

    @staticmethod
    def _cpu_partial_batch_params(augmentation: _BasicAugmentationBase, input: torch.Tensor) -> dict[str, torch.Tensor]:
        params = augmentation.forward_parameters(input.shape)
        assert params["batch_prob"].device.type == "cpu"
        params["batch_prob"] = torch.tensor([True, False])
        return params

    def test_2d_augmentation_blends_cpu_params_with_accelerator_input(self, device, dtype):
        input = torch.arange(12, device=device, dtype=dtype).reshape(2, 1, 2, 3) / 12
        augmentation = RandomInvert(p=0.5)
        params = self._cpu_partial_batch_params(augmentation, input)

        output = augmentation(input, params=params)

        expected = torch.stack([1 - input[0], input[1]])
        assert output.device == input.device
        self.assert_close(output, expected)

    def test_2d_geometric_matrix_blends_cpu_params_with_accelerator_input(self, device, dtype):
        input = torch.arange(12, device=device, dtype=dtype).reshape(2, 1, 2, 3)
        augmentation = RandomHorizontalFlip(p=0.5)
        params = self._cpu_partial_batch_params(augmentation, input)

        output = augmentation(input, params=params)
        matrix = augmentation.transform_matrix

        expected = torch.stack([input[0].flip(-1), input[1]])
        assert output.device == input.device
        assert matrix is not None
        assert matrix.device == input.device
        self.assert_close(output, expected)
        self.assert_close(matrix[1], torch.eye(3, device=device, dtype=dtype))

    def test_3d_geometric_matrix_blends_cpu_params_with_accelerator_input(self, device, dtype):
        input = torch.arange(24, device=device, dtype=dtype).reshape(2, 1, 2, 2, 3)
        augmentation = RandomHorizontalFlip3D(p=0.5)
        params = self._cpu_partial_batch_params(augmentation, input)

        output = augmentation(input, params=params)
        matrix = augmentation.transform_matrix

        expected = torch.stack([input[0].flip(-1), input[1]])
        assert output.device == input.device
        assert matrix is not None
        assert matrix.device == input.device
        self.assert_close(output, expected)
        self.assert_close(matrix[1], torch.eye(4, device=device, dtype=dtype))

    def test_mix_augmentation_blends_cpu_params_with_accelerator_input(self, device, dtype):
        input = torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 2, 2)
        augmentation = RandomMixUpV2(lambda_val=(0.25, 0.25), p=1.0, data_keys=["input"])
        params = self._cpu_partial_batch_params(augmentation, input)
        params["mixup_pairs"] = torch.tensor([1, 0])

        output = augmentation(input, params=params)

        expected = torch.stack([input[0] * 0.75 + input[1] * 0.25, input[1]])
        assert output.device == input.device
        self.assert_close(output, expected)


# Pins for the shared `AugmentationBase2D` contract. Every literal below was generated by the body of the pin
# that carries it. Fixtures are asymmetric (H != W, a hot pixel off both centre lines) and every random draw is
# either seeded inside the test or made deterministic with `p=1.0` plus a point range such as
# `degrees=(45.0, 45.0)`.

# Representative constructible augmentations for the statelessness pin. Some other augmentations keep their
# sampling range in a `_param_generator.*` buffer; see `test_convention_param_generator_range_buffers_reach_the_draws`.
_STATELESS_REPRESENTATIVES = {
    "RandomHorizontalFlip": lambda: K.RandomHorizontalFlip(p=1.0),
    "RandomAffine": lambda: K.RandomAffine(degrees=(45.0, 45.0), p=1.0),
    "ColorJiggle": lambda: K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0),
    "Normalize": lambda: K.Normalize(0.5, 0.5, p=1.0),
    "AugmentationSequential": lambda: K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0)),
}


def _assert_params_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_params_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_params_equal(actual_value, expected_value)
    else:
        assert actual == expected


# name -> (constructor, batch shape, reference CPU RNG operations).
# Compare states against operations on the installed PyTorch rather than version-specific float literals.
# This pins consumption, not the actual random values or the assignment of values to parameter keys.
_RNG_FINGERPRINTS = {
    "RandomHorizontalFlip(p=1.0)": (lambda: K.RandomHorizontalFlip(p=1.0), (4, 3, 6, 8), []),
    "RandomHorizontalFlip(p=0.0)": (lambda: K.RandomHorizontalFlip(p=0.0), (4, 3, 6, 8), []),
    "RandomHorizontalFlip(p=0.5)": (lambda: K.RandomHorizontalFlip(p=0.5), (4, 3, 6, 8), [("rand", (4,))]),
    "CenterCrop": (lambda: K.CenterCrop((4, 6)), (4, 3, 6, 8), []),
    "Resize": (lambda: K.Resize((8, 12)), (4, 3, 6, 8), []),
    # The affine draws a gate, angle and two translations. Point ranges still consume random values.
    "RandomAffine": (
        lambda: K.RandomAffine(degrees=(30.0, 30.0), translate=(0.1, 0.1)),
        (4, 3, 6, 8),
        [("rand", (4,))] * 4,
    ),
    "RandomRotation": (lambda: K.RandomRotation(degrees=(30.0, 30.0)), (4, 3, 6, 8), [("rand", (4,))] * 2),
    "RandomCrop": (lambda: K.RandomCrop((4, 6)), (4, 3, 6, 8), [("rand", (4,))] * 2),
    "RandomPerspective": (lambda: K.RandomPerspective(0.5), (4, 3, 6, 8), [("rand", (4,)), ("rand", (4, 4, 2))]),
    "RandomElasticTransform": (K.RandomElasticTransform, (4, 3, 6, 8), [("rand", (4,)), ("rand", (4, 2, 6, 8))]),
    "ColorJiggle": (
        lambda: K.ColorJiggle(0.2, 0.2, 0.2, 0.1),
        (4, 3, 6, 8),
        [("rand", (4,))] * 4 + [("randperm", (4,))],
    ),
    "RandomGaussianNoise": (
        lambda: K.RandomGaussianNoise(std=0.1),
        (4, 3, 6, 8),
        [("rand", (4,)), ("randn", (4, 3, 6, 8))],
    ),
    "AugmentationSequential": (
        lambda: K.AugmentationSequential(K.RandomAffine(degrees=(10.0, 90.0), p=1.0)),
        (4, 3, 6, 8),
        [("rand", (4,))],
    ),
    "ImageSequential": (
        lambda: K.ImageSequential(K.RandomAffine(degrees=(10.0, 90.0), p=1.0)),
        (4, 3, 6, 8),
        [("rand", (4,))],
    ),
    "VideoSequential": (
        lambda: K.VideoSequential(K.RandomAffine(degrees=(10.0, 90.0), p=1.0)),
        (4, 2, 3, 6, 8),
        [("rand", (4,))],
    ),
    "PatchSequential": (
        lambda: K.PatchSequential(K.RandomAffine(degrees=(10.0, 90.0), p=1.0), grid_size=(2, 2), patchwise_apply=False),
        (4, 3, 6, 8),
        # One draw per patch row: B * rows * columns = 4 * 2 * 2 (#4421).
        [("rand", (16,))],
    ),
}


@pytest.mark.usefixtures("restore_torch_rng")
class TestConventionAugmentationBase2D(BaseTester):
    """Pins for the contract every 2D augmentation inherits from ``AugmentationBase2D``."""

    def test_convention_unbatched_input_is_promoted_unless_keepdim(self, device, dtype):
        # Convention pin: an augmentation always works on (B, C, H, W). A (C, H, W) input is promoted to
        # (1, C, H, W) and a bare (H, W) to (1, 1, H, W); `keepdim=True` restores the input rank on the way
        # out and never drops a real batch dimension.
        x = torch.rand(3, 6, 8, device=device, dtype=dtype)
        assert K.RandomHorizontalFlip(p=1.0)(x).shape == (1, 3, 6, 8)
        assert K.RandomHorizontalFlip(p=1.0, keepdim=True)(x).shape == (3, 6, 8)
        assert K.RandomHorizontalFlip(p=1.0)(torch.rand(6, 8, device=device, dtype=dtype)).shape == (1, 1, 6, 8)
        batched = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        assert K.RandomHorizontalFlip(p=1.0, keepdim=True)(batched).shape == (2, 3, 6, 8)
        one_batched = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
        assert K.RandomHorizontalFlip(p=1.0, keepdim=True)(one_batched).shape == (1, 3, 6, 8)

    def test_convention_integer_input_is_rejected(self, device):
        # Convention pin: the dtype policy is enforced - an integer image raises `TypeError` naming the four
        # accepted float dtypes, for `uint8` and `int64` alike.
        expected = r"Expected input of \[torch.bfloat16, torch.float16, torch.float32, torch.float64\]"
        for bad in (torch.uint8, torch.int64):
            with pytest.raises(TypeError, match=expected):
                K.RandomHorizontalFlip(p=1.0)(torch.zeros(1, 1, 4, 4, device=device, dtype=bad))

    def test_convention_wrong_input_rank_raises_one_error_from_every_entry_point_4424(self, device, dtype):
        # Convention pin (#4424): one user error, one exception and one message, whichever entry point sees it.
        # The class, `validate_tensor` and the containers used to raise `ValueError`, `RuntimeError` or
        # `AssertionError` with four different messages; `InputRankError` is all three types, so a caller's
        # existing `except` clause still catches it.
        x5 = torch.rand(1, 2, 3, 6, 8, device=device, dtype=dtype)
        message = r"Input size must have a shape of either \(H, W\), \(C, H, W\) or \(\*, C, H, W\)\. Got"
        entry_points = {
            "class": lambda: K.RandomHorizontalFlip(p=1.0)(x5),
            "validate_tensor": lambda: K.RandomHorizontalFlip(p=1.0).validate_tensor(x5),
            "AugmentationSequential": lambda: K.AugmentationSequential(
                K.RandomHorizontalFlip(p=1.0), data_keys=["input"]
            )(x5),
            "ImageSequential": lambda: K.ImageSequential(K.RandomHorizontalFlip(p=1.0))(x5),
        }
        for name, call in entry_points.items():
            with pytest.raises(InputRankError, match=message) as info:
                call()
            for legacy_type in (ValueError, RuntimeError, AssertionError, ShapeError):
                assert isinstance(info.value, legacy_type), (name, legacy_type)
            assert info.value.actual_shape == tuple(x5.shape), name

    def test_convention_validate_tensor_accepts_the_ranks_forward_accepts_4424(self, device, dtype):
        # `forward` promotes (H, W) and (C, H, W); `validate_tensor` used to reject both (#4424).
        aug = K.RandomHorizontalFlip(p=1.0)
        for shape in ((6, 8), (3, 6, 8), (2, 3, 6, 8)):
            x = torch.rand(*shape, device=device, dtype=dtype)
            aug.validate_tensor(x)
            assert aug(x).shape[-2:] == (6, 8)
        with pytest.raises(InputRankError):
            aug.validate_tensor(torch.rand(8, device=device, dtype=dtype))

    def test_convention_wrong_input_rank_in_patch_video_and_3d_entry_points_4424(self, device, dtype):
        flips = [K.RandomHorizontalFlip(p=1.0) for _ in range(4)]
        with pytest.raises(InputRankError, match=r"must have a shape of \(B, C, H, W\)\. Got"):
            K.PatchSequential(*flips, grid_size=(2, 2))(torch.rand(1, 2, 3, 8, 8, device=device, dtype=dtype))
        with pytest.raises(InputRankError, match=r"must have a shape of \(B, T, C, H, W\)\. Got") as info:
            K.VideoSequential(K.RandomHorizontalFlip(p=1.0))(torch.rand(2, 3, 6, 8, device=device, dtype=dtype))
        assert isinstance(info.value, AssertionError)
        volume_message = r"must have a shape of either \(D, H, W\), \(C, D, H, W\) or \(\*, C, D, H, W\)\. Got"
        x6 = torch.rand(1, 1, 2, 3, 4, 5, device=device, dtype=dtype)
        with pytest.raises(InputRankError, match=volume_message):
            RandomHorizontalFlip3D(p=1.0)(x6)
        with pytest.raises(InputRankError, match=volume_message):
            RandomHorizontalFlip3D(p=1.0).validate_tensor(x6)
        RandomHorizontalFlip3D(p=1.0).validate_tensor(torch.rand(3, 4, 5, device=device, dtype=dtype))

    def test_convention_wrong_input_rank_in_augmentation_sequential_video_and_3d_4424(self, device, dtype):
        # Only the 5D layout is listed: a rank-3 or rank-4 video does not run through the container.
        video = K.AugmentationSequential(K.VideoSequential(K.RandomHorizontalFlip(p=1.0), data_format="BCTHW"))
        with pytest.raises(InputRankError, match=r"must have a shape of \(B, C, T, H, W\)\. Got"):
            video(torch.rand(1, 1, 2, 3, 6, 8, device=device, dtype=dtype))
        # Rank 4 is ambiguous next to a 3D child, so no message lists (C, D, H, W).
        volume = K.AugmentationSequential(RandomHorizontalFlip3D(p=1.0), data_keys=["input"])
        for shape in ((2, 3, 5, 6), (5, 6), (1, 2, 1, 3, 5, 6)):
            with pytest.raises(InputRankError, match=r"either \(D, H, W\) or \(B, C, D, H, W\)\. Got") as info:
                volume(torch.rand(*shape, device=device, dtype=dtype))
            assert isinstance(info.value, RuntimeError)

    def test_convention_numeric_ranges_return_cpu_parameters_independently_of_input(self, device, dtype):
        # With numeric ranges and unchanged CPU defaults, returned parameter placement is independent of
        # the image. This pin checks returned tensors, not the sampling backend/precision (covered below).
        aug = K.RandomAffine(degrees=(45.0, 45.0), translate=(0.2, 0.2), p=1.0)
        out = aug(torch.rand(2, 3, 6, 8, device=device, dtype=dtype))
        assert aug._params["angle"].dtype == torch.get_default_dtype()
        assert all(v.device.type == "cpu" for v in aug._params.values())
        assert aug.transform_matrix.dtype == dtype
        assert out.dtype == dtype

    def test_convention_p_is_per_sample_and_p_batch_gates_the_whole_batch(self):
        # Convention pin: `p` is a per-sample Bernoulli and `p_batch` a single Bernoulli that gates the whole
        # batch; a sample is augmented only when both fire. Drawn on the cpu generator, so no device/dtype
        # fixture is involved; seeded inside the test.
        flip = K.RandomHorizontalFlip
        assert flip(p=1.0, p_batch=0.0).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist() == [0.0] * 4
        assert flip(p=0.0, p_batch=1.0).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist() == [0.0] * 4
        assert flip(p=1.0, p_batch=1.0).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist() == [1.0] * 4
        batch_rows = []
        for seed in range(8):
            torch.manual_seed(seed)
            batch_rows.append(flip(p=1.0, p_batch=0.5).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist())
        assert all(len(set(row)) == 1 for row in batch_rows)
        assert {row[0] for row in batch_rows} == {0.0, 1.0}
        # `p=0.5` is per sample: mixed rows exist.
        sample_rows = []
        for seed in range(4):
            torch.manual_seed(seed)
            sample_rows.append(flip(p=0.5).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist())
        assert any(len(set(row)) > 1 for row in sample_rows)

    def test_convention_p_batch_draw_precedes_the_per_sample_gate(self):
        torch.manual_seed(3)
        actual = K.RandomHorizontalFlip(p=0.5, p_batch=0.5).forward_parameters((4, 3, 6, 8))["batch_prob"]
        actual_state = torch.random.get_rng_state()
        torch.manual_seed(3)
        batch_gate = torch.rand(1) < 0.5
        per_sample_gate = torch.rand(4) < 0.5
        expected = (batch_gate * per_sample_gate).to(actual.dtype)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_state, torch.random.get_rng_state())

    @pytest.mark.parametrize("augmentation_cls", [K.RandomJigsaw, K.RandomMosaic])
    def test_convention_jigsaw_and_mosaic_use_p_per_sample(self, augmentation_cls):
        rows = []
        for seed in range(4):
            torch.manual_seed(seed)
            rows.append(augmentation_cls(p=0.5).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist())
        assert any(len(set(row)) > 1 for row in rows)

    @pytest.mark.parametrize("augmentation_cls", [K.RandomMixUpV2, K.RandomCutMixV2, K.PatchMix])
    def test_convention_mixup_cutmix_and_patchmix_map_p_to_a_batch_gate_4425(self, augmentation_cls):
        # Their constructors map `p` to the mix base's batch-wide probability, unlike Jigsaw and Mosaic.
        rows = []
        for seed in range(8):
            torch.manual_seed(seed)
            kwargs: dict = {"use_correct_lambda": True} if augmentation_cls is K.RandomCutMixV2 else {}
            # PatchMix's default patch_size of 16 does not fit this 6x8 input.
            if augmentation_cls is K.PatchMix:
                kwargs = {"patch_size": 4}
            params = augmentation_cls(p=0.5, **kwargs).forward_parameters((4, 3, 6, 8))
            rows.append(params["batch_prob"].tolist())
        assert all(len(set(row)) == 1 for row in rows)
        assert {row[0] for row in rows} == {0.0, 1.0}

    @pytest.mark.parametrize(
        ("factory", "shape"),
        [
            (lambda: K.RandomAffine(30.0, p=1.0), (2, 3, 6, 8)),
            (lambda: K.RandomRotation3D(30.0, p=1.0), (2, 1, 4, 6, 8)),
        ],
    )
    def test_convention_p_batch_attribute_gates_any_augmentation_4425(self, factory, shape, device, dtype):
        aug = factory()
        aug.p_batch = 0.0
        x = torch.rand(shape, device=device, dtype=dtype)
        out = aug(x)
        assert torch.equal(out, x)
        assert not aug._params["batch_prob"].any()

    @pytest.mark.parametrize(
        "augmentation_cls", [K.RandomMixUpV2, K.RandomCutMixV2, K.PatchMix, K.RandomJigsaw, K.RandomMosaic]
    )
    def test_convention_mix_replay_requires_batch_prob(self, augmentation_cls):
        kwargs = {"use_correct_lambda": True} if augmentation_cls is K.RandomCutMixV2 else {}
        with pytest.raises(KeyError, match="batch_prob"):
            augmentation_cls(p=1.0, **kwargs)(torch.ones(2, 3, 6, 8), params={})

    @pytest.mark.parametrize(
        ("augmentation_cls", "shape"),
        [
            (K.RandomTransplantation, (4, 3, 6, 8)),
            (K.RandomTransplantation3D, (4, 3, 6, 8, 10)),
        ],
    )
    def test_convention_transplantation_p_batch_gate(self, augmentation_cls, shape):
        # Both public transplantation classes expose the base's p_batch gate. A zero gate skips the entire
        # batch even though p would otherwise apply every sample; one applies every sample.
        for p_batch, expected in ((0.0, 0.0), (1.0, 1.0)):
            augmentation = augmentation_cls(p=1.0, p_batch=p_batch)
            assert augmentation.forward_parameters(shape)["batch_prob"].tolist() == [expected] * shape[0]

    @pytest.mark.parametrize(
        ("augmentation_cls", "spatial_shape"),
        [
            (K.RandomTransplantation, (6, 8)),
            (K.RandomTransplantation3D, (3, 6, 8)),
        ],
    )
    def test_convention_transplantation_supports_a_mask_only_call(self, augmentation_cls, spatial_shape, device):
        # Images are optional; a missing mask raises a named error (#4777). Three donors make the cyclic donor
        # direction observable: p=1 replaces every acceptor with its preceding donor.
        mask = torch.arange(3, device=device, dtype=torch.int64).reshape(3, *((1,) * len(spatial_shape)))
        mask = mask.expand(3, *spatial_shape)
        augmentation = augmentation_cls(p=1.0)
        output = augmentation(mask, data_keys=["mask"])
        assert isinstance(output, torch.Tensor)  # a single output is a bare tensor
        self.assert_close(output, torch.roll(mask, 1, dims=0))

    def test_convention_same_on_batch_collapses_the_draw_to_one_sample(self):
        # Convention pin: `same_on_batch=True` collapses every per-sample draw - the `p` gate included - to a
        # single value broadcast over the batch. Same seeds as the pin above, whose `p=0.5` rows are mixed.
        rows = []
        for seed in range(4):
            torch.manual_seed(seed)
            rows.append(K.RandomHorizontalFlip(p=0.5, same_on_batch=True).forward_parameters((4, 3, 6, 8)))
        assert [r["batch_prob"].tolist() for r in rows] == [[1.0] * 4, [0.0] * 4, [0.0] * 4, [1.0] * 4]
        torch.manual_seed(0)
        angles = K.RandomAffine(degrees=(10.0, 90.0), same_on_batch=True).forward_parameters((4, 3, 6, 8))["angle"]
        assert angles.unique().numel() == 1
        torch.manual_seed(0)
        angles = K.RandomAffine(degrees=(10.0, 90.0), same_on_batch=False).forward_parameters((4, 3, 6, 8))["angle"]
        assert angles.unique().numel() == 4

    @pytest.mark.parametrize("augmentation_cls", [K.ColorJiggle, K.ColorJitter])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_convention_color_adjustment_order_is_shared_across_the_batch(self, augmentation_cls, same_on_batch):
        # The color adjustments have one random permutation per call, independently of same_on_batch. The
        # individual factors do honor same_on_batch.
        params = augmentation_cls(0.2, 0.2, 0.2, 0.1, same_on_batch=same_on_batch).forward_parameters((3, 3, 6, 8))
        order = params["order"]
        assert order.shape == (4,)
        assert torch.equal(order.sort().values, torch.arange(4, device=order.device, dtype=order.dtype))
        if same_on_batch:
            for name in ("brightness_factor", "contrast_factor", "saturation_factor", "hue_factor"):
                assert params[name].unique().numel() == 1
        else:
            for name in ("brightness_factor", "contrast_factor", "saturation_factor", "hue_factor"):
                assert params[name].unique().numel() > 1

    def test_convention_global_seed_reproduces_the_draw(self):
        # Convention pin: reproducibility goes through the global torch CPU generator - `torch.manual_seed`
        # before the draw reproduces `forward_parameters` key for key, bitwise.
        torch.manual_seed(1)
        first = K.RandomAffine(degrees=45.0, translate=(0.2, 0.2)).forward_parameters((2, 1, 8, 8))
        torch.manual_seed(1)
        second = K.RandomAffine(degrees=45.0, translate=(0.2, 0.2)).forward_parameters((2, 1, 8, 8))
        assert set(first) == set(second)
        assert all(torch.equal(first[k], second[k]) for k in first)

    def test_wart_generator_is_rejected_at_construction_and_swallowed_by_forward_4427(self, device, dtype):
        # Wart pin (#4427): there is no per-instance RNG. `generator=` raises `TypeError` on a class and on a
        # container, and `forward` silently accepts and drops it - together with any other unknown keyword -
        # because `forward` funnels its `**kwargs` into `override_parameters`. The passed generator is never
        # consumed, so the output still follows the global seed.
        with pytest.raises(TypeError):
            K.RandomAffine(degrees=45.0, generator=torch.Generator())
        with pytest.raises(TypeError):
            K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), generator=torch.Generator())
        x = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        gen = torch.Generator()
        gen.manual_seed(2718)
        torch.manual_seed(0)
        with_generator = K.RandomAffine(degrees=(10.0, 90.0), p=1.0)(x, generator=gen)
        torch.manual_seed(0)
        without = K.RandomAffine(degrees=(10.0, 90.0), p=1.0)(x)
        assert torch.equal(with_generator, without)
        # any other unknown keyword is swallowed just as silently
        assert K.RandomAffine(degrees=(10.0, 90.0), p=1.0)(x, nonsense=1).shape == (2, 3, 6, 8)

    def test_convention_params_replay_is_bitwise_and_the_dict_is_stored_by_reference(self, device, dtype):
        # Convention pin for a complete generated dictionary: `params=` replaces the previous draw,
        # stores the dictionary by reference and replays this affine bitwise without changing its entries.
        aug = K.RandomAffine(degrees=(45.0, 45.0), translate=(0.2, 0.2), p=1.0)
        x = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        first = aug(x)
        params = dict(aug._params)
        before = {k: v.clone() for k, v in params.items()}
        second = aug(x, params=params)
        assert torch.equal(first, second)
        assert aug._params is params
        assert set(aug._params) == set(before)
        assert all(torch.equal(before[k], params[k]) for k in before)

    def test_convention_params_without_batch_prob_are_extended_in_place(self, device, dtype):
        aug = K.RandomHorizontalFlip(p=0.0)
        image = torch.arange(16, device=device, dtype=dtype).reshape(1, 1, 4, 4)
        params = {}
        output = aug(image, params=params)
        assert aug._params is params
        assert set(params) == {"batch_prob"}
        assert params["batch_prob"].tolist() == [True]
        self.assert_close(output, image.flip(-1))

    def test_convention_rigid_base_requires_data_key_handlers(self, device, dtype):
        class Identity(K.RigidAffineAugmentationBase2D):
            def compute_transformation(self, input, params, flags):
                return self.identity_matrix(input)

            def apply_transform(self, input, params, flags, transform=None):
                return input

        aug = Identity(p=1.0)
        image = torch.ones(1, 1, 4, 4, device=device, dtype=dtype)
        self.assert_close(aug(image), image)
        boxes = Boxes.from_tensor(torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        keypoints = Keypoints(torch.tensor([[[1.0, 1.0]]], device=device, dtype=dtype))
        for handler, data in (
            (aug.transform_masks, image),
            (aug.transform_boxes, boxes),
            (aug.transform_keypoints, keypoints),
        ):
            with pytest.raises(NotImplementedError):
                handler(data, aug._params, aug.flags, transform=aug.transform_matrix)
        for key, data in (("mask", image), ("bbox_xyxy", boxes), ("keypoints", keypoints)):
            with pytest.raises(NotImplementedError):
                K.AugmentationSequential(aug, data_keys=["input", key])(image, data)

    def test_convention_bhw_mask_uses_tensor_not_list_5598(self, device, dtype):
        # A batched (B, H, W) mask takes its batch size from ``batch_prob`` (#5598). A mask list instead holds
        # one tensor per image sample (#4477); wrapping the full batch in a one-element list is invalid.
        image = torch.arange(2 * 64, device=device).to(dtype).reshape(2, 1, 8, 8)
        sequence = K.AugmentationSequential(K.RandomCrop((4, 4), p=1.0), data_keys=["input", "mask"])
        output_image, output_mask = sequence(image, image[:, 0])
        assert output_mask.shape == (2, 1, 4, 4)
        self.assert_close(output_mask, output_image, rtol=0, atol=0)
        with pytest.raises(ValueError, match="one entry per sample"):
            sequence(image, [image[:, 0]])

    @pytest.mark.parametrize("first_dim", [2, 3])
    def test_convention_3d_mask_keepdim_returns_its_rank_5598(self, device, dtype, first_dim):
        # Under keepdim a 3-D mask whose first dimension is the image batch stays (B, H, W); otherwise it is a
        # single (C, H, W) mask that is broadcast across the image batch.
        image = torch.zeros(2, 1, 8, 8, device=device, dtype=dtype)
        mask = torch.arange(first_dim * 64, device=device).to(dtype).reshape(first_dim, 8, 8)
        sequence = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask"], keepdim=True)
        _, output_mask = sequence(image, mask)
        expected = (
            mask.flip(-1)
            if first_dim == image.shape[0]
            else mask.flip(-1).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
        )
        self.assert_close(output_mask, expected, rtol=0, atol=0)

    def test_convention_video_bthw_mask_follows_the_clip_5598(self, device, dtype):
        # VideoSequential flattens a (B, T, H, W) mask to (B * T, H, W). With same_on_frame=True the child draws for B
        # and its gates are broadcast to B * T, which is the batch the mask has; the drawn shape says B (#5598).
        clip = torch.arange(2 * 3 * 64, device=device).to(dtype).reshape(2, 3, 1, 8, 8)
        video = K.VideoSequential(K.RandomHorizontalFlip(p=1.0))
        sequence = K.AugmentationSequential(video, data_keys=["input", "mask"])
        output_clip, output_mask = sequence(clip, clip[:, :, 0])
        self.assert_close(output_mask, output_clip, rtol=0, atol=0)

    def test_convention_direct_geometric_mask_handler_rejects_bool(self, device, dtype):
        # The container casts bool masks around geometric dispatch. Calling the geometric handler directly
        # instead takes the image dtype guard, so float masks work while bool masks raise TypeError.
        augmentation = K.RandomHorizontalFlip(p=1.0)
        image = torch.ones(2, 1, 3, 4, device=device, dtype=dtype)
        augmentation(image)
        float_mask = torch.tensor(
            [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        transformed = augmentation.transform_masks(
            float_mask, augmentation._params, augmentation.flags, transform=augmentation.transform_matrix
        )
        self.assert_close(transformed, float_mask.flip(-1))
        with pytest.raises(TypeError, match="Expected input of"):
            augmentation.transform_masks(
                float_mask.bool(), augmentation._params, augmentation.flags, transform=augmentation.transform_matrix
            )

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_intensity_boxes_pass_through_direct_dispatch_and_container_4480(self, device, dtype, p):
        image = torch.ones(1, 1, 4, 4, device=device, dtype=dtype)
        boxes = Boxes.from_tensor(torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        augmentation = K.RandomInvert(p=p)
        augmentation(image)
        direct = augmentation.transform_boxes(boxes, augmentation._params, augmentation.flags)
        assert direct.mode == boxes.mode
        self.assert_close(direct.data, boxes.data)
        container = K.AugmentationSequential(K.RandomInvert(p=1.0), data_keys=["input", "bbox_xyxy"])
        _, output_boxes = container(image, boxes)
        self.assert_close(output_boxes.data, boxes.data)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize("as_objects", [False, True])
    @pytest.mark.parametrize("annotations_first", [False, True])
    def test_convention_container_dispatches_custom_rigid_annotations_4481(
        self, device, dtype, p, as_objects, annotations_first
    ):
        # Every handler reads the matrix the container passes, so a handler called with ``transform=None`` fails.
        class ShiftRight(K.RigidAffineAugmentationBase2D):
            def compute_transformation(self, input, params, flags):
                matrix = self.identity_matrix(input).clone()
                matrix[:, 0, 2] = 1.0
                return matrix

            def apply_transform(self, input, params, flags, transform=None):
                return input.roll(1, dims=-1)

            def apply_non_transform_mask(self, input, params, flags, transform=None):
                return input

            def apply_transform_mask(self, input, params, flags, transform=None):
                return input.roll(int(transform[0, 0, 2]), dims=-1)

            def apply_transform_box(self, input, params, flags, transform=None):
                return input.transform_boxes_(transform)

            def apply_transform_keypoint(self, input, params, flags, transform=None):
                return input.transform_keypoints_(transform)

        image = torch.zeros(1, 1, 4, 5, device=device, dtype=dtype)
        image[0, 0, 1, 1] = 1.0
        mask = image.clone()
        boxes = Boxes.from_tensor(torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        keypoints = Keypoints(torch.tensor([[[1.0, 1.0]]], device=device, dtype=dtype))
        augmentation = ShiftRight(p=p)
        sequence = K.AugmentationSequential(augmentation, data_keys=["input", "mask", "bbox_xyxy", "keypoints"])
        box_input = boxes if as_objects else boxes.to_tensor("xyxy")
        keypoint_input = keypoints if as_objects else keypoints.data
        if annotations_first:
            output_mask, output_boxes, output_keypoints, output_image = sequence(
                mask,
                box_input,
                keypoint_input,
                image,
                data_keys=["mask", "bbox_xyxy", "keypoints", "input"],
            )
        else:
            output_image, output_mask, output_boxes, output_keypoints = sequence(image, mask, box_input, keypoint_input)

        expected_boxes = boxes.to_tensor("xyxy") + torch.tensor([p, 0.0, p, 0.0], device=device, dtype=dtype)
        expected_keypoints = keypoints.data + torch.tensor([p, 0.0], device=device, dtype=dtype)
        actual_boxes = output_boxes.to_tensor("xyxy") if isinstance(output_boxes, Boxes) else output_boxes
        actual_keypoints = output_keypoints.data if isinstance(output_keypoints, Keypoints) else output_keypoints
        self.assert_close(output_image, image.roll(int(p), dims=-1))
        self.assert_close(output_mask, output_image)
        self.assert_close(actual_boxes, expected_boxes)
        self.assert_close(actual_keypoints, expected_keypoints)
        self.assert_close(sequence.transform_matrix[:, 0, 2], torch.tensor([p], device=device, dtype=dtype))
        # A direct call with the same parameters and matrix agrees with the container.
        params, flags, transform = augmentation._params, augmentation.flags, augmentation.transform_matrix
        self.assert_close(
            augmentation.transform_boxes(boxes, params, flags, transform).to_tensor("xyxy"), expected_boxes
        )
        self.assert_close(
            augmentation.transform_keypoints(keypoints, params, flags, transform).data, expected_keypoints
        )
        # A list of masks takes the per-entry path, which passes the matrix too.
        mask_sequence = K.AugmentationSequential(augmentation, data_keys=["input", "mask"])
        if annotations_first:
            output_masks, list_output_image = mask_sequence([mask[0]], image, data_keys=["mask", "input"])
        else:
            list_output_image, output_masks = mask_sequence(image, [mask[0]])
        self.assert_close(list_output_image, output_image)
        self.assert_close(output_masks[0], output_image)

    @pytest.mark.parametrize("key", ["mask", "bbox_xyxy", "keypoints"])
    def test_convention_container_transforms_image_first_for_geometric_children(self, device, dtype, key):
        # Annotation handlers read the matrix the image call records. With the annotation listed before the image,
        # a built-in geometric child used to raise on the first call and reuse the previous call's matrix after it.
        image = torch.linspace(0, 1, 2 * 3 * 16 * 20, device=device, dtype=dtype).reshape(2, 3, 16, 20)
        annotations = {
            "mask": (image[:, :1] > 0.5).to(dtype),
            "bbox_xyxy": torch.tensor([[[2.0, 3.0, 8.0, 9.0]], [[4.0, 1.0, 12.0, 7.0]]], device=device, dtype=dtype),
            "keypoints": torch.tensor(
                [[[3.0, 4.0], [10.0, 6.0]], [[5.0, 5.0], [15.0, 12.0]]], device=device, dtype=dtype
            ),
        }
        annotation = annotations[key]
        reference = K.AugmentationSequential(RandomAffine(30.0, p=1.0))
        sequence = K.AugmentationSequential(RandomAffine(30.0, p=1.0))
        for seed in (0, 1):
            torch.manual_seed(seed)
            expected_image, expected = reference(image, annotation, data_keys=["input", key])
            torch.manual_seed(seed)
            output, output_image = sequence(annotation, image, data_keys=[key, "input"])
            self.assert_close(output_image, expected_image)
            self.assert_close(output, expected)

    def test_convention_container_transforms_image_first_for_nested_geometric_child(self, device, dtype):
        image = torch.linspace(0, 1, 2 * 3 * 16 * 20, device=device, dtype=dtype).reshape(2, 3, 16, 20)
        keypoints = torch.tensor([[[3.0, 4.0], [10.0, 6.0]], [[5.0, 5.0], [15.0, 12.0]]], device=device, dtype=dtype)
        reference = K.AugmentationSequential(RandomAffine(30.0, p=1.0))
        sequence = K.AugmentationSequential(K.ImageSequential(RandomAffine(30.0, p=1.0)))

        # A fresh nested child used to raise because its matrix was not recorded yet.
        torch.manual_seed(1)
        expected_image, expected_keypoints = reference(image, keypoints, data_keys=["input", "keypoints"])
        torch.manual_seed(1)
        output_keypoints, output_image = sequence(keypoints, image, data_keys=["keypoints", "input"])
        self.assert_close(output_image, expected_image)
        self.assert_close(output_keypoints, expected_keypoints)

        # A prior call used to leave a matrix behind and silently apply that matrix to later calls.
        sequence(image, keypoints, data_keys=["input", "keypoints"])
        for seed in range(2, 6):
            torch.manual_seed(seed)
            expected_image, expected_keypoints = reference(image, keypoints, data_keys=["input", "keypoints"])
            torch.manual_seed(seed)
            output_keypoints, output_image = sequence(keypoints, image, data_keys=["keypoints", "input"])
            self.assert_close(output_image, expected_image)
            self.assert_close(output_keypoints, expected_keypoints)

    def test_convention_container_transforms_image_first_for_auto_policy(self, device, dtype):
        image = torch.linspace(0, 1, 2 * 3 * 16 * 20, device=device, dtype=dtype).reshape(2, 3, 16, 20)
        keypoints = torch.tensor([[[3.0, 4.0], [10.0, 6.0]], [[5.0, 5.0], [15.0, 12.0]]], device=device, dtype=dtype)
        reference = K.AugmentationSequential(K.auto.TrivialAugment())
        sequence = K.AugmentationSequential(K.auto.TrivialAugment())
        sequence(image, keypoints, data_keys=["input", "keypoints"])
        for seed in (2, 4):
            torch.manual_seed(seed)
            expected_image, expected_keypoints = reference(image, keypoints, data_keys=["input", "keypoints"])
            torch.manual_seed(seed)
            output_keypoints, output_image = sequence(keypoints, image, data_keys=["keypoints", "input"])
            self.assert_close(output_image, expected_image)
            self.assert_close(output_keypoints, expected_keypoints)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_intensity_annotation_overrides_in_container_5113(self, device, dtype, p):
        class Blackout(K.IntensityAugmentationBase2D):
            def apply_transform(self, input, params, flags, transform=None):
                return torch.zeros_like(input)

            def apply_transform_mask(self, input, params, flags, transform=None):
                return torch.zeros_like(input)

            def apply_transform_box(self, input, params, flags, transform=None):
                return Boxes(torch.zeros_like(input.data), mode=input.mode)

            def apply_transform_keypoint(self, input, params, flags, transform=None):
                return Keypoints(torch.zeros_like(input.data))

        aug = Blackout(p=p)
        image = torch.ones(1, 1, 6, 8, device=device, dtype=dtype)
        mask = torch.ones(1, 1, 6, 8, device=device, dtype=dtype)
        boxes = Boxes.from_tensor(torch.tensor([[[1.0, 1.0, 4.0, 3.0]]], device=device, dtype=dtype))
        keypoints = Keypoints(torch.tensor([[[2.0, 2.0]]], device=device, dtype=dtype))

        seq = K.AugmentationSequential(aug, data_keys=["input", "mask", "bbox_xyxy", "keypoints"])
        out_img, out_mask, out_boxes, out_kp = seq(image, mask, boxes, keypoints)
        scale = 0.0 if p == 1.0 else 1.0
        self.assert_close(out_img, image * scale)
        self.assert_close(out_mask, mask * scale)
        self.assert_close(out_boxes.data, boxes.data * scale)
        self.assert_close(out_kp.data, keypoints.data * scale)

    def test_convention_random_erasing_also_erases_container_masks(self, device, dtype):
        image = torch.ones(1, 1, 6, 8, device=device, dtype=dtype)
        mask = torch.ones_like(image)
        augmentation = K.AugmentationSequential(
            K.RandomErasing(scale=(0.5, 0.5), ratio=(1.0, 1.0), value=0.0, p=1.0), data_keys=["input", "mask"]
        )
        _, output_mask = augmentation(image, mask)
        assert torch.count_nonzero(output_mask) < output_mask.numel()

    @pytest.mark.parametrize("plasma_cls", [K.RandomPlasmaBrightness, K.RandomPlasmaContrast, K.RandomPlasmaShadow])
    def test_convention_random_plasma_replay_is_bitwise(self, plasma_cls, device, dtype):
        # Convention pin: since #4462 (fixing #4445) the three `RandomPlasma*` classes draw their fractal noise
        # in `generate_parameters` and store it under `params["plasma"]`, so a `params=` replay is bitwise
        # without reseeding, while a fresh forward under a different seed draws different noise.
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        aug = plasma_cls(p=1.0)
        x = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        torch.manual_seed(0)
        first = aug(x)
        params = aug._params
        assert "plasma" in params
        torch.manual_seed(5)
        assert torch.equal(aug(x, params=params), first)
        torch.manual_seed(5)
        assert not torch.equal(aug(x), first)

    def test_convention_random_dissolving_replay_needs_the_latent_rng(self, device, dtype):
        # Exercise the real augmentation/filter/encoder path with a lightweight VAE distribution.
        # No diffusion checkpoint is downloaded; only prompt setup and denoising are bypassed.
        image = torch.ones(1, 3, 4, 4, device=device, dtype=dtype)
        latent_dist = Mock()
        latent_dist.sample.side_effect = lambda: torch.rand_like(image)
        model = SimpleNamespace(
            device=device,
            vae=SimpleNamespace(
                dtype=dtype,
                encode=Mock(return_value={"latent_dist": latent_dist}),
                decode=lambda latent: {"sample": latent},
            ),
            tokenizer=None,
            scheduler=SimpleNamespace(set_timesteps=Mock(), timesteps=[0]),
        )
        wrapper = _DissolvingWraper_HF(model)
        with patch.object(StableDiffusionDissolving, "__init__", lambda self, *a, **kw: ImageModule.__init__(self)):
            aug = K.RandomDissolving(step_range=(1.0, 1.0), p=1.0)
        aug._dslv.model = wrapper
        with (
            patch.object(wrapper, "init_prompt"),
            patch.object(wrapper, "one_step_dissolve", side_effect=lambda latent, step: latent),
        ):
            torch.manual_seed(0)
            first = aug(image)
            params = {key: value.clone() for key, value in aug._params.items()}
            second = aug(image, params=params)
            assert not torch.equal(first, second)
            assert set(params) == {"step_range_factor", "batch_prob", "forward_input_shape"}
            torch.manual_seed(5)
            replay = aug(image, params=params)
            torch.manual_seed(5)
            self.assert_close(aug(image, params=params), replay, rtol=0, atol=0)
        assert latent_dist.sample.call_count == 4

    @pytest.mark.parametrize("name", list(_STATELESS_REPRESENTATIVES))
    def test_convention_augmentations_are_stateless_modules(self, name):
        # These numeric-range configurations have no parameters, buffers or state_dict entries.
        # Parameter-valued ranges are covered separately below. They survive pickle and deepcopy while
        # carrying the last `_params`. Other classes register range buffers; that distinction is pinned in
        # `test_convention_param_generator_range_buffers_reach_the_draws`.
        aug = _STATELESS_REPRESENTATIVES[name]()
        assert len(aug.state_dict()) == 0
        assert list(aug.buffers()) == []
        assert list(aug.parameters()) == []
        aug(torch.rand(2, 3, 6, 8))
        for clone in (pickle.loads(pickle.dumps(aug)), copy.deepcopy(aug)):  # noqa: S301
            assert isinstance(clone, type(aug))
            _assert_params_equal(clone._params, aug._params)

    def test_convention_parameter_valued_ranges_receive_gradients(self, device, dtype):
        if not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("grid_sample backward is unavailable for this device/dtype")
        degrees = torch.nn.Parameter(torch.tensor([10.0, 20.0], device=device, dtype=dtype))
        aug = K.RandomRotation(degrees, p=1.0)
        assert dict(aug.named_parameters()) == {"_param_generator.degrees": degrees}
        self.assert_close(aug.state_dict()["_param_generator.degrees"], degrees)
        image = torch.arange(64, device=device, dtype=dtype).reshape(1, 1, 8, 8) / 64
        torch.manual_seed(0)
        aug(image).sum().backward()
        assert degrees.grad is not None
        assert torch.isfinite(degrees.grad).all()
        assert torch.count_nonzero(degrees.grad) == 2

    def test_convention_param_generator_range_buffers_reach_the_draws(self):
        # Classes that keep their sampling range in a `_param_generator.*` buffer expose it in `state_dict()` (and,
        # inside a container, under a prefixed key), and `load_state_dict` from an instance with a different range
        # changes the buffer, the draw and the `repr` of the generator (#6572).
        assert list(K.RandomRotation(degrees=(80.0, 81.0)).state_dict()) == ["_param_generator.degrees"]
        assert list(K.AugmentationSequential(K.RandomRotation(degrees=(10.0, 90.0))).state_dict()) == [
            "RandomRotation_0._param_generator.degrees"
        ]
        source = K.RandomRotation(degrees=(10.0, 11.0))
        target = K.RandomRotation(degrees=(80.0, 81.0))
        assert target.state_dict()["_param_generator.degrees"].tolist() == [80.0, 81.0]
        target.load_state_dict(source.state_dict())
        assert target.state_dict()["_param_generator.degrees"].tolist() == [10.0, 11.0]
        torch.manual_seed(0)
        drawn = target.forward_parameters((4, 3, 6, 8))["degrees"]
        assert bool(((drawn >= 10.0) & (drawn <= 11.0)).all())
        # A range given as numbers is loaded as numbers.
        assert "degrees=[10.0, 11.0]" in repr(target)

    def test_wart_sampler_precision_is_separate_from_returned_dtype_4426(self):
        # #4426: flips when the returned parameters follow the requested dtype like `batch_prob` does.
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            for aug, key, sampler in (
                (K.RandomAffine(degrees=(10.0, 90.0), p=1.0), "angle", "degree_sampler"),
                (K.RandomRotation(degrees=(10.0, 90.0), p=1.0), "degrees", None),
            ):

                def distribution(aug=aug, key=key, sampler=sampler):
                    generator = aug._param_generator
                    return getattr(generator, sampler) if sampler else generator.sampler_dict[key]

                assert distribution().low.dtype == torch.float32
                torch.set_default_dtype(torch.float64)
                assert aug.forward_parameters((2, 3, 6, 8))[key].dtype == torch.float64
                assert distribution().low.dtype == torch.float32
                aug.set_rng_device_and_dtype(torch.device("cpu"), torch.float64)
                assert distribution().low.dtype == torch.float64
                torch.set_default_dtype(torch.float32)
                params = aug.forward_parameters((2, 3, 6, 8))
                assert params[key].dtype == torch.float32
                assert params["batch_prob"].dtype == torch.float64
        finally:
            torch.set_default_dtype(original_dtype)

    def test_wart_set_rng_device_moves_sampling_but_not_returned_parameters_4426(self, device):
        # #4426: flips when the returned angle follows the requested device like `batch_prob` does; only an
        # accelerator leg can see that, since on CPU both devices are the same.
        aug = K.RandomAffine(degrees=(10.0, 90.0), p=1.0)
        aug.set_rng_device_and_dtype(device, torch.float32)
        assert aug._param_generator.degree_sampler.low.device == device
        if device.type == "cuda":

            def get_device_state():
                return torch.cuda.get_rng_state(device)

        elif device.type == "mps":
            get_device_state = torch.mps.get_rng_state
        elif device.type == "cpu":
            get_device_state = torch.random.get_rng_state
        else:
            pytest.skip("RNG state assertions cover CPU, CUDA and MPS")
        cpu_before = torch.random.get_rng_state()
        device_before = get_device_state()
        params = aug.forward_parameters((2, 3, 6, 8))
        assert not torch.equal(device_before, get_device_state())
        if device.type != "cpu":
            assert torch.equal(cpu_before, torch.random.get_rng_state())
        assert params["batch_prob"].device == device
        assert params["angle"].device.type == "cpu"

    def test_convention_tensor_ranges_determine_returned_parameter_placement(self, device, dtype):
        degrees = torch.tensor([10.0, 20.0], device=device, dtype=dtype)
        for aug, key in ((K.RandomAffine(degrees, p=1.0), "angle"), (K.RandomRotation(degrees, p=1.0), "degrees")):
            aug.set_rng_device_and_dtype(torch.device("cpu"), torch.float32)
            drawn = aug.forward_parameters((2, 3, 6, 8))[key]
            assert drawn.device == degrees.device
            assert drawn.dtype == degrees.dtype

    def test_convention_rng_consumption_fingerprint(self):
        # Pin CPU float32 consumption relative to this PyTorch build's generator, not its numeric sequence.
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            for name, (make, shape, operations) in _RNG_FINGERPRINTS.items():
                torch.manual_seed(0)
                make().forward_parameters(shape)
                actual_state = torch.random.get_rng_state()
                torch.manual_seed(0)
                for operation, draw_shape in operations:
                    kwargs = {"device": "cpu"}
                    if operation != "randperm":
                        kwargs["dtype"] = torch.float32
                    getattr(torch, operation)(*draw_shape, **kwargs)
                assert torch.equal(actual_state, torch.random.get_rng_state()), name
        finally:
            torch.set_default_dtype(original_dtype)

    def test_convention_zero_batch_passes_through(self, device, dtype):
        # Convention pin (#4115 family, empty in -> empty out): a `B = 0` batch survives the flip, the affine
        # and an intensity op, and survives `AugmentationSequential`, keeping the (0, C, H, W) shape.
        if device.type == "mps" and torch_version_lt(2, 6, 0):
            pytest.skip("torch 2.5.1 MPS asserts on an empty placeholder tensor inside grid_sample")
        empty = torch.rand(0, 3, 6, 8, device=device, dtype=dtype)
        assert K.RandomHorizontalFlip(p=1.0)(empty).shape == (0, 3, 6, 8)
        assert K.RandomAffine(degrees=(45.0, 45.0), p=1.0)(empty).shape == (0, 3, 6, 8)
        assert K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0)(empty).shape == (0, 3, 6, 8)
        container = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input"])
        assert container(empty).shape == (0, 3, 6, 8)

    @pytest.mark.parametrize("augmentation", [K.RandomGaussianIllumination, K.RandomLinearCornerIllumination])
    @pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
    def test_minmax_illumination_empty_batch_4429(self, augmentation, probability, device, dtype):
        data = torch.empty(0, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = augmentation(p=probability)(data)
        assert output.shape == data.shape
        assert output.dtype == dtype
        assert output.device == device
        output.sum().backward()
        assert data.grad is not None
        assert data.grad.shape == data.shape

    @pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("clip_output", [False, True])
    def test_autocontrast_empty_batch_4429(self, probability, clip_output, device, dtype):
        data = torch.empty(0, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        with pytest.warns(DeprecationWarning, match="clip_output"):
            augmentation = K.RandomAutoContrast(p=probability, clip_output=clip_output)
        output = augmentation(data)
        replay = augmentation(data, params=augmentation._params)
        for actual in (output, replay):
            assert actual.shape == data.shape
            assert actual.dtype == dtype
            assert actual.device == device
        output.sum().backward()
        assert data.grad is not None
        assert data.grad.shape == data.shape
        sequence = K.AugmentationSequential(augmentation, data_keys=["input"])
        assert sequence(data).shape == data.shape

    @pytest.mark.parametrize(
        ("augmentation", "shape"),
        [
            pytest.param(lambda: K.RandomCrop((4, 6), p=1.0), (0, 3, 4, 6), id="RandomCrop"),
            pytest.param(lambda: K.RandomCrop((8, 10), pad_if_needed=True, p=1.0), (0, 3, 8, 10), id="RandomCrop-pad"),
            pytest.param(lambda: K.RandomResizedCrop((4, 4), p=1.0), (0, 3, 4, 4), id="RandomResizedCrop"),
            pytest.param(lambda: K.LongestMaxSize(16, p=1.0), (0, 3, 12, 16), id="LongestMaxSize"),
            pytest.param(lambda: K.SmallestMaxSize(12, p=1.0), (0, 3, 12, 16), id="SmallestMaxSize"),
            pytest.param(lambda: K.Resize(10, side="long"), (0, 3, 7, 10), id="Resize-int-long"),
            pytest.param(lambda: K.Resize(10, side="short"), (0, 3, 10, 13), id="Resize-int-short"),
            pytest.param(
                lambda: K.RandomAutoContrast(p=1.0),
                (0, 3, 6, 8),
            ),
            pytest.param(lambda: K.RandomEqualize(p=1.0), (0, 3, 6, 8), id="RandomEqualize"),
            pytest.param(lambda: K.RandomPosterize(p=1.0), (0, 3, 6, 8), id="RandomPosterize"),
            pytest.param(lambda: K.RandomSharpness(p=1.0), (0, 3, 6, 8), id="RandomSharpness"),
            pytest.param(
                lambda: K.RandomGaussianBlur((3, 5), (0.5, 1.5), "constant", p=1.0),
                (0, 3, 6, 8),
                id="RandomGaussianBlur",
            ),
            pytest.param(lambda: K.RandomMotionBlur(3, 35.0, 0.5, p=1.0), (0, 3, 6, 8), id="RandomMotionBlur"),
            pytest.param(lambda: K.RandomJPEG(p=1.0), (0, 3, 6, 8), id="RandomJPEG"),
        ],
    )
    def test_convention_zero_batch_is_empty_in_empty_out(self, augmentation, shape, device, dtype):
        # Convention pin (#4429): each listed augmentation preserves an empty batch with its intended
        # output geometry. Other empty-batch configurations remain tracked by the umbrella issue.
        empty = torch.rand(0, 3, 6, 8, device=device, dtype=dtype)
        assert augmentation()(empty).shape == shape

    @pytest.mark.parametrize(
        ("augmentation", "input_shape", "output_size"),
        [
            pytest.param(lambda p: K.RandomCrop((4, 6), p=p), (3, 6, 8), (4, 6), id="RandomCrop"),
            pytest.param(lambda p: K.RandomCrop((4, 6), padding=1, p=p), (3, 6, 8), (4, 6), id="RandomCrop-padding"),
            pytest.param(lambda p: K.CenterCrop((4, 6), p=p), (3, 6, 8), (4, 6), id="CenterCrop"),
            pytest.param(lambda p: K.RandomResizedCrop((4, 4), p=p), (3, 6, 8), (4, 4), id="RandomResizedCrop"),
            pytest.param(lambda p: K.Resize((4, 5), p=p), (3, 6, 8), (4, 5), id="Resize-tuple"),
            pytest.param(lambda p: K.Resize(10, side="long", p=p), (3, 6, 8), (7, 10), id="Resize-int-long"),
            pytest.param(lambda p: K.LongestMaxSize(16, p=p), (3, 6, 8), (12, 16), id="LongestMaxSize"),
            pytest.param(lambda p: K.SmallestMaxSize(12, p=p), (3, 6, 8), (12, 16), id="SmallestMaxSize"),
            pytest.param(lambda p: K.RandomCrop3D((4, 4, 4), p=p), (1, 6, 8, 10), (4, 4, 4), id="RandomCrop3D"),
            pytest.param(lambda p: K.CenterCrop3D((4, 4, 4), p=p), (1, 6, 8, 10), (4, 4, 4), id="CenterCrop3D"),
        ],
    )
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_zero_batch_takes_configured_size_under_any_gate_4429(
        self, augmentation, input_shape, output_size, p, device, dtype
    ):
        # An empty batch has no sample for the gate to skip, so a shape-changing augmentation returns its configured
        # output size whatever ``p`` is (#4429), as 0.8.3 did for the 2D crops. A non-empty batch the gate skips
        # entirely still comes back at its input size.
        channels = input_shape[0]
        aug = augmentation(p)
        empty = torch.rand(0, *input_shape, device=device, dtype=dtype)
        assert aug(empty).shape == (0, channels, *output_size)
        skipped = torch.rand(2, *input_shape, device=device, dtype=dtype)
        assert augmentation(0.0)(skipped).shape == skipped.shape


_AUGMENTATIONS = [K.RandomEqualize, K.RandomEqualize3D, K.RandomPosterize, K.RandomSharpness]


def _shape(augmentation, batch=0):
    return (batch, 2, 3, 5, 7) if augmentation is K.RandomEqualize3D else (batch, 3, 6, 8)


def _assert_params_unchanged(actual, snapshot):
    assert actual.keys() == snapshot.keys()
    for name in actual:
        assert torch.equal(actual[name], snapshot[name]), name


class TestEmptyIntensityAugmentation(BaseTester):
    @pytest.mark.parametrize("augmentation", _AUGMENTATIONS)
    @pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_direct_callers_4429(self, augmentation, probability, same_on_batch, device, dtype):
        image = torch.empty(_shape(augmentation), device=device, dtype=dtype, requires_grad=True)
        aug = augmentation(p=probability, same_on_batch=same_on_batch).to(device=device, dtype=dtype)
        params = aug.forward_parameters(image.shape)
        # Exercise the function itself even when a container or the gate could bypass it.
        transformed = aug.apply_transform(image, params, aug.flags)
        assert transformed.shape == image.shape
        assert transformed.device == device
        assert transformed.dtype == dtype
        transformed.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape
        image.grad = None
        output = aug(image, params=params)
        assert output.shape == image.shape
        assert output.device == device
        assert output.dtype == dtype
        assert params["forward_input_shape"].tolist() == list(image.shape)
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("augmentation", _AUGMENTATIONS)
    @pytest.mark.parametrize("state", ["copied", "after_forward", "fresh"])
    def test_saved_parameters_replay_4429(self, augmentation, state, device, dtype):
        image = torch.empty(_shape(augmentation), device=device, dtype=dtype, requires_grad=True)
        aug = augmentation(p=1).to(device=device, dtype=dtype)
        output = aug(image)
        params = copy.deepcopy(aug._params)
        snapshot = copy.deepcopy(params)
        if state == "after_forward":
            later_shape = (*_shape(augmentation, 2)[:-2], 9, 11)
            aug(torch.rand(later_shape).to(device=device, dtype=dtype))
        elif state == "fresh":
            aug = augmentation(p=1).to(device=device, dtype=dtype)
        cpu_rng = torch.random.get_rng_state().clone()
        replayed = aug(image, params=params)
        assert replayed.shape == image.shape
        assert replayed.dtype == dtype
        assert replayed.device == device
        self.assert_close(replayed, output, rtol=0, atol=0)
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        _assert_params_unchanged(params, snapshot)
        replayed.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("container", ["image", "augmentation", "patch_same", "patch_valid"])
    def test_2d_container_replay_4429(self, container, device, dtype):
        children = [K.RandomEqualize(p=1), K.RandomPosterize(p=1), K.RandomSharpness(p=1)]
        if container == "image":
            sequence = K.ImageSequential(*children)
        elif container == "augmentation":
            sequence = K.AugmentationSequential(*children, data_keys=["input", "mask"])
        else:
            sequence = K.PatchSequential(
                *children, grid_size=(1, 3), padding=container.split("_")[1], patchwise_apply=True, random_apply=False
            )
        sequence = sequence.to(device=device, dtype=dtype)
        image = torch.empty(0, 3, 9, 13, device=device, dtype=dtype, requires_grad=True)
        if container == "augmentation":
            mask = torch.empty(0, 1, 9, 13, device=device, dtype=dtype)
            output, output_mask = sequence(image, mask)
            assert output_mask.shape == mask.shape
            replayed, replayed_mask = sequence(image, mask, params=copy.deepcopy(sequence._params))
            assert replayed_mask.shape == mask.shape
        else:
            output = sequence(image)
            replayed = sequence(image, params=copy.deepcopy(sequence._params))
        expected_shape = (0, 3, 9, 12 if container == "patch_valid" else 13)
        assert output.shape == replayed.shape == expected_shape
        assert output.device == device
        assert output.dtype == dtype
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("layout", ["BTCHW", "BCTHW"])
    @pytest.mark.parametrize("same_on_frame", [False, True])
    def test_video_replay_4429(self, layout, same_on_frame, device, dtype):
        shape = (0, 2, 3, 6, 8) if layout == "BTCHW" else (0, 3, 2, 6, 8)
        video = torch.empty(shape, device=device, dtype=dtype, requires_grad=True)
        sequence = K.VideoSequential(
            K.RandomEqualize(p=1),
            K.RandomPosterize(p=1),
            K.RandomSharpness(p=1),
            data_format=layout,
            same_on_frame=same_on_frame,
        ).to(device=device, dtype=dtype)
        output = sequence(video)
        params = copy.deepcopy(sequence._params)
        replayed = sequence(video, params=params)
        assert output.shape == replayed.shape == video.shape
        assert output.device == device
        assert output.dtype == dtype
        for item in params:
            assert item.data["forward_input_shape"][-2:].tolist() == [6, 8]
        replayed.sum().backward()
        assert video.grad is not None
        assert video.grad.shape == video.shape

    def test_3d_container_replay_4429(self, device, dtype):
        image = torch.empty(0, 2, 3, 5, 7, device=device, dtype=dtype, requires_grad=True)
        sequence = K.AugmentationSequential(K.RandomEqualize3D(p=1), data_keys=["input"]).to(device=device, dtype=dtype)
        output = sequence(image)
        replayed = sequence(image, params=copy.deepcopy(sequence._params))
        assert output.shape == replayed.shape == image.shape
        assert output.device == device
        assert output.dtype == dtype
        replayed.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("augmentation", _AUGMENTATIONS)
    def test_dynamo_direct_transform_4429(self, augmentation, device, dtype, torch_optimizer):
        image = torch.empty(_shape(augmentation), device=device, dtype=dtype)
        aug = augmentation(p=1).to(device=device, dtype=dtype)
        params = aug.forward_parameters(image.shape)

        def run(input):
            return aug.apply_transform(input, params, aug.flags)

        expected = run(image)
        compiled = torch_optimizer(run)
        output = compiled(image)
        assert output.shape == image.shape
        assert output.device == device
        assert output.dtype == dtype
        self.assert_close(output, expected)


class TestStaticProbabilityIgnoresHandMadeGate(BaseTester):
    """With static ``p=1.0, p_batch=1.0`` every handler transforms every sample (#5585).

    A ``batch_prob`` put into ``params`` is not consulted: the image, the matrix, the labels and ``inverse`` agree.
    """

    @pytest.mark.parametrize("augmentation_cls", [K.RandomHorizontalFlip, K.RandomVerticalFlip])
    def test_round_trip_restores_the_batch(self, augmentation_cls, device, dtype):
        x = torch.rand(3, 3, 8, 12, device=device, dtype=dtype)
        aug = augmentation_cls(p=1.0)
        params = aug.forward_parameters(x.shape)
        params["batch_prob"] = torch.tensor([0.0, 1.0, 0.0])

        out = aug(x, params=copy.deepcopy(params))

        assert not torch.equal(out[0], x[0])  # the hand-made gate is not read at p=1
        self.assert_close(aug.inverse(out, params=copy.deepcopy(params)), x)

    def test_keypoints_and_matrix_follow_the_image(self, device, dtype):
        x = torch.rand(2, 1, 8, 12, device=device, dtype=dtype)
        points = torch.tensor([[[2.0, 3.0]], [[2.0, 3.0]]], device=device, dtype=dtype)
        module = K.RandomHorizontalFlip(p=1.0)
        aug = K.AugmentationSequential(module, data_keys=["input", "keypoints"])
        params = module.forward_parameters(x.shape)
        params["batch_prob"] = torch.tensor([0.0, 1.0])

        out, out_points = aug(x, points, params=[K.container.params.ParamItem("RandomHorizontalFlip_0", params)])

        self.assert_close(out, x.flip(-1))
        self.assert_close(out_points[..., 0], 11.0 - points[..., 0])
        self.assert_close(aug.inverse(out, out_points, params=aug._params)[1], points)

    @pytest.mark.parametrize(
        ("augmentation_cls", "kwargs", "shape"),
        [
            pytest.param(K.RandomAffine, {"degrees": 30.0}, (2, 1, 8, 12), id="RandomAffine"),
            pytest.param(
                K.RandomCrop,
                {"size": (8, 12), "padding": (1, 2, 3, 0), "cropping_mode": "resample"},
                (2, 1, 8, 12),
                id="RandomCrop-resample",
            ),
            pytest.param(K.RandomAffine3D, {"degrees": 30.0}, (2, 1, 4, 6, 6), id="RandomAffine3D"),
        ],
    )
    def test_hand_made_gate_matches_the_sampled_gate(self, augmentation_cls, kwargs, shape, device, dtype):
        x = torch.rand(*shape, device=device, dtype=dtype)
        aug = augmentation_cls(p=1.0, **kwargs)
        params = aug.forward_parameters(x.shape)
        hand_made = copy.deepcopy(params)
        hand_made["batch_prob"] = torch.tensor([0.0, 1.0])

        out = aug(x, params=copy.deepcopy(params))
        matrix = aug.transform_matrix

        self.assert_close(aug(x, params=copy.deepcopy(hand_made)), out)
        self.assert_close(aug.transform_matrix, matrix)
        if len(shape) == 4:
            expected = aug.inverse(out, params=copy.deepcopy(params))
            self.assert_close(aug.inverse(out, params=copy.deepcopy(hand_made)), expected)

    def test_masks_and_boxes_follow_the_image(self, device, dtype):
        x = torch.rand(2, 1, 8, 12, device=device, dtype=dtype)
        mask = (torch.rand(2, 1, 8, 12, device=device) > 0.5).to(dtype)
        boxes = torch.tensor([[[1.0, 2.0, 4.0, 5.0]], [[1.0, 2.0, 4.0, 5.0]]], device=device, dtype=dtype)
        module = K.RandomHorizontalFlip(p=1.0)
        aug = K.AugmentationSequential(module, data_keys=["input", "mask", "bbox_xyxy"])
        params = module.forward_parameters(x.shape)
        params["batch_prob"] = torch.tensor([0.0, 1.0])

        out, out_mask, out_boxes = aug(
            x, mask, boxes, params=[K.container.params.ParamItem("RandomHorizontalFlip_0", params)]
        )

        self.assert_close(out_mask, mask.flip(-1))
        _, inv_mask, inv_boxes = aug.inverse(out, out_mask, out_boxes, params=aug._params)
        self.assert_close(inv_mask, mask)
        self.assert_close(inv_boxes, boxes)


def _learnable(*values: float) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor(values if len(values) > 1 else values[0]))


# name -> (constructor, volumetric input, range receives a gradient). One augmentation per generator family that builds
# its samplers from a range tensor, plus RandomRGBShift (a 0-d half-width) and a mix augmentation. The ranges are made
# on the CPU and move with the module, the way a model holding the augmentation is moved.
_LEARNABLE_RANGES = {
    "RandomBrightness": (lambda: K.RandomBrightness(_learnable(0.8, 1.2), p=1.0), False, True),
    "RandomRGBShift": (lambda: K.RandomRGBShift(_learnable(0.1), _learnable(0.2), _learnable(0.3), p=1.0), False, True),
    "ColorJiggle": (
        lambda: K.ColorJiggle(_learnable(0.8, 1.2), _learnable(0.8, 1.2), _learnable(0.8, 1.2), _learnable(-0.1, 0.1)),
        False,
        True,
    ),
    "ColorJitter": (
        lambda: K.ColorJitter(_learnable(0.8, 1.2), _learnable(0.8, 1.2), _learnable(0.8, 1.2), _learnable(-0.1, 0.1)),
        False,
        True,
    ),
    "RandomAffine": (
        lambda: K.RandomAffine(
            _learnable(-10.0, 10.0), _learnable(0.1, 0.1), _learnable(0.9, 1.1), _learnable(-5.0, 5.0), p=1.0
        ),
        False,
        True,
    ),
    "RandomRotation": (lambda: K.RandomRotation(_learnable(10.0, 20.0), p=1.0), False, True),
    "RandomShear": (lambda: K.RandomShear(_learnable(-5.0, 5.0), p=1.0), False, True),
    "RandomTranslate": (lambda: K.RandomTranslate(_learnable(-0.1, 0.1), _learnable(-0.1, 0.1), p=1.0), False, True),
    "RandomPerspective": (lambda: K.RandomPerspective(_learnable(0.5), p=1.0), False, True),
    "RandomMotionBlur": (
        lambda: K.RandomMotionBlur(3, _learnable(-30.0, 30.0), _learnable(-0.5, 0.5), p=1.0),
        False,
        True,
    ),
    "RandomResizedCrop": (
        lambda: K.RandomResizedCrop((6, 6), scale=_learnable(0.5, 1.0), ratio=_learnable(0.75, 1.33), p=1.0),
        False,
        False,
    ),
    "RandomErasing": (lambda: K.RandomErasing(_learnable(0.02, 0.33), _learnable(0.3, 3.3), p=1.0), False, False),
    "RandomMixUpV2": (lambda: K.RandomMixUpV2(_learnable(0.0, 1.0), p=1.0), False, False),
    "RandomAffine3D": (
        lambda: K.RandomAffine3D(
            _learnable(-10.0, 10.0), _learnable(0.1, 0.1, 0.1), _learnable(0.9, 1.1), _learnable(-5.0, 5.0), p=1.0
        ),
        True,
        True,
    ),
    "RandomRotation3D": (lambda: K.RandomRotation3D(_learnable(-10.0, 10.0), p=1.0), True, True),
    "RandomPerspective3D": (lambda: K.RandomPerspective3D(_learnable(0.5), p=1.0), True, True),
    "RandomMotionBlur3D": (
        lambda: K.RandomMotionBlur3D(3, _learnable(-30.0, 30.0), _learnable(-0.5, 0.5), p=1.0),
        True,
        True,
    ),
}
_DIFFERENTIABLE_RANGES = [name for name, (_, _, differentiable) in _LEARNABLE_RANGES.items() if differentiable]

# name -> (constructor, (2,) range, 0-d half-width). Ranges made directly on the test device and dtype. The four colour
# classes used to check theirs with ``_range_bound``, which kept a CPU float32 copy (or, for a half-width, a derived
# range) that the module did not own; RandomAffine and ColorJitter check theirs in the generator and are controls.
_RANGES_ON_DEVICE = {
    "RandomBrightness": (lambda r: K.RandomBrightness(r, p=1.0), (0.8, 1.2), 0.2),
    "RandomContrast": (lambda r: K.RandomContrast(r, p=1.0), (0.8, 1.2), 0.2),
    "RandomSaturation": (lambda r: K.RandomSaturation(r, p=1.0), (0.8, 1.2), 0.2),
    "RandomHue": (lambda r: K.RandomHue(r, p=1.0), (-0.1, 0.1), 0.1),
    "RandomAffine": (lambda r: K.RandomAffine(r, p=1.0), (-10.0, 10.0), 10.0),
    "ColorJitter": (lambda r: K.ColorJitter(r, p=1.0), (0.8, 1.2), 0.2),
}

# name -> (class, range name, (2,) range, 0-d half-width, the range a checkpoint holds for that half-width). Before the
# four colour classes owned a learnable range, their checkpoints held one only as the generator's ``<name>_factor``, a
# CPU float32 (2,) range, unless it was a CPU float32 (2,) parameter. For a 0-d half-width that is the range derived
# from it, with the lower end floored at the domain: the saved ranges are what
# ``RandomX(nn.Parameter(torch.tensor(half_width))).state_dict()`` gives at 0ee23ec52.
_UNOWNED_RANGE_CHECKPOINTS = {
    "RandomBrightness": (K.RandomBrightness, "brightness", (0.8, 1.2), 0.2, (0.8, 1.2)),
    "RandomContrast": (K.RandomContrast, "contrast", (0.8, 1.2), 0.3, (0.7, 1.3)),
    "RandomSaturation": (K.RandomSaturation, "saturation", (0.7, 1.2), 1.5, (0.0, 2.5)),
    "RandomHue": (K.RandomHue, "hue", (-0.1, 0.2), 0.1, (-0.1, 0.1)),
}


def _torch_save_round_trip(module: torch.nn.Module) -> torch.nn.Module:
    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


_COPIERS = {
    "deepcopy": copy.deepcopy,
    "pickle": lambda module: pickle.loads(pickle.dumps(module)),  # noqa: S301
    "torch.save": _torch_save_round_trip,
}


class TestLearnableRangeCopies(BaseTester):
    """``copy.deepcopy``, pickle and ``torch.save`` of augmentations whose range is a learnable ``nn.Parameter``."""

    @staticmethod
    def _case(name, device, dtype):
        make, volumetric, _ = _LEARNABLE_RANGES[name]
        torch.manual_seed(0)
        aug = make().to(device=device, dtype=dtype)
        shape = (2, 3, 4, 8, 8) if volumetric else (2, 3, 8, 8)
        return aug, torch.rand(shape, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)

    @staticmethod
    def _draw(module, seed, *inputs):
        torch.manual_seed(seed)
        return module(*inputs)

    @staticmethod
    def _ranges(module):
        # Read the dispatcher's parameters through its sequences, which it registers as child modules only since #6449.
        members = module.augmentations if isinstance(module, K.ManyToOneAugmentationDispather) else [module]
        return [p for member in members for p in member.parameters()]

    @pytest.mark.parametrize("name", list(_LEARNABLE_RANGES))
    def test_deepcopy_draws_like_the_original(self, name, device, dtype):
        aug, x = self._case(name, device, dtype)
        before = copy.deepcopy(aug)
        self._draw(aug, 1, x)
        after = copy.deepcopy(aug)
        expected = self._draw(aug, 2, x)
        for clone in (before, after):
            assert all(q is not p for p, q in zip(aug.parameters(), clone.parameters()))
            self.assert_close(self._draw(clone, 2, x), expected, rtol=0, atol=0)
            _assert_params_equal(clone._params, aug._params)

    @pytest.mark.parametrize("name", list(_LEARNABLE_RANGES))
    def test_deepcopy_keeps_the_last_draw_detached(self, name, device, dtype):
        aug, x = self._case(name, device, dtype)
        self._draw(aug, 1, x)
        clone = copy.deepcopy(aug)
        _assert_params_equal(clone._params, aug._params)
        assert all(value.grad_fn is None for value in clone._params.values() if isinstance(value, torch.Tensor))
        if getattr(aug, "_transform_matrix_params", None) is aug._params:
            # Records that share values in the original still share them in the copy.
            assert clone._transform_matrix_params is clone._params

    @pytest.mark.parametrize("grad_mode", ["enabled", "no_grad"])
    @pytest.mark.parametrize("copier", list(_COPIERS))
    @pytest.mark.parametrize("name", _DIFFERENTIABLE_RANGES)
    def test_gradients_reach_the_range_of_the_copy(self, name, copier, grad_mode, device, dtype):
        aug, x = self._case(name, device, dtype)
        # A copy made under ``torch.no_grad()``, as an EMA or evaluation copy often is, trains like any other.
        with torch.no_grad() if grad_mode == "no_grad" else contextlib.nullcontext():
            clone = _COPIERS[copier](aug)
        self._draw(aug, 3, x).sum().backward()
        expected = [None if p.grad is None else p.grad.clone() for p in aug.parameters()]
        assert any(grad is not None and bool(grad.ne(0).any()) for grad in expected)
        aug.zero_grad(set_to_none=True)

        self._draw(clone, 3, x).sum().backward()

        assert all(p.grad is None for p in aug.parameters())
        clone_ranges = list(clone.parameters())
        assert len(clone_ranges) == len(expected)
        for q, grad in zip(clone_ranges, expected):
            if grad is None:
                assert q.grad is None
            else:
                self.assert_close(q.grad, grad)

    def test_copy_keeps_the_state_dict(self, device, dtype):
        aug, x = self._case("RandomRotation", device, dtype)
        self._draw(aug, 1, x)
        clone = copy.deepcopy(aug)
        assert list(clone.state_dict()) == list(aug.state_dict()) == ["_param_generator.degrees"]
        for key, value in aug.state_dict().items():
            self.assert_close(clone.state_dict()[key], value, rtol=0, atol=0)
        target = _LEARNABLE_RANGES["RandomRotation"][0]().to(device=device, dtype=dtype)
        target.load_state_dict(clone.state_dict())
        self.assert_close(target._param_generator.degrees, aug._param_generator.degrees, rtol=0, atol=0)

    def test_copy_follows_device_and_dtype_moves(self, device, dtype):
        aug, x = self._case("RandomAffine", torch.device("cpu"), torch.float32)
        clone = copy.deepcopy(aug)
        aug.to(device=device, dtype=dtype)
        clone.to(device=device, dtype=dtype)
        x = x.to(device=device, dtype=dtype)
        expected = self._draw(aug, 2, x)
        actual = self._draw(clone, 2, x)
        self.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        for q in clone.parameters():
            assert q.device == x.device
            assert q.dtype == dtype
            assert q.grad is not None

    @pytest.mark.parametrize("container", ["AugmentationSequential", "ManyToOneAugmentationDispather"])
    def test_container_copy_draws_and_trains_like_the_original(self, container, device, dtype):
        def sequence():
            return K.AugmentationSequential(
                _LEARNABLE_RANGES["RandomBrightness"][0](),
                _LEARNABLE_RANGES["RandomAffine"][0](),
                data_keys=["input", "mask", "bbox_xyxy"],
            ).to(device=device, dtype=dtype)

        def outputs(out):
            # AugmentationSequential returns [image, mask, boxes]; the dispatcher returns one such list per sequence.
            return [out] if container == "AugmentationSequential" else out

        torch.manual_seed(0)
        module = (
            sequence()
            if container == "AugmentationSequential"
            else K.ManyToOneAugmentationDispather(sequence(), sequence())
        )
        image = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        mask = (image[:, :1] > 0.5).to(dtype)
        boxes = torch.tensor([[[1.0, 1.0, 5.0, 6.0]]], device=device, dtype=dtype).repeat(2, 1, 1)
        before = copy.deepcopy(module)
        self._draw(module, 1, image, mask, boxes)
        after = copy.deepcopy(module)
        expected = outputs(self._draw(module, 2, image, mask, boxes))
        sum(out[0].sum() for out in expected).backward()
        expected_grads = [p.grad.clone() for p in self._ranges(module)]
        for p in self._ranges(module):
            p.grad = None

        for clone in (before, after):
            actual = outputs(self._draw(clone, 2, image, mask, boxes))
            for actual_out, expected_out in zip(actual, expected):
                for actual_value, expected_value in zip(actual_out, expected_out):
                    self.assert_close(actual_value, expected_value, rtol=0, atol=0)
            sum(out[0].sum() for out in actual).backward()
            assert all(p.grad is None for p in self._ranges(module))
            clone_ranges = self._ranges(clone)
            assert len(clone_ranges) == len(expected_grads) == 5 * len(expected)
            for q, grad in zip(clone_ranges, expected_grads):
                self.assert_close(q.grad, grad)

    def test_averaged_model_of_a_learnable_range(self, device, dtype):
        aug, x = self._case("RandomAffine", device, dtype)
        averaged = torch.optim.swa_utils.AveragedModel(aug)
        averaged.update_parameters(aug)
        expected = self._draw(aug, 2, x)
        self.assert_close(self._draw(averaged, 2, x), expected, rtol=0, atol=0)
        for p, q in zip(aug.parameters(), averaged.module.parameters()):
            assert q is not p
            self.assert_close(q, p, rtol=0, atol=0)

    @pytest.mark.parametrize("form", ["range", "half-width"])
    @pytest.mark.parametrize("name", list(_RANGES_ON_DEVICE))
    def test_range_made_on_the_device_is_owned_and_copied(self, name, form, device, dtype):
        make, pair, half_width = _RANGES_ON_DEVICE[name]
        param = torch.nn.Parameter(torch.tensor(pair if form == "range" else half_width, device=device, dtype=dtype))
        torch.manual_seed(0)
        aug = make(param)
        owned = list(aug.parameters())
        assert len(owned) == 1
        assert owned[0] is param
        x = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)
        # The same draws as the same augmentation built from a plain tensor with these values.
        expected = self._draw(make(param.detach().clone()), 2, x)
        self.assert_close(self._draw(aug, 2, x), expected, rtol=0, atol=0)

        before = copy.deepcopy(aug)
        self._draw(aug, 1, x)
        after = copy.deepcopy(aug)
        self._draw(aug, 2, x).sum().backward()
        grad = param.grad.clone()
        assert bool(grad.ne(0).any())
        param.grad = None
        for clone in (before, after):
            (clone_param,) = clone.parameters()
            assert clone_param is not param
            actual = self._draw(clone, 2, x)
            self.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
            assert param.grad is None
            self.assert_close(clone_param.grad, grad)

        if device.type != "mps":  # MPS has no float64 to move to
            other = torch.float32 if dtype == torch.float64 else torch.float64
            aug.to(dtype=other)
            assert param.dtype == other

    def test_rgb_shift_takes_a_half_width_tensor_in_any_dtype(self, device, dtype):
        # A 0-d limit is a half-width. Its bounds are not computed from the tensor, so a value that rounds when the
        # generator samples in float32 is not reported out of its own bounds.
        limit = torch.tensor(0.3, device=device, dtype=dtype)
        aug = K.RandomRGBShift(limit, limit, limit, p=1.0)
        params = aug.forward_parameters((2, 3, 4, 4))
        for key in ("r_shift", "g_shift", "b_shift"):
            assert bool((params[key].abs() <= float(limit) * (1 + 1e-3)).all())

    @pytest.mark.parametrize("name", ["RandomBrightness", "RandomAffine"])
    def test_load_mapped_to_the_cpu_rebuilds_the_samplers_there(self, name, device, dtype):
        if device.type == "cpu":
            pytest.skip("needs a device to map from")
        # ``torch.load(map_location=...)`` moves the tensors of a module but not its generator's ``device``. The
        # samplers are rebuilt where the tensors went, so a module saved on an accelerator loads where there is none.
        aug, x = self._case(name, device, dtype)
        self._draw(aug, 1, x)
        buffer = io.BytesIO()
        torch.save(aug, buffer)
        buffer.seek(0)
        generator = torch.load(buffer, map_location="cpu", weights_only=False)._param_generator
        moved = copy.deepcopy(aug).to("cpu")._param_generator
        assert generator.device == moved.device == torch.device("cpu")
        torch.manual_seed(2)
        expected = moved(x.shape)
        torch.manual_seed(2)
        actual = generator(x.shape)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            self.assert_close(actual[key], value, rtol=0, atol=0)
        sum(value.sum() for value in actual.values() if value.is_floating_point()).backward()
        assert all(p.grad is not None and p.device.type == "cpu" for p in generator.parameters())

    @pytest.mark.parametrize("form", ["range", "half-width"])
    @pytest.mark.parametrize("name", list(_UNOWNED_RANGE_CHECKPOINTS))
    def test_strict_load_of_a_checkpoint_without_the_range(self, name, form, device, dtype):
        make, key, pair, half_width, saved_range = _UNOWNED_RANGE_CHECKPOINTS[name]
        value = torch.tensor(pair if form == "range" else half_width, device=device, dtype=dtype)
        checkpoint = {f"_param_generator.{key}_factor": torch.tensor(pair if form == "range" else saved_range)}
        aug = make(torch.nn.Parameter(value.clone()), p=1.0)
        for target in (aug, pickle.loads(pickle.dumps(aug))):  # noqa: S301
            (param,) = target.parameters()
            with torch.no_grad():
                param.zero_()
            target.load_state_dict(checkpoint)
            self.assert_close(param, value)
            # A checkpoint of the module as it is now still loads, and its own layout is kept.
            target.load_state_dict(make(torch.nn.Parameter(value.clone()), p=1.0).state_dict())
            assert set(target.state_dict()) == {key, f"_param_generator.{key}_factor"}


# name -> (constructor, key of the draw in ``_params``, (2,) range, 0-d half-width, center and floor of a half-width,
# a wider range, a wider half-width). What ``_range_bound`` derives from a half-width ``f`` is
# ``[max(center - f, floor), center + f]`` (#6572), and a draw is ``low + u * (high - low)`` for a ``u`` the seed fixes.
_FOLLOWED_RANGES = {
    "RandomBrightness": (K.RandomBrightness, "brightness_factor", (0.9, 1.1), 0.1, (1.0, 0.0), (0.5, 1.5), 0.5),
    "RandomContrast": (K.RandomContrast, "contrast_factor", (0.9, 1.1), 0.1, (1.0, 0.0), (0.5, 1.5), 0.5),
    "RandomSaturation": (K.RandomSaturation, "saturation_factor", (0.9, 1.1), 0.1, (1.0, 0.0), (0.5, 1.5), 0.5),
    "RandomHue": (K.RandomHue, "hue_factor", (-0.05, 0.05), 0.05, (0.0, -0.5), (-0.4, 0.4), 0.4),
    "RandomRGBShift": (K.RandomRGBShift, "r_shift", (-0.05, 0.05), 0.05, (0.0, float("-inf")), (-0.4, 0.4), 0.4),
    "RandomRotation": (K.RandomRotation, "degrees", (-10.0, 10.0), 10.0, (0.0, -360.0), (-40.0, 40.0), 40.0),
    "RandomAffine": (K.RandomAffine, "angle", (-10.0, 10.0), 10.0, (0.0, -360.0), (-40.0, 40.0), 40.0),
    "ColorJitter": (K.ColorJitter, "brightness_factor", (0.9, 1.1), 0.1, (1.0, 0.0), (0.5, 1.5), 0.5),
}
_RANGE_FORMS = ["range", "half-width"]


def _range_ends(value, center, floor):
    # Independent of ``_range_bound``: the two forms of a range as the issue states them.
    if value.dim() == 1:
        return value[0], value[1]
    return torch.maximum(center - value, torch.full_like(value, floor)), center + value


class _Clamped(torch.nn.Module):
    """A parametrization that keeps a range inside ``[low, high]``."""

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.clamp(self.low, self.high)


def _make_constrained(make, name, form, limit, device, dtype):
    # A range built outside ``limit``: the parametrization has to bring it back. A (2,) range is kept in
    # ``[limit / 2, limit]``, a half-width in ``[0, limit]``.
    start = (limit * 1.5, limit * 2.0) if form == "range" else limit * 1.5
    param = torch.nn.Parameter(torch.tensor(start, device=device, dtype=dtype))
    aug = make(param).to(device=device, dtype=dtype)
    low = limit / 2 if form == "range" else 0.0
    torch.nn.utils.parametrize.register_parametrization(aug._param_generator, name, _Clamped(low, limit))
    return aug


@pytest.mark.usefixtures("restore_torch_rng")
class TestLearnableRangeFollowsItsValue(BaseTester):
    """Every forward draws from the current value of a learnable range, so it trains for more than one step (#6572)."""

    @staticmethod
    def _make(name, form, device, dtype):
        make, key, pair, half_width, (center, floor), wide_pair, wide_half_width = _FOLLOWED_RANGES[name]
        value = pair if form == "range" else half_width
        wide = wide_pair if form == "range" else wide_half_width
        param = torch.nn.Parameter(torch.tensor(value, device=device, dtype=dtype))
        aug = make(param, p=1.0).to(device=device, dtype=dtype)
        return aug, param, key, (center, floor), torch.tensor(wide, device=device, dtype=dtype)

    @staticmethod
    def _skip_without_backward(name, device, dtype):
        if name in ("RandomRotation", "RandomAffine") and not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("grid_sample backward is unavailable for this device/dtype")

    def _assert_drawn_from(self, aug, key, low, high, seed=7):
        # 2048 draws of a uniform: deterministic under the seed, inside the range, and spanning nearly all of it. The
        # stale range of the issue is a fifth of the width of the wide ranges used here.
        torch.manual_seed(seed)
        drawn = aug.forward_parameters(torch.Size((2048, 3, 8, 8)))[key].detach().to(low.device)
        tolerance = (high - low).abs() * 0.01
        assert drawn.min() >= low - tolerance
        assert drawn.max() <= high + tolerance
        assert drawn.max() - drawn.min() >= (high - low) * 0.95

    @pytest.mark.parametrize("form", _RANGE_FORMS)
    @pytest.mark.parametrize("name", list(_FOLLOWED_RANGES))
    def test_second_step_draws_follow_the_optimizer(self, name, form, device, dtype):
        self._skip_without_backward(name, device, dtype)
        aug, param, key, (center, floor), _ = self._make(name, form, device, dtype)
        x = torch.rand(4, 3, 8, 8, device=device, dtype=dtype, requires_grad=True)
        # A step of a fixed size: the gradient only decides its direction, and a small one would hide a stale range.
        opt = torch.optim.SGD(aug.parameters(), lr=abs(float(_FOLLOWED_RANGES[name][3])) / 2)

        torch.manual_seed(3)
        aug(x).sum().backward()
        assert param.grad is not None
        assert torch.count_nonzero(param.grad) > 0
        first = aug._params[key].detach().clone()
        low_1, high_1 = _range_ends(param.detach().clone(), center, floor)
        param.grad.sign_()
        opt.step()
        opt.zero_grad()

        torch.manual_seed(3)
        aug(x).sum().backward()
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        low_2, high_2 = _range_ends(param.detach(), center, floor)
        expected = low_2 + (first - low_1) / (high_1 - low_1) * (high_2 - low_2)
        # A few units in the last place of the range: ``expected`` recovers the unit draw from a rounded draw. In half
        # precision that is two units on a range of 10 degrees, past the default tolerance.
        atol = 8 * torch.finfo(dtype).eps * max(abs(float(low_2)), abs(float(high_2)))
        self.assert_close(aug._params[key].detach(), expected, rtol=0, atol=atol)
        assert not torch.equal(aug._params[key].detach(), first)

    @pytest.mark.parametrize("name", _DIFFERENTIABLE_RANGES)
    def test_every_family_trains_for_two_steps(self, name, device, dtype):
        make, volumetric, _ = _LEARNABLE_RANGES[name]
        if not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("grid_sample backward is unavailable for this device/dtype")
        aug = make().to(device=device, dtype=dtype)
        shape = (2, 3, 4, 8, 8) if volumetric else (2, 3, 8, 8)
        x = torch.rand(shape, device=device, dtype=dtype, requires_grad=True)
        opt = torch.optim.SGD(aug.parameters(), lr=1e-3)
        for _ in range(3):
            aug(x).sum().backward()
            opt.step()
            opt.zero_grad()

    def test_mixup_follows_its_range(self, device, dtype):
        param = _learnable(0.0, 0.1).to(device=device, dtype=dtype)
        param = torch.nn.Parameter(param.detach())
        aug = K.RandomMixUpV2(param, p=1.0).to(device=device, dtype=dtype)
        with torch.no_grad():
            param.copy_(torch.tensor([0.5, 1.0], device=device, dtype=dtype))
        x = torch.rand(1024, 1, 2, 2, device=device, dtype=dtype, requires_grad=True)
        aug(x).sum().backward()
        drawn = aug._params["mixup_lambdas"].detach()
        assert drawn.min() >= 0.5
        assert drawn.max() <= 1.0
        assert param.grad is not None

    @pytest.mark.parametrize("how", ["in-place", "load_state_dict", "assign", "deepcopy", "pickle", "AveragedModel"])
    @pytest.mark.parametrize("form", _RANGE_FORMS)
    @pytest.mark.parametrize("name", list(_FOLLOWED_RANGES))
    def test_an_update_reaches_the_draws(self, name, form, how, device, dtype):
        self._skip_without_backward(name, device, dtype)
        aug, param, key, (center, floor), wide = self._make(name, form, device, dtype)
        aug(torch.rand(2, 3, 8, 8, device=device, dtype=dtype))  # The samplers are built and have drawn.
        target = aug
        if how == "AveragedModel":
            target = torch.optim.swa_utils.AveragedModel(aug)
        if how in ("load_state_dict", "assign"):
            state = aug.state_dict()
            state.update({k: wide.clone() for k, _ in aug.named_parameters(remove_duplicate=False)})
            aug.load_state_dict(state, assign=how == "assign")
        else:
            with torch.no_grad():
                param.copy_(wide)
        if how == "AveragedModel":
            target.update_parameters(aug)
            target = target.module
        elif how == "deepcopy":
            target = copy.deepcopy(aug)
        elif how == "pickle":
            target = pickle.loads(pickle.dumps(aug))  # noqa: S301
        low, high = _range_ends(wide, center, floor)
        self._assert_drawn_from(target, key, low, high)
        # And the update of that copy or load still has a graph to the range it now draws from.
        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype, requires_grad=True)
        for _ in range(2):
            target(x).sum().backward()
        # ``assign`` splits a parameter the module shares with its generator, as it splits any tied parameters.
        assert all(p.grad is not None for p in target._param_generator.parameters())

    @pytest.mark.parametrize("form", _RANGE_FORMS)
    def test_a_copy_follows_its_own_range(self, form, device, dtype):
        aug, param, key, (center, floor), wide = self._make("RandomRotation", form, device, dtype)
        clone = copy.deepcopy(aug)
        (clone_param,) = clone.parameters()
        with torch.no_grad():
            clone_param.copy_(wide)
        low, high = _range_ends(wide, center, floor)
        self._assert_drawn_from(clone, key, low, high)
        low, high = _range_ends(param.detach(), center, floor)
        self._assert_drawn_from(aug, key, low, high)

    @pytest.mark.parametrize("name, key", [("RandomRotation", "degrees"), ("RandomAffine", "angle")])
    def test_plain_leaf_that_requires_grad_trains_too(self, name, key, device, dtype):
        self._skip_without_backward(name, device, dtype)
        leaf = torch.tensor(10.0, device=device, dtype=dtype, requires_grad=True)
        aug = getattr(K, name)(leaf, p=1.0).to(device=device, dtype=dtype)
        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype, requires_grad=True)
        aug(x).sum().backward()
        with torch.no_grad():
            leaf.copy_(torch.tensor(40.0, device=device, dtype=dtype))
        aug(x).sum().backward()
        self._assert_drawn_from(aug, key, -leaf.detach(), leaf.detach())
        if name == "RandomAffine":
            return  # its generator keeps no range in ``state_dict``
        # A load writes the leaf in place, and the generator keeps drawing from it with its graph.
        leaf.grad = None
        aug.load_state_dict({"_param_generator.degrees": torch.tensor(20.0)})
        aug(x).sum().backward()
        assert leaf.grad is not None
        self._assert_drawn_from(aug, key, -leaf.detach(), leaf.detach())

    def test_range_outside_its_domain_keeps_training_moving_and_copying(self, device, dtype):
        # The domain is checked when the range is first built; reading a range on every forward would wait for the
        # GPU, and refusing to move or copy a model because of one step of its optimizer would lose the run.
        param = torch.nn.Parameter(torch.tensor(0.9, device=device, dtype=dtype))
        aug = K.RandomBrightness(param, p=1.0).to(device=device, dtype=dtype)
        with torch.no_grad():
            param.fill_(1.5)  # [-0.5, 2.5], past the (0, 2) bounds of the constructor
        self._assert_drawn_from(aug, "brightness_factor", torch.zeros_like(param), 1 + param.detach())
        self._assert_drawn_from(copy.deepcopy(aug), "brightness_factor", torch.zeros_like(param), 1 + param.detach())
        clone = pickle.loads(pickle.dumps(aug))  # noqa: S301
        self._assert_drawn_from(clone, "brightness_factor", torch.zeros_like(param), 1 + param.detach())
        if device.type != "mps":  # MPS has no float64 to move to
            moved = clone.to(dtype=torch.float32 if dtype == torch.float64 else torch.float64)
            self._assert_drawn_from(moved, "brightness_factor", torch.tensor(0.0).to(param), 1 + param.detach())
        # A new module with that range is still refused.
        with pytest.raises(ValueError, match="out of bounds"):
            K.RandomBrightness(torch.nn.Parameter(param.detach().clone()), p=1.0)

    @pytest.mark.parametrize(
        "make, key, value",
        [
            (K.ColorJitter, "brightness_factor", (-0.5, 1.5)),
            (K.RandomGamma, "gamma_factor", (-0.5, 1.5)),
            (K.RandomRotation, "degrees", (-400.0, 400.0)),
            (K.RandomAffine, "angle", (-400.0, 400.0)),
            (K.RandomShear, "shear_x", (5.0, -5.0)),
        ],
    )
    def test_range_outside_its_domain_keeps_drawing(self, make, key, value, device, dtype):
        # Each validator a generator calls directly is off in a rebuild, as well as those in ``_range_bound``.
        low, high = value
        start = (0.5, 1.0) if make in (K.ColorJitter, K.RandomGamma) else (-1.0, 1.0)
        param = torch.nn.Parameter(torch.tensor(start, device=device, dtype=dtype))
        aug = make(param, p=1.0).to(device=device, dtype=dtype)
        with torch.no_grad():
            param.copy_(torch.tensor(value, device=device, dtype=dtype))
        torch.manual_seed(0)
        drawn = aug.forward_parameters(torch.Size((2048, 3, 8, 8)))[key].detach()
        assert drawn.max() - drawn.min() >= abs(high - low) * 0.9

    def test_load_of_a_learnable_range_outside_its_domain_draws_from_it(self, device, dtype):
        # The ``ValueError`` of a load is for a range that is not learnable: a trained one may be out of its domain.
        param = torch.nn.Parameter(torch.tensor(10.0, device=device, dtype=dtype))
        aug = K.RandomRotation(param, p=1.0).to(device=device, dtype=dtype)
        aug.load_state_dict({"_param_generator.degrees": torch.tensor(500.0)})
        torch.manual_seed(0)
        drawn = aug.forward_parameters(torch.Size((2048, 3, 8, 8)))["degrees"].detach()
        assert drawn.max() > 360.0

    def test_volumetric_range_outside_its_domain_keeps_drawing(self, device, dtype):
        param = torch.nn.Parameter(torch.tensor(10.0, device=device, dtype=dtype))
        aug = K.RandomRotation3D(param, p=1.0).to(device=device, dtype=dtype)
        with torch.no_grad():
            param.fill_(400.0)
        torch.manual_seed(0)
        drawn = aug.forward_parameters(torch.Size((2048, 3, 4, 8, 8)))["yaw"].detach()
        assert drawn.max() - drawn.min() >= 800.0 * 0.9

    @pytest.mark.parametrize("form", _RANGE_FORMS)
    @pytest.mark.parametrize("name", ["RandomBrightness", "RandomRotation", "RandomAffine", "ColorJitter"])
    def test_range_on_an_accelerator_with_the_module_on_the_cpu_follows_too(self, name, form, device, dtype):
        if device.type == "cpu":
            pytest.skip("needs a range on another device than the module")
        self._skip_without_backward(name, device, dtype)
        make, key, pair, half_width, (center, floor), wide_pair, wide_half_width = _FOLLOWED_RANGES[name]
        param = torch.nn.Parameter(torch.tensor(pair if form == "range" else half_width, device=device, dtype=dtype))
        wide = torch.tensor(wide_pair if form == "range" else wide_half_width, device=device, dtype=dtype)
        aug = make(param, p=1.0)
        assert aug._param_generator.device == torch.device("cpu")
        opt = torch.optim.SGD([param], lr=1e-3)
        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        for _ in range(2):
            aug(x).sum().backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            param.copy_(wide)
        low, high = _range_ends(wide, center, floor)
        self._assert_drawn_from(aug, key, low.cpu(), high.cpu())
        aug.load_state_dict(
            {k: wide.clone() * 0.5 for k, _ in aug.named_parameters(remove_duplicate=False)}, strict=False
        )
        low, high = _range_ends(wide * 0.5, center, floor)
        self._assert_drawn_from(aug, key, low.cpu(), high.cpu())

        # ``assign`` replaces the parameter, and a new one can have the version counter of the old one.
        aug = make(
            torch.nn.Parameter(torch.tensor(pair if form == "range" else half_width, device=device, dtype=dtype)), p=1.0
        )
        aug(x)
        state = {k: wide.clone() for k, _ in aug.named_parameters(remove_duplicate=False)}
        aug.load_state_dict(state, strict=False, assign=True)
        low, high = _range_ends(wide, center, floor)
        self._assert_drawn_from(aug, key, low.cpu(), high.cpu())

    @pytest.mark.parametrize("volumetric", [False, True])
    def test_rebuild_adds_no_state_dict_key(self, volumetric, device, dtype):
        # Assigning the range itself to an attribute registers it, if the rebuild is the first time their dtypes match.
        other = torch.float32 if dtype == torch.float64 else torch.float64
        make = K.RandomPerspective3D if volumetric else K.RandomPerspective
        aug = make(torch.nn.Parameter(torch.tensor(0.3, dtype=other)), p=1.0).to(device=device, dtype=dtype)
        keys = list(aug.state_dict())
        shape = (2, 3, 4, 8, 8) if volumetric else (2, 3, 8, 8)
        aug(torch.rand(shape, device=device, dtype=dtype))
        assert list(aug.state_dict()) == keys

    @pytest.mark.parametrize("placement", ["same device", "module on the cpu"])
    def test_draws_outside_grad_mode_do_not_stick(self, placement, device, dtype):
        # With the module on the CPU, the range is copied only when something changed since the last draw. A draw under
        # ``no_grad`` after an optimizer step built samplers without a graph, and the next training forward kept them.
        if placement == "module on the cpu" and device.type == "cpu":
            pytest.skip("needs a range on another device than the module")
        aug, param, _, _, _ = self._make("RandomBrightness", "half-width", device, dtype)
        if placement == "module on the cpu":
            aug = K.RandomBrightness(param, p=1.0)
            assert aug._param_generator.device == torch.device("cpu")
        opt = torch.optim.SGD([param], lr=1e-3)
        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        for _ in range(2):
            with torch.no_grad():
                aug(x)
            with torch.inference_mode():
                aug(x)
            aug(x).sum().backward()
            assert param.grad is not None
            assert torch.count_nonzero(param.grad) > 0
            opt.step()
            opt.zero_grad(set_to_none=True)

    def test_a_failed_rebuild_leaves_validation_on(self, device, dtype):
        param = torch.nn.Parameter(torch.tensor(10.0, device=device, dtype=dtype))
        aug = K.RandomRotation(param, p=1.0).to(device=device, dtype=dtype)
        with (
            patch.object(UniformDistribution, "__init__", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            aug(torch.rand(1, 3, 8, 8, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="out of bounds"):
            K.RandomRotation((-500.0, 500.0))

    @staticmethod
    def _out_of_domain(device, dtype):
        param = torch.nn.Parameter(torch.tensor(10.0, device=device, dtype=dtype))
        aug = K.RandomRotation(param, p=1.0).to(device=device, dtype=dtype)
        with torch.no_grad():
            param.fill_(400.0)  # A validator that runs on this raises.
        return aug

    def test_a_rebuild_leaves_other_threads_validating(self, device, dtype):
        # Another thread's constructor and checkpoint load, while this thread's rebuild is inside ``make_samplers``.
        aug = self._out_of_domain(device, dtype)
        inside, release = threading.Event(), threading.Event()
        make_samplers = K.random_generator.PlainUniformGenerator.make_samplers

        def blocked(generator, *args):
            if generator is aug._param_generator:
                inside.set()
                release.wait(30)
            make_samplers(generator, *args)

        errors = []

        def draw():
            try:
                aug.forward_parameters(torch.Size((2, 3, 8, 8)))
            except Exception as error:
                errors.append(error)

        with patch.object(K.random_generator.PlainUniformGenerator, "make_samplers", blocked):
            thread = threading.Thread(target=draw)
            thread.start()
            try:
                assert inside.wait(30)
                with pytest.raises(ValueError, match="out of bounds"):
                    K.RandomRotation((-500.0, 500.0))
                with pytest.raises(ValueError, match="out of bounds"):
                    K.RandomRotation((-10.0, 10.0)).load_state_dict(
                        {"_param_generator.degrees": torch.tensor([-500.0, 500.0])}
                    )
            finally:
                release.set()
                thread.join()
        assert not errors

    def test_overlapping_rebuilds_do_not_end_each_other(self, device, dtype):
        # A second thread's rebuild ends inside the first one's: the first must not start validating its range.
        first, second = self._out_of_domain(device, dtype), self._out_of_domain(device, dtype)
        inside, release = threading.Event(), threading.Event()
        make_samplers = K.random_generator.PlainUniformGenerator.make_samplers

        def blocked(generator, *args):
            if generator is first._param_generator:
                inside.set()
                release.wait(30)
            make_samplers(generator, *args)

        errors = []

        def draw():
            try:
                first.forward_parameters(torch.Size((2, 3, 8, 8)))
            except Exception as error:
                errors.append(error)

        with patch.object(K.random_generator.PlainUniformGenerator, "make_samplers", blocked):
            thread = threading.Thread(target=draw)
            thread.start()
            try:
                assert inside.wait(30)
                second.forward_parameters(torch.Size((2, 3, 8, 8)))
            finally:
                release.set()
                thread.join()
        assert not errors

    @pytest.mark.parametrize(
        "name",
        ["RandomBrightness", "RandomTranslate", "RandomAffine3D isotropic scale", "RandomAffine3D scale per axis"],
    )
    def test_a_draw_during_a_rebuild_finds_every_sampler(self, name, device, dtype):
        # One module shared by threads: a draw that runs while another thread rebuilds the samplers finds them all, old
        # or new. The rebuild used to empty them or set them to ``None`` first, which raised a ``KeyError`` or drew
        # zero translations and unit scales. The draw is made here from inside the rebuild, at a helper it calls.
        from kornia.augmentation.random_generator._2d import plain_uniform, translate
        from kornia.augmentation.random_generator._3d import affine

        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        if name == "RandomBrightness":
            aug = K.RandomBrightness(torch.nn.Parameter(torch.tensor(0.2, device=device, dtype=dtype)), p=1.0)
            module, helper = plain_uniform, "_range_bound"
        elif name == "RandomTranslate":
            param = torch.nn.Parameter(torch.tensor([-0.2, 0.2], device=device, dtype=dtype))
            aug = K.RandomTranslate(param, (-0.2, 0.2), p=1.0)
            module, helper = translate, "_range_bound"
        else:
            scale = (0.5, 1.5) if name.endswith("isotropic scale") else ((0.5, 1.5),) * 3
            degrees = torch.nn.Parameter(torch.tensor(10.0, device=device, dtype=dtype))
            aug = K.RandomAffine3D(degrees, translate=(0.1, 0.1, 0.1), scale=scale, p=1.0)
            module, helper = affine, "_as_range_tensor"
            x = torch.rand(2, 3, 4, 8, 8, device=device, dtype=dtype)
        aug = aug.to(device=device, dtype=dtype)
        aug(x)
        generator = aug._param_generator
        draws = []

        def drawing(*args, **kwargs):
            draws.append(generator.forward(x.shape))
            return original(*args, **kwargs)

        original = getattr(module, helper)
        with patch.object(module, helper, drawing):
            aug(x)
        assert draws
        for drawn in draws:
            if name == "RandomBrightness":
                assert "brightness_factor" in drawn
            elif name == "RandomTranslate":
                assert torch.count_nonzero(drawn["translate_x"]) > 0
                assert torch.count_nonzero(drawn["translate_y"]) > 0
            else:
                assert torch.count_nonzero(drawn["translations"]) > 0
                assert bool((drawn["scale"] != 1).any())

    def test_a_nested_rebuild_restores_the_outer_one(self, device, dtype):
        outer, inner = self._out_of_domain(device, dtype), self._out_of_domain(device, dtype)
        make_samplers = K.random_generator.PlainUniformGenerator.make_samplers

        def nested(generator, *args):
            if generator is outer._param_generator:
                inner.forward_parameters(torch.Size((2, 3, 8, 8)))
            make_samplers(generator, *args)

        with patch.object(K.random_generator.PlainUniformGenerator, "make_samplers", nested):
            outer.forward_parameters(torch.Size((2, 3, 8, 8)))
        with pytest.raises(ValueError, match="out of bounds"):
            K.RandomRotation((-500.0, 500.0))

    @pytest.mark.parametrize("volumetric, make", [(False, K.RandomRotation), (False, K.RandomBrightness)])
    def test_range_that_is_not_learnable_pays_nothing(self, volumetric, make, device, dtype):
        aug = make((0.8, 1.1), p=1.0).to(device=device, dtype=dtype)
        generator = aug._param_generator
        assert not generator._forward_pre_hooks
        samplers = dict(generator.sampler_dict)
        aug(torch.rand(2, 3, 8, 8, device=device, dtype=dtype))
        assert all(generator.sampler_dict[name] is sampler for name, sampler in samplers.items())

    @pytest.mark.parametrize("name", ["RandomBrightness", "RandomRotation"])
    def test_generator_pickled_without_the_hook_follows_its_range(self, name, device, dtype):
        # What a pickle from before #6572 unpickles to: no forward pre-hook on the generator.
        aug, _, key, (center, floor), wide = self._make(name, "range", device, dtype)
        aug._param_generator._forward_pre_hooks.clear()
        clone = pickle.loads(pickle.dumps(aug))  # noqa: S301
        (clone_param,) = clone.parameters()
        with torch.no_grad():
            clone_param.copy_(wide)
        self._assert_drawn_from(clone, key, *_range_ends(wide, center, floor))

    @pytest.mark.parametrize(
        "make, key, grad_key",
        [
            (K.RandomRotation, "degrees", "_param_generator.degrees"),
            (partial(K.RandomBrightness, clip_output=False), "brightness_factor", "_param_generator.brightness_factor"),
        ],
    )
    @pytest.mark.parametrize("value", [(0.8, 1.1), 0.1], ids=_RANGE_FORMS)
    def test_gradcheck_of_the_range(self, value, make, key, grad_key, device):
        param = torch.nn.Parameter(torch.tensor(value, device=device, dtype=torch.float64))
        aug = make(param, p=1.0).to(device=device, dtype=torch.float64)
        x = (torch.rand(1, 1, 5, 5, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.2).to(device, torch.float64)

        def draw(range_value):
            torch.manual_seed(5)
            return torch.func.functional_call(aug, {grad_key: range_value}, (x,))

        # On a stale range the output ignores ``range_value``, which gradcheck would accept as a zero Jacobian.
        assert torch.count_nonzero(torch.autograd.functional.jacobian(draw, param.detach().clone())) > 0
        self.gradcheck(draw, (param.detach().clone(),), fast_mode=False)

    @pytest.mark.parametrize(
        "make, loaded, read",
        [
            (
                _LEARNABLE_RANGES["RandomErasing"][0],
                {"scale": (0.25, 0.25), "ratio": (1.0, 1.0)},
                lambda params: (params["widths"], params["heights"]),
            ),
            (
                _LEARNABLE_RANGES["RandomResizedCrop"][0],
                {"scale": (0.25, 0.25), "ratio": (1.0, 1.0)},
                lambda params: (
                    params["src"][:, 1, 0] - params["src"][:, 0, 0] + 1,
                    params["src"][:, 2, 1] - params["src"][:, 1, 1] + 1,
                ),
            ),
            (
                lambda: K.RandomCutMixV2(cut_size=_learnable(0.2, 0.8), p=1.0, use_correct_lambda=True),
                {"cut_size": (0.75, 0.75)},
                lambda params: (params["crop_src"][..., 1, 0] - params["crop_src"][..., 0, 0] + 1,),
            ),
        ],
        ids=["RandomErasing", "RandomResizedCrop", "RandomCutMixV2"],
    )
    def test_generator_that_reads_its_range_gets_it_from_a_load(self, make, loaded, read, device, dtype):
        # A quarter of the area of an 8 x 8 image with a ratio of 1 is a 4 x 4 box, and a lambda of 0.75 cuts 4 pixels.
        aug = make().to(device=device, dtype=dtype)
        generator = aug._param_generator
        assert not generator._forward_pre_hooks
        state = aug.state_dict()
        for key in state:
            state[key] = torch.tensor(loaded[key.split(".")[-1].lstrip("_")], device=device, dtype=dtype)
        aug.load_state_dict(state)
        params = aug.forward_parameters(torch.Size((64, 3, 8, 8)))
        for size in read(params):
            self.assert_close(size.detach(), torch.full_like(size, 4.0))

    @pytest.mark.parametrize(
        "make, key, expected",
        [
            (lambda r: K.RandomPosterize(r, p=1.0), "bits_factor", (5.0, 5.0)),
            (lambda r: K.RandomRotation90(r, p=1.0), "times", (1.0, 1.0)),
        ],
        ids=["RandomPosterize", "RandomRotation90"],
    )
    def test_whole_number_generator_gets_its_range_from_a_load(self, make, key, expected, device, dtype):
        start = (2.0, 6.0) if key == "bits_factor" else (0.0, 0.0)
        aug = make(torch.nn.Parameter(torch.tensor(start, device=device, dtype=dtype))).to(device=device, dtype=dtype)
        assert not aug._param_generator._forward_pre_hooks
        aug.load_state_dict({f"_param_generator.{key}": torch.tensor(expected)})
        drawn = aug.forward_parameters(torch.Size((64, 3, 8, 8)))[key].detach()
        assert drawn.min() == expected[0]
        assert drawn.max() == expected[1]

    @pytest.mark.parametrize("updated", [(0.5, 1.5), (1.0, 1.0)], ids=["fractional", "odd"])
    def test_rotation90_in_place_update_leaves_its_decisions_consistent(self, updated, device, dtype):
        # The bounds, the whole-range rule and the non-square check all come from the range as it was built. Refreshing
        # the bounds alone drew 0 from a range of [0.5, 1.5], and a quarter turn silently cropped a non-square image.
        param = torch.nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
        aug = K.RandomRotation90(param, p=1.0).to(device=device, dtype=dtype)
        with torch.no_grad():
            param.copy_(torch.tensor(updated, device=device, dtype=dtype))
        image = torch.rand(1, 1, 4, 8, device=device, dtype=dtype)
        aug(image)
        assert aug._params["times"].detach().unique().tolist() == [0.0]
        assert aug._param_generator._whole_range
        # A load brings all of them over together.
        with torch.no_grad():
            param.zero_()
        aug.load_state_dict({"_param_generator.times": torch.tensor(updated)})
        with pytest.raises(ValueError, match="quarter turns"):
            aug(image)

    @pytest.mark.parametrize("copier", list(_COPIERS))
    @pytest.mark.parametrize("updated", [(0.25, 0.25), (1.0, 1.0), (0.5, 1.5)], ids=["fixed", "odd", "fractional"])
    def test_rotation90_copy_of_an_in_place_update_decides_like_a_fresh_module(self, updated, copier, device, dtype):
        # A copy of a learnable range draws from the range as it is now, so the whole-range rule and the non-square
        # check have to come from it too. They used to be copied: a copy of [0.25, 0.25] drew 0 three times in four,
        # and a copy of [1, 1] gave a non-square image a quarter turn that cropped it.
        copy_of = _COPIERS[copier]
        param = torch.nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
        aug = K.RandomRotation90(param, p=1.0).to(device=device, dtype=dtype)
        with torch.no_grad():
            param.copy_(torch.tensor(updated, device=device, dtype=dtype))
        copied = copy_of(aug)
        fresh = K.RandomRotation90(torch.nn.Parameter(torch.tensor(updated, device=device, dtype=dtype)), p=1.0).to(
            device=device, dtype=dtype
        )
        assert copied._param_generator._whole_range == fresh._param_generator._whole_range
        assert copied._allows_odd_turns == fresh._allows_odd_turns
        torch.manual_seed(0)
        drawn = copied.forward_parameters(torch.Size((256, 1, 4, 4)))["times"]
        torch.manual_seed(0)
        self.assert_close(drawn, fresh.forward_parameters(torch.Size((256, 1, 4, 4)))["times"], rtol=0, atol=0)
        if fresh._allows_odd_turns:
            with pytest.raises(ValueError, match="quarter turns"):
                copied(torch.rand(1, 1, 4, 8, device=device, dtype=dtype))

    def test_rotation90_copy_of_a_buffer_written_in_place_draws_like_the_original(self, device, dtype):
        # A buffer that is not learnable keeps its samplers in a copy, and with them its decisions: the copy still draws
        # no turn, so it must still take a non-square image.
        aug = K.RandomRotation90((0, 0), p=1.0).to(device=device, dtype=dtype)
        aug._param_generator.times.fill_(1.0)
        copied = copy.deepcopy(aug)
        image = torch.rand(1, 1, 4, 8, device=device, dtype=dtype)
        self.assert_close(copied(image), aug(image), rtol=0, atol=0)
        assert copied._params["times"].unique().tolist() == [0.0]

    def test_rotation90_loads_a_range_without_float64(self, device, monkeypatch):
        # MPS has no float64: a loaded range is on the device by then, so it cannot be converted to find if it is whole.
        dtype = torch.float32
        aug = K.RandomRotation90((0, 0), p=1.0).to(device=device, dtype=dtype)
        original = torch.Tensor.to

        def to_without_float64(tensor, *args, **kwargs):
            if torch.float64 in args or kwargs.get("dtype") == torch.float64:
                raise TypeError("Cannot convert to float64, as on MPS")
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", to_without_float64)
        aug.load_state_dict({"_param_generator.times": torch.tensor([0.5, 1.5], device=device, dtype=dtype)})
        assert not aug._param_generator._whole_range

    @pytest.mark.parametrize(
        "make, name, key, limit",
        [
            (lambda r: K.RandomRotation(r, p=1.0), "degrees", "degrees", 10.0),
            (lambda r: K.RandomAffine(r, p=1.0), "degrees", "angle", 10.0),
            (lambda r: K.RandomBrightness(r, p=1.0), "brightness_factor", "brightness_factor", 0.1),
            (lambda r: K.ColorJitter(r, p=1.0), "brightness", "brightness_factor", 0.1),
        ],
        ids=["RandomRotation", "RandomAffine", "RandomBrightness", "ColorJitter"],
    )
    @pytest.mark.parametrize("form", _RANGE_FORMS)
    def test_parametrization_constrains_the_draws(self, make, name, key, limit, form, device, dtype):
        # The docs recommend one for a range that training must keep inside its domain. It moves the parameter out of
        # ``_parameters``, and the attribute is the constrained value.
        aug = _make_constrained(make, name, form, limit, device, dtype)
        center = 1.0 if "brightness" in name else 0.0
        torch.manual_seed(0)
        drawn = aug.forward_parameters(torch.Size((2048, 3, 8, 8)))[key].detach()
        # A few units in the last place of the drawn values, which half precision needs around a center of 1.
        slack = limit * 0.01 + 4 * torch.finfo(dtype).eps * max(center, limit)
        if form == "half-width":
            assert (drawn - center).abs().max() <= limit + slack
            assert drawn.max() - drawn.min() >= 2 * limit * 0.9
        else:
            assert drawn.min() >= limit / 2 - slack
            assert drawn.max() <= limit + slack
        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype, requires_grad=True)
        if name == "degrees" and not supports_bilinear_2d_grid_sample_backward(device, dtype):
            return
        for _ in range(2):
            aug(x).sum().backward()

    @pytest.mark.parametrize(
        "make, name, shape",
        [
            (lambda r: K.RandomPerspective(r, p=1.0), "distortion_scale", (2, 3, 8, 8)),
            (lambda r: K.RandomPerspective3D(r, p=1.0), "distortion_scale", (2, 3, 4, 8, 8)),
        ],
        ids=["RandomPerspective", "RandomPerspective3D"],
    )
    def test_parametrization_of_a_range_kept_by_the_generator(self, make, name, shape, device, dtype):
        # These generators assign the range itself to an attribute, which a parametrization's output cannot replace.
        param = torch.nn.Parameter(torch.tensor(0.9, device=device, dtype=dtype))
        aug = make(param).to(device=device, dtype=dtype)
        generator = aug._param_generator
        torch.nn.utils.parametrize.register_parametrization(generator, name, _Clamped(0.0, 0.1))
        aug(torch.rand(shape, device=device, dtype=dtype))
        self.assert_close(generator._distortion_scale.detach(), torch.tensor(0.1, device=device, dtype=dtype))

    def test_parametrization_of_the_ranges_of_the_volumetric_affine(self, device, dtype):
        aug = K.RandomAffine3D(10.0, _learnable(0.9, 0.9, 0.9), p=1.0).to(device=device, dtype=dtype)
        generator = aug._param_generator
        torch.nn.utils.parametrize.register_parametrization(generator, "translate", _Clamped(0.0, 0.1))
        aug(torch.rand(2, 3, 4, 8, 8, device=device, dtype=dtype))
        self.assert_close(generator._translate.detach(), torch.full((3,), 0.1, device=device, dtype=dtype))

    def test_replaced_parameter_with_the_module_on_the_cpu_is_followed(self, device, dtype):
        # The version counter of a new parameter can equal that of the old one.
        if device.type == "cpu":
            pytest.skip("needs a range on another device than the module")
        aug = K.RandomRotation3D(torch.nn.Parameter(torch.tensor(10.0, device=device, dtype=dtype)), p=1.0)
        aug(torch.rand(2, 3, 4, 8, 8, device=device, dtype=dtype))
        aug._param_generator.degrees = torch.nn.Parameter(torch.tensor(40.0, device=device, dtype=dtype))
        torch.manual_seed(0)
        yaw = aug.forward_parameters(torch.Size((1024, 3, 4, 8, 8)))["yaw"].detach()
        assert yaw.max() - yaw.min() >= 80.0 * 0.9

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_compile_of_a_learnable_range_trains_and_follows_it(
        self, device, dtype, torch_optimizer, optimizer_backend
    ):
        if optimizer_backend == "jit":
            pytest.skip("TorchScript cannot script an augmentation")
        if not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("grid_sample backward is unavailable for this device/dtype")
        for name in ("RandomBrightness", "RandomRotation", "RandomAffine"):
            for form in _RANGE_FORMS:
                aug, param, key, (center, floor), wide = self._make(name, form, device, dtype)
                # ``fullgraph``: a graph break in the rebuild raises here instead of running that part eagerly.
                compiled = torch_optimizer(aug, fullgraph=True)
                x = torch.rand(1024, 1, 4, 4, device=device, dtype=dtype, requires_grad=True)
                if name == "RandomBrightness":
                    x = torch.rand(1024, 3, 4, 4, device=device, dtype=dtype, requires_grad=True)
                compiled(x).sum().backward()
                assert param.grad is not None
                param.grad = None
                with torch.no_grad():
                    param.copy_(wide)
                compiled(x).sum().backward()
                assert param.grad is not None
                if key not in aug._params:
                    continue  # Dynamo in torch 2.5 does not write ``_params`` back; the graph above held in full.
                drawn = aug._params[key].detach()
                low, high = _range_ends(wide, center, floor)
                assert drawn.min() >= low - (high - low) * 0.01
                assert drawn.max() <= high + (high - low) * 0.01
                assert drawn.max() - drawn.min() >= (high - low) * 0.9


@pytest.mark.usefixtures("restore_torch_rng")
class TestLoadedRangeReachesTheDraws(BaseTester):
    """``load_state_dict`` of a range buffer that is not learnable reaches the draws, at load time only (#6572)."""

    # name -> (module of a narrow range, module of a far range, key of the draw, ends of the far range)
    _CASES = {
        "RandomRotation": (
            lambda: K.RandomRotation((80.0, 81.0), p=1.0),
            lambda: K.RandomRotation((10.0, 11.0), p=1.0),
            "degrees",
            (10.0, 11.0),
        ),
        "RandomBrightness": (
            lambda: K.RandomBrightness((1.5, 1.5), p=1.0),
            lambda: K.RandomBrightness((0.1, 0.2), p=1.0),
            "brightness_factor",
            (0.1, 0.2),
        ),
        "half-width": (
            lambda: K.RandomContrast(0.1, p=1.0),
            lambda: K.RandomContrast(0.5, p=1.0),
            "contrast_factor",
            (0.5, 1.5),
        ),
        "tensor": (
            lambda: K.RandomRotation(torch.tensor(5.0), p=1.0),
            lambda: K.RandomRotation(torch.tensor(40.0), p=1.0),
            "degrees",
            (-40.0, 40.0),
        ),
        "RandomRotation90": (
            lambda: K.RandomRotation90((0, 0), p=1.0),
            lambda: K.RandomRotation90((2, 2), p=1.0),
            "times",
            (2.0, 2.0),
        ),
    }

    @pytest.mark.parametrize("name", list(_CASES))
    def test_load_reaches_the_draws(self, name, device, dtype):
        target, source, key, (low, high) = self._CASES[name]
        target, source = target().to(device=device, dtype=dtype), source()
        target.load_state_dict(source.state_dict())
        torch.manual_seed(0)
        drawn = target.forward_parameters(torch.Size((1024, 3, 8, 8)))[key]
        assert drawn.min() >= low - 1e-2
        assert drawn.max() <= high + 1e-2
        assert drawn.max() - drawn.min() >= (high - low) * 0.9
        # The generator's ``repr`` shows the range it draws from, and so does the augmentation's.
        (loaded,) = target.state_dict().values()
        (entry,) = [factor for factor, *_ in target._param_generator.samplers]
        self.assert_close(torch.as_tensor(entry, device=loaded.device, dtype=loaded.dtype), loaded)
        assert f"{entry}" in repr(target._param_generator)
        assert f"{entry}" in repr(target)

    # name -> (module of the given range, a narrow range, a far range exact in float32, key of the draw)
    _MOVED_CASES = {
        "tensors": (
            lambda gamma: K.RandomGamma(torch.tensor(gamma), torch.tensor([0.9, 1.1]), p=1.0),
            [0.8, 1.2],
            [0.5, 0.625],
            "gamma_factor",
        ),
        "three tensors": (
            lambda r: K.RandomRGBShift(torch.tensor(r), torch.tensor([-0.1, 0.1]), torch.tensor([-0.1, 0.1]), p=1.0),
            [-0.1, 0.1],
            [-0.375, -0.25],
            "r_shift",
        ),
        "numbers": (lambda degrees: K.RandomRotation(tuple(degrees), p=1.0), [-80.0, 80.0], [10.0, 11.0], "degrees"),
    }

    @pytest.mark.parametrize("order", ["move, load, move back", "load, move"])
    @pytest.mark.parametrize("name", list(_MOVED_CASES))
    def test_load_of_one_range_of_several_keeps_up_with_moves(self, name, order, device, dtype):
        # The buffer of a loaded range moves with the module, while the other entries stay as they were given. Put next
        # to them, it made every forward of a moved module refuse their mix of devices or dtypes, and it left the draws
        # behind on a later move. A loaded module draws like one built with the loaded range and moved the same way.
        if device.type == "cpu" and dtype == torch.float32:
            pytest.skip("nothing moves")
        make, narrow, far, key = self._MOVED_CASES[name]
        target, reference, state = make(narrow), make(far), make(far).state_dict()
        if order == "load, move":
            target.load_state_dict(state)
            target.to(device=device, dtype=dtype)
            reference.to(device=device, dtype=dtype)
        else:
            target.to(device=device, dtype=dtype)
            target.load_state_dict(state)
            target.to(device="cpu", dtype=torch.float32)
        torch.manual_seed(0)
        drawn = target.forward_parameters(torch.Size((64, 3, 8, 8)))[key]
        torch.manual_seed(0)
        expected = reference.forward_parameters(torch.Size((64, 3, 8, 8)))[key]
        self.assert_close(drawn, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("source", [0.2, (-0.2, 0.2)], ids=["half-width", "pair"])
    def test_rgb_shift_loads_a_wider_valid_range(self, source, device, dtype):
        # A number limit used to bound the range by itself, so a valid checkpoint of a wider shift was refused.
        target = K.RandomRGBShift(0.1, 0.1, 0.1, p=1.0).to(device=device, dtype=dtype)
        target.load_state_dict(K.RandomRGBShift(source, 0.1, 0.1, p=1.0).state_dict())
        torch.manual_seed(0)
        drawn = target.forward_parameters(torch.Size((1024, 3, 8, 8)))["r_shift"]
        assert drawn.min() >= -0.2 - 1e-3
        assert drawn.max() <= 0.2 + 1e-3
        assert drawn.max() - drawn.min() >= 0.4 * 0.9

    def test_load_inside_a_container_follows_the_prefix(self, device, dtype):
        target = K.AugmentationSequential(K.RandomRotation((80.0, 81.0), p=1.0)).to(device=device, dtype=dtype)
        source = K.AugmentationSequential(K.RandomRotation((10.0, 11.0), p=1.0))
        target.load_state_dict(source.state_dict())
        target(torch.rand(4, 3, 8, 8, device=device, dtype=dtype))
        degrees = target._params[0].data["degrees"]
        assert degrees.min() >= 10.0 - 1e-3
        assert degrees.max() <= 11.0 + 1e-3

    def test_load_of_the_same_range_changes_nothing(self, device):
        # The sampler keeps the given numbers in the generator's dtype: 0.1 exactly in float64, not 0.1 rounded to the
        # float32 of the buffer. A checkpoint that holds the same range, in either dtype, must keep them.
        if device.type == "mps":
            pytest.skip("MPS has no float64, and the exact double is the point of this test")
        target = K.RandomRotation((-0.1, 0.1), p=1.0).to(device=device, dtype=torch.float64)
        samplers = dict(target._param_generator.sampler_dict)
        low = samplers["degrees"].low.item()
        target.load_state_dict(K.RandomRotation((-0.1, 0.1), p=1.0).state_dict())
        target.load_state_dict(target.state_dict())
        assert target._param_generator.sampler_dict["degrees"] is samplers["degrees"]
        assert low == -0.1

    @pytest.mark.parametrize("assign", [False, True])
    def test_load_of_an_invalid_range_is_refused_like_the_constructor(self, assign, device, dtype):
        target = K.RandomRotation((-10.0, 10.0), p=1.0)
        with pytest.raises(ValueError, match="out of bounds"):
            target.load_state_dict({"_param_generator.degrees": torch.tensor([-500.0, 500.0])}, assign=assign)
        # The refused range is not kept: the buffer goes back to the range it had, and the samplers, which used to be
        # left empty, keep it. A move used to raise again from the refused value in the buffer.
        assert target.state_dict()["_param_generator.degrees"].tolist() == [-10.0, 10.0]
        drawn = target.forward_parameters(torch.Size((1024, 3, 8, 8)))["degrees"]
        assert drawn.abs().max() <= 10.0
        moved = torch.float16 if device.type == "mps" else (torch.float32 if dtype == torch.float64 else torch.float64)
        target.to(device=device, dtype=moved)
        drawn = target.forward_parameters(torch.Size((1024, 3, 8, 8)))["degrees"]
        assert drawn.abs().max() <= 10.0
        assert drawn.max() - drawn.min() >= 20.0 * 0.9

    def test_rotation90_load_updates_what_it_derives_from_the_range(self, device, dtype):
        # A whole range floors its draws and only allows odd turns on a square input when it can draw one.
        target = K.RandomRotation90((0, 0), p=1.0).to(device=device, dtype=dtype)
        target(torch.rand(1, 3, 4, 8, device=device, dtype=dtype))
        target.load_state_dict(K.RandomRotation90((1, 1), p=1.0).state_dict())
        with pytest.raises(ValueError, match="square"):
            target(torch.rand(1, 3, 4, 8, device=device, dtype=dtype))
        assert target._param_generator._whole_range
        target.load_state_dict({"_param_generator.times": torch.tensor([0.5, 1.5])})
        assert not target._param_generator._whole_range
