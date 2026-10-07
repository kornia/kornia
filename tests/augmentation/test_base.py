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

import copy
import pickle
import re
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
from kornia.augmentation._3d.geometric.horizontal_flip import RandomHorizontalFlip3D
from kornia.augmentation._3d.intensity.motion_blur import RandomMotionBlur3D
from kornia.augmentation.base import _BasicAugmentationBase
from kornia.augmentation.container.params import ParamItem
from kornia.core import ImageModule
from kornia.core._compat import torch_version_lt
from kornia.filters.dissolving import StableDiffusionDissolving, _DissolvingWraper_HF
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from testing.base import BaseTester, supports_bilinear_2d_grid_sample_backward, supports_reflect_padding


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
# sampling range in a `_param_generator.*` buffer; see `test_wart_param_generator_range_buffers_are_inert_4428`.
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

    def test_wart_wrong_input_rank_has_entry_point_specific_errors_4424(self, device, dtype):
        # Wart pin (#4424): one user error can raise different exception types depending on which entry point
        # sees it. The class path raises `ValueError` from `transform_tensor`, the container path raises
        # `RuntimeError` with a different message, and `validate_tensor`'s own
        # `RuntimeError` (which also rejects the legal (C, H, W) rank). `forward` promotes legal unbatched
        # inputs before validation.
        x5 = torch.rand(1, 2, 3, 6, 8, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Input size must have a shape of either"):
            K.RandomHorizontalFlip(p=1.0)(x5)
        with pytest.raises(RuntimeError, match=r"input shape expected to be in \(2, 4\)"):
            K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input"])(x5)
        with pytest.raises(RuntimeError, match=r"Expect \(B, C, H, W\)"):
            K.RandomHorizontalFlip(p=1.0).validate_tensor(x5)
        with pytest.raises(RuntimeError, match=r"Expect \(B, C, H, W\)"):
            # a rank `forward` accepts, rejected by the validator behind it
            K.RandomHorizontalFlip(p=1.0).validate_tensor(torch.rand(3, 6, 8, device=device, dtype=dtype))

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
    def test_wart_mixup_cutmix_and_patchmix_map_p_to_a_batch_gate_4425(self, augmentation_cls):
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
        # `test_wart_param_generator_range_buffers_are_inert_4428`.
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

    def test_wart_param_generator_range_buffers_are_inert_4428(self):
        # Wart pin (#4428): classes that keep their sampling range in a `_param_generator.*` buffer
        # expose it in `state_dict()` (and, inside a container, under a prefixed key), but the buffer is a
        # dead copy: `load_state_dict` from an instance with a different range changes the buffer and changes
        # neither the draw nor the `repr`, so a `state_dict` round trip is a silent no-op.
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
        assert bool(((drawn >= 80.0) & (drawn <= 81.0)).all())
        assert "degrees=(80.0, 81.0)" in repr(target)

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

    def test_wart_zero_batch_raises_a_validation_error_4429(self, device, dtype):
        # Wart pin (#4429): `B = 0` is not uniformly "empty in, empty out". RandomAutoContrast raises its
        # deliberate validation error; the int-size resizes are pinned in the convention test below.
        with pytest.raises(ValueError, match=re.escape("Invalid input tensor, it is empty.")):
            K.RandomAutoContrast(p=1.0)(torch.rand(0, 3, 6, 8, device=device, dtype=dtype))

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
                marks=pytest.mark.xfail(strict=True, raises=ValueError, reason="Tracked in #4429"),
            ),
        ],
    )
    def test_convention_zero_batch_is_empty_in_empty_out(self, augmentation, shape, device, dtype):
        # Strict xfail (#4429): each listed augmentation must preserve an empty batch with its intended
        # output geometry. Per-class markers turn an unrelated repair into a failure rather than an XFAIL.
        empty = torch.rand(0, 3, 6, 8, device=device, dtype=dtype)
        assert augmentation()(empty).shape == shape


class TestHandMadeBatchProbWithStaticP(BaseTester):
    """A hand-made ``batch_prob`` is the gate whatever ``p`` is (#5585).

    ``p=1.0, p_batch=1.0`` takes a fast path that skips the non-transform branch and the blend, which is right for
    the all-ones gate that ``forward_parameters`` samples. A caller that replaces ``params["batch_prob"]`` used to
    have it ignored by the image and the transformation matrix but honored by ``inverse``, so an image row, its
    matrix and its labels disagreed about whether the row was transformed.
    """

    GATE = (0.0, 1.0, 0.0)

    @pytest.mark.parametrize(
        ("augmentation_cls", "kwargs"),
        [
            pytest.param(K.RandomAffine, {"degrees": 30.0}, id="RandomAffine"),
            pytest.param(K.RandomRotation, {"degrees": 45.0}, id="RandomRotation"),
            pytest.param(K.RandomHorizontalFlip, {}, id="RandomHorizontalFlip"),
            pytest.param(K.RandomPerspective, {"distortion_scale": 0.5}, id="RandomPerspective"),
            pytest.param(K.RandomCrop, {"size": (8, 12), "padding": (1, 2, 3, 0)}, id="RandomCrop-equal-size-padded"),
            pytest.param(K.RandomResizedCrop, {"size": (8, 12)}, id="RandomResizedCrop"),
            pytest.param(K.RandomBrightness, {"brightness": (0.5, 1.5)}, id="RandomBrightness"),
            pytest.param(K.RandomGaussianBlur, {"kernel_size": (3, 3), "sigma": (0.5, 1.5)}, id="RandomGaussianBlur"),
            pytest.param(K.RandomErasing, {}, id="RandomErasing"),
        ],
    )
    def test_skipped_rows_keep_the_input_and_the_rest_match_the_general_path(
        self, augmentation_cls, kwargs, device, dtype
    ):
        torch.manual_seed(0)
        x = torch.rand(3, 3, 8, 12, device=device, dtype=dtype)
        gate = torch.tensor(self.GATE)
        skipped, applied = gate == 0, gate == 1

        static = augmentation_cls(p=1.0, **kwargs)
        # `p < 1` never takes the fast path, so it is the reference for what the gate means.
        general = augmentation_cls(p=0.999, **kwargs)
        params = general.forward_parameters(x.shape)
        params["batch_prob"] = gate.clone()

        out_static = static(x, params=copy.deepcopy(params))
        out_general = general(x, params=copy.deepcopy(params))

        self.assert_close(out_static, out_general)
        assert torch.equal(out_static[skipped], x[skipped])
        assert not torch.equal(out_static[applied], x[applied])  # the gate selected a row that really changes

        matrix = getattr(static, "transform_matrix", None)
        if matrix is not None:
            self.assert_close(matrix, general.transform_matrix)
            if matrix.shape[-2:] == (3, 3):
                eye = torch.eye(3, device=matrix.device, dtype=matrix.dtype).expand(int(skipped.sum()), -1, -1)
                self.assert_close(matrix[skipped], eye)

    def test_issue_5585_round_trip(self, device, dtype):
        img = torch.rand(2, 1, 6, 10, device=device, dtype=dtype)
        aug = K.RandomAffine(90, p=1.0)
        params = aug.forward_parameters(img.shape)
        params["batch_prob"] = torch.tensor([0.0, 1.0])

        out = aug(img, params=copy.deepcopy(params))
        inv = aug.inverse(out, params=copy.deepcopy(params))

        assert torch.equal(out[0], img[0])  # forward leaves the skipped row alone ...
        assert torch.equal(inv[0], out[0])  # ... and so does inverse
        assert not torch.equal(out[1], img[1])

    @pytest.mark.parametrize("augmentation_cls", [K.RandomHorizontalFlip, K.RandomVerticalFlip])
    def test_round_trip_restores_the_batch(self, augmentation_cls, device, dtype):
        x = torch.rand(3, 3, 8, 12, device=device, dtype=dtype)
        aug = augmentation_cls(p=1.0)
        params = aug.forward_parameters(x.shape)
        params["batch_prob"] = torch.tensor(self.GATE)

        out = aug(x, params=copy.deepcopy(params))
        self.assert_close(aug.inverse(out, params=copy.deepcopy(params)), x)

    @pytest.mark.parametrize(
        ("augmentation_cls", "kwargs"),
        [
            pytest.param(K.RandomAffine3D, {"degrees": (30.0, 30.0, 30.0)}, id="RandomAffine3D"),
            pytest.param(K.RandomDepthicalFlip3D, {}, id="RandomDepthicalFlip3D"),
        ],
    )
    def test_3d_skipped_rows_keep_the_input(self, augmentation_cls, kwargs, device, dtype):
        torch.manual_seed(0)
        x = torch.rand(3, 1, 4, 6, 6, device=device, dtype=dtype)
        gate = torch.tensor(self.GATE)
        static, general = augmentation_cls(p=1.0, **kwargs), augmentation_cls(p=0.999, **kwargs)
        params = general.forward_parameters(x.shape)
        params["batch_prob"] = gate.clone()

        out_static = static(x, params=copy.deepcopy(params))

        self.assert_close(out_static, general(x, params=copy.deepcopy(params)))
        assert torch.equal(out_static[gate == 0], x[gate == 0])
        assert not torch.equal(out_static[gate == 1], x[gate == 1])

    def test_labels_follow_the_same_gate_as_the_image(self, device, dtype):
        torch.manual_seed(0)
        x = torch.rand(3, 1, 8, 12, device=device, dtype=dtype)
        mask = (torch.rand(3, 1, 8, 12, device=device, dtype=dtype) > 0.5).to(dtype)
        points = torch.tensor([[[2.0, 3.0]]] * 3, device=device, dtype=dtype)
        quads = torch.tensor([[[[2.0, 3.0], [6.0, 3.0], [6.0, 5.0], [2.0, 5.0]]]] * 3, device=device, dtype=dtype)
        gate = torch.tensor(self.GATE)

        module = K.RandomAffine(30.0, p=1.0)
        aug = K.AugmentationSequential(module, data_keys=["input", "keypoints", "bbox", "mask"])
        params = module.forward_parameters(x.shape)
        params["batch_prob"] = gate.clone()

        out_x, out_points, out_boxes, out_mask = aug(
            x,
            Keypoints(points.clone()),
            Boxes(quads.clone()),
            mask.clone(),
            params=[ParamItem("RandomAffine_0", params)],
        )

        skipped, applied = gate == 0, gate == 1
        assert torch.equal(out_x[skipped], x[skipped])
        assert torch.equal(out_points.data[skipped], points[skipped])
        assert torch.equal(out_boxes.data[skipped], quads[skipped])
        assert torch.equal(out_mask[skipped], mask[skipped])
        # The selected row moved in all four, so the checks above are not vacuous.
        assert not torch.equal(out_x[applied], x[applied])
        assert not torch.equal(out_points.data[applied], points[applied])
        assert not torch.equal(out_boxes.data[applied], quads[applied])
        # The matrix a container composes is the identity on the skipped rows.
        self.assert_close(module.transform_matrix[skipped], torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1))

    @pytest.mark.parametrize(
        ("augmentation_cls", "kwargs"),
        [
            pytest.param(K.Resize, {"size": (4, 6)}, id="Resize"),
            pytest.param(K.RandomCrop, {"size": (4, 6)}, id="RandomCrop"),
        ],
    )
    def test_shape_changing_augmentation(self, augmentation_cls, kwargs, device, dtype):
        x = torch.rand(3, 3, 8, 12, device=device, dtype=dtype)
        aug = augmentation_cls(p=1.0, **kwargs)
        params = aug.forward_parameters(x.shape)

        # A mixed gate cannot keep the skipped rows at their shape, as for `p < 1` (#4497): it raises instead of
        # transforming every row.
        mixed = copy.deepcopy(params)
        mixed["batch_prob"] = torch.tensor(self.GATE)
        with pytest.raises(ValueError, match="mixes applied and skipped rows"):
            aug(x, params=mixed)

        # A gate that skips every row leaves the batch untouched.
        none = copy.deepcopy(params)
        none["batch_prob"] = torch.zeros(3)
        assert torch.equal(aug(x, params=none), x)

    def test_default_gate_still_takes_the_fast_path(self, device, dtype):
        x = torch.rand(3, 3, 8, 12, device=device, dtype=dtype)
        aug = K.RandomHorizontalFlip(p=1.0)
        params = aug.forward_parameters(x.shape)
        assert params["batch_prob"].tolist() == [1.0, 1.0, 1.0]

        with patch.object(aug, "apply_non_transform", wraps=aug.apply_non_transform) as non_transform:
            aug(x, params=copy.deepcopy(params))
            non_transform.assert_not_called()  # the skipped branch and the blend are still elided

            params["batch_prob"] = torch.tensor([1.0, 0.0, 1.0])
            aug(x, params=params)
            non_transform.assert_called_once()

    def test_params_without_a_gate_are_all_applied(self):
        aug = K.RandomHorizontalFlip(p=1.0)
        assert aug._is_always_applied({})
        assert aug._is_always_applied({"batch_prob": torch.ones(0)})  # an empty batch has nothing to skip

    @pytest.mark.parametrize("p, p_batch", [(0.5, 1.0), (1.0, 0.5), (0.0, 1.0)])
    def test_non_static_probabilities_never_take_the_fast_path(self, p, p_batch):
        aug = K.RandomHorizontalFlip(p=p, p_batch=p_batch)
        assert not aug._is_always_applied({"batch_prob": torch.ones(3)})

    @pytest.mark.parametrize("flag", ["is_compiling", "is_exporting"])
    def test_capture_decides_by_the_static_probabilities_alone(self, flag):
        # Under torch.compile / export the gate cannot be read without a graph break, so the hand-made gate is
        # honored in eager mode only. `object()` has no ordering: reading it would raise.
        aug = K.RandomHorizontalFlip(p=1.0)
        with patch(f"kornia.augmentation.base.{flag}", return_value=True):
            assert aug._is_always_applied({"batch_prob": object()})
        assert not K.RandomHorizontalFlip(p=0.5)._is_always_applied({"batch_prob": object()})
