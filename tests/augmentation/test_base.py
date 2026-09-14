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

    @pytest.mark.parametrize("keepdim", (True, False))
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
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("RandomMixUpV2 promotes half inputs because mixup_lambdas are float32")

        input = torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 2, 2)
        augmentation = RandomMixUpV2(lambda_val=(0.25, 0.25), p=1.0, data_keys=["input"])
        params = self._cpu_partial_batch_params(augmentation, input)
        params["mixup_pairs"] = torch.tensor([1, 0])

        output = augmentation(input, params=params)

        expected = torch.stack([input[0] * 0.75 + input[1] * 0.25, input[1]])
        assert output.device == input.device
        self.assert_close(output, expected)


# Batch-6 convention pins for the `kornia.augmentation` shared contract (anchor 1: `AugmentationBase2D`).
# Every literal below was generated by the body of the pin that carries it, executed on this worktree on
# 2026-09-11 with `.venv/bin/python` (torch 2.14.0, python 3.11, cpu, float32) unless the comment says
# otherwise. Fixtures are asymmetric (H != W, a hot pixel off both centre lines) and every random draw is
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
        # One draw per patch row: B * rows * columns = 4 * 2 * 2 (B * C = 12 before #4460 fixed #4421).
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu):
        # (1, 3, 6, 8) / (3, 6, 8) / (1, 1, 6, 8) / (2, 3, 6, 8).
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu):
        # "Expected input of [torch.bfloat16, torch.float16, torch.float32, torch.float64]. Got torch.uint8".
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu).
        # The repair window picks one exception type; do not "correct" this pin here.
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0) for a float64
        # input: every `_params` entry `cpu`, the float entries `torch.float32`, output and matrix float64.
        # Only the `--dtype=float64` leg separates the two dtypes; on a float32 leg (and on mps, which has no
        # float64) the dtype half is a tautology and the device half is what bites.
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu).
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
            kwargs = {"use_correct_lambda": True} if augmentation_cls is K.RandomCutMixV2 else {}
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
    def test_convention_transplantation_requires_masks_and_supports_mask_only(
        self, augmentation_cls, spatial_shape, device
    ):
        # A mask is required to choose the transplant, but images are optional. Three donors make the cyclic
        # donor direction observable: p=1 replaces every acceptor with its preceding donor.
        mask = torch.arange(3, device=device, dtype=torch.int64).reshape(3, *((1,) * len(spatial_shape)))
        mask = mask.expand(3, *spatial_shape)
        augmentation = augmentation_cls(p=1.0)
        output = augmentation(mask, data_keys=["mask"])
        self.assert_close(output, torch.roll(mask, 1, dims=0))
        image = torch.zeros((3, 3, *spatial_shape), device=device)
        with pytest.raises(IndexError, match="tuple index out of range"):
            augmentation(image)

    def test_convention_same_on_batch_collapses_the_draw_to_one_sample(self):
        # Convention pin: `same_on_batch=True` collapses every per-sample draw - the `p` gate included - to a
        # single value broadcast over the batch. Same seeds as the pin above, whose `p=0.5` rows are mixed.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): seeds 0..3
        # give [1,1,1,1], [0,0,0,0], [0,0,0,0], [1,1,1,1], and the angles are constant across the batch.
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): True.
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): TypeError,
        # TypeError, output shape (2, 3, 6, 8), and the two seeded outputs bitwise equal.
        with pytest.raises(TypeError, match="unexpected keyword argument 'generator'"):
            K.RandomAffine(degrees=45.0, generator=torch.Generator())
        with pytest.raises(TypeError, match="unexpected keyword argument 'generator'"):
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): bitwise True,
        # `aug._params is params` True, no key added, no value mutated.
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

    def test_wart_intensity_container_boxes_passthrough_but_direct_dispatch_raises_4480(self, device, dtype):
        image = torch.ones(1, 1, 4, 4, device=device, dtype=dtype)
        boxes = Boxes.from_tensor(torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        augmentation = K.RandomInvert(p=1.0)
        augmentation(image)
        with pytest.raises(NotImplementedError):
            augmentation.transform_boxes(boxes, augmentation._params, augmentation.flags)
        container = K.AugmentationSequential(K.RandomInvert(p=1.0), data_keys=["input", "bbox_xyxy"])
        _, output_boxes = container(image, boxes)
        self.assert_close(output_boxes.data, boxes.data)

    def test_wart_container_skips_custom_rigid_box_handlers_4481(self, device, dtype):
        class ShiftBoxes(K.RigidAffineAugmentationBase2D):
            def compute_transformation(self, input, params, flags):
                return self.identity_matrix(input)

            def apply_transform(self, input, params, flags, transform=None):
                return input

            def apply_transform_box(self, input, params, flags, transform=None):
                return Boxes(input.data + 1, mode=input.mode)

        image = torch.ones(1, 1, 4, 4, device=device, dtype=dtype)
        boxes = Boxes.from_tensor(torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        augmentation = ShiftBoxes(p=1.0)
        direct = augmentation.transform_boxes(boxes, augmentation.forward_parameters(image.shape), augmentation.flags)
        self.assert_close(direct.data, boxes.data + 1)
        _, output_boxes = K.AugmentationSequential(augmentation, data_keys=["input", "bbox_xyxy"])(image, boxes)
        self.assert_close(output_boxes.data, boxes.data)

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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): 0 state_dict
        # entries, no buffers, no parameters, pickle and deepcopy OK for each representative.
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): the buffer
        # moves [80.0, 81.0] -> [10.0, 11.0] while the draw stays in [80, 81]
        # ([80.30742645263672, 80.63407897949219, 80.49009704589844, 80.89644622802734]) and the repr still
        # says degrees=(80.0, 81.0).
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

    def test_convention_sampler_precision_is_separate_from_returned_dtype_4426(self):
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

    def test_convention_set_rng_device_moves_sampling_before_casting_parameters_4426(self, device):
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
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): (0, 3, 6, 8)
        # from all four.
        if device.type == "mps" and torch_version_lt(2, 6, 0):
            pytest.skip("torch 2.5.1 MPS asserts on an empty placeholder tensor inside grid_sample")
        empty = torch.rand(0, 3, 6, 8, device=device, dtype=dtype)
        assert K.RandomHorizontalFlip(p=1.0)(empty).shape == (0, 3, 6, 8)
        assert K.RandomAffine(degrees=(45.0, 45.0), p=1.0)(empty).shape == (0, 3, 6, 8)
        assert K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0)(empty).shape == (0, 3, 6, 8)
        container = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input"])
        assert container(empty).shape == (0, 3, 6, 8)

    @pytest.mark.parametrize(
        "name,error,message",
        [
            ("RandomCrop", IndexError, "list index out of range"),
            ("LongestMaxSize", KeyError, "output_size"),
            ("RandomAutoContrast", ValueError, "Invalid input tensor, it is empty."),
            ("Normalize", RuntimeError, "cannot reshape tensor of 0 elements"),
        ],
    )
    def test_wart_zero_batch_raises_in_four_exception_families_4429(self, name, error, message, device, dtype):
        # Wart pin (#4429): `B = 0` is not uniformly "empty in, empty out". These representative
        # augmentations raise through distinct failure paths; RandomAutoContrast is a deliberate validation
        # error and the others expose implementation details.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu):
        # RandomCrop IndexError('list index out of range'), LongestMaxSize KeyError('output_size'),
        # RandomAutoContrast ValueError('Invalid input tensor, it is empty.'),
        # Normalize RuntimeError('cannot reshape tensor of 0 elements into shape [0, 3, -1] ...').
        builders = {
            "RandomCrop": lambda: K.RandomCrop((4, 6), p=1.0),
            "LongestMaxSize": lambda: K.LongestMaxSize(16, p=1.0),
            "RandomAutoContrast": lambda: K.RandomAutoContrast(p=1.0),
            "Normalize": lambda: K.Normalize(0.5, 0.5, p=1.0),
        }
        with pytest.raises(error, match=re.escape(message)):
            builders[name]()(torch.rand(0, 3, 6, 8, device=device, dtype=dtype))

    @pytest.mark.parametrize(
        ("augmentation", "shape"),
        [
            pytest.param(
                lambda: K.RandomCrop((4, 6), p=1.0),
                (0, 3, 4, 6),
                marks=pytest.mark.xfail(strict=True, raises=IndexError, reason="Tracked in #4429"),
            ),
            pytest.param(
                lambda: K.LongestMaxSize(16, p=1.0),
                (0, 3, 12, 16),
                marks=pytest.mark.xfail(strict=True, raises=KeyError, reason="Tracked in #4429"),
            ),
            pytest.param(
                lambda: K.RandomAutoContrast(p=1.0),
                (0, 3, 6, 8),
                marks=pytest.mark.xfail(strict=True, raises=ValueError, reason="Tracked in #4429"),
            ),
            pytest.param(
                lambda: K.Normalize(0.5, 0.5, p=1.0),
                (0, 3, 6, 8),
                marks=pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Tracked in #4429"),
            ),
        ],
    )
    def test_convention_zero_batch_is_empty_in_empty_out(self, augmentation, shape, device, dtype):
        # Strict xfail (#4429): each listed augmentation must preserve an empty batch with its intended
        # output geometry. Per-class markers turn an unrelated repair into a failure rather than an XFAIL.
        empty = torch.rand(0, 3, 6, 8, device=device, dtype=dtype)
        assert augmentation()(empty).shape == shape
