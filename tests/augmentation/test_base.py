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
import inspect
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

from testing.base import BaseTester


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

# Representative constructible augmentations for the statelessness pin. The blanket claim "no augmentation
# registers a buffer" is false for 16 of the 69 concrete classes (they keep their sampling range in a
# `_param_generator.*` buffer, see `test_wart_param_generator_range_buffers_are_inert_4428`), so this list
# is a representative set, not a sweep over the package.
_STATELESS_REPRESENTATIVES = {
    "RandomHorizontalFlip": lambda: K.RandomHorizontalFlip(p=1.0),
    "RandomAffine": lambda: K.RandomAffine(degrees=(45.0, 45.0), p=1.0),
    "ColorJiggle": lambda: K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0),
    "Normalize": lambda: K.Normalize(0.5, 0.5, p=1.0),
    "AugmentationSequential": lambda: K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0)),
}

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
        [("rand", (12,))],
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

    def test_convention_integer_input_is_rejected(self, device):
        # Convention pin: the dtype policy is enforced - an integer image raises `TypeError` naming the four
        # accepted float dtypes, for `uint8` and `int64` alike.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu):
        # "Expected input of [torch.bfloat16, torch.float16, torch.float32, torch.float64]. Got torch.uint8".
        for bad in (torch.uint8, torch.int64):
            with pytest.raises(TypeError, match="Expected input of"):
                K.RandomHorizontalFlip(p=1.0)(torch.zeros(1, 1, 4, 4, device=device, dtype=bad))

    def test_wart_wrong_input_rank_raises_three_exception_types_4424(self, device, dtype):
        # Wart pin (#4424): one user error - the wrong input rank - raises three different exception types
        # depending on which entry point sees it. The class path raises `ValueError` from `transform_tensor`,
        # the container path raises `RuntimeError` with a different message, and `validate_tensor`'s own
        # `RuntimeError` (which also rejects the legal (C, H, W) rank) is unreachable from `forward`.
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
        flip = K.RandomHorizontalFlip  # one of the two concrete classes that accept `p_batch` (see #4425)
        assert flip(p=1.0, p_batch=0.0).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist() == [0.0] * 4
        assert flip(p=0.0, p_batch=1.0).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist() == [0.0] * 4
        assert flip(p=1.0, p_batch=1.0).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist() == [1.0] * 4
        # `p_batch=0.5`: every seed gives an all-0 or an all-1 row - one draw for the batch, seeds 0..7 give
        # [1,1,1,1], [0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0].
        batch_rows = []
        for seed in range(8):
            torch.manual_seed(seed)
            batch_rows.append(flip(p=1.0, p_batch=0.5).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist())
        assert batch_rows == [[1.0] * 4, [0.0] * 4, [0.0] * 4, [1.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4]
        # `p=0.5` is per sample: seeds 0..3 give [1,0,1,1], [0,1,1,0], [0,1,0,1], [1,1,1,1] - mixed rows exist.
        sample_rows = []
        for seed in range(4):
            torch.manual_seed(seed)
            sample_rows.append(flip(p=0.5).forward_parameters((4, 3, 6, 8))["batch_prob"].tolist())
        assert sample_rows == [[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]

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

    def test_wart_p_batch_is_accepted_by_only_two_concrete_classes_4425(self):
        # Wart pin (#4425): `p_batch` is documented on the bases and reaches the constructor of exactly two of
        # the 69 concrete `kornia.augmentation` classes - `RandomHorizontalFlip` and `RandomVerticalFlip`.
        # Every other class raises `TypeError`, and `RandomDissolving` swallows it through `**kwargs`.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0): 78 exported
        # augmentation classes, 9 bases, 69 concrete, accepted by ['RandomHorizontalFlip', 'RandomVerticalFlip'].
        classes = [n for n in K.__all__ if isinstance(getattr(K, n, None), type)]
        classes = [n for n in classes if issubclass(getattr(K, n), _BasicAugmentationBase)]
        bases = [n for n in classes if n.endswith(("Base2D", "Base3D", "BaseV2"))]
        concrete = [n for n in classes if n not in bases]
        accepted = [n for n in concrete if "p_batch" in inspect.signature(getattr(K, n).__init__).parameters]
        assert len(concrete) == 69
        assert accepted == ["RandomHorizontalFlip", "RandomVerticalFlip"]
        with pytest.raises(TypeError, match="unexpected keyword argument 'p_batch'"):
            K.RandomAffine(degrees=45.0, p_batch=0.5)
        assert repr(K.RandomHorizontalFlip(p=1.0, p_batch=0.5)).startswith(
            "RandomHorizontalFlip(p=1.0, p_batch=0.5, same_on_batch=False)"
        )

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

    def test_wart_random_plasma_replay_4445(self, device, dtype):
        # Wart pin (#4445): the three `RandomPlasma*`
        # classes draw their fractal noise inside `apply_transform` from the global generator instead of in
        # `generate_parameters`, so `params=` replay is NOT bitwise (RandomDissolving also samples during
        # application). Replaying the same `params=` under
        # the same global seed is reproducible; replaying it without reseeding is not.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): replay
        # max|difference| 0.5715941786766052, two reseeded replays bitwise equal.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("RandomPlasmaBrightness builds its fractal noise in float32")
        aug = K.RandomPlasmaBrightness(p=1.0)
        x = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        torch.manual_seed(0)
        first = aug(x)
        params = aug._params
        replayed = aug(x, params=params)
        assert not torch.equal(first, replayed)
        torch.manual_seed(5)
        one = aug(x, params=params)
        torch.manual_seed(5)
        two = aug(x, params=params)
        assert torch.equal(one, two)

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
        # carrying its last `_params`. (16 of the 69 concrete classes DO register a range buffer; that
        # deviation is pinned in `test_wart_param_generator_range_buffers_are_inert_4428`.)
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): 0 state_dict
        # entries, no buffers, no parameters, pickle and deepcopy OK for each representative.
        aug = _STATELESS_REPRESENTATIVES[name]()
        assert len(aug.state_dict()) == 0
        assert list(aug.buffers()) == []
        assert list(aug.parameters()) == []
        aug(torch.rand(2, 3, 6, 8))
        for clone in (pickle.loads(pickle.dumps(aug)), copy.deepcopy(aug)):  # noqa: S301
            assert isinstance(clone, type(aug))
            assert len(clone._params) == len(aug._params)

    def test_convention_parameter_valued_ranges_receive_gradients(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the CPU rotation backward requires float32 or float64")
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
        # Wart pin (#4428): the 16 classes that keep their sampling range in a `_param_generator.*` buffer
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
        # Wart pin (#4429): `B = 0` is not uniformly "empty in, empty out" - 20 classes raise on it, in six
        # exception families. These four are one class per family, all raw internal errors rather than a
        # validation message.
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

    @pytest.mark.xfail(strict=True, reason="Tracked in #4429")
    def test_convention_zero_batch_is_empty_in_empty_out(self, device, dtype):
        # Strict xfail (#4429): the settled convention for a degenerate batch is the #4115 rule - empty in,
        # empty out - which the flip, the affine, an intensity op and `AugmentationSequential` already follow
        # (pinned above). This asserts the intended behavior for the 20 classes that raise instead; it XFAILs
        # today and turns XPASS the moment the repair lands, which is the signal to delete the wart pin above.
        # Executed 2026-09-11 (torch 2.14.0, cpu): the first builder already raises IndexError.
        empty = torch.rand(0, 3, 6, 8, device=device, dtype=dtype)
        for aug in (
            K.RandomCrop((4, 6), p=1.0),
            K.LongestMaxSize(16, p=1.0),
            K.RandomAutoContrast(p=1.0),
            K.Normalize(0.5, 0.5, p=1.0),
        ):
            assert aug(empty).shape[0] == 0
