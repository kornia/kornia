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

"""Explicit augmentation RNG streams and their isolation from global state."""

import concurrent.futures
import copy
import inspect
import io
import pickle
from unittest import mock

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation.base import _BasicAugmentationBase
from kornia.augmentation.random_generator import AffineGenerator, Rotation90Generator
from kornia.augmentation.utils import _adapted_rsampling

from testing.augmentation.sync_registry import AUG_CASES, EXCLUDED
from testing.base import BaseTester


def _equal_params(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal_params(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal_params(a, b)
    else:
        assert left == right


_CASES = [case for case in AUG_CASES if case.kind in ("2d", "3d", "mix") and case.name in K.__all__]


class TestExplicitGenerator(BaseTester):
    def test_registry_covers_exported_augmentations(self):
        """Every concrete built-in with stream support has a parameter-sampling case."""
        exported = {
            name
            for name in K.__all__
            if inspect.isclass(aug := getattr(K, name))
            and issubclass(aug, _BasicAugmentationBase)
            and EXCLUDED.get(name) != "abstract base"
            and aug._supports_explicit_generator
        }
        assert {case.name for case in _CASES} == exported

    @pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
    @pytest.mark.parametrize("same_on_batch", [False, True])
    @pytest.mark.parametrize("sampling_device", ["cpu", "cuda"])
    def test_parameter_draws_match_global_stream_without_consuming_it(
        self, case, same_on_batch, sampling_device, dtype
    ):
        """An explicit stream replaces the generator, not the sampling algorithm or draw order."""
        if sampling_device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA sampling needs a CUDA device")
        aug = case.build()
        aug.p = 0.5
        aug.same_on_batch = same_on_batch
        aug.set_rng_device_and_dtype(torch.device(sampling_device), dtype)
        shape = torch.Size((4, 3, 8, 32, 32) if case.kind == "3d" else (4, 3, 32, 32))
        torch.manual_seed(123)
        # Some CUDA-configured samplers mix global CPU and CUDA draws; an explicit stream owns both.
        kwargs = (
            {"generator": torch.Generator(device=sampling_device).manual_seed(123)} if sampling_device == "cuda" else {}
        )
        expected = aug.forward_parameters(shape, **kwargs)
        torch.manual_seed(789)
        before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        actual = aug.forward_parameters(shape, generator=torch.Generator(device=sampling_device).manual_seed(123))
        _equal_params(actual, expected)
        assert torch.equal(before, torch.random.get_rng_state())
        if cuda_before is not None:
            assert torch.equal(cuda_before, torch.cuda.get_rng_state())
        assert aug._generator is None

    @pytest.mark.parametrize(
        ("factory", "parameter"),
        [
            pytest.param(lambda: K.RandomAffine(180.0, p=1.0), "angle", id="affine"),
            pytest.param(lambda: K.RandomRotation90((0, 3), p=1.0), "times", id="rotation90"),
        ],
    )
    def test_issue_reproduction_4427(self, factory, parameter, device, dtype):
        """Call-local streams control the sampled transform and reproduce it from the same seed."""
        x = torch.linspace(0, 1, 256, device=device, dtype=dtype).reshape(4, 1, 8, 8)
        aug = factory()
        a = aug(x, generator=torch.Generator().manual_seed(1))
        draw = aug._params[parameter].clone()
        b = aug(x, generator=torch.Generator().manual_seed(2))
        assert not torch.equal(draw, aug._params[parameter])
        assert not torch.equal(a, b)
        self.assert_close(a, aug(x, generator=torch.Generator().manual_seed(1)))

    @pytest.mark.parametrize("sampling_device", ["cpu", "cuda"])
    def test_nested_random_apply_and_replay(self, sampling_device, device, dtype):
        """Order selection, gates, noise, and nested children share the call's stream."""
        if sampling_device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA streams need a CUDA device")
        x = torch.linspace(0, 1, 768, device=device, dtype=dtype).reshape(4, 3, 8, 8)
        seq = K.AugmentationSequential(
            K.ImageSequential(
                K.RandomHorizontalFlip(p=0.5), K.RandomGaussianNoise(p=1.0), K.RandomRotation90((0, 3), p=1.0)
            ),
            K.RandomAffine(30.0, p=0.5),
            K.ColorJitter(0.2, 0.2, 0.2, 0.2),
            random_apply=(2, 3),
        )
        seq.to(device=sampling_device, dtype=dtype)
        before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        expected = seq(x, generator=torch.Generator(device=sampling_device).manual_seed(19))
        params = seq._params
        assert torch.equal(before, torch.random.get_rng_state())
        torch.rand(17)
        actual = seq(x, generator=torch.Generator(device=sampling_device).manual_seed(19))
        assert torch.equal(expected, actual)
        generator = torch.Generator(device=sampling_device).manual_seed(42)
        state = generator.get_state().clone()
        assert torch.equal(seq(x, params=params, generator=generator), expected)
        assert torch.equal(generator.get_state(), state)
        if cuda_before is not None:
            assert torch.equal(cuda_before, torch.cuda.get_rng_state())
        assert not torch.equal(before, torch.random.get_rng_state())  # only the unrelated draw consumed it

    @pytest.mark.parametrize("persistent", [False, True])
    def test_rotation90_generator_stream(self, persistent, device, dtype):
        """The standalone quarter-turn sampler accepts persistent and call-local streams."""
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Native streams need a CPU or CUDA device")
        sampler = Rotation90Generator((0, 3)).to(device=device, dtype=dtype)
        stream = torch.Generator(device=device).manual_seed(13)
        if persistent:
            sampler.set_generator(stream)
        before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        expected = sampler((8, 3, 16, 16), **({} if persistent else {"generator": stream}))
        stream.manual_seed(13)
        actual = sampler((8, 3, 16, 16), **({} if persistent else {"generator": stream}))
        _equal_params(actual, expected)
        assert torch.equal(before, torch.random.get_rng_state())
        if cuda_before is not None:
            assert torch.equal(cuda_before, torch.cuda.get_rng_state())
        assert sampler._generator is (stream if persistent else None)

    def test_persistent_streams_and_temporary_override(self):
        """A per-call override restores different persistent streams on shared descendants."""
        child = K.RandomAffine(90.0, p=1.0).set_generator(torch.Generator().manual_seed(3))
        seq = K.ImageSequential(child).set_generator(torch.Generator().manual_seed(7))
        child_generator = torch.Generator().manual_seed(11)
        child.set_generator(child_generator)
        outer_generator = seq._generator
        state = child_generator.get_state().clone()
        seq.forward_parameters(torch.Size((2, 1, 8, 8)), generator=torch.Generator().manual_seed(29))
        assert seq._generator is outer_generator
        assert child._generator is child_generator
        assert torch.equal(state, child_generator.get_state())
        seq.set_generator(None)
        assert all(m._generator is None for m in seq.modules() if hasattr(m, "_generator"))

    @pytest.mark.parametrize("serialization", ["deepcopy", "pickle", "torch"])
    @pytest.mark.parametrize("nested", [False, True])
    def test_copy_with_learnable_range(self, serialization, nested, device, dtype):
        """A copy retains its stream and differentiates through its own learnable range."""
        bounds = torch.nn.Parameter(torch.tensor([0.8, 1.2], device=device, dtype=dtype))
        aug = K.RandomBrightness(bounds, p=1.0)
        module = K.ImageSequential(aug) if nested else aug
        module.set_generator(torch.Generator().manual_seed(3))
        image = torch.full((2, 1, 4, 4), 0.25, device=device, dtype=dtype)
        global_state = torch.random.get_rng_state().clone()
        module(image)

        with torch.no_grad():
            if serialization == "deepcopy":
                clone = copy.deepcopy(module)
            elif serialization == "pickle":
                clone = pickle.loads(pickle.dumps(module))  # noqa: S301
            else:
                buffer = io.BytesIO()
                torch.save(module, buffer)
                buffer.seek(0)
                clone = torch.load(buffer, weights_only=False)

        assert clone._generator is not module._generator
        assert torch.equal(clone._generator.get_state(), module._generator.get_state())
        child = clone[0] if nested else clone
        assert child._generator is clone._generator
        assert child._param_generator._generator is clone._generator
        expected = module(image)
        expected_grad = torch.autograd.grad(expected.sum(), bounds)[0]
        original_state = module._generator.get_state().clone()
        actual = clone(image)
        self.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(original_state, module._generator.get_state())
        assert torch.equal(global_state, torch.random.get_rng_state())
        actual.sum().backward()
        (copied_bounds,) = clone.parameters()
        assert copied_bounds is not bounds
        assert bounds.grad is None
        assert bool(copied_bounds.grad.ne(0).any())
        self.assert_close(copied_bounds.grad, expected_grad)

    @pytest.mark.parametrize("container", ["video", "patch"])
    def test_video_and_patch_streams(self, container, device, dtype):
        """Container-specific parameter sampling shares the stream and can be replayed."""
        children = [K.RandomHorizontalFlip(p=0.5), K.RandomBrightness((0.7, 1.3), p=1.0)]
        if container == "video":
            seq = K.VideoSequential(*children, same_on_frame=True, data_format="BTCHW")
            x = torch.linspace(0, 1, 1152, device=device, dtype=dtype).reshape(2, 3, 3, 8, 8)
        else:
            seq = K.PatchSequential(*children, grid_size=(2, 2), patchwise_apply=False)
            x = torch.linspace(0, 1, 384, device=device, dtype=dtype).reshape(2, 3, 8, 8)
        torch.manual_seed(13)
        expected = seq(x)
        before = torch.random.get_rng_state().clone()
        actual = seq(x, generator=torch.Generator().manual_seed(13))
        assert torch.equal(actual, expected)
        assert torch.equal(before, torch.random.get_rng_state())
        assert torch.equal(seq(x, params=seq._params), expected)

    def test_joint_image_mask_stream(self, device, dtype):
        """The same sampled geometry is applied to aligned inputs."""
        image = torch.arange(192, device=device, dtype=dtype).reshape(4, 1, 6, 8)
        mask = image.clone()
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5), data_keys=["input", "mask"])
        before = torch.random.get_rng_state().clone()
        result, transformed_mask = seq(image, mask, generator=torch.Generator().manual_seed(3))
        assert torch.equal(result, transformed_mask)
        assert not torch.equal(result, image)
        assert torch.equal(before, torch.random.get_rng_state())

    def test_restore_after_exception(self):
        """An invalid input cannot leave a call-local stream installed on the module."""
        aug = K.RandomAffine(30.0).set_generator(torch.Generator().manual_seed(1))
        previous = aug._generator
        with pytest.raises((TypeError, ValueError, AssertionError)):
            aug(torch.ones(2, 3, 4, 4, dtype=torch.int64), generator=torch.Generator().manual_seed(2))
        assert aug._generator is previous
        assert aug._param_generator._generator is previous

    def test_two_thread_streams(self):
        """Different modules can sample concurrently without borrowing the global RNG."""

        def draw(seed):
            aug = K.RandomAffine(180.0, p=0.5).set_generator(torch.Generator().manual_seed(seed))
            return [aug.forward_parameters(torch.Size((8, 1, 8, 8))) for _ in range(4)]

        expected = [draw(1), draw(2)]
        before = torch.random.get_rng_state().clone()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            actual = list(pool.map(draw, [1, 2]))
        _equal_params(actual, expected)
        assert torch.equal(before, torch.random.get_rng_state())

    def test_standalone_parameter_generator(self):
        """Parameter generators expose the same persistent and call-local interfaces."""
        sampler = AffineGenerator(45.0)
        shape = (4, 1, 8, 8)
        expected = sampler(shape, generator=torch.Generator().manual_seed(4))
        sampler.set_generator(torch.Generator().manual_seed(4))
        _equal_params(sampler(shape), expected)

    @pytest.mark.parametrize("factory", [lambda: K.RandomAffine3D(30.0), lambda: K.RandomRotation3D(30.0)])
    def test_three_dimensional_sampling(self, factory):
        """The 3D generators use the same isolated sampling path."""
        aug = factory()
        shape = torch.Size((4, 1, 8, 8, 8))
        torch.manual_seed(3)
        expected = aug.forward_parameters(shape)
        before = torch.random.get_rng_state().clone()
        _equal_params(aug.forward_parameters(shape, generator=torch.Generator().manual_seed(3)), expected)
        assert torch.equal(before, torch.random.get_rng_state())

    def test_beta_reparameterization(self, device):
        """An explicit generator preserves the native Beta sample and its concentration gradients."""
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Native Dirichlet sampling needs a CPU or CUDA generator")
        concentrations = torch.tensor([0.2, 1.0, 5.0], device=device, dtype=torch.float64, requires_grad=True)
        beta = torch.distributions.Beta(concentrations, concentrations + 0.3)
        torch.manual_seed(5)
        expected = beta.rsample((7,))
        expected_gradient = torch.autograd.grad(expected.sum(), concentrations, retain_graph=True)[0]
        before = torch.random.get_rng_state().clone()
        actual = _adapted_rsampling((7,), beta, generator=torch.Generator(device=device).manual_seed(5))
        self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(torch.autograd.grad(actual.sum(), concentrations)[0], expected_gradient, rtol=0, atol=0)
        assert torch.equal(before, torch.random.get_rng_state())

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: K.RandomAffine(30.0, p=0.5),
            lambda: K.RandomCrop((4, 5)),
            lambda: K.RandomGaussianNoise(p=1.0),
            lambda: K.RandomCutMixV2(beta=0.5, p=1.0, use_correct_lambda=True),
            # Distribution and randperm samplers that must draw on the sampling device.
            lambda: K.RandomMixUpV2(p=1.0),
            lambda: K.RandomMosaic(output_size=(4, 4), p=1.0),
            lambda: K.RandomMotionBlur(kernel_size=(3, 5), angle=30.0, direction=0.5, p=1.0),
            lambda: K.RandomMotionBlur3D(kernel_size=(3, 5), angle=30.0, direction=0.5, p=1.0),
            lambda: K.RandomSaltAndPepperNoise(amount=0.1, salt_vs_pepper=0.5, p=1.0),
        ],
    )
    def test_sampling_device_stream(self, factory, device):
        """Explicit native-device gates and distribution draws leave both global streams untouched."""
        if device.type not in ("cpu", "cuda"):
            pytest.skip("This tests native CPU/CUDA generator sampling")
        aug = factory()
        aug.set_rng_device_and_dtype(device, torch.float32)
        shape = torch.Size((4, 3, 8, 8))
        torch.manual_seed(7)
        # Global CUDA sampling can also consume the CPU stream. An explicit CUDA stream owns every draw.
        expected = aug.forward_parameters(
            shape, **({"generator": torch.Generator(device=device).manual_seed(7)} if device.type == "cuda" else {})
        )
        cpu_state = torch.random.get_rng_state().clone()
        device_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else cpu_state
        actual = aug.forward_parameters(shape, generator=torch.Generator(device=device).manual_seed(7))
        _equal_params(actual, expected)
        assert torch.equal(cpu_state, torch.random.get_rng_state())
        if device.type == "cuda":
            assert torch.equal(device_state, torch.cuda.get_rng_state(device))

    @pytest.mark.parametrize("factory", [K.RandomMixUpV2, K.RandomMosaic])
    def test_default_host_range_permutation_stays_on_cpu(self, factory, device):
        """Configuring CUDA sampling does not move a host-range permutation off the global CPU stream."""
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Native generator sampling needs a CPU or CUDA device")
        aug = factory(p=1.0)
        aug.set_rng_device_and_dtype(device, torch.float32)
        with mock.patch.object(torch, "randperm", wraps=torch.randperm) as randperm:
            aug.generate_parameters(torch.Size((4, 3, 8, 8)))
        assert randperm.call_count == 1
        assert torch.device(randperm.call_args.kwargs["device"]).type == "cpu"

    def test_default_motion_blur_index_stays_on_cpu(self, device):
        """The unbound motion-blur kernel index retains its global CPU draw."""
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Native generator sampling needs a CPU or CUDA device")
        aug = K.RandomMotionBlur((3, 5), 30.0, 0.5, p=1.0)
        aug.set_rng_device_and_dtype(device, torch.float32)
        params = aug.generate_parameters(torch.Size((4, 3, 8, 8)))
        assert params["idx"].device.type == "cpu"

    @pytest.mark.parametrize("factory", [K.RandomMotionBlur, K.RandomMotionBlur3D])
    def test_default_motion_blur_kernel_range_stays_on_cpu(self, factory, device, dtype):
        """Kernel-size bounds keep the host/default dtype independently of the angle sampler."""
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Native generator sampling needs a CPU or CUDA device")
        aug = factory((3, 5), 30.0, 0.5, p=1.0)
        aug.set_rng_device_and_dtype(device, dtype)
        assert aug._param_generator.ksize_sampler.low.device.type == "cpu"
        assert aug._param_generator.ksize_sampler.low.dtype == torch.get_default_dtype()

    def test_dynamo_explicit_parameter_replay_4427(self, device, dtype, torch_optimizer):
        """Sample an isolated stream eagerly, then capture deterministic transform application."""
        aug = K.RandomAffine(30.0, p=0.5).set_generator(torch.Generator().manual_seed(7))
        x = torch.linspace(0, 1, 192, device=device, dtype=dtype).reshape(4, 1, 6, 8)
        params = aug.forward_parameters(x.shape)
        state = aug._generator.get_state().clone()
        expected = aug(x, params=params)
        compiled = torch_optimizer(aug, fullgraph=True)
        self.assert_close(compiled(x, params=params), expected)
        assert torch.equal(state, aug._generator.get_state())

    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_tps_sampling_device_stream(self, device, dtype, same_on_batch):
        """TPS draws on its configured device before converting control-point noise to the output dtype."""
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Native generator sampling needs a CPU or CUDA generator")
        aug = K.RandomThinPlateSpline(scale=0.2, same_on_batch=same_on_batch, p=1.0)
        aug.set_rng_device_and_dtype(device, dtype)
        dist = torch.distributions.Uniform(aug.dist.low.to(device), aug.dist.high.to(device), validate_args=False)
        torch.manual_seed(17)
        expected_noise = dist.rsample((1 if same_on_batch else 4, 5, 2)).to(dtype).expand(4, 5, 2)
        cpu_state = torch.random.get_rng_state().clone()
        device_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else cpu_state
        params = aug.forward_parameters(
            torch.Size((4, 1, 8, 8)), generator=torch.Generator(device=device).manual_seed(17)
        )
        self.assert_close(params["dst"], params["src"] + expected_noise, rtol=0, atol=0)
        assert params["dst"].device == expected_noise.device
        assert params["dst"].dtype == dtype
        assert torch.equal(cpu_state, torch.random.get_rng_state())
        if device.type == "cuda":
            assert torch.equal(device_state, torch.cuda.get_rng_state(device))

    def test_unsupported_policy_and_invalid_generator(self):
        """Unsupported policies fail before installing a stream or consuming global randomness."""
        from kornia.augmentation.auto import RandAugment

        aug = K.RandomAffine(45.0)
        with pytest.raises(TypeError, match=r"torch\.Generator"):
            aug.set_generator(4)
        seq = K.ImageSequential(aug, RandAugment(n=1, m=5))
        with pytest.raises(NotImplementedError, match="RandAugment"):
            seq.set_generator(torch.Generator())
        assert seq._generator is None
        assert aug._generator is None

    def test_opt_out_flag_rejects_subclasses(self):
        """A module that opts out of explicit streams is rejected through its subclasses too."""

        class OptedOut(K.RandomAffine):
            _supports_explicit_generator = False

        with pytest.raises(NotImplementedError, match="OptedOut"):
            OptedOut(45.0).set_generator(torch.Generator())
        with pytest.raises(NotImplementedError, match="OptedOut"):
            K.ImageSequential(OptedOut(45.0)).set_generator(torch.Generator())
        # The flag defaults to opt-in, so an ordinary subclass still accepts a stream.
        K.RandomAffine(45.0).set_generator(torch.Generator().manual_seed(1))
