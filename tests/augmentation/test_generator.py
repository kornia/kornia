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

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation.random_generator import AffineGenerator
from kornia.augmentation.utils import _adapted_rsampling

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


_FACTORIES = [
    lambda: K.RandomHorizontalFlip(p=0.5),
    lambda: K.RandomAffine(45.0, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=15.0, p=0.5),
    lambda: K.RandomCrop((8, 8), padding=1),
    lambda: K.RandomResizedCrop((8, 8)),
    lambda: K.RandomPerspective(0.5),
    lambda: K.ColorJitter(0.2, 0.2, 0.2, 0.2),
    lambda: K.ColorJiggle(0.2, 0.2, 0.2, 0.2),
    lambda: K.RandomGaussianNoise(p=1.0),
    lambda: K.RandomGaussianBlur((3, 3), (0.2, 1.0)),
    lambda: K.RandomMotionBlur(3, 30.0, 0.5),
    lambda: K.RandomElasticTransform(kernel_size=(3, 3), sigma=(1.0, 1.0)),
    lambda: K.RandomThinPlateSpline(p=1.0),
    lambda: K.RandomPlasmaBrightness(p=1.0),
    lambda: K.RandomPlasmaContrast(p=1.0),
    lambda: K.RandomPlasmaShadow(p=1.0),
    lambda: K.RandomJigsaw(grid=(2, 2)),
    lambda: K.RandomChannelShuffle(p=1.0),
    lambda: K.RandomChannelDropout(p=1.0),
    lambda: K.RandomRain(number_of_drops=(1, 3), drop_height=(1, 3), drop_width=(-1, 1)),
    lambda: K.RandomMixUpV2(p=1.0),
    lambda: K.RandomCutMixV2(num_mix=2, beta=0.5, p=1.0, use_correct_lambda=True),
    lambda: K.RandomSaltAndPepperNoise(p=1.0),
]


class TestExplicitGenerator(BaseTester):
    @pytest.mark.parametrize("factory", _FACTORIES, ids=lambda f: f().__class__.__name__)
    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_parameter_draws_match_global_stream_without_consuming_it(self, factory, same_on_batch):
        """An explicit stream replaces the generator, not the sampling algorithm or draw order."""
        aug = factory()
        aug.same_on_batch = same_on_batch
        shape = torch.Size((4, 3, 16, 16))
        torch.manual_seed(123)
        expected = aug.forward_parameters(shape)
        torch.manual_seed(789)
        before = torch.random.get_rng_state().clone()
        actual = aug.forward_parameters(shape, generator=torch.Generator().manual_seed(123))
        _equal_params(actual, expected)
        assert torch.equal(before, torch.random.get_rng_state())
        assert aug._generator is None

    def test_issue_reproduction_4427(self, device, dtype):
        """The previously swallowed generator keyword controls affine draws."""
        x = torch.linspace(0, 1, 96, device=device, dtype=dtype).reshape(2, 1, 6, 8)
        aug = K.RandomAffine(180.0, p=1.0)
        a = aug(x, generator=torch.Generator().manual_seed(1))
        angle = aug._params["angle"].clone()
        b = aug(x, generator=torch.Generator().manual_seed(2))
        assert not torch.equal(angle, aug._params["angle"])
        assert not torch.equal(a, b)
        self.assert_close(a, aug(x, generator=torch.Generator().manual_seed(1)))

    def test_nested_random_apply_and_replay(self, device, dtype):
        """Order selection, gates, noise, and nested children share the call's stream."""
        x = torch.linspace(0, 1, 768, device=device, dtype=dtype).reshape(4, 3, 8, 8)
        seq = K.AugmentationSequential(
            K.ImageSequential(K.RandomHorizontalFlip(p=0.5), K.RandomGaussianNoise(p=1.0)),
            K.RandomAffine(30.0, p=0.5),
            K.ColorJitter(0.2, 0.2, 0.2, 0.2),
            random_apply=(2, 3),
        )
        before = torch.random.get_rng_state().clone()
        expected = seq(x, generator=torch.Generator().manual_seed(19))
        params = seq._params
        assert torch.equal(before, torch.random.get_rng_state())
        torch.rand(17)
        actual = seq(x, generator=torch.Generator().manual_seed(19))
        assert torch.equal(expected, actual)
        generator = torch.Generator().manual_seed(42)
        state = generator.get_state().clone()
        assert torch.equal(seq(x, params=params, generator=generator), expected)
        assert torch.equal(generator.get_state(), state)
        assert not torch.equal(before, torch.random.get_rng_state())  # only the unrelated draw consumed it

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
        expected = aug.forward_parameters(shape)
        cpu_state = torch.random.get_rng_state().clone()
        device_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else cpu_state
        actual = aug.forward_parameters(shape, generator=torch.Generator(device=device).manual_seed(7))
        _equal_params(actual, expected)
        assert torch.equal(cpu_state, torch.random.get_rng_state())
        if device.type == "cuda":
            assert torch.equal(device_state, torch.cuda.get_rng_state(device))

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
