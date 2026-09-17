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

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from kornia.feature import SIFTFeatureScaleSpace, get_laf_center, get_laf_orientation, laf_is_filled
from kornia.feature.sift.scale_space import _SIFTScaleSpaceDescriptor, _SIFTScaleSpaceDetector
from kornia.filters import spatial_gradient

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


class TestSharedSIFTScaleSpace(BaseTester):
    def test_trilinear_histogram_bins(self, device, dtype):
        # Hand-computed votes in (row, column, angle) order. Cancel Gaussian
        # weighting to isolate spatial interpolation and the angle 7 -> 0 seam.
        work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        xx = torch.tensor([-0.6, -0.2, 0.2, 0.6, 0.0], device=device, dtype=work_dtype)
        yy = torch.tensor([0.6, -0.6, 0.2, -0.2, 0.2], device=device, dtype=work_dtype)
        mass = xx.new_tensor([1.0, 2.0, 3.0, 4.0, 2.0])
        mag = (mass * torch.exp(0.78125 * (xx.square() + yy.square()))).view(1, 1, -1)
        angle = xx.new_tensor([0.25, 1.5, 7.75, 4.0, 2.5]).view(1, 1, -1) * (torch.pi / 4)
        actual = _SIFTScaleSpaceDescriptor._descriptor_histograms(mag, angle, xx, yy).reshape(4, 4, 8)
        expected = torch.zeros_like(actual)
        expected[3, 0, 0], expected[3, 0, 1] = 0.75, 0.25
        expected[0, 1, 1:3] = 1.0
        expected[2, 2, 7], expected[2, 2, 0] = 0.75, 2.25
        expected[1, 3, 4] = 4.0
        expected[2, 1:3, 2:4] = 0.5
        self.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_histogram_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        xx = torch.tensor([-0.8, -0.15, 0.35, 0.9], device=device, dtype=torch.float64)
        yy = xx.flip(0)
        mag = xx.new_tensor([0.3, 0.6, 1.0, 0.4]).view(1, 1, -1)
        angle = xx.new_tensor([0.2, 1.3, 3.1, 5.6]).view(1, 1, -1)
        self.gradcheck(lambda m, a: _SIFTScaleSpaceDescriptor._descriptor_histograms(m, a, xx, yy), (mag, angle))

    def test_separable_histogram_matches_dense_values_and_gradients(self, device, dtype):
        # _sample_gradients promotes reduced precision before this private
        # histogram helper; keep half leaves to check their returned gradients.
        work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        coordinate = torch.tensor([-0.6, 0.0, 0.6], device=device, dtype=work_dtype)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        xx, yy = xx.reshape(-1), yy.reshape(-1)
        actual_mag = torch.linspace(0.1, 1.8, 18, device=device, dtype=dtype).reshape(2, 1, 9).requires_grad_()
        expected_mag = actual_mag.detach().clone().requires_grad_()
        # Include both directions across the 7 -> 0 seam without putting a
        # finite-difference probe exactly on an angular-bin boundary.
        actual_angle = (
            torch.tensor(
                [-0.13, 0.21, 1.07, 2.31, 3.53, 4.89, 5.97, 2 * torch.pi + 0.17, -2 * torch.pi + 1.43],
                device=device,
                dtype=dtype,
            )
            .repeat(2)
            .reshape(2, 1, 9)
            .requires_grad_()
        )
        expected_angle = actual_angle.detach().clone().requires_grad_()
        bins = torch.arange(4, device=device, dtype=work_dtype)
        separable = (1.0 - (2.5 * coordinate[:, None] + 1.5 - bins).abs()).clamp_min(0.0)

        actual = _SIFTScaleSpaceDescriptor._descriptor_histograms(
            actual_mag.to(work_dtype), actual_angle.to(work_dtype), xx, yy, separable
        )
        expected = _SIFTScaleSpaceDescriptor._descriptor_histograms(
            expected_mag.to(work_dtype), expected_angle.to(work_dtype), xx, yy
        )
        self.assert_close(actual, expected)
        weights = torch.linspace(0.1, 1.0, actual.numel(), device=device, dtype=work_dtype).reshape_as(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        self.assert_close(actual_mag.grad, expected_mag.grad)
        self.assert_close(actual_angle.grad, expected_angle.grad)

    def test_separable_histogram_gradcheck_cpu_double(self, device):
        if device.type != "cpu":
            pytest.skip("covers the CPU float64 derivative of the separable path")
        coordinate = torch.tensor([-0.6, 0.0, 0.6], device=device, dtype=torch.float64)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        xx, yy = xx.reshape(-1), yy.reshape(-1)
        bins = torch.arange(4, device=device, dtype=torch.float64)
        separable = (1.0 - (2.5 * coordinate[:, None] + 1.5 - bins).abs()).clamp_min(0.0)
        mag = torch.linspace(0.1, 0.9, 9, device=device, dtype=torch.float64).reshape(1, 1, 9)
        angle = torch.tensor(
            [-0.13, 0.21, 1.07, 2.31, 3.53, 4.89, 5.97, 2 * torch.pi + 0.17, -2 * torch.pi + 1.43],
            device=device,
            dtype=torch.float64,
        ).reshape(1, 1, 9)
        self.gradcheck(
            lambda m, a: _SIFTScaleSpaceDescriptor._descriptor_histograms(m, a, xx, yy, separable), (mag, angle)
        )

    @pytest.mark.parametrize("components", [["scale_pyr"], ["subpix"], ["scale_pyr", "subpix"]])
    def test_checkpoint_round_trip(self, components):
        eager = SIFTFeatureScaleSpace(8, descriptor_backend="pyramid")
        compiled = SIFTFeatureScaleSpace(8, descriptor_backend="pyramid", compile_modules=components)
        assert eager.state_dict().keys() == compiled.state_dict().keys()
        compiled.load_state_dict(eager.state_dict(), strict=True)
        eager.load_state_dict(compiled.state_dict(), strict=True)
        checkpoint = io.BytesIO()
        torch.save(compiled, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=False)
        image = torch.rand(1, 1, 17, 19)
        for expected, actual in zip(eager(image), restored(image)):
            self.assert_close(actual, expected)

    def test_single_pyramid_build_and_exact_images(self, device, dtype, monkeypatch):
        feature = SIFTFeatureScaleSpace(8, descriptor_backend="pyramid").to(device, dtype)
        image = torch.rand(1, 1, 65, 67, device=device, dtype=dtype)
        built = []
        original = feature.detector.scale_pyr.forward

        def build(image):
            result = original(image)
            built.append(result)
            return result

        original_descriptor = feature.descriptor.forward

        def describe(pyramid, *args, **kwargs):
            assert pyramid is built[-1]
            return original_descriptor(pyramid, *args, **kwargs)

        monkeypatch.setattr(feature.detector.scale_pyr, "forward", build)
        monkeypatch.setattr(feature.descriptor, "forward", describe)
        lafs, responses, desc = feature(image)
        assert len(built) == 1
        assert desc.shape == (1, 8, 128)
        assert torch.isfinite(desc).all()
        assert torch.isfinite(responses).all()
        assert torch.isfinite(lafs).all()

    def test_provenance_survives_topk_and_padding(self, device, dtype):
        detector = SIFTFeatureScaleSpace(40, upright=True, descriptor_backend="pyramid").to(device, dtype).detector
        image = torch.rand(2, 1, 65, 67, device=device, dtype=dtype)
        image[1] = 0
        responses, lafs, filled, pyramid, octaves, levels = detector._detect_with_pyramid(image, 40)
        assert torch.equal(filled, laf_is_filled(lafs))
        assert (octaves[~filled] == -1).all()
        assert (levels[~filled] == -1).all()
        assert not filled[1].any()
        expected_lafs, expected_responses = detector(image)
        self.assert_close(responses, expected_responses)
        self.assert_close(lafs, expected_lafs)
        for octave, images in enumerate(pyramid):
            selected = filled & (octaves == octave)
            # Selected layer is nearest to the continuous refined scale, not an
            # octave guessed from frame size (octaves have overlapping scales).
            sigma = lafs[..., 0, 0][selected] / (6 * 0.5 * 2**octave)
            expected = (3 * torch.log2(sigma / 1.6)).round().long().clamp(0, images.shape[2] - 1)
            assert torch.equal(levels[selected], expected)

    def test_selected_layer_and_rotation(self, device, dtype):
        axis = torch.arange(64, device=device, dtype=dtype)
        horizontal = axis[None, :].expand(64, 64)
        vertical = axis[:, None].expand(64, 64)
        pyramid = [torch.stack([horizontal, vertical])[None, None]]
        # Same location/scale, different Gaussian provenance must give different
        # orientations even though the returned canonical descriptors coincide.
        lafs = torch.tensor([[[[6.0, 0, 16], [0, 6.0, 16]]] * 2], device=device, dtype=dtype)
        octaves = torch.zeros(1, 2, device=device, dtype=torch.long)
        levels = torch.tensor([[0, 1]], device=device)
        oriented, desc = _SIFTScaleSpaceDescriptor()(pyramid, lafs, octaves, levels)
        self.assert_close(get_laf_center(oriented), get_laf_center(lafs))
        angles = get_laf_orientation(oriented).flatten()
        self.assert_close(angles.abs(), angles.new_tensor([0, 90]), atol=0.02, rtol=0)
        self.assert_close(desc[0, 0], desc[0, 1], atol=0.002, rtol=0.002)
        self.assert_close(desc.norm(dim=-1), torch.ones(1, 2, device=device, dtype=dtype))

    def test_mask_padding_and_empty(self, device, dtype):
        feature = SIFTFeatureScaleSpace(4, descriptor_backend="pyramid").to(device, dtype)
        image = torch.rand(2, 1, 40, 40, device=device, dtype=dtype)
        lafs, responses, desc = feature(image, torch.zeros_like(image))
        assert not lafs.any()
        assert not responses.any()
        assert not desc.any()
        feature.detector.num_features = 0
        lafs, responses, desc = feature(image)
        assert lafs.shape == (2, 0, 2, 3)
        assert desc.shape == (2, 0, 128)

    @pytest.mark.parametrize("case", ["flat", "masked", "num_features", "empty_batch"])
    def test_public_empty_outputs_are_graph_connected(self, device, dtype, case):
        if case == "empty_batch":
            image = torch.rand(0, 1, 40, 40, device=device, dtype=dtype, requires_grad=True)
        elif case == "flat":
            image = torch.zeros(1, 1, 40, 40, device=device, dtype=dtype, requires_grad=True)
        else:
            image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype, requires_grad=True)
        feature = SIFTFeatureScaleSpace(4, descriptor_backend="pyramid").to(device, dtype)
        if case == "num_features":
            feature.detector.num_features = 0
        mask = torch.zeros_like(image) if case == "masked" else None

        lafs, responses, descriptors = feature(image, mask)

        assert not lafs.any() and not responses.any() and not descriptors.any()
        assert lafs.requires_grad and responses.requires_grad and descriptors.requires_grad
        (lafs.sum() + responses.sum() + descriptors.sum()).backward()
        assert image.grad is not None
        assert torch.isfinite(image.grad).all() and not image.grad.any()

    def test_float16_tiny_float32_mask_is_suppressed(self, device, dtype):
        if dtype != torch.float16:
            pytest.skip("exercises mask preservation across float16 detector promotion")
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype)
        detector = _SIFTScaleSpaceDetector(1, _FixedPyramid(dog)).to(device, dtype)
        mask = torch.full((1, 1, 32, 32), 1e-10, device=device, dtype=torch.float32)
        lafs, responses = detector(image, mask)
        assert not lafs.any() and not responses.any()

    def test_flat_gradient_backward(self, device, dtype):
        image = torch.zeros(1, 1, 1, 32, 32, device=device, dtype=dtype, requires_grad=True)
        lafs = torch.tensor([[[[3.0, 0, 8], [0, 3.0, 8]]]], device=device, dtype=dtype)
        ids = torch.zeros(1, 1, device=device, dtype=torch.long)
        _, desc = _SIFTScaleSpaceDescriptor()([image], lafs, ids, ids)
        desc.sum().backward()
        assert torch.isfinite(image.grad).all()

    def test_gradients_built_once_per_used_layer(self, device, dtype, monkeypatch):
        import kornia.feature.sift.scale_space as implementation

        pyramid = [torch.rand(2, 1, 3, 32, 32, device=device, dtype=dtype)]
        lafs = torch.tensor([[[[3.0, 0, 8], [0, 3.0, 8]]] * 3] * 2, device=device, dtype=dtype)
        # Exercise both ends of an atlas layer: border replication must stay in
        # that layer instead of interpolating a neighbouring Gaussian image.
        # Original-image centres 0.5/15.5 become octave-0 centres 1/31 because
        # its pixel distance is 0.5. Batch 1 uses the upper atlas layer at y=1.
        lafs[:, 0, :, 2] = 0.5
        lafs[:, 2, :, 2] = 15.5
        octaves = torch.zeros(2, 3, device=device, dtype=torch.long)
        levels = torch.tensor([[0, 0, 2], [2, 0, 2]], device=device)
        calls = []
        original = implementation._SIFTScaleSpaceDescriptor._gradient_atlas

        def gradient(image, levels, work_dtype):
            calls.append(image)
            return original(image, levels, work_dtype)

        monkeypatch.setattr(implementation._SIFTScaleSpaceDescriptor, "_gradient_atlas", staticmethod(gradient))
        module = _SIFTScaleSpaceDescriptor()
        _, desc = module(pyramid, lafs, octaves, levels)
        # Used layers are batched into one gradient call per octave, including
        # both images; the unused middle layer is never differentiated.
        assert len(calls) == 1
        assert calls[0].shape == (2, 1, 3, 32, 32)
        for batch in range(2):
            for feature in range(3):
                _, single = module(
                    [pyramid[0][batch : batch + 1]],
                    lafs[batch : batch + 1, feature : feature + 1],
                    octaves[batch : batch + 1, feature : feature + 1],
                    levels[batch : batch + 1, feature : feature + 1],
                )
                self.assert_close(desc[batch, feature], single[0, 0])

    @pytest.mark.parametrize("height,width", [(5, 7), (1, 7), (7, 1)])
    def test_gradient_atlas_matches_spatial_gradient_and_backward(self, device, dtype, height, width):
        levels = torch.tensor([0, 2], device=device)
        actual_images = torch.rand(2, 1, 3, height, width, device=device, dtype=dtype, requires_grad=True)
        expected_images = actual_images.detach().clone().requires_grad_()
        work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        actual = _SIFTScaleSpaceDescriptor._gradient_atlas(actual_images, levels, work_dtype)
        selected = expected_images[:, 0].index_select(1, levels).to(work_dtype)
        expected = spatial_gradient(selected.reshape(-1, 1, height, width), mode="diff")[:, 0]
        expected = expected.reshape(2, 2, 2, height, width).permute(0, 2, 1, 3, 4).reshape(2, 2, -1, width)
        self.assert_close(actual, expected)
        weights = torch.linspace(0.1, 1.0, actual.numel(), device=device, dtype=work_dtype).reshape_as(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        assert actual_images.grad is not None and expected_images.grad is not None
        self.assert_close(actual_images.grad, expected_images.grad)

    def test_grid_inference_matches_autograd_path(self, device, dtype):
        lafs = torch.tensor(
            [
                [[[2.0, 0.3, 0.0], [-0.2, 1.5, 0.0]], [[1.0, 0.0, 6.0], [0.0, 2.0, 4.0]]],
                [[[2.0, 0.3, 8.0], [-0.2, 1.5, 5.0]], [[1.0, 0.0, 3.0], [0.0, 2.0, 8.0]]],
            ],
            device=device,
            dtype=dtype,
        )
        layers = torch.tensor([[0, 2], [2, 0]], device=device)
        inference, _, _ = _SIFTScaleSpaceDescriptor._grid(lafs, layers, 9, 11, 27, 19)
        autograd, _, _ = _SIFTScaleSpaceDescriptor._grid(lafs.detach().clone().requires_grad_(), layers, 9, 11, 27, 19)
        self.assert_close(inference, autograd)

    def test_upright_image_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        image = torch.rand(1, 1, 1, 12, 12, device=device, dtype=torch.float64)
        lafs = torch.tensor([[[[1.0, 0, 3], [0, 1.0, 3]]]], device=device, dtype=torch.float64)
        ids = torch.zeros(1, 1, device=device, dtype=torch.long)
        module = _SIFTScaleSpaceDescriptor()
        # CUDA grid sampling accumulates repeated pixel contributions atomically.
        # Repeated float64 backward differs by ~6e-16 on both SIFT implementations.
        self.gradcheck(
            lambda image: module([image], lafs, ids, ids, upright=True)[1],
            (image,),
            nondet_tol=1e-12 if device.type == "cuda" else 0.0,
        )

    def test_specialized_detector_does_not_use_generic_detection(self, device, dtype, monkeypatch):
        from kornia.feature import ScaleSpaceDetector
        from kornia.feature.sift.scale_space import _SIFTScaleSpaceDetector

        def forbidden(*args, **kwargs):
            raise AssertionError("optimized SIFT must not call the generic detector")

        monkeypatch.setattr(ScaleSpaceDetector, "forward", forbidden)
        feature = SIFTFeatureScaleSpace(8, descriptor_backend="pyramid").to(device, dtype)
        assert isinstance(feature.detector, _SIFTScaleSpaceDetector)
        lafs, responses, descriptors = feature(torch.rand(1, 1, 33, 35, device=device, dtype=dtype))
        assert lafs.shape == (1, 8, 2, 3)
        assert responses.dtype == descriptors.dtype == lafs.dtype == dtype


class TestSIFTScalePyramid(BaseTester):
    def test_precise_double_grid(self, device, dtype):
        from kornia.feature.sift.scale_space import _SIFTScalePyramid

        if not supports_replicate_padding(device, dtype):
            pytest.skip("direct pyramid helper requires native replicate padding; the detector promotes half inputs")
        image = torch.arange(35, device=device, dtype=dtype).reshape(1, 1, 5, 7)
        doubled = _SIFTScalePyramid._double(image)
        self.assert_close(doubled[..., ::2, ::2], image)
        self.assert_close(doubled[..., 0::2, 1:-1:2], 0.5 * (image[..., :-1] + image[..., 1:]))
        assert doubled.shape == (1, 1, 10, 14)

    def test_next_octave_is_exact_decimation(self, device, dtype):
        from kornia.feature.sift.scale_space import _SIFTScalePyramid

        if not supports_replicate_padding(device, dtype) or not supports_reflect_padding(device, dtype):
            pytest.skip("direct pyramid helper requires native border padding; the detector promotes half inputs")
        image = torch.rand(1, 1, 65, 67, device=device, dtype=dtype)
        pyramid = _SIFTScalePyramid().to(device, dtype)(image)
        for previous, current in zip(pyramid, pyramid[1:]):
            h, w = previous.shape[-2:]
            self.assert_close(current[:, :, 0], previous[:, :, 3, : 2 * (h // 2) : 2, : 2 * (w // 2) : 2])
            assert current.shape[2] == 6

    def test_pyramid_backend_rejects_unknown_compile_component(self):
        with pytest.raises(ValueError, match="compile_modules"):
            SIFTFeatureScaleSpace(descriptor_backend="pyramid", compile_modules=["resp"])


class _FixedPyramid(nn.Module):
    def __init__(self, dog: torch.Tensor) -> None:
        super().__init__()
        self.dog = dog

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        gaussian = torch.cat([torch.zeros_like(self.dog[:, :1]), self.dog.cumsum(1)], 1).unsqueeze(1)
        return [gaussian.to(image)]


def _quadratic_dog(
    device: torch.device,
    dtype: torch.dtype,
    amplitude: float = 1.0,
    curvature: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (2.2, 10.3, 11.2),
) -> torch.Tensor:
    # The detector constructs its pyramid in float32 for reduced-precision
    # inputs; preserve the analytic curvature in this substitute pyramid too.
    dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    s = torch.arange(5, device=device, dtype=dtype).view(1, 5, 1, 1)
    y = torch.arange(32, device=device, dtype=dtype).view(1, 1, 32, 1)
    x = torch.arange(32, device=device, dtype=dtype).view(1, 1, 1, 32)
    cs, cy, cx = curvature
    return amplitude * (
        20.0 - cs * (s - center[0]).square() - cy * (y - center[1]).square() - cx * (x - center[2]).square()
    )


class TestSIFTScaleSpaceDetector(BaseTester):
    @pytest.mark.parametrize("case", ["random", "quadratic", "singular", "outside"])
    def test_cuda_refinement_matches_reference(self, device, dtype, case):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("CUDA implementation; CPU checks its arithmetic against the reference")
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the specialized detector promotes reduced precision before refinement")
        generator = torch.Generator().manual_seed(23)
        dog = torch.rand(1, 5, 32, 32, generator=generator, dtype=dtype)
        index = torch.arange(24)
        b, s, y, x = index * 0, index % 5, (index * 3) % 32, (index * 7) % 32
        if case == "quadratic":
            dog = _quadratic_dog(torch.device("cpu"), dtype)
            s, y, x = index * 0 + 2, index % 3 + 9, index % 4 + 10
        elif case == "singular":
            dog.zero_()
        elif case == "outside":
            # The eager implementation exits before reading these NaNs. Extra
            # fixed-trip CUDA iterations must not introduce NaNs in backward.
            dog.fill_(float("nan"))
            s = index * 0 - 1
        reference_image = dog.clone().requires_grad_()
        image = dog.to(device).requires_grad_()
        detector = _SIFTScaleSpaceDetector(24, nn.Identity())
        expected = detector._refine(reference_image, b, s, y, x)
        actual = detector._refine_cuda(image, *(value.to(device) for value in (b, s, y, x)))
        for value, reference in zip(actual, expected):
            self.assert_close(value, reference.to(device), rtol=0, atol=0)
        if case == "quadratic":
            assert actual[-1].all()
        weights = torch.linspace(0.1, 1.0, index.numel(), dtype=dtype)
        reference_loss = sum((value * weights).sum() for value in expected[4:7]) + reference_image[..., :0].sum()
        loss = sum((value * weights.to(device)).sum() for value in actual[4:7]) + image[..., :0].sum()
        reference_loss.backward()
        loss.backward()
        assert image.grad is not None and reference_image.grad is not None
        assert torch.isfinite(image.grad).all()
        self.assert_close(image.grad, reference_image.grad.to(device))

    def test_dynamo_refinement(self, device, dtype, torch_optimizer):
        dog = _quadratic_dog(device, dtype)
        detector = _SIFTScaleSpaceDetector(1, _FixedPyramid(dog))
        b = torch.tensor([0], device=device)
        s, y, x = b + 2, b + 10, b + 11
        expected = detector._refine(dog, b, s, y, x)
        actual = torch_optimizer(detector._refine)(dog, b, s, y, x)
        for value, reference in zip(actual, expected):
            self.assert_close(value, reference)

    @pytest.mark.parametrize("neighbour", [(0, 1, 1), (1, 1, 1), (-1, 0, 0)])
    @pytest.mark.parametrize("sign", [-1.0, 1.0])
    def test_rejects_equal_diagonal_and_adjacent_scale(self, device, dtype, neighbour, sign):
        dog = torch.zeros(1, 5, 32, 32, device=device, dtype=dtype)
        dog[0, 2, 16, 16] = sign
        ds, dy, dx = neighbour
        dog[0, 2 + ds, 16 + dy, 16 + dx] = sign
        detector = _SIFTScaleSpaceDetector(2, _FixedPyramid(dog))
        lafs, responses = detector(torch.zeros(1, 1, 32, 32, device=device, dtype=dtype))
        assert not lafs.any() and not responses.any()

    def test_checkerboard_equal_diagonals_reject_before_neighbourhood_gather(self, device, dtype, monkeypatch):
        y = torch.arange(32, device=device).view(1, 1, 32, 1)
        x = torch.arange(32, device=device).view(1, 1, 1, 32)
        dog = (1 - 2 * ((x + y) % 2)).to(torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype)
        dog = dog.expand(1, 5, -1, -1)
        gaussian = torch.cat([torch.zeros_like(dog[:, :1]), dog.cumsum(1)], 1).unsqueeze(1)
        detector = _SIFTScaleSpaceDetector(1, nn.Identity()).to(device, dtype)

        def forbidden(*args, **kwargs):
            raise AssertionError("checkerboard candidates must be rejected before sparse gathering")

        monkeypatch.setattr(detector, "_neighbourhood", forbidden)
        result = detector._octave(gaussian, 0, None)
        assert all(value.numel() == 0 for value in result)

    def test_scale_rejected_spikes_screen_neighbourhoods_in_bounded_chunks(self, device, dtype, monkeypatch):
        if device.type not in ("cpu", "mps") or dtype != torch.float32:
            pytest.skip("exercises the CPU/MPS sparse-screening chunk limits")
        size = 512
        coordinates = torch.arange(6, size - 5, 3, device=device)
        dog = torch.zeros(1, 5, size, size, device=device, dtype=dtype)
        # Every searchable scale sees the same isolated spatial maxima, so all
        # candidates reach sparse screening but fail strict scale comparison.
        dog[:, :, coordinates[:, None], coordinates] = 1.0
        gaussian = torch.cat([torch.zeros_like(dog[:, :1]), dog.cumsum(1)], 1).unsqueeze(1)
        detector = _SIFTScaleSpaceDetector(1, nn.Identity())
        original = detector._neighbourhood
        calls = []

        def record(*args):
            calls.append(args[1].numel())
            return original(*args)

        monkeypatch.setattr(detector, "_neighbourhood", record)
        with torch.inference_mode():
            result = detector._octave(gaussian, 0, None)
        limit = 16384 if device.type == "cpu" else 65536
        assert len(calls) > 1 and sum(calls) > limit and max(calls) <= limit
        assert all(value.numel() == 0 for value in result)

    def test_many_middle_scale_extrema_refine_in_chunks(self, device, dtype, monkeypatch):
        if device.type != "cpu" or dtype != torch.float32:
            pytest.skip("exercises the CPU refinement chunk limit")
        size = 512
        coordinates = torch.arange(6, size - 5, 3, device=device)
        dog = torch.zeros(1, 5, size, size, device=device, dtype=dtype)
        dog[:, 2, coordinates[:, None], coordinates] = 1.0
        gaussian = torch.cat([torch.zeros_like(dog[:, :1]), dog.cumsum(1)], 1).unsqueeze(1)
        detector = _SIFTScaleSpaceDetector(1, nn.Identity())
        original = detector._refine
        calls = []

        def record(*args):
            calls.append(args[1].numel())
            return original(*args)

        monkeypatch.setattr(detector, "_refine", record)
        with torch.inference_mode():
            *_, responses, _, _, _ = detector._octave(gaussian, 0, None)
        expected_count = coordinates.numel() ** 2
        assert len(calls) > 1 and sum(calls) == expected_count and max(calls) <= 16384
        assert responses.numel() == expected_count
        self.assert_close(responses, torch.ones_like(responses))

    def test_sparse_neighbourhood_values_and_backward(self, device, dtype):
        dog = torch.rand(2, 5, 13, 15, device=device, dtype=dtype, requires_grad=True)
        b = torch.tensor([0, 1, 0], device=device)
        s = torch.tensor([1, 3, 1], device=device)
        y = torch.tensor([5, 7, 5], device=device)
        x = torch.tensor([6, 9, 6], device=device)
        actual = _SIFTScaleSpaceDetector._neighbourhood(dog, b, s, y, x)
        expected = torch.stack([dog[0, 0:3, 4:7, 5:8], dog[1, 2:5, 6:9, 8:11], dog[0, 0:3, 4:7, 5:8]])
        self.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), dog)[0]
        expected_grad = torch.autograd.grad(expected.sum(), dog)[0]
        self.assert_close(actual_grad, expected_grad)

    def test_refines_analytic_extremum_and_keeps_tiny_amplitude(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype)
        detector = _SIFTScaleSpaceDetector(4, _FixedPyramid(dog)).to(device, dtype)
        lafs, responses = detector(image)
        filled = laf_is_filled(lafs)
        assert filled[0, 0]
        expected = torch.tensor([11.2 * 0.5, 10.3 * 0.5], device=device, dtype=dtype)
        self.assert_close(get_laf_center(lafs)[0, 0], expected, rtol=2e-3, atol=2e-3)
        amplitude = 1e-4 if dtype == torch.float16 else 1e-8
        tiny_lafs, tiny_responses = _SIFTScaleSpaceDetector(4, _FixedPyramid(dog * amplitude)).to(device, dtype)(image)
        assert laf_is_filled(tiny_lafs)[0, 0]
        self.assert_close(get_laf_center(tiny_lafs)[0, 0], get_laf_center(lafs)[0, 0], rtol=2e-3, atol=2e-3)
        self.assert_close(tiny_responses[0, 0], responses[0, 0] * amplitude, rtol=5e-3, atol=1e-12)

    def test_keeps_edge_like_anisotropic_extremum(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype, curvature=(1.0, 1e-3, 1.0))
        lafs, _ = _SIFTScaleSpaceDetector(4, _FixedPyramid(dog)).to(device, dtype)(image)
        assert laf_is_filled(lafs)[0, 0]

    def test_minimum_is_ranked_by_absolute_response_and_output_is_padded(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        maximum = _quadratic_dog(device, dtype)
        minimum = -_quadratic_dog(device, dtype) + 1.0
        # Put a stronger negative extremum in the second batch item; both signs
        # are valid and responses are absolute refined DoG values.
        dog = torch.cat([maximum, minimum * 2.0], 0)
        lafs, responses = _SIFTScaleSpaceDetector(3, _FixedPyramid(dog)).to(device, dtype)(image.expand(2, -1, -1, -1))
        assert laf_is_filled(lafs)[:, 0].all()
        assert (responses[:, 0] > 0).all()
        assert (~laf_is_filled(lafs)[:, 1:]).all()
        assert not responses[:, 1:].any()

    def test_mask_broadcast_zero_and_fractional_weight(self, device, dtype):
        image = torch.zeros(2, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype).expand(2, -1, -1, -1).clone()
        detector = _SIFTScaleSpaceDetector(2, _FixedPyramid(dog)).to(device, dtype)
        _, unmasked = detector(image)
        mask = torch.full((1, 1, 32, 32), 0.25, device=device, dtype=dtype)
        lafs, weighted = detector(image, mask)
        assert laf_is_filled(lafs)[:, 0].all()
        self.assert_close(weighted[:, 0], unmasked[:, 0] * 0.25, rtol=2e-3, atol=2e-3)
        zero_lafs, zero_responses = detector(image, torch.zeros_like(mask))
        assert not zero_lafs.any() and not zero_responses.any()

    def test_flat_and_singular_refinement_reject_without_oob_access(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        flat = torch.zeros(1, 5, 32, 32, device=device, dtype=dtype)
        lafs, responses = _SIFTScaleSpaceDetector(2, _FixedPyramid(flat)).to(device, dtype)(image)
        assert not lafs.any() and not responses.any()
        detector = _SIFTScaleSpaceDetector(1, _FixedPyramid(flat)).to(device, dtype)
        b = torch.zeros(1, device=device, dtype=torch.long)
        s = torch.full_like(b, 2)
        y = torch.full_like(b, 10)
        x = torch.full_like(b, 10)
        *_, converged = detector._refine(flat, b, s, y, x)
        assert not converged.any()

    def test_fractional_offsets_at_right_bottom_border(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype, center=(2.2, 26.3, 26.2))
        lafs, _ = _SIFTScaleSpaceDetector(1, _FixedPyramid(dog))(image)
        assert laf_is_filled(lafs).all()
        expected = torch.tensor([26.2, 26.3], device=device, dtype=dtype) * 0.5
        self.assert_close(get_laf_center(lafs)[0, 0], expected, atol=0.01, rtol=0.001)

    def test_topk_sorts_by_refined_response(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        weaker = _quadratic_dog(device, dtype)
        stronger = _quadratic_dog(device, dtype, amplitude=2.0, center=(2.2, 10.3, 21.2))
        dog = torch.maximum(weaker, stronger)
        lafs, scores = _SIFTScaleSpaceDetector(2, _FixedPyramid(dog))(image)
        assert laf_is_filled(lafs).all()
        assert scores[0, 0] > scores[0, 1]
        expected = torch.tensor([[21.2, 10.3], [11.2, 10.3]], device=device, dtype=dtype) * 0.5
        self.assert_close(get_laf_center(lafs)[0], expected, atol=0.01, rtol=0.001)

    def test_refinement_backward_is_finite(self, device):
        dog = _quadratic_dog(device, torch.float32).requires_grad_()
        detector = _SIFTScaleSpaceDetector(1, _FixedPyramid(dog))
        lafs, responses = detector(torch.zeros(1, 1, 32, 32, device=device))
        (lafs.sum() + responses.sum()).backward()
        assert torch.isfinite(dog.grad).all()
