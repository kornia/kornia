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

import pytest
import torch

from kornia.feature import SIFTFeatureScaleSpace, get_laf_center, get_laf_orientation, laf_is_filled
from kornia.feature.sift_scale_space import _SIFTScaleSpaceDescriptor

from testing.base import BaseTester


class TestSharedSIFTScaleSpace(BaseTester):
    def test_single_pyramid_build_and_exact_images(self, device, dtype, monkeypatch):
        feature = SIFTFeatureScaleSpace(8, descriptor_backend="pyramid").to(device, dtype)
        image = torch.rand(1, 1, 65, 67, device=device, dtype=dtype)
        built = []
        original = feature.detector.scale_pyr.forward

        def build(image):
            result = original(image)
            built.append(result[0])
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
        detector = SIFTFeatureScaleSpace(40, upright=True).to(device, dtype).detector
        image = torch.rand(2, 1, 65, 67, device=device, dtype=dtype)
        image[1] = 0
        responses, lafs, filled, pyramid, octaves, levels = detector._detect_with_pyramid(image, 40)
        assert torch.equal(filled, laf_is_filled(lafs))
        assert (octaves[~filled] == -1).all()
        assert (levels[~filled] == -1).all()
        assert not filled[1].any()
        expected_responses, expected_lafs = detector.detect(image, 40)
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

    def test_flat_gradient_backward(self, device, dtype):
        image = torch.zeros(1, 1, 1, 32, 32, device=device, dtype=dtype, requires_grad=True)
        lafs = torch.tensor([[[[3.0, 0, 8], [0, 3.0, 8]]]], device=device, dtype=dtype)
        ids = torch.zeros(1, 1, device=device, dtype=torch.long)
        _, desc = _SIFTScaleSpaceDescriptor()([image], lafs, ids, ids)
        desc.sum().backward()
        assert torch.isfinite(image.grad).all()

    def test_gradients_built_once_per_used_layer(self, device, dtype, monkeypatch):
        import kornia.feature.sift_scale_space as implementation

        pyramid = [torch.rand(2, 1, 3, 32, 32, device=device, dtype=dtype)]
        lafs = torch.tensor([[[[3.0, 0, 8], [0, 3.0, 8]]] * 3] * 2, device=device, dtype=dtype)
        octaves = torch.zeros(2, 3, device=device, dtype=torch.long)
        levels = torch.tensor([[0, 0, 2], [2, 0, 2]], device=device)
        calls = []
        original = implementation.spatial_gradient

        def gradient(image, *args, **kwargs):
            calls.append(image)
            return original(image, *args, **kwargs)

        monkeypatch.setattr(implementation, "spatial_gradient", gradient)
        module = _SIFTScaleSpaceDescriptor()
        _, desc = module(pyramid, lafs, octaves, levels)
        assert len(calls) == 2
        for batch in range(2):
            for feature in range(3):
                _, single = module(
                    [pyramid[0][batch : batch + 1]],
                    lafs[batch : batch + 1, feature : feature + 1],
                    octaves[batch : batch + 1, feature : feature + 1],
                    levels[batch : batch + 1, feature : feature + 1],
                )
                self.assert_close(desc[batch, feature], single[0, 0])

    def test_upright_image_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        image = torch.rand(1, 1, 1, 12, 12, device=device, dtype=torch.float64)
        lafs = torch.tensor([[[[1.0, 0, 3], [0, 1.0, 3]]]], device=device, dtype=torch.float64)
        ids = torch.zeros(1, 1, device=device, dtype=torch.long)
        module = _SIFTScaleSpaceDescriptor()
        self.gradcheck(lambda image: module([image], lafs, ids, ids, upright=True)[1], (image,))
