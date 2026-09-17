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
# limitations under the License.

from __future__ import annotations

import torch

from kornia.feature import SIFTDescriptorFromPyramid, get_laf_orientation, laf_from_center_scale_ori

from testing.base import BaseTester


class TestSIFTDescriptorFromPyramid(BaseTester):
    def test_shape_norm_and_empty(self, device, dtype):
        image = torch.rand(2, 1, 64, 80, device=device, dtype=dtype)
        xy = torch.tensor([[[24.0, 24.0], [48.0, 40.0]], [[20.0, 30.0], [52.0, 36.0]]], device=device, dtype=dtype)
        scale = torch.full((2, 2, 1, 1), 8.0, device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, scale)
        feature = SIFTDescriptorFromPyramid().to(device, dtype)
        oriented, descriptors = feature.orient_and_describe(image, lafs)
        assert oriented.shape == lafs.shape
        assert descriptors.shape == (2, 2, 128)
        self.assert_close(descriptors.norm(dim=-1), torch.ones(2, 2, device=device, dtype=dtype))
        empty_lafs, empty_descriptors = feature.orient_and_describe(image, lafs[:, :0])
        assert empty_lafs.shape == (2, 0, 2, 3)
        assert empty_descriptors.shape == (2, 0, 128)

    def test_horizontal_gradient_keeps_orientation(self, device, dtype):
        image = torch.arange(64, device=device, dtype=dtype).reshape(1, 1, 1, 64).expand(1, 1, 64, 64)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype))
        oriented, _ = SIFTDescriptorFromPyramid(rootsift=False).to(device, dtype).orient_and_describe(image, lafs)
        self.assert_close(
            get_laf_orientation(oriented), torch.zeros(1, 1, 1, device=device, dtype=dtype), rtol=0.0, atol=1e-2
        )

    def test_flat_input_has_finite_backward(self, device, dtype):
        image = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype, requires_grad=True)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype))
        _, descriptors = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        descriptors.sum().backward()
        assert torch.isfinite(image.grad).all()

    def test_odd_spatial_pooling(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype))
        _, descriptors = (
            SIFTDescriptorFromPyramid(spatial_bin_size=3).to(device, dtype).orient_and_describe(image, lafs)
        )
        assert descriptors.shape == (1, 1, 128)

    def test_invalid_laf_and_mixed_laf_dtype(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        laf_dtype = torch.float32 if device.type == "mps" else torch.float64
        lafs = torch.zeros(1, 2, 2, 3, device=device, dtype=laf_dtype)
        lafs[0, 1] = torch.tensor([[8.0, 0.0, 32.0], [0.0, 8.0, 32.0]], device=device, dtype=laf_dtype)
        oriented, descriptors = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        assert oriented.dtype == lafs.dtype
        assert descriptors.dtype == dtype
        self.assert_close(descriptors[0, 0], torch.zeros(128, device=device, dtype=dtype))
        assert torch.isfinite(descriptors).all()

    def test_upright_preserves_laf_orientation(self, device, dtype):
        image = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        xy = torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(
            xy,
            torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype),
            torch.full((1, 1, 1), 25.0, device=device, dtype=dtype),
        )
        oriented, _ = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs, upright=True)
        self.assert_close(get_laf_orientation(oriented), get_laf_orientation(lafs))

    def test_affine_gradient_is_canonicalized(self, device, dtype):
        # I(x,y)=x+y has a known gradient, so A^T(1,1) tests shear,
        # anisotropic magnitude, nonzero orientation, and the LAF rotation sign.
        axis = torch.arange(65, device=device, dtype=dtype)
        image = (axis[:, None] + axis[None, :])[None, None]
        lafs = torch.tensor([[[[8.0, 2.0, 32.0], [0.0, 6.0, 32.0]]]], device=device, dtype=dtype)
        oriented, descriptors = (
            SIFTDescriptorFromPyramid(rootsift=False).to(device, dtype).orient_and_describe(image, lafs)
        )
        gradient = oriented[..., :2, :2].transpose(-1, -2) @ torch.ones(2, 1, device=device, dtype=dtype)
        # Histogram discretization interpolates a 45-degree input between bins.
        assert gradient[..., 0, 0].min() > 0
        assert (gradient[..., 1, 0].abs() / gradient[..., 0, 0]).max() < 0.06
        assert descriptors.reshape(1, 1, 8, 16).sum(-1).argmax(-1).item() == 0

    def test_odd_pyramid_coordinates(self, device, dtype):
        # Linear images remain linear away from borders. Sampling any octave at
        # the transformed frame centre must return the original pixel coordinate.
        h, w = 65, 67
        x = torch.arange(w, device=device, dtype=dtype)[None].expand(h, w)
        y = torch.arange(h, device=device, dtype=dtype)[:, None].expand(h, w)
        feature = SIFTDescriptorFromPyramid().to(device, dtype)
        lafs = torch.tensor([[[[20.0, 0.0, 27.0], [0.0, 20.0, 31.0]]]], device=device, dtype=dtype)
        pyramid = feature._pyramid(torch.stack([x, y])[None])
        level_lafs = feature._laf_at_level(lafs, pyramid, 1)
        sampled = feature._sample(pyramid[1], pyramid[1], level_lafs, 1)
        self.assert_close(sampled.flatten(), lafs[0, 0, :, 2])

    def test_nonfinite_frame_backward(self, device, dtype):
        image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype, requires_grad=True)
        lafs = torch.tensor(
            [[[[8.0, 0.0, float("nan")], [0.0, 8.0, 20.0]], [[8.0, 0.0, 20.0], [0.0, 8.0, 20.0]]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        _, desc = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        desc.sum().backward()
        self.assert_close(desc[0, 0], torch.zeros_like(desc[0, 0]))
        assert torch.isfinite(image.grad).all()
        assert torch.isfinite(lafs.grad).all()

    def test_gradcheck(self, device):
        image = torch.rand(1, 1, 8, 8, device=device, dtype=torch.float64)
        lafs = torch.tensor([[[[2.0, 0.0, 3.0], [0.0, 2.0, 3.0]]]], device=device, dtype=torch.float64)
        feature = SIFTDescriptorFromPyramid(spatial_bin_size=3).to(device, torch.float64)
        self.gradcheck(lambda image: feature(image, lafs), (image,))

    def test_empty_batch(self, device, dtype):
        image = torch.empty(0, 1, 40, 40, device=device, dtype=dtype)
        lafs = torch.empty(0, 3, 2, 3, device=device, dtype=dtype)
        oriented, desc = SIFTDescriptorFromPyramid().to(device, dtype).orient_and_describe(image, lafs)
        assert oriented.shape == (0, 3, 2, 3)
        assert desc.shape == (0, 3, 128)

    def test_descriptor_forward_does_not_reorient(self, device, dtype, monkeypatch):
        image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype)
        lafs = torch.tensor([[[[6.0, 2.0, 20.0], [-2.0, 6.0, 20.0]]]], device=device, dtype=dtype)
        original = lafs.clone()
        descriptor = SIFTDescriptorFromPyramid().to(device, dtype)

        def unexpected_orientation(*args):
            raise AssertionError("Descriptor forward must respect the supplied orientation")

        monkeypatch.setattr(descriptor, "_orientation", unexpected_orientation)
        descriptors = descriptor(image, lafs)
        assert descriptors.shape == (1, 1, 128)
        self.assert_close(lafs, original)

    def test_orientation_and_description_share_one_pyramid(self, device, dtype, monkeypatch):
        image = torch.rand(1, 1, 40, 40, device=device, dtype=dtype)
        lafs = torch.tensor([[[[8.0, 0.0, 20.0], [0.0, 8.0, 20.0]]]], device=device, dtype=dtype)
        descriptor = SIFTDescriptorFromPyramid().to(device, dtype)
        original = descriptor._pyramid
        calls = []

        def counted_pyramid(image):
            calls.append(image.shape)
            return original(image)

        monkeypatch.setattr(descriptor, "_pyramid", counted_pyramid)
        descriptor.orient_and_describe(image, lafs)
        assert len(calls) == 1
