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

import pytest
import torch

from kornia.core.exceptions import ShapeError
from kornia.feature.aliked import ALIKED
from kornia.feature.dedode import DeDoDe
from kornia.feature.sandesc import SANDesc
from kornia.geometry import normalize_pixel_coordinates

from testing.base import BaseTester


def _ramp_image(batch, channels, height, width, device=None, dtype=torch.float32):
    """Deterministic ``[0, 1]`` image: a horizontal ramp broadcast over channels (no RNG)."""
    ramp = torch.arange(width, device=device, dtype=dtype) / max(width - 1, 1)
    return ramp.expand(height, width)[None, None].expand(batch, channels, height, width).contiguous()


def _ramp_keypoints(batch, n, device=None, dtype=torch.float32):
    """Deterministic interior keypoints in ``(-1, 1)``: a diagonal sweep, no RNG.

    Endpoints stay off the image border so ``grid_sample`` does not return the
    zero-padding region (which would yield zero-norm descriptors).
    """
    coords = torch.linspace(-0.9, 0.9, n, device=device, dtype=dtype)
    return torch.stack([coords, coords], dim=-1)[None].expand(batch, n, 2).contiguous()


class TestSANDesc(BaseTester):
    def test_smoke(self, device, dtype):
        model = SANDesc().to(device, dtype).eval()
        img = _ramp_image(1, 3, 64, 64, device=device, dtype=dtype)
        volume = model(img)
        assert volume.shape[0] == 1
        assert volume.shape[-2:] == img.shape[-2:]

    @pytest.mark.parametrize("skip_connection", [False, True])
    @pytest.mark.parametrize("spatial_attention", [False, True])
    @pytest.mark.parametrize("third_block", [False, True])
    def test_smoke_options(self, device, dtype, skip_connection, spatial_attention, third_block):
        """The attention/skip variants are what ``from_pretrained`` builds."""
        model = (
            SANDesc(
                skip_connection=skip_connection,
                spatial_attention=spatial_attention,
                third_block=third_block,
            )
            .to(device, dtype)
            .eval()
        )
        volume = model(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype))
        assert volume.shape[-2:] == (32, 32)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_keypoints", [1, 10])
    def test_cardinality(self, device, dtype, batch_size, num_keypoints):
        model = SANDesc().to(device, dtype).eval()
        images = _ramp_image(batch_size, 3, 32, 48, device=device, dtype=dtype)
        keypoints = _ramp_keypoints(batch_size, num_keypoints, device=device, dtype=dtype)
        descriptors, volume = model.describe(images, keypoints, return_desc_volume=True)
        assert descriptors.shape == (batch_size, num_keypoints, 128)
        assert volume.shape == (batch_size, 128, 32, 48)

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = SANDesc().to(device, dtype).eval()
        img = _ramp_image(1, 3, 32, 32, device=device, dtype=dtype)
        op_optimized = torch_optimizer(model)
        self.assert_close(model(img), op_optimized(img))

    def test_describe(self, device, dtype):
        model = SANDesc().to(device, dtype).eval()
        des_dim = model(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype)).shape[1]
        images = _ramp_image(2, 3, 64, 64, device=device, dtype=dtype)
        keypoints = _ramp_keypoints(2, 10, device=device, dtype=dtype)
        descriptors = model.describe(images, keypoints)
        assert descriptors.shape == (2, 10, des_dim)
        norms = descriptors.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))

        descriptors, volume = model.describe(images, keypoints, return_desc_volume=True)
        assert volume.shape == (2, des_dim, 64, 64)

    def test_exception(self, device, dtype):
        model = SANDesc().to(device, dtype).eval()
        # forward requires spatial dims that are multiples of 16
        with pytest.raises(ValueError):
            model(_ramp_image(1, 3, 30, 32, device=device, dtype=dtype))
        # forward requires a (B, C, H, W) image
        with pytest.raises(ShapeError):
            model(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype)[0])
        # describe requires (B, N, 2) keypoints
        with pytest.raises(ShapeError):
            model.describe(
                _ramp_image(1, 3, 32, 32, device=device, dtype=dtype),
                _ramp_keypoints(1, 4, device=device, dtype=dtype)[..., 0],
            )
        # unknown normalization / activation are rejected at construction
        with pytest.raises(ValueError):
            SANDesc(norm="not_a_norm")
        with pytest.raises(ValueError):
            SANDesc(activation="bad")
        # the block wiring is fixed, so channel lists of other lengths are rejected
        with pytest.raises(ValueError):
            SANDesc(down_output_channels=[16, 32, 64, 64])
        with pytest.raises(ValueError):
            SANDesc(up_output_channels=[64, 64, 64])
        # unknown detector in from_pretrained (raised before any download)
        with pytest.raises(ValueError):
            SANDesc.from_pretrained(detector="not_a_detector")

    def test_gradcheck(self, device):
        img = _ramp_image(1, 3, 16, 16, device=device, dtype=torch.float64)
        model = SANDesc().to(device, img.dtype).eval()
        self.gradcheck(model, (img,), eps=1e-4, atol=1e-4, nondet_tol=1e-4)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("detector", "expected"),
        [
            (
                "aliked",
                [
                    [-0.157611, 0.027860, -0.117423, 0.083916, -0.105181, -0.028156],
                    [0.100992, -0.076817, 0.221748, -0.015796, 0.059756, 0.232232],
                    [-0.128114, -0.042794, 0.219844, 0.101577, -0.005671, 0.068021],
                ],
            ),
            (
                "dedode",
                [
                    [0.115363, -0.077948, 0.049350, 0.029223, 0.051257, 0.001052],
                    [0.106579, -0.124094, -0.052825, -0.010934, 0.080602, -0.002354],
                    [-0.036962, -0.198061, 0.025702, -0.019343, 0.017758, 0.014779],
                ],
            ),
        ],
    )
    def test_pretrained_values(self, device, detector, expected):
        """Descriptors from the pretrained weights match a saved reference.

        The input is a deterministic horizontal color ramp (no RNG), so the
        reference is reproducible across runs/platforms and guards against
        regressions in the forward/sampling path.
        """
        # Snippet used to generate ``expected`` (requires the pretrained weights):
        # img = _ramp_image(1, 3, 256, 256)  # deterministic horizontal ramp, no RNG
        # kpts = torch.tensor([[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]])
        # expected = SANDesc.from_pretrained(detector=detector).eval().describe(img, kpts)[0, :, :6]
        img = _ramp_image(1, 3, 256, 256, device=device)
        kpts = torch.tensor([[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]], device=device)
        model = SANDesc.from_pretrained(detector=detector).to(device).eval()
        # cuDNN runs float32 convolutions in TF32 by default, which costs ~5e-4 of accuracy
        # on CUDA and would exceed the tolerance below; conftest only pins matmul precision.
        with torch.backends.cudnn.flags(allow_tf32=False), torch.no_grad():
            desc = model.describe(img, kpts)
        assert desc.shape == (1, 3, 128)
        self.assert_close(desc[0, :, :6], torch.tensor(expected, device=device), atol=1e-4, rtol=1e-4)

    @pytest.mark.slow
    def test_pretrained_aliked_pipeline(self, device):
        """ALIKED detects keypoints; SANDesc (aliked weights) describes them."""
        aliked = ALIKED.from_pretrained(model_name="aliked-n16", device=device)
        sandesc = SANDesc.from_pretrained(detector="aliked").to(device).eval()

        img = _ramp_image(1, 3, 480, 480, device=device)
        des_dim = sandesc(_ramp_image(1, 3, 64, 64, device=device)).shape[1]
        with torch.no_grad():
            features = aliked(img)[0]
            keypoints = normalize_pixel_coordinates(features.keypoints, img.shape[-2], img.shape[-1])
            descriptors = sandesc.describe(img, keypoints[None])[0]

        assert features.n > 0
        assert descriptors.shape == (features.n, des_dim)
        norms = descriptors.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))

    @pytest.mark.slow
    def test_pretrained_dedode_pipeline(self, device):
        """DeDoDe detects keypoints; SANDesc (dedode weights) describes them."""
        dedode = DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-upright").to(device)
        sandesc = SANDesc.from_pretrained(detector="dedode").to(device).eval()

        img = _ramp_image(1, 3, 480, 480, device=device)
        des_dim = sandesc(_ramp_image(1, 3, 64, 64, device=device)).shape[1]
        with torch.no_grad():
            keypoints, _scores = dedode.detect(img, n=1000)
            descriptors = sandesc.describe(img, keypoints)[0]

        assert descriptors.shape == (keypoints.shape[1], des_dim)
        norms = descriptors.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))
