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
from kornia.feature.sandesc import SANDesc
from kornia.geometry import normalize_pixel_coordinates

from testing.base import BaseTester


def _ramp_image(
    batch: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic ``[0, 1]`` image: a horizontal ramp broadcast over channels (no RNG)."""
    ramp = torch.arange(width, device=device, dtype=dtype) / max(width - 1, 1)
    return ramp.expand(height, width)[None, None].expand(batch, channels, height, width).contiguous()


def _ramp_keypoints(
    batch: int, n: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
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
        volume = model.extract_dense_map(img)
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
        volume = model.extract_dense_map(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype))
        assert volume.shape[-2:] == (32, 32)
        # attention and the extra blocks hang off the skip path, so they must not add
        # parameters that the forward pass never reaches
        extra = [n for n, _ in model.named_parameters() if "cbam" in n or "block2" in n or "block3" in n]
        assert extra == [] or skip_connection

    def test_non_uniform_channels(self, device, dtype):
        """``align`` sits on the residual path, so its input is the coarser block's width."""
        model = SANDesc(skip_connection=True, up_output_channels=[64, 128, 64, 32]).to(device, dtype).eval()
        volume = model.extract_dense_map(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype))
        assert volume.shape == (1, 32, 32, 32)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_keypoints", [1, 10])
    def test_cardinality(self, device, dtype, batch_size, num_keypoints):
        model = SANDesc().to(device, dtype).eval()
        images = _ramp_image(batch_size, 3, 32, 48, device=device, dtype=dtype)
        keypoints = _ramp_keypoints(batch_size, num_keypoints, device=device, dtype=dtype)
        descriptors = model.describe(images, keypoints)
        volume = model.describe(images)
        assert descriptors.shape == (batch_size, num_keypoints, 128)
        assert volume.shape == (batch_size, 128, 32, 48)

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = SANDesc().to(device, dtype).eval()
        img = _ramp_image(1, 3, 32, 32, device=device, dtype=dtype)
        op_optimized = torch_optimizer(model.extract_dense_map)
        self.assert_close(model.extract_dense_map(img), op_optimized(img))

    def test_describe(self, device, dtype):
        model = SANDesc().to(device, dtype).eval()
        des_dim = model.extract_dense_map(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype)).shape[1]
        images = _ramp_image(2, 3, 64, 64, device=device, dtype=dtype)
        keypoints = _ramp_keypoints(2, 10, device=device, dtype=dtype)
        descriptors = model.describe(images, keypoints)
        assert descriptors.shape == (2, 10, des_dim)
        norms = descriptors.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))

        volume = model.describe(images)
        assert volume.shape == (2, des_dim, 64, 64)
        norms = volume.norm(dim=1)
        self.assert_close(norms, torch.ones_like(norms))

    def test_align_corners(self, device, dtype):
        model = SANDesc().to(device, dtype).eval()
        images = _ramp_image(1, 3, 32, 32, device=device, dtype=dtype)
        keypoints = _ramp_keypoints(1, 5, device=device, dtype=dtype)

        # the default keypoint_align_corners is ALIKED's convention
        assert model.keypoint_align_corners is True
        default = model.describe(images, keypoints)
        self.assert_close(default, model.describe(images, keypoints, align_corners=True))

        # a per-call override takes precedence over the instance attribute
        overridden = model.describe(images, keypoints, align_corners=False)
        assert not torch.allclose(default, overridden)

        # setting the instance attribute changes the default used when no override is passed
        model.keypoint_align_corners = False
        self.assert_close(model.describe(images, keypoints), overridden)

    @pytest.mark.parametrize(("height", "width"), [(30, 32), (32, 30), (30, 33)])
    def test_pad_if_not_divisible(self, device, dtype, height, width):
        """Height-only, width-only and both-non-divisible each raise, or pad and crop back."""
        model = SANDesc().to(device, dtype).eval()
        images = _ramp_image(1, 3, height, width, device=device, dtype=dtype)
        keypoints = _ramp_keypoints(1, 10, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            model.extract_dense_map(images)
        volume = model.extract_dense_map(images, pad_if_not_divisible=True)
        assert volume.shape[-2:] == (height, width)
        descriptors = model.describe(images, keypoints, pad_if_not_divisible=True)
        assert descriptors.shape == (1, 10, volume.shape[1])

    @pytest.mark.parametrize(("height", "width"), [(1, 1), (4, 7), (8, 8), (15, 16)])
    def test_small_images(self, device, dtype, height, width):
        """Images smaller than the 16x downsampling factor survive the padded round trip."""
        model = SANDesc().to(device, dtype).eval()
        images = _ramp_image(1, 3, height, width, device=device, dtype=dtype)
        volume = model.extract_dense_map(images, pad_if_not_divisible=True)
        assert volume.shape[-2:] == (height, width)
        assert torch.isfinite(volume).all()

        keypoints = _ramp_keypoints(1, 4, device=device, dtype=dtype)
        descriptors = model.describe(images, keypoints, pad_if_not_divisible=True)
        assert torch.isfinite(descriptors).all()

    def test_large_image(self, device):
        """A resolution well past the sizes used elsewhere still produces a finite volume."""
        model = SANDesc().to(device).eval()
        images = _ramp_image(1, 3, 512, 512, device=device)
        with torch.no_grad():
            volume = model.extract_dense_map(images)
        assert volume.shape == (1, 128, 512, 512)
        assert torch.isfinite(volume).all()

    def test_extreme_value_images(self, device, dtype):
        """Constant images have zero spatial variance; normalization must not divide by zero."""
        model = SANDesc().to(device, dtype).eval()
        keypoints = _ramp_keypoints(1, 4, device=device, dtype=dtype)
        for value in (0.0, 1.0):
            images = torch.full((1, 3, 32, 32), value, device=device, dtype=dtype)
            volume = model.extract_dense_map(images)
            assert torch.isfinite(volume).all()
            descriptors = model.describe(images, keypoints)
            assert torch.isfinite(descriptors).all()
            norms = descriptors.norm(dim=-1)
            self.assert_close(norms, torch.ones_like(norms))

    def test_out_of_bounds_keypoints(self, device, dtype):
        """Keypoints outside [-1, 1] sample the zero padding: zero descriptors, never NaN.

        This is the failure mode of passing pixel coordinates instead of normalized ones, so
        the zero norm is the signal the caller has to notice -- it must not become NaN, which
        would only surface much later in a matching pipeline.
        """
        model = SANDesc().to(device, dtype).eval()
        images = _ramp_image(1, 3, 32, 32, device=device, dtype=dtype)
        keypoints = torch.tensor([[[5.0, 5.0], [-3.0, 0.0], [0.0, 0.0]]], device=device, dtype=dtype)
        descriptors = model.describe(images, keypoints)
        assert torch.isfinite(descriptors).all()
        norms = descriptors.norm(dim=-1)[0]
        self.assert_close(norms[:2], torch.zeros_like(norms[:2]))
        self.assert_close(norms[2], torch.ones_like(norms[2]))

    def test_wrong_n_channels(self, device, dtype):
        """Channel mismatch raises a descriptive ValueError, as DISK's U-Net does."""
        model = SANDesc().to(device, dtype).eval()
        for channels in (1, 4):
            with pytest.raises(ValueError):
                model.extract_dense_map(_ramp_image(1, channels, 32, 32, device=device, dtype=dtype))

        # the check keys off ch_in, so a matching model accepts the same input
        gray_model = SANDesc(ch_in=1).to(device, dtype).eval()
        volume = gray_model.extract_dense_map(_ramp_image(1, 1, 32, 32, device=device, dtype=dtype))
        assert volume.shape == (1, 128, 32, 32)

    def test_empty_keypoints(self, device, dtype):
        """Describing zero keypoints returns an empty (B, 0, D) tensor instead of crashing."""
        model = SANDesc().to(device, dtype).eval()
        images = _ramp_image(1, 3, 32, 32, device=device, dtype=dtype)
        descriptors = model.describe(images, torch.zeros(1, 0, 2, device=device, dtype=dtype))
        assert descriptors.shape == (1, 0, 128)
        assert torch.isfinite(descriptors).all()

    def test_exception(self, device, dtype):
        model = SANDesc().to(device, dtype).eval()
        # forward requires spatial dims that are multiples of 16 unless padding is requested
        with pytest.raises(ValueError):
            model.extract_dense_map(_ramp_image(1, 3, 30, 32, device=device, dtype=dtype))
        # forward requires a (B, C, H, W) image
        with pytest.raises(ShapeError):
            model.extract_dense_map(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype)[0])
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

    def test_keypoint_detector(self, device):
        """``keypoint_detector`` builds the paired detector and fixes the sampling convention."""
        model = SANDesc(keypoint_detector=True, num_keypoints=16).to(device).eval()
        assert model.keypoint_detector is not None
        assert model.keypoint_align_corners is True

        images = _ramp_image(2, 3, 64, 64, device=device)
        with torch.no_grad():
            keypoints, scores, descriptors = model(images)
        assert keypoints.shape == (2, 16, 2)
        assert scores.shape == (2, 16)
        assert descriptors.shape == (2, 16, 128)
        norms = descriptors.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))

        # keypoints come back in pixel coordinates, inside the image
        assert (keypoints[..., 0] >= 0).all() and (keypoints[..., 0] <= 64).all()
        assert (keypoints[..., 1] >= 0).all() and (keypoints[..., 1] <= 64).all()

    def test_keypoint_detector_has_no_descriptors(self, device):
        """SANDesc supplies the descriptors, so the detector's own head is not built."""
        model = SANDesc(keypoint_detector=True, num_keypoints=16).to(device).eval()
        assert model.keypoint_detector.desc_head is None

    def test_forward_without_detector_exception(self, device, dtype):
        # forward needs a detector, which the default constructor does not build
        model = SANDesc().to(device, dtype).eval()
        assert model.keypoint_detector is None
        with pytest.raises(RuntimeError):
            model(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype))
        # asking for more keypoints than the image has pixels is caught before the detector runs
        budget = SANDesc(keypoint_detector=True, num_keypoints=5000).to(device, dtype).eval()
        with pytest.raises(ValueError):
            budget(_ramp_image(1, 3, 32, 32, device=device, dtype=dtype))

    def test_gradcheck(self, device):
        img = _ramp_image(1, 3, 16, 16, device=device, dtype=torch.float64)
        model = SANDesc().to(device, img.dtype).eval()
        self.gradcheck(model.extract_dense_map, (img,), eps=1e-4, atol=1e-4, nondet_tol=1e-4)

    @pytest.mark.slow
    def test_pretrained_values(self, device):
        """Descriptors from the pretrained weights match a saved reference.

        The input is a deterministic horizontal color ramp (no RNG), so the
        reference is reproducible across runs/platforms and guards against
        regressions in the forward/sampling path.
        """
        # Snippet used to generate ``expected`` (requires the pretrained weights):
        # img = _ramp_image(1, 3, 256, 256)  # deterministic horizontal ramp, no RNG
        # kpts = torch.tensor([[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]])
        # expected = SANDesc.from_pretrained(load_detector=False).eval().describe(img, kpts)[0, :, :6]
        expected = [
            [0.012115, 0.012292, -0.015393, -0.012649, 0.017301, -0.000850],
            [0.027959, -0.001189, 0.034381, 0.020057, 0.035708, -0.046235],
            [0.030159, -0.021707, 0.084352, 0.031746, 0.031509, -0.097772],
        ]
        img = _ramp_image(1, 3, 256, 256, device=device)
        kpts = torch.tensor([[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]], device=device)
        model = SANDesc.from_pretrained(load_detector=False).to(device).eval()
        assert model.keypoint_detector is None
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
        sandesc = SANDesc.from_pretrained(load_detector=False).to(device).eval()

        img = _ramp_image(1, 3, 480, 480, device=device)
        des_dim = sandesc.extract_dense_map(_ramp_image(1, 3, 64, 64, device=device)).shape[1]
        with torch.no_grad():
            features = aliked(img)[0]
            keypoints = normalize_pixel_coordinates(features.keypoints, img.shape[-2], img.shape[-1])
            descriptors = sandesc.describe(img, keypoints[None])[0]

        assert features.n > 0
        assert descriptors.shape == (features.n, des_dim)
        norms = descriptors.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))

    @pytest.mark.slow
    def test_pretrained_forward(self, device):
        """``from_pretrained`` bundles a pretrained detector by default, ready for ``forward``.

        Also covers that the strict ``load_state_dict`` of the descriptor checkpoint still
        succeeds once a detector submodule is attached to the model.
        """
        model = SANDesc.from_pretrained(num_keypoints=64).to(device).eval()
        assert model.keypoint_detector is not None
        assert model.keypoint_detector.desc_head is None

        img = _ramp_image(1, 3, 256, 256, device=device)
        with torch.no_grad():
            keypoints, scores, descriptors = model(img)

        assert keypoints.shape == (1, 64, 2)
        assert scores.shape == (1, 64)
        assert descriptors.shape == (1, 64, 128)
        assert (keypoints[..., 0] >= 0).all() and (keypoints[..., 0] <= 255).all()
        assert (keypoints[..., 1] >= 0).all() and (keypoints[..., 1] <= 255).all()
        norms = descriptors.norm(dim=-1)
        self.assert_close(norms, torch.ones_like(norms))
