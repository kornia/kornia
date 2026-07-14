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

import torch

import kornia
from kornia.feature.scale_space_detector import MultiResolutionDetector, ScaleSpaceDetector, get_default_detector_config
from kornia.geometry.subpix import ConvQuadInterp3d

from testing.base import BaseTester


class TestScaleSpaceDetector(BaseTester):
    def test_shape(self, device, dtype):
        inp = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        n_feats = 10
        det = ScaleSpaceDetector(n_feats).to(device, dtype)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, n_feats, 2, 3])
        assert resps.shape == torch.Size([1, n_feats])

    def test_shape_batch(self, device, dtype):
        inp = torch.rand(7, 1, 32, 32, device=device, dtype=dtype)
        n_feats = 10
        det = ScaleSpaceDetector(n_feats).to(device, dtype)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([7, n_feats, 2, 3])
        assert resps.shape == torch.Size([7, n_feats])

    def test_toy(self, device, dtype):
        inp = torch.zeros(1, 1, 33, 33, device=device, dtype=dtype)
        inp[:, :, 13:-13, 13:-13] = 1.0
        n_feats = 1
        det = ScaleSpaceDetector(n_feats, resp_module=kornia.feature.BlobHessian(), mr_size=3.0).to(device, dtype)
        lafs, resps = det(inp)
        expected_laf = torch.tensor([[[[8.4260, 0.0000, 16.0], [0.0, 8.4260, 16.0]]]], device=device, dtype=dtype)
        expected_resp = torch.tensor([[0.1159]], device=device, dtype=dtype)
        self.assert_close(lafs, expected_laf, rtol=0.001, atol=1e-03)
        self.assert_close(resps, expected_resp, rtol=0.001, atol=1e-03)

    def test_toy_mask(self, device, dtype):
        inp = torch.zeros(1, 1, 33, 33, device=device, dtype=dtype)
        inp[:, :, 13:-13, 13:-13] = 1.0

        mask = torch.zeros(1, 1, 33, 33, device=device, dtype=dtype)
        mask[:, :, 1:-1, 3:-3] = 1.0

        n_feats = 1
        det = ScaleSpaceDetector(n_feats, resp_module=kornia.feature.BlobHessian(), mr_size=3.0).to(device, dtype)
        lafs, resps = det(inp, mask)
        expected_laf = torch.tensor([[[[8.4260, 0.0000, 16.0], [0.0, 8.4260, 16.0]]]], device=device, dtype=dtype)
        expected_resp = torch.tensor([[0.1159]], device=device, dtype=dtype)
        self.assert_close(lafs, expected_laf, rtol=0.001, atol=1e-03)
        self.assert_close(resps, expected_resp, rtol=0.001, atol=1e-03)

    def test_minima_are_also_good(self, device, dtype):
        # Image with a bright blob (local max) and dark blob (local min).
        # With minima_are_also_good=True both should contribute to detections.
        inp = torch.ones(1, 1, 33, 33, device=device, dtype=dtype) * 0.5
        inp[:, :, 10:14, 10:14] = 1.0  # bright blob → local maximum
        inp[:, :, 10:14, 20:24] = 0.0  # dark blob → local minimum
        n_feats = 2
        det_max_only = ScaleSpaceDetector(n_feats, resp_module=kornia.feature.BlobHessian(), mr_size=3.0).to(
            device, dtype
        )
        det_minmax = ScaleSpaceDetector(
            n_feats, resp_module=kornia.feature.BlobHessian(), mr_size=3.0, minima_are_also_good=True
        ).to(device, dtype)
        lafs_max, resps_max = det_max_only(inp)
        lafs_minmax, resps_minmax = det_minmax(inp)
        assert lafs_max.shape == torch.Size([1, n_feats, 2, 3])
        assert lafs_minmax.shape == torch.Size([1, n_feats, 2, 3])
        # minmax detector should find a higher total response magnitude (it sees both blobs).
        assert resps_minmax.abs().sum() >= resps_max.abs().sum()

    def test_scale_space_response_mode(self, device, dtype):
        # Smoke test: scale_space_response=True uses a different internal code path.
        # BlobDoG operates on the 5D scale-space tensor directly.
        inp = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        n_feats = 5
        det = ScaleSpaceDetector(n_feats, resp_module=kornia.feature.BlobDoG(), scale_space_response=True).to(
            device, dtype
        )
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, n_feats, 2, 3])
        assert resps.shape == torch.Size([1, n_feats])

    def test_few_detections_padding(self, device, dtype):
        # Constant image → very few (possibly zero) NMS candidates; output must still
        # have the requested shape because the detect() method pads with zeros.
        inp = torch.ones(1, 1, 32, 32, device=device, dtype=dtype)
        n_feats = 20
        det = ScaleSpaceDetector(n_feats, subpix_module=ConvQuadInterp3d(10)).to(device, dtype)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, n_feats, 2, 3])
        assert resps.shape == torch.Size([1, n_feats])

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 7, 7
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        # Use ConvQuadInterp3d for gradcheck — IterativeQuadInterp3d uses non-differentiable
        # indexed in-place assignments that are incompatible with torch.autograd.gradcheck.
        det = ScaleSpaceDetector(2, subpix_module=ConvQuadInterp3d(10)).to(device)
        self.gradcheck(det, patches, nondet_tol=1e-4)


class TestMultiResolutionDetector(BaseTester):
    def _make_detector(self, num_features: int = 50, **config_overrides):
        cfg = get_default_detector_config()
        cfg.update(config_overrides)
        return MultiResolutionDetector(kornia.feature.BlobHessian(), num_features=num_features, config=cfg)

    def test_shape(self, device, dtype):
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = self._make_detector().to(device, dtype)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, 50, 2, 3])
        assert resps.shape == torch.Size([1, 50])

    def test_shape_non_square(self, device, dtype):
        inp = torch.rand(1, 1, 48, 96, device=device, dtype=dtype)
        det = self._make_detector().to(device, dtype)
        lafs, _ = det(inp)
        assert lafs.shape == torch.Size([1, 50, 2, 3])

    def test_lafs_inside_image(self, device, dtype):
        # All detected LAF centers should lie within the image boundaries.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = self._make_detector(num_features=20).to(device, dtype)
        lafs, _ = det(inp)
        cx = lafs[0, :, 0, 2]
        cy = lafs[0, :, 1, 2]
        assert (cx >= 0).all() and (cx <= 64).all()
        assert (cy >= 0).all() and (cy <= 64).all()

    def test_no_upscale_levels(self, device, dtype):
        # up_levels=0 disables the upsampling branch; should still produce valid output.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        cfg = get_default_detector_config()
        cfg["up_levels"] = 0
        cfg["pyramid_levels"] = 2
        det = MultiResolutionDetector(kornia.feature.BlobHessian(), num_features=20, config=cfg).to(device, dtype)
        lafs, _ = det(inp)
        assert lafs.shape == torch.Size([1, 20, 2, 3])

    def test_with_upscale_levels(self, device, dtype):
        # up_levels > 0 exercises the upsampling code path.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        cfg = get_default_detector_config()
        cfg["up_levels"] = 2
        cfg["pyramid_levels"] = 1
        det = MultiResolutionDetector(kornia.feature.BlobHessian(), num_features=20, config=cfg).to(device, dtype)
        lafs, _ = det(inp)
        assert lafs.shape == torch.Size([1, 20, 2, 3])

    def test_score_threshold_reduces_detections(self, device, dtype):
        # A very high score_threshold should leave no real detections: all returned responses
        # will be the sentinel fill value (very negative), while shape remains fixed.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det_no_thresh = self._make_detector(num_features=50).to(device, dtype)
        det_high_thresh = MultiResolutionDetector(
            kornia.feature.BlobHessian(), num_features=50, score_threshold=1e6
        ).to(device, dtype)
        lafs_no_thresh, resps_no_thresh = det_no_thresh(inp)
        lafs_high_thresh, resps_high_thresh = det_high_thresh(inp)
        assert lafs_high_thresh.shape == lafs_no_thresh.shape
        # With an impossibly high threshold all slots contain the fill sentinel (< 0),
        # while real detections always have positive responses.
        assert resps_no_thresh.max().item() > 0
        assert (resps_high_thresh <= 0).all()

    def test_smoke_with_blob_image(self, device, dtype):
        # Synthetic image with a bright blob — detector should find it.
        inp = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        inp[:, :, 28:36, 28:36] = 1.0
        det = self._make_detector(num_features=5).to(device, dtype)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, 5, 2, 3])
        assert resps.abs().max().item() > 0
