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

import kornia
from kornia.core.check import ShapeError
from kornia.feature.scale_space_detector import (
    _MAX_ABS_SIN_12,
    MultiResolutionDetector,
    ScaleSpaceDetector,
    get_default_detector_config,
)
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

    def test_mask_does_not_promote_the_response_dtype(self, device, dtype):
        # The mask is resampled and cast onto the response map, so a mask in a different dtype
        # cannot pull the detector's output away from the image dtype.
        inp = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        other = torch.float64 if dtype != torch.float64 else torch.float32
        if device.type == "mps" and other == torch.float64:  # MPS has no float64
            other = torch.float32 if dtype != torch.float32 else torch.float16
        assert other != dtype
        mask = torch.ones(1, 1, 32, 32, device=device, dtype=other)
        det = ScaleSpaceDetector(5).to(device, dtype)
        lafs, resps = det(inp, mask)
        assert lafs.dtype == dtype
        assert resps.dtype == dtype

    def test_mask_spatial_size_must_match_the_image(self, device, dtype):
        # `_create_octave_mask` resamples the mask onto every octave, so without this check a
        # mask of any size is stretched silently onto the wrong geometry.
        det = ScaleSpaceDetector(5).to(device, dtype)
        inp = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        with pytest.raises(Exception, match="spatial size"):
            det(inp, torch.ones(1, 1, 8, 8, device=device, dtype=dtype))
        with pytest.raises(Exception, match="spatial size"):
            det(inp, torch.ones(1, 1, 32, 16, device=device, dtype=dtype))

    def test_mask_must_be_single_channel(self, device, dtype):
        # `_create_octave_mask` broadcasts the mask over the scale levels, so a channel axis would
        # land on the level axis: an error when the counts differ, the wrong weighting when they match.
        det = ScaleSpaceDetector(5).to(device, dtype)
        inp = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        with pytest.raises(Exception, match="1, H, W"):
            det(inp, torch.ones(2, 3, 32, 32, device=device, dtype=dtype))
        with pytest.raises(Exception, match="batch"):
            det(inp, torch.ones(3, 1, 32, 32, device=device, dtype=dtype))
        for b in (1, 2):
            lafs, _ = det(inp, torch.ones(b, 1, 32, 32, device=device, dtype=dtype))
            assert lafs.shape == torch.Size([2, 5, 2, 3])

    def test_unfilled_slots_carry_a_zero_laf(self, device, dtype):
        # A slot that no candidate filled -- the padding -- has a zero response and a zero LAF.
        # The converse is deliberately not true: a candidate the border check rejects keeps its
        # coordinates beside its zeroed response, so the zero LAFs are a strict subset of the
        # zero-response slots. Inferring the padding from `response == 0` would collapse the two.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = ScaleSpaceDetector(50).to(device, dtype)
        resps, lafs = det.detect(inp, 50)
        zero_laf = (lafs[0] == 0).all(dim=-1).all(dim=-1)
        empty = resps[0] == 0
        assert bool(zero_laf.any()), "expected this image to under-fill 50 slots"
        assert bool((empty[zero_laf]).all()), "a zero LAF must carry a zero response"
        assert bool((empty & ~zero_laf).any()), "a border-rejected candidate keeps its frame"
        lafs_fwd, resps_fwd = det(inp)
        assert torch.equal(resps_fwd, resps) and torch.equal(lafs_fwd, lafs)

    def test_a_zero_response_maximum_keeps_its_laf(self, device, dtype):
        # The response function is pluggable and may be signed, so an exact zero can be a genuine
        # maximum rather than an unfilled slot. Classifying the padding by `response == 0` erased
        # all three detections here.
        class SignedResponse(torch.nn.Module):
            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                out = -torch.ones_like(x)
                out[:, :, out.shape[2] // 2, out.shape[-2] // 2, out.shape[-1] // 2] = 0.0
                return out

        det = ScaleSpaceDetector(5, resp_module=SignedResponse(), scale_space_response=True).to(device, dtype)
        lafs, resps = det(torch.zeros(1, 1, 96, 96, device=device, dtype=dtype))
        assert (resps == 0).all()
        centers = lafs[0, :, :, 2]
        found = (centers == 48).all(dim=-1)
        assert int(found.sum()) == 3, f"expected the three centre maxima, got {centers.tolist()}"

    def test_batched_underfill_does_not_leak_the_topk_sentinel(self, device, dtype):
        # For B > 1 the octave top-K ranks over the whole volume with non-candidates masked to
        # `finfo.min / 2`. An image with fewer maxima than requested gets those sentinels back;
        # they are not detections and must not reach the caller as a response or as a LAF.
        det = ScaleSpaceDetector().to(device, dtype)
        lafs, resps = det(torch.zeros(2, 1, 32, 32, device=device, dtype=dtype))
        assert (resps == 0).all(), f"sentinel leaked: min response {resps.min().item()}"
        assert (lafs == 0).all()
        # and the single-image path agrees, which it did not before
        lafs1, resps1 = det(torch.zeros(1, 1, 32, 32, device=device, dtype=dtype))
        assert torch.equal(resps1[0], resps[0]) and torch.equal(lafs1[0], lafs[0])

    def test_padding_survives_the_affine_and_orientation_modules(self, device, dtype):
        # Same contract as `MultiResolutionDetector.forward`: the shape and orientation modules
        # normalise a zero frame into a finite one, so the padding is re-applied after them.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("LAFAffineShapeEstimator uses linalg.inv, which has no half-precision kernel")
        det = ScaleSpaceDetector(
            10, aff_module=kornia.feature.LAFAffineShapeEstimator(19), ori_module=kornia.feature.LAFOrienter(19)
        ).to(device, dtype)
        lafs, resps = det(torch.zeros(1, 1, 64, 64, device=device, dtype=dtype))
        assert (resps == 0).all()
        assert (lafs == 0).all()

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

    def test_inline_boundary_constant_matches_reference(self):
        """`_MAX_ABS_SIN_12` must be the x-extent `laf_to_boundary_points(laf, 12)` actually produces.

        `_process_octave` inlines `laf_is_inside_image(scale_laf(lafs, 0.5), octave, 5)` for the
        isotropic LAFs it builds. That inline form is only equivalent to the reference if the
        constant is the maximum ``|sin|`` over the angles the reference samples, which are
        ``linspace(0, 2 * pi, n_pts - 1)`` -- spacing ``2 * pi / 10``, not ``2 * pi / 11`` (#4064).
        Device- and dtype-independent: this pins a Python float against a sampling convention.
        """
        cx, cy, half_s = 10.0, 20.0, 3.0
        laf = kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[cx, cy]]], dtype=torch.float64),
            torch.full((1, 1, 1, 1), half_s, dtype=torch.float64),
        )
        pts = kornia.feature.laf_to_boundary_points(laf, 12)
        # `laf_to_boundary_points` builds its angles in float32 before casting, so compare at
        # float32 resolution -- the wrong spacing is off by 4e-2, four orders of magnitude more.
        assert abs(((pts[..., 0].max() - cx) / half_s).item() - _MAX_ABS_SIN_12) < 1e-6
        # ... and the y-extent the same block hardcodes as `half_s` (max|cos| = 1).
        assert abs(((pts[..., 1].max() - cy) / half_s).item() - 1.0) < 1e-6

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
        # A very high score_threshold should leave no real detections: every slot is padding,
        # which reads as an exactly zero response, while the shape remains fixed.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det_no_thresh = self._make_detector(num_features=50).to(device, dtype)
        det_high_thresh = MultiResolutionDetector(
            kornia.feature.BlobHessian(), num_features=50, score_threshold=1e6
        ).to(device, dtype)
        lafs_no_thresh, resps_no_thresh = det_no_thresh(inp)
        lafs_high_thresh, resps_high_thresh = det_high_thresh(inp)
        assert lafs_high_thresh.shape == lafs_no_thresh.shape
        # With an impossibly high threshold every slot is padding: a zero response and a zero
        # LAF. Real detections always have positive responses.
        assert resps_no_thresh.max().item() > 0
        assert (resps_high_thresh == 0).all()
        assert (lafs_high_thresh == 0).all()

    def test_short_result_is_padded_with_zeros(self, device, dtype):
        # A level with fewer above-threshold maxima than its quota must not return the
        # `torch.finfo(dtype).min / 2` top-K sentinel, nor the arbitrary border coordinates
        # `topk` picks once every remaining candidate is tied at that value.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = self._make_detector(num_features=100).to(device, dtype)
        lafs, resps = det(inp)

        assert resps.shape == torch.Size([1, 100])
        assert (resps >= 0).all()
        padding = resps[0] == 0
        assert bool(padding.any()), "expected this image to under-fill 100 slots"
        assert (lafs[0][padding] == 0).all()
        assert (resps[0][~padding] > 0).all()

    def test_mask_suppresses_detections(self, device, dtype):
        # Two blobs: a bright one top-left, a dimmer one bottom-right. Unmasked, the bright
        # blob wins; with a mask covering only the bottom-right quadrant the dim blob must be
        # the best -- and nothing may be detected outside the mask.
        inp = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        inp[:, :, 14:19, 14:19] = 1.0
        inp[:, :, 46:51, 46:51] = 0.5
        mask = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        mask[:, :, 32:, 32:] = 1.0

        det = self._make_detector(num_features=10).to(device, dtype)
        lafs_free, resps_free = det(inp)
        lafs_masked, resps_masked = det(inp, mask)

        assert lafs_free[0, int(resps_free[0].argmax()), 0, 2].item() < 32
        assert lafs_masked[0, int(resps_masked[0].argmax()), 0, 2].item() > 32
        found = resps_masked[0] != 0
        assert bool(found.any())
        # The mask is resampled bilinearly onto every level, so its edge softens by a level
        # pixel or two; 40 keeps the bound well clear of that without weakening the check
        # (the unmasked run puts its best detection at ~17.8).
        assert (lafs_masked[0][found][:, :, 2] >= 40).all()

    def test_mask_spatial_size_must_match_the_image(self, device, dtype):
        # `KORNIA_CHECK_SHAPE`'s named dims are free per call, so on their own they let a mask
        # of any size through to be stretched onto the image; the explicit check is what rejects it.
        det = self._make_detector(num_features=10).to(device, dtype)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        with pytest.raises(Exception, match="spatial size"):
            det(inp, torch.ones(1, 1, 32, 32, device=device, dtype=dtype))
        with pytest.raises(Exception, match="spatial size"):
            det(inp, torch.ones(1, 1, 64, 32, device=device, dtype=dtype))

    def test_bool_mask_matches_float_mask(self, device, dtype):
        # A boolean mask has no bilinear kernel; it is cast before the resample and gives the
        # same detections as the equivalent 0/1 float mask instead of raising.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        mask = torch.zeros(1, 1, 64, 64, device=device, dtype=torch.bool)
        mask[:, :, 32:, 32:] = True
        det = self._make_detector(num_features=10).to(device, dtype)
        lafs_bool, resps_bool = det(inp, mask)
        lafs_float, resps_float = det(inp, mask.to(dtype))
        assert bool((resps_bool != 0).any())
        assert torch.equal(resps_bool, resps_float)
        assert torch.equal(lafs_bool, lafs_float)
        # ... and the mask is doing something, so the two arms cannot agree vacuously the way
        # they would if the mask were ignored altogether.
        _lafs_free, resps_free = det(inp)
        assert not torch.equal(resps_bool, resps_free)

    def test_padding_survives_the_affine_and_orientation_modules(self, device, dtype):
        # `forward` runs `aff` and `ori` after `detect`; `LAFAffineShapeEstimator` normalises a
        # zero frame into a finite 1e-5-scale one, so the zero-LAF contract has to be re-applied.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("LAFAffineShapeEstimator uses linalg.inv, which has no half-precision kernel")
        det = MultiResolutionDetector(
            kornia.feature.BlobHessian(),
            num_features=10,
            aff_module=kornia.feature.LAFAffineShapeEstimator(19),
            ori_module=kornia.feature.LAFOrienter(19),
        ).to(device, dtype)
        lafs, resps = det(torch.zeros(1, 1, 64, 64, device=device, dtype=dtype))
        assert (resps == 0).all()
        assert (lafs == 0).all()

    def test_zero_mask_detects_nothing(self, device, dtype):
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        mask = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        det = self._make_detector(num_features=20).to(device, dtype)
        lafs, resps = det(inp, mask)
        assert (resps == 0).all()
        assert (lafs == 0).all()

    @staticmethod
    def _require_half_response(device, half_dtype):
        if device.type == "mps":
            pytest.skip("MPS autocast changes the effective dtype")
        # `BlobHessian` reaches `spatial_gradient`, which pads with mode="replicate".
        # `replication_pad2d` has no CPU float16 kernel before torch 2.6.
        try:
            torch.nn.functional.pad(torch.zeros(1, 1, 4, 4, device=device, dtype=half_dtype), (1, 1, 1, 1), "replicate")
        except RuntimeError as err:
            pytest.skip(f"torch has no replicate-padding kernel for {half_dtype} on {device.type}: {err}")

    @pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_dtype_is_preserved(self, device, half_dtype):
        # `detect_features_on_single_level` used to hardcode `.float()` on the pixel
        # coordinates, so half-precision input came back with float32 LAFs beside
        # half-precision responses.
        self._require_half_response(device, half_dtype)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=half_dtype)
        det = self._make_detector(num_features=20).to(device, half_dtype)
        lafs, resps = det(inp)
        assert lafs.dtype == half_dtype
        assert resps.dtype == half_dtype

    @pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
    def test_mask_does_not_promote_the_response_dtype(self, device, half_dtype):
        # A float32 mask multiplied into a half-precision response map used to promote the
        # responses to float32 while the LAFs stayed half.
        self._require_half_response(device, half_dtype)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=half_dtype)
        mask = torch.ones(1, 1, 64, 64, device=device, dtype=torch.float32)
        det = self._make_detector(num_features=20).to(device, half_dtype)
        lafs, resps = det(inp, mask)
        assert lafs.dtype == half_dtype
        assert resps.dtype == half_dtype

    def test_multichannel_response_stays_inside_the_image(self, device, dtype):
        # `BlobHessian` keeps the image channels, so an RGB image gives a 3-channel response.
        # Decoding the flat top-K index with the width alone placed a candidate from channel `c`
        # at `y + c * H`, i.e. outside the image for every channel past the first.
        torch.manual_seed(3)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        det = self._make_detector(num_features=100).to(device, dtype)
        lafs, resps = det(inp)
        found = resps[0] != 0
        assert bool(found.any())
        cx = lafs[0, :, 0, 2]
        cy = lafs[0, :, 1, 2]
        assert (cx >= 0).all() and (cx <= 63).all()
        assert (cy >= 0).all() and (cy <= 63).all()

    def test_result_is_padded_to_num_features(self, device, dtype):
        # An 8x8 image is smaller than the 15px border `remove_borders` strips, so there is
        # genuinely nothing to detect and every slot is padding. A level is also capped at its
        # own pixel count (64 here), so the concatenated result used to come back short.
        cfg = get_default_detector_config()
        cfg["pyramid_levels"] = 0
        cfg["up_levels"] = 0
        tiny = MultiResolutionDetector(kornia.feature.BlobHessian(), num_features=100, config=cfg).to(device, dtype)
        lafs, resps = tiny(torch.rand(1, 1, 8, 8, device=device, dtype=dtype))
        assert lafs.shape == torch.Size([1, 100, 2, 3])
        assert resps.shape == torch.Size([1, 100])
        # Say so explicitly, so this test cannot pass by codifying a dummy result elsewhere.
        assert (resps == 0).all()
        assert (lafs == 0).all()

    def test_short_result_is_sorted(self, device, dtype):
        # `detect` used to run its final top-K only when the levels had produced *more* slots than
        # `num_features`, so a short result came back in level order with each level's own padding
        # left in place and the real detections scattered through it.
        det = self._make_detector(num_features=6000, pyramid_levels=2, up_levels=0).to(device, dtype)
        resps, _lafs = det.detect(torch.rand(1, 1, 48, 48, device=device, dtype=dtype))
        found = int((resps[0] != 0).sum())
        assert 0 < found < 6000, f"expected a short result, got {found} detections"
        assert bool((resps[0][:found] != 0).all()), "the detections do not come first"
        assert torch.equal(resps[0], torch.sort(resps[0], descending=True).values)

    def test_single_feature_request_returns_a_real_detection(self, device, dtype):
        # With the default configuration and `num_features=1` every proportional quota truncates
        # to zero (the shares are 0.508 .. 0.016), so every level was queried for zero candidates
        # and the result was a padded dummy on an image full of maxima.
        torch.manual_seed(11)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        lafs_one, resps_one = self._make_detector(num_features=1).to(device, dtype)(inp)
        lafs_two, resps_two = self._make_detector(num_features=2).to(device, dtype)(inp)

        assert resps_one.shape == torch.Size([1, 1])
        assert resps_one[0, 0] > 0
        # Asking for one feature must return the same best feature as asking for two.
        self.assert_close(resps_one[0, 0], resps_two[0].max())
        self.assert_close(lafs_one[0, 0], lafs_two[0, int(resps_two[0].argmax())])

    def test_negative_score_threshold_is_rejected(self, device, dtype):
        # NMS writes an exact zero at every suppressed position, so a negative threshold would
        # admit all of them -- and collide with the zero response that marks an unfilled slot.
        with pytest.raises(ValueError):
            MultiResolutionDetector(kornia.feature.BlobHessian(), num_features=10, score_threshold=-1.0)

    def test_detect_rejects_a_batch(self, device, dtype):
        # `detect` is public and documents `(1, C, H, W)`; a larger batch would be flattened
        # across the batch axis and silently return coordinates for the wrong image.
        det = self._make_detector(num_features=10).to(device, dtype)
        with pytest.raises(ShapeError):
            det.detect(torch.rand(2, 1, 64, 64, device=device, dtype=dtype))

    def test_smoke_with_blob_image(self, device, dtype):
        # Synthetic image with a bright blob — detector should find it.
        inp = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        inp[:, :, 28:36, 28:36] = 1.0
        det = self._make_detector(num_features=5).to(device, dtype)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, 5, 2, 3])
        assert resps.abs().max().item() > 0
