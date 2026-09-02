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
    _resize_mask,
    get_default_detector_config,
)
from kornia.geometry.subpix import ConvQuadInterp3d

from testing.base import BaseTester, supports_conv2d, supports_replicate_padding, supports_topk


def _require_affine_orientation_kernels(device: torch.device, dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16):
        return
    if device.type == "mps":
        pytest.skip("MPS autocast changes the effective dtype")
    probes = (
        ("replicate-padding", supports_replicate_padding),
        ("conv2d", supports_conv2d),
        ("topk", supports_topk),
    )
    for name, probe in probes:
        if not probe(device, dtype):
            pytest.skip(f"no {name} kernel for {dtype} on {device.type}")


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
        # The scale is `sigma * 2 ** (level / num_levels)` after a sub-pixel refinement; float16
        # resolves it to ~3e-3 and bfloat16, with its 8-bit mantissa, to ~3e-2.
        rtol = {torch.float16: 5e-3, torch.bfloat16: 4e-2}.get(dtype, 1e-3)
        self.assert_close(lafs, expected_laf, rtol=rtol, atol=1e-03)
        self.assert_close(resps, expected_resp, rtol=rtol, atol=1e-03)

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
        # The scale is `sigma * 2 ** (level / num_levels)` after a sub-pixel refinement; float16
        # resolves it to ~3e-3 and bfloat16, with its 8-bit mantissa, to ~3e-2.
        rtol = {torch.float16: 5e-3, torch.bfloat16: 4e-2}.get(dtype, 1e-3)
        self.assert_close(lafs, expected_laf, rtol=rtol, atol=1e-03)
        self.assert_close(resps, expected_resp, rtol=rtol, atol=1e-03)

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
        # A slot that no detection filled -- the padding, or a candidate whose frame reaches
        # outside the image -- has a zero response and a zero LAF, and every real detection sorts
        # ahead of it. On `main` a border-rejected candidate kept its coordinates beside a zero
        # response, i.e. a keypoint that was never detected.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = ScaleSpaceDetector(50).to(device, dtype)
        resps, lafs = det.detect(inp, 50)
        zero_laf = (lafs[0] == 0).all(dim=-1).all(dim=-1)
        empty = resps[0] == 0
        assert bool(zero_laf.any()), "expected this image to under-fill 50 slots"
        assert bool((~zero_laf).any()), "expected this image to yield real detections"
        assert torch.equal(zero_laf, empty)
        n_real = int((~zero_laf).sum())
        assert bool((~zero_laf[:n_real]).all()), "real detections must come first"
        lafs_fwd, resps_fwd = det(inp)
        assert torch.equal(resps_fwd, resps) and torch.equal(lafs_fwd, lafs)

    def test_negative_detections_sort_before_the_padding(self, device, dtype):
        # A signed response function can have every maximum below zero. The padding must still
        # come last: it is ranked with a sentinel and zeroed only afterwards.
        class NegatedHessian(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.resp = kornia.feature.BlobHessian()

            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                return self.resp(x) - 1.0

        if dtype in (torch.float16, torch.bfloat16):
            # The shifted response is flat at half precision and yields no maxima at all.
            pytest.skip("a Hessian response offset by 1.0 has no resolution left in half precision")
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = ScaleSpaceDetector(50, resp_module=NegatedHessian()).to(device, dtype)
        lafs, resps = det(inp)
        filled = lafs[0].ne(0).any(dim=-1).any(dim=-1)
        n_real = int(filled.sum())
        assert 0 < n_real < 50, f"expected a partially filled result, got {n_real}"
        assert bool(filled[:n_real].all()) and not bool(filled[n_real:].any())
        assert (resps[0, :n_real] < 0).all()
        assert (resps[0, n_real:] == 0).all()

    def test_float_mask_never_promotes_a_down_weighted_negative_response(self, device, dtype):
        # A weight scales a score toward the worst, not toward zero: a negative score is divided
        # by the weight, a positive one multiplied. A plain multiply pulled a down-weighted negative
        # score toward zero and ranked it above every full-weight one.
        class SignedSpikes(torch.nn.Module):
            # Four maxima on a -1 floor, all negative: two in the left half, two in the right.
            spikes = {(24, 24): -0.20, (24, 72): -0.25, (72, 24): -0.40, (72, 72): -0.30}

            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                out = -torch.ones_like(x)
                if out.shape[-1] == 96:  # the full-resolution octave only; the coarser ones stay flat
                    for (y, xx), v in self.spikes.items():
                        out[:, :, out.shape[2] // 2, y, xx] = v
                return out

        det = ScaleSpaceDetector(4, resp_module=SignedSpikes(), scale_space_response=True, mr_size=3.0).to(
            device, dtype
        )
        img = torch.zeros(1, 1, 96, 96, device=device, dtype=dtype)
        lafs_plain, resps_plain = det(img)
        assert bool(lafs_plain.ne(0).any(dim=-1).any(dim=-1).all()), "expected all four spikes"
        assert resps_plain[0].tolist() == sorted(resps_plain[0].tolist(), reverse=True)
        mask = torch.ones(1, 1, 96, 96, device=device, dtype=dtype)
        mask[..., 48:] = 0.5  # the right half is down-weighted
        lafs, resps = det(img, mask)
        xs = lafs[0, :, 0, 2]
        right = xs >= 48
        # Same four frames; the full-weight (left, x=24) ones keep their score, the down-weighted
        # (right, x=72) ones get worse, and the down-weighted best (-0.25) no longer ranks above the
        # full-weight runner-up (-0.40) -- a multiply would have reported it as -0.125 and put it first.
        assert int(right.sum()) == 2
        self.assert_close(resps[0][~right], torch.tensor([-0.20, -0.40], device=device, dtype=dtype))
        self.assert_close(resps[0][right], torch.tensor([-0.50, -0.60], device=device, dtype=dtype))
        assert resps[0].tolist() == sorted(resps[0].tolist(), reverse=True)
        assert xs[0] < 48 and xs[1] < 48, f"a down-weighted candidate outranked a full-weight one: {xs.tolist()}"

    def test_a_weighted_negative_response_never_sorts_below_the_padding(self, device, dtype):
        # A weight *divides* a negative score, and a small weight can push the quotient past the
        # dtype's range: -inf ties with the `-inf` an unfilled slot carries into the ranking, so a
        # non-detection could outrank a real one. The weighted score is kept finite instead.
        floor = -float(torch.finfo(dtype).max) / 4
        spike = -float(torch.finfo(dtype).max) / 8

        class HugeNegativeSpikes(torch.nn.Module):
            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                out = torch.full_like(x, floor)
                if out.shape[-1] == 96:
                    for y, xx in ((24, 24), (24, 72), (72, 24), (72, 72)):
                        out[:, :, out.shape[2] // 2, y, xx] = spike
                return out

        det = ScaleSpaceDetector(4, resp_module=HugeNegativeSpikes(), scale_space_response=True, mr_size=3.0).to(
            device, dtype
        )
        img = torch.zeros(1, 1, 96, 96, device=device, dtype=dtype)
        mask = torch.full((1, 1, 96, 96), 1e-3, device=device, dtype=dtype)
        lafs, resps = det(img, mask)
        filled = lafs.ne(0).any(dim=-1).any(dim=-1)
        assert int(filled.sum()) == 4, "expected all four spikes"
        assert torch.isfinite(resps).all(), f"a weighted score overflowed: {resps.tolist()}"

    def test_an_extreme_finite_response_still_outranks_the_padding(self, device, dtype):
        # `finfo.min / 2` is not below every finite response: four maxima at `-2e38` (float32) on a
        # `finfo.min` floor are real detections, and the single-image path returned all four while
        # the batched top-K preferred the sentinel and returned none. Padding is ranked with `-inf`,
        # and a finite response, however extreme, sorts ahead of it on both paths.
        floor = float(torch.finfo(dtype).min)
        spike = -float(torch.finfo(dtype).max) * 0.6

        class ExtremeSpikes(torch.nn.Module):
            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                out = torch.full_like(x, floor)
                if out.shape[-1] == 96:
                    for y, xx in ((24, 24), (24, 72), (72, 24), (72, 72)):
                        out[:, :, out.shape[2] // 2, y, xx] = spike
                return out

        det = ScaleSpaceDetector(4, resp_module=ExtremeSpikes(), scale_space_response=True, mr_size=3.0).to(
            device, dtype
        )
        img = torch.zeros(2, 1, 96, 96, device=device, dtype=dtype)
        lafs1, resps1 = det(img[:1])
        lafs2, resps2 = det(img)
        filled1 = lafs1.ne(0).any(dim=-1).any(dim=-1)
        filled2 = lafs2.ne(0).any(dim=-1).any(dim=-1)
        assert int(filled1.sum()) == 4, f"single-image path lost a detection: {resps1.tolist()}"
        assert int(filled2[0].sum()) == 4 and int(filled2[1].sum()) == 4, f"batched path lost one: {resps2.tolist()}"
        assert torch.isfinite(resps2).all()
        self.assert_close(resps2[0], resps1[0])

    def test_subpixel_refinement_cannot_cross_the_mask(self, device, dtype):
        # The mask is checked at the integer NMS site, but the sub-pixel step moves the centre. A
        # parabola peaked at x = 31.6 has its discrete maximum at x = 32; a mask allowing only
        # x >= 32 passed that site and returned a centre of 31.6, inside the zero region. The refined
        # centre is re-checked against the resampled mask conservatively (every pixel it touches).
        class Parabola(torch.nn.Module):
            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                B, L, H, W = x.shape[0], x.shape[2], x.shape[-2], x.shape[-1]
                xs = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, 1, W)
                ys = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, 1, H, 1)
                ls = torch.arange(L, device=x.device, dtype=x.dtype).view(1, 1, L, 1, 1)
                return (10.0 - (xs - 31.6) ** 2 - (ys - 48.0) ** 2 - (ls - L // 2) ** 2).expand(B, 1, L, H, W).clone()

        det = ScaleSpaceDetector(1, resp_module=Parabola(), scale_space_response=True, mr_size=1.0).to(device, dtype)
        img = torch.zeros(1, 1, 96, 96, device=device, dtype=dtype)
        mask = torch.zeros(1, 1, 96, 96, device=device, dtype=torch.bool)
        mask[..., 32:] = True
        lafs, resps = det(img, mask)
        assert bool((lafs == 0).all()) and bool((resps == 0).all()), (
            f"a refined centre at {lafs[0, 0, :, 2].tolist()} lies in the masked-out region"
        )
        # Control: the same peak is returned, refined, when the mask allows the pixels it touches.
        mask[..., 31:] = True
        lafs, resps = det(img, mask)
        assert bool(resps[0, 0] > 0)
        self.assert_close(lafs[0, 0, :, 2], torch.tensor([31.6, 48.0], device=device, dtype=dtype), atol=0.2, rtol=0)

    def test_fill_sentinel_follows_the_response_dtype(self, device, dtype):
        # The top-K sentinel is written into the *response* tensor, so it must fit that dtype: a
        # response module that emits a narrower dtype than the image -- a learned response under
        # autocast -- raised "value cannot be converted to type at::Half without overflow".
        if not supports_topk(device, torch.float16):
            # The top-K ranks the response, which this module narrows to float16; torch 2.1.2 has
            # no float16 CPU `topk` kernel.
            pytest.skip(f"no topk kernel for torch.float16 on {device.type}")

        class NarrowResponse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.resp = kornia.feature.BlobHessian()

            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                return self.resp(x, sigmas).to(torch.float16)

        torch.manual_seed(0)
        inp = torch.rand(2, 1, 64, 64, device=device, dtype=dtype)
        det = ScaleSpaceDetector(20, resp_module=NarrowResponse()).to(device, dtype)
        for b in (1, 2):
            lafs, resps = det(inp[:b])
            assert resps.dtype == torch.float16
            assert torch.isfinite(resps).all()
            filled = lafs.ne(0).any(dim=-1).any(dim=-1)
            assert 0 < int(filled[0].sum()) < 20
            assert bool((resps[~filled] == 0).all())

    def test_float_weights_above_one_are_clamped(self, device, dtype):
        # A float 0/255 mask -- an OpenCV mask after `.astype(np.float32)` -- is weights, and a weight
        # above one scaled every score by 255. Weights are clamped to one, so it means what the
        # integer 0/255 mask means.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = ScaleSpaceDetector(50).to(device, dtype)
        lafs_b, resps_b = det(inp, torch.ones(1, 1, 64, 64, device=device, dtype=torch.bool))
        lafs_f, resps_f = det(inp, torch.full((1, 1, 64, 64), 255.0, device=device, dtype=dtype))
        assert torch.equal(resps_f, resps_b) and torch.equal(lafs_f, lafs_b)

    def test_a_zero_response_maximum_keeps_its_laf(self, device, dtype):
        # The response function is pluggable and may be signed, so an exact zero can be a genuine
        # maximum rather than an unfilled slot. Classifying the padding by `response == 0` erased
        # all three detections here.
        class SignedResponse(torch.nn.Module):
            def forward(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
                out = -torch.ones_like(x)
                out[:, :, out.shape[2] // 2, out.shape[-2] // 2, out.shape[-1] // 2] = 0.0
                return out

        # `mr_size=3` keeps the coarsest octave's frame inside the 96 px image; at the default 6 it
        # is 76.8 px wide, and a candidate whose frame leaves the image is not a detection.
        det = ScaleSpaceDetector(5, resp_module=SignedResponse(), scale_space_response=True, mr_size=3.0).to(
            device, dtype
        )
        lafs, resps = det(torch.zeros(1, 1, 96, 96, device=device, dtype=dtype))
        assert (resps == 0).all()
        centers = lafs[0, :, :, 2]
        found = (centers == 48).all(dim=-1)
        assert int(found.sum()) == 3, f"expected the three centre maxima, got {centers.tolist()}"

    def test_batched_underfill_does_not_leak_the_topk_sentinel(self, device, dtype):
        # For B > 1 the octave top-K ranks over the whole volume with non-candidates masked to
        # `finfo.min / 2`. An image with fewer maxima than requested gets those sentinels back;
        # they are not detections and must not reach the caller as a response or as a LAF.
        torch.manual_seed(0)
        inp = torch.rand(2, 1, 64, 64, device=device, dtype=dtype)
        det = ScaleSpaceDetector().to(device, dtype)
        lafs, resps = det(inp)
        filled = lafs.ne(0).any(dim=-1).any(dim=-1)
        assert 0 < int(filled[0].sum()) < det.num_features, "expected a partially filled result"
        assert bool((resps[~filled] == 0).all()), f"sentinel leaked: min response {resps.min().item()}"
        assert (resps > torch.finfo(dtype).min / 4).all()
        # and the single-image path agrees, which it did not before. Up to tolerance, not bit-for-bit:
        # the response convolutions pick batch-size-dependent kernels on some CPUs (ubuntu CI), so the
        # same candidates carry last-ULP-different scores.
        lafs1, resps1 = det(inp[:1])
        self.assert_close(resps1[0], resps[0])
        self.assert_close(lafs1[0], lafs[0])

    def test_padding_survives_the_affine_and_orientation_modules(self, device, dtype):
        # Same contract as `MultiResolutionDetector.forward`: the shape and orientation modules
        # may transform a zero frame differently by dtype, so the padding is re-applied after them.
        _require_affine_orientation_kernels(device, dtype)
        det = ScaleSpaceDetector(
            10, aff_module=kornia.feature.LAFAffineShapeEstimator(19), ori_module=kornia.feature.LAFOrienter(19)
        ).to(device, dtype)
        lafs, resps = det(torch.zeros(1, 1, 64, 64, device=device, dtype=dtype))
        assert (resps == 0).all()
        assert (lafs == 0).all()

    def test_forward_uses_overridden_detect(self, device, dtype):
        class CustomDetector(ScaleSpaceDetector):
            def detect(self, img, num_feats, mask=None):
                responses = img.new_full((img.shape[0], num_feats), 123.0)
                lafs = img.new_full((img.shape[0], num_feats, 2, 3), 7.0)
                return responses, lafs

        inp = torch.zeros(2, 1, 32, 32, device=device, dtype=dtype)
        det = CustomDetector(3).to(device, dtype)
        lafs, responses = det(inp)
        assert torch.equal(responses, inp.new_full((2, 3), 123.0))
        assert torch.equal(lafs, inp.new_full((2, 3, 2, 3), 7.0))

    @pytest.mark.parametrize("subpix", ["adaptive", "conv"])
    def test_color_input_with_single_channel_response(self, device, dtype, subpix):
        class ColorResponse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.response = kornia.feature.BlobHessian()

            def forward(self, x: torch.Tensor, _sigmas: torch.Tensor) -> torch.Tensor:
                return self.response(x.mean(dim=1, keepdim=True))

        torch.manual_seed(3)
        inp = torch.rand(1, 3, 96, 96, device=device, dtype=dtype)
        kwargs = {} if subpix == "adaptive" else {"subpix_module": ConvQuadInterp3d(10)}
        det = ScaleSpaceDetector(20, resp_module=ColorResponse(), **kwargs).to(device, dtype)
        lafs, responses = det(inp)
        assert lafs.shape == torch.Size([1, 20, 2, 3])
        assert responses.shape == torch.Size([1, 20])
        valid = lafs.ne(0).any(dim=-1).any(dim=-1)
        centers = lafs[..., 2][valid]
        assert bool(valid.any())
        assert (centers[:, 0] >= 0).all() and (centers[:, 0] <= 95).all()
        assert (centers[:, 1] >= 0).all() and (centers[:, 1] <= 95).all()

    def test_multichannel_response_is_rejected(self, device, dtype):
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        detector = ScaleSpaceDetector(20, resp_module=kornia.feature.BlobHessian()).to(device, dtype)
        with pytest.raises(Exception, match="one response map"):
            detector(inp)

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
        # The mask is resampled conservatively onto every level, so nothing is detected on the
        # zero side of its edge at any level.
        assert (lafs_masked[0][found][:, :, 2] >= 32).all()

    def test_mask_edge_does_not_create_maxima(self, device, dtype):
        # A blob wholly inside the zero region, next to the mask edge. Multiplying the response by
        # a resampled mask before non-maxima suppression carved an edge into it, and the bilinear
        # ramp of that edge was a "maximum" on the suppressed side (kornia#4102).
        inp = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        inp[:, :, 30:34, 28:32] = 1.0
        mask = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        mask[:, :, :, 32:] = 1.0
        det = self._make_detector(num_features=5).to(device, dtype)
        lafs, resps = det(inp, mask)
        assert (resps == 0).all(), f"detected on the masked side: {lafs[0][resps[0] != 0][:, 0, 2].tolist()}"
        assert (lafs == 0).all()

    def test_thin_masked_stripe_survives_downsampling(self, device, dtype):
        # A two-pixel zero stripe is narrower than the sampling step of a coarse level, so an
        # interpolated mask reads 1.0 there and the stripe is gone; the conservative resample keeps
        # every level pixel it touches at zero.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 256, 256, device=device, dtype=dtype)
        mask = torch.ones(1, 1, 256, 256, device=device, dtype=dtype)
        mask[:, :, :, 120:122] = 0.0
        det = self._make_detector(num_features=2000).to(device, dtype)
        lafs, resps = det(inp, mask)
        xs = lafs[0][resps[0] != 0][:, 0, 2]
        assert xs.numel() > 100
        assert not bool(((xs >= 120) & (xs < 122)).any()), sorted(xs[(xs >= 119) & (xs < 123)].tolist())

    @pytest.mark.parametrize("src, dst", [((100, 100), (45, 45)), ((38, 38), (90, 90)), ((7, 9), (3, 4))])
    def test_mask_resample_is_the_cpu_adaptive_min_pool_on_every_device(self, device, dtype, src, dst):
        # `F.adaptive_max_pool2d` on MPS returns the wrong shape when the output is larger than the
        # input (the KeyNet up-level, the double-image octave) and pools other windows than CPU for
        # a non-integer ratio, so the resampled mask would depend on the device. The hand-rolled
        # resample pins the CPU windows on every device.
        torch.manual_seed(0)
        mask = torch.rand(2, 1, *src, device=device, dtype=dtype)
        ref = torch.empty(1, 1, *dst, device=device, dtype=dtype)
        expected = -torch.nn.functional.adaptive_max_pool2d(-mask.cpu().float(), dst)
        got = _resize_mask(mask, ref)
        assert got.shape == (2, 1, *dst)
        assert got.dtype == dtype
        self.assert_close(got.cpu().float(), expected.to(got.cpu().float().dtype))

    def test_float_weights_above_one_are_clamped(self, device, dtype):
        # The weighting runs before the `score_threshold` test, so a weight above one lifted
        # sub-threshold maxima over it. A float 0/255 mask now means what the integer one means.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = self._make_detector().to(device, dtype)
        lafs_b, resps_b = det(inp, torch.ones(1, 1, 64, 64, device=device, dtype=torch.bool))
        lafs_f, resps_f = det(inp, torch.full((1, 1, 64, 64), 255.0, device=device, dtype=dtype))
        assert torch.equal(resps_f, resps_b) and torch.equal(lafs_f, lafs_b)

    def test_float_mask_weights_the_score_and_keeps_the_candidates(self, device, dtype):
        # A graded mask used to be multiplied into the response before the NMS and the sub-pixel
        # step, which moved maxima and their refined positions. It now weights only the score of a
        # maximum found in the unweighted response: same detections, same positions, ranked by weight.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = self._make_detector(num_features=2000).to(device, dtype)
        lafs_plain, resps_plain = det(inp)
        ramp = torch.linspace(0.2, 1.0, 64, device=device, dtype=dtype).view(1, 1, 1, 64).expand(1, 1, 64, 64)
        lafs, resps = det(inp, ramp.contiguous())
        keep_plain, keep = resps_plain[0] != 0, resps[0] != 0
        assert int(keep.sum()) == int(keep_plain.sum()) > 20
        # Same set of frames, up to the order the weighted score imposes.

        def order(t: torch.Tensor) -> torch.Tensor:
            key = t[:, 0, 2].cpu().double() * 1000 + t[:, 1, 2].cpu().double()
            return t[torch.sort(key).indices.to(t.device)]

        self.assert_close(order(lafs[0][keep]), order(lafs_plain[0][keep_plain]))
        ratio = resps[0][keep] / resps_plain[0][keep]
        assert float(ratio.min()) >= 0.2 - 1e-2 and float(ratio.max()) <= 1.0 + 1e-2
        assert not torch.allclose(resps[0][keep], resps_plain[0][keep_plain])

    def test_response_map_must_match_the_level_size(self, device, dtype):
        # A response index is decoded as a level pixel with no offset, so a valid-convolution net
        # that returns a smaller map put every keypoint one pixel up and left of its peak and had
        # the mask resampled onto the shifted geometry. Nothing can infer the offset from the shape,
        # so a map of another size is rejected rather than silently misplaced.
        kernel = torch.ones(1, 1, 3, 3, device=device, dtype=dtype) / 9
        det = MultiResolutionDetector(lambda x: torch.nn.functional.conv2d(x, kernel), num_features=8).to(device, dtype)
        with pytest.raises(Exception, match=r"spatial size"):
            det(torch.rand(1, 1, 64, 64, device=device, dtype=dtype))

    def test_a_nan_weight_does_not_consume_a_slot(self, device, dtype):
        # `s * NaN` sorts first in `topk`, so one NaN pixel in a float mask took a slot on every
        # pyramid level and displaced a real maximum (18 filled slots became 13); `ScaleSpaceDetector`
        # gates its candidates with `weight > 0`, which drops NaN, and this detector does the same.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = self._make_detector(num_features=20).to(device, dtype)
        lafs_ref, _ = det(inp)
        mask = torch.ones(1, 1, 64, 64, device=device, dtype=dtype)
        mask[0, 0, 30, 30] = float("nan")
        lafs, resps = det(inp, mask)
        filled_ref = lafs_ref.ne(0).any(dim=-1).any(dim=-1)
        filled = lafs.ne(0).any(dim=-1).any(dim=-1)
        assert torch.isfinite(resps).all()
        # the NaN pixel suppresses its neighbourhood on every level, nothing more
        assert int(filled_ref.sum()) - 4 <= int(filled.sum()) <= int(filled_ref.sum())

    def test_mask_spatial_size_must_match_the_image(self, device, dtype):
        # `KORNIA_CHECK_SHAPE`'s named dims are free per call, so on their own they let a mask
        # of any size through to be stretched onto the image; the explicit check is what rejects it.
        det = self._make_detector(num_features=10).to(device, dtype)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        with pytest.raises(Exception, match="spatial size"):
            det(inp, torch.ones(1, 1, 32, 32, device=device, dtype=dtype))
        with pytest.raises(Exception, match="spatial size"):
            det(inp, torch.ones(1, 1, 64, 32, device=device, dtype=dtype))

    @pytest.mark.parametrize("mask_kind", ["bool", "uint8_255", "int32_100"])
    def test_binary_masks_match_float_mask(self, device, dtype, mask_kind):
        # A boolean or integer mask is binary: any non-zero value keeps a position, so an OpenCV
        # 0/255 mask means the same as a 0/1 one and does not scale the responses by 255.
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        mask = torch.zeros(1, 1, 64, 64, device=device, dtype=torch.bool)
        mask[:, :, 32:, 32:] = True
        if mask_kind == "uint8_255":
            mask = mask.to(torch.uint8) * 255
        elif mask_kind == "int32_100":
            mask = mask.to(torch.int32) * 100
        det = self._make_detector(num_features=10).to(device, dtype)
        lafs_bool, resps_bool = det(inp, mask)
        lafs_float, resps_float = det(inp, mask.ne(0).to(dtype))
        assert bool((resps_bool != 0).any())
        assert torch.equal(resps_bool, resps_float)
        assert torch.equal(lafs_bool, lafs_float)
        # ... and the mask is doing something, so the two arms cannot agree vacuously the way
        # they would if the mask were ignored altogether.
        _lafs_free, resps_free = det(inp)
        assert not torch.equal(resps_bool, resps_free)

    def test_padding_survives_the_affine_and_orientation_modules(self, device, dtype):
        # `forward` runs `aff` and `ori` after `detect`; they may transform a zero frame differently
        # by dtype, so the zero-LAF contract has to be re-applied.
        _require_affine_orientation_kernels(device, dtype)
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
        if not supports_replicate_padding(device, half_dtype):
            pytest.skip(f"no replicate-padding kernel for {half_dtype} on {device.type}")

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

    def test_color_input_with_single_channel_response(self, device, dtype):
        class ColorResponse(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.mean(dim=1, keepdim=True)

        torch.manual_seed(3)
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        det = MultiResolutionDetector(ColorResponse(), num_features=100).to(device, dtype)
        lafs, resps = det(inp)
        found = resps[0] != 0
        assert bool(found.any())
        cx = lafs[0, :, 0, 2]
        cy = lafs[0, :, 1, 2]
        assert (cx >= 0).all() and (cx <= 63).all()
        assert (cy >= 0).all() and (cy <= 63).all()

    def test_multichannel_response_is_rejected(self, device, dtype):
        inp = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        detector = self._make_detector(num_features=100).to(device, dtype)
        with pytest.raises(Exception, match="one response map"):
            detector(inp)

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

    def test_small_request_spreads_across_scales(self, device, dtype):
        # `detect` used to apportion `num_features` across pyramid levels by flooring each level's
        # fractional share independently (`int(x) for x in shares`). The shares favor the finest
        # level so heavily (0.508 .. 0.016 with the default config) that a small `num_features`
        # truncated every other level's quota to zero: five well-separated, equally strong blobs and
        # `num_features=3` returned three near-duplicate detections of the single strongest blob
        # instead of covering several of them. Largest-remainder (Hamilton) apportionment hands the
        # shortfall to the levels with the largest fractional remainder instead, so a small request
        # still reaches more than one scale and covers more than one blob.
        img = torch.zeros(1, 1, 96, 96, device=device, dtype=dtype)
        centers = [(20, 20), (20, 70), (48, 48), (76, 20), (76, 70)]
        for y, x in centers:
            img[:, :, y - 2 : y + 3, x - 2 : x + 3] = 1.0

        det = self._make_detector(num_features=3).to(device, dtype)
        _, lafs = det.detect(img)
        laf_centers = lafs[0, :, :, 2]
        covered = sum(
            1 for y, x in centers if bool(((laf_centers - laf_centers.new_tensor([x, y])).abs().amax(dim=1) < 4).any())
        )
        assert covered >= 2, f"expected the 3-feature request to reach at least 2 of the 5 blobs, got {covered}"

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

    def test_forward_uses_overridden_detect(self, device, dtype):
        # `detect` is the public extension point; `forward` reads occupancy off its zero-LAF
        # padding rather than off the response, so an override is honoured whole.
        class CustomDetector(MultiResolutionDetector):
            def detect(self, img, mask=None):
                responses = img.new_full((1, self.num_features), 123.0)
                lafs = img.new_full((1, self.num_features, 2, 3), 7.0)
                return responses, lafs

        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        det = CustomDetector(kornia.feature.BlobHessian(), num_features=3).to(device, dtype)
        lafs, responses = det(inp)
        assert torch.equal(responses, inp.new_full((1, 3), 123.0))
        assert torch.equal(lafs, inp.new_full((1, 3, 2, 3), 7.0))

    def test_three_argument_level_override_still_runs_unmasked(self, device, dtype):
        # `detect_features_on_single_level` is public and gained a keyword-only `mask`; a subclass
        # written against the historical three-argument signature keeps working without a mask
        # and gets a TypeError, not silent misbehaviour, with one.
        class LegacyLevel(MultiResolutionDetector):
            def detect_features_on_single_level(self, level_img, num_kp, factor):
                return super().detect_features_on_single_level(level_img, num_kp, factor)

        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        det = LegacyLevel(kornia.feature.BlobHessian(), num_features=10).to(device, dtype)
        ref = self._make_detector(num_features=10).to(device, dtype)
        lafs, resps = det(inp)
        lafs_ref, resps_ref = ref(inp)
        assert torch.equal(lafs, lafs_ref) and torch.equal(resps, resps_ref)
        with pytest.raises(TypeError):
            det(inp, torch.ones(1, 1, 64, 64, device=device, dtype=dtype))

    def test_smoke_with_blob_image(self, device, dtype):
        # Synthetic image with a bright blob — detector should find it.
        inp = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        inp[:, :, 28:36, 28:36] = 1.0
        det = self._make_detector(num_features=5).to(device, dtype)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, 5, 2, 3])
        assert resps.abs().max().item() > 0
