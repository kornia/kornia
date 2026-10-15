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

from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch import nn

import kornia
from kornia.feature import LocalFeatureMatcher

from testing.base import BaseTester


def _render(x0: float, w: int, h: int = 48, dtype: torch.dtype = torch.float32) -> torch.Tensor:  # float32: MPS too
    ys, xs = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype) + x0, indexing="ij")
    return torch.stack(
        [
            0.5 + 0.2 * torch.sin(0.21 * xs + 0.13 * ys + c) + 0.15 * torch.cos(0.17 * ys - 0.07 * xs + 2 * c)
            for c in range(3)
        ]
    )[None]


class _ShiftMatcher(LocalFeatureMatcher):
    """Exact correspondences: right pixel p is left pixel p + (shift_b, 0); samples in `empty` get none.

    The right image is sampled every 5 px along x and every `ystep` px along y.
    """

    def __init__(self, shifts, empty=(), float32_keypoints=False, ystep=5.0):
        nn.Module.__init__(self)
        self.shifts, self.empty, self.float32_keypoints, self.ystep = shifts, set(empty), float32_keypoints, ystep

    def forward(self, data):
        im0, im1 = data["image0"], data["image1"]
        kp0, kp1, idx = [torch.empty(0, 2)], [torch.empty(0, 2)], [torch.empty(0, dtype=torch.long)]
        for b in range(im1.shape[0]):
            if b in self.empty:
                continue
            ys, xs = torch.meshgrid(
                torch.arange(2, im1.shape[-2] - 2, self.ystep), torch.arange(2, im1.shape[-1] - 2, 5.0), indexing="ij"
            )
            p1 = torch.stack([xs.flatten(), ys.flatten()], -1)
            p0 = p1 + torch.tensor([self.shifts[b], 0.0])
            keep = (p0[:, 0] >= 0) & (p0[:, 0] <= im0.shape[-1] - 1)
            kp0.append(p0[keep])
            kp1.append(p1[keep])
            idx.append(torch.full((int(keep.sum()),), b))
        return {
            "keypoints0": torch.cat(kp0).to(im0.device if self.float32_keypoints else im0),
            "keypoints1": torch.cat(kp1).to(im0.device if self.float32_keypoints else im0),
            "batch_indexes": torch.cat(idx),
        }


class TestImageStitcher(BaseTester):
    @pytest.mark.parametrize("img_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shift", [40.0, 40.5, 40.7])
    def test_subpixel_shift_has_no_dark_seam(self, shift, img_dtype, device):
        # C-12: the mask was warped with nearest and the image bilinearly: s = 40.5 left a dark column (error 0.43),
        # s = 40.7 an extra dark trailing column (width 97, truth 96). The mask and the image share one warp, so they
        # share the sampling grid in half precision too (on MPS a half image keeps a half grid).
        torch.manual_seed(0)
        left, right = _render(0, 64).to(device, img_dtype), _render(shift, 56).to(device, img_dtype)
        with torch.no_grad():
            out = kornia.contrib.ImageStitcher(_ShiftMatcher([shift]))(left, right)
        assert out.shape[-1] == int(shift) + 56
        truth = _render(0, out.shape[-1]).to(device)
        # bilinear resampling of the smooth texture, no black mixed in and no covered pixel dropped: a dark or dropped
        # pixel is off by 0.3 or more for these shifts; the bound is interpolation error plus the bf16 keypoint rounding
        tol = {torch.float32: 0.02, torch.float16: 0.02, torch.bfloat16: 0.04}[img_dtype]
        assert (out.float() - truth).abs().max().item() < tol

    @pytest.mark.parametrize("img_dtype", [torch.float16, torch.bfloat16])
    def test_float32_homography_with_half_image(self, img_dtype, device):
        # The warp runs in float32, so a float32 homography (exact float32 keypoints) works with a half image.
        left, right = _render(0, 64).to(device, img_dtype), _render(40.5, 56).to(device, img_dtype)
        matcher = _ShiftMatcher([40.5], float32_keypoints=True)
        with torch.no_grad():
            out = kornia.contrib.ImageStitcher(matcher)(left, right)
        assert out.dtype == img_dtype
        tol = {torch.float16: 0.02, torch.bfloat16: 0.04}[img_dtype]
        assert (out.float() - _render(0, out.shape[-1]).to(device)).abs().max().item() < tol

    @pytest.mark.parametrize("img_dtype", [torch.float32, torch.float16])
    def test_image_and_mask_share_one_warp(self, img_dtype, device):
        # I2: on MPS a half image is sampled on a half grid; a mask warped in a separate (float32) call lands on another
        # grid and marks pixels the image samples outside of as covered. One call keeps both on the same grid.
        left, right = _render(0, 64).to(device, img_dtype), _render(40.5, 56).to(device, img_dtype)
        stitcher = kornia.contrib.ImageStitcher(_ShiftMatcher([40.5]))
        with (
            patch(
                "kornia.contrib.image_stitching.warp_perspective", wraps=kornia.geometry.transform.warp_perspective
            ) as w,
            torch.no_grad(),
        ):
            stitcher.stitch_pair(left, right)
        assert w.call_count == 1
        assert w.call_args.args[0].shape[1] == 6  # three image channels and three mask channels
        assert w.call_args.args[0].dtype == torch.float32  # also for a half image

    def test_batch_of_two_equals_each_sample_alone(self, device):
        left = torch.cat([_render(0, 64), _render(3, 64)]).to(device)
        right = torch.cat([_render(40, 56), _render(31.5, 56)]).to(device)
        stitcher = kornia.contrib.ImageStitcher(_ShiftMatcher([40.0, 28.5]))
        with torch.no_grad():
            pair = stitcher.stitch_pair(left, right)[0]
            for b in range(2):
                alone = kornia.contrib.ImageStitcher(_ShiftMatcher([[40.0, 28.5][b]])).stitch_pair(
                    left[b : b + 1], right[b : b + 1]
                )[0]
                self.assert_close(pair[b : b + 1], alone)

    def test_estimate_transform_names_a_gap_without_batch_size(self):
        # Samples 0 and 2 matched, sample 1 not: the sample count comes from the largest batch index.
        stitcher = kornia.contrib.ImageStitcher(_ShiftMatcher([0.0]))
        kp = torch.rand(8, 2)
        idx = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        with pytest.raises(RuntimeError, match="sample 1"):
            stitcher.estimate_transform(keypoints0=kp, keypoints1=kp, batch_indexes=idx)

    def test_unmatched_sample_is_named(self, device):
        # C-15: one unmatched sample failed the batch with grid_sampler's batch-size error.
        left = torch.cat([_render(0, 64)] * 2).to(device)
        right = torch.cat([_render(40, 56), _render(30, 56)]).to(device)
        with pytest.raises(RuntimeError, match="sample 1"):
            kornia.contrib.ImageStitcher(_ShiftMatcher([40.0, 30.0], empty=(1,)))(left, right)

    def test_every_sample_unmatched_names_sample_0(self, device):
        # With the batch size known, an all-empty match set is reported per sample, like a single unmatched one.
        left = torch.cat([_render(0, 64)] * 2).to(device)
        right = torch.cat([_render(40, 56), _render(30, 56)]).to(device)
        with pytest.raises(RuntimeError, match="sample 0: 0 matches"):
            kornia.contrib.ImageStitcher(_ShiftMatcher([40.0, 30.0], empty=(0, 1)))(left, right)

    def test_no_matches_without_batch_size(self):
        # Without a batch size there is no sample to name.
        stitcher = kornia.contrib.ImageStitcher(_ShiftMatcher([0.0]))
        empty = torch.empty(0, 2)
        with pytest.raises(RuntimeError, match="No matched keypoints found"):
            stitcher.estimate_transform(keypoints0=empty, keypoints1=empty, batch_indexes=torch.empty(0).long())

    @pytest.mark.parametrize(
        "idx, batch_size, message",
        [
            ([0, 1, 2], 2, "batch index 2 is outside the batch of 2 samples"),
            ([0, 1], 1, "batch index 1 is outside the batch of 1 sample\\."),
            ([0, 1, -1], 2, "batch index -1 is negative"),
            ([-2], None, "batch index -2 is negative"),
        ],
    )
    def test_batch_index_outside_the_batch_is_named(self, idx, batch_size, message):
        # Matches of a batch index outside [0, batch_size) were silently left out.
        stitcher = kornia.contrib.ImageStitcher(_ShiftMatcher([0.0]))
        idx = torch.tensor(idx).repeat_interleave(4)
        kp = torch.rand(len(idx), 2) * 50
        with pytest.raises(RuntimeError, match=message):
            stitcher.estimate_transform(batch_size=batch_size, keypoints0=kp, keypoints1=kp, batch_indexes=idx)

    def test_integer_shift_at_large_coordinates_keeps_the_border(self, device):
        # For integer shifts the warp's float32 sampling positions are off by up to ~1.5 * eps * max(H_out, W_out) px
        # (general homographies up to ~2.9), so the coverage of a completely covered border pixel reads below 1 - 1e-4
        # on a wide panorama. Here 8 x 4096 images shifted by 3 px: the right image's last column read 0.99951 and was
        # dropped (the panorama was one column short).
        left, right = _render(0, 4096, h=8).to(device), _render(3, 4096, h=8).to(device)
        with torch.no_grad():
            out = kornia.contrib.ImageStitcher(_ShiftMatcher([3.0], ystep=1.0))(left, right)
        assert out.shape[-1] == 3 + 4096
        assert (out - _render(0, 3 + 4096, h=8).to(device)).abs().max().item() < 0.02

    def test_coverage_tolerance_scales_with_the_panorama(self, device):
        # The tolerance sits between the warp's round-off and a genuinely partial pixel. On this 8 x 8192 panorama a
        # coverage 3 * eps * 8192 below 1 is kept (above the worst round-off measured: 2.86 * eps * max(H_out, W_out)
        # for general homographies, 1.5 for integer shifts, on canvases too large for a test) and 0.99 is dropped.
        left, right = _render(0, 4096, h=8).to(device), _render(3, 4096, h=8).to(device)
        real_warp = kornia.geometry.transform.warp_perspective

        def warp(*args, **kwargs):
            out = real_warp(*args, **kwargs)
            out[:, 3:, 0, 3 + 4095] = 1 - 3 * torch.finfo(torch.float32).eps * 8192
            out[:, 3:, 1, 3 + 4095] = 0.99
            return out

        with patch("kornia.contrib.image_stitching.warp_perspective", warp), torch.no_grad():
            _, mask = kornia.contrib.ImageStitcher(_ShiftMatcher([3.0], ystep=1.0)).stitch_pair(left, right)
        assert mask[0, 0, 0, 3 + 4095].item() == 1
        assert mask[0, 0, 1, 3 + 4095].item() == 0

    @pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
    def test_float32_warp_under_autocast_keeps_the_border(self, autocast_dtype, device):
        # Codex review 5529721894: under autocast the warp's homography matmuls ran in half precision, the coverage of
        # completely covered border pixels fell below the float32 tolerance, and they were dropped: black pixels in
        # the new columns and the panorama one column short (66, truth 67). The conftest skips tests named "autocast"
        # on MPS; the same case failed there and passes with the fix.
        if not torch.amp.is_autocast_available(device.type):
            pytest.skip(f"autocast is not available on {device.type}")
        left, right = torch.ones(1, 3, 48, 64, device=device), torch.ones(1, 3, 48, 64, device=device)
        stitcher = kornia.contrib.ImageStitcher(_ShiftMatcher([3.0]))
        homography = torch.tensor([[[1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device)
        stitcher.estimate_transform = lambda *args, **kwargs: homography  # exact, to isolate the resampling
        with torch.autocast(device_type=device.type, dtype=autocast_dtype), torch.no_grad():
            out = stitcher(left, right)
        assert out.shape[-1] == 64 + 3
        assert out[..., 64:].min().item() > 0.99999

    @pytest.mark.parametrize("estimator", ["ransac", "vanilla"])
    def test_smoke(self, estimator, device, dtype):
        B, C, H, W = 1, 3, 6, 6
        sample1 = torch.tensor(
            [
                [0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251],
                [0.2071, 0.6297, 0.3653, 0.8513, 0.8549, 0.5509],
                [0.2868, 0.2063, 0.4451, 0.3593, 0.7204, 0.0731],
                [0.9699, 0.1078, 0.8829, 0.4132, 0.7572, 0.6948],
                [0.5209, 0.5932, 0.8797, 0.6286, 0.7653, 0.1132],
                [0.8559, 0.6721, 0.6267, 0.5691, 0.7437, 0.9592],
            ],
            dtype=dtype,
            device=device,
        )
        sample2 = torch.tensor(
            [
                [0.3887, 0.2214, 0.3742, 0.1953, 0.7405, 0.2529],
                [0.2332, 0.9314, 0.9575, 0.5575, 0.4134, 0.4355],
                [0.7369, 0.0331, 0.0914, 0.8994, 0.9936, 0.4703],
                [0.1049, 0.5137, 0.2674, 0.4990, 0.7447, 0.7213],
                [0.4414, 0.5550, 0.6361, 0.1081, 0.3305, 0.5196],
                [0.2147, 0.2816, 0.6679, 0.7878, 0.5070, 0.3055],
            ],
            dtype=dtype,
            device=device,
        )
        sample1 = sample1.expand((B, C, H, W))
        sample2 = sample2.expand((B, C, H, W))
        return_value = {
            "keypoints0": torch.tensor(
                [
                    [0.1546, 0.9391],
                    [0.8077, 0.1051],
                    [0.6768, 0.5596],
                    [0.5092, 0.7195],
                    [0.2856, 0.8889],
                    [0.4342, 0.0203],
                    [0.6701, 0.0585],
                    [0.3828, 0.9038],
                    [0.7301, 0.0762],
                    [0.7864, 0.4490],
                    [0.3509, 0.0756],
                    [0.6782, 0.9297],
                    [0.4132, 0.3664],
                    [0.3134, 0.5039],
                    [0.2073, 0.2552],
                ],
                device=device,
                dtype=dtype,
            ),
            "keypoints1": torch.tensor(
                [
                    [0.2076, 0.2669],
                    [0.9679, 0.8137],
                    [0.9536, 0.8317],
                    [0.3718, 0.2456],
                    [0.3875, 0.8450],
                    [0.7592, 0.1687],
                    [0.5173, 0.6760],
                    [0.9446, 0.4570],
                    [0.6164, 0.1867],
                    [0.4732, 0.1786],
                    [0.4090, 0.8089],
                    [0.9742, 0.8943],
                    [0.5996, 0.7427],
                    [0.7038, 0.9210],
                    [0.6272, 0.0796],
                ],
                device=device,
                dtype=dtype,
            ),
            "confidence": torch.tensor(
                [
                    0.9314,
                    0.5951,
                    0.4187,
                    0.0318,
                    0.1434,
                    0.7952,
                    0.8306,
                    0.7511,
                    0.6407,
                    0.7379,
                    0.4363,
                    0.9220,
                    0.8453,
                    0.5075,
                    0.8141,
                ],
                device=device,
                dtype=dtype,
            ),
            "batch_indexes": torch.zeros((15,), device=device, dtype=dtype),
        }
        with patch(
            "kornia.contrib.ImageStitcher.on_matcher", new_callable=PropertyMock, return_value=lambda x: return_value
        ):
            # NOTE: This will need to download the pretrained weights.
            # To avoid that, we mock as below
            matcher = kornia.feature.LoFTR(None)
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator=estimator).to(device=device, dtype=dtype)
            torch.manual_seed(1)  # issue kornia#2027
            out = stitcher(sample1, sample2)
            assert out.shape[:-1] == torch.Size([1, 3, 6])
            assert out.shape[-1] <= 12

    @pytest.mark.slow
    def test_exception(self, device, dtype):
        B, C, H, W = 1, 3, 224, 224
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample2 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        # NOTE: This will need to download the pretrained weights.
        matcher = kornia.feature.LoFTR(None)

        with pytest.raises(NotImplementedError):
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator="random").to(device=device, dtype=dtype)

        stitcher = kornia.contrib.ImageStitcher(matcher).to(device=device, dtype=dtype)
        with pytest.raises(RuntimeError):
            stitcher(sample1, sample2)
