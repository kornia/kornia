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

import sys

import pytest
import torch

from kornia.feature.disk import DISK, DISKFeatures

from testing.base import (
    BaseTester,
    supports_bicubic_2d_grid_sample,
    supports_bilinear_2d_grid_sample,
    supports_replicate_padding,
)
from testing.casts import dict_to


class TestDisk(BaseTester):
    def test_smoke(self, dtype, device):
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        assert all(isinstance(e, DISKFeatures) for e in output)

    def test_smoke_n_detections(self, dtype, device):
        """Unless we give it an actual image and use pretrained weights, we can't expect the number of detections
        to really match the limit.
        """
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        output = disk(inp, n=100)
        assert isinstance(output, list)
        assert len(output) == 1
        assert all(isinstance(e, DISKFeatures) for e in output)

    @pytest.mark.slow
    def test_smoke_pretrained(self, device):
        disk = DISK.from_pretrained(checkpoint="depth", device=device)
        inp = torch.ones(1, 3, 64, 64, device=device)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        assert all(isinstance(e, DISKFeatures) for e in output)

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["disk_outdoor"], indirect=True)
    def test_pretrained_outdoor(self, device, dtype, data):
        disk = DISK.from_pretrained(checkpoint="depth", device=device).to(dtype)
        data_dev = dict_to(data, device, dtype)
        num_feat = 256
        with torch.no_grad():
            out = disk(data_dev["img1"], num_feat)
        if device.type != "cpu":
            pytest.skip("Reference keypoints were computed on CPU; NMS outcomes differ on non-CPU devices")
        self.assert_close(out[0].keypoints, data_dev["disk1"][0].keypoints.to(device=device, dtype=dtype))
        self.assert_close(out[0].descriptors, data_dev["disk1"][0].descriptors.to(device=device, dtype=dtype))

    def test_heatmap_and_dense_descriptors(self, dtype, device):
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        heatmaps, descriptors = disk.heatmap_and_dense_descriptors(inp)

        assert heatmaps.shape == (1, 1, 64, 64)
        assert descriptors.shape == (1, 128, 64, 64)
        assert heatmaps.dtype == dtype
        assert descriptors.dtype == dtype

    def test_half_model_returns_float32_keypoints(self, device, dtype):
        # R4: keypoints are integer pixels, and bfloat16 holds no odd integer above 256, so a bfloat16 DISK put every
        # keypoint past x = 256 on an even pixel. They are float32 now; scores and descriptors keep the model dtype.
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision rule")
        torch.manual_seed(0)
        disk = DISK().to(device, dtype)
        image = torch.rand(1, 3, 112, 400, generator=torch.Generator().manual_seed(0)).to(device, dtype)
        with torch.no_grad():
            (feats,) = disk(image)
        assert feats.keypoints.dtype == torch.float32
        assert feats.detection_scores.dtype == dtype
        assert feats.descriptors.dtype == dtype
        x = feats.keypoints[:, 0]
        assert (x[x > 256] % 2 != 0).any()  # odd pixels past 256, which bfloat16 cannot hold

    def test_not_divisible_by_16(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 3, 72, 64, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)

        _ = disk(inp, pad_if_not_divisible=True)

        inp = torch.ones(1, 3, 64, 72, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)

        _ = disk(inp, pad_if_not_divisible=True)

        inp = torch.ones(1, 3, 72, 72, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)

        _ = disk(inp, pad_if_not_divisible=True)

    def test_wrong_n_channels(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 1, 64, 64, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)


class TestDiskKeypointSelection(BaseTester):
    """``n`` is an upper bound: with ``n`` maxima or fewer, every maximum is a keypoint (#5864)."""

    class _FixedUNet(torch.nn.Module):
        # The public ``unet`` argument of DISK, returning a fixed output.
        def __init__(self, out: torch.Tensor) -> None:
            super().__init__()
            self.out = out

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.out

    @staticmethod
    def _heatmaps(peaks, batch=1, h=48, w=80):
        # (batch, h, w) heatmaps of -1 with the given peaks (image, x, y, logit)
        heatmaps = torch.full((batch, h, w), -1.0, dtype=torch.float64)
        for b, x, y, logit in peaks:
            heatmaps[b, y, x] = logit
        return heatmaps

    def _features(self, heatmaps, device, dtype, **kwargs):
        # DISK whose U-Net returns the heatmaps and descriptor channels (column, row, 1), so a descriptor names its
        # pixel; returns per image the sorted (x, y, score) of its keypoints and the features themselves
        b, h, w = heatmaps.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
        )
        out = torch.stack([xx.expand(b, h, w), yy.expand(b, h, w), torch.ones(b, h, w, dtype=torch.float64), heatmaps])
        disk = DISK(desc_dim=3, unet=self._FixedUNet(out.transpose(0, 1).to(device, dtype))).eval()
        feats = disk(torch.rand(b, 3, h, w, device=device, dtype=dtype), **kwargs)
        kept = [
            sorted((x, y, s) for (x, y), s in zip(f.keypoints.tolist(), f.detection_scores.tolist())) for f in feats
        ]
        return kept, feats

    def test_n_at_or_above_the_number_of_maxima_returns_every_maximum_5864(self, device, dtype):
        # Five maxima of a 48 x 80 heatmap. n = 4 keeps the four strongest; n = 5 and n = 55 return the same keypoints,
        # descriptors and scores as n = None, in the same (raster) order.
        peaks = [(0, 13, 21, 2.0), (0, 50, 9, 1.5), (0, 60, 33, 1.0), (0, 27, 40, 0.5), (0, 70, 14, 3.0)]
        expected = sorted((float(x), float(y), s) for _, x, y, s in peaks)
        # relabelling: the transposed heatmap gives the transposed keypoints
        peaks_t = [(b, y, x, s) for b, x, y, s in peaks]
        for heatmaps, expected_i in (
            (self._heatmaps(peaks), expected),
            (self._heatmaps(peaks_t, h=80, w=48), sorted((y, x, s) for x, y, s in expected)),
        ):
            (kept_all,), (feats_all,) = self._features(heatmaps, device, dtype)
            assert kept_all == expected_i
            (kept_4,), _ = self._features(heatmaps, device, dtype, n=4)
            assert kept_4 == [k for k in expected_i if k[2] != 0.5]
            for n in (5, 55):
                (kept_n,), (feats_n,) = self._features(heatmaps, device, dtype, n=n)
                assert kept_n == expected_i
                self.assert_close(feats_n.keypoints, feats_all.keypoints, rtol=0.0, atol=0.0)
                self.assert_close(feats_n.descriptors, feats_all.descriptors, rtol=0.0, atol=0.0)
                self.assert_close(feats_n.detection_scores, feats_all.detection_scores, rtol=0.0, atol=0.0)

    def test_n_is_applied_per_image_and_an_image_without_maxima_returns_none_5864(self, device, dtype):
        # n = 3 on a batch of three: five maxima give the three strongest, two give both, and a heatmap without a
        # maximum gives no keypoint.
        peaks = [(0, 13, 21, 2.0), (0, 50, 9, 1.5), (0, 60, 33, 1.0), (0, 27, 40, 0.5), (0, 70, 14, 3.0)]
        peaks += [(1, 33, 30, 1.25), (1, 8, 12, 0.75)]
        kept, _ = self._features(self._heatmaps(peaks, batch=3), device, dtype, n=3)
        assert kept == [
            [(13.0, 21.0, 2.0), (50.0, 9.0, 1.5), (70.0, 14.0, 3.0)],
            [(8.0, 12.0, 0.75), (33.0, 30.0, 1.25)],
            [],
        ]

    def test_n_counts_the_maxima_above_score_threshold_5864(self, device, dtype):
        # n counts the maxima above score_threshold: of five maxima, three are strictly above 1.0, so n = 3, 4 and 10
        # keep those three, and n = 2 the two strongest of them.
        peaks = [(0, 13, 21, 2.0), (0, 50, 9, 1.5), (0, 60, 33, 1.0), (0, 27, 40, 0.5), (0, 70, 14, 3.0)]
        heatmaps = self._heatmaps(peaks)
        scores = {}
        for n in (None, 2, 3, 4, 10):
            (kept,), _ = self._features(heatmaps, device, dtype, n=n, score_threshold=1.0)
            scores[n] = sorted(s for _, _, s in kept)
        three = [1.5, 2.0, 3.0]
        assert scores == {None: three, 2: [2.0, 3.0], 3: three, 4: three, 10: three}

    def test_n_takes_maxima_tied_at_the_cut_in_raster_order_5864(self, device, dtype):
        # With more maxima than n, the n strongest are kept, and maxima tied at the n-th score are taken in raster
        # order (row by row) until there are n. The maxima (50, 9) and (20, 33) tie at 2.0 across the cut at n = 2:
        # (50, 9) comes first in raster order and is kept. Transposing the heatmap swaps their raster order, so the
        # transposed (20, 33) is kept instead. n = 0 keeps nothing.
        peaks = [(0, 13, 21, 3.0), (0, 50, 9, 2.0), (0, 20, 33, 2.0), (0, 60, 40, 1.0)]
        peaks_t = [(b, y, x, s) for b, x, y, s in peaks]
        expected = {
            0: [],
            1: [(13.0, 21.0, 3.0)],
            2: [(13.0, 21.0, 3.0), (50.0, 9.0, 2.0)],
            3: [(13.0, 21.0, 3.0), (20.0, 33.0, 2.0), (50.0, 9.0, 2.0)],
            4: [(13.0, 21.0, 3.0), (20.0, 33.0, 2.0), (50.0, 9.0, 2.0), (60.0, 40.0, 1.0)],
        }
        expected_t = {n: sorted((y, x, s) for x, y, s in kept) for n, kept in expected.items()}
        expected_t[2] = [(21.0, 13.0, 3.0), (33.0, 20.0, 2.0)]
        for heatmaps, expected_i in (
            (self._heatmaps(peaks), expected),
            (self._heatmaps(peaks_t, h=80, w=48), expected_t),
        ):
            kept = {n: self._features(heatmaps, device, dtype, n=n)[0][0] for n in expected_i}
            assert kept == expected_i


class _FixedUNet(torch.nn.Module):
    """A U-Net stand-in that returns a fixed output and records its input."""

    def __init__(self, output):
        super().__init__()
        self.output = output
        self.seen = []

    def forward(self, images):
        self.seen.append(images)
        return self.output


class TestConventionsDISK(BaseTester):
    @staticmethod
    def _disk(peaks, device, dtype, h=48, w=80):
        # DISK with its U-Net replaced (the public unet argument) by a fixed output: heatmap channel 128 is -1 with the
        # given peaks (x, y, logit); descriptor channels 0, 1, 2 are the column, the row and 1, so a descriptor names
        # its pixel.
        out = torch.zeros(1, 129, h, w, dtype=torch.float64)
        out[0, 128] = -1.0
        for x, y, logit in peaks:
            out[0, 128, y, x] = logit
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
        )
        out[0, 0], out[0, 1], out[0, 2] = xx, yy, torch.ones_like(xx)
        return DISK(unet=_FixedUNet(out.to(device, dtype))).eval()

    def test_convention_disk_keypoints_are_the_xy_pixels_of_the_heatmap_maxima(self, device, dtype):
        # Peaks at (x, y) = (13, 21), (50, 9), (60, 33) of a 48 x 80 heatmap. The keypoints are those pixels as (x, y),
        # in raster (row-major) order rather than by score; detection_scores are the heatmap values there and each
        # descriptor is the L2-normalised dense descriptor at its pixel.
        peaks = [(13, 21, 2.0), (50, 9, 1.5), (60, 33, 1.0)]
        expected = torch.tensor([[50.0, 9.0], [13.0, 21.0], [60.0, 33.0]], device=device, dtype=dtype)
        image = torch.rand(1, 3, 48, 80, device=device, dtype=dtype)
        (feats,) = self._disk(peaks, device, dtype)(image)
        # the keypoints are float32 for a half-precision network, the scores and descriptors keep its dtype
        self.assert_close(feats.keypoints, expected.to(torch.promote_types(dtype, torch.float32)), rtol=0.0, atol=0.0)
        self.assert_close(feats.detection_scores, torch.tensor([1.5, 2.0, 1.0], device=device, dtype=dtype))
        named = torch.cat([expected, torch.ones_like(expected[:, :1])], dim=-1)
        self.assert_close(feats.descriptors[:, :3], torch.nn.functional.normalize(named, dim=-1), low_tolerance=True)
        assert feats.descriptors[:, 3:].abs().max() == 0
        # heatmap_and_dense_descriptors returns the dense descriptors as the network outputs them, not normalised
        _heatmaps, dense = self._disk(peaks, device, dtype).heatmap_and_dense_descriptors(image)
        self.assert_close(dense[0, :3, 21, 13], torch.tensor([13.0, 21.0, 1.0], device=device, dtype=dtype))
        # relabelling: the transposed heatmap gives the transposed keypoints, again in raster order
        disk_t = self._disk([(y, x, s) for x, y, s in peaks], device, dtype, h=80, w=48)
        (feats_t,) = disk_t(image.transpose(-1, -2))
        assert feats_t.keypoints.tolist() == [[21.0, 13.0], [9.0, 50.0], [33.0, 60.0]]

    def test_convention_disk_takes_raw_rgb_and_pads_bottom_right_to_a_multiple_of_16(self, device, dtype):
        # The U-Net sees the RGB input as given (cvlab-epfl/disk divides uint8 images by 255 before the network). With
        # pad_if_not_divisible a 45 x 75 image is zero-padded at the bottom and right to 48 x 80, and the maps are
        # cropped back to 45 x 75 before the suppression: the peaks at (40, 45) and (77, 10) lie in the padding, and
        # (73, 20) is 1 px from the cropped right edge, so none of them is a keypoint; (72, 30), 2 px from it, is.
        peaks = [(13, 21, 2.0), (72, 30, 1.0), (73, 20, 1.0), (40, 45, 3.0), (77, 10, 3.0)]
        disk = self._disk(peaks, device, dtype)
        image = torch.rand(1, 3, 45, 75, device=device, dtype=dtype)
        (feats,) = disk(image, pad_if_not_divisible=True)
        seen = disk.unet.seen[0]
        assert seen.shape == (1, 3, 48, 80)
        self.assert_close(seen[..., :45, :75], image, rtol=0.0, atol=0.0)
        assert seen[..., 45:, :].abs().max() == 0
        assert seen[..., :, 75:].abs().max() == 0
        assert feats.keypoints.tolist() == [[13.0, 21.0], [72.0, 30.0]]

    def test_convention_disk_default_unet_sees_a_common_gain_only_through_the_norms_eps(self, device, dtype):
        # The default U-Net instance-normalises every channel before each convolution but the first, which removes a
        # common positive gain of the input together with the first convolution's bias, so [0, 255] gives the maps,
        # keypoints and descriptors of [0, 1] up to the norms' eps (random weights) at ordinary contrast. At low
        # contrast eps weighs against the variance and the gain shows. An offset is not removed: the first
        # convolution's zero padding does not move with it.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            disk = DISK().eval()
            image = torch.rand(1, 3, 48, 80)
        disk, image = disk.to(device, dtype), image.to(device, dtype)
        tol = {torch.float16: 2e-2, torch.bfloat16: 1e-1}.get(dtype, 3e-3)  # the maps span about 3
        # torch 2.5.1's CPU float16 instance norm overflows on a [0, 255] image, so float16 scales by 16 instead
        scale = 16.0 if dtype == torch.float16 else 255.0
        with torch.no_grad():
            heatmap, dense = disk.heatmap_and_dense_descriptors(image)
            heatmap_s, dense_s = disk.heatmap_and_dense_descriptors(scale * image)
            self.assert_close(heatmap_s, heatmap, rtol=0.0, atol=tol)
            self.assert_close(dense_s, dense, rtol=0.0, atol=tol)
            if dtype in (torch.float32, torch.float64):  # half-precision rounding can swap near-tied maxima
                (feats,), (feats_s,) = disk(image), disk(scale * image)
                assert feats.n > 10
                assert torch.equal(feats_s.keypoints, feats.keypoints)
                self.assert_close(feats_s.descriptors, feats.descriptors, rtol=0.0, atol=tol)
            assert (disk.heatmap_and_dense_descriptors(image + 0.3)[0] - heatmap).abs().max() > 0.2
            low = 0.01 * image  # intensities spread over 0.01: the same gain moves the heatmap by about 0.5
            heatmap_low = disk.heatmap_and_dense_descriptors(low)[0]
            assert (disk.heatmap_and_dense_descriptors(scale * low)[0] - heatmap_low).abs().max() > 0.2

    def test_convention_disk_n_keeps_the_strongest_and_score_threshold_is_strict(self, device, dtype):
        # n is an upper bound on the maxima above score_threshold: with more maxima than n, forward keeps the n
        # strongest, and with n of them it keeps all; score_threshold keeps the scores strictly above it. The keypoints
        # stay in raster order. TestDiskKeypointSelection pins the count above score_threshold and the ties at the cut.
        disk = self._disk([(13, 21, 2.0), (50, 9, 1.5), (60, 33, 1.0)], device, dtype)
        image = torch.rand(1, 3, 48, 80, device=device, dtype=dtype)
        kept = {}
        for kwargs in ({}, {"n": 1}, {"n": 2}, {"n": 3}, {"score_threshold": 1.0}, {"score_threshold": 1.5}):
            (feats,) = disk(image, **kwargs)
            kept[str(kwargs)] = feats.detection_scores.tolist()
        assert kept == {
            "{}": [1.5, 2.0, 1.0],
            "{'n': 1}": [2.0],
            "{'n': 2}": [1.5, 2.0],
            "{'n': 3}": [1.5, 2.0, 1.0],
            "{'score_threshold': 1.0}": [1.5, 2.0],
            "{'score_threshold': 1.5}": [2.0],
        }

    @pytest.mark.parametrize("model", ["DISK", "ALIKED", "XFeat"])
    def test_convention_learned_detectors_keep_strict_maxima_whose_window_fits_in_the_image(self, device, dtype, model):
        # DISK, ALIKED and XFeat keep a position only if it is greater than every other score of its 5 x 5 window and
        # the window lies in the image: the peaks one pixel from each border, the tied pair at (40, 50) and (41, 50),
        # and the flank (12, 30), (14, 30) of the peak at (10, 30) are not keypoints. This departs on purpose from
        # cvlab-epfl/disk, Shiaoming/ALIKED and verlab/accelerated_features, whose max-pool NMS clips the window at the
        # border and keeps ties, and, for ALIKED, re-admits the flank. ALIKED's soft-argmax moves the (10, 30) peak
        # towards its flank by 2 * (e**-0.5 - e**-8.9) / (1 + e**-0.5 + 23 * e**-8.9) = 0.7534 px; in half precision
        # ALIKED's positions are checked to 0.25 px, enough to tell the surviving maxima apart.
        from kornia.feature import ALIKED, XFeat

        if model == "ALIKED" and not (
            supports_replicate_padding(device, dtype) and supports_bilinear_2d_grid_sample(device, dtype)
        ):
            pytest.skip("ALIKED pads with replication and samples with bilinear grid_sample, absent for this dtype")
        border = [(1, 40, 0.95), (94, 20, 0.95), (60, 1, 0.95), (45, 62, 0.95)]
        inside = [(2, 10, 0.6), (93, 50, 0.6), (70, 2, 0.6), (20, 61, 0.6)]
        tie_and_flank = [(40, 50, 0.7), (41, 50, 0.7), (10, 30, 0.9), (12, 30, 0.85), (14, 30, 0.8)]
        peaks = border + inside + tie_and_flank
        kept = [[70.0, 2.0], [2.0, 10.0], [10.7534 if model == "ALIKED" else 10.0, 30.0], [93.0, 50.0], [20.0, 61.0]]
        half = dtype in (torch.float16, torch.bfloat16)
        tol = (0.25 if half else 1e-4) if model == "ALIKED" else 0.0
        for h, w, peaks_i, expected in (
            (64, 96, peaks, kept),
            (96, 64, [(y, x, s) for x, y, s in peaks], [[y, x] for x, y in kept]),  # relabelling: transpose
        ):
            image = torch.rand(1, 3, h, w, device=device, dtype=dtype)
            if model == "DISK":
                out = torch.zeros(1, 129, h, w, dtype=torch.float64)
                out[0, 128] = -1.0
                for x, y, s in peaks_i:
                    out[0, 128, y, x] = s
                (feats,) = DISK(unet=_FixedUNet(out.to(device, dtype))).eval()(image)
                keypoints = feats.keypoints
            elif model == "ALIKED":
                score_map = torch.full((1, 1, h, w), 0.01, dtype=torch.float64)
                for x, y, s in peaks_i:
                    score_map[0, 0, y, x] = s
                logits = torch.logit(score_map).to(device, dtype)
                aliked = ALIKED(model_name="aliked-t16").to(device, dtype).eval()
                aliked.score_head.register_forward_hook(lambda m, i, o, logits=logits: logits)
                with torch.no_grad():
                    (feats,) = aliked(image)
                keypoints = feats.keypoints
            else:
                if not supports_bicubic_2d_grid_sample(device, torch.float32):
                    pytest.skip("XFeat samples its descriptors bicubically, absent on this device")
                # keypoint logits: dustbin 10, a peak of score s gets 10 + 2 s in its cell's channel (y % 8) * 8 + x % 8
                logits = torch.zeros(1, 65, h // 8, w // 8)
                logits[0, 64] = 10.0
                for x, y, s in peaks_i:
                    logits[0, (y % 8) * 8 + x % 8, y // 8, x // 8] = 10.0 + 2.0 * s
                outputs = (torch.zeros(1, 64, h // 8, w // 8), logits, torch.full((1, 1, h // 8, w // 8), 0.5))
                xfeat = XFeat().to(device).eval()
                outputs = tuple(t.to(device) for t in outputs)
                xfeat.net.register_forward_hook(lambda m, i, o, outputs=outputs: outputs)
                keypoints = xfeat.detectAndCompute(image)[0]["keypoints"]
            order = torch.argsort(keypoints[:, 1] * 1000 + keypoints[:, 0])  # raster order
            reference = torch.tensor(sorted(expected, key=lambda p: (p[1], p[0])), device=device)
            self.assert_close(keypoints[order].float(), reference, rtol=0.0, atol=tol)
