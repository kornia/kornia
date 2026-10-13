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

from testing.base import BaseTester
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

    def test_n_keeps_the_scores_above_the_n_plus_first_so_a_tie_at_the_cut_is_dropped_5864(self, device, dtype):
        # With more maxima than n, the keypoints are the maxima scoring strictly above the (n + 1)-th strongest. The
        # maxima tied at 2.0, (50, 9) and (20, 33), straddle the cut at n = 2, so neither is kept: picking one would
        # depend on the raster order of the pixels, which the transposed heatmap reverses. n = 0 keeps nothing, and
        # n = 4, the number of maxima, keeps all of them.
        peaks = [(0, 13, 21, 3.0), (0, 50, 9, 2.0), (0, 20, 33, 2.0), (0, 60, 40, 1.0)]
        peaks_t = [(b, y, x, s) for b, x, y, s in peaks]
        for heatmaps in (self._heatmaps(peaks), self._heatmaps(peaks_t, h=80, w=48)):
            scores = {}
            for n in (0, 1, 2, 3, 4):
                (kept,), _ = self._features(heatmaps, device, dtype, n=n)
                scores[n] = sorted(s for _, _, s in kept)
            assert scores == {0: [], 1: [3.0], 2: [3.0], 3: [2.0, 2.0, 3.0], 4: [1.0, 2.0, 2.0, 3.0]}
