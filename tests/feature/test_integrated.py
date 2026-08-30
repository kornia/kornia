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
from torch import nn

import kornia
from kornia.core._compat import torch_version_le
from kornia.feature import (
    DescriptorMatcher,
    GFTTAffNetHardNet,
    KeyNetHardNet,
    LAFDescriptor,
    LocalFeature,
    ScaleSpaceDetector,
    SIFTDescriptor,
    SIFTFeature,
    extract_patches_from_pyramid,
    get_laf_center,
    get_laf_descriptors,
    get_laf_orientation,
    get_laf_scale,
)
from kornia.feature.integrated import LocalFeatureMatcher
from kornia.geometry import RANSAC, resize, transform_points

from testing.base import BaseTester, supports_replicate_padding
from testing.casts import dict_to


class TestGetLAFDescriptors(BaseTester):
    def test_same(self, device, dtype):
        B, C, H, W = 1, 3, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        descs_test_from_rgb = get_laf_descriptors(img, lafs, sift, PS, True)
        descs_test_from_gray = get_laf_descriptors(img_gray, lafs, sift, PS, True)

        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        self.assert_close(descs_test_from_rgb, descs_reference)
        self.assert_close(descs_test_from_gray, descs_reference)

    def test_gradcheck(self, device):
        dtype = torch.float64
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        centers = torch.tensor([[H / 2.0, W / 2.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 5.0, (H + W) / 6.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)

        class _MeanPatch(nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        desc = _MeanPatch()
        self.gradcheck(
            get_laf_descriptors,
            (img, lafs, desc, PS, True),
            eps=1e-3,
            atol=1e-3,
            nondet_tol=1e-3,
        )


class TestGetLAFDescriptorsEmptyPath(BaseTester):
    """The empty-LAF result must agree with the non-empty one (see #4065)."""

    @pytest.mark.parametrize("output_dims", [64, 128, 238])
    def test_empty_matches_the_descriptor_width(self, device, dtype, output_dims):
        """The width came from a hardcoded 128 rather than from the descriptor module."""
        img = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        desc = kornia.feature.MKDDescriptor(patch_size=32, output_dims=output_dims).to(device, dtype)
        laf = kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[32.0, 32.0]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype),
        )
        non_empty = kornia.feature.get_laf_descriptors(img, laf, desc, 32)
        with pytest.warns(UserWarning):
            empty = kornia.feature.get_laf_descriptors(
                img, torch.zeros(1, 0, 2, 3, device=device, dtype=dtype), desc, 32
            )
        assert empty.shape == (1, 0, non_empty.shape[-1])
        assert empty.dtype == non_empty.dtype
        assert empty.device.type == non_empty.device.type

    @pytest.mark.parametrize("descriptor", ["four_dim", "dense_sift"])
    def test_empty_matches_a_non_2d_descriptor(self, device, dtype, descriptor):
        """The width must be what `.view(B, N, -1)` produces, not the probe's last dimension.

        The non-empty path flattens *everything* the descriptor produced per patch, so taking
        `probe.shape[-1]` agreed only for descriptors whose output is 2-D. `patch_descriptor` is
        typed as a bare `nn.Module`; `DenseSIFTDescriptor` returns `(N, C, H, W)` and so does any
        user module of that shape, and those got a narrower empty result that could not be
        concatenated with a non-empty one -- the very defect this path exists to remove (#4065).
        """

        class FourDimDescriptor(nn.Module):
            """Minimal descriptor whose output is not 2-D."""

            def forward(self, patches: torch.Tensor) -> torch.Tensor:
                return patches.new_zeros(patches.shape[0], 3, 2, 5)

        desc = (
            FourDimDescriptor()
            if descriptor == "four_dim"
            else kornia.feature.DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
        )
        desc = desc.to(device, dtype)
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        laf = kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[16.0, 16.0]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), 8.0, device=device, dtype=dtype),
        )
        non_empty = kornia.feature.get_laf_descriptors(img, laf, desc, 32)
        with pytest.warns(UserWarning):
            empty = kornia.feature.get_laf_descriptors(
                img, torch.zeros(1, 0, 2, 3, device=device, dtype=dtype), desc, 32
            )
        assert empty.shape == (1, 0, non_empty.shape[-1])
        # which is the point: an empty batch has to be concatenable with a non-empty one
        assert torch.cat([non_empty, empty], dim=1).shape == non_empty.shape


class TestLocalFeatureMatcherEmptyPath(BaseTester):
    """`no_match_output` must keep the rank the success path uses (see #4065)."""

    def test_empty_lafs_keep_the_batch_dimension(self, device, dtype):
        matcher = kornia.feature.LocalFeatureMatcher(
            kornia.feature.SIFTFeature(50), kornia.feature.DescriptorMatcher("snn", 0.9)
        )
        out = matcher.no_match_output(device, dtype)
        # the success path returns `.view(1, -1, 2, 3)`
        assert out["lafs0"].shape == (1, 0, 2, 3)
        assert out["lafs1"].shape == (1, 0, 2, 3)
        # so indexing the batch works either way
        assert out["lafs0"][0].shape == (0, 2, 3)
        assert out["keypoints0"].shape == (0, 2)


class TestLAFDescriptor(BaseTester):
    def test_same(self, device, dtype):
        B, C, H, W = 1, 3, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        lafsift = LAFDescriptor(sift, PS)
        descs_test = lafsift(img, lafs)
        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        self.assert_close(descs_test, descs_reference)

    def test_empty(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        lafs = torch.zeros(B, 0, 2, 3, device=device)
        sift = SIFTDescriptor(PS).to(device)
        lafsift = LAFDescriptor(sift, PS)
        descs_test = lafsift(img, lafs)
        assert descs_test.shape == (B, 0, 128)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        centers = torch.tensor([[H / 2.0, W / 2.0], [2.0 * H / 3.0, W / 2.0]], device=device).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 5.0, (H + W) / 6.0], device=device).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)

        class _MeanPatch(nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        lafdesc = LAFDescriptor(_MeanPatch(), PS)
        self.gradcheck(lafdesc, (img, lafs), eps=1e-3, atol=1e-3, nondet_tol=1e-3)


class TestLocalFeature(BaseTester):
    def test_smoke(self, device, dtype):
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(32)
        local_feature = LocalFeature(det, desc).to(device, dtype)
        assert local_feature is not None

    def test_same(self, device, dtype):
        B, C, H, W = 1, 1, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(PS)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS)).to(device, dtype)
        lafs, responses, descs = local_feature(img)
        lafs1, responses1 = det(img)
        self.assert_close(lafs, lafs1)
        self.assert_close(responses, responses1)
        patches = extract_patches_from_pyramid(img, lafs1, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs1 = desc(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        self.assert_close(descs, descs1)

    def test_scale(self, device, dtype):
        B, C, H, W = 1, 1, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(PS)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS), 1.0).to(device, dtype)
        local_feature2 = LocalFeature(det, LAFDescriptor(desc, PS), 2.0).to(device, dtype)
        lafs, _responses, _descs = local_feature(img)
        lafs2, _responses2, _descs2 = local_feature2(img)
        self.assert_close(get_laf_center(lafs), get_laf_center(lafs2))
        self.assert_close(get_laf_orientation(lafs), get_laf_orientation(lafs2))
        self.assert_close(2.0 * get_laf_scale(lafs), get_laf_scale(lafs2))

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = LocalFeature(ScaleSpaceDetector(2), LAFDescriptor(SIFTDescriptor(PS), PS)).to(device, img.dtype)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, nondet_tol=1e-8)


class TestSIFTFeature(BaseTester):
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        sift = SIFTFeature()
        assert sift is not None

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = SIFTFeature(2, True).to(device)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, fast_mode=False)


class TestSIFTFeatureHalfPrecision(BaseTester):
    def test_lafs_and_descriptors_are_finite(self, device):
        # The orienter's unguarded `sqrt`/`atan2` gave a NaN LAF at a *filled* slot, which the
        # padding mask cannot cover, and 128 NaN descriptor values in that row with it.
        if device.type == "mps":
            pytest.skip("MPS autocast changes the effective dtype")
        if not supports_replicate_padding(device, torch.float16):
            pytest.skip("no replicate-padding kernel for float16 on this device")
        torch.manual_seed(0)
        img = torch.rand(1, 1, 64, 64, device=device, dtype=torch.float16)
        with torch.no_grad():
            lafs, _, descs = SIFTFeature(50).to(device, torch.float16)(img)
        assert lafs.dtype == torch.float16 and descs.dtype == torch.float16
        assert int(lafs[0].ne(0).any(dim=-1).any(dim=-1).sum()) > 0
        assert torch.isfinite(lafs).all()
        assert torch.isfinite(descs).all()


class TestKeyNetHardNetFeature(BaseTester):
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        sift = KeyNetHardNet(2).to(device, dtype)
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        out = sift(img)
        assert out is not None

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = KeyNetHardNet(2, True).to(device).to(device)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, fast_mode=False)


class TestGFTTAffNetHardNet(BaseTester):
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        feat = GFTTAffNetHardNet().to(device, dtype)
        assert feat is not None

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64)
        local_feature = GFTTAffNetHardNet(2, True).to(device, img.dtype)
        self.gradcheck(local_feature, img, eps=1e-4, atol=1e-4, fast_mode=False)


class TestLocalFeatureMatcher(BaseTester):
    def test_smoke(self, device):
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher("snn", 0.8)).to(device)
        assert matcher is not None

    @pytest.mark.parametrize("match_type", ["nn", "mnn"])
    def test_detector_padding_is_not_matched(self, device, dtype, match_type):
        # A fixed-shape detector with no maxima returns zero LAFs. Their descriptors are finite,
        # but NN/MNN would otherwise report them as false correspondences at the image origin.
        feat = SIFTFeature(8, upright=True, score_threshold=1e6).to(device, dtype)
        matcher = LocalFeatureMatcher(feat, DescriptorMatcher(match_type, 0.8)).to(device, dtype)
        image0 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        image1 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        with torch.no_grad():
            out = matcher({"image0": image0, "image1": image1})
        assert out["keypoints0"].shape == (0, 2)
        assert out["keypoints1"].shape == (0, 2)
        assert out["lafs0"].shape == (1, 0, 2, 3)
        assert out["lafs1"].shape == (1, 0, 2, 3)

    def test_laf_and_descriptor_counts_must_agree(self, device, dtype):
        # The padding mask is derived from the LAFs and applied to the descriptors, so a count
        # mismatch used to surface as an opaque boolean-index `IndexError` deep in the loop.
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher("nn", 0.8)).to(device, dtype)
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        lafs = torch.rand(1, 3, 2, 3, device=device, dtype=dtype)
        descs = torch.rand(1, 5, 128, device=device, dtype=dtype)
        images = {"image0": img, "image1": img}
        data = {**images, "lafs0": lafs, "descriptors0": descs, "lafs1": lafs, "descriptors1": descs[:, :3]}
        with pytest.raises(Exception, match=r"lafs0 and descriptors0"):
            matcher(data)
        data = {**images, "lafs0": lafs, "descriptors0": descs[:, :3], "lafs1": lafs, "descriptors1": descs}
        with pytest.raises(Exception, match=r"lafs1 and descriptors1"):
            matcher(data)

    def test_masks_are_forwarded_to_the_detector(self, device, dtype):
        feat = SIFTFeature(8, upright=True).to(device, dtype)
        matcher = LocalFeatureMatcher(feat, DescriptorMatcher("nn", 0.8)).to(device, dtype)
        image0 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        image1 = torch.rand(1, 1, 64, 64, device=device, dtype=dtype)
        # LocalFeatureMatcher historically documents masks without a channel dimension; it adds
        # that singleton axis before forwarding to the detector's (B, 1, H, W) mask contract.
        mask = torch.zeros(1, 64, 64, device=device, dtype=torch.bool)
        with torch.no_grad():
            out = matcher({"image0": image0, "image1": image1, "mask0": mask, "mask1": mask})
        assert out["keypoints0"].shape == (0, 2)
        assert out["keypoints1"].shape == (0, 2)

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_nomatch(self, device, dtype, data):
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(100), DescriptorMatcher("snn", 0.8)).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            out = matcher({"image0": data_dev["image0"], "image1": 0 * data_dev["image0"]})
        assert len(out["keypoints0"]) == 0

    @pytest.mark.skip("Takes too long time (but works)")
    def test_gradcheck(self, device):
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher("nn", 1.0)).to(device)
        patches = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float64)
        patches05 = resize(patches, (48, 48))

        def proxy_forward(x, y):
            return matcher({"image0": x, "image1": y})["keypoints0"]

        self.gradcheck(proxy_forward, (patches, patches05), eps=1e-4, atol=1e-4)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_sift(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(1000), DescriptorMatcher("snn", 0.8)).to(device, dtype)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        with torch.no_grad():
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_sift_preextract(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        feat = SIFTFeature(1000).to(device, dtype)
        matcher = LocalFeatureMatcher(feat, DescriptorMatcher("snn", 0.8)).to(device)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]

        lafs, _, descs = feat(data_dev["image0"])
        data_dev["lafs0"] = lafs
        data_dev["descriptors0"] = descs

        lafs2, _, descs2 = feat(data_dev["image1"])
        data_dev["lafs1"] = lafs2
        data_dev["descriptors1"] = descs2

        with torch.no_grad():
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_gftt(self, device, dtype, data):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(1000), DescriptorMatcher("snn", 0.8)).to(device, dtype)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        with torch.no_grad():
            torch.manual_seed(0)
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Fails for bached torch.linalg.solve")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_real_keynet(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(KeyNetHardNet(500), DescriptorMatcher("snn", 0.9)).to(device, dtype)
        ransac = RANSAC("homography", 1.0, 1024, 5).to(device, dtype)
        data_dev = dict_to(data, device, dtype)
        pts_src = data_dev["pts0"]
        pts_dst = data_dev["pts1"]
        with torch.no_grad():
            out = matcher(data_dev)
        homography, inliers = ransac(out["keypoints0"], out["keypoints1"])
        assert inliers.sum().item() > 50  # we have enough inliers
        # Reprojection error of 5px is OK
        self.assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)

    @pytest.mark.skip("ScaleSpaceDetector now is not jittable")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        inputs = {"image0": patches, "image1": patches2x}
        model = LocalFeatureMatcher(SIFTDescriptor(32), DescriptorMatcher("snn", 0.8)).to(device).eval()
        model_jit = torch.jit.script(model)

        out = model(inputs)
        out_jit = model_jit(inputs)
        for k, v in out.items():
            self.assert_close(v, out_jit[k])
