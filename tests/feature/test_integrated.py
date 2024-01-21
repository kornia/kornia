import sys

import pytest
import torch
from torch import nn

import kornia
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
from kornia.utils._compat import torch_version_le
from testing.base import BaseTester
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
        lafs, responses, descs = local_feature(img)
        lafs2, responses2, descs2 = local_feature2(img)
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
