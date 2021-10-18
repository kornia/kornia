import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
import kornia
from kornia.feature import *
from kornia.geometry import (
    ConvQuadInterp3d,
    ScalePyramid,
    resize,
    RANSAC,
    transform_points
)
from kornia.testing import assert_close


class TestGetLAFDescriptors:
    def test_same(self, device, dtype):
        B, C, H, W = 1, 3, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.],
                                [2.0 * H / 3.0, W / 2.]],
                               device=device,
                               dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0],
                              device=device,
                              dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.],
                           device=device,
                           dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        descs_test_from_rgb = get_laf_descriptors(img, lafs, sift, PS, True)
        descs_test_from_gray = get_laf_descriptors(img_gray,
                                                   lafs, sift, PS, True)

        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        assert_close(descs_test_from_rgb, descs_reference)
        assert_close(descs_test_from_gray, descs_reference)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        centers = torch.tensor([[H / 3.0, W / 3.],
                                [2.0 * H / 3.0, W / 2.]],
                               device=device,
                               dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0],
                              device=device,
                              dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.],
                           device=device,
                           dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        lafs = utils.tensor_to_gradcheck_var(lafs)  # to var

        class MeanPatch(nn.Module):
            def forward(self, input):
                return input.mean(dim=(2, 3))
        desc = MeanPatch()
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(get_laf_descriptors,
                         (img, lafs, desc, PS, True),
                         eps=1e-3, atol=1e-3, raise_exception=True)


class TestLAFDescriptor:
    def test_same(self, device, dtype):
        B, C, H, W = 1, 3, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.],
                                [2.0 * H / 3.0, W / 2.]],
                               device=device,
                               dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0],
                              device=device,
                              dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.],
                           device=device,
                           dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        lafsift = LAFDescriptor(sift, PS)
        descs_test = lafsift(img, lafs)
        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        assert_close(descs_test, descs_reference)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        centers = torch.tensor([[H / 3.0, W / 3.],
                                [2.0 * H / 3.0, W / 2.]],
                               device=device,
                               dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0],
                              device=device,
                              dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.],
                           device=device,
                           dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        lafs = utils.tensor_to_gradcheck_var(lafs)  # to var

        class MeanPatch(nn.Module):
            def forward(self, input):
                return input.mean(dim=(2, 3))
        desc = MeanPatch()
        lafdesc = LAFDescriptor(MeanPatch(), PS)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(lafdesc,
                         (img, lafs),
                         eps=1e-3, atol=1e-3, raise_exception=True)


class TestLocalFeature:
    def test_smoke(self, device, dtype):
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(32)
        local_feature = LocalFeature(det, desc).to(device, dtype)

    def test_same(self, device, dtype):
        B, C, H, W = 1, 1, 64, 64
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(PS)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS)).to(device, dtype)
        lafs, responses, descs = local_feature(img)
        lafs1, responses1 = det(img)
        assert_close(lafs, lafs1)
        assert_close(responses, responses1)
        patches = extract_patches_from_pyramid(img, lafs1, PS)
        B1, N1, CH1, H1, W1 = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs1 = desc(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        assert_close(descs, descs1)

    @pytest.mark.skip("Takes too long time (but works)")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        local_feature = LocalFeature(ScaleSpaceDetector(2),
                                     LAFDescriptor(SIFTDescriptor(PS), PS)).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(local_feature, img, eps=1e-4, atol=1e-4, raise_exception=True)


class TestSIFTFeature:
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        sift = SIFTFeature()

    def test_same(self, device, dtype):
        PS = 41
        num_features = 5
        data = torch.load("data/test/loftr_indoor_and_fundamental_data.pt")
        img = data['image0'].to(device=device, dtype=dtype)
        img = resize(img, (128, 128))
        det = ScaleSpaceDetector(num_features,
                                 resp_module=BlobDoG(),
                                 nms_module=ConvQuadInterp3d(10),
                                 scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
                                 ori_module=PassLAF(),
                                 scale_space_response=True,
                                 minima_are_also_good=True,
                                 mr_size=6.0).to(device)
        desc = SIFTDescriptor(PS, rootsift=True)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS)).to(device, dtype)
        sift_feature = SIFTFeature(5, True).to(device, dtype)
        with torch.no_grad():
            lafs, responses, descs = local_feature(img)
            lafs1, responses1, descs1 = sift_feature(img)
        assert_close(lafs, lafs1)
        assert_close(responses, responses1)
        assert_close(descs, descs1)

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        local_feature = SIFTFeature(2, True).to(device).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(local_feature, img, eps=1e-4, atol=1e-4, raise_exception=True)


class TestGFTTAffNetHardNet:
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        feat = GFTTAffNetHardNet()

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        local_feature = GFTTAffNetHardNet(2, True).to(device, torch.double)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(local_feature, img, eps=1e-4, atol=1e-4, raise_exception=True)


class TestLocalFeatureMatcher:
    def test_smoke(self, device):
        matcher = LocalFeatureMatcher(SIFTFeature(5),
                                      DescriptorMatcher('snn', 0.8)).to(device)

    def test_nomatch(self, device, dtype):
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(100),
                                      DescriptorMatcher('snn', 0.8)).to(device)
        ransac = RANSAC('homography', 1.0, 2048, 10).to(device)
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        with torch.no_grad():
            out = matcher({"image0": data["image0"],
                           "image1": 0 * data["image0"]})
        assert (len(out['keypoints0']) == 0)

    @pytest.mark.skip("Takes too long time (but works)")
    def test_gradcheck(self, device):
        matcher = LocalFeatureMatcher(SIFTFeature(5),
                                      DescriptorMatcher('nn', 1.0)).to(device)
        patches = torch.rand(1, 1, 32, 32, device=device)
        patches05 = resize(patches, (48, 48))
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        patches05 = utils.tensor_to_gradcheck_var(patches05)  # to var

        def proxy_forward(x, y):
            return matcher({"image0": x, "image1": y})["keypoints0"]
        assert gradcheck(proxy_forward, (patches, patches05), eps=1e-4, atol=1e-4, raise_exception=True)

    def test_real_sift(self, device, dtype):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(2000),
                                      DescriptorMatcher('snn', 0.8)).to(device)
        ransac = RANSAC('homography', 1.0, 2048, 10).to(device)
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        with torch.no_grad():
            out = matcher(data)
        homography, inliers = ransac(out['keypoints0'], out['keypoints1'])
        assert (inliers.sum().item() > 50)  # we have enough inliers
        # Reprojection error of 5px is OK
        assert_close(
            transform_points(homography[None], pts_src[None]),
            pts_dst[None],
            rtol=5e-2,
            atol=5)

    def test_real_sift_preextract(self, device, dtype):
        # This is not unit test, but that is quite good integration test
        feat = SIFTFeature(2000)
        matcher = LocalFeatureMatcher(feat,
                                      DescriptorMatcher('snn', 0.8)).to(device)
        ransac = RANSAC('homography', 1.0, 2048, 10).to(device)
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        lafs, resps, descs = feat(data["image0"])
        data["lafs0"] = lafs
        data["descriptors0"] = descs
        lafs2, resps2, descs2 = feat(data["image1"])
        data["lafs1"] = lafs2
        data["descriptors1"] = descs2

        with torch.no_grad():
            out = matcher(data)
        homography, inliers = ransac(out['keypoints0'], out['keypoints1'])
        assert (inliers.sum().item() > 50)  # we have enough inliers
        # Reprojection error of 5px is OK
        assert_close(
            transform_points(homography[None], pts_src[None]),
            pts_dst[None],
            rtol=5e-2,
            atol=5)

    def test_real_gftt(self, device, dtype):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(2000),
                                      DescriptorMatcher('snn', 0.8)).to(device)
        ransac = RANSAC('homography', 1.0, 2048, 10).to(device)
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        with torch.no_grad():
            out = matcher(data)
        homography, inliers = ransac(out['keypoints0'], out['keypoints1'])
        assert (inliers.sum().item() > 50)  # we have enough inliers
        # Reprojection error of 5px is OK
        assert_close(
            transform_points(homography[None], pts_src[None]),
            pts_dst[None],
            rtol=5e-2,
            atol=5)

    @pytest.mark.skip("ScaleSpaceDetector now is not jittable")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        input = {"image0": patches, "image1": patches2x}
        matcher = LocalFeatureMatcher(ScaleSpaceDetector(50),
                                      SIFTDescriptor(32),
                                      DescriptorMatcher('snn', 0.8)).to(device).eval()
        model_jit = torch.jit.script(LocalFeatureMatcher(ScaleSpaceDetector(50),
                                                         SIFTDescriptor(32),
                                                         DescriptorMatcher('snn', 0.8)).to(device).eval())
        out = model(input)
        out_jit = model(input)
        for k, v in out.items():
            assert_close(v, out_jit[k])
