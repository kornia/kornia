import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.feature import SIFTDescriptor, extract_patches_from_pyramid,get_laf_descriptors,LAFDescriptor,ScaleSpaceDetector, LocalFeature, GFTTAffNetHardNet,SIFTFeature,DescriptorMatcher
from kornia.feature.integrated import LocalFeatureMatcher
from kornia.geometry import RANSAC, resize, transform_points
from kornia.testing import assert_close


@pytest.fixture
def data():
    url = 'https://github.com/kornia/data_test/blob/main/loftr_outdoor_and_homography_data.pt?raw=true'
    return torch.hub.load_state_dict_from_url(url)


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
        dtype = torch.double
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        centers = torch.tensor([[H / 3.0, W / 3.],
                                [2.0 * H / 3.0, W / 2.]],
                               device=device,
                               dtype=torch.double).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0],
                              device=device, dtype=torch.double).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.],
                           device=device,
                           dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        lafs = utils.tensor_to_gradcheck_var(lafs)  # to var

        class _MeanPatch(nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        desc = _MeanPatch()
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
        dtype = torch.double
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

        class _MeanPatch(nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(2, 3))

        lafdesc = LAFDescriptor(_MeanPatch(), PS)
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

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device)
        local_feature = SIFTFeature(2, True).to(device).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(local_feature, img, eps=1e-4, atol=1e-4, raise_exception=True)


class TestGFTTAffNetHardNet:
    # The real test is in TestLocalFeatureMatcher
    def test_smoke(self, device, dtype):
        feat = GFTTAffNetHardNet().to(device, dtype)
        assert feat is not None

    @pytest.mark.skip("jacobian not well computed")
    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        img = torch.rand(B, C, H, W, device=device)
        local_feature = GFTTAffNetHardNet(2, True).to(device, torch.double)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(local_feature, img, eps=1e-4, atol=1e-4, raise_exception=True)


class TestLocalFeatureMatcher:
    def test_smoke(self, device):
        matcher = LocalFeatureMatcher(SIFTFeature(5),
                                      DescriptorMatcher('snn', 0.8)).to(device)

    def test_nomatch(self, device, dtype, data):
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(100),
                                      DescriptorMatcher('snn', 0.8)).to(device, dtype)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
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

    def test_real_sift(self, device, dtype, data):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(2000),
                                      DescriptorMatcher('snn', 0.8)).to(device, dtype)
        ransac = RANSAC('homography', 1.0, 2048, 10).to(device, dtype)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        with torch.no_grad():
            out = matcher(data)
        torch.random.manual_seed(0)
        homography, inliers = ransac(out['keypoints0'], out['keypoints1'])
        assert (inliers.sum().item() > 50)  # we have enough inliers
        # Reprojection error of 5px is OK
        assert_close(
            transform_points(homography[None], pts_src[None]),
            pts_dst[None],
            rtol=5e-2,
            atol=5)

    def test_real_sift_preextract(self, device, dtype, data):
        # This is not unit test, but that is quite good integration test
        feat = SIFTFeature(2000)
        matcher = LocalFeatureMatcher(feat,
                                      DescriptorMatcher('snn', 0.8)).to(device)
        ransac = RANSAC('homography', 1.0, 2048, 10).to(device, dtype)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        lafs, _, descs = feat(data["image0"])
        data["lafs0"] = lafs
        data["descriptors0"] = descs
        lafs2, _, descs2 = feat(data["image1"])
        data["lafs1"] = lafs2
        data["descriptors1"] = descs2

        with torch.no_grad():
            out = matcher(data)
        torch.random.manual_seed(0)
        homography, inliers = ransac(out['keypoints0'], out['keypoints1'])
        assert (inliers.sum().item() > 50)  # we have enough inliers
        # Reprojection error of 5px is OK
        assert_close(
            transform_points(homography[None], pts_src[None]),
            pts_dst[None],
            rtol=5e-2,
            atol=5)

    def test_real_gftt(self, device, dtype, data):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(2000),
                                      DescriptorMatcher('snn', 0.8)).to(device, dtype)
        ransac = RANSAC('homography', 1.0, 2048, 10).to(device, dtype)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        pts_src = data['pts0'].to(device, dtype)
        pts_dst = data['pts1'].to(device, dtype)
        with torch.no_grad():
            out = matcher(data)
        torch.random.manual_seed(0)
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
        inputs = {"image0": patches, "image1": patches2x}
        model = LocalFeatureMatcher(ScaleSpaceDetector(50),
                                    SIFTDescriptor(32),
                                    DescriptorMatcher('snn', 0.8)).to(device).eval()
        model_jit = torch.jit.script(model)

        out = model(inputs)
        out_jit = model_jit(inputs)
        for k, v in out.items():
            assert_close(v, out_jit[k])
