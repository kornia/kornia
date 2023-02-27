import pytest
import torch

import kornia
import kornia.testing as utils  # test utils
from kornia.geometry import transform_points
from kornia.geometry.conversions import denormalize_homography
from kornia.geometry.transform import ImageRegistrator
from kornia.testing import assert_close


class TestSimilarity:
    def test_smoke(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            sim = kornia.geometry.transform.Similarity(r, sc, sh).to(device, dtype)
            assert_close(sim(), expected, atol=1e-4, rtol=1e-4)

    def test_smoke_inverse(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            s = kornia.geometry.transform.Similarity(r, sc, sh).to(device, dtype)
            assert_close(s.forward_inverse(), expected, atol=1e-4, rtol=1e-4)

    def test_scale(self, device, dtype):
        sc = 0.5
        sim = kornia.geometry.transform.Similarity(True, True, True).to(device, dtype)
        sim.scale.data *= sc
        expected = torch.tensor([[0.5, 0, 0.0], [0, 0.5, 0], [0, 0, 1]], device=device, dtype=dtype)[None]
        inv_expected = torch.tensor([[2.0, 0, 0.0], [0, 2.0, 0], [0, 0, 1]], device=device, dtype=dtype)[None]
        assert_close(sim.forward_inverse(), inv_expected, atol=1e-4, rtol=1e-4)
        assert_close(sim(), expected, atol=1e-4, rtol=1e-4)

    def test_repr(self, device, dtype):
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            s = kornia.geometry.transform.Similarity(r, sc, sh).to(device, dtype)
            assert s is not None


class TestHomography:
    def test_smoke(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        h = kornia.geometry.transform.Homography().to(device, dtype)
        assert_close(h(), expected, atol=1e-4, rtol=1e-4)

    def test_smoke_inverse(self, device, dtype):
        expected = torch.eye(3, device=device, dtype=dtype)[None]
        h = kornia.geometry.transform.Homography().to(device, dtype)
        assert_close(h.forward_inverse(), expected, atol=1e-4, rtol=1e-4)

    def test_repr(self, device, dtype):
        h = kornia.geometry.transform.Homography().to(device, dtype)
        assert h is not None


class TestImageRegistrator:
    @pytest.mark.parametrize("model_type", ['homography', 'similarity', 'translation', 'scale', 'rotation'])
    def test_smoke(self, device, dtype, model_type):
        ir = kornia.geometry.transform.ImageRegistrator(model_type).to(device, dtype)
        assert ir is not None

    def test_registration_toy(self, device, dtype):
        ch, height, width = 3, 16, 18
        homography = torch.eye(3, device=device, dtype=dtype)[None]
        homography[..., 0, 0] = 1.05
        homography[..., 1, 1] = 1.05
        homography[..., 0, 2] = 0.01
        img_src = torch.rand(1, ch, height, width, device=device, dtype=dtype)
        img_dst = kornia.geometry.homography_warp(img_src, homography, (height, width), align_corners=False)
        IR = ImageRegistrator('Similarity', num_iterations=500, lr=3e-4, pyramid_levels=2).to(device, dtype)
        model = IR.register(img_src, img_dst)
        assert_close(model, homography, atol=1e-3, rtol=1e-3)
        model, intermediate = IR.register(img_src, img_dst, output_intermediate_models=True)
        assert len(intermediate) == 2

    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_registration_real(self, device, dtype, data):
        data_dev = utils.dict_to(data, device, dtype)
        IR = ImageRegistrator('homography', num_iterations=1200, lr=2e-2, pyramid_levels=5).to(device, dtype)
        model = IR.register(data_dev['image0'], data_dev['image1'])
        homography_gt = torch.inverse(data_dev['H_gt'])
        homography_gt = homography_gt / homography_gt[2, 2]
        h0, w0 = data['image0'].shape[2], data['image0'].shape[3]
        h1, w1 = data['image1'].shape[2], data['image1'].shape[3]

        model_denormalized = denormalize_homography(model, (h0, w0), (h1, w1))
        model_denormalized = model_denormalized / model_denormalized[0, 2, 2]

        bbox = torch.tensor([[[0, 0], [w0, 0], [w0, h0], [0, h0]]], device=device, dtype=dtype)
        bbox_in_2_gt = transform_points(homography_gt[None], bbox)
        bbox_in_2_gt_est = transform_points(model_denormalized, bbox)
        # The tolerance is huge, because the error is in pixels
        # and transformation is quite significant, so
        # 15 px  reprojection error is not super huge
        assert_close(bbox_in_2_gt, bbox_in_2_gt_est, atol=15, rtol=0.1)
