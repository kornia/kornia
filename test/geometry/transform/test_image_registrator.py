import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.geometry.transform import ImageRegistrator
from kornia.testing import assert_close


class TestSimilarity:
    def test_smoke(self, device, dtype):
        expected = torch.eye(3, device=device)[None]
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            sim = kornia.geometry.transform.Similarity(r, sc, sh).to(device)
            assert_close(sim(), expected, atol=1e-4, rtol=1e-4)

    def test_smoke_inverse(self, device, dtype):
        expected = torch.eye(3, device=device)[None]
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            s = kornia.geometry.transform.Similarity(r, sc, sh).to(device)
            assert_close(s.forward_inverse(), expected, atol=1e-4, rtol=1e-4)

    def test_scale(self, device, dtype):
        sc = 0.5
        sim = kornia.geometry.transform.Similarity(True, True, True).to(device)
        sim.scale.data *= sc
        expected = torch.tensor([[0.5, 0, 0.],
                                [0, 0.5, 0],
                                [0, 0, 1]], device=device)[None]
        inv_expected = torch.tensor([[2.0, 0, 0.],
                                    [0, 2.0, 0],
                                    [0, 0, 1]], device=device)[None]
        assert_close(sim.forward_inverse(), inv_expected, atol=1e-4, rtol=1e-4)
        assert_close(sim(), expected, atol=1e-4, rtol=1e-4)

    def test_repr(self, device, dtype):
        for r, sc, sh in zip([True, False], [True, False], [True, False]):
            s = kornia.geometry.transform.Similarity(r, sc, sh).to(device)
            print(s)


class TestHomography:
    def test_smoke(self, device, dtype):
        expected = torch.eye(3, device=device)[None]
        h = kornia.geometry.transform.Homography().to(device)
        assert_close(h(), expected, atol=1e-4, rtol=1e-4)

    def test_smoke_inverse(self, device, dtype):
        expected = torch.eye(3, device=device)[None]
        h = kornia.geometry.transform.Homography().to(device)
        assert_close(h.forward_inverse(), expected, atol=1e-4, rtol=1e-4)

    def test_repr(self, device, dtype):
        h = kornia.geometry.transform.Homography().to(device)
        print(h)


class TestImageRegistrator:
    def test_smoke(self, device, dtype):
        for model_type in ['homography',
                           'similarity',
                           'translation',
                           'scale',
                           'rotation']:
            ir = kornia.geometry.transform.ImageRegistrator(model_type).to(device)
            print(ir)

    def test_registration(self, device):
        ch, height, width = 3, 16, 18
        homography = torch.eye(3, device=device)[None]
        homography[..., 0, 0] = 1.05
        homography[..., 1, 1] = 1.05
        homography[..., 0, 2] = 0.01
        img_src = torch.rand(1, ch, height, width, device=device)
        img_dst = kornia.geometry.homography_warp(img_src,
                                                  homography,
                                                  (height, width),
                                                  align_corners=False)
        IR = ImageRegistrator('Similarity',
                              num_iterations=500,
                              lr=3e-4,
                              pyramid_levels=2).to(device=device)
        model = IR.register(img_src, img_dst)
        assert_close(model, homography, atol=1e-3, rtol=1e-3)
        model, intermediate = IR.register(img_src,
                                          img_dst,
                                          output_intermediate_models=True)
        assert len(intermediate) == 2
