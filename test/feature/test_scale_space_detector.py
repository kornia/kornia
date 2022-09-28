from __future__ import annotations

import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.testing import assert_close


class TestScaleSpaceDetector:
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
        expected_laf = torch.tensor([[[[9.5823, 0.0000, 16.0], [0.0, 9.5823, 16.0]]]], device=device, dtype=dtype)
        expected_resp = torch.tensor([[0.0857]], device=device, dtype=dtype)
        assert_close(lafs, expected_laf, rtol=0.001, atol=1e-03)
        assert_close(resps, expected_resp, rtol=0.001, atol=1e-03)

    def test_toy_mask(self, device, dtype):
        inp = torch.zeros(1, 1, 33, 33, device=device, dtype=dtype)
        inp[:, :, 13:-13, 13:-13] = 1.0

        mask = torch.zeros(1, 1, 33, 33, device=device, dtype=dtype)
        mask[:, :, 1:-1, 3:-3] = 1.0

        n_feats = 1
        det = ScaleSpaceDetector(n_feats, resp_module=kornia.feature.BlobHessian(), mr_size=3.0).to(device, dtype)
        lafs, resps = det(inp, mask)
        expected_laf = torch.tensor([[[[9.5823, 0.0000, 16.0], [0.0, 9.5823, 16.0]]]], device=device, dtype=dtype)
        expected_resp = torch.tensor([[0.0857]], device=device, dtype=dtype)
        assert_close(lafs, expected_laf, rtol=0.001, atol=1e-03)
        assert_close(resps, expected_resp, rtol=0.001, atol=1e-03)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 7, 7
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ScaleSpaceDetector(2).to(device), patches, raise_exception=True, nondet_tol=1e-4)
