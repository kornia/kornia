import pytest
import torch
import kornia.testing as utils  # test utils
import kornia

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.scale_space_detector import *


class TestScaleSpaceDetector:
    def test_shape(self, device):
        inp = torch.rand(1, 1, 32, 32, device=device)
        n_feats = 10
        det = ScaleSpaceDetector(n_feats).to(device)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, n_feats, 2, 3])
        assert resps.shape == torch.Size([1, n_feats])

    def test_shape_batch(self, device):
        inp = torch.rand(7, 1, 32, 32, device=device)
        n_feats = 10
        det = ScaleSpaceDetector(n_feats).to(device)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([7, n_feats, 2, 3])
        assert resps.shape == torch.Size([7, n_feats])

    def test_print(self, device):
        sift = ScaleSpaceDetector()
        sift.__repr__()

    def test_toy(self, device):
        inp = torch.zeros(1, 1, 33, 33, device=device)
        inp[:, :, 13:-13, 13:-13] = 1.0
        n_feats = 1
        det = ScaleSpaceDetector(n_feats,
                                 resp_module=kornia.feature.BlobHessian(),
                                 mr_size=3.0).to(device)
        lafs, resps = det(inp)
        expected_laf = torch.tensor([[[[9.5823, 0.0000, 16.0], [0.0, 9.5823, 16.0]]]], device=device)
        expected_resp = torch.tensor([[0.0857]], device=device)
        assert_allclose(lafs, expected_laf, rtol=0.001, atol=1e-03)
        assert_allclose(resps, expected_resp, rtol=0.001, atol=1e-03)

    def test_toy_mask(self, device):
        if "cuda" in str(device):
            pytest.skip("this cuda test is broken")

        inp = torch.zeros(1, 1, 33, 33, device=device)
        inp[:, :, 13:-13, 13:-13] = 1.0

        mask = torch.zeros(1, 1, 33, 33, device=device)
        mask[:, :, 1:-1, 3:-3] = 1.0

        n_feats = 1
        det = ScaleSpaceDetector(n_feats,
                                 resp_module=kornia.feature.BlobHessian(),
                                 mr_size=3.0).to(device)
        lafs, resps = det(inp, mask)
        expected_laf = torch.tensor([[[[9.5823, 0.0000, 16.0], [0.0, 9.5823, 16.0]]]], device=device)
        expected_resp = torch.tensor([[0.0857]], device=device)
        assert_allclose(lafs, expected_laf, rtol=0.001, atol=1e-03)
        assert_allclose(resps, expected_resp, rtol=0.001, atol=1e-03)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 31, 21
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ScaleSpaceDetector(2).to(device), patches,
                         raise_exception=True, nondet_tol=1e-4)
