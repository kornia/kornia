import pytest
import torch
import kornia.testing as utils  # test utils
import kornia
from test.common import device

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
        expected_laf = torch.tensor([[[[6.0548, 0.0000, 16.0], [0.0, 6.0548, 16.0]]]], device=device)
        expected_resp = torch.tensor([[0.0806]], device=device)
        assert_allclose(expected_laf, lafs)
        assert_allclose(expected_resp, resps)

    def test_toy_strict_maxima(self, device):
        inp = torch.zeros(1, 1, 22, 22, device=device)
        inp[:, :, 8:-8, 8:-8] = 1.0
        n_feats = 4
        nms = kornia.geometry.ConvSoftArgmax3d(kernel_size=(3, 3, 3),  # nms windows size (scale, height, width)
                                               stride=(1, 1, 1),  # stride (scale, height, width)
                                               padding=(1, 1, 1),
                                               temperature=0.1,
                                               strict_maxima_bonus=1.)
        det = ScaleSpaceDetector(n_feats,
                                 resp_module=kornia.feature.CornerHarris(0.04),
                                 nms_module=nms,
                                 mr_size=1.0).to(device)
        lafs, resps = det(inp)
        expected_laf = torch.tensor([[[[1.7385, 0.0000, 11.9998],
                                       [-0.0000, 1.7385, 9.0002]],
                                      [[1.7385, 0.0000, 9.0002],
                                       [-0.0000, 1.7385, 9.0002]],
                                      [[1.7385, 0.0000, 9.0002],
                                       [-0.0000, 1.7385, 11.9998]],
                                      [[1.7385, 0.0000, 11.9998],
                                       [-0.0000, 1.7385, 11.9998]]]], device=device)
        expected_resp = torch.tensor([[0.0012, 0.0012, 0.0012, 0.0012]], device=device)
        assert_allclose(expected_laf, lafs)
        assert_allclose(expected_resp, resps)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 31, 21
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ScaleSpaceDetector(2).to(device), patches,
                         raise_exception=True, nondet_tol=1e-4)
