import pytest
import kornia.testing as utils  # test utils
import kornia

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.scale_space_detector import *


class TestScaleSpaceDetector:
    def test_shape(self):
        inp = torch.rand(1, 1, 32, 32)
        n_feats = 10
        det = ScaleSpaceDetector(n_feats)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([1, n_feats, 2, 3])
        assert resps.shape == torch.Size([1, n_feats])

    def test_shape_batch(self):
        inp = torch.rand(7, 1, 32, 32)
        n_feats = 10
        det = ScaleSpaceDetector(n_feats)
        lafs, resps = det(inp)
        assert lafs.shape == torch.Size([7, n_feats, 2, 3])
        assert resps.shape == torch.Size([7, n_feats])

    def test_print(self):
        sift = ScaleSpaceDetector()
        sift.__repr__()

    def test_toy(self):
        inp = torch.zeros(1, 1, 33, 33)
        inp[:, :, 13:-13, 13:-13] = 1.0
        n_feats = 1
        det = ScaleSpaceDetector(n_feats,
                                 resp_module=kornia.feature.BlobHessian(),
                                 mr_size=3.0)
        lafs, resps = det(inp)
        expected_laf = torch.tensor([[[[6.0548, 0.0000, 16.0], [0.0, 6.0548, 16.0]]]])
        expected_resp = torch.tensor([[0.0806]])
        assert_allclose(expected_laf, lafs)
        assert_allclose(expected_resp, resps)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 1, 31, 21
        patches = torch.rand(batch_size, channels, height, width)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(ScaleSpaceDetector(2), (patches),
                         raise_exception=True)
