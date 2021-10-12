import pytest
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature.matching import *
from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.feature import SIFTDescriptor
from kornia.geometry import resize
from kornia.testing import assert_close


class TestMatchNN:
    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(1, 4, 4), (2, 5, 128), (6, 2, 32)])
    def test_shape(self, num_desc1, num_desc2, dim, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        desc2 = torch.rand(num_desc2, dim, device=device)

        dists, idxs = match_nn(desc1, desc2)
        assert idxs.shape == (num_desc1, 2)
        assert dists.shape == (num_desc1, 1)

    def test_matching(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        dists, idxs = match_nn(desc1, desc2)
        expected_dists = torch.tensor([0, 0, 0.5, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('nn').to(device)
        dists1, idxs1 = match_nn(desc1, desc2)
        assert_close(dists1, expected_dists)
        assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device)
        desc2 = torch.rand(7, 8, device=device)
        desc1 = utils.tensor_to_gradcheck_var(desc1)  # to var
        desc2 = utils.tensor_to_gradcheck_var(desc2)  # to var
        assert gradcheck(match_mnn, (desc1, desc2), raise_exception=True, nondet_tol=1e-4)


class TestMatchMNN:
    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(1, 4, 4), (2, 5, 128), (6, 2, 32)])
    def test_shape(self, num_desc1, num_desc2, dim, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        desc2 = torch.rand(num_desc2, dim, device=device)

        dists, idxs = match_mnn(desc1, desc2)
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= num_desc1

    def test_matching(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        dists, idxs = match_mnn(desc1, desc2)
        expected_dists = torch.tensor([0, 0, 0.5, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('mnn').to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        assert_close(dists1, expected_dists)
        assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device)
        desc2 = torch.rand(7, 8, device=device)
        desc1 = utils.tensor_to_gradcheck_var(desc1)  # to var
        desc2 = utils.tensor_to_gradcheck_var(desc2)  # to var
        assert gradcheck(match_mnn, (desc1, desc2), raise_exception=True, nondet_tol=1e-4)


class TestMatchSNN:
    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(2, 4, 4), (2, 5, 128), (6, 2, 32)])
    def test_shape(self, num_desc1, num_desc2, dim, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        desc2 = torch.rand(num_desc2, dim, device=device)

        dists, idxs = match_snn(desc1, desc2)
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= num_desc1

    def test_matching1(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        dists, idxs = match_snn(desc1, desc2, 0.8)
        expected_dists = torch.tensor([0, 0, 0.35355339059327373, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('snn', 0.8).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        assert_close(dists1, expected_dists)
        assert_close(idxs1, expected_idx)

    def test_matching2(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        dists, idxs = match_snn(desc1, desc2, 0.1)
        expected_dists = torch.tensor([0.0, 0, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('snn', 0.1).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        assert_close(dists1, expected_dists)
        assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device)
        desc2 = torch.rand(7, 8, device=device)
        desc1 = utils.tensor_to_gradcheck_var(desc1)  # to var
        desc2 = utils.tensor_to_gradcheck_var(desc2)  # to var
        assert gradcheck(match_snn, (desc1, desc2, 0.8), raise_exception=True, nondet_tol=1e-4)


class TestMatchSMNN:
    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(2, 4, 4), (2, 5, 128), (6, 2, 32)])
    def test_shape(self, num_desc1, num_desc2, dim, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        desc2 = torch.rand(num_desc2, dim, device=device)

        dists, idxs = match_smnn(desc1, desc2, 0.8)
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= num_desc1
        assert dists.shape[0] <= num_desc2

    def test_matching1(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        dists, idxs = match_smnn(desc1, desc2, 0.8)
        expected_dists = torch.tensor([0, 0, 0.5423, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('smnn', 0.8).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        assert_close(dists1, expected_dists)
        assert_close(idxs1, expected_idx)

    def test_matching2(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        dists, idxs = match_smnn(desc1, desc2, 0.1)
        expected_dists = torch.tensor([0.0, 0, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('smnn', 0.1).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        assert_close(dists1, expected_dists)
        assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device)
        desc2 = torch.rand(7, 8, device=device)
        desc1 = utils.tensor_to_gradcheck_var(desc1)  # to var
        desc2 = utils.tensor_to_gradcheck_var(desc2)  # to var
        matcher = DescriptorMatcher('smnn', 0.8).to(device)
        assert gradcheck(match_smnn, (desc1, desc2, 0.8), raise_exception=True, nondet_tol=1e-4)
        assert gradcheck(matcher, (desc1, desc2), raise_exception=True, nondet_tol=1e-4)

    @pytest.mark.jit
    @pytest.mark.parametrize("match_type", ["nn", "snn", "mnn", "smnn"])
    def test_jit(self, match_type, device, dtype):
        desc1 = torch.rand(5, 8, device=device, dtype=dtype)
        desc2 = torch.rand(7, 8, device=device, dtype=dtype)
        matcher = DescriptorMatcher(match_type, 0.8).to(device)
        matcher_jit = torch.jit.script(DescriptorMatcher(match_type, 0.8).to(device))
        assert_close(matcher(desc1, desc2)[0], matcher_jit(desc1, desc2)[0])
        assert_close(matcher(desc1, desc2)[1], matcher_jit(desc1, desc2)[1])


class TestLocalFeatureMatcher:
    def test_smoke(self, device):
        matcher = LocalFeatureMatcher(ScaleSpaceDetector(10),
                                      SIFTDescriptor(32),
                                      DescriptorMatcher('snn', 0.8)).to(device)

    @pytest.mark.skip("Takes too long time (but works)")
    def test_gradcheck(self, device):
        matcher = LocalFeatureMatcher(ScaleSpaceDetector(4),
                                      SIFTDescriptor(8, 2, 1),
                                      DescriptorMatcher('nn', 1.0)).to(device)
        patches = torch.rand(1, 1, 32, 32, device=device)
        patches05 = resize(patches, (48, 48))
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        patches05 = utils.tensor_to_gradcheck_var(patches05)  # to var
        def proxy_forward(x, y):
            return matcher({"image0": x, "image1": y})["keypoints0"]
        assert gradcheck(proxy_forward, (patches, patches05), eps=1e-4, atol=1e-4, raise_exception=True)

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
