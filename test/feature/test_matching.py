import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature.laf import laf_from_center_scale_ori
from kornia.feature.matching import DescriptorMatcher, match_fginn, match_mnn, match_nn, match_smnn, match_snn
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

    def test_nomatch(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0]], device=device)

        dists, idxs = match_snn(desc1, desc2, 0.8)
        assert len(dists) == 0
        assert len(idxs) == 0

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

    def test_nomatch(self, device):
        desc1 = torch.tensor([[0, 0.0]], device=device)
        desc2 = torch.tensor([[5, 5.0]], device=device)

        dists, idxs = match_smnn(desc1, desc2, 0.8)
        assert len(dists) == 0
        assert len(idxs) == 0

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


class TestMatchFGINN:
    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(2, 4, 4), (2, 5, 128), (6, 2, 32)])
    def test_shape_one_way(self, num_desc1, num_desc2, dim, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        desc2 = torch.rand(num_desc2, dim, device=device)
        lafs1 = torch.rand(1, num_desc1, 2, 3, device=device)
        lafs2 = torch.rand(1, num_desc2, 2, 3, device=device)

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.9, 1000)
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= num_desc1

    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(2, 4, 4), (2, 5, 128), (6, 2, 32)])
    def test_shape_two_way(self, num_desc1, num_desc2, dim, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        desc2 = torch.rand(num_desc2, dim, device=device)
        lafs1 = torch.rand(1, num_desc1, 2, 3, device=device)
        lafs2 = torch.rand(1, num_desc2, 2, 3, device=device)

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.9, 1000, mutual=True)
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= num_desc1
        assert dists.shape[0] <= num_desc2

    def test_matching1(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 2.0)
        expected_dists = torch.tensor([0, 0, 0.3536, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists, rtol=0.001, atol=1e-3)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('fginn', 0.8, {"spatial_th": 2.0}).to(device)
        dists1, idxs1 = matcher(desc1, desc2, lafs1, lafs2)
        assert_close(dists1, expected_dists, rtol=0.001, atol=1e-3)
        assert_close(idxs1, expected_idx)

    def test_matching_mutual(self, device):
        desc1 = torch.tensor([[0, 0.1], [1, 1], [2, 2], [3, 3.0], [5, 5.0], [0.0, 0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 2.0, mutual=True)
        expected_dists = torch.tensor([0, 0.3535, 0, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[1, 3], [2, 2], [3, 1], [4, 0], [5, 4]], device=device)
        assert_close(dists, expected_dists, rtol=0.001, atol=1e-3)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('fginn', 0.8, {"spatial_th": 2.0, "mutual": True}).to(device)
        dists1, idxs1 = matcher(desc1, desc2, lafs1, lafs2)
        assert_close(dists1, expected_dists, rtol=0.001, atol=1e-3)
        assert_close(idxs1, expected_idx)

    def test_nomatch(self, device):
        desc1 = torch.tensor([[0, 0.0]], device=device)
        desc2 = torch.tensor([[5, 5.0]], device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8)
        assert len(dists) == 0
        assert len(idxs) == 0

    def test_matching2(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 0.0001)
        expected_dists = torch.tensor([0, 0, 0.3536, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        assert_close(dists, expected_dists, rtol=0.001, atol=1e-3)
        assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher('fginn', 0.8, {"spatial_th": 2.0}).to(device)
        dists1, idxs1 = matcher(desc1, desc2, lafs1, lafs2)
        assert_close(dists1, expected_dists, rtol=0.001, atol=1e-3)
        assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device)
        desc2 = torch.rand(7, 8, device=device)
        center1 = torch.rand(1, 5, 2, device=device)
        center2 = torch.rand(1, 7, 2, device=device)
        lafs1 = laf_from_center_scale_ori(center1)
        lafs2 = laf_from_center_scale_ori(center2)
        desc1 = utils.tensor_to_gradcheck_var(desc1)  # to var
        desc2 = utils.tensor_to_gradcheck_var(desc2)  # to var
        lafs1 = utils.tensor_to_gradcheck_var(lafs1)  # to var
        lafs2 = utils.tensor_to_gradcheck_var(lafs2)  # to var
        assert gradcheck(match_fginn, (desc1, desc2, lafs1, lafs2, 0.8, 0.05), raise_exception=True, nondet_tol=1e-4)

    @pytest.mark.jit
    @pytest.mark.skip("keyword-arg expansion is not supported")
    def test_jit(self, device, dtype):
        desc1 = torch.rand(5, 8, device=device, dtype=dtype)
        desc2 = torch.rand(7, 8, device=device, dtype=dtype)
        center1 = torch.rand(1, 5, 2, device=device)
        center2 = torch.rand(1, 7, 2, device=device)
        lafs1 = laf_from_center_scale_ori(center1)
        lafs2 = laf_from_center_scale_ori(center2)
        matcher = DescriptorMatcher('fginn', 0.8).to(device)
        matcher_jit = torch.jit.script(DescriptorMatcher('fginn', 0.8).to(device))
        assert_close(matcher(desc1, desc2)[0], matcher_jit(desc1, desc2, lafs1, lafs2)[0])
        assert_close(matcher(desc1, desc2)[1], matcher_jit(desc1, desc2, lafs1, lafs2)[1])
