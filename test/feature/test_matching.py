import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature.matching import DescriptorMatcher, match_mnn, match_nn, match_smnn, match_snn
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
