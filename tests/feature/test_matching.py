# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

from kornia.core._compat import torch_version_le
from kornia.feature import matching
from kornia.feature.integrated import LightGlueMatcher
from kornia.feature.laf import laf_from_center_scale_ori
from kornia.feature.matching import (
    DescriptorMatcher,
    DescriptorMatcherWithSteerer,
    GeometryAwareDescriptorMatcher,
    _cdist,
    match_adalam,
    match_fginn,
    match_mnn,
    match_nn,
    match_smnn,
    match_snn,
)
from kornia.feature.steerers import DiscreteSteerer

from testing.base import BaseTester, supports_matmul
from testing.casts import dict_to


class TestMatchNN(BaseTester):
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
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)

        dists1, idxs1 = match_nn(desc1, desc2)
        self.assert_close(dists1, expected_dists)
        self.assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device, dtype=torch.float64)
        desc2 = torch.rand(7, 8, device=device, dtype=torch.float64)
        self.gradcheck(match_mnn, (desc1, desc2), nondet_tol=1e-4)


class TestMatchMNN(BaseTester):
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
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher("mnn").to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        self.assert_close(dists1, expected_dists)
        self.assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device, dtype=torch.float64)
        desc2 = torch.rand(7, 8, device=device, dtype=torch.float64)
        self.gradcheck(match_mnn, (desc1, desc2), nondet_tol=1e-4)


class TestMatchSNN(BaseTester):
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
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher("snn", 0.8).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        self.assert_close(dists1, expected_dists)
        self.assert_close(idxs1, expected_idx)

    def test_matching2(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        dists, idxs = match_snn(desc1, desc2, 0.1)
        expected_dists = torch.tensor([0.0, 0, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [3, 1], [4, 0]], device=device)
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher("snn", 0.1).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        self.assert_close(dists1, expected_dists)
        self.assert_close(idxs1, expected_idx)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device, dtype=torch.float64)
        desc2 = torch.rand(7, 8, device=device, dtype=torch.float64)
        self.gradcheck(match_snn, (desc1, desc2, 0.8), nondet_tol=1e-4)


class TestMatchSMNN(BaseTester):
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
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher("smnn", 0.8).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        self.assert_close(dists1, expected_dists)
        self.assert_close(idxs1, expected_idx)

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
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)
        matcher = DescriptorMatcher("smnn", 0.1).to(device)
        dists1, idxs1 = matcher(desc1, desc2)
        self.assert_close(dists1, expected_dists)
        self.assert_close(idxs1, expected_idx)

    @pytest.mark.parametrize(
        "match_type, d1, d2",
        [
            ("nn", 0, 10),
            ("nn", 10, 0),
            ("nn", 0, 0),
            ("snn", 0, 10),
            ("snn", 10, 0),
            ("snn", 0, 0),
            ("mnn", 0, 10),
            ("mnn", 10, 0),
            ("mnn", 0, 0),
            ("smnn", 0, 10),
            ("smnn", 10, 0),
            ("smnn", 0, 0),
        ],
    )
    def test_empty_nocrash(self, match_type, d1, d2, device, dtype):
        desc1 = torch.empty(d1, 8, device=device, dtype=dtype)
        desc2 = torch.empty(d2, 8, device=device, dtype=dtype)
        matcher = DescriptorMatcher(match_type, 0.8).to(device)
        dists, idxs = matcher(desc1, desc2)
        assert dists is not None
        assert idxs is not None

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device, dtype=torch.float64)
        desc2 = torch.rand(7, 8, device=device, dtype=torch.float64)
        matcher = DescriptorMatcher("smnn", 0.8).to(device)
        self.gradcheck(match_smnn, (desc1, desc2, 0.8), nondet_tol=1e-4)
        self.gradcheck(matcher, (desc1, desc2), nondet_tol=1e-4)

    @pytest.mark.parametrize("match_type", ["nn", "snn", "mnn", "smnn"])
    def test_jit(self, match_type, device, dtype):
        desc1 = torch.rand(5, 8, device=device, dtype=dtype)
        desc2 = torch.rand(7, 8, device=device, dtype=dtype)
        matcher = DescriptorMatcher(match_type, 0.8).to(device)
        matcher_jit = torch.jit.script(DescriptorMatcher(match_type, 0.8).to(device))
        self.assert_close(matcher(desc1, desc2)[0], matcher_jit(desc1, desc2)[0])
        self.assert_close(matcher(desc1, desc2)[1], matcher_jit(desc1, desc2)[1])


class TestMatchFGINN(BaseTester):
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

    def test_matching1(self, device, dtype):
        desc1 = torch.tensor([[0, 0.0], [1, 1.001], [2, 2], [3, 3.0], [5, 5.0]], dtype=dtype, device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1.001], [0, 0.0]], dtype=dtype, device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 0.01)
        expected_dists = torch.tensor([0, 0, 0.3536, 0, 0], dtype=dtype, device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        self.assert_close(dists, expected_dists, rtol=0.001, atol=1e-3)
        self.assert_close(idxs, expected_idx)
        matcher = GeometryAwareDescriptorMatcher("fginn", {"spatial_th": 0.01}).to(device)
        dists1, idxs1 = matcher(desc1, desc2, lafs1, lafs2)
        self.assert_close(dists1, expected_dists, rtol=0.001, atol=1e-3)
        self.assert_close(idxs1, expected_idx)

    def test_matching_mutual(self, device, dtype):
        desc1 = torch.tensor([[0, 0.1], [1, 1.001], [2, 2], [3, 3.0], [5, 5.0], [0.0, 0]], dtype=dtype, device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1.001], [0, 0.0]], dtype=dtype, device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 2.0, mutual=True)
        expected_dists = torch.tensor([0, 0.1768, 0, 0, 0], dtype=dtype, device=device).view(-1, 1)
        expected_idx = torch.tensor([[1, 3], [2, 2], [3, 1], [4, 0], [5, 4]], device=device)
        self.assert_close(dists, expected_dists, rtol=0.001, atol=1e-3)
        self.assert_close(idxs, expected_idx)
        matcher = GeometryAwareDescriptorMatcher("fginn", {"spatial_th": 2.0, "mutual": True}).to(device)
        dists1, idxs1 = matcher(desc1, desc2, lafs1, lafs2)
        self.assert_close(dists1, expected_dists, rtol=0.001, atol=1e-3)
        self.assert_close(idxs1, expected_idx)

    def test_nomatch(self, device, dtype):
        desc1 = torch.tensor([[0, 0.0]], dtype=dtype, device=device)
        desc2 = torch.tensor([[5, 5.0]], dtype=dtype, device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8)
        assert len(dists) == 0
        assert len(idxs) == 0

    def test_matching2(self, device, dtype):
        desc1 = torch.tensor([[0, 0.0], [1, 1.001], [2, 2], [3, 3.0], [5, 5.0]], dtype=dtype, device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1.001], [0, 0.0]], dtype=dtype, device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 2.0)
        expected_dists = torch.tensor([0, 0, 0.1768, 0, 0], dtype=dtype, device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        self.assert_close(dists, expected_dists, rtol=0.001, atol=1e-3)
        self.assert_close(idxs, expected_idx)
        matcher = GeometryAwareDescriptorMatcher("fginn", {"spatial_th": 2.0}).to(device)
        dists1, idxs1 = matcher(desc1, desc2, lafs1, lafs2)
        self.assert_close(dists1, expected_dists, rtol=0.001, atol=1e-3)
        self.assert_close(idxs1, expected_idx)

    @staticmethod
    def _suppression_case(device, dtype):
        """A textbook FGINN case.

        The 2nd nearest neighbour in descriptor space sits 1 px from the 1st, so it is a repeated
        detection of the same structure and must not be used for the ratio test. The effective 2nd
        neighbour is then ``desc2[2]``, giving ``0.1 / 0.5 = 0.2`` rather than ``0.1 / 0.2 = 0.5``.
        """
        desc2 = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.5, 0.0], [0.9, 0.0]], device=device, dtype=dtype)
        xy2 = torch.tensor(
            [
                [10.0, 10.0],  # 1st NN
                [11.0, 10.0],  # 1 px away -> suppressed by the spatial check
                [50.0, 50.0],  # far -> the effective 2nd NN
                [90.0, 90.0],
            ],
            device=device,
            dtype=dtype,
        )
        lafs2 = laf_from_center_scale_ori(xy2[None], torch.ones(1, 4, 1, 1, device=device, dtype=dtype))
        return desc2, lafs2

    def test_second_neighbour_near_the_first_is_suppressed(self, device, dtype):
        desc2, lafs2 = self._suppression_case(device, dtype)
        query = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        desc1 = torch.stack([query, query])
        lafs1 = laf_from_center_scale_ori(
            torch.zeros(1, 2, 2, device=device, dtype=dtype), torch.ones(1, 2, 1, 1, device=device, dtype=dtype)
        )

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, th=0.3, spatial_th=5.0)

        expected_dists = torch.tensor([0.2, 0.2], device=device, dtype=dtype).view(-1, 1)
        expected_idx = torch.tensor([[0, 0], [1, 0]], device=device)
        self.assert_close(dists, expected_dists, rtol=1e-3, atol=1e-3)
        self.assert_close(idxs, expected_idx)

    def test_result_is_independent_of_the_other_queries(self, device, dtype):
        """A query's ratio must not depend on unrelated queries in the same batch (see #4062)."""
        desc2, lafs2 = self._suppression_case(device, dtype)
        query = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        # a decoy whose own nearest neighbour is desc2[3], i.e. a different candidate ordering
        decoy = torch.tensor([1.0, 0.0], device=device, dtype=dtype)

        ratios = []
        for first in (query, decoy):
            desc1 = torch.stack([first, query])
            lafs1 = laf_from_center_scale_ori(
                torch.zeros(1, 2, 2, device=device, dtype=dtype),
                torch.ones(1, 2, 1, 1, device=device, dtype=dtype),
            )
            dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, th=0.3, spatial_th=5.0)
            hit = dists[idxs[:, 0] == 1]
            assert len(hit) == 1, "the second query must match regardless of the first"
            ratios.append(hit[0, 0])

        self.assert_close(ratios[0], ratios[1], rtol=1e-4, atol=1e-4)
        self.assert_close(ratios[1], torch.tensor(0.2, device=device, dtype=dtype), rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        desc1 = torch.rand(5, 8, device=device, dtype=torch.float64)
        desc2 = torch.rand(7, 8, device=device, dtype=torch.float64)
        center1 = torch.rand(1, 5, 2, device=device, dtype=torch.float64)
        center2 = torch.rand(1, 7, 2, device=device, dtype=torch.float64)
        lafs1 = laf_from_center_scale_ori(center1)
        lafs2 = laf_from_center_scale_ori(center2)
        self.gradcheck(match_fginn, (desc1, desc2, lafs1, lafs2, 0.8, 0.05), nondet_tol=1e-4)

    @pytest.mark.skip("keyword-arg expansion is not supported")
    def test_jit(self, device, dtype):
        desc1 = torch.rand(5, 8, device=device, dtype=dtype)
        desc2 = torch.rand(7, 8, device=device, dtype=dtype)
        center1 = torch.rand(1, 5, 2, device=device)
        center2 = torch.rand(1, 7, 2, device=device)
        lafs1 = laf_from_center_scale_ori(center1)
        lafs2 = laf_from_center_scale_ori(center2)
        matcher = GeometryAwareDescriptorMatcher("fginn", 0.8).to(device)
        matcher_jit = torch.jit.script(GeometryAwareDescriptorMatcher("fginn", 0.8).to(device))
        self.assert_close(matcher(desc1, desc2)[0], matcher_jit(desc1, desc2, lafs1, lafs2)[0])
        self.assert_close(matcher(desc1, desc2)[1], matcher_jit(desc1, desc2, lafs1, lafs2)[1])


class TestAdalam(BaseTester):
    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["adalam_idxs"], indirect=True)
    def test_real(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            dists, idxs = match_adalam(data_dev["descs1"], data_dev["descs2"], data_dev["lafs1"], data_dev["lafs2"])
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= data_dev["descs1"].shape[0]
        assert dists.shape[0] <= data_dev["descs2"].shape[0]
        expected_idxs = data_dev["expected_idxs"].long()
        self.assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["adalam_idxs"], indirect=True)
    def test_single_nocrash(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            _dists, _idxs = match_adalam(
                data_dev["descs1"], data_dev["descs2"][:1], data_dev["lafs1"], data_dev["lafs2"][:, :1]
            )
            _dists, _idxs = match_adalam(
                data_dev["descs1"][:1], data_dev["descs2"], data_dev["lafs1"][:, :1], data_dev["lafs2"]
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["adalam_idxs"], indirect=True)
    def test_small_user_conf(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        adalam_config = {"device": device}
        with torch.no_grad():
            _dists, _idxs = match_adalam(
                data_dev["descs1"], data_dev["descs2"][:1], data_dev["lafs1"], data_dev["lafs2"][:, :1]
            )
            _dists, _idxs = match_adalam(
                data_dev["descs1"], data_dev["descs2"], data_dev["lafs1"], data_dev["lafs2"], config=adalam_config
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["adalam_idxs"], indirect=True)
    def test_empty_nocrash(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            _dists, _idxs = match_adalam(
                data_dev["descs1"],
                torch.empty(0, 128, device=device, dtype=dtype),
                data_dev["lafs1"],
                torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
            )
            _dists, _idxs = match_adalam(
                torch.empty(0, 128, device=device, dtype=dtype),
                data_dev["descs2"],
                torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
                data_dev["lafs2"],
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["adalam_idxs"], indirect=True)
    def test_small(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            _dists, _idxs = match_adalam(
                data_dev["descs1"][:4], data_dev["descs2"][:4], data_dev["lafs1"][:, :4], data_dev["lafs2"][:, :4]
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["adalam_idxs"], indirect=True)
    def test_seeds_fail(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        with torch.no_grad():
            _dists, _idxs = match_adalam(
                data_dev["descs1"][:100],
                data_dev["descs2"][:100],
                data_dev["lafs1"][:, :100],
                data_dev["lafs2"][:, :100],
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["adalam_idxs"], indirect=True)
    def test_module(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        matcher = GeometryAwareDescriptorMatcher("adalam", {"device": device}).to(device, dtype)
        with torch.no_grad():
            dists, idxs = matcher(data_dev["descs1"], data_dev["descs2"], data_dev["lafs1"], data_dev["lafs2"])
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= data_dev["descs1"].shape[0]
        assert dists.shape[0] <= data_dev["descs2"].shape[0]
        expected_idxs = data_dev["expected_idxs"].long()
        self.assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)


class TestLightGlueDISK(BaseTester):
    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason="Needs autocast")
    @pytest.mark.parametrize("data", ["lightglue_idxs"], indirect=True)
    def test_real(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        config = {"depth_confidence": -1, "width_confidence": -1}
        lg = LightGlueMatcher("disk", config).to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            dists, idxs = lg(data_dev["descs1"], data_dev["descs2"], data_dev["lafs1"], data_dev["lafs2"])
        assert idxs.shape[1] == 2
        assert dists.shape[1] == 1
        assert idxs.shape[0] == dists.shape[0]
        assert dists.shape[0] <= data_dev["descs1"].shape[0]
        assert dists.shape[0] <= data_dev["descs2"].shape[0]
        if device.type == "cpu":
            expected_idxs = data_dev["lightglue_disk_idxs"].long()
            self.assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["lightglue_idxs"], indirect=True)
    def test_single_nocrash(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        lg = LightGlueMatcher("disk").to(device, dtype).eval()
        with torch.no_grad():
            _dists, _idxs = lg(data_dev["descs1"], data_dev["descs2"][:1], data_dev["lafs1"], data_dev["lafs2"][:, :1])
            _dists, _idxs = lg(data_dev["descs1"][:1], data_dev["descs2"], data_dev["lafs1"][:, :1], data_dev["lafs2"])

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["lightglue_idxs"], indirect=True)
    def test_empty_nocrash(self, device, dtype, data):
        torch.random.manual_seed(0)
        # This is not unit test, but that is quite good integration test
        data_dev = dict_to(data, device, dtype)
        lg = LightGlueMatcher("disk").to(device, dtype).eval()
        with torch.no_grad():
            _dists, _idxs = lg(
                data_dev["descs1"],
                torch.empty(0, 256, device=device, dtype=dtype),
                data_dev["lafs1"],
                torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
            )
            _dists, _idxs = lg(
                torch.empty(0, 256, device=device, dtype=dtype),
                data_dev["descs2"],
                torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
                data_dev["lafs2"],
            )


class TestLightGlueHardNet(BaseTester):
    def test_smoke(self):
        lg = LightGlueMatcher("doghardnet")
        assert isinstance(lg, LightGlueMatcher)


class TestLightGlueMatcherOrientations(BaseTester):
    @staticmethod
    def _record_lightglue_inputs(monkeypatch, device):
        seen = {}

        class _RecordingLightGlue(torch.nn.Module):
            # Stands in for the network, whose construction downloads weights; the matcher only
            # prepares its inputs.
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, data):
                seen.update(data)
                matches = torch.full((1, data["image0"]["keypoints"].shape[1]), -1, device=device)
                return {"matches0": matches, "matching_scores0": torch.zeros_like(matches, dtype=torch.float32)}

        monkeypatch.setattr("kornia.feature.integrated.LightGlue", _RecordingLightGlue)
        return seen

    def test_float64_orientations_wrap_by_full_precision_two_pi_5127(self, device, monkeypatch):
        if device.type == "mps":
            pytest.skip("float64 is unavailable on MPS")
        seen = self._record_lightglue_inputs(monkeypatch, device)
        degrees = torch.tensor([[-90.0, -45.0, 30.0], [-135.0, 60.0, -30.0]], device=device, dtype=torch.float64)
        xy = torch.rand(2, 3, 2, device=device, dtype=torch.float64) * 50
        scale = torch.full((2, 3, 1, 1), 4.0, device=device, dtype=torch.float64)
        lafs = laf_from_center_scale_ori(xy, scale, degrees.unsqueeze(-1))
        descriptors = torch.rand(3, 128, device=device, dtype=torch.float64)
        LightGlueMatcher("disk")(descriptors, descriptors, lafs[:1], lafs[1:])
        # Negative angles move into [0, 2 pi); the float32 pi added 1.7e-7 rad too much.
        expected = torch.remainder(torch.deg2rad(degrees), 2 * torch.pi)
        self.assert_close(seen["image0"]["oris"], expected[:1], rtol=0.0, atol=1e-12)
        self.assert_close(seen["image1"]["oris"], expected[1:], rtol=0.0, atol=1e-12)

    def test_sift_passes_the_training_keypoint_convention(self, device, dtype, monkeypatch):
        seen = self._record_lightglue_inputs(monkeypatch, device)
        degrees = torch.tensor([[30.0, -90.0, 180.0]], device=device, dtype=dtype)
        xy = torch.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]], device=device, dtype=dtype)
        scale = torch.full((1, 3, 1, 1), 12.0, device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy, scale, degrees.unsqueeze(-1))
        descriptors = torch.rand(3, 128, device=device, dtype=dtype)
        LightGlueMatcher("sift")(descriptors, descriptors, lafs, lafs)
        # The negated orientation in (-pi, pi], and sigma of the 12-pixel 6-sigma frame.
        expected = torch.tensor([[-torch.pi / 6, torch.pi / 2, torch.pi]], device=device, dtype=dtype)
        for image in ("image0", "image1"):
            self.assert_close(seen[image]["oris"], expected)
            self.assert_close(seen[image]["scales"], torch.full((1, 3), 2.0, device=device, dtype=dtype))


class TestMatchSteererGlobal(BaseTester):
    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(1, 4, 4), (2, 5, 128), (6, 2, 32), (32, 32, 8)])
    @pytest.mark.parametrize("matching_mode", ["nn", "mnn", "snn", "smnn"])
    @pytest.mark.parametrize("fast", [False, True])
    def test_shape(self, num_desc1, num_desc2, dim, matching_mode, fast, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        generator = torch.rand(dim, dim, device=device)
        steerer = DiscreteSteerer(generator)
        desc2 = steerer(desc1)

        matcher = DescriptorMatcherWithSteerer(
            steerer=steerer, steerer_order=3, steer_mode="global", match_mode=matching_mode
        )

        dists, idxs, _num_rot = matcher(
            desc1,
            desc2,
            subset_size=max(1, min(num_desc1 // 2, num_desc2 // 2)) if fast else None,
        )
        assert dists.shape[1] == 1
        assert dists.shape[0] <= num_desc1
        assert idxs.shape[1] == 2
        assert idxs.shape[0] == dists.shape[0]

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_normalize_is_finite_on_a_zero_descriptor(self, device, desc_dtype):
        if not supports_matmul(device, desc_dtype):
            # The half-precision `_cdist` fallback multiplies the descriptors, and the steerer
            # steers through `F.linear`; torch 2.1.2 has no float16 CPU `addmm` kernel.
            pytest.skip(f"no matmul kernel for {desc_dtype} on {device.type}")
        # `F.normalize`'s default eps underflows in float16, so a zero row -- the descriptor of a
        # padded slot -- became NaN, and `cdist` then poisoned its whole row and column.
        torch.manual_seed(0)
        desc1 = torch.rand(5, 8, device=device, dtype=desc_dtype)
        desc2 = torch.rand(6, 8, device=device, dtype=desc_dtype)
        desc1[1] = 0
        desc2[4] = 0
        # torch 2.2.2 and older have no bfloat16 CPU `eye` kernel. 0 and 1 are exact in every
        # float dtype, so building in float32 and casting gives the identical matrix.
        steerer = DiscreteSteerer(torch.eye(8, device=device, dtype=torch.float32).to(desc_dtype))
        matcher = DescriptorMatcherWithSteerer(steerer=steerer, steerer_order=2, steer_mode="global", match_mode="mnn")
        dists, idxs, _ = matcher(desc1, desc2, normalize=True)
        assert torch.isfinite(dists).all()
        assert idxs.shape[0] >= 1

    def test_matching(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        # rotate desc2 270 deg anti-clockwise
        desc2 = desc2[:, [1, 0]]
        desc2[:, 0] = -desc2[:, 0]

        generator = torch.tensor([[0.0, 1], [-1, 0]], device=device)
        steerer = DiscreteSteerer(generator)
        matcher = DescriptorMatcherWithSteerer(steerer=steerer, steerer_order=4, steer_mode="global", match_mode="mnn")

        dists, idxs, num_rot = matcher(desc1, desc2)
        expected_dists = torch.tensor([0, 0, 0.5, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)

        assert num_rot == 3


class TestMatchSteererLocal(BaseTester):
    @pytest.mark.parametrize("num_desc1, num_desc2, dim", [(1, 4, 4), (2, 5, 128), (6, 2, 32)])
    def test_shape(self, num_desc1, num_desc2, dim, device):
        desc1 = torch.rand(num_desc1, dim, device=device)
        generator = torch.rand(dim, dim, device=device)
        steerer = DiscreteSteerer(generator)
        desc2 = steerer(desc1)
        desc2[:1] = steerer(desc2[:1])

        matcher = DescriptorMatcherWithSteerer(steerer=steerer, steerer_order=3, steer_mode="local", match_mode="mnn")

        dists, idxs, _num_rot = matcher(desc1, desc2)
        assert dists.shape[1] == 1
        assert idxs.shape == (dists.shape[0], 2)
        assert dists.shape[0] == num_desc1

    def test_matching(self, device):
        desc1 = torch.tensor([[0, 0.0], [1, 1], [2, 2], [3, 3.0], [5, 5.0]], device=device)
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.3, 2.4], [1, 1], [0, 0.0]], device=device)

        # rotate second to last element of desc2 90 deg anti-clockwise
        desc2[-2] = desc2[-2, [1, 0]]
        desc2[-2, 1] = -desc2[-2, 1]

        # rotate first two elements of desc2 270 deg anti-clockwise
        desc2[:2] = desc2[:2, [1, 0]]
        desc2[:2, 0] = -desc2[:2, 0]

        generator = torch.tensor([[0.0, 1], [-1, 0]], device=device)
        steerer = DiscreteSteerer(generator)
        matcher = DescriptorMatcherWithSteerer(steerer=steerer, steerer_order=4, steer_mode="local", match_mode="mnn")

        dists, idxs, num_rot = matcher(desc1, desc2)
        expected_dists = torch.tensor([0, 0, 0.5, 0, 0], device=device).view(-1, 1)
        expected_idx = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], device=device)
        self.assert_close(dists, expected_dists)
        self.assert_close(idxs, expected_idx)

        assert num_rot is None


class TestCDist(BaseTester):
    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_zero_distance_gradient(self, device, desc_dtype):
        if not supports_matmul(device, desc_dtype):
            pytest.skip(f"no matmul kernel for {desc_dtype} on {device.type}")
        d1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]], device=device, dtype=desc_dtype, requires_grad=True)
        d2 = torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]], device=device, dtype=desc_dtype, requires_grad=True)
        dists = _cdist(d1, d2)
        assert torch.isfinite(dists).all()
        assert dists[0, 0] == 0.0
        assert dists[2, 0] == 0.0
        assert dists[1, 2] == 0.0
        loss = dists.sum()
        loss.backward()
        assert d1.grad is not None and torch.isfinite(d1.grad).all()
        assert d2.grad is not None and torch.isfinite(d2.grad).all()


class TestMatchSMNNBatched(BaseTester):
    def test_large_common_descriptor_offset(self, device):
        a = torch.tensor([[[10000.0, 10000.0], [10001.0, 10000.0], [10000.0, 10002.0]]], device=device)
        b = a.flip(1)
        ratios, indices = matching.match_smnn_batched(a, b)
        reference_ratio = torch.zeros(3, 1, device=device)
        reference_index = torch.tensor([[0, 2], [1, 1], [2, 0]], device=device)
        self.assert_close(indices[:, 1:], reference_index)
        self.assert_close(ratios, reference_ratio)

    def test_undefined_provided_distance_ratios_have_finite_gradients(self, device, dtype):
        a = torch.zeros(1, 3, 4, device=device, dtype=dtype)
        b = torch.zeros(1, 4, 4, device=device, dtype=dtype)
        dm = torch.tensor(
            [[[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 2.0, 3.0], [2.0, 3.0, 0.1, 4.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        ratios, indices = matching.match_smnn_batched(a, b, 1.0, dm=dm)
        assert indices.tolist() == [[0, 2, 2]]
        ratios.sum().backward()
        assert torch.isfinite(dm.grad).all()

    def test_ambiguous_zero_distance_gradients(self, device, dtype):
        a = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]]], device=device, dtype=dtype, requires_grad=True)
        b = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 1.0]]], device=device, dtype=dtype, requires_grad=True)
        ratios, indices = matching.match_smnn_batched(a, b)
        assert indices.tolist() == [[0, 2, 2]]
        ratios.sum().backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()

    def test_dynamo(self, device, dtype, torch_optimizer):
        a = torch.rand(2, 3, 4, device=device, dtype=dtype)
        b = torch.rand(2, 4, 4, device=device, dtype=dtype)
        expected = matching.match_smnn_batched(a, b)
        actual = torch_optimizer(matching.match_smnn_batched)(a, b)
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])

    def test_masked_gradients_are_finite(self, device, dtype):
        a = torch.rand(2, 3, 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.rand(2, 4, 4, device=device, dtype=dtype, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, False]], device=device)
        ratios, _ = matching.match_smnn_batched(a, b, 1.0, mask1=mask)
        ratios.sum().backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        self.assert_close(a.grad[:, 2], torch.zeros_like(a.grad[:, 2]))

    @pytest.mark.parametrize("batch,n,m", [(1, 5, 7), (3, 5, 7), (2, 8, 3)])
    def test_matches_independent_pairs(self, device, dtype, batch, n, m):
        a = torch.rand(batch, n, 8, device=device, dtype=dtype)
        b = torch.rand(batch, m, 8, device=device, dtype=dtype)
        ratios, indices = matching.match_smnn_batched(a, b, 0.95)
        assert ratios.shape == (len(indices), 1)
        assert indices.shape[1] == 3
        assert indices.dtype == torch.long
        assert ratios.device == a.device and ratios.dtype == a.dtype
        for i in range(batch):
            # The new API accumulates half-precision L2 distances in float32;
            # compare to that reference rather than legacy half norm cancellation.
            if dtype in (torch.float16, torch.bfloat16):
                expected_ratio, expected_index = match_smnn(a[i].float(), b[i].float(), 0.95)
                expected_ratio = expected_ratio.to(dtype)
            else:
                expected_ratio, expected_index = match_smnn(a[i], b[i], 0.95)
            self.assert_close(indices[indices[:, 0] == i, 1:], expected_index)
            self.assert_close(ratios[indices[:, 0] == i], expected_ratio)

    def test_masks_preserve_original_indices(self, device, dtype):
        a = torch.tensor(
            [[[0.0, 0.0], [10.0, 10.0], [1.0, 2.0], [2.0, 1.0]], [[0.0, 0.0], [0.0, 0.0], [3.0, 3.0], [4.0, 4.0]]],
            device=device,
            dtype=dtype,
        )
        b = torch.tensor(
            [
                [[2.0, 1.0], [0.0, 0.0], [1.0, 2.0], [10.0, 10.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            device=device,
            dtype=dtype,
        )
        mask_a = torch.tensor([[True, False, True, True], [False, False, True, True]], device=device)
        mask_b = torch.tensor([[True, True, True, False, False], [False, True, True, False, False]], device=device)
        ratios, indices = matching.match_smnn_batched(a, b, 0.95, mask1=mask_a, mask2=mask_b)
        for i in range(2):
            valid_a, valid_b = mask_a[i].nonzero().flatten(), mask_b[i].nonzero().flatten()
            r, ix = match_smnn(a[i, valid_a], b[i, valid_b], 0.95)
            original = torch.stack((valid_a[ix[:, 0]], valid_b[ix[:, 1]]), dim=1)
            self.assert_close(indices[indices[:, 0] == i, 1:], original)
            self.assert_close(ratios[indices[:, 0] == i], r)

    def test_provided_distance_matrix_and_ties(self, device, dtype):
        a = torch.zeros(2, 3, 4, device=device, dtype=dtype)
        b = torch.zeros(2, 4, 4, device=device, dtype=dtype)
        dm = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [2.0, 0.1, 3.0, 4.0], [3.0, 4.0, 0.2, 5.0]],
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 2.0, 3.0], [2.0, 3.0, 0.1, 4.0]],
            ],
            device=device,
            dtype=dtype,
        )
        ratios, indices = matching.match_smnn_batched(a, b, 0.95, dm)
        for i in range(2):
            r, ix = match_smnn(a[i], b[i], 0.95, dm[i])
            self.assert_close(indices[indices[:, 0] == i, 1:], ix)
            self.assert_close(ratios[indices[:, 0] == i], r)

    @pytest.mark.parametrize("batch,n,m", [(0, 3, 4), (2, 0, 4), (2, 3, 0), (2, 1, 4), (2, 4, 1)])
    def test_empty_and_short(self, device, dtype, batch, n, m):
        a = torch.empty(batch, n, 4, device=device, dtype=dtype)
        b = torch.empty(batch, m, 4, device=device, dtype=dtype)
        r, ix = matching.match_smnn_batched(a, b)
        assert r.shape == (0, 1) and ix.shape == (0, 3)
        assert r.dtype == dtype and ix.device == a.device

    def test_short_valid_rows_return_no_matches(self, device, dtype):
        a = torch.rand(2, 3, 4, device=device, dtype=dtype)
        b = torch.rand(2, 4, 4, device=device, dtype=dtype)
        mask = torch.tensor([[True, False, False], [False, False, False]], device=device)
        r, ix = matching.match_smnn_batched(a, b, mask1=mask)
        assert r.shape == (0, 1) and ix.shape == (0, 3)

    @pytest.mark.parametrize("problem", ["batch", "dim", "mask_shape", "mask_dtype", "dm_shape"])
    def test_invalid_inputs(self, device, dtype, problem):
        a = torch.rand(2, 3, 4, device=device, dtype=dtype)
        b = torch.rand(2, 5, 4, device=device, dtype=dtype)
        kwargs = {}
        if problem == "batch":
            b = b[:1]
        elif problem == "dim":
            b = b[..., :3]
        elif problem == "mask_shape":
            kwargs["mask1"] = torch.ones(2, 2, dtype=torch.bool, device=device)
        elif problem == "mask_dtype":
            kwargs["mask2"] = torch.ones(2, 5, dtype=dtype, device=device)
        else:
            kwargs["dm"] = torch.rand(2, 3, 4, dtype=dtype, device=device)
        with pytest.raises(ValueError):
            matching.match_smnn_batched(a, b, **kwargs)

    def test_gradcheck(self, device):
        a = torch.rand(2, 3, 4, device=device, dtype=torch.float64)
        b = torch.rand(2, 4, 4, device=device, dtype=torch.float64)
        self.gradcheck(matching.match_smnn_batched, (a, b, 1.0))
