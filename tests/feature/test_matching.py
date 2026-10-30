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

import math

import pytest
import torch

from kornia.core._compat import torch_version_le
from kornia.feature import matching
from kornia.feature.adalam import AdalamFilter, get_adalam_default_config
from kornia.feature.integrated import LightGlueMatcher, LocalFeatureMatcher
from kornia.feature.laf import get_laf_center, get_laf_orientation, get_laf_scale, laf_from_center_scale_ori
from kornia.feature.lightglue import LightGlue
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
    @pytest.mark.parametrize("mps_fallback", [False, True])
    def test_mutual_matches_keep_integer_indices(self, device, dtype, monkeypatch, mps_fallback):
        # Half precision cannot distinguish all descriptor indices: float16 rounds 2049 to
        # 2048, and bfloat16 rounds 257 to 256. Only the last pair below is mutual.
        count = 2050 if dtype == torch.float16 else 258 if dtype == torch.bfloat16 else 16
        desc = torch.zeros(count, 1, device=device, dtype=dtype)
        dm = torch.full((count, count), 10.0, device=device, dtype=dtype)
        dm[-2, -2] = 1.0
        dm[-1, -2] = 0.5
        dm[-1, -1] = 0.1
        if mps_fallback:
            monkeypatch.setattr(matching, "is_mps_tensor_safe", lambda _: True)

        ratios, indices = match_smnn(desc, desc, 0.95, dm)

        expected_indices = torch.tensor([[count - 1, count - 1]], device=device, dtype=torch.long)
        self.assert_close(indices, expected_indices)
        self.assert_close(ratios, torch.tensor([[0.2]], device=device, dtype=dtype))

    def test_reverse_ratios_follow_the_matches(self, device, dtype):
        # Reverse matches arrive in target-index order; forward matches arrive in source-index order.
        # Reject source 1 in the reverse direction and source 3 in the forward direction.
        dm = torch.tensor(
            [[4.0, 10.0, 1.0], [10.0, 1.5, 10.0], [1.0, 10.0, 10.0], [2.0, 2.0, 10.0]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        ratios, indices = match_smnn(dm[:, :1], dm[0, :, None], 0.7, dm)
        self.assert_close(indices, torch.tensor([[0, 2], [2, 0]], device=device, dtype=torch.long))
        self.assert_close(ratios, torch.tensor([[0.25], [0.5]], device=device, dtype=dtype))
        ratios.sum().backward()
        expected_grad = torch.tensor(
            [[-0.0625, 0.0, 0.25], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.25, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(dm.grad, expected_grad)

    def test_nonempty_directional_matches_have_empty_intersection(self, device, dtype):
        dm = torch.tensor([[1.0, 2.0], [0.5, 0.75]], device=device, dtype=dtype)
        # The forward ratio test keeps 0 -> 0; the reverse test keeps 0 -> 1 and 1 -> 1.
        ratios, indices = match_smnn(dm[:, :1], dm[0, :, None], 0.6, dm)
        assert ratios.shape == (0, 1)
        assert indices.shape == (0, 2)
        assert indices.dtype == torch.long

    def test_target_without_reverse_match_is_not_mutual(self, device, dtype):
        # Sources 0 and 1 both match target 0, which ties between them and has no reverse match. The last reverse
        # match (target 2 -> source 0) names source 0, and the first (target 1 -> source 1) names source 1.
        dm = torch.tensor([[1.0, 10.0, 2.0], [1.0, 3.0, 10.0], [10.0, 10.0, 10.0]], device=device, dtype=dtype)
        ratios, indices = match_smnn(dm[:, :1], dm[0, :, None], 0.7, dm)
        assert ratios.shape == (0, 1)
        assert indices.shape == (0, 2)

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
        # Exact half-precision coordinates keep the ratio independent of input rounding.
        desc2 = torch.tensor([[5, 5.0], [3, 3.0], [2.25, 2.25], [1, 1.001], [0, 0.0]], dtype=dtype, device=device)
        lafs1 = laf_from_center_scale_ori(desc1[None])
        lafs2 = laf_from_center_scale_ori(desc2[None])

        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 0.01)
        # The third query's nearest distances are sqrt(2) / 4 and sqrt(2).
        expected_dists = torch.tensor([0, 0, 0.25, 0, 0], dtype=dtype, device=device).view(-1, 1)
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


def _adalam_scene(device, H=240, W=320, n_in=400, n_out=200, n_extra2=37, D=32, seed=0):
    # n_in affine inliers (a 10 degree rotation, 1.2 scale and a shift) plus n_out outliers, in an H x W image.
    # The second image has n_extra2 more outliers, so the two keypoint counts differ.
    g = torch.Generator().manual_seed(seed)
    xy1 = torch.rand(n_in + n_out, 2, generator=g) * torch.tensor([W - 1.0, H - 1.0])
    a = math.radians(10.0)
    R = 1.2 * torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    xy2 = torch.cat(
        [
            (xy1[:n_in] - torch.tensor([160.0, 120.0])) @ R.T
            + torch.tensor([165.0, 118.0])
            + 0.7 * torch.randn(n_in, 2, generator=g),
            torch.rand(n_out + n_extra2, 2, generator=g) * torch.tensor([W - 1.0, H - 1.0]),
        ]
    )
    d1 = torch.nn.functional.normalize(torch.randn(n_in + n_out, D, generator=g), dim=1)
    noisy = d1[:n_in] + 0.15 * torch.randn(n_in, D, generator=g)
    d2 = torch.nn.functional.normalize(torch.cat([noisy, torch.randn(n_out + n_extra2, D, generator=g)]), dim=1)
    s = torch.full((1, n_in + n_out, 1, 1), 8.0)
    lafs1 = laf_from_center_scale_ori(xy1[None], s)
    lafs2 = laf_from_center_scale_ori(xy2[None], 1.2 * torch.full((1, n_in + n_out + n_extra2, 1, 1), 8.0))
    return d1.to(device), d2.to(device), lafs1.to(device), lafs2.to(device), (H, W)


class TestLightGlueMatcherSize(BaseTester):
    def test_swapped_hw_names_hw_5724(self, device):
        # LightGlueMatcher takes hw as (height, width). Its constructor downloads weights, so the matcher is built
        # around a random-weight LightGlue, enough to reach the range check; forward reads only these two attributes.
        matcher = LightGlueMatcher.__new__(LightGlueMatcher)
        torch.nn.Module.__init__(matcher)
        matcher.feature_name = "disk"
        matcher.matcher = LightGlue(
            features=None, input_dim=64, descriptor_dim=64, n_layers=2, num_heads=4, flash=False
        )
        matcher = matcher.to(device).eval()
        H, W = 40, 120
        xy = torch.tensor([[[10.0, 20.0], [100.0, 30.0], [60.0, 5.0]]], device=device)
        lafs = laf_from_center_scale_ori(xy, torch.full((1, 3, 1, 1), 4.0, device=device))
        desc = torch.rand(3, 64, device=device)
        with torch.no_grad():
            matcher(desc, desc, lafs, lafs, hw1=(H, W), hw2=(H, W))  # (height, width) is accepted
            with pytest.raises(Exception, match=r"\(height, width\) for the ``hw1``"):
                matcher(desc, desc, lafs, lafs, hw1=(W, H), hw2=(W, H))


class TestAdalamInputs(BaseTester):
    def test_dm_distances_reproduce_the_default_matches_5724(self, device):
        d1, d2, l1, l2, hw = _adalam_scene(device)
        assert len(d1) != len(d2)
        q0, i0 = match_adalam(d1, d2, l1, l2, hw1=hw, hw2=hw)
        # dm holds plain distances like the dm of the other matchers; AdaLAM squares them internally
        q1, i1 = match_adalam(d1, d2, l1, l2, hw1=hw, hw2=hw, dm=torch.cdist(d1, d2))
        assert len(i0) > 100
        assert torch.equal(i0.sort(0).values, i1.sort(0).values)
        self.assert_close(q0.sort(0).values, q1.sort(0).values, low_tolerance=True)

    def test_dm_drives_the_putative_matches_5724(self, device):
        d1, d2, l1, l2, hw = _adalam_scene(device)
        _, i0 = match_adalam(d1, d2, l1, l2, hw1=hw, hw2=hw)
        dm = torch.cdist(d1, d2)
        dm[:200] = dm[:200].flip(1)
        _, i1 = match_adalam(d1, d2, l1, l2, hw1=hw, hw2=hw, dm=dm)
        assert not torch.equal(i0, i1)
        # for the first 200 rows the flipped distances prefer a mirrored candidate, which is not an inlier
        assert len(i1) < len(i0)
        flipped_rows = i1[:, 0] < 200
        assert (i1[flipped_rows, 1] != i1[flipped_rows, 0]).all()

    def test_dm_ratio_comes_from_dm_5724(self, device):
        d1, d2, l1, l2, hw = _adalam_scene(device)
        dm = torch.cdist(d1, d2)
        _, i0 = match_adalam(d1, d2, l1, l2, hw1=hw, hw2=hw, dm=dm)
        # a constant added to the squared distances keeps every nearest neighbour and pushes every ratio to 1, so
        # fewer putative matches pass the ratio test of the seeds
        _, i1 = match_adalam(d1, d2, l1, l2, hw1=hw, hw2=hw, dm=(dm**2 + 10.0).sqrt())
        assert len(i1) < len(i0) - 100

    def test_dm_wrong_shape_5724(self, device):
        d1, d2, l1, l2, hw = _adalam_scene(device)
        with pytest.raises(Exception, match="dm must have shape"):
            match_adalam(d1, d2, l1, l2, hw1=hw, hw2=hw, dm=torch.zeros(len(d1), len(d2) - 1, device=device))
        # also when a single descriptor would end the matching early
        with pytest.raises(Exception, match="dm must have shape"):
            match_adalam(d1[:1], d2, l1[:, :1], l2, hw1=hw, hw2=hw, dm=torch.zeros(5, 5, device=device))

    def test_hw_list_equals_tuple_5724(self, device):
        d1, d2, l1, l2, (H, W) = _adalam_scene(device)
        # an image much larger than the keypoint spread changes the seed radius, so a list that is ignored differs
        tuple_out = match_adalam(d1, d2, l1, l2, hw1=(4 * H, 4 * W), hw2=(4 * H, 4 * W))
        list_out = match_adalam(d1, d2, l1, l2, hw1=[4 * H, 4 * W], hw2=[4 * H, 4 * W])
        none_out = match_adalam(d1, d2, l1, l2)
        assert len(tuple_out[1]) != len(none_out[1])
        assert torch.equal(tuple_out[1], list_out[1])

    @pytest.mark.parametrize("config", [{"ransac_iters": 128}, {}])
    def test_device_follows_input_with_config_5724(self, device, config):
        d1, d2, l1, l2, hw = _adalam_scene(device)
        quality, idxs = match_adalam(d1, d2, l1, l2, config=config, hw1=hw, hw2=hw)
        assert idxs.device == d1.device
        assert quality.device == d1.device

    def test_module_device_follows_input_5724(self, device):
        d1, d2, l1, l2, _ = _adalam_scene(device)
        quality, idxs = GeometryAwareDescriptorMatcher("adalam")(d1, d2, l1, l2)
        assert idxs.device == d1.device
        assert quality.device == d1.device

    def test_explicit_device_is_kept_5724(self, device):
        d1, d2, l1, l2, hw = _adalam_scene(device)
        _, idxs = match_adalam(d1, d2, l1, l2, config={"device": torch.device("cpu")}, hw1=hw, hw2=hw)
        assert idxs.device.type == "cpu"


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


def _sine_pattern(shape, frequency, chirp, phase, device, dtype):
    """Deterministic descriptors in ``[-1, 1]`` without the random generator, the same on every device."""
    index = torch.arange(math.prod(shape), dtype=torch.float64)
    values = torch.sin(frequency * index + chirp * (index % shape[-1]) ** 2 + phase).view(shape)
    return values.to(dtype).to(device)


class TestCDist(BaseTester):
    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("use_autocast", [False, True])
    @pytest.mark.parametrize("scripted", [False, True])
    def test_half_precision_nearby_descriptors(self, device, desc_dtype, use_autocast, scripted):
        if use_autocast and device.type not in ("cpu", "cuda"):
            pytest.skip("autocast coverage requires CPU or CUDA")
        desc1 = torch.full((1, 128), 0.0625, device=device, dtype=desc_dtype)
        desc2 = desc1.repeat(2, 1)
        desc2[0, 0] += 0.03125
        desc2[1, 0] += 0.0078125
        # Only one coordinate differs, so its absolute difference is the Euclidean distance.
        expected = torch.tensor([[0.03125, 0.0078125]], device=device, dtype=desc_dtype)
        distance_fn = torch.jit.script(_cdist) if scripted else _cdist
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=use_autocast):
            actual = distance_fn(desc1, desc2)
            distances, indices = match_nn(desc1, desc2)
            ratios, ratio_indices = match_snn(desc1, desc2)
        self.assert_close(actual, expected)
        self.assert_close(distances, expected[:, 1:])
        self.assert_close(indices, torch.tensor([[0, 1]], device=device))
        self.assert_close(ratios, torch.tensor([[0.25]], device=device, dtype=desc_dtype))
        self.assert_close(ratio_indices, indices)
        assert actual.dtype == desc_dtype

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_large_descriptors(self, device, desc_dtype):
        desc1 = torch.tensor([[300.0, 300.0]], device=device, dtype=desc_dtype, requires_grad=True)
        desc2 = torch.tensor([[304.0, 300.0]], device=device, dtype=desc_dtype, requires_grad=True)
        distances = _cdist(desc1, desc2)
        self.assert_close(distances, torch.tensor([[4.0]], device=device, dtype=desc_dtype))
        distances.sum().backward()
        self.assert_close(desc1.grad, torch.tensor([[-1.0, 0.0]], device=device, dtype=desc_dtype))
        self.assert_close(desc2.grad, torch.tensor([[1.0, 0.0]], device=device, dtype=desc_dtype))

    @pytest.mark.parametrize("desc_dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_export_branch_ignores_autocast(self, device, desc_dtype, monkeypatch):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("autocast coverage requires CPU or CUDA")
        # Export traces the manual branch; autocast must not lower its float32 matmul back to half.
        monkeypatch.setattr("kornia.feature.matching.is_exporting", lambda: True)
        desc1 = torch.full((1, 128), 0.0625, device=device, dtype=desc_dtype)
        desc2 = desc1.repeat(2, 1)
        desc2[0, 0] += 0.03125
        desc2[1, 0] += 0.0078125
        with torch.autocast(device_type=device.type):
            actual = _cdist(desc1, desc2)
        self.assert_close(actual, torch.tensor([[0.03125, 0.0078125]], device=device, dtype=desc_dtype))
        assert actual.dtype == desc_dtype

    @pytest.mark.parametrize(("dtype1", "dtype2"), [(torch.float16, torch.float32), (torch.float16, torch.bfloat16)])
    def test_mixed_dtypes_raise(self, device, dtype1, dtype2):
        # Only same-dtype half inputs are promoted; a mixed pair is not silently computed in float32.
        with pytest.raises(RuntimeError):
            _cdist(torch.ones(2, 4, device=device, dtype=dtype1), torch.ones(3, 4, device=device, dtype=dtype2))

    @pytest.mark.parametrize("exporting", [False, True])
    @pytest.mark.parametrize("rows", [10, 30])
    def test_mixed_pair_under_mixed_precision_is_computed(self, device, monkeypatch, rows, exporting):
        # Inside autocast a mixed pair of float16, bfloat16 and float32 is computed in float32, by `torch.cdist` and by
        # the fallback that export takes; more than 25 rows also send `torch.cdist` through its matrix form. Autocast
        # does not cast float64, so a pair with it raises, as outside autocast.
        if device.type not in ("cpu", "cuda", "mps"):
            pytest.skip("autocast region of CPU, CUDA and MPS only")
        monkeypatch.setattr(matching, "is_exporting", lambda: exporting)
        low = torch.bfloat16 if device.type == "cpu" else torch.float16
        d1 = _sine_pattern((rows, 16), 0.71, 0.13, 0.0, device, torch.float32)
        d2 = _sine_pattern((rows + 5, 16), 0.53, 0.29, 1.0, device, torch.float32)
        with torch.autocast(device.type, dtype=low):
            for a, b in ((d1.to(low), d2), (d1, d2.to(low))):
                actual = _cdist(a, b)
                assert actual.dtype == a.dtype
                expected = torch.cdist(a.cpu().double(), b.cpu().double())
                self.assert_close(actual.cpu().double(), expected.to(a.dtype).double(), rtol=1e-2, atol=1e-3)
            if device.type == "cpu":
                with pytest.raises(RuntimeError, match="same dtype"):
                    _cdist(d1.to(low), d2.double())

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
        assert d1.grad is not None
        assert torch.isfinite(d1.grad).all()
        assert d2.grad is not None
        assert torch.isfinite(d2.grad).all()

    @pytest.mark.parametrize("exporting", [False, True])
    def test_matches_float64_cdist(self, device, dtype, monkeypatch, exporting):
        # Export takes the matrix-multiplication fallback, as MPS does before torch 2.13. Every distance is at least
        # 2.6, where that form does not cancel; more than 25 rows also send `torch.cdist` itself through it.
        monkeypatch.setattr(matching, "is_exporting", lambda: exporting)
        desc1 = _sine_pattern((30, 16), 0.71, 0.13, 0.0, device, dtype).requires_grad_()
        desc2 = _sine_pattern((40, 16), 0.53, 0.29, 1.0, device, dtype).requires_grad_()
        reference1 = desc1.detach().cpu().double().requires_grad_()
        reference2 = desc2.detach().cpu().double().requires_grad_()
        expected = torch.cdist(reference1, reference2)
        expected_grads = torch.autograd.grad(expected.sum(), (reference1, reference2))
        actual = _cdist(desc1, desc2)
        actual_grads = torch.autograd.grad(actual.sum(), (desc1, desc2))
        self.assert_close(actual, expected.to(dtype).to(device))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_close(actual_grad, expected_grad.to(dtype).to(device))


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
        assert torch.isfinite(a.grad).all()
        assert torch.isfinite(b.grad).all()

    @pytest.mark.parametrize("masked", [False, True])
    def test_dynamo(self, device, dtype, torch_optimizer, masked):
        a = torch.rand(2, 3, 4, device=device, dtype=dtype)
        b = torch.rand(2, 4, 4, device=device, dtype=dtype)
        kwargs = {"mask1": torch.tensor([[True, True, False], [True, True, True]], device=device)} if masked else {}
        expected = matching.match_smnn_batched(a, b, **kwargs)
        actual = torch_optimizer(matching.match_smnn_batched)(a, b, **kwargs)
        self.assert_close(actual[0], expected[0])
        self.assert_close(actual[1], expected[1])

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("rows", [20, 30])
    def test_export_differences_match_cdist(self, device, dtype, monkeypatch, masked, rows):
        # Export accumulates direct differences in chunks of 16 coordinates, as MPS does for masks, for at most 25
        # rows and before torch 2.13. Every ratio is at least 0.02 from the threshold.
        a = _sine_pattern((2, rows, 20), 0.71, 0.13, 0.0, device, dtype)
        noise = _sine_pattern((2, rows, 20), 0.53, 0.29, 1.0, device, dtype)
        b = torch.cat((a.flip(1) + 0.1 * noise, 2.0 * _sine_pattern((2, 4, 20), 0.83, 0.17, 2.0, device, dtype)), 1)
        kwargs = {}
        if masked:
            limits = torch.tensor([[rows + 4], [rows]], device=device)
            kwargs["mask1"] = torch.arange(rows, device=device).expand(2, rows) % 7 != 3
            kwargs["mask2"] = torch.arange(rows + 4, device=device).expand(2, rows + 4) < limits
        results = []
        for exporting in (False, True):
            monkeypatch.setattr(matching, "is_exporting", lambda exporting=exporting: exporting)
            a_, b_ = a.clone().requires_grad_(), b.clone().requires_grad_()
            ratios, indices = matching.match_smnn_batched(a_, b_, 0.95, **kwargs)
            results.append((ratios, indices, *torch.autograd.grad(ratios.sum(), (a_, b_))))
        (ratios, indices, *grads), (export_ratios, export_indices, *export_grads) = results
        assert len(indices) > 20
        self.assert_close(export_indices, indices)
        self.assert_close(export_ratios, ratios)
        for export_grad, grad in zip(export_grads, grads):
            # Unmasked pairs over 25 rows take the matrix-multiplication form, which moves the gradients by 1e-5.
            self.assert_close(export_grad, grad, low_tolerance=not masked and rows > 25)

    def test_masked_gradients_are_finite(self, device, dtype):
        a = torch.rand(2, 3, 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.rand(2, 4, 4, device=device, dtype=dtype, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, False]], device=device)
        ratios, _ = matching.match_smnn_batched(a, b, 1.0, mask1=mask)
        ratios.sum().backward()
        assert torch.isfinite(a.grad).all()
        assert torch.isfinite(b.grad).all()
        self.assert_close(a.grad[:, 2], torch.zeros_like(a.grad[:, 2]))

    @pytest.mark.parametrize("batch,n,m", [(1, 5, 7), (3, 5, 7), (2, 8, 3)])
    def test_matches_independent_pairs(self, device, dtype, batch, n, m):
        a = torch.rand(batch, n, 8, device=device, dtype=dtype)
        b = torch.rand(batch, m, 8, device=device, dtype=dtype)
        ratios, indices = matching.match_smnn_batched(a, b, 0.95)
        assert ratios.shape == (len(indices), 1)
        assert indices.shape[1] == 3
        assert indices.dtype == torch.long
        assert ratios.device == a.device
        assert ratios.dtype == a.dtype
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

    @pytest.mark.parametrize("mask_first, mask_second", [(True, False), (False, True), (True, True)])
    def test_masks_large_offset_padded_descriptors(self, device, mask_first, mask_second):
        a = torch.full((1, 30, 2), -1e4, device=device)
        b = torch.full((1, 31, 2), -2e4, device=device)
        a[0, [1, 8, 29]] = torch.tensor([[10000.0, 10000.0], [10001.0, 10000.0], [10000.0, 10002.0]], device=device)
        b[0, [3, 15, 30]] = a[0, [1, 8, 29]]
        mask1 = torch.zeros(1, 30, dtype=torch.bool, device=device) if mask_first else None
        mask2 = torch.zeros(1, 31, dtype=torch.bool, device=device) if mask_second else None
        if mask1 is not None:
            mask1[0, [1, 8, 29]] = True
        if mask2 is not None:
            mask2[0, [3, 15, 30]] = True
        ratios, indices = matching.match_smnn_batched(a, b, mask1=mask1, mask2=mask2)
        expected = torch.tensor([[0, 1, 3], [0, 8, 15], [0, 29, 30]], device=device)
        self.assert_close(indices, expected)
        self.assert_close(ratios, torch.zeros(3, 1, device=device))

    @pytest.mark.parametrize("mask_first, mask_second", [(True, False), (False, True), (True, True)])
    def test_nonfinite_padding_matches_compact_gradients(self, device, dtype, mask_first, mask_second):
        a_padding = [float("nan"), float("inf")] if mask_first else [4.0, 0.0]
        b_padding = [float("-inf"), float("nan")] if mask_second else [4.0, 0.0]
        a = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], a_padding]], device=device, dtype=dtype, requires_grad=True
        )
        b = torch.tensor(
            [[[0.1, 0.0], [1.2, 0.0], [3.0, 0.0], b_padding]], device=device, dtype=dtype, requires_grad=True
        )
        mask1 = torch.tensor([[True, True, True, False]], device=device) if mask_first else None
        mask2 = torch.tensor([[True, True, True, False]], device=device) if mask_second else None
        ratios, indices = matching.match_smnn_batched(a, b, 1.0, mask1=mask1, mask2=mask2)
        ratios.sum().backward()

        valid_a = mask1[0].nonzero().flatten() if mask1 is not None else torch.arange(4, device=device)
        valid_b = mask2[0].nonzero().flatten() if mask2 is not None else torch.arange(4, device=device)
        compact_a = a.detach()[:, valid_a].clone().requires_grad_()
        compact_b = b.detach()[:, valid_b].clone().requires_grad_()
        reference_ratios, reference_indices = matching.match_smnn_batched(compact_a, compact_b, 1.0)
        reference_ratios.sum().backward()

        self.assert_close(ratios, reference_ratios)
        self.assert_close(indices, reference_indices)
        self.assert_close(a.grad[:, valid_a], compact_a.grad)
        self.assert_close(b.grad[:, valid_b], compact_b.grad)
        assert torch.isfinite(a.grad).all()
        assert torch.isfinite(b.grad).all()
        assert a.grad[:, valid_a].abs().sum() > 0
        assert b.grad[:, valid_b].abs().sum() > 0
        if mask_first:
            self.assert_close(a.grad[:, 3], torch.zeros_like(a.grad[:, 3]))
        if mask_second:
            self.assert_close(b.grad[:, 3], torch.zeros_like(b.grad[:, 3]))

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
        assert r.shape == (0, 1)
        assert ix.shape == (0, 3)
        assert r.dtype == dtype
        assert ix.device == a.device

    def test_short_valid_rows_return_no_matches(self, device, dtype):
        a = torch.rand(2, 3, 4, device=device, dtype=dtype)
        b = torch.rand(2, 4, 4, device=device, dtype=dtype)
        mask = torch.tensor([[True, False, False], [False, False, False]], device=device)
        r, ix = matching.match_smnn_batched(a, b, mask1=mask)
        assert r.shape == (0, 1)
        assert ix.shape == (0, 3)

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


class TestMatchAdalamNoMatch(BaseTester):
    """The paths of AdaLAM that find nothing to match return ``(dists, idxs)`` like the others (#5848)."""

    HW = (180, 320)

    def _scene(self, device, dtype):
        # One-hot descriptors: every distance between the two sets is sqrt(2), so every ratio is 1 > 0.95, no seed
        # survives and no ratio-test match remains.
        d1 = torch.eye(12, device=device, dtype=dtype)[:6]
        d2 = torch.eye(12, device=device, dtype=dtype)[6:]
        g = torch.Generator().manual_seed(4)
        size = torch.tensor([self.HW[1] - 10.0, self.HW[0] - 10.0])
        lafs1, lafs2 = (
            laf_from_center_scale_ori(torch.rand(1, 6, 2, generator=g) * size + 5.0).to(device, dtype) for _ in range(2)
        )
        return d1, d2, lafs1, lafs2

    def _case(self, case, device, dtype):
        d1, d2, lafs1, lafs2 = self._scene(device, dtype)
        if case == "empty desc1":
            return d1[:0], d2, lafs1[:, :0], lafs2
        if case == "empty desc2":
            return d1, d2[:0], lafs1, lafs2[:, :0]
        if case == "one row in desc1":
            return d1[:1], d2, lafs1[:, :1], lafs2
        if case == "one row in desc2":
            return d1, d2[:1], lafs1, lafs2[:, :1]
        return d1, d2, lafs1, lafs2

    def _assert_no_match(self, dists, idxs, device):
        # AdaLAM computes in float32 whatever the descriptor dtype, and so does its output when something matches
        assert dists.shape == (0, 1)
        assert dists.dtype == torch.float32
        assert idxs.shape == (0, 2)
        assert idxs.dtype == torch.long
        assert dists.device.type == idxs.device.type == device.type

    CASES = ("empty desc1", "empty desc2", "one row in desc1", "one row in desc2", "no seed and no ratio-test match")

    @pytest.mark.parametrize("case", CASES)
    def test_match_adalam(self, device, dtype, case):
        d1, d2, lafs1, lafs2 = self._case(case, device, dtype)
        dists, idxs = match_adalam(d1, d2, lafs1, lafs2, hw1=self.HW, hw2=self.HW)
        self._assert_no_match(dists, idxs, device)

    @pytest.mark.parametrize("case", CASES)
    def test_geometry_aware_descriptor_matcher(self, device, dtype, case):
        d1, d2, lafs1, lafs2 = self._case(case, device, dtype)
        dists, idxs = GeometryAwareDescriptorMatcher("adalam")(d1, d2, lafs1, lafs2)
        self._assert_no_match(dists, idxs, device)

    @pytest.mark.parametrize("case", CASES)
    def test_adalam_filter_match_and_filter(self, device, dtype, case):
        d1, d2, lafs1, lafs2 = self._case(case, device, dtype)
        config = get_adalam_default_config()
        config["device"] = device
        args = (
            get_laf_center(lafs1).reshape(-1, 2),
            get_laf_center(lafs2).reshape(-1, 2),
            d1,
            d2,
            self.HW,
            self.HW,
            get_laf_orientation(lafs1).reshape(-1),
            get_laf_orientation(lafs2).reshape(-1),
            get_laf_scale(lafs1).reshape(-1),
            get_laf_scale(lafs2).reshape(-1),
        )
        # AdalamFilter returns the indices first
        idxs, dists = AdalamFilter(config).match_and_filter(*args, return_dist=True)
        self._assert_no_match(dists, idxs, device)
        idxs = AdalamFilter(config).match_and_filter(*args, return_dist=False)
        assert idxs.shape == (0, 2)
        assert idxs.dtype == torch.long

    def test_no_match_follows_the_output_of_a_match(self, device, dtype):
        # An exact copy of desc1[0] in desc2 gives one ratio-test match (ratio 0) on the no-seed path: the empty output
        # has its dtypes and device.
        d1, d2, lafs1, lafs2 = self._scene(device, dtype)
        d2 = torch.cat([d1[:1], d2[1:]])
        dists, idxs = match_adalam(d1, d2, lafs1, lafs2, hw1=self.HW, hw2=self.HW)
        assert idxs.tolist() == [[0, 0]]
        assert dists.tolist() == [[0.0]]
        empty_dists, empty_idxs = match_adalam(d1[:1], d2, lafs1[:, :1], lafs2, hw1=self.HW, hw2=self.HW)
        assert (empty_dists.dtype, empty_dists.device) == (dists.dtype, dists.device)
        assert (empty_idxs.dtype, empty_idxs.device) == (idxs.dtype, idxs.device)


class TestMatchFGINNSecondNeighbour(BaseTester):
    """The second neighbour of FGINN is the nearest candidate at least ``spatial_th`` from the nearest one (#5849)."""

    @staticmethod
    def _lafs(xy, device, dtype):
        centers = torch.tensor([xy], device=device, dtype=dtype)
        return laf_from_center_scale_ori(centers, torch.full((1, len(xy), 1, 1), 4.0, device=device, dtype=dtype))

    @staticmethod
    def _scales(dtype):
        # Powers of two keep dm exact in every dtype. Float16 cannot hold distances of order 1e6.
        big = 2.0**20 if torch.finfo(dtype).max > 1e7 else 2.0**13
        return (1.0 / big, 1.0, big)

    def _match(self, dm, xy2, device, dtype, **kwargs):
        lafs1 = self._lafs([[10.0, 10.0]], device, dtype)
        lafs2 = self._lafs(xy2, device, dtype)
        desc1 = torch.zeros(1, 2, device=device, dtype=dtype)
        desc2 = torch.zeros(len(xy2), 2, device=device, dtype=dtype)
        return match_fginn(desc1, desc2, lafs1, lafs2, th=0.8, spatial_th=10.0, dm=dm, **kwargs)

    # desc2 index 0 is the nearest neighbour, index 1 lies 2 px from it, index 2 lies 90 px from it
    NEAR_AND_FAR = [[10.0, 10.0], [12.0, 10.0], [100.0, 10.0]]
    # every candidate lies within spatial_th = 10 of the nearest neighbour
    ALL_NEAR = [[10.0, 10.0], [12.0, 10.0], [13.0, 11.0]]

    def test_a_candidate_near_the_nearest_neighbour_is_never_the_second_at_any_scale(self, device, dtype):
        for scale in self._scales(dtype):
            dm = torch.tensor([[2.0, 2.5, 5.0]], device=device, dtype=dtype) * scale
            dists, idxs = self._match(dm, self.NEAR_AND_FAR, device, dtype)
            assert idxs.tolist() == [[0, 0]]
            # 2 / 5, not 2 / 2.5: the ratio does not depend on the scale of dm
            self.assert_close(dists, torch.tensor([[0.4]], device=device, dtype=dtype))

    @pytest.mark.parametrize("mutual", [False, True])
    def test_all_candidates_near_the_nearest_neighbour_give_ratio_zero_at_any_scale(self, device, dtype, mutual):
        for scale in self._scales(dtype):
            dm = torch.tensor([[0.5, 0.75, 1.0]], device=device, dtype=dtype) * scale
            dists, idxs = self._match(dm, self.ALL_NEAR, device, dtype, mutual=mutual)
            assert idxs.tolist() == [[0, 0]]
            # there is no second neighbour: the match is accepted with ratio 0, never nan
            assert dists.tolist() == [[0.0]]

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_distance_matrix(self, device, int_dtype):
        # an integer dm, such as Hamming distances between binary descriptors
        for xy2, expected in ((self.NEAR_AND_FAR, 0.4), (self.ALL_NEAR, 0.0)):
            dm = torch.tensor([[2, 3, 5]], device=device, dtype=int_dtype)
            dists, idxs = self._match(dm, xy2, device, torch.float32)
            assert idxs.tolist() == [[0, 0]]
            # promoted to the default floating dtype
            assert dists.dtype == torch.get_default_dtype()
            self.assert_close(dists, torch.tensor([[expected]], device=device))

    def test_gradients_are_finite(self, device, dtype):
        for xy2, expected in ((self.NEAR_AND_FAR, [[0.2, 0.0, -0.08]]), (self.ALL_NEAR, [[0.0, 0.0, 0.0]])):
            dm = torch.tensor([[2.0, 2.5, 5.0]], device=device, dtype=dtype, requires_grad=True)
            dists, _ = self._match(dm, xy2, device, dtype)
            dists.sum().backward()
            # d(dm[0] / dm[2]) / d dm = (1 / dm[2], 0, -dm[0] / dm[2] ** 2)
            self.assert_close(dm.grad, torch.tensor(expected, device=device, dtype=dtype))

    def test_gradcheck(self, device):
        lafs1 = self._lafs([[10.0, 10.0]], device, torch.float64)
        lafs2 = self._lafs(self.NEAR_AND_FAR, device, torch.float64)
        desc = torch.zeros(3, 2, device=device, dtype=torch.float64)
        dm = torch.tensor([[2.0, 2.5, 5.0]], device=device, dtype=torch.float64)
        self.gradcheck(lambda dm: match_fginn(desc[:1], desc, lafs1, lafs2, 0.8, 10.0, dm=dm)[0], (dm,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        lafs1 = self._lafs([[10.0, 10.0]], device, dtype)
        desc = torch.zeros(3, 2, device=device, dtype=dtype)
        op = torch_optimizer(match_fginn)
        for xy2 in (self.NEAR_AND_FAR, self.ALL_NEAR):
            lafs2 = self._lafs(xy2, device, dtype)
            dm = torch.tensor([[2.0, 2.5, 5.0], [2.5, 2.0, 5.0]], device=device, dtype=dtype)
            expected = match_fginn(desc[:2], desc, lafs1.expand(1, 2, 2, 3), lafs2, 0.8, 10.0, dm=dm)
            actual = op(desc[:2], desc, lafs1.expand(1, 2, 2, 3), lafs2, 0.8, 10.0, dm=dm)
            self.assert_close(actual[0], expected[0])
            self.assert_close(actual[1], expected[1])


class TestConventionsMatching(BaseTester):
    # Descriptors with x != y and no tie in any row or column of their distance matrix. Nearest neighbours in _D2:
    # 0 -> 3, 1 -> 0, 2 -> 0, and row 0 of _D2 is nearest to row 1 of _D1, so 2 -> 0 is not mutual. Squared
    # distances, torch.cdist(_D1, _D2) ** 2: [[85, 20, 40, 17], [13, 122, 130, 113], [40, 137, 85, 80]].
    _D1 = ((9.0, 6.0), (0.0, 3.0), (0.0, 6.0))
    _D2 = ((2.0, 0.0), (11.0, 2.0), (7.0, 12.0), (8.0, 10.0))

    def _descriptors(self, device, dtype):
        return (
            torch.tensor(self._D1, device=device, dtype=dtype),
            torch.tensor(self._D2, device=device, dtype=dtype),
        )

    @staticmethod
    def _lafs(centres, device, dtype):
        # frames with a non-zero orientation; the matchers read their centres only
        xy = torch.tensor(centres, device=device, dtype=dtype)[None]
        n = xy.shape[1]
        ori = torch.tensor([30.0, -70.0, 110.0, 15.0, -160.0][:n], device=device, dtype=dtype).view(1, n, 1)
        return laf_from_center_scale_ori(xy, torch.full((1, n, 1, 1), 4.0, device=device, dtype=dtype), ori)

    def test_convention_match_snn_returns_the_distance_ratio(self, device, dtype):
        d1 = torch.tensor([[0.0, 0.0], [10.0, 0.0]], device=device, dtype=dtype)
        d2 = torch.tensor([[0.0, 0.75], [0.0, -1.0], [10.0, 3.0]], device=device, dtype=dtype)
        # descriptor 0: nearest 0.75, second 1.0 -> ratio 0.75 (kept at th=0.8);
        # descriptor 1: nearest 3.0 (index 2), second sqrt(100 + 0.75**2) -> ratio 0.29916
        nn_dists, nn_idxs = match_nn(d1, d2)
        snn_dists, snn_idxs = match_snn(d1, d2, 0.8)
        self.assert_close(nn_dists.flatten(), torch.tensor([0.75, 3.0], device=device, dtype=dtype))
        self.assert_close(
            snn_dists.flatten(), torch.tensor([0.75, 3.0 / math.sqrt(100.5625)], device=device, dtype=dtype)
        )
        assert nn_idxs.tolist() == snn_idxs.tolist() == [[0, 0], [1, 2]]
        assert snn_dists.dtype == dtype
        assert snn_idxs.dtype == torch.long
        # relabelling: row k of the permuted desc2 is row perm[k], so the second index follows and the ratios stay
        perm = [2, 0, 1]
        perm_dists, perm_idxs = match_snn(d1, d2[perm], 0.8)
        assert perm_idxs.tolist() == [[0, 1], [1, 0]]
        self.assert_close(perm_dists, snn_dists)

    def test_convention_symmetric_ratio_matchers_return_the_larger_of_both_ratios(self, device, dtype):
        d1, d2 = self._descriptors(device, dtype)
        # pair (0, 3): ratio sqrt(17 / 20) from desc1 to desc2, sqrt(17 / 80) back -> the forward one;
        # pair (1, 0): ratio sqrt(13 / 113) forward, sqrt(13 / 40) back -> the backward one.
        expected = torch.tensor([[math.sqrt(17 / 20)], [math.sqrt(13 / 40)]], device=device, dtype=dtype)
        dists, idxs = match_smnn(d1, d2, 0.95)
        assert idxs.tolist() == [[0, 3], [1, 0]]
        self.assert_close(dists, expected)
        # swapping the inputs swaps the index columns and keeps each pair's ratio
        swapped_dists, swapped_idxs = match_smnn(d2, d1, 0.95)
        assert swapped_idxs.tolist() == [[0, 1], [3, 0]]
        self.assert_close(swapped_dists, expected.flip(0))
        # match_smnn_batched gives the same pairs and ratios per item; item 1 has desc2 permuted (old 3 -> new 0,
        # old 0 -> new 2)
        perm = [3, 1, 0, 2]
        ratios, indices = matching.match_smnn_batched(torch.stack([d1, d1]), torch.stack([d2, d2[perm]]), 0.95)
        assert indices.tolist() == [[0, 0, 3], [0, 1, 0], [1, 0, 0], [1, 1, 2]]
        self.assert_close(ratios, torch.cat([expected, expected]))

    @pytest.mark.parametrize("matcher", ["snn", "smnn", "smnn_batched", "fginn"])
    def test_convention_ratio_thresholds_are_inclusive(self, device, dtype, matcher):
        # Distance matrices whose ratios are exact in every dtype. First: row 0 has 1 / 2 = 0.5 to its nearest
        # neighbour, column 0 has 1 / 4 = 0.25 back, row 1 has 3 / 4 = 0.75. Second: the 0.5 is the backward ratio
        # of the same pair (column 0: 1 / 2) and the forward one is 1 / 4.
        cases = [[[1.0, 2.0, 5.0], [4.0, 3.0, 6.0]], [[1.0, 4.0, 6.0], [2.0, 3.0, 5.0]]]
        if matcher in ("snn", "fginn"):
            cases = cases[:1]  # one-directional: no backward ratio
        desc1 = torch.zeros(2, 4, device=device, dtype=dtype)
        desc2 = torch.zeros(3, 4, device=device, dtype=dtype)
        lafs1 = self._lafs([[33.0, 21.0], [91.0, 57.0]], device, dtype)
        # candidate centres far more than spatial_th apart, so the geometric check changes nothing
        lafs2 = self._lafs([[30.0, 20.0], [90.0, 55.0], [140.0, 12.0]], device, dtype)

        def run(dm, th):
            if matcher == "snn":
                return match_snn(desc1, desc2, th, dm)
            if matcher == "smnn":
                return match_smnn(desc1, desc2, th, dm)
            if matcher == "fginn":
                return match_fginn(desc1, desc2, lafs1, lafs2, th, 10.0, dm=dm)
            ratios, indices = matching.match_smnn_batched(desc1[None], desc2[None], th, dm[None])
            return ratios, indices[:, 1:]

        for rows in cases:
            dm = torch.tensor(rows, device=device, dtype=dtype)
            dists, idxs = run(dm, 0.5)
            assert idxs.tolist() == [[0, 0]]  # a ratio equal to th is kept
            self.assert_close(dists, torch.tensor([[0.5]], device=device, dtype=dtype))
            assert run(dm, 0.49)[1].shape[0] == 0

    def test_convention_match_mnn_keeps_mutual_nearest_neighbours_only(self, device, dtype):
        d1, d2 = self._descriptors(device, dtype)
        assert match_nn(d1, d2)[1].tolist() == [[0, 3], [1, 0], [2, 0]]
        dists, idxs = match_mnn(d1, d2)
        # row 0 of desc2 is nearest to row 1 of desc1, so the nearest neighbour 2 -> 0 is dropped
        assert idxs.tolist() == [[0, 3], [1, 0]]
        assert idxs.dtype == torch.long
        self.assert_close(dists, torch.tensor([[math.sqrt(17.0)], [math.sqrt(13.0)]], device=device, dtype=dtype))
        # relabelling: row k of the permuted desc2 is row perm[k] (old 3 -> new 2, old 0 -> new 1)
        perm = [2, 0, 3, 1]
        assert match_mnn(d1, d2[perm])[1].tolist() == [[0, 2], [1, 1]]
        # swapping the inputs (now more rows in desc1 than in desc2) swaps the index columns; row order is not part
        # of the contract
        swapped_dists, swapped_idxs = match_mnn(d2, d1)
        order = swapped_idxs[:, 0].argsort()
        assert swapped_idxs[order].tolist() == [[0, 1], [3, 0]]
        self.assert_close(
            swapped_dists[order], torch.tensor([[math.sqrt(13.0)], [math.sqrt(17.0)]], device=device, dtype=dtype)
        )

    def test_convention_matchers_need_one_or_two_descriptors_per_side(self, device, dtype):
        d1, d2 = self._descriptors(device, dtype)
        lafs1 = self._lafs([[33.0, 21.0], [91.0, 57.0], [140.0, 70.0]], device, dtype)
        lafs2 = self._lafs([[30.0, 20.0], [90.0, 55.0], [140.0, 12.0], [5.0, 60.0]], device, dtype)
        # an empty side, first or second, gives (0, 1) values in the descriptor dtype (float32 for match_adalam) and
        # (0, 2) int64 indices ((0, 3) for match_smnn_batched)
        for a, b, lafs_a, lafs_b in ((d1[:0], d2, lafs1[:, :0], lafs2), (d1, d2[:0], lafs1, lafs2[:, :0])):
            outputs = [(match(a, b), dtype) for match in (match_nn, match_mnn, match_snn, match_smnn)]
            outputs.append((match_fginn(a, b, lafs_a, lafs_b), dtype))
            outputs.append((match_adalam(a, b, lafs_a, lafs_b), torch.float32))
            for (dists, idxs), dists_dtype in outputs:
                assert dists.shape == (0, 1)
                assert dists.dtype == dists_dtype
                assert idxs.shape == (0, 2)
                assert idxs.dtype == torch.long
            ratios, indices = matching.match_smnn_batched(a[None], b[None])
            assert ratios.shape == (0, 1)
            assert ratios.dtype == dtype
            assert indices.shape == (0, 3)
            assert indices.dtype == torch.long
        # one row in desc2: nn and mnn still match, the ratio tests have no second neighbour
        assert match_nn(d1, d2[:1])[1].tolist() == [[0, 0], [1, 0], [2, 0]]
        assert match_mnn(d1, d2[:1])[1].tolist() == [[1, 0]]
        assert match_snn(d1, d2[:1], 0.95)[1].shape == (0, 2)
        assert match_fginn(d1, d2[:1], lafs1, lafs2[:, :1], 0.95)[1].shape == (0, 2)
        assert match_adalam(d1, d2[:1], lafs1, lafs2[:, :1])[1].shape == (0, 2)
        # one row in desc1: match_snn keeps its match (ratio sqrt(17 / 20) = 0.92), the symmetric tests and AdaLAM
        # need two rows on both sides
        assert match_snn(d1[:1], d2, 0.95)[1].tolist() == [[0, 3]]
        assert match_smnn(d1[:1], d2, 0.95)[1].shape == (0, 2)
        assert matching.match_smnn_batched(d1[None, :1], d2[None], 0.95)[1].shape == (0, 3)
        assert match_adalam(d1[:1], d2, lafs1[:, :1], lafs2)[1].shape == (0, 2)

    def test_convention_matchers_read_a_given_distance_matrix_as_it_is(self, device, dtype):
        d1, d2 = self._descriptors(device, dtype)
        # squared L2 distances, exact in every dtype: torch.cdist(d1, d2) ** 2
        dm = torch.tensor(
            [[85.0, 20.0, 40.0, 17.0], [13.0, 122.0, 130.0, 113.0], [40.0, 137.0, 85.0, 80.0]],
            device=device,
            dtype=dtype,
        )
        # nn and mnn return the entries of dm
        nn_dists, nn_idxs = match_nn(d1, d2, dm=dm)
        assert nn_idxs.tolist() == match_nn(d1, d2)[1].tolist() == [[0, 3], [1, 0], [2, 0]]
        self.assert_close(nn_dists.flatten(), torch.tensor([17.0, 13.0, 40.0], device=device, dtype=dtype))
        self.assert_close(match_mnn(d1, d2, dm=dm)[0].flatten(), torch.tensor([17.0, 13.0], device=device, dtype=dtype))
        # the ratio test takes the ratio of what dm holds: squared distances square the ratio, so th is squared too
        snn_dists, snn_idxs = match_snn(d1, d2, 0.8**2, dm=dm)
        assert snn_idxs.tolist() == match_snn(d1, d2, 0.8)[1].tolist() == [[1, 0], [2, 0]]
        self.assert_close(snn_dists.flatten(), torch.tensor([13 / 113, 40 / 80], device=device, dtype=dtype))

    def test_convention_matcher_modules_have_their_own_threshold_defaults(self, device, dtype):
        # pair (0, 0): ratio 0.825 forward, 0.825 / sqrt(11.175**2 + 5**2) = 0.067 backward;
        # pair (1, 2): ratio 4 / sqrt(11.175**2 + 5**2) = 0.32673 forward, 4 / sqrt(144 + 81) backward
        d1 = torch.tensor([[0.0, 0.0], [12.0, 5.0]], device=device, dtype=dtype)
        d2 = torch.tensor([[0.825, 0.0], [0.0, -1.0], [12.0, 9.0]], device=device, dtype=dtype)
        lafs1 = self._lafs([[33.0, 21.0], [91.0, 57.0]], device, dtype)
        lafs2 = self._lafs([[30.0, 20.0], [90.0, 55.0], [140.0, 12.0]], device, dtype)
        # DescriptorMatcher uses th = 0.8 in every mode, match_smnn defaults to 0.95
        assert DescriptorMatcher("smnn")(d1, d2)[1].tolist() == [[1, 2]]
        assert match_smnn(d1, d2)[1].tolist() == [[0, 0], [1, 2]]
        # match_snn defaults to 0.8 too, so the two agree
        assert DescriptorMatcher("snn")(d1, d2)[1].tolist() == match_snn(d1, d2)[1].tolist() == [[1, 2]]
        # GeometryAwareDescriptorMatcher("fginn") uses th = 0.85, match_fginn defaults to 0.8
        dists, idxs = GeometryAwareDescriptorMatcher("fginn")(d1, d2, lafs1, lafs2)
        assert idxs.tolist() == [[0, 0], [1, 2]]
        self.assert_close(
            dists.flatten(), torch.tensor([0.825, 4.0 / math.sqrt(149.880625)], device=device, dtype=dtype)
        )
        assert match_fginn(d1, d2, lafs1, lafs2)[1].tolist() == [[1, 2]]

    @pytest.mark.parametrize("unit", [1.0, 3.0, 0.25])
    def test_convention_match_fginn_spatial_th_is_in_the_units_of_the_lafs2_centres(self, device, dtype, unit):
        # Query 0's candidates lie 5, 10, 25 and 40 away in descriptor space (3-4-5 triangles), so its ratio is
        # 5 / 10 = 0.5 with candidate 1 as the 2nd neighbour and 5 / 25 = 0.2 when candidate 1 is discarded as a
        # repeat of candidate 0. Query 1's nearest neighbour is candidate 4 (distance 2), the next candidate 2,
        # sqrt(25**2 + 13**2) = sqrt(794) away.
        d1 = torch.tensor([[0.0, 0.0], [40.0, 7.0]], device=device, dtype=dtype)
        d2 = torch.tensor([[3.0, 4.0], [6.0, 8.0], [15.0, 20.0], [24.0, 32.0], [40.0, 9.0]], device=device, dtype=dtype)
        # candidate 1's centre exactly spatial_th = 10 from candidate 0's competes; 9.6 away it is discarded
        for centre1, ratio0 in (((36.0, 28.0), 0.5), ((36.0, 27.5), 0.2)):
            centres2 = [[30.0, 20.0], list(centre1), [80.0, 50.0], [5.0, 60.0], [120.0, 90.0]]
            lafs2 = self._lafs([[unit * x, unit * y] for x, y in centres2], device, dtype)
            lafs1 = self._lafs([[33.0, 21.0], [91.0, 57.0]], device, dtype)
            dists, idxs = match_fginn(d1, d2, lafs1, lafs2, 0.8, 10.0 * unit)
            assert idxs.tolist() == [[0, 0], [1, 4]]
            self.assert_close(
                dists.flatten(), torch.tensor([ratio0, 2.0 / math.sqrt(794.0)], device=device, dtype=dtype)
            )
            # lafs1 is not read
            other_lafs1 = self._lafs([[7.0, 150.0], [64.0, 3.0]], device, dtype)
            assert match_fginn(d1, d2, other_lafs1, lafs2, 0.8, 10.0 * unit)[1].tolist() == idxs.tolist()
            # relabelling: permuting desc2 together with lafs2 (old 0 -> new 2, old 4 -> new 0) moves the indices
            perm = [4, 2, 0, 3, 1]
            perm_dists, perm_idxs = match_fginn(d1, d2[perm], lafs1, lafs2[:, perm], 0.8, 10.0 * unit)
            assert perm_idxs.tolist() == [[0, 2], [1, 0]]
            self.assert_close(perm_dists, dists)

    def test_convention_match_fginn_takes_the_second_neighbour_among_the_10_nearest(self, device, dtype):
        # desc2 has 12 rows at distances 1 to 12 from the query. The 10 nearest lie within spatial_th = 10 of the
        # nearest one's centre, the other two far from it.
        centres2 = [[100.0 + 0.4 * i, 100.0] for i in range(10)] + [[400.0, 300.0], [10.0, 20.0]]
        scale = torch.full((1, 12, 1, 1), 4.0, device=device, dtype=dtype)
        lafs2 = laf_from_center_scale_ori(torch.tensor([centres2], device=device, dtype=dtype), scale)
        lafs1 = self._lafs([[33.0, 21.0]], device, dtype)
        desc1 = torch.zeros(1, 2, device=device, dtype=dtype)
        desc2 = torch.zeros(12, 2, device=device, dtype=dtype)
        dm = torch.arange(1.0, 13.0, device=device, dtype=dtype)[None]
        # every candidate is excluded, and the two far rows are not candidates: the ratio is 0 and the match is kept
        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 10.0, dm=dm)
        assert idxs.tolist() == [[0, 0]]
        assert dists.tolist() == [[0.0]]
        # a far row among the 10 nearest (distance 10) is the second neighbour: ratio 1 / 10
        dm_far = dm.clone()
        dm_far[0, 9], dm_far[0, 10] = 11.0, 10.0
        dists, idxs = match_fginn(desc1, desc2, lafs1, lafs2, 0.8, 10.0, dm=dm_far)
        assert idxs.tolist() == [[0, 0]]
        self.assert_close(dists, torch.tensor([[0.1]], device=device, dtype=dtype))

    @staticmethod
    def _oriented_adalam_scene():
        """``_adalam_scene`` with orientations (degrees) and scales.

        The 400 inliers turn by 10 degrees and grow by 1.2 between the images; the outliers get random values.
        """
        d1, d2, lafs1, lafs2, hw = _adalam_scene(torch.device("cpu"))
        g = torch.Generator().manual_seed(7)
        n1, n2 = lafs1.shape[1], lafs2.shape[1]
        ori1 = torch.rand(1, n1, 1, generator=g) * 360.0 - 180.0
        ori2 = torch.rand(1, n2, 1, generator=g) * 360.0 - 180.0
        ori2[:, :400] = ori1[:, :400] + 10.0
        scale1 = torch.rand(1, n1, 1, 1, generator=g) * 12.0 + 4.0
        scale2 = torch.rand(1, n2, 1, 1, generator=g) * 12.0 + 4.0
        scale2[:, :400] = 1.2 * scale1[:, :400]
        # a random per-keypoint scale factor in [1/4, 4]
        jitter = torch.exp(torch.empty(1, n1, 1, 1).uniform_(-math.log(4.0), math.log(4.0), generator=g))
        frames1 = (get_laf_center(lafs1), scale1, ori1)
        frames2 = (get_laf_center(lafs2), scale2, ori2)
        return d1, d2, frames1, frames2, jitter, hw

    def test_convention_match_adalam_reads_the_orientation_and_scale_change_between_the_images(self, device, dtype):
        d1, d2, (c1, s1, o1), (c2, s2, o2), jitter, hw = self._oriented_adalam_scene()
        d1, d2 = d1.to(device, dtype), d2.to(device, dtype)

        def run(frames1, frames2):
            lafs1 = laf_from_center_scale_ori(*frames1).to(device, dtype)
            lafs2 = laf_from_center_scale_ori(*frames2).to(device, dtype)
            return match_adalam(d1, d2, lafs1, lafs2, hw1=hw, hw2=hw)

        quality, idxs = run((c1, s1, o1), (c2, s2, o2))
        assert len(idxs) > 350
        # Only the changes between the images are read: negating or shifting both orientations, or scaling both
        # LAF sets, keeps the inliers' common turn (-10 degrees after negation) and 1.2 scale ratio.
        for frames1, frames2 in (
            ((c1, s1, -o1), (c2, s2, -o2)),
            ((c1, s1, o1 + 37.0), (c2, s2, o2 + 37.0)),
            ((c1, 3.0 * s1, o1), (c2, 3.0 * s2, o2)),
        ):
            other_quality, other_idxs = run(frames1, frames2)
            assert torch.equal(other_idxs, idxs)
            self.assert_close(other_quality, quality)
        # Negating one side breaks the inliers' common turn (orientation_difference_threshold 30 is in degrees),
        # and per-keypoint scale changes break their common scale ratio.
        assert len(run((c1, s1, -o1), (c2, s2, o2))[1]) < 0.6 * len(idxs)
        assert len(run((c1, s1 * jitter, o1), (c2, s2, o2))[1]) < 0.9 * len(idxs)

    def test_convention_match_adalam_returns_float32_inverse_confidences(self, device, dtype):
        d1, d2, frames1, frames2, _, (h, w) = self._oriented_adalam_scene()
        d1, d2 = d1.to(device, dtype), d2.to(device, dtype)
        lafs1 = laf_from_center_scale_ori(*frames1).to(device, dtype)
        lafs2 = laf_from_center_scale_ori(*frames2).to(device, dtype)
        quality, idxs = match_adalam(d1, d2, lafs1, lafs2, hw1=(h, w), hw2=(h, w))
        # the first value is 1 / (local RANSAC confidence) in float32 whatever the descriptor dtype, and a match
        # needs a confidence of at least min_confidence (default 200)
        assert quality.dtype == torch.float32
        assert quality.shape == (len(idxs), 1)
        assert len(idxs) > 350
        assert quality.min() > 0
        assert quality.max() <= 1 / 200
        quality400, _ = match_adalam(d1, d2, lafs1, lafs2, config={"min_confidence": 400}, hw1=(h, w), hw2=(h, w))
        assert len(quality400) > 350
        assert quality400.max() <= 1 / 400
        # hw1 and hw2 enter only as the image area (the seed radius): (H, W) and (W, H) agree, and sizes of 64 times
        # the area agree with each other whatever their aspect ratio, while they change the matches
        assert torch.equal(match_adalam(d1, d2, lafs1, lafs2, hw1=(w, h), hw2=(w, h))[1], idxs)
        sizes = [(8 * h, 8 * w), (8 * w, 8 * h), (64 * h, w), (h, 64 * w)]
        larger = [match_adalam(d1, d2, lafs1, lafs2, hw1=size, hw2=size)[1] for size in sizes]
        assert len(larger[0]) < 0.6 * len(idxs)
        assert all(torch.equal(other, larger[0]) for other in larger[1:])

    def test_convention_local_feature_matcher_returns_laf_centres_and_one_minus_the_matcher_value(self, device, dtype):
        class _NotCalled(torch.nn.Module):
            def forward(self, *args):
                raise AssertionError("the pre-extracted lafs and descriptors must be used")

        d1, d2 = self._descriptors(device, dtype)
        pad = torch.tensor([[50.0, 50.0]], device=device, dtype=dtype)  # identical in both images
        lafs_a = self._lafs([[12.5, 40.0], [70.0, 9.0], [33.0, 55.0]], device, dtype)[0]
        lafs_b = self._lafs([[14.0, 41.0], [71.0, 11.0], [35.0, 52.0], [60.0, 30.0]], device, dtype)[0]

        def slots(lafs, descs, order, pad_slot):
            # the real features in `order` with an all-zero padding LAF inserted at `pad_slot`
            lafs, descs = list(lafs[order]), list(descs[order])
            lafs.insert(pad_slot, torch.zeros_like(lafs[0]))
            descs.insert(pad_slot, pad[0])
            return torch.stack(lafs), torch.stack(descs)

        # pair 0: features in order, padding in slot 1 / slot 2; pair 1: image 0 reversed and image 1 permuted
        lafs0, descs0 = zip(slots(lafs_a, d1, [0, 1, 2], 1), slots(lafs_a, d1, [2, 1, 0], 0))
        lafs1, descs1 = zip(slots(lafs_b, d2, [0, 1, 2, 3], 2), slots(lafs_b, d2, [3, 0, 2, 1], 4))
        images = torch.zeros(2, 1, 48, 64, device=device, dtype=dtype)
        data = {
            "image0": images,
            "image1": images,
            "lafs0": torch.stack(lafs0),
            "descriptors0": torch.stack(descs0),
            "lafs1": torch.stack(lafs1),
            "descriptors1": torch.stack(descs1),
        }
        out = LocalFeatureMatcher(_NotCalled(), DescriptorMatcher("snn", 0.8))(data)
        # On the real features snn keeps desc1 rows 1 and 2, both with desc2 row 0 (ratios sqrt(13 / 113) and
        # sqrt(40 / 80)); the padding descriptors match each other at distance 0, but zero LAFs are dropped first.
        r1, r2 = math.sqrt(13 / 113), math.sqrt(40 / 80)
        expected0 = torch.tensor([[70.0, 9.0], [33.0, 55.0], [33.0, 55.0], [70.0, 9.0]], device=device, dtype=dtype)
        self.assert_close(out["keypoints0"], expected0)
        self.assert_close(out["keypoints1"], torch.tensor([[14.0, 41.0]] * 4, device=device, dtype=dtype))
        self.assert_close(out["confidence"], 1.0 - torch.tensor([r1, r2, r2, r1], device=device, dtype=dtype))
        assert out["batch_indexes"].tolist() == [0, 0, 1, 1]
        self.assert_close(out["lafs0"][0], lafs_a[[1, 2, 2, 1]])
        self.assert_close(out["lafs1"][0], lafs_b[[0, 0, 0, 0]])

    def test_convention_matchers_return_dists_in_the_dtype_of_a_given_dm(self, device, dtype):
        d1, d2 = self._descriptors(device, dtype)
        other = torch.float16 if dtype == torch.float32 else torch.float32
        # torch.cdist(d1, d2) ** 2, exact in every dtype, in a dtype other than that of the descriptors
        dm = torch.tensor(
            [[85.0, 20.0, 40.0, 17.0], [13.0, 122.0, 130.0, 113.0], [40.0, 137.0, 85.0, 80.0]],
            device=device,
            dtype=other,
        )
        for match in (match_nn, match_mnn, match_snn, match_smnn):
            dists, idxs = match(d1, d2, dm=dm)
            assert len(idxs) > 0
            assert dists.dtype == other

    def test_convention_match_adalam_without_seeds_returns_the_squared_ratio_test(self, device, dtype):
        # 8 keypoints leave no AdaLAM seed, so the matches are those of a ratio test on squared distances,
        # ratio <= 0.95, and the first value is that ratio (float32), not an inverse confidence
        g = torch.Generator().manual_seed(9)
        d1 = torch.nn.functional.normalize(torch.randn(8, 24, generator=g), dim=1)
        noisy = d1 + 0.6 * torch.randn(8, 24, generator=g)
        d2 = torch.nn.functional.normalize(noisy, dim=1)[torch.randperm(8, generator=g)]
        frames = [
            laf_from_center_scale_ori(
                torch.rand(1, 8, 2, generator=g) * torch.tensor([300.0, 170.0]) + 5.0,
                torch.full((1, 8, 1, 1), 5.0),
                torch.rand(1, 8, 1, generator=g) * 90.0,
            )
            for _ in range(2)
        ]
        d1, d2 = d1.to(device, dtype), d2.to(device, dtype)
        lafs1, lafs2 = (f.to(device, dtype) for f in frames)
        quality, idxs = match_adalam(d1, d2, lafs1, lafs2, hw1=(180, 310), hw2=(180, 310))
        vals, nearest = (torch.cdist(d1.float(), d2.float()) ** 2).topk(2, dim=1, largest=False)
        ratio = vals[:, 0] / vals[:, 1]
        keep = ratio <= 0.95
        assert 0 < int(keep.sum()) < 8
        assert idxs.tolist() == torch.stack([torch.where(keep)[0], nearest[keep, 0]], 1).tolist()
        assert quality.dtype == torch.float32
        self.assert_close(quality.flatten(), ratio[keep], rtol=1e-5, atol=1e-5)

    def test_convention_match_adalam_ignores_a_common_rotation_or_scaling_of_one_image(self, device, dtype):
        d1, d2, (c1, s1, o1), (c2, s2, o2), _, hw = self._oriented_adalam_scene()
        d1, d2 = d1.to(device, dtype), d2.to(device, dtype)

        def run(frames1, frames2):
            lafs1 = laf_from_center_scale_ori(*frames1).to(device, dtype)
            lafs2 = laf_from_center_scale_ori(*frames2).to(device, dtype)
            return match_adalam(d1, d2, lafs1, lafs2, hw1=hw, hw2=hw)

        quality, idxs = run((c1, s1, o1), (c2, s2, o2))
        assert len(idxs) > 350
        # Turning or scaling the LAFs of one image alike changes the orientation change or the scale ratio of every
        # match by the same amount, so the inliers stay consistent with each other although their common turn of
        # 10 degrees becomes -113 or -90 degrees and their scale ratio of 1.2 becomes 0.48 or 0.4.
        for frames1, frames2 in (
            ((c1, s1, o1), (c2, 0.4 * s2, o2 - 123.0)),
            ((c1, 3.0 * s1, o1 + 100.0), (c2, s2, o2)),
        ):
            other_quality, other_idxs = run(frames1, frames2)
            assert torch.equal(other_idxs, idxs)
            self.assert_close(other_quality, quality)

    def test_convention_match_adalam_warns_about_an_unknown_config_key_and_ignores_it(self, device, dtype):
        d1, d2, frames1, frames2, _, hw = self._oriented_adalam_scene()
        d1, d2 = d1.to(device, dtype), d2.to(device, dtype)
        lafs1 = laf_from_center_scale_ori(*frames1).to(device, dtype)
        lafs2 = laf_from_center_scale_ori(*frames2).to(device, dtype)
        quality, idxs = match_adalam(d1, d2, lafs1, lafs2, hw1=hw, hw2=hw)
        # "mnn" is not read from the configuration either
        for key in ("foo", "mnn"):
            with pytest.warns(UserWarning, match=rf"not recognized \({key}\)"):
                other_quality, other_idxs = match_adalam(d1, d2, lafs1, lafs2, config={key: 1}, hw1=hw, hw2=hw)
            assert torch.equal(other_idxs, idxs)
            self.assert_close(other_quality, quality)

    def test_convention_match_adalam_reads_each_image_size_for_its_own_image(self, device, dtype):
        d1, d2, frames1, frames2, _, (h, w) = self._oriented_adalam_scene()
        d1, d2 = d1.to(device, dtype), d2.to(device, dtype)
        lafs1 = laf_from_center_scale_ori(*frames1).to(device, dtype)
        lafs2 = laf_from_center_scale_ori(*frames2).to(device, dtype)
        idxs = match_adalam(d1, d2, lafs1, lafs2, hw1=(h, w), hw2=(h, w))[1]
        # The area of each image sets the neighbourhood radius in that image: a smaller second image alone, or a
        # smaller first image alone, changes the matches, and (H, W) and (W, H) of that size agree.
        smaller2 = match_adalam(d1, d2, lafs1, lafs2, hw1=(h, w), hw2=(h // 2, w // 2))[1]
        assert 0 < len(smaller2) < 0.9 * len(idxs)
        assert torch.equal(match_adalam(d1, d2, lafs1, lafs2, hw1=(h, w), hw2=(w // 2, h // 2))[1], smaller2)
        smaller1 = match_adalam(d1, d2, lafs1, lafs2, hw1=(h // 3, w // 3), hw2=(h, w))[1]
        assert 0 < len(smaller1) < 0.9 * len(idxs)
        assert torch.equal(match_adalam(d1, d2, lafs1, lafs2, hw1=(w // 3, h // 3), hw2=(h, w))[1], smaller1)

    def test_convention_match_adalam_keeps_exact_correspondences_5866(self, device, dtype):
        d1, _, (c1, s1, o1), _, _, hw = self._oriented_adalam_scene()
        desc = d1.to(device, dtype)
        lafs = laf_from_center_scale_ori(c1, s1, o1).to(device, dtype)
        # Exact matches must survive the local verification as well as the no-seed fallback.
        exact = match_adalam(desc, desc, lafs, lafs, hw1=hw, hw2=hw)[1]
        assert exact.tolist() == [[i, i] for i in range(len(desc))]
        shift = torch.randn(c1.shape, generator=torch.Generator().manual_seed(11)) * 0.1
        moved = laf_from_center_scale_ori(c1 + shift, s1, o1).to(device, dtype)
        assert len(match_adalam(desc, desc, lafs, moved, hw1=hw, hw2=hw)[1]) > 0.95 * len(desc)
        # With 8 keypoints no seed survives, so the ratio-test fallback runs and matches each keypoint to itself.
        few = match_adalam(desc[:8], desc[:8], lafs[:, :8], lafs[:, :8], hw1=hw, hw2=hw)[1]
        assert few.tolist() == [[i, i] for i in range(8)]
