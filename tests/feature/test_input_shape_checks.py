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

"""Input shape checks of LAFOrienter and PatchAffineShapeEstimator (issue #6530)."""

import pytest
import torch

import kornia.feature as KF
from kornia.core.check import ShapeError

from testing.base import BaseTester


def _assert_rejected_up_front(call):
    """The input must be refused by a shape check, not fail later with a RuntimeError or pass silently."""
    with pytest.raises(ShapeError):
        call()


def _laf(device, dtype, batch_size=1, num_laf=1):
    centers = torch.tensor([60.0, 50.0], device=device, dtype=dtype).expand(batch_size, num_laf, 2).contiguous()
    scales = torch.full((batch_size, num_laf, 1, 1), 8.0, device=device, dtype=dtype)
    return KF.laf_from_center_scale_ori(centers, scales)


class TestLAFOrienterInputShape(BaseTester):
    def test_smoke(self, device, dtype):
        laf = _laf(device, dtype)
        img = torch.rand(1, 1, 120, 160, device=device, dtype=dtype)
        out = KF.LAFOrienter(32).to(device, dtype)(laf, img)
        assert out.shape == laf.shape

    @pytest.mark.parametrize("channels", [2, 3])
    def test_exception(self, channels, device, dtype):
        orienter = KF.LAFOrienter(32).to(device, dtype)
        laf = _laf(device, dtype)
        img = torch.rand(1, channels, 120, 160, device=device, dtype=dtype)
        _assert_rejected_up_front(lambda: orienter(laf, img))

    @pytest.mark.parametrize("batch_size, num_laf", [(1, 1), (2, 3)])
    def test_cardinality(self, batch_size, num_laf, device, dtype):
        laf = _laf(device, dtype, batch_size, num_laf)
        img = torch.rand(batch_size, 1, 120, 160, device=device, dtype=dtype)
        out = KF.LAFOrienter(32).to(device, dtype)(laf, img)
        assert out.shape == (batch_size, num_laf, 2, 3)

class TestPatchAffineShapeEstimatorInputShape(BaseTester):
    def test_smoke(self, device, dtype):
        patch = torch.rand(2, 1, 19, 19, device=device, dtype=dtype)
        out = KF.PatchAffineShapeEstimator(19).to(device, dtype)(patch)
        assert out.shape == (2, 1, 3)

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 1, 1, 19),  # used to broadcast silently and return (B, 1, 3)
            (2, 1, 19, 1),
            (2, 1, 18, 19),
            (2, 1, 20, 20),  # used to fail late with a broadcasting RuntimeError
            (2, 3, 19, 19),
        ],
    )
    def test_exception(self, shape, device, dtype):
        estimator = KF.PatchAffineShapeEstimator(19).to(device, dtype)
        patch = torch.rand(*shape, device=device, dtype=dtype)
        _assert_rejected_up_front(lambda: estimator(patch))

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_cardinality(self, batch_size, device, dtype):
        patch = torch.rand(batch_size, 1, 19, 19, device=device, dtype=dtype)
        out = KF.PatchAffineShapeEstimator(19).to(device, dtype)(patch)
        assert out.shape == (batch_size, 1, 3)

    def test_module(self, device, dtype):
        # the expected patch size follows the constructor argument
        estimator = KF.PatchAffineShapeEstimator(32).to(device, dtype)
        out = estimator(torch.rand(2, 1, 32, 32, device=device, dtype=dtype))
        assert out.shape == (2, 1, 3)
        _assert_rejected_up_front(lambda: estimator(torch.rand(2, 1, 19, 19, device=device, dtype=dtype)))
