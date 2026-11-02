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


def _laf():
    return KF.laf_from_center_scale_ori(torch.tensor([[[60.0, 50.0]]]), torch.full((1, 1, 1, 1), 8.0))


class TestLAFOrienterInputShape:
    @pytest.mark.parametrize("channels", [2, 3])
    def test_rejects_multichannel_image(self, channels):
        with pytest.raises(ShapeError):
            KF.LAFOrienter(32)(_laf(), torch.rand(1, channels, 120, 160))

    def test_accepts_grayscale_image(self):
        laf = _laf()
        out = KF.LAFOrienter(32)(laf, torch.rand(1, 1, 120, 160))
        assert out.shape == laf.shape


class TestPatchAffineShapeEstimatorInputShape:
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 1, 1, 19),
            (2, 1, 19, 1),
            (2, 1, 18, 19),
            (2, 1, 20, 20),
            (2, 3, 19, 19),
        ],
    )
    def test_rejects_wrong_patch_shape(self, shape):
        with pytest.raises(ShapeError):
            KF.PatchAffineShapeEstimator(19)(torch.rand(*shape))

    def test_accepts_matching_patch(self):
        out = KF.PatchAffineShapeEstimator(19)(torch.rand(2, 1, 19, 19))
        assert out.shape == (2, 1, 3)

    def test_patch_size_follows_constructor(self):
        out = KF.PatchAffineShapeEstimator(32)(torch.rand(2, 1, 32, 32))
        assert out.shape == (2, 1, 3)
        with pytest.raises(ShapeError):
            KF.PatchAffineShapeEstimator(32)(torch.rand(2, 1, 19, 19))
