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

from kornia.morphology import bottom_hat, closing, dilation, erosion, gradient, opening, top_hat
from kornia.morphology import morphology as morphology_module

from testing.base import BaseTester

# Each composite with the expression it is defined as, and whether its output is non-negative.
COMPOSITES = {
    "opening": (opening, lambda x, k, **o: dilation(erosion(x, k, **o), k, **o), False),
    "closing": (closing, lambda x, k, **o: erosion(dilation(x, k, **o), k, **o), False),
    "gradient": (gradient, lambda x, k, **o: dilation(x, k, **o) - erosion(x, k, **o), True),
    "top_hat": (top_hat, lambda x, k, **o: x - opening(x, k, **o), True),
    "bottom_hat": (bottom_hat, lambda x, k, **o: closing(x, k, **o) - x, True),
}


class TestComposites(BaseTester):
    @pytest.mark.parametrize("name", list(COMPOSITES))
    def test_convention_composite_is_its_definition(self, device, dtype, monkeypatch, name):
        # Each composite evaluates its defining expression with the same kernel and options in every half,
        # so the two sides are bitwise equal in every dtype. A non-default `border_value` and `origin` would
        # show a half that fell back to its default; the engine is checked by recording it.
        op, definition, non_negative = COMPOSITES[name]
        l_kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)

        assert torch.equal(op(tensor, l_kernel), definition(tensor, l_kernel))
        if non_negative:
            assert (op(tensor, l_kernel) >= 0).all()
        options = {"border_type": "constant", "border_value": 0.5, "origin": [0, 0]}
        assert torch.equal(op(tensor, l_kernel, **options), definition(tensor, l_kernel, **options))

        seen = []
        resolve = morphology_module._resolve_engine

        def record(engine, *args):
            seen.append(engine)
            return resolve(engine, *args)

        monkeypatch.setattr(morphology_module, "_resolve_engine", record)
        op(tensor, l_kernel, engine="unfold")
        assert seen == ["unfold", "unfold"]

    @pytest.mark.parametrize("name", ["opening", "closing", "top_hat", "bottom_hat"])
    def test_convention_composite_convolution_handles_infinite_intermediates_4734(self, device, dtype, name):
        if not dtype.is_floating_point:
            pytest.skip("Infinity regression requires a floating-point dtype.")

        op = COMPOSITES[name][0]
        kernel = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        tensor = torch.tensor([[[[0.2, 0.5, 0.9, 0.4]]]], device=device, dtype=dtype)

        convolved = op(tensor, kernel, engine="convolution")
        unfolded = op(tensor, kernel, engine="unfold")

        assert not torch.isnan(convolved).any()
        self.assert_close(convolved, unfolded)
