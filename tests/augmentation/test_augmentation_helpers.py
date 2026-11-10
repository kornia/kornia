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

import re

import pytest
import torch

from kornia.augmentation.utils.helpers import _validate_shape, override_parameters


class TestValidateShape:
    def test_accepts_any_listed_rank(self):
        _validate_shape(torch.Size([1, 2, 3, 4]), required_shapes=("BCHW", "BCDHW"))
        _validate_shape(torch.Size([1, 2, 3, 4, 5]), required_shapes=("BCHW", "BCDHW"))

    def test_error_lists_every_required_shape(self):
        shape = torch.Size([3, 4])
        message = f"Expected input shape in ('BCHW', 'BCDHW'). Got {shape}."
        with pytest.raises(TypeError, match=re.escape(message)):
            _validate_shape(shape, required_shapes=("BCHW", "BCDHW"))


class TestOverrideParameters:
    @pytest.mark.parametrize("in_place", [False, True])
    def test_missing_key_is_ignored(self, in_place):
        params = {"angle": 1}
        result = override_parameters(params, {"angle": 2, "typo": 3}, in_place=in_place)
        assert result == {"angle": 2}
        assert params == {"angle": 2 if in_place else 1}

    def test_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="typo"):
            override_parameters({"angle": 1}, {"typo": 2}, if_none_exist="raise")

    @pytest.mark.parametrize("overrides", [None, {}])
    def test_no_override_preserves_identity(self, overrides):
        params = {"angle": 1}
        assert override_parameters(params, overrides) is params

    def test_invalid_missing_key_policy(self):
        with pytest.raises(ValueError, match="not a valid option"):
            override_parameters({"angle": 1}, {"typo": 2}, if_none_exist="invalid")
