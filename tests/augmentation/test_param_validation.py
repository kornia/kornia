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

from typing import cast

import pytest
import torch

from kornia.augmentation.utils.param_validation import _common_param_check, _range_bound, _tuple_range_reader


class TestParamValidation:
    def test_common_param_check(self):
        # Valid cases
        _common_param_check(batch_size=1, same_on_batch=True)
        _common_param_check(batch_size=0, same_on_batch=False)

        # Invalid batch size
        with pytest.raises(AssertionError):
            _common_param_check(batch_size=-1)

        # Invalid same_on_batch type
        with pytest.raises(AssertionError):
            _common_param_check(batch_size=1, same_on_batch=cast(bool, "invalid"))

    def test_tuple_range_reader_valid(self):
        device = torch.device("cpu")
        # Case 1: single float should produce [-x, x]
        res = _tuple_range_reader(10.0, 2, device=device)
        assert res.shape == (2, 2)
        assert torch.allclose(res, torch.tensor([[-10.0, 10.0], [-10.0, 10.0]]))

        # Case 2: tuple (min, max) should be repeated for target_size
        res = _tuple_range_reader((5.0, 10.0), 2, device=device)
        assert res.shape == (2, 2)
        assert torch.allclose(res, torch.tensor([[5.0, 10.0], [5.0, 10.0]]))

        # Case 3: tensor input (min, max)
        res = _tuple_range_reader(torch.tensor([5.0, 10.0]), 2, device=device)
        assert res.shape == (2, 2)
        assert torch.allclose(res, torch.tensor([[5.0, 10.0], [5.0, 10.0]]))

    def test_tuple_range_reader_errors(self):
        # Negative single number
        with pytest.raises(ValueError):
            _tuple_range_reader(-10, 2)

        # Invalid type
        with pytest.raises(TypeError):
            _tuple_range_reader("invalid", 2)

    def test_range_bound_errors(self):
        # Test factor < 0 for single number
        with pytest.raises(ValueError):
            _range_bound(-1.0, "param", center=0, bounds=(-10, 10))
