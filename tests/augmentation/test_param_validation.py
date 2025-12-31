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

from kornia.augmentation.utils.param_validation import (
    _common_param_check,
    _range_bound,
    _tuple_range_reader,
)


class TestParamValidation:
    @pytest.mark.parametrize(
        "batch_size, same_on_batch",
        [
            (1, True),
            (0, False),
            (1, None),
        ],
    )
    def test_common_param_check_valid(self, batch_size, same_on_batch):
        """Valid combinations of batch_size and same_on_batch should not raise."""
        _common_param_check(batch_size=batch_size, same_on_batch=same_on_batch)

    @pytest.mark.parametrize("batch_size", [-1])
    def test_common_param_check_invalid_batch_size(self, batch_size):
        """Negative batch_size should raise an assertion error."""
        with pytest.raises(AssertionError):
            _common_param_check(batch_size=batch_size)

    @pytest.mark.parametrize("same_on_batch", [cast(bool, "invalid")])
    def test_common_param_check_invalid_same_on_batch(self, same_on_batch):
        """
        Invalid runtime values for same_on_batch should raise.

        typing.cast is used to inject an invalid value at runtime
        without breaking static type checking of the test itself.
        """
        with pytest.raises(AssertionError):
            _common_param_check(batch_size=1, same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        "input_param, target_size, expected",
        [
            (10.0, 2, torch.tensor([[-10.0, 10.0], [-10.0, 10.0]], dtype=torch.float32)),
            ((5.0, 10.0), 2, torch.tensor([[5.0, 10.0], [5.0, 10.0]], dtype=torch.float32)),
            (torch.tensor([5.0, 10.0]), 2, torch.tensor([[5.0, 10.0], [5.0, 10.0]], dtype=torch.float32)),
            ([5.0, 10.0], 2, torch.tensor([[5.0, 10.0], [5.0, 10.0]], dtype=torch.float32)),
            (torch.tensor([1.0, 2.0]), 2, torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)),
            ([(5.0, 10.0), (3.0, 8.0)], 2, torch.tensor([[5.0, 10.0], [3.0, 8.0]], dtype=torch.float32)),
            (10.0, 1, torch.tensor([[-10.0, 10.0]], dtype=torch.float32)),
            (
                torch.tensor([[5.0, 10.0], [3.0, 8.0]]),
                2,
                torch.tensor([[5.0, 10.0], [3.0, 8.0]], dtype=torch.float32),
            ),
        ],
        ids=[
            "float-symmetric-2",
            "tuple-2",
            "tensor-1d",
            "list",
            "tensor-1d-alt",
            "list-of-tuples",
            "float-symmetric-1",
            "tensor-2d",
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            torch.device("cpu"),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            ),
        ],
    )
    def test_tuple_range_reader_valid(self, input_param, target_size, expected, device):
        """Supported input formats should expand correctly across devices."""
        res = _tuple_range_reader(input_param, target_size, device=device)
        assert res.shape == (target_size, 2)
        torch.testing.assert_close(res, expected.to(device))

    @pytest.mark.parametrize(
        "args, kwargs, expected_exception, match_msg",
        [
            ((-10, 2), {}, ValueError, None),
            (("invalid", 2), {}, TypeError, None),
            ((torch.rand(2, 3), 2), {}, ValueError, "Degrees must be a"),
            (([1, 2, 3], 2), {}, TypeError, "If not pass a torch.tensor"),
            ((["a", 1.0], 2), {}, TypeError, "If not pass a torch.tensor"),
        ],
    )
    def test_tuple_range_reader_errors(self, args, kwargs, expected_exception, match_msg):
        """Invalid inputs should raise the appropriate exception."""
        if match_msg is None:
            with pytest.raises(expected_exception):
                _tuple_range_reader(*args, **kwargs)
        else:
            with pytest.raises(expected_exception, match=match_msg):
                _tuple_range_reader(*args, **kwargs)

    @pytest.mark.parametrize(
        "factor, center, bounds, check, expected_exception, match_msg",
        [
            (-1.0, 0, (-10, 10), "singular", ValueError, None),
            (10.0, 0, None, "singular", ValueError, "`center` and `bounds` cannot be None"),
            ((-10, 10), 0, (-5, 5), "singular", ValueError, "param out of bounds"),
            ((10, 5), 0, None, "joint", ValueError, "should be smaller than"),
            ("invalid", 0, (-10, 10), "singular", TypeError, None),
            ((-10.0, 10.0), 0, (-5, 5), "singular", ValueError, "param out of bounds"),
        ],
    )
    def test_range_bound_errors(self, factor, center, bounds, check, expected_exception, match_msg):
        """Invalid parameter combinations should raise."""
        if match_msg is None:
            with pytest.raises(expected_exception):
                _range_bound(factor, "param", center=center, bounds=bounds, check=check)
        else:
            with pytest.raises(expected_exception, match=match_msg):
                _range_bound(factor, "param", center=center, bounds=bounds, check=check)

    @pytest.mark.parametrize(
        "factor, center, bounds, check, expected",
        [
            (10.0, 0, (-10, 10), "singular", torch.tensor([-10.0, 10.0], dtype=torch.float32)),
            (10.0, 0, (-5, 5), "singular", torch.tensor([-5.0, 5.0], dtype=torch.float32)),
            (0.2, 1.0, (0, 2), "singular", torch.tensor([0.8, 1.2], dtype=torch.float32)),
            ((5.0, 10.0), 0, None, "singular", torch.tensor([5.0, 10.0], dtype=torch.float32)),
            ([-5.0, 5.0], 0, (-10, 10), "singular", torch.tensor([-5.0, 5.0], dtype=torch.float32)),
            (torch.tensor([5.0, 10.0]), 0, None, "singular", torch.tensor([5.0, 10.0], dtype=torch.float32)),
            ((10.0, 5.0), 0, None, "singular", torch.tensor([10.0, 5.0], dtype=torch.float32)),
        ],
        ids=[
            "float-clamp-full",
            "float-clamp-partial",
            "float-center-offset",
            "tuple-input",
            "list-input",
            "tensor-input",
            "singular-min-gt-max",
        ],
    )
    def test_range_bound_valid(self, factor, center, bounds, check, expected):
        """Valid inputs should produce the expected bounded range."""
        res = _range_bound(factor, "param", center=center, bounds=bounds, check=check)
        torch.testing.assert_close(res, expected)
