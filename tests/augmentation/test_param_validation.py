

import pytest
import torch

from typing import cast
from kornia.augmentation.utils.param_validation import _common_param_check, _range_bound, _tuple_range_reader


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
        _common_param_check(batch_size=batch_size, same_on_batch=same_on_batch)

    @pytest.mark.parametrize("batch_size", [-1])
    def test_common_param_check_invalid_batch_size(self, batch_size):
        with pytest.raises(AssertionError):
            _common_param_check(batch_size=batch_size)

    @pytest.mark.parametrize("same_on_batch", [cast(bool, "invalid")])
    def test_common_param_check_invalid_same_on_batch(self, same_on_batch):
        with pytest.raises(AssertionError):
            _common_param_check(batch_size=1, same_on_batch=same_on_batch)


    @pytest.mark.parametrize(
        "input_param, target_size, expected",
        [
            (10.0, 2, torch.tensor([[-10.0, 10.0], [-10.0, 10.0]])),
            ((5.0, 10.0), 2, torch.tensor([[5.0, 10.0], [5.0, 10.0]])),
            (torch.tensor([5.0, 10.0]), 2, torch.tensor([[5.0, 10.0], [5.0, 10.0]])),
            ([5.0, 10.0], 2, torch.tensor([[5.0, 10.0], [5.0, 10.0]])),
            (torch.tensor([1.0, 2.0]), 2, torch.tensor([[1.0, 2.0], [1.0, 2.0]])),
            ([(5.0, 10.0), (3.0, 8.0)], 2, torch.tensor([[5.0, 10.0], [3.0, 8.0]])),
            (10.0, 1, torch.tensor([[-10.0, 10.0]])),
            (torch.tensor([[5.0, 10.0], [3.0, 8.0]]), 2, torch.tensor([[5.0, 10.0], [3.0, 8.0]])),
        ],
    )
    def test_tuple_range_reader_valid(self, input_param, target_size, expected):
        device = torch.device("cpu")
        res = _tuple_range_reader(input_param, target_size, device=device)
        assert res.shape == (target_size, 2)
        assert torch.allclose(res, expected)

    @pytest.mark.parametrize(
        "args, kwargs, expected_exception, match_msg",
        [
            ((-10, 2), {}, ValueError, None),
            (("invalid", 2), {}, TypeError, None),
            # Invalid tensor shape (2x3)
            ((torch.rand(2, 3), 2), {}, ValueError, "Degrees must be a"),
            # Wrong length list (3 elements)
            (([1, 2, 3], 2), {}, TypeError, "If not pass a tensor"),
            # Mixed types
            ((["a", 1.0], 2), {}, TypeError, "If not pass a tensor"),
        ]
    )
    def test_tuple_range_reader_errors(self, args, kwargs, expected_exception, match_msg):
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
        ]
    )
    def test_range_bound_errors(self, factor, center, bounds, check, expected_exception, match_msg):
        with pytest.raises(expected_exception, match=match_msg):
            _range_bound(factor, "param", center=center, bounds=bounds, check=check)

    @pytest.mark.parametrize(
        "factor, center, bounds, check, expected",
        [
            # Float input: [center - x, center + x] clamped to bounds
            (10.0, 0, (-10, 10), "singular", torch.tensor([-10.0, 10.0])),
            (10.0, 0, (-5, 5), "singular", torch.tensor([-5.0, 5.0])), # Clamped
            (0.2, 1.0, (0, 2), "singular", torch.tensor([0.8, 1.2])),

            # Tuple/List input: [min, max] strictly within bounds
            ((5.0, 10.0), 0, None, "singular", torch.tensor([5.0, 10.0])),
            ([-5.0, 5.0], 0, (-10, 10), "singular", torch.tensor([-5.0, 5.0])),

            # Tensor input
            (torch.tensor([5.0, 10.0]), 0, None, "singular", torch.tensor([5.0, 10.0])),

            # Check singular allows min > max
            ((10.0, 5.0), 0, None, "singular", torch.tensor([10.0, 5.0])),
        ]
    )
    def test_range_bound_valid(self, factor, center, bounds, check, expected):
        res = _range_bound(factor, "param", center=center, bounds=bounds, check=check)
        assert torch.allclose(res, expected)

    # test_range_bound_errors_expanded merged into test_range_bound_errors