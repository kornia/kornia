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

import kornia.augmentation as K
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
    def test_tuple_range_reader_valid(self, input_param, target_size, expected, device):
        """Supported input formats should expand correctly across devices."""
        res = _tuple_range_reader(input_param, target_size, device=device)
        assert res.shape == (target_size, 2)
        torch.testing.assert_close(res, expected.to(device))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tuple_range_reader_honours_cuda_device(self):
        """`_tuple_range_reader` allocates its own tensors, so the CUDA branch needs its own test.

        The `device` fixture above follows `--device`, which is `cpu` in CI and in a default local
        run, so the cross-product alone would never place the reader on an accelerator.
        """
        cuda = torch.device("cuda")
        res = _tuple_range_reader([(5.0, 10.0), (3.0, 8.0)], 2, device=cuda)
        assert res.device.type == "cuda"
        torch.testing.assert_close(res, torch.tensor([[5.0, 10.0], [3.0, 8.0]], device=cuda))

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
            # The scalar rows name the whole message on purpose. The reported tensor is the unclamped
            # `[center - x, center + x]`, which only the scalar branch's own raise produces: with that raise gone
            # the trailing `_joint_range_check` / `_singular_range_check` still rejects the same inputs,
            # but reports the lower end already floored (`[-5., 10.]`, `[-0.5000, 0.6000]`, `[0., 4.]`).
            (10.0, 0, (-5, 5), "singular", ValueError, r"param out of bounds\. .*got tensor\(\[-10\.,  10\.\]\)"),
            (0.6, 0, (-0.5, 0.5), "joint", ValueError, r"param out of bounds\. .*got tensor\(\[-0\.6000,  0\.6000\]\)"),
            (3.0, 1.0, (0, 2), "joint", ValueError, r"param out of bounds\. .*got tensor\(\[-2\.,  4\.\]\)"),
            # -inf hits the earlier "non negative" guard, not the finiteness check: the row pins that order.
            (float("-inf"), 0, (0, float("inf")), "joint", ValueError, "must be non negative"),
            (float("inf"), 0, (0, float("inf")), "joint", ValueError, "must be finite"),
            (float("nan"), 0, (0, float("inf")), "joint", ValueError, "must be finite"),
            (torch.tensor(float("inf")), 0, (0, float("inf")), "joint", ValueError, "must be finite"),
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
            (1.5, 1.0, (0, float("inf")), "joint", torch.tensor([0.0, 2.5], dtype=torch.float32)),
            (0.5, 0.0, (0, float("inf")), "joint", torch.tensor([0.0, 0.5], dtype=torch.float32)),
            (50.0, 50.0, (1, 100), "joint", torch.tensor([1.0, 100.0], dtype=torch.float32)),
            (0.2, 1.0, (0, 2), "singular", torch.tensor([0.8, 1.2], dtype=torch.float32)),
            ((5.0, 10.0), 0, None, "singular", torch.tensor([5.0, 10.0], dtype=torch.float32)),
            ([-5.0, 5.0], 0, (-10, 10), "singular", torch.tensor([-5.0, 5.0], dtype=torch.float32)),
            (torch.tensor([5.0, 10.0]), 0, None, "singular", torch.tensor([5.0, 10.0], dtype=torch.float32)),
            ((10.0, 5.0), 0, None, "singular", torch.tensor([10.0, 5.0], dtype=torch.float32)),
        ],
        ids=[
            "float-symmetric-full",
            "float-floored-at-zero",
            "float-floored-at-zero-center-zero",
            "float-floored-at-lower-bound",
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

    # `_range_bound` is shared, so rejecting a scalar past the upper bound (#4563) reaches the geometric
    # classes as well as the intensity ones, and each reports the unclamped `[center - x, center + x]`.
    # The 3D `degrees`, `shears` and `angle` arguments read their scalar through `_tuple_range_reader`
    # instead, which has no bound of its own, so they are deliberately not covered here.
    # Snippet used to generate expected:
    #   for ctor in (lambda: K.RandomRotation(400.0), lambda: K.RandomShear(400.0),
    #                lambda: K.RandomTranslate(2.0, 0.1), lambda: K.RandomRotation90(5),
    #                lambda: K.RandomAffine(30.0, translate=2.0)):
    #       try: ctor()
    #       except ValueError as e: print(e)
    # executed 2026-09-16 (torch 2.14.0, cpu) -> the five messages pinned below.
    @pytest.mark.parametrize(
        "ctor, match_msg",
        [
            (lambda: K.RandomRotation(400.0), r"degrees out of bounds\. .*got tensor\(\[-400\.,  400\.\]\)"),
            (lambda: K.RandomShear(400.0), r"shear-x out of bounds\. .*got tensor\(\[-400\.,  400\.\]\)"),
            (lambda: K.RandomTranslate(2.0, 0.1), r"translate_x out of bounds\. .*got tensor\(\[-2\.,  2\.\]\)"),
            (lambda: K.RandomRotation90(5), r"times out of bounds\. .*got tensor\(\[-5\.,  5\.\]\)"),
            (lambda: K.RandomAffine(30.0, translate=2.0), r"translate out of bounds\. .*got tensor\(\[-2\.,  2\.\]\)"),
        ],
        ids=["rotation-degrees", "shear", "translate", "rotation90-times", "affine-translate"],
    )
    def test_geometric_scalar_magnitude_past_the_upper_bound_raises(self, ctor, match_msg):
        """A scalar that overshoots a geometric parameter's bound is rejected at construction."""
        with pytest.raises(ValueError, match=match_msg):
            ctor()

    # The 3D readers took a scalar or an explicit range without any bound, so an angle past one
    # turn was sampled instead of rejected (#4617). Executed 2026-09-17 (torch 2.5.0, cpu).
    @pytest.mark.parametrize(
        "ctor, match_msg",
        [
            (lambda: K.RandomRotation3D(400.0), r"degrees out of bounds\. .*got tensor\(\[-400\.,  400\.\]\)"),
            (
                lambda: K.RandomRotation3D((-400.0, 400.0)),
                r"degrees out of bounds\. .*got tensor\(\[-400\.,  400\.\]\)",
            ),
            (lambda: K.RandomAffine3D(400.0), r"degrees out of bounds\. .*got tensor\(\[-400\.,  400\.\]\)"),
            (
                lambda: K.RandomAffine3D(30.0, shears=400.0),
                r"shears out of bounds\. .*got tensor\(\[-400\.,  400\.\]\)",
            ),
            (
                lambda: K.RandomMotionBlur3D(3, 400.0, 0.5),
                r"angle out of bounds\. .*got tensor\(\[-400\.,  400\.\]\)",
            ),
        ],
        ids=["rotation3d-scalar", "rotation3d-range", "affine3d-degrees", "affine3d-shears", "motion-blur3d-angle"],
    )
    def test_3d_angle_past_one_turn_raises(self, ctor, match_msg):
        """A 3D angle range wider than one turn is rejected, as the 2D one is."""
        with pytest.raises(ValueError, match=match_msg):
            ctor()

    @pytest.mark.parametrize(
        "ctor",
        [
            lambda: K.RandomRotation3D(360.0),
            lambda: K.RandomRotation3D((-360.0, 360.0)),
            lambda: K.RandomRotation3D((10.0, 20.0, 30.0)),
            lambda: K.RandomAffine3D(30.0, shears=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0)),
            lambda: K.RandomMotionBlur3D(3, 360.0, 0.5),
        ],
        ids=["at-the-bound", "explicit-at-the-bound", "per-axis", "per-axis-shears", "motion-blur-at-the-bound"],
    )
    def test_3d_angles_inside_the_bound_still_construct(self, ctor):
        """The bound is inclusive, and the per-axis forms are unaffected."""
        assert ctor() is not None
