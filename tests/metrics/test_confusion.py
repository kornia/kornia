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

import kornia
from kornia.core.exceptions import BaseError

from testing.base import BaseTester


class TestConfusionMatrix(BaseTester):
    def test_two_classes(self, device, dtype):
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor([[[3, 1], [0, 4]]], device=device, dtype=torch.float32)
        self.assert_close(conf_mat, conf_mat_real)

    def test_two_classes_batch2(self, device, dtype):
        batch_size = 2
        num_classes = 2
        actual = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long).repeat(batch_size, 1)
        predicted = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1]], device=device, dtype=torch.long).repeat(batch_size, 1)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor([[[3, 1], [0, 4]], [[3, 1], [0, 4]]], device=device, dtype=torch.float32)
        self.assert_close(conf_mat, conf_mat_real)

    def test_three_classes(self, device, dtype):
        num_classes = 3
        actual = torch.tensor([[2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 1, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor([[[4, 1, 2], [3, 0, 2], [1, 2, 1]]], device=device, dtype=torch.float32)
        self.assert_close(conf_mat, conf_mat_real)

    def test_four_classes_one_missing(self, device, dtype):
        num_classes = 4
        actual = torch.tensor([[3, 3, 1, 1, 2, 1, 1, 3, 2, 2, 1, 1, 2, 3, 2, 1]], device=device, dtype=torch.long)
        predicted = torch.tensor([[3, 2, 1, 1, 1, 1, 1, 2, 1, 3, 3, 2, 1, 1, 3, 3]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 0, 0], [0, 4, 1, 2], [0, 3, 0, 2], [0, 1, 2, 1]]], device=device, dtype=torch.float32
        )
        self.assert_close(conf_mat, conf_mat_real)

    def test_three_classes_normalized(self, device, dtype):
        num_classes = 3
        normalized = True
        actual = torch.tensor([[2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 1, 0]], device=device, dtype=torch.long)
        predicted = torch.tensor([[2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2]], device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes, normalized)

        conf_mat_real = torch.tensor(
            [[[0.5000, 0.3333, 0.4000], [0.3750, 0.0000, 0.4000], [0.1250, 0.6667, 0.2000]]],
            device=device,
            dtype=torch.float32,
        )

        self.assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_perfect(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]], device=device, dtype=torch.float32
        )
        self.assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_nonperfect(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[0, 0, 1, 1], [0, 3, 0, 1], [2, 2, 1, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[3, 0, 0, 1], [1, 3, 0, 0], [0, 0, 4, 0], [0, 1, 0, 3]]], device=device, dtype=torch.float32
        )
        self.assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_missing(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[3, 3, 1, 1], [3, 3, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 0, 4], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]], device=device, dtype=torch.float32
        )
        self.assert_close(conf_mat, conf_mat_real)

    def test_four_classes_2d_one_class_no_predicted(self, device, dtype):
        num_classes = 4
        actual = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )
        predicted = torch.tensor(
            [[[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]]], device=device, dtype=torch.long
        )

        conf_mat = kornia.metrics.confusion_matrix(predicted, actual, num_classes)
        conf_mat_real = torch.tensor(
            [[[0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]], device=device, dtype=torch.float32
        )
        self.assert_close(conf_mat, conf_mat_real)

    def test_empty_batch(self, device, dtype):
        num_classes = 3
        pred = torch.zeros(0, 4, 4, device=device, dtype=torch.long)
        target = torch.zeros(0, 4, 4, device=device, dtype=torch.long)

        conf_mat = kornia.metrics.confusion_matrix(pred, target, num_classes)
        assert conf_mat.shape == (0, num_classes, num_classes)
        assert conf_mat.dtype == torch.float32
        assert conf_mat.device == pred.device

    def test_exception_shape_mismatch(self, device, dtype):
        pred = torch.zeros(1, 4, dtype=torch.long, device=device)
        target = torch.zeros(1, 5, dtype=torch.long, device=device)
        with pytest.raises(ValueError, match="same shape"):
            kornia.metrics.confusion_matrix(pred, target, num_classes=2)

    def test_exception_num_classes_too_small(self, device, dtype):
        pred = torch.zeros(1, 4, dtype=torch.long, device=device)
        with pytest.raises(ValueError, match="at least two"):
            kornia.metrics.confusion_matrix(pred, pred, num_classes=1)

    def test_exception_not_integer_labels(self, device, dtype):
        # The two type checks were dead (#5549): a list reached `.dtype` and a float tensor reached `bincount`.
        labels = torch.tensor([[0, 1, 0]], device=device, dtype=torch.long)
        with pytest.raises(TypeError, match="pred must be a tensor"):
            kornia.metrics.confusion_matrix([[0, 1, 0]], labels, num_classes=3)
        with pytest.raises(BaseError, match="target must have an integer dtype"):
            kornia.metrics.confusion_matrix(labels, labels.to(dtype), num_classes=3)

    def test_exception_out_of_range(self, device, dtype):
        # An out-of-range prediction landed in another cell and an out-of-range target in a raw torch error (#5549).
        for pred, target, name, span in (
            ([0, 3, 2], [0, 1, 2], "pred", r"\[0, 3\]"),
            ([-1, 1, 2], [0, 1, 2], "pred", r"\[-1, 2\]"),
            ([0, 1, 2], [0, 3, 2], "target", r"\[0, 3\]"),
            ([0, 1, 2], [0, 255, 2], "target", r"\[0, 255\]"),
        ):
            message = rf"Input {name} must contain values in \[0, 3\)\. Got values in {span}"
            with pytest.raises(BaseError, match=message):
                kornia.metrics.confusion_matrix(
                    torch.tensor([pred], device=device), torch.tensor([target], device=device), num_classes=3
                )

    @pytest.mark.parametrize("label_dtype, num_classes", [(torch.uint8, 21), (torch.int16, 182), (torch.int32, 21)])
    def test_small_integer_dtypes_match_int64(self, label_dtype, num_classes, device, dtype):
        # The cell index was formed in the label dtype and wrapped, for uint8 from num_classes = 17 (#5549).
        labels = torch.tensor([[[num_classes - 1, 0, 5], [5, num_classes - 1, 0]]], device=device, dtype=label_dtype)
        actual = kornia.metrics.confusion_matrix(labels, labels, num_classes)
        expected = kornia.metrics.confusion_matrix(labels.long(), labels.long(), num_classes)
        assert actual.dtype == torch.float32
        self.assert_close(actual, expected, rtol=0, atol=0)
        assert actual[0].nonzero().tolist() == [[0, 0], [5, 5], [num_classes - 1, num_classes - 1]]

    def test_bool_masks_count_as_binary_labels(self, device, dtype):
        pred = torch.tensor([[True, False, True, True]], device=device)
        target = torch.tensor([[True, False, False, True]], device=device)
        expected = torch.tensor([[[1, 1], [0, 2]]], device=device, dtype=torch.float32)
        self.assert_close(kornia.metrics.confusion_matrix(pred, target, num_classes=2), expected)
