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
from kornia.core._compat import torch_version_lt
from kornia.core.exceptions import BaseError

from testing.base import BaseTester


class TestConfusionMatrix(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("label_dtype", [torch.long, torch.uint8])
    @pytest.mark.parametrize("normalized", [False, True])
    def test_transposed_label_maps(self, device, batch_size, label_dtype, normalized):
        pred = torch.tensor([[[0, 1, 2], [2, 1, 0]], [[2, 0, 1], [1, 2, 0]]], device=device, dtype=label_dtype)[
            :batch_size
        ].transpose(1, 2)
        target = torch.tensor([[[0, 2, 2], [1, 1, 0]], [[2, 0, 0], [1, 2, 1]]], device=device, dtype=label_dtype)[
            :batch_size
        ].transpose(1, 2)
        expected = torch.tensor(
            [[[2, 0, 0], [0, 1, 1], [0, 1, 1]], [[1, 1, 0], [1, 1, 0], [0, 0, 2]]],
            device=device,
            dtype=torch.float32,
        )[:batch_size]
        if normalized:
            expected = expected / 2

        assert not pred.is_contiguous()
        assert not target.is_contiguous()
        self.assert_close(kornia.metrics.confusion_matrix(pred, target, 3, normalized), expected)

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
            [[[4 / 7, 1 / 7, 2 / 7], [3 / 5, 0, 2 / 5], [1 / 4, 2 / 4, 1 / 4]]],
            device=device,
            dtype=torch.float32,
        )

        self.assert_close(conf_mat, conf_mat_real)

    def test_normalized_target_rows_batch(self, device, dtype):
        pred = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 0], [0, 1, 1, 2, 2, 2, 2, 2]], device=device, dtype=torch.long)
        target = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 2, 2, 2]], device=device, dtype=torch.long)
        counts = torch.tensor(
            [[[2, 1, 0], [0, 2, 0], [1, 0, 2]], [[1, 2, 2], [0, 0, 0], [0, 0, 3]]],
            device=device,
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [[[2 / 3, 1 / 3, 0], [0, 1, 0], [1 / 3, 0, 2 / 3]], [[1 / 5, 2 / 5, 2 / 5], [0, 0, 0], [0, 0, 1]]],
            device=device,
            dtype=torch.float32,
        )

        self.assert_close(kornia.metrics.confusion_matrix(pred, target, 3), counts)
        normalized = kornia.metrics.confusion_matrix(pred, target, 3, normalized=True)
        self.assert_close(normalized, expected)
        self.assert_close(
            normalized.sum(dim=2), torch.tensor([[1, 1, 1], [1, 0, 1]], device=device, dtype=torch.float32)
        )
        assert not torch.allclose(normalized, counts / (counts.sum(dim=1, keepdim=True) + 1e-6))
        assert torch.count_nonzero(normalized[1, 1]) == 0
        assert normalized.dtype == torch.float32
        assert normalized.device == pred.device

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

    def test_zero_pixel_samples_give_zero_matrices(self, device, dtype):
        # The range check reads the label minimum, which a sample without pixels does not have: it is skipped there.
        for shape in ((2, 0), (2, 3, 0)):
            labels = torch.zeros(shape, device=device, dtype=torch.long)
            expected = torch.zeros(2, 3, 3, device=device, dtype=torch.float32)
            self.assert_close(kornia.metrics.confusion_matrix(labels, labels, num_classes=3), expected, rtol=0, atol=0)

    @pytest.mark.skipif(torch_version_lt(2, 14, 0), reason="torch.export cannot capture the bincount before torch 2.14")
    def test_export_counts_like_eager(self, device, dtype):
        # The range check reads the data, which torch.export cannot capture, so it is skipped under export.
        class _ConfusionMatrix(torch.nn.Module):
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return kornia.metrics.confusion_matrix(pred, target, num_classes=3)

        pred = torch.tensor([[0, 1, 2, 2], [1, 1, 0, 2]], device=device)
        target = torch.tensor([[0, 2, 2, 1], [1, 0, 0, 2]], device=device)
        exported = torch.export.export(_ConfusionMatrix(), (pred, target), strict=True).module()
        for p, t in ((pred, target), (target, pred.flip(-1))):
            self.assert_close(exported(p, t), kornia.metrics.confusion_matrix(p, t, num_classes=3), rtol=0, atol=0)


class TestConventionsConfusionMatrix(BaseTester):
    # Two (H, W) = (3, 4) label maps, three classes; the class counts differ between target and prediction (rows 3, 5,
    # 4 against columns 3, 4, 5 in sample 0) and between the two samples.
    PRED = [[[0, 0, 1, 2], [1, 1, 2, 2], [0, 2, 2, 1]], [[2, 2, 2, 2], [0, 1, 1, 0], [0, 0, 1, 2]]]
    TARGET = [[[0, 1, 1, 2], [1, 1, 1, 2], [0, 0, 2, 2]], [[2, 2, 1, 1], [0, 0, 1, 0], [1, 0, 1, 2]]]

    def test_convention_confusion_matrix_rows_are_targets_per_sample(self, device, dtype):
        """confusion_matrix returns one float32 count matrix per sample, cm[b, target, prediction]."""
        pred = torch.tensor(self.PRED, device=device)
        target = torch.tensor(self.TARGET, device=device)
        # Snippet used to generate expected (scikit-learn 1.9.0, same orientation: rows y_true, columns y_pred):
        #   [sklearn.metrics.confusion_matrix(np.ravel(t), np.ravel(p), labels=[0, 1, 2]) for p, t in zip(PRED, TARGET)]
        expected = torch.tensor(
            [[[2.0, 0.0, 1.0], [1.0, 3.0, 1.0], [0.0, 1.0, 3.0]], [[3.0, 1.0, 0.0], [1.0, 2.0, 2.0], [0.0, 0.0, 3.0]]],
            device=device,
        )
        cm = kornia.metrics.confusion_matrix(pred, target, 3)
        assert cm.dtype == torch.float32
        self.assert_close(cm, expected)
        # swapping the arguments transposes every matrix
        self.assert_close(kornia.metrics.confusion_matrix(target, pred, 3), expected.transpose(-2, -1))
        # relabel: renaming class c to perm[c] in both maps permutes the rows and the columns alike
        perm = torch.tensor([2, 0, 1], device=device)
        inv = torch.argsort(perm)
        self.assert_close(kornia.metrics.confusion_matrix(perm[pred], perm[target], 3), expected[:, inv][:, :, inv])
        # the first axis is always the batch: a flat (N,) label vector gives N one-pixel matrices, not one matrix
        flat = kornia.metrics.confusion_matrix(pred[0, 0], target[0, 0], 3)
        assert flat.shape == (4, 3, 3)
        self.assert_close(flat.sum(0), torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device))
