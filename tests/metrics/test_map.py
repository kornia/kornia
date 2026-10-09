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


class TestMeanAveragePrecision(BaseTester):
    @pytest.mark.parametrize("coordinates", [[[100, 50, 150, 100.0]], [[0, 0, 512, 512]]])
    def test_smoke(self, coordinates, device, dtype):
        boxes = torch.tensor(coordinates, device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.7], device=device, dtype=dtype)

        gt_boxes = torch.tensor(coordinates, device=device, dtype=dtype)
        gt_labels = torch.tensor([1], device=device, dtype=torch.long)

        mean_ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)

        self.assert_close(mean_ap[0], torch.tensor(1.0, device=device, dtype=dtype))
        self.assert_close(mean_ap[1][1], 1.0)

    def test_recall_per_class(self, device, dtype):
        # Two objects of different classes in one image, each detected exactly. Recall is taken
        # over the objects of the class being scored, so both classes reach recall 1 and AP 1.
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], device=device, dtype=dtype)
        labels = torch.tensor([1, 2], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)

        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [boxes], [labels], 3)

        self.assert_close(mean_ap, torch.tensor(1.0, device=device, dtype=dtype))
        self.assert_close(ap[1], 1.0)
        self.assert_close(ap[2], 1.0)

    def test_recall_per_class_over_images(self, device, dtype):
        # Two images. Class 1 has 3 objects and 2 exact detections: recall 1/3, 2/3 at precision 1, so 7 of the 11
        # recall thresholds (0 to 0.6) are reached and AP = 7/11. Class 2 has 2 objects, the higher-scored detection
        # is a false positive and the other is exact: recall 0, 1/2 at precision 0, 1/2, so AP = 6 * 0.5 / 11. Class 3
        # is predicted but has no objects: AP is undefined and excluded from mAP. The recall denominators
        # (3, 2, 0) differ from the detection counts (2, 2, 1), from the detected counts and from the total (5).
        gt_boxes = [
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]]),
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        ]
        gt_labels = [torch.tensor([1, 1, 2]), torch.tensor([1, 2])]
        boxes = [
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 50.0, 50.0]]),
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [60.0, 60.0, 70.0, 70.0], [80.0, 80.0, 90.0, 90.0]]),
        ]
        labels = [torch.tensor([1, 2]), torch.tensor([1, 2, 3])]
        scores = [torch.tensor([0.9, 0.8]), torch.tensor([0.7, 0.95, 0.6])]

        def to(tensors, dtype):
            return [t.to(device=device, dtype=dtype) for t in tensors]

        mean_ap, ap = kornia.metrics.mean_average_precision(
            to(boxes, dtype),
            to(labels, torch.long),
            to(scores, dtype),
            to(gt_boxes, dtype),
            to(gt_labels, torch.long),
            4,
        )

        expected = torch.tensor([7 / 11, 3 / 11, -1.0], device=device, dtype=dtype)
        self.assert_close(torch.tensor([ap[1], ap[2], ap[3]], device=device, dtype=dtype), expected)
        self.assert_close(mean_ap, expected[:2].mean())

    @pytest.mark.parametrize("n_classes", [3, 4, 5])
    def test_absent_classes_do_not_change_map_5540(self, device, dtype, n_classes):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [30.0, 5.0, 45.0, 12.0]], device=device, dtype=dtype)
        labels = torch.tensor([1, 2], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)

        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [boxes], [labels], n_classes)

        self.assert_close(mean_ap, boxes.new_tensor(1.0))
        assert mean_ap.shape == torch.Size([])
        assert mean_ap.dtype == dtype
        assert mean_ap.device == boxes.device
        assert set(ap) == set(range(1, n_classes))
        self.assert_close(ap[1], 1.0)
        self.assert_close(ap[2], 1.0)
        assert all(ap[c] == -1.0 for c in range(3, n_classes))

    def test_detections_without_ground_truth_5540(self, device, dtype):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [30.0, 5.0, 45.0, 12.0]], device=device, dtype=dtype)
        labels = torch.tensor([1, 2], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)

        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [boxes[:1]], [labels[:1]], 3)

        self.assert_close(mean_ap, boxes.new_tensor(1.0))
        self.assert_close(ap[1], 1.0)
        assert ap[2] == -1.0

    @pytest.mark.parametrize("n_detections", [0, 1])
    def test_ground_truth_without_detections_5540(self, device, dtype, n_detections):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [30.0, 5.0, 45.0, 12.0]], device=device, dtype=dtype)
        labels = torch.tensor([1, 2], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)

        mean_ap, ap = kornia.metrics.mean_average_precision(
            [boxes[:n_detections]], [labels[:n_detections]], [scores[:n_detections]], [boxes], [labels], 4
        )

        self.assert_close(mean_ap, boxes.new_tensor(n_detections / 2))
        self.assert_close(ap[1], float(n_detections))
        assert ap[2] == 0.0
        assert ap[3] == -1.0

    @pytest.mark.parametrize("n_classes", [1, 3])
    @pytest.mark.parametrize("with_background_gt", [False, True])
    @pytest.mark.parametrize("with_detections", [False, True])
    def test_no_foreground_ground_truth_5540(self, device, dtype, n_classes, with_background_gt, with_detections):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0]], device=device, dtype=dtype)
        labels = torch.tensor([n_classes - 1], device=device, dtype=torch.long)
        scores = torch.tensor([0.9], device=device, dtype=dtype)
        gt_labels = torch.zeros(1, device=device, dtype=torch.long)

        mean_ap, ap = kornia.metrics.mean_average_precision(
            [boxes[: int(with_detections)]],
            [labels[: int(with_detections)]],
            [scores[: int(with_detections)]],
            [boxes[: int(with_background_gt)]],
            [gt_labels[: int(with_background_gt)]],
            n_classes,
        )

        self.assert_close(mean_ap, boxes.new_tensor(-1.0))
        assert mean_ap.dtype == dtype
        assert mean_ap.device == boxes.device
        assert ap == dict.fromkeys(range(1, n_classes), -1.0)

    @pytest.mark.parametrize("invalid_label", [-1, 3, 5])
    @pytest.mark.parametrize("label_source", ["pred_labels", "gt_labels"])
    def test_label_range_5540(self, device, dtype, invalid_label, label_source):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        invalid = torch.tensor([invalid_label], device=device, dtype=torch.long)
        scores = torch.tensor([0.9], device=device, dtype=dtype)
        pred_labels = [labels, invalid if label_source == "pred_labels" else labels]
        gt_labels = [labels, invalid if label_source == "gt_labels" else labels]

        with pytest.raises(BaseError, match=rf"{label_source} must satisfy 0 <= label < n_classes \(3\)"):
            kornia.metrics.mean_average_precision([boxes] * 2, pred_labels, [scores] * 2, [boxes] * 2, gt_labels, 3)

    @pytest.mark.parametrize("label_source", ["pred_labels", "gt_labels"])
    def test_fractional_labels_5629(self, device, dtype, label_source):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        invalid = torch.tensor([1.5], device=device, dtype=dtype)
        scores = torch.tensor([0.9], device=device, dtype=dtype)
        pred_labels = [labels, invalid if label_source == "pred_labels" else labels]
        gt_labels = [labels, invalid if label_source == "gt_labels" else labels]

        with pytest.raises(BaseError, match=rf"{label_source} must contain integer-valued labels"):
            kornia.metrics.mean_average_precision([boxes] * 2, pred_labels, [scores] * 2, [boxes] * 2, gt_labels, 3)

    @pytest.mark.parametrize("label_source", ["pred_labels", "gt_labels", "both"])
    def test_integer_valued_float_labels_5629(self, device, dtype, label_source):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [30.0, 5.0, 45.0, 12.0]], device=device, dtype=dtype)
        labels = torch.tensor([1, 2], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)
        expected_map, expected_ap = kornia.metrics.mean_average_precision(
            [boxes], [labels], [scores], [boxes], [labels], 3
        )
        pred_labels = labels.to(dtype) if label_source in ("pred_labels", "both") else labels
        gt_labels = labels.to(dtype) if label_source in ("gt_labels", "both") else labels

        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [pred_labels], [scores], [boxes], [gt_labels], 3)

        self.assert_close(expected_map, boxes.new_tensor(1.0))
        self.assert_close(expected_ap[1], 1.0)
        self.assert_close(expected_ap[2], 1.0)
        self.assert_close(mean_ap, expected_map)
        assert ap == expected_ap

    @pytest.mark.parametrize("invalid_label", [-1.0, 3.0, -0.5, 3.5])
    @pytest.mark.parametrize("label_source", ["pred_labels", "gt_labels"])
    def test_float_label_range_5629(self, device, dtype, invalid_label, label_source):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0]], device=device, dtype=dtype)
        labels = torch.tensor([1.0], device=device, dtype=dtype)
        invalid = torch.tensor([invalid_label], device=device, dtype=dtype)
        scores = torch.tensor([0.9], device=device, dtype=dtype)
        pred_labels = [invalid if label_source == "pred_labels" else labels]
        gt_labels = [invalid if label_source == "gt_labels" else labels]

        with pytest.raises(BaseError, match=rf"{label_source} must satisfy 0 <= label < n_classes \(3\)"):
            kornia.metrics.mean_average_precision([boxes], pred_labels, [scores], [boxes], gt_labels, 3)

    def test_background_is_excluded(self, device, dtype):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [30.0, 5.0, 45.0, 12.0]], device=device, dtype=dtype)
        labels = torch.tensor([0, 1], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)

        # The foreground detection misses its object. A perfect background detection must not raise mAP to 0.5.
        gt_boxes = boxes.clone()
        gt_boxes[1] += 100
        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [gt_boxes], [labels], 2)
        self.assert_close(mean_ap, boxes.new_tensor(0.0))
        assert ap == {1: 0.0}

    def test_false_positive_in_image_without_class_ground_truth(self, device, dtype):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.9], device=device, dtype=dtype)

        # Class 1 has GT in the second image: its higher-scored detection in the first image still counts as FP.
        mean_ap, ap = kornia.metrics.mean_average_precision(
            [boxes, boxes], [labels, labels], [scores, scores - 0.1], [boxes[:0], boxes], [labels[:0], labels], 2
        )
        self.assert_close(mean_ap, boxes.new_tensor(0.5))
        self.assert_close(ap[1], 0.5)

    def test_raise(self, device, dtype):
        boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.7], device=device, dtype=dtype)

        gt_boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        gt_labels = torch.tensor([1], device=device, dtype=torch.long)

        with pytest.raises(BaseError, match="same length"):
            _ = kornia.metrics.mean_average_precision(boxes[0], [labels], [scores], [gt_boxes], [gt_labels], 2)

    def test_exception_list_lengths(self, device, dtype):
        # A list-length mismatch names the five lengths; a per-image size mismatch names the sizes (#5551).
        boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [30.0, 5.0, 45.0, 12.0]], device=device, dtype=dtype)
        labels = torch.tensor([1, 2], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)

        with pytest.raises(BaseError, match=r"same length.*pred_boxes 2.*gt_boxes 1"):
            kornia.metrics.mean_average_precision([boxes] * 2, [labels] * 2, [scores] * 2, [boxes], [labels], 3)
        with pytest.raises(BaseError, match=r"same length.*pred_scores 1"):
            kornia.metrics.mean_average_precision([boxes] * 2, [labels] * 2, [scores], [boxes] * 2, [labels] * 2, 3)
        with pytest.raises(BaseError, match=r"one row per object.*2 boxes and 1 labels"):
            kornia.metrics.mean_average_precision([boxes], [labels], [scores], [boxes], [labels[:1]], 3)
        with pytest.raises(BaseError, match=r"one row per detection.*2 boxes, 2 labels and 1 scores"):
            kornia.metrics.mean_average_precision([boxes], [labels], [scores[:1]], [boxes], [labels], 3)
        with pytest.raises(BaseError, match=r"one row per detection.*1 boxes, 2 labels and 2 scores"):
            kornia.metrics.mean_average_precision([boxes[:1]], [labels], [scores], [boxes], [labels], 3)
        # each of the five lists in turn one image short
        lists = [[boxes] * 2, [labels] * 2, [scores] * 2, [boxes] * 2, [labels] * 2]
        for i, name in enumerate(("pred_boxes", "pred_labels", "pred_scores", "gt_boxes", "gt_labels")):
            args = [x[:1] if j == i else x for j, x in enumerate(lists)]
            with pytest.raises(BaseError, match=rf"same length.*{name} 1"):
                kornia.metrics.mean_average_precision(*args, 3)

    def test_exception_per_image_counts(self, device, dtype):
        # The counts are checked per image, so mismatches that cancel over the images do not pair the rows of one
        # image with another's (#5580).
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device, dtype=dtype)
        b = torch.tensor([[50.0, 50.0, 60.0, 60.0]], device=device, dtype=dtype)
        none = torch.zeros(0, 4, device=device, dtype=dtype)
        boxes = [torch.cat([a, b]), none]
        labels = [torch.tensor([1, 1], device=device), torch.tensor([], device=device, dtype=torch.long)]
        scores = [torch.tensor([0.9, 0.8], device=device, dtype=dtype), torch.zeros(0, device=device, dtype=dtype)]
        one_each = [torch.tensor([1], device=device), torch.tensor([1], device=device)]
        one_score_each = [
            torch.tensor([0.9], device=device, dtype=dtype),
            torch.tensor([0.8], device=device, dtype=dtype),
        ]

        mean_ap, _ = kornia.metrics.mean_average_precision(boxes, labels, scores, boxes, labels, 2)
        self.assert_close(mean_ap, torch.tensor(1.0, device=device, dtype=dtype))
        with pytest.raises(BaseError, match=r"one row per object in every image. Got image 0: 2 boxes and 1 labels"):
            kornia.metrics.mean_average_precision(boxes, labels, scores, boxes, one_each, 2)
        with pytest.raises(
            BaseError, match=r"one row per detection in every image. Got image 0: 2 boxes, 1 labels and 1 scores"
        ):
            kornia.metrics.mean_average_precision(boxes, one_each, one_score_each, boxes, labels, 2)
        # the first image is consistent, so the second is named
        with pytest.raises(BaseError, match=r"Got image 1: 2 boxes and 1 labels"):
            kornia.metrics.mean_average_precision(
                [a, a, b],
                one_each + one_each[:1],
                one_score_each + one_score_each[:1],
                [a, boxes[0], none],
                one_each + one_each[:1],
                2,
            )
        # the first image holds the extra row, of each kind the checks compare in turn
        with pytest.raises(BaseError, match=r"Got image 0: 1 boxes and 2 labels"):
            kornia.metrics.mean_average_precision(boxes, labels, scores, [a, b], labels, 2)
        with pytest.raises(BaseError, match=r"Got image 0: 2 boxes, 1 labels and 2 scores"):
            kornia.metrics.mean_average_precision(boxes, one_each, scores, boxes, labels, 2)
        with pytest.raises(BaseError, match=r"Got image 0: 1 boxes, 1 labels and 2 scores"):
            kornia.metrics.mean_average_precision([a, b], one_each, scores, boxes, labels, 2)

    @pytest.mark.parametrize("box_dtype", [torch.int64, torch.int32, torch.uint8])
    def test_integer_boxes_match_the_float_result(self, device, box_dtype):
        # Integer boxes give the AP of their float copies in float32, as mean_iou_bbox computes them (#5551). The
        # ranked detections are TP, FP, TP: recall 0.5, 0.5, 1 at precision 1, 0.5, 2/3, so AP = (6 + 5 * 2 / 3) / 11.
        gt_boxes = torch.tensor([[0, 0, 10, 20], [30, 5, 45, 12]], device=device, dtype=box_dtype)
        gt_labels = torch.tensor([1, 1], device=device, dtype=torch.long)
        boxes = torch.tensor([[0, 0, 10, 20], [100, 100, 120, 130], [30, 5, 45, 12]], device=device, dtype=box_dtype)
        labels = torch.tensor([1, 1, 1], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)

        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)
        expected, expected_ap = kornia.metrics.mean_average_precision(
            [boxes.float()], [labels], [scores], [gt_boxes.float()], [gt_labels], 2
        )

        assert mean_ap.dtype == torch.float32
        self.assert_close(mean_ap, torch.tensor((6 + 5 * 2 / 3) / 11, device=device))
        self.assert_close(mean_ap, expected, rtol=0, atol=0)
        assert ap == expected_ap

    @pytest.mark.parametrize(
        "pred_dtype, gt_dtype",
        [
            (torch.int64, torch.float16),
            (torch.float16, torch.int64),
            (torch.float16, torch.float32),
            (torch.float32, torch.float16),
        ],
    )
    def test_mixed_box_dtypes_compute_in_the_promoted_floating_dtype(self, device, pred_dtype, gt_dtype):
        # Integer boxes count as float32, and the two box sets meet in their promoted dtype, as in mean_iou_bbox. The
        # TP/FP counts used to take the prediction dtype and the precisions the ground-truth dtype, so float16 boxes
        # with float32 ones gave a float16 AP. Same TP, FP, TP ranking as above.
        gt_boxes = torch.tensor([[0, 0, 10, 20], [30, 5, 45, 12]], device=device)
        gt_labels = torch.tensor([1, 1], device=device, dtype=torch.long)
        boxes = torch.tensor([[0, 0, 10, 20], [100, 100, 120, 130], [30, 5, 45, 12]], device=device)
        labels = torch.tensor([1, 1, 1], device=device, dtype=torch.long)
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)

        mean_ap, ap = kornia.metrics.mean_average_precision(
            [boxes.to(pred_dtype)], [labels], [scores], [gt_boxes.to(gt_dtype)], [gt_labels], 2
        )
        expected, expected_ap = kornia.metrics.mean_average_precision(
            [boxes.float()], [labels], [scores], [gt_boxes.float()], [gt_labels], 2
        )

        assert mean_ap.dtype == torch.float32
        self.assert_close(mean_ap, expected, rtol=0, atol=0)
        assert ap == expected_ap

    def test_recall_on_an_exact_tenth_reaches_its_threshold_5083(self, device, dtype):
        # 10 objects, ranked detections TP, FP, then 9 TP: recall passes through every tenth, precision drops at the FP.
        gt_boxes = torch.tensor([[i * 20.0, 0.0, i * 20.0 + 10.0, 10.0] for i in range(10)], device=device, dtype=dtype)
        gt_labels = torch.ones(10, device=device, dtype=torch.long)
        boxes = torch.cat(
            [gt_boxes[:1], torch.tensor([[500.0, 500.0, 510.0, 510.0]], device=device, dtype=dtype), gt_boxes[1:]]
        )
        labels = torch.ones(11, device=device, dtype=torch.long)
        scores = torch.linspace(1.0, 0.5, 11, device=device, dtype=dtype)

        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)

        # Precision 1 at the recall thresholds 0 and 0.1, then 10/11 at the nine others: (2 + 9 * 10 / 11) / 11
        expected = torch.tensor(112.0 / 121.0, device=device, dtype=dtype)
        self.assert_close(mean_ap, expected)
        self.assert_close(torch.tensor(ap[1], device=device, dtype=dtype), expected)

    def test_recall_on_every_tenth_reaches_its_threshold_5083(self, device, dtype):
        # 20 objects, ranked detections TP, then (FP, TP) 19 times: the j-th TP has recall j / 20 and precision
        # j / (2j - 1), above every later precision. Recall 2i / 20 is the first to reach the threshold i / 10, exactly,
        # so each of the 11 thresholds sets its own term, and a threshold it misses takes the next TP's lower precision.
        gt_boxes = torch.tensor([[i * 20.0, 0.0, i * 20.0 + 10.0, 10.0] for i in range(20)], device=device, dtype=dtype)
        gt_labels = torch.ones(20, device=device, dtype=torch.long)
        fp_box = torch.tensor([[500.0, 500.0, 510.0, 510.0]], device=device, dtype=dtype)
        boxes = torch.cat([gt_boxes[:1]] + [torch.cat([fp_box, gt_boxes[j : j + 1]]) for j in range(1, 20)])
        labels = torch.ones(39, device=device, dtype=torch.long)
        scores = torch.linspace(1.0, 0.5, 39, device=device, dtype=dtype)

        mean_ap, ap = kornia.metrics.mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)

        # Precision 1 at the threshold 0, then 2i / (4i - 1) at the threshold i / 10.
        expected = torch.tensor((1.0 + sum(2 * i / (4 * i - 1) for i in range(1, 11))) / 11, device=device, dtype=dtype)
        self.assert_close(mean_ap, expected)
        self.assert_close(torch.tensor(ap[1], device=device, dtype=dtype), expected)


class TestConventionsMeanAveragePrecision(BaseTester):
    BOX_1 = [0.0, 0.0, 10.0, 20.0]  # 10 x 20
    BOX_2 = [30.0, 30.0, 45.0, 40.0]  # 15 x 10
    FAR = [100.0, 100.0, 120.0, 110.0]  # overlaps nothing

    @staticmethod
    def _map(device, dtype, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, n_classes, threshold=0.5):
        """Run mean_average_precision on per-image lists of Python lists."""
        return kornia.metrics.mean_average_precision(
            [torch.tensor(b, device=device, dtype=dtype) for b in pred_boxes],
            [torch.tensor(lbl, device=device, dtype=torch.long) for lbl in pred_labels],
            [torch.tensor(s, device=device, dtype=dtype) for s in pred_scores],
            [torch.tensor(b, device=device, dtype=dtype) for b in gt_boxes],
            [torch.tensor(lbl, device=device, dtype=torch.long) for lbl in gt_labels],
            n_classes,
            threshold,
        )

    def _assert_aps(self, ap, expected, device, dtype):
        """Compare the per-class dict: the same class ids, and values close in the box dtype."""
        assert sorted(ap) == sorted(expected)
        keys = sorted(expected)
        self.assert_close(
            torch.tensor([ap[k] for k in keys], device=device, dtype=dtype),
            torch.tensor([expected[k] for k in keys], device=device, dtype=dtype),
        )

    def test_convention_mean_average_precision_pools_detections_over_images(self, device, dtype):
        """mAP ranks the detections of a class over all images at once and returns (0-d mAP, {class: AP}) fractions."""
        # Image A: one exact detection (score 0.5). Image B: a false positive (0.9), then an exact detection (0.3).
        # Pooled ranking FP, TP, TP over 2 objects: precision 2/3 at recall 1, so AP = 2/3. Image by image the APs
        # are 1 and 1/2 (mean 3/4). COCO (pycocotools 2.0.11 via torchmetrics 1.9.0) also pools: 0.666667.
        mean_ap, ap = self._map(
            device,
            dtype,
            [[self.BOX_1], [self.FAR, self.BOX_2]],
            [[1], [1, 1]],
            [[0.5], [0.9, 0.3]],
            [[self.BOX_1], [self.BOX_2]],
            [[1], [1]],
            2,
        )
        assert mean_ap.shape == ()
        assert mean_ap.dtype == dtype
        self.assert_close(mean_ap, torch.tensor(2.0 / 3.0, device=device, dtype=dtype))
        assert isinstance(ap, dict)
        assert isinstance(ap[1], float)
        self._assert_aps(ap, {1: 2.0 / 3.0}, device, dtype)

    def test_convention_mean_average_precision_class_zero_is_background(self, device, dtype):
        """Classes 1 ... n_classes - 1 are scored; class 0 objects and detections are never looked at."""
        # Class 0 has an object that nobody detects and a confident detection on the class-1 box; scored like any
        # class (as COCO does) it would add AP 0. Class 1: one exact detection, AP 1. Class 2: a false positive ranked
        # above the exact detection, precision 1/2 at recall 1, AP 1/2.
        class_zero_box = [50.0, 0.0, 60.0, 30.0]
        pred = ([[self.BOX_1, self.BOX_1, self.FAR, self.BOX_2]], [[0.99, 0.9, 0.95, 0.6]])
        gt_boxes = [[class_zero_box, self.BOX_1, self.BOX_2]]
        for c1, c2 in ((1, 2), (2, 1)):  # relabel: swapping classes 1 and 2 swaps their entries, the mAP stays
            mean_ap, ap = self._map(device, dtype, pred[0], [[0, c1, c2, c2]], pred[1], gt_boxes, [[0, c1, c2]], 3)
            self.assert_close(mean_ap, torch.tensor(0.75, device=device, dtype=dtype))
            self._assert_aps(ap, {c1: 1.0, c2: 0.5}, device, dtype)

    def test_convention_mean_average_precision_missed_class_scores_zero(self, device, dtype):
        """A class with ground truth and no detection has AP 0 and enters the mean, as in COCO."""
        # COCO (pycocotools 2.0.11 via torchmetrics 1.9.0) gives the same (0.5, {1: 1.0, 2: 0.0})
        mean_ap, ap = self._map(device, dtype, [[self.BOX_1]], [[1]], [[0.9]], [[self.BOX_1, self.BOX_2]], [[1, 2]], 3)
        self.assert_close(mean_ap, torch.tensor(0.5, device=device, dtype=dtype))
        self._assert_aps(ap, {1: 1.0, 2: 0.0}, device, dtype)

    def test_convention_mean_average_precision_iou_must_exceed_threshold(self, device, dtype):
        """A detection matches only when its IoU is strictly greater than threshold."""
        # [0, 0, 10, 10] against the 10 x 20 box: IoU exactly 0.5. py-faster-rcnn's voc_eval uses the same `>`, but its
        # +1 areas give this pair IoU 0.524, so py-faster-rcnn, the VOC devkit (VOCevaldet.m `ovmax >= minoverlap`) and
        # COCO all count it as a match (AP 1).
        half_box = [[[0.0, 0.0, 10.0, 10.0]]]
        for threshold, expected in ((0.5, 0.0), (0.4999, 1.0)):
            mean_ap, ap = self._map(device, dtype, half_box, [[1]], [[0.9]], [[self.BOX_1]], [[1]], 2, threshold)
            self.assert_close(mean_ap, torch.tensor(expected, device=device, dtype=dtype))
            self._assert_aps(ap, {1: expected}, device, dtype)

    def test_convention_mean_average_precision_greedy_voc_matching(self, device, dtype):
        """A detection takes its highest-IoU object; if that one is already matched, it is a false positive."""
        objects = [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 16.0]]]
        # The second detection overlaps the first object (IoU 10/11) more than the second one (IoU 0.6875 > 0.5). The
        # first object is taken, so it is a false positive: TP, FP over 2 objects -> recall 1/2 at precision 1 ->
        # 6 of the 11 recall thresholds -> AP 6/11. COCO matches it to the free second object instead: AP 1.
        detections = [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 11.0]]]
        mean_ap, _ = self._map(device, dtype, detections, [[1, 1]], [[0.9, 0.8]], objects, [[1, 1]], 2)
        self.assert_close(mean_ap, torch.tensor(6.0 / 11.0, device=device, dtype=dtype))
        # The highest-IoU object, not the first one above the threshold: each detection equals one object and overlaps
        # the other at IoU 5/6, so both are true positives -> AP 1. Taking the first object above the threshold would
        # match the second detection to the taken first object, a false positive -> AP 6/11.
        objects = [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 12.0]]]
        mean_ap, _ = self._map(device, dtype, objects, [[1, 1]], [[0.9, 0.8]], objects, [[1, 1]], 2)
        self.assert_close(mean_ap, torch.tensor(1.0, device=device, dtype=dtype))

    def test_wart_mean_average_precision_drops_fractional_labels_5629(self, device, dtype):
        """A fractional label passes the range check and matches no class, so its boxes are dropped (#5629)."""
        boxes = [torch.tensor([self.BOX_1, self.BOX_2], device=device, dtype=dtype)]
        scores = [torch.tensor([0.9, 0.8], device=device, dtype=dtype)]

        def run(labels):
            labels = [torch.tensor(labels, device=device, dtype=dtype)]
            return kornia.metrics.mean_average_precision(boxes, labels, scores, boxes, labels, 3)

        # control: integer-valued floating-point labels give the result of int64 labels, two perfect classes
        expected_map, expected_ap = kornia.metrics.mean_average_precision(
            boxes, [torch.tensor([1, 2], device=device)], scores, boxes, [torch.tensor([1, 2], device=device)], 3
        )
        mean_ap, ap = run([1.0, 2.0])
        self.assert_close(mean_ap, expected_map)
        self._assert_aps(ap, expected_ap, device, dtype)
        self._assert_aps(ap, {1: 1.0, 2: 1.0}, device, dtype)
        # label 1.25 (exact in every float dtype; rounding or truncating it gives class 1, so those changes flip this
        # pin too): its detection and its object enter no class, so class 1 reads "no objects" (-1) and the mAP is
        # class 2's alone, without an error
        mean_ap, ap = run([1.25, 2.0])
        self.assert_close(mean_ap, torch.tensor(1.0, device=device, dtype=dtype))
        self._assert_aps(ap, {1: -1.0, 2: 1.0}, device, dtype)
