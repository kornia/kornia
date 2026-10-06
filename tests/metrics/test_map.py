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
        # is predicted but has no objects: every detection is a false positive and AP = 0. The recall denominators
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

        expected = torch.tensor([7 / 11, 3 / 11, 0.0], device=device, dtype=dtype)
        self.assert_close(torch.tensor([ap[1], ap[2], ap[3]], device=device, dtype=dtype), expected)
        self.assert_close(mean_ap, expected.mean())

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
