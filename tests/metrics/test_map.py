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

from testing.base import BaseTester


class TestMeanAveragePrecision(BaseTester):
    def test_smoke(self, device, dtype):
        boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
        labels = torch.tensor([1], device=device, dtype=torch.long)
        scores = torch.tensor([0.7], device=device, dtype=dtype)

        gt_boxes = torch.tensor([[100, 50, 150, 100.0]], device=device, dtype=dtype)
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

        with pytest.raises(AssertionError):
            _ = kornia.metrics.mean_average_precision(boxes[0], [labels], [scores], [gt_boxes], [gt_labels], 2)
