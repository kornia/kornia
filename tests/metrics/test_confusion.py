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

import torch

import kornia

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
