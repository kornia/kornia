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

import copy

import pytest
import torch

from kornia.augmentation import (
    AugmentationSequential,
    RandomCutMixV2,
    RandomJigsaw,
    RandomMixUpV2,
    RandomMosaic,
    RandomTransplantation,
    RandomTransplantation3D,
)

from testing.base import BaseTester


class TestRandomMixUpV2(BaseTester):
    def test_smoke(self, device, dtype):
        f = RandomMixUpV2()
        repr = "RandomMixUpV2(lambda_val=None, p=1.0, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr, str(f)

    def test_random_mixup_p1(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)
        lam = torch.tensor([0.1320, 0.3074], device=device, dtype=dtype)

        expected = torch.stack(
            [
                torch.ones(1, 3, 4, device=device, dtype=dtype) * (1 - lam[0]),
                torch.ones(1, 3, 4, device=device, dtype=dtype) * lam[1],
            ]
        )

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[:, 0], label)
        self.assert_close(out_label[:, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        self.assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_p0(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(p=0.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

        expected = input.clone()

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_lam0(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(lambda_val=(0.0, 0.0), p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)
        lam = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

        expected = input.clone()

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[:, 0], label)
        self.assert_close(out_label[:, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        self.assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_same_on_batch(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(same_on_batch=True, p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)
        lam = torch.tensor([0.0885, 0.0885], device=device, dtype=dtype)

        expected = torch.stack(
            [
                torch.ones(1, 3, 4, device=device, dtype=dtype) * (1 - lam[0]),
                torch.ones(1, 3, 4, device=device, dtype=dtype) * lam[1],
            ]
        )

        out_image, out_label = f(input, label)
        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[:, 0], label)
        self.assert_close(out_label[:, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        self.assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_boxes_concatenated_xyxy(self, device, dtype):
        # Boxes are non-spatial annotations: MixUp must concatenate, not blend.
        # With B=2, N=2 boxes each, output must be (B, 2N, 4).
        torch.manual_seed(1)
        f = RandomMixUpV2(p=1.0, data_keys=["input", "bbox_xyxy"])

        images = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)
        boxes = torch.tensor(
            [[[1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 6.0, 6.0]], [[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 5.0, 5.0]]],
            device=device,
            dtype=dtype,
        )

        _, out_boxes = f(images, boxes)

        # Output must double the object dimension.
        assert out_boxes.shape == (2, 4, 4), f"Expected (2, 4, 4), got {out_boxes.shape}"

        # First N boxes for each image are the original boxes.
        self.assert_close(out_boxes[:, :2, :], boxes, rtol=1e-4, atol=1e-4)

        # Second N boxes for each image are the boxes from the paired (permuted) image.
        pairs = f._params["mixup_pairs"]
        self.assert_close(out_boxes[:, 2:, :], boxes[pairs], rtol=1e-4, atol=1e-4)

    def test_boxes_concatenated_xywh(self, device, dtype):
        torch.manual_seed(1)
        f = RandomMixUpV2(p=1.0, data_keys=["input", "bbox_xywh"])

        images = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)
        boxes = torch.tensor(
            [[[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]], [[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 4.0, 4.0]]],
            device=device,
            dtype=dtype,
        )

        _, out_boxes = f(images, boxes)

        assert out_boxes.shape == (2, 4, 4), f"Expected (2, 4, 4), got {out_boxes.shape}"

    def test_boxes_preserved_when_p0(self, device, dtype):
        # When p=0 (no augmentation) boxes must pass through unchanged.
        torch.manual_seed(1)
        f = RandomMixUpV2(p=0.0, data_keys=["input", "bbox_xyxy"])

        images = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)
        boxes = torch.tensor(
            [[[1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 6.0, 6.0]], [[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 5.0, 5.0]]],
            device=device,
            dtype=dtype,
        )

        _, out_boxes = f(images, boxes)

        # Shape and values unchanged: no concatenation when transform is skipped.
        assert out_boxes.shape == boxes.shape, f"Expected {boxes.shape}, got {out_boxes.shape}"
        self.assert_close(out_boxes, boxes, rtol=1e-4, atol=1e-4)

    def test_boxes_image_label_unchanged(self, device, dtype):
        # Adding bbox key must not change image or label output.
        torch.manual_seed(42)
        images = torch.rand(2, 1, 4, 4, device=device, dtype=dtype)
        label = torch.tensor([0, 1], device=device, dtype=dtype)
        boxes = torch.tensor(
            [[[0.0, 0.0, 2.0, 2.0]], [[1.0, 1.0, 3.0, 3.0]]],
            device=device,
            dtype=dtype,
        )

        f_no_box = RandomMixUpV2(p=1.0, data_keys=["input", "class"])
        torch.manual_seed(42)
        out_img_ref, out_lbl_ref = f_no_box(images.clone(), label.clone())

        f_with_box = RandomMixUpV2(p=1.0, data_keys=["input", "class", "bbox_xyxy"])
        torch.manual_seed(42)
        out_img, out_lbl, _ = f_with_box(images.clone(), label.clone(), boxes.clone())

        self.assert_close(out_img, out_img_ref, rtol=1e-4, atol=1e-4)
        self.assert_close(out_lbl, out_lbl_ref, rtol=1e-4, atol=1e-4)


class TestRandomCutMixV2(BaseTester):
    def test_smoke(self):
        f = RandomCutMixV2(data_keys=["input", "class"], use_correct_lambda=True)
        expected_repr = "RandomCutMixV2(cut_size=None, beta=None, num_mix=1, p=1.0, p_batch=1.0, same_on_batch=False)"
        assert str(f) == expected_repr

    def test_random_mixup_p1(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMixV2(p=1.0, data_keys=["input", "class"], use_correct_lambda=True)

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[0, :, 0], label)
        self.assert_close(out_label[0, :, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        self.assert_close(out_label[0, :, 2], torch.tensor([0.5, 0.5], device=device, dtype=dtype))

    def test_random_mixup_p0(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMixV2(p=0.0, data_keys=["input", "class"], use_correct_lambda=True)

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device)

        expected = input.clone()
        exp_label = torch.tensor([[[1, 1, 0], [0, 0, 0]]], device=device, dtype=dtype)

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label, exp_label)

    def test_random_mixup_beta0(self, device, dtype):
        torch.manual_seed(76)
        # beta 0 => resample 0.5 area
        # beta cannot be 0 after torch 1.8.0
        f = RandomCutMixV2(beta=1e-7, p=1.0, data_keys=["input", "class"], use_correct_lambda=True)

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[0, :, 0], label)
        self.assert_close(out_label[0, :, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        # cut area = 4 / 12, but with use_correct_lambda=True the lambda calculation is different
        self.assert_close(
            out_label[0, :, 2],
            torch.tensor([0.66667, 0.66667], device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_random_mixup_num2(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMixV2(num_mix=5, p=1.0, data_keys=["input", "class"], use_correct_lambda=True)

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[:, :, 0], label.view(1, -1).expand(5, 2))
        self.assert_close(
            out_label[:, :, 1], torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], device=device, dtype=dtype)
        )
        # Updated expected values for use_correct_lambda=True
        self.assert_close(
            out_label[:, :, 2],
            torch.tensor(
                [[0.9167, 0.6667], [1.0, 0.8333], [0.5, 0.9167], [0.9167, 1.0], [0.5, 0.6667]],
                device=device,
                dtype=dtype,
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_random_mixup_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        f = RandomCutMixV2(same_on_batch=True, p=1.0, data_keys=["input", "class"], use_correct_lambda=True)

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_label[0, :, 0], label)
        self.assert_close(out_label[0, :, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        self.assert_close(
            out_label[0, :, 2], torch.tensor([0.5000, 0.5000], device=device, dtype=dtype), rtol=1e-4, atol=1e-4
        )

    def test_boxes_target_fully_outside_cut_kept(self, device, dtype):
        # A target box that does not intersect the cut rectangle must be kept unchanged.
        #
        # Setup:
        #   image size 8x8, cut rect (xyxy_plus) [2,2,5,5] (4x4 pixels),
        #   target box [6,6,7,7] xyxy_plus (bottom-right corner, fully outside cut).
        #   source (permuted image) has a single zero-area box at [0,0,0,0].
        from kornia.geometry.bbox import bbox_generator

        f = RandomCutMixV2(p=1.0, data_keys=["input", "bbox_xyxy"], use_correct_lambda=True, min_area=1.0)
        imgs = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)

        # xyxy: width = xmax - xmin, so [6,6,7,7] xyxy_plus -> [6,6,8,8] xyxy
        boxes = torch.tensor([[[6.0, 6.0, 8.0, 8.0]], [[0.0, 0.0, 1.0, 1.0]]], device=device, dtype=dtype)  # (2, 1, 4)

        # cut rect: x_start=2, y_start=2, width=4, height=4 -> TL=(2,2), BR=(5,5)
        crop = bbox_generator(
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
        ).unsqueeze(0)  # (1, 2, 4, 2)

        params = {
            "mix_pairs": torch.tensor([[1, 0]], device=device),
            "crop_src": crop,
            "image_shape": torch.tensor([8.0, 8.0], device=device, dtype=dtype),
            "batch_prob": torch.tensor([True, True]),
            "dtype": torch.tensor(6),
        }

        _, out_boxes = f(imgs, boxes, params=params)
        # Image 0 output: first slot = target box kept, second = clipped source (zero-area, zeroed)
        # Target box [6,6,8,8] is fully outside cut → first slot unchanged
        self.assert_close(
            out_boxes[0, 0],
            torch.tensor([6.0, 6.0, 8.0, 8.0], device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_boxes_target_fully_inside_cut_dropped(self, device, dtype):
        # A target box that is completely covered by the cut rectangle must be zeroed out.
        #
        # Setup:
        #   cut rect (xyxy_plus) [2,2,5,5]; target box [2,2,5,5] xyxy_plus (same as cut -> fully inside).
        from kornia.geometry.bbox import bbox_generator

        f = RandomCutMixV2(p=1.0, data_keys=["input", "bbox_xyxy"], use_correct_lambda=True, min_area=1.0)
        imgs = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)

        # xyxy_plus [2,2,5,5] -> xyxy [2,2,6,6]
        boxes = torch.tensor([[[2.0, 2.0, 6.0, 6.0]], [[0.0, 0.0, 1.0, 1.0]]], device=device, dtype=dtype)

        crop = bbox_generator(
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
        ).unsqueeze(0)

        params = {
            "mix_pairs": torch.tensor([[1, 0]], device=device),
            "crop_src": crop,
            "image_shape": torch.tensor([8.0, 8.0], device=device, dtype=dtype),
            "batch_prob": torch.tensor([True, True]),
            "dtype": torch.tensor(6),
        }

        _, out_boxes = f(imgs, boxes, params=params)
        # First output slot for image 0 should be zeroed (dropped box).
        # Zeroed xyxy_plus [0,0,0,0] -> xyxy [0,0,1,1]
        self.assert_close(
            out_boxes[0, 0],
            torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_boxes_target_partially_inside_cut_clipped_kept(self, device, dtype):
        # A target box that partially overlaps the cut rectangle is kept if its remaining visible
        # area (original_area - intersection_area) is >= min_area.
        #
        # Setup:
        #   cut rect (xyxy_plus) [2,2,5,5]; target box [1,1,3,3] xyxy_plus.
        #   Intersection = [2,2,3,3] -> area = 1x1 = 1.
        #   Original area = 2x2 = 4.  Visible = 3 >= min_area=1.0 -> kept unchanged.
        from kornia.geometry.bbox import bbox_generator

        f = RandomCutMixV2(p=1.0, data_keys=["input", "bbox_xyxy"], use_correct_lambda=True, min_area=1.0)
        imgs = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)

        # xyxy_plus [1,1,3,3] -> xyxy [1,1,4,4]
        boxes = torch.tensor([[[1.0, 1.0, 4.0, 4.0]], [[0.0, 0.0, 1.0, 1.0]]], device=device, dtype=dtype)

        crop = bbox_generator(
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
        ).unsqueeze(0)

        params = {
            "mix_pairs": torch.tensor([[1, 0]], device=device),
            "crop_src": crop,
            "image_shape": torch.tensor([8.0, 8.0], device=device, dtype=dtype),
            "batch_prob": torch.tensor([True, True]),
            "dtype": torch.tensor(6),
        }

        _, out_boxes = f(imgs, boxes, params=params)
        # Visible area = 3 >= 1.0 -> first slot of image 0 kept at original coords
        self.assert_close(
            out_boxes[0, 0],
            torch.tensor([1.0, 1.0, 4.0, 4.0], device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_boxes_target_partially_inside_cut_dropped_when_below_min_area(self, device, dtype):
        # A target box whose visible area after cut is below min_area must be zeroed out.
        #
        # Setup:
        #   cut rect (xyxy_plus) [2,2,5,5]; target box [1,1,3,3] xyxy_plus.
        #   Visible area = 3.  With min_area=4.0 the box should be dropped.
        from kornia.geometry.bbox import bbox_generator

        f = RandomCutMixV2(p=1.0, data_keys=["input", "bbox_xyxy"], use_correct_lambda=True, min_area=4.0)
        imgs = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)

        # xyxy_plus [1,1,3,3] -> xyxy [1,1,4,4]
        boxes = torch.tensor([[[1.0, 1.0, 4.0, 4.0]], [[0.0, 0.0, 1.0, 1.0]]], device=device, dtype=dtype)

        crop = bbox_generator(
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
        ).unsqueeze(0)

        params = {
            "mix_pairs": torch.tensor([[1, 0]], device=device),
            "crop_src": crop,
            "image_shape": torch.tensor([8.0, 8.0], device=device, dtype=dtype),
            "batch_prob": torch.tensor([True, True]),
            "dtype": torch.tensor(6),
        }

        _, out_boxes = f(imgs, boxes, params=params)
        # Visible area = 3 < min_area=4 -> dropped, zeroed in xyxy_plus -> xyxy [0,0,1,1]
        self.assert_close(
            out_boxes[0, 0],
            torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_boxes_source_inside_cut_kept_clipped(self, device, dtype):
        # A source box that intersects the cut rectangle must be clipped to the cut bounds and added.
        #
        # Setup:
        #   cut rect (xyxy_plus) [2,2,5,5]; source box [4,4,6,6] xyxy_plus.
        #   Intersection with cut = [4,4,5,5] -> area=1 >= min_area=1.0 -> kept, clipped.
        from kornia.geometry.bbox import bbox_generator

        f = RandomCutMixV2(p=1.0, data_keys=["input", "bbox_xyxy"], use_correct_lambda=True, min_area=1.0)
        imgs = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)

        # Target image 0 has no real boxes; source image 1 has the box of interest.
        # xyxy_plus [4,4,6,6] -> xyxy [4,4,7,7]
        boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]], [[4.0, 4.0, 7.0, 7.0]]], device=device, dtype=dtype)

        crop = bbox_generator(
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
        ).unsqueeze(0)

        params = {
            "mix_pairs": torch.tensor([[1, 0]], device=device),
            "crop_src": crop,
            "image_shape": torch.tensor([8.0, 8.0], device=device, dtype=dtype),
            "batch_prob": torch.tensor([True, True]),
            "dtype": torch.tensor(6),
        }

        _, out_boxes = f(imgs, boxes, params=params)
        # Image 0: second slot = clipped source box.
        # Source [4,4,6,6] xyxy_plus clipped to cut [2,2,5,5] -> [4,4,5,5] xyxy_plus -> xyxy [4,4,6,6]
        self.assert_close(
            out_boxes[0, 1],
            torch.tensor([4.0, 4.0, 6.0, 6.0], device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_boxes_source_outside_cut_dropped(self, device, dtype):
        # A source box that has no intersection with the cut rectangle must be zeroed out.
        #
        # Setup:
        #   cut rect (xyxy_plus) [2,2,5,5]; source box [6,6,8,8] xyxy_plus (outside cut).
        from kornia.geometry.bbox import bbox_generator

        f = RandomCutMixV2(p=1.0, data_keys=["input", "bbox_xyxy"], use_correct_lambda=True, min_area=1.0)
        imgs = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)

        # xyxy_plus [6,6,8,8] -> xyxy [6,6,9,9]
        boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]], [[6.0, 6.0, 9.0, 9.0]]], device=device, dtype=dtype)

        crop = bbox_generator(
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([2.0, 2.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
            torch.tensor([4.0, 4.0], device=device, dtype=dtype),
        ).unsqueeze(0)

        params = {
            "mix_pairs": torch.tensor([[1, 0]], device=device),
            "crop_src": crop,
            "image_shape": torch.tensor([8.0, 8.0], device=device, dtype=dtype),
            "batch_prob": torch.tensor([True, True]),
            "dtype": torch.tensor(6),
        }

        _, out_boxes = f(imgs, boxes, params=params)
        # Image 0: second slot = source box fully outside cut -> zeroed -> xyxy [0,0,1,1]
        self.assert_close(
            out_boxes[0, 1],
            torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype),
            rtol=1e-4,
            atol=1e-4,
        )


class TestRandomMosaic(BaseTester):
    def test_smoke(self):
        f = RandomMosaic(data_keys=["input", "class"])
        repr = (
            "RandomMosaic(output_size=None, mosaic_grid=(2, 2), start_ratio_range=(0.3, 0.7), p=0.7,"
            " p_batch=1.0, same_on_batch=False, mosaic_grid=(2, 2), output_size=None, min_bbox_size=0.0,"
            " padding_mode=constant, resample=bilinear, align_corners=True, cropping_mode=slice)"
        )
        assert str(f) == repr

    def test_numerical(self, device, dtype):
        torch.manual_seed(76)
        f = RandomMosaic(p=1.0, data_keys=["input", "bbox_xyxy"])

        input = torch.stack(
            [torch.ones(1, 8, 8, device=device, dtype=dtype), torch.zeros(1, 8, 8, device=device, dtype=dtype)]
        )
        boxes = torch.tensor([[[4, 5, 6, 7], [1, 2, 3, 4]], [[2, 2, 6, 6], [0, 0, 0, 0]]], device=device, dtype=dtype)

        out_image, out_box = f(input, boxes)

        expected = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    ]
                ],
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )

        expected_box = torch.tensor(
            [
                [
                    [0.7074, 0.7099, 2.7074, 2.7099],
                    [0.0000, 0.0000, 1.0000, 1.0000],
                    [0.0000, 5.7099, 2.7074, 8.0000],
                    [0.0000, 2.7099, 1.0000, 4.7099],
                    [7.0000, 0.7099, 8.0000, 2.7099],
                    [5.7074, 0.0000, 7.7074, 1.0000],
                    [7.0000, 7.0000, 8.0000, 8.0000],
                    [5.7074, 5.7099, 7.7074, 7.7099],
                ],
                [
                    [0.0000, 0.0000, 1.0000, 2.8313],
                    [0.0000, 0.0000, 1.0000, 1.0000],
                    [0.0000, 7.0000, 1.0000, 8.0000],
                    [0.0000, 6.8313, 1.0000, 8.0000],
                    [4.5036, 0.0000, 8.0000, 2.8313],
                    [1.5036, 0.0000, 3.5036, 1.0000],
                    [4.5036, 6.8313, 8.0000, 8.0000],
                    [1.5036, 3.8313, 3.5036, 5.8313],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        self.assert_close(out_box, expected_box, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_p(self, p, device, dtype):
        torch.manual_seed(76)
        f = RandomMosaic(output_size=(300, 300), p=p, data_keys=["input", "bbox_xyxy"])

        input = torch.randn((2, 3, 224, 224), device=device, dtype=dtype)
        boxes = torch.tensor(
            [
                # image 1
                [[70.0, 5, 150, 100], [60, 180, 175, 220]],  # head  # feet
                # image 2
                [[75, 30, 175, 140], [0, 0, 0, 0]],  # head  # placeholder
            ],
            device=device,
            dtype=dtype,
        )

        f(input, boxes)

    def test_mask_numerical(self, device, dtype):
        # Verify mask compositing uses same 2x2 grid coordinates as image.
        # Generation snippet:
        #   torch.manual_seed(76)
        #   f = RandomMosaic(p=1.0, data_keys=["input", "mask"])
        #   masks = torch.stack([ones(1,8,8)*10, ones(1,8,8)*20])
        #   out_image, out_mask = f(input_img, masks)
        torch.manual_seed(76)
        f = RandomMosaic(p=1.0, data_keys=["input", "mask"])

        input_img = torch.stack(
            [
                torch.ones(1, 8, 8, device=device, dtype=dtype) * 0.5,
                torch.ones(1, 8, 8, device=device, dtype=dtype) * 0.25,
            ]
        )
        masks = torch.stack(
            [
                torch.ones(1, 8, 8, device=device, dtype=dtype) * 10.0,
                torch.ones(1, 8, 8, device=device, dtype=dtype) * 20.0,
            ]
        )

        _out_image, out_mask = f(input_img, masks)

        assert out_mask.shape == torch.Size([2, 1, 8, 8])
        expected_mask = torch.tensor(
            [
                [
                    [
                        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0],
                    ]
                ],
                [
                    [
                        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out_mask, expected_mask, rtol=1e-4, atol=1e-4)

    def test_mask_output_size(self, device, dtype):
        # Mask is resized to output_size together with image; shape must match.
        torch.manual_seed(76)
        f = RandomMosaic(output_size=(300, 300), p=1.0, data_keys=["input", "mask"])

        input_img = torch.randn(2, 3, 224, 224, device=device, dtype=dtype)
        masks = torch.zeros(2, 1, 224, 224, device=device, dtype=dtype)

        out_image, out_mask = f(input_img, masks)
        assert out_image.shape == torch.Size([2, 3, 300, 300])
        assert out_mask.shape == torch.Size([2, 1, 300, 300])

    def test_mask_non_transform(self, device, dtype):
        # When p=0.0 the mask is padded to output_size but not composited.
        torch.manual_seed(76)
        f = RandomMosaic(output_size=(300, 300), p=0.0, data_keys=["input", "mask"])

        input_img = torch.randn(2, 3, 224, 224, device=device, dtype=dtype)
        masks = torch.zeros(2, 1, 224, 224, device=device, dtype=dtype)

        _out_image, out_mask = f(input_img, masks)
        assert out_mask.shape == torch.Size([2, 1, 300, 300])

    def test_mask_and_boxes_jointly(self, device, dtype):
        # All three data keys (image, mask, bbox_xyxy) routed simultaneously through same params.
        torch.manual_seed(76)
        f = RandomMosaic(p=1.0, data_keys=["input", "mask", "bbox_xyxy"])

        input_img = torch.stack(
            [torch.ones(1, 8, 8, device=device, dtype=dtype), torch.zeros(1, 8, 8, device=device, dtype=dtype)]
        )
        masks = torch.stack(
            [
                torch.ones(1, 8, 8, device=device, dtype=dtype) * 10.0,
                torch.ones(1, 8, 8, device=device, dtype=dtype) * 20.0,
            ]
        )
        boxes = torch.tensor([[[4.0, 5, 6, 7], [1, 2, 3, 4]], [[2, 2, 6, 6], [0, 0, 0, 0]]], device=device, dtype=dtype)

        out_image, out_mask, out_box = f(input_img, masks, boxes)
        assert out_image.shape == torch.Size([2, 1, 8, 8])
        assert out_mask.shape == torch.Size([2, 1, 8, 8])
        assert out_box.shape[0] == 2


class TestRandomJigsaw(BaseTester):
    def test_smoke(self, device, dtype):
        f = RandomJigsaw(data_keys=["input"])
        repr = "RandomJigsaw(grid=(4, 4), p=0.5, p_batch=1.0, same_on_batch=False, grid=(4, 4))"
        assert str(f) == repr

        # Test square and non-square images.
        f = RandomJigsaw(grid=(2, 2), p=1.0, data_keys=["input"])
        input = torch.arange(64, device=device, dtype=dtype).reshape(2, 1, 4, 8)
        f(input)
        input = torch.arange(64, device=device, dtype=dtype).reshape(2, 1, 8, 4)
        f(input)
        input = torch.arange(32, device=device, dtype=dtype).reshape(2, 1, 4, 4)
        f(input)

    def test_numerical(self, device, dtype):
        torch.manual_seed(22)
        f = RandomJigsaw(grid=(2, 2), p=1.0, data_keys=["input"])

        input = torch.arange(32, device=device, dtype=dtype).reshape(2, 1, 4, 4)

        out_image = f(input)

        expected = torch.tensor(
            [
                [[[2.0, 3.0, 0.0, 1.0], [6.0, 7.0, 4.0, 5.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]],
                [
                    [
                        [16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0],
                        [24.0, 25.0, 26.0, 27.0],
                        [28.0, 29.0, 30.0, 31.0],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(out_image, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_p(self, p, same_on_batch, device, dtype):
        torch.manual_seed(76)
        f = RandomJigsaw(p=p, data_keys=["input"], same_on_batch=same_on_batch)

        input = torch.randn((12, 3, 256, 256), device=device, dtype=dtype)

        f(input)


class TestRandomTransplantation(BaseTester):
    def test_smoke(self, device, dtype):
        torch.manual_seed(22)

        mask = torch.zeros(2, 3, 3, device=device, dtype=dtype)
        mask[0, 0:2, 0:2] = 1
        mask[1, 1:2, 1:2] = 2
        image = mask.clone().unsqueeze(dim=1)

        f = RandomTransplantation(p=1, excluded_labels=[0])
        image_out, mask_out = f(image, mask)

        mask_out_expected = torch.tensor(
            [[[1, 1, 0], [1, 2, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]]], device=device, dtype=dtype
        )

        self.assert_close(mask_out, mask_out_expected)
        self.assert_close(image_out, mask_out_expected.unsqueeze(dim=1))

    def test_mask_only(self, device, dtype):
        torch.manual_seed(22)

        mask = torch.zeros(2, 3, 3, device=device, dtype=dtype)
        mask[0, 0:2, 0:2] = 1
        mask[1, 1:2, 1:2] = 2

        f = RandomTransplantation(p=1, excluded_labels=[0], data_keys=["mask"])
        mask_out = f(mask)

        mask_out_expected = torch.tensor(
            [[[1, 1, 0], [1, 2, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]]], device=device, dtype=dtype
        )

        self.assert_close(mask_out, mask_out_expected)

    @pytest.mark.parametrize("n_spatial", [2, 3, 4])
    def test_module(self, n_spatial, device, dtype):
        torch.manual_seed(22)

        spatial_dimensions = [10] * n_spatial
        image = torch.rand(4, 3, *spatial_dimensions, device=device, dtype=dtype)
        mask = torch.zeros(4, *spatial_dimensions, device=device, dtype=dtype)
        mask_additional = torch.randint(0, 2, (4, *spatial_dimensions), device=device, dtype=dtype)

        selection = torch.zeros(*spatial_dimensions, device=device, dtype=torch.bool)
        selection[[slice(0, 5)] * n_spatial] = True
        assert selection.sum() == 5**n_spatial

        # Transplant rectangle from the (i - 1)-th to the i-th image
        for i in range(4):
            mask[i, selection] = i + 1

        image_copy = image.clone()
        mask_copy = mask.clone()
        mask_additional_copy = mask_additional.clone()

        f = RandomTransplantation(p=1, excluded_labels=[0])
        image_out, mask_out, mask_additional_out = f(image, mask, mask_additional, data_keys=["input", "mask", "mask"])

        self.assert_close(image, image_copy)
        self.assert_close(mask, mask_copy)
        self.assert_close(mask_additional, mask_additional_copy)

        for i in range(4):
            selection_moved = mask_out[i, selection]
            selection_unchanged = mask_out[i, ~selection]
            self.assert_close(selection_moved, torch.full_like(selection_moved, (i - 1) % 4 + 1))
            self.assert_close(selection_unchanged, torch.full_like(selection_unchanged, 0))
            self.assert_close(image_out[i, :, selection], image[(i - 1) % 4, :, selection])
            self.assert_close(image_out[i, :, ~selection], image[i, :, ~selection])
            self.assert_close(mask_additional_out[i, selection], mask_additional[(i - 1) % 4, selection])
            self.assert_close(mask_additional_out[i, ~selection], mask_additional[i, ~selection])

    def test_apply_none(self, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(4, 3, 10, 10, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, 10, 10), device=device, dtype=dtype)

        f = RandomTransplantation(p=0)
        image_out, mask_out = f(image, mask)

        assert torch.all(f._params["batch_prob"] == 0)
        assert len(f._params["selected_labels"]) == 0

        self.assert_close(image_out, image)
        self.assert_close(mask_out, mask)

    @pytest.mark.parametrize("wrapper", [AugmentationSequential, lambda x: x])
    def test_repeating(self, wrapper, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(4, 3, 10, 10, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, 10, 10), device=device, dtype=dtype)

        f = wrapper(RandomTransplantation(p=0.5))
        image_out, mask_out = f(image, mask, data_keys=["input", "mask"])
        image_out_same, mask_out_same = f(image, mask, params=f._params, data_keys=["input", "mask"])
        image_out_different, mask_out_different = f(image, mask, data_keys=["input", "mask"])

        self.assert_close(image_out, image_out_same)
        self.assert_close(mask_out, mask_out_same)
        with pytest.raises(AssertionError):
            self.assert_close(image_out, image_out_different)
        with pytest.raises(AssertionError):
            self.assert_close(mask_out, mask_out_different)

    @pytest.mark.parametrize("wrapper", [AugmentationSequential, lambda x: x])
    def test_data_keys(self, wrapper, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(4, 3, 10, 10, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, 10, 10), device=device, dtype=dtype)

        f = wrapper(RandomTransplantation(p=1))
        torch.manual_seed(22)
        image_out, mask_out = f(image, mask, data_keys=["input", "mask"])
        torch.manual_seed(22)
        mask_out2, image_out2 = f(mask, image, data_keys=["mask", "input"])

        self.assert_close(image_out, image_out2)
        self.assert_close(mask_out, mask_out2)

    @pytest.mark.parametrize("wrapper", [AugmentationSequential])
    def test_dict_input(self, wrapper, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(4, 3, 10, 10, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, 10, 10), device=device, dtype=dtype)

        f = wrapper(RandomTransplantation(p=1), data_keys=None)
        torch.manual_seed(22)
        dict_input = {"image": image, "mask": mask}
        aug_dict_output = f(dict_input)
        torch.manual_seed(22)
        dict_input2 = {"mask": mask, "image": image}
        aug_dict_output2 = f(dict_input2)

        image_out = aug_dict_output["image"]
        mask_out = aug_dict_output["mask"]
        image_out2 = aug_dict_output2["image"]
        mask_out2 = aug_dict_output2["mask"]

        self.assert_close(image_out, image_out2)
        self.assert_close(mask_out, mask_out2)

    @pytest.mark.parametrize("n_spatial", [2, 3])
    def test_sequential(self, n_spatial, device, dtype):
        torch.manual_seed(22)
        spatial_dimensions = [10] * n_spatial
        image = torch.rand(4, 3, *spatial_dimensions, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, *spatial_dimensions), device=device, dtype=dtype)

        if n_spatial == 2:
            f = RandomTransplantation(p=1)
        elif n_spatial == 3:
            f = RandomTransplantation3D(p=1)
        else:
            raise ValueError("n_spatial must be 2 or 3 since AugmentationSequential only supports 2D and 3D input")

        torch.manual_seed(22)
        image_out, mask_out = f(image, mask)

        torch.manual_seed(22)
        image_out2, mask_out2 = AugmentationSequential(f)(image, mask, data_keys=["image", "mask"])

        self.assert_close(image_out, image_out2)
        self.assert_close(mask_out, mask_out2)

    @pytest.mark.parametrize(
        "input_shape_image, input_shape_mask, target_shape_image",
        [
            [(1, 2, 3, 4), (1, 3, 4), (1, 2, 3, 4)],  # (B, C, H, W)
            [(1, 2, 5, 3, 4), (1, 5, 3, 4), (1, 2, 5, 3, 4)],  # (B, C, D, H, W)
            [(1, 1, 1, 1), (1, 1, 1), (1, 1, 1, 1)],  # (B, C, H, W)
        ],
    )
    def test_cardinality(self, input_shape_image, input_shape_mask, target_shape_image, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(input_shape_image, device=device, dtype=dtype)
        mask = torch.randint(0, 2, input_shape_mask, device=device, dtype=dtype)

        f = RandomTransplantation(p=1)
        image_out, mask_out = f(image, mask)

        assert image_out.shape == target_shape_image
        assert mask_out.shape == torch.Size([s for i, s in enumerate(target_shape_image) if i != 1])

    def test_gradcheck(self, device):
        torch.manual_seed(22)
        image = torch.rand(1, 3, 2, 2, device=device, dtype=torch.float64)
        mask = torch.randint(0, 2, (1, 2, 2), device=device, dtype=torch.float64)

        self.gradcheck(RandomTransplantation(p=1.0), (image, mask))

    def test_exception(self, device, dtype):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        torch.manual_seed(22)
        image = torch.rand(1, 3, 2, 2, device=device, dtype=torch.float64)
        mask = torch.randint(0, 2, (1, 2, 2), device=device, dtype=torch.float64)
        f = RandomTransplantation(p=1.0)
        f(image, mask)
        params = f._params

        with pytest.raises(Exception, match="excluded_labels must be a 1-dimensional"):
            RandomTransplantation(p=1.0, excluded_labels=torch.tensor([[0, 1]], device=device, dtype=dtype))

        with pytest.raises(Exception, match=r"Length of keys.*does not match number of inputs"):
            f = RandomTransplantation(p=1.0)
            f(image, mask, data_keys=["input", "mask", "mask"])

        with pytest.raises(Exception, match=r"selected_labels must be a 1-dimensional torch\.tensor"):
            params_copy = copy.deepcopy(params)
            params_copy["selected_labels"] = torch.tensor([[0, 1]], device=device, dtype=dtype)
            del params_copy["selection"]
            f(image, mask, params=params_copy)

        with pytest.raises(Exception, match="There cannot be more selected labels"):
            params_copy = copy.deepcopy(params)
            params_copy["selected_labels"] = torch.tensor([0, 1], device=device, dtype=dtype)
            del params_copy["selection"]
            f(image, mask, params=params_copy)

        with pytest.raises(Exception, match="Every image input must have one additional dimension"):
            f(image.unsqueeze(dim=-1), mask)

        with pytest.raises(Exception, match="The dimensions of the input image and segmentation mask must match"):
            image = torch.rand(1, 3, 2, 5, device=device, dtype=torch.float64)
            f(image, mask)
