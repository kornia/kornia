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

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import kornia.augmentation as K
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestRandomCropAnnotations(BaseTester):
    @staticmethod
    def inputs(batch_size, device, dtype):
        # Each pixel identifies its position and image, independently of the crop's matrix.
        image = torch.arange(batch_size * 60, device=device, dtype=dtype).reshape(batch_size, 1, 6, 10)
        image = image / (batch_size * 60)
        mask = torch.zeros_like(image)
        mask[..., 2, 3] = 1
        mask[..., 3, 4] = 1
        points = torch.tensor([[[3.0, 2.0], [4.0, 3.0]]], device=device, dtype=dtype).repeat(batch_size, 1, 1)
        boxes = torch.tensor([[[2.0, 1.0, 5.0, 3.0]]], device=device, dtype=dtype).repeat(batch_size, 1, 1)
        return image, mask, points, boxes

    @staticmethod
    def fixed_params(seq, shape, batch_prob):
        params = seq.forward_parameters(shape)
        params[0].data["batch_prob"] = torch.tensor(batch_prob)
        h, w = seq[0].flags["size"]
        # Top-left (2, 3) in padded coordinates; the second image starts at (3, 2).
        src = torch.tensor(
            [
                [[2.0, 3.0], [w + 1.0, 3.0], [w + 1.0, h + 2.0], [2.0, h + 2.0]],
                [[3.0, 2.0], [w + 2.0, 2.0], [w + 2.0, h + 1.0], [3.0, h + 1.0]],
            ]
        )
        params[0].data["src"] = src[: shape[0]].to(params[0].data["src"])
        return params

    @pytest.mark.parametrize("cropping_mode", ["slice", "resample"])
    @pytest.mark.parametrize("p,batch_size", [(0.0, 1), (0.5, 2)])
    @pytest.mark.parametrize(
        "padding,size,pad_if_needed",
        [
            (None, (4, 8), False),
            (1, (4, 8), False),
            ((1, 2), (4, 8), False),
            ((1, 2, 3, 4), (4, 8), False),
            ((0, 0, 2, 3), (4, 8), False),
            (None, (8, 12), True),
        ],
    )
    def test_skipped_is_identity_4473(self, cropping_mode, p, batch_size, padding, size, pad_if_needed, device, dtype):
        inputs = self.inputs(batch_size, device, dtype)
        originals = tuple(value.clone() for value in inputs)
        crop = K.RandomCrop(size, padding=padding, pad_if_needed=pad_if_needed, p=p, cropping_mode=cropping_mode)
        seq = K.AugmentationSequential(crop, data_keys=["input", "mask", "keypoints", "bbox_xyxy"])
        params = seq.forward_parameters(inputs[0].shape)
        params[0].data["batch_prob"].zero_()
        saved_params = deepcopy(params)

        output = seq(*inputs, params=params)

        for actual, expected in zip(output, originals):
            self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(crop.transform_matrix, torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3))
        replay = seq(*inputs, params=saved_params)
        for actual, expected in zip(replay, originals):
            self.assert_close(actual, expected, rtol=0, atol=0)
        if cropping_mode == "resample":
            for actual, expected in zip(seq.inverse(*output, params=saved_params), originals):
                self.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(inputs, originals):
            self.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in params[0].data.items():
            self.assert_close(value, saved_params[0].data[name], rtol=0, atol=0)

    @pytest.mark.parametrize("cropping_mode", ["slice", "resample"])
    @pytest.mark.parametrize("p", [0.5, 1.0])
    def test_applied_matches_pixels(self, cropping_mode, p, device, dtype):
        inputs = self.inputs(2, device, dtype)
        seq = K.AugmentationSequential(
            K.RandomCrop((4, 8), padding=(1, 2), p=p, cropping_mode=cropping_mode),
            data_keys=["input", "mask", "keypoints", "bbox_xyxy"],
        )
        params = self.fixed_params(seq, inputs[0].shape, [1.0, 1.0])
        expected = (
            torch.stack([inputs[0][0, :, 1:5, 1:9], inputs[0][1, :, :4, 2:10]]),
            torch.stack([inputs[1][0, :, 1:5, 1:9], inputs[1][1, :, :4, 2:10]]),
            torch.tensor([[[2.0, 1.0], [3.0, 2.0]], [[1.0, 2.0], [2.0, 3.0]]], device=device, dtype=dtype),
            torch.tensor([[[1.0, 0.0, 4.0, 2.0]], [[0.0, 1.0, 3.0, 3.0]]], device=device, dtype=dtype),
        )

        output = seq(*inputs, params=params)
        for actual, target in zip(output, expected):
            self.assert_close(actual, target)
        for actual, target in zip(seq(*inputs, params=deepcopy(params)), expected):
            self.assert_close(actual, target)
        if cropping_mode == "resample":
            restored = seq.inverse(*output, params=params)
            assert restored[0].shape == inputs[0].shape
            assert restored[1].shape == inputs[1].shape
            self.assert_close(restored[2], inputs[2])
            self.assert_close(restored[3], inputs[3])

    @pytest.mark.parametrize("cropping_mode", ["slice", "resample"])
    def test_same_size_mixed_forward_4473(self, cropping_mode, device, dtype):
        inputs = self.inputs(2, device, dtype)
        seq = K.AugmentationSequential(
            # Nearest sampling isolates row selection from half-precision interpolation at padded edges.
            K.RandomCrop((6, 10), padding=(1, 2), p=0.5, cropping_mode=cropping_mode, resample="nearest"),
            data_keys=["input", "mask", "keypoints", "bbox_xyxy"],
        )
        params = self.fixed_params(seq, inputs[0].shape, [1.0, 0.0])
        params[0].data["src"][1] = params[0].data["src"][0]
        expected = [value.clone() for value in inputs]
        for index in [0, 1]:
            expected[index][0].zero_()
            expected[index][0, :, :-1, :-1] = inputs[index][0, :, 1:, 1:]
        expected[2][0] -= 1
        expected[3][0] -= 1

        output = seq(*inputs, params=params)

        for actual, target in zip(output, expected):
            self.assert_close(actual, target)
        # Equal spatial sizes let the image path select rows. Mixed shape-changing crops
        # and mixed inverse have separate size/unpadding issues and are not asserted here.
        for actual, target in zip(output, inputs):
            self.assert_close(actual[1], target[1], rtol=0, atol=0)

    @pytest.mark.parametrize("box_format", ["bbox", "bbox_xyxy", "bbox_xywh"])
    @pytest.mark.parametrize("unbatched", [False, True])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_box_formats(self, box_format, unbatched, p, device, dtype):
        image = self.inputs(1, device, dtype)[0]
        if box_format == "bbox":
            boxes = torch.tensor([[[[2.0, 1.0], [5.0, 1.0], [5.0, 3.0], [2.0, 3.0]]]], device=device, dtype=dtype)
            shift = torch.ones_like(boxes)
        elif box_format == "bbox_xywh":
            boxes = torch.tensor([[[2.0, 1.0, 4.0, 3.0]]], device=device, dtype=dtype)
            shift = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(boxes)
        else:
            boxes = torch.tensor([[[2.0, 1.0, 5.0, 3.0]]], device=device, dtype=dtype)
            shift = torch.ones_like(boxes)
        expected = boxes - p * shift
        if unbatched:
            boxes, expected = boxes[0], expected[0]
        seq = K.AugmentationSequential(
            K.RandomCrop((4, 8), padding=(1, 2), p=p, cropping_mode="resample"),
            data_keys=["input", box_format],
        )
        params = self.fixed_params(seq, image.shape, [p])

        output = seq(image, boxes, params=params)

        self.assert_close(output[1], expected)
        self.assert_close(seq.inverse(*output, params=params)[1], boxes)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_object_inputs_preserve_metadata(self, p, device, dtype):
        image, _, points, box_tensor = self.inputs(2, device, dtype)
        box_list = [box_tensor[0], box_tensor[1].repeat(2, 1)]
        boxes = Boxes.from_tensor(box_list, mode="xyxy_plus")
        keypoints = Keypoints(points.clone())
        original_boxes = boxes.data.clone()
        seq = K.AugmentationSequential(
            K.RandomCrop((4, 8), padding=(1, 2), p=p), data_keys=["input", "keypoints", "bbox_xyxy"]
        )
        params = self.fixed_params(seq, image.shape, [p, p])

        _, out_points, out_boxes = seq(image, keypoints, boxes, params=params)

        assert isinstance(out_points, Keypoints)
        assert isinstance(out_boxes, Boxes)
        assert out_boxes.mode == boxes.mode
        out_box_list = out_boxes.to_tensor()
        assert isinstance(out_box_list, list)
        assert len(out_box_list) == len(box_list)
        shifts = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 2.0, 0.0]], device=device, dtype=dtype)
        for actual, original, shift in zip(out_box_list, box_list, shifts):
            self.assert_close(actual, original - p * shift)
        self.assert_close(out_points.data, points - p * shifts[:, None, :2])
        self.assert_close(boxes.data, original_boxes, rtol=0, atol=0)
        self.assert_close(keypoints.data, points, rtol=0, atol=0)

    def test_skipped_crop_followed_by_flip_4473(self, device, dtype):
        inputs = self.inputs(2, device, dtype)
        seq = K.AugmentationSequential(
            K.RandomCrop((4, 8), padding=(1, 2), p=0.0),
            K.RandomHorizontalFlip(p=1.0),
            data_keys=["input", "mask", "keypoints", "bbox_xyxy"],
        )
        expected_points = torch.tensor([[[6.0, 2.0], [5.0, 3.0]]], device=device, dtype=dtype).repeat(2, 1, 1)
        expected_boxes = torch.tensor([[[4.0, 1.0, 7.0, 3.0]]], device=device, dtype=dtype).repeat(2, 1, 1)

        image, mask, points, boxes = seq(*inputs)

        self.assert_close(image, inputs[0].flip(-1))
        self.assert_close(mask, inputs[1].flip(-1))
        self.assert_close(points, expected_points)
        self.assert_close(boxes, expected_boxes)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_gradcheck(self, p, device):
        image, mask, points, boxes = self.inputs(1, device, torch.float64)
        seq = K.AugmentationSequential(
            K.RandomCrop((4, 8), padding=(1, 2), p=p, cropping_mode="resample"),
            data_keys=["input", "mask", "keypoints", "bbox_xyxy"],
        )
        params = self.fixed_params(seq, image.shape, [p])

        def apply(image, points, boxes):
            output = seq(image, mask, points, boxes, params=params)
            return output[0], output[2], output[3]

        self.gradcheck(apply, (image, points, boxes))

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_dynamo(self, device, dtype, torch_optimizer):
        inputs = self.inputs(1, device, dtype)
        seq = K.AugmentationSequential(
            K.RandomCrop((4, 8), padding=(1, 2), p=0.0, cropping_mode="resample"),
            data_keys=["input", "mask", "keypoints", "bbox_xyxy"],
        )
        params = self.fixed_params(seq, inputs[0].shape, [0.0])

        output = torch_optimizer(seq)(*inputs, params=params)

        for actual, expected in zip(output, inputs):
            self.assert_close(actual, expected, rtol=0, atol=0)
