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
from kornia.constants import DataKey
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestRandomCropAnnotations(BaseTester):
    def test_slice_crop_boxes_match_pixels_when_only_height_is_oversized(self, device, dtype):
        height, width, size = 329, 1209, 416
        x = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width).expand(1, 1, height, width)
        boxes = torch.tensor([[[600.0, 100.0, 700.0, 200.0]]], device=device, dtype=dtype)
        seq = K.AugmentationSequential(
            K.RandomCrop((size, size), p=1.0, cropping_mode="slice"), data_keys=["input", "bbox_xyxy"]
        )

        out, out_boxes = seq(x, boxes)
        x_offset = out[0, 0, 0, 0]

        self.assert_close(out_boxes[0, 0, [0, 2]], boxes[0, 0, [0, 2]] - x_offset)

    @pytest.mark.parametrize(
        ("input_size", "size"),
        [((5, 12), (10, 8)), ((12, 5), (8, 10)), ((5, 6), (10, 9))],
        ids=["height", "width", "both"],
    )
    def test_slice_crop_annotations_sit_on_the_stretched_pixels_5481(self, input_size, size, device, dtype):
        """Sampling the output image at a transformed point must give back the point's source coordinates."""
        height, width = input_size
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        ramp = torch.stack([xs, ys])[None]  # each pixel stores its own source (x, y)
        seq = K.AugmentationSequential(
            K.RandomCrop(size, p=1.0, cropping_mode="slice"), data_keys=["input", "keypoints", "bbox_xyxy"]
        )
        h, w = size
        # An oversized axis starts at 0 and is stretched; an axis that fits is cropped, here from 1.
        x0, y0 = float(w < width), float(h < height)
        x1, y1 = x0 + min(w, width) - 1, y0 + min(h, height) - 1
        params = seq.forward_parameters(ramp.shape)
        params[0].data["src"] = ramp.new_tensor(
            [[[x0, y0], [x0 + w - 1, y0], [x0 + w - 1, y0 + h - 1], [x0, y0 + h - 1]]]
        )
        points = ramp.new_tensor([[[x0 + 1, y0 + 1], [x1 - 1, y1 - 1]]])
        boxes = points.view(1, 1, 4)

        out, out_points, out_boxes = seq(ramp, points, boxes, params=params)

        self.assert_close(out_boxes[0, 0].view(2, 2), out_points[0])
        # The stored coordinates are separable, so read x along the first row and y down the first column.
        x_of_column, y_of_row = out[0, 0, 0], out[0, 1, :, 0]
        for (x, y), (px, py) in zip(points[0].tolist(), out_points[0].tolist()):
            assert 0 < px < size[1] - 1 and 0 < py < size[0] - 1  # inside the rows and columns the resize clamps
            x_lo, y_lo = int(px // 1), int(py // 1)
            sampled_x = torch.lerp(x_of_column[x_lo], x_of_column[x_lo + 1], px - x_lo)
            sampled_y = torch.lerp(y_of_row[y_lo], y_of_row[y_lo + 1], py - y_lo)
            self.assert_close(torch.stack((sampled_x, sampled_y)), ramp.new_tensor([x, y]), low_tolerance=True)

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
    def test_per_channel_image_fill_defaults_masks_to_background(self, cropping_mode, device, dtype):
        image = torch.zeros(1, 3, 2, 2, device=device, dtype=dtype)
        mask = torch.ones(1, 1, 2, 2, device=device, dtype=dtype)
        fill = (0.25, 0.5, 0.75)

        def apply(mask_input, mask_fill=None):
            extra_args = None if mask_fill is None else {DataKey.MASK: {"fill": mask_fill}}
            seq = K.AugmentationSequential(
                K.RandomCrop((4, 4), padding=1, fill=fill, p=1.0, cropping_mode=cropping_mode),
                data_keys=["input", "mask"],
                extra_args=extra_args,
            )
            return seq(image, mask_input)

        padded_image, padded_mask = apply(mask)
        expected_image = image.new_tensor(fill).view(1, 3, 1, 1).expand(1, 3, 4, 4).clone()
        expected_image[:, :, 1:3, 1:3] = image
        expected_mask = mask.new_zeros(1, 1, 4, 4)
        expected_mask[:, :, 1:3, 1:3] = mask
        self.assert_close(padded_image, expected_image)
        self.assert_close(padded_mask, expected_mask)

        _, overridden_mask = apply(mask, 9.0)
        expected_override = mask.new_full((1, 1, 4, 4), 9.0)
        expected_override[:, :, 1:3, 1:3] = mask
        self.assert_close(overridden_mask, expected_override)

        _, padded_mask_list = apply([mask])
        self.assert_close(padded_mask_list[0], expected_mask)
        _, overridden_mask_list = apply([mask], 9.0)
        self.assert_close(overridden_mask_list[0], expected_override)

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
        # Equal spatial sizes let the image path select rows. A mixed gate on a shape-changing crop raises
        # (test_mixed_gate_with_shape_change_raises_4497); the mixed inverse is in test_same_size_mixed_inverse_5512.
        for actual, target in zip(output, inputs):
            self.assert_close(actual[1], target[1], rtol=0, atol=0)

    def test_same_size_mixed_inverse_5512(self, device, dtype):
        # The inverse of an equal-size padded crop returns the unpadded input size, so, like forward (#4473),
        # it selects rows: the applied rows are inverted and unpadded, the skipped rows are left alone.
        inputs = self.inputs(2, device, dtype)
        seq = K.AugmentationSequential(
            K.RandomCrop((6, 10), padding=(1, 2, 3, 0), p=0.5, cropping_mode="resample", resample="nearest"),
            data_keys=["input", "mask", "keypoints", "bbox_xyxy"],
        )
        mixed = self.fixed_params(seq, inputs[0].shape, [0.0, 1.0])
        output = seq(*inputs, params=deepcopy(mixed))
        forward_output = [value.clone() for value in output]
        restored = seq.inverse(*output, params=deepcopy(mixed))
        # The inverse writes the applied rows into a copy, not into the forward output it was given.
        for actual, before in zip(output, forward_output):
            self.assert_close(actual, before, rtol=0, atol=0)
        whole = self.fixed_params(seq, inputs[0].shape, [1.0, 1.0])
        reference = seq.inverse(*seq(*inputs, params=deepcopy(whole)), params=deepcopy(whole))
        for actual, skipped, applied in zip(restored, inputs, reference):
            assert actual.shape == skipped.shape
            self.assert_close(actual[0], skipped[0], rtol=0, atol=0)
            self.assert_close(actual[1], applied[1], rtol=0, atol=0)

        # A padded crop that changes the size still rejects a mixed gate, naming the shapes the caller sees.
        aug = K.RandomCrop((4, 8), padding=(1, 2, 3, 0), p=0.5, cropping_mode="resample")
        params = aug.forward_parameters(inputs[0].shape)
        params["batch_prob"] = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError, match=r"from \(1, 6, 10\) to \(1, 4, 8\)"):
            aug(inputs[0], params=deepcopy(params))
        with pytest.raises(ValueError, match=r"from \(1, 4, 8\) to \(1, 6, 10\)"):
            aug.inverse(torch.zeros(2, 1, 4, 8, device=device, dtype=dtype), params=deepcopy(params))

    @pytest.mark.parametrize(
        "make_aug",
        [
            lambda: K.RandomCrop((20, 26), padding=(1, 2), p=0.5, cropping_mode="slice"),
            lambda: K.RandomCrop((20, 26), padding=(1, 2), p=0.5, cropping_mode="resample"),
            lambda: K.Resize((20, 26), p=0.5),
        ],
        ids=["crop-slice", "crop-resample", "resize"],
    )
    def test_mixed_gate_with_shape_change_raises_4497(self, make_aug, device, dtype):
        # A batch holds one sample shape, so rows skipped by the gate cannot keep their own size.
        # Before #4497 they were silently transformed too, and inverse failed with a shape mismatch.
        aug = make_aug()
        image = torch.rand(4, 1, 24, 32, device=device, dtype=dtype)
        params = aug.forward_parameters(image.shape)
        params["batch_prob"] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="mixes applied and skipped rows"):
            aug(image, params=deepcopy(params))
        if aug.flags.get("cropping_mode", "resample") == "resample":
            output = torch.rand(4, 1, 20, 26, device=device, dtype=dtype)
            with pytest.raises(ValueError, match="mixes applied and skipped rows"):
                aug.inverse(output, params=deepcopy(params))

        # A whole-batch gate in either direction is still accepted.
        for gate in ([1.0] * 4, [0.0] * 4):
            params["batch_prob"] = torch.tensor(gate)
            output = aug(image, params=deepcopy(params))
            expected = (20, 26) if gate[0] else (24, 32)
            assert output.shape == (4, 1, *expected)

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


class TestRandomCropPaddingMatrix(BaseTester):
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("size", [(4, 8), (6, 10)])
    def test_dynamo_mixed_padding_4801(self, device, dtype, torch_optimizer, size):
        image = torch.zeros(2, 1, 6, 10, device=device, dtype=dtype)
        image[:, 0, 1, 3] = 1
        crop = K.RandomCrop(size, padding=(1, 2), p=0.5, cropping_mode="resample", resample="nearest")
        params = crop.forward_parameters(image.shape)
        params["batch_prob"] = image.new_tensor([0, 1])
        params["src"] = (
            image.new_tensor([[[1, 1], [size[1], 1], [size[1], size[0]], [1, size[0]]]]).expand(2, -1, -1).clone()
        )
        if size != image.shape[-2:]:
            # Skipped rows cannot keep their shape in a cropped batch (#4497).
            with pytest.raises(ValueError, match="mixes applied and skipped rows"):
                torch_optimizer(crop)(image, params=params)
            return
        output = torch_optimizer(crop)(image, params=params)
        padded = torch.nn.functional.pad(image, (1, 1, 2, 2))
        expected = padded[..., 1 : size[0] + 1, 1 : size[1] + 1].clone()
        expected[0] = image[0]
        self.assert_close(output, expected, rtol=0, atol=0)
        matrix = image.new_tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 1], [0, 0, 1]]])
        self.assert_close(crop.transform_matrix, matrix, rtol=0, atol=0)

    @pytest.mark.parametrize("mode", ["slice", "resample"])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    @pytest.mark.parametrize(
        "padding,automatic,size",
        [(None, False, (4, 6)), ((2, 1, 3, 2), False, (6, 9)), (None, True, (6, 9)), ((3, 0, 1, 2), True, (6, 9))],
    )
    def test_original_coordinates_match_pixels_and_annotations_4801(
        self, mode, same_on_batch, padding, automatic, size, device, dtype
    ):
        torch.manual_seed(4)
        image = torch.zeros(2, 1, 5, 7, device=device, dtype=dtype)
        image[..., 2, 3] = 1
        points = image.new_tensor([[[3, 2]]]).expand(2, -1, -1).clone()
        boxes = image.new_tensor([[[[2, 1], [3, 1], [3, 2], [2, 2]]]]).expand(2, -1, -1, -1).clone()
        crop = K.RandomCrop(
            size,
            padding=padding,
            pad_if_needed=automatic,
            cropping_mode=mode,
            same_on_batch=same_on_batch,
            resample="nearest",
            p=1.0,
        )
        seq = K.AugmentationSequential(crop, K.RandomHorizontalFlip(p=1.0), data_keys=["input", "keypoints", "bbox"])
        output, out_points, out_boxes = seq(image, points, boxes)
        matrix = seq.transform_matrix.clone()
        params = deepcopy(seq._params)
        homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], -1)
        mapped = (homogeneous @ matrix.transpose(-1, -2))[..., :2]
        self.assert_close(out_points, mapped)
        box_homogeneous = torch.cat([boxes[:, 0], torch.ones_like(boxes[:, 0, :, :1])], -1)
        mapped_boxes = (box_homogeneous @ matrix.transpose(-1, -2))[..., :2]
        # Tensor-form boxes reorder corners after a flip; compare their coordinate bounds.
        self.assert_close(out_boxes[:, 0].amin(1), mapped_boxes.amin(1))
        self.assert_close(out_boxes[:, 0].amax(1), mapped_boxes.amax(1))
        for row in range(2):
            y, x = divmod(output[row, 0].argmax().item(), size[1])
            self.assert_close(mapped[row, 0], image.new_tensor([x, y]), rtol=0, atol=0)
            assert output[row, 0, y, x] == 1
        if same_on_batch:
            self.assert_close(crop.transform_matrix[0], crop.transform_matrix[1], rtol=0, atol=0)
        replay = seq(image, points, boxes, params=params)
        for actual, expected in zip(replay, (output, out_points, out_boxes)):
            self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(seq.transform_matrix, matrix, rtol=0, atol=0)
        if mode == "resample":
            restored, restored_points, restored_boxes = seq.inverse(*replay, params=params)
            assert restored.shape == image.shape
            self.assert_close(restored_points, points)
            self.assert_close(restored_boxes, boxes)
            self.assert_close(restored[..., 2, 3], image[..., 2, 3])

    @pytest.mark.parametrize("resample", ["bilinear", "nearest"])
    def test_full_padded_image_inverse_4801(self, device, dtype, resample):
        image = torch.arange(20, device=device, dtype=dtype).reshape(1, 1, 4, 5) / 20
        crop = K.RandomCrop((9, 10), padding=(2, 1, 3, 4), cropping_mode="resample", resample=resample, p=1.0)
        output = crop(image)
        if resample == "nearest":
            self.assert_close(crop.inverse(output), image, rtol=0, atol=0)
        else:
            # Float16 normalized grids lose subpixel precision even for the upstream
            # identity warp. Budget one epsilon over the padded extent; keep the
            # default tolerances for other dtypes and the exact nearest control above.
            atol = torch.finfo(dtype).eps * max(output.shape[-2:]) if dtype == torch.float16 else None
            self.assert_close(crop.inverse(output), image, rtol=0 if atol is not None else None, atol=atol)
