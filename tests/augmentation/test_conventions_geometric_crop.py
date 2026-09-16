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

import pytest
import torch

import kornia.augmentation as K
from kornia.constants import DataKey
from kornia.geometry.transform import crop_by_transform_mat, resize

from testing.base import BaseTester, supports_2d_border_padding


class TestGeometricCropConventions(BaseTester):
    """Batch-6 convention pins for crop, resize, and pad augmentations."""

    def test_convention_center_crop_size_mode_and_inverse(self, device, dtype):
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        expected = x[..., 1:5, 2:6]

        slice_crop = K.CenterCrop(4, cropping_mode="slice")
        self.assert_close(slice_crop(x), expected)
        with pytest.raises(NotImplementedError, match="only applicable for resample"):
            slice_crop.inverse(expected)

        resample_crop = K.CenterCrop((4, 4), cropping_mode="resample")
        resample_output = resample_crop(x)
        assert resample_output.shape == expected.shape
        assert resample_crop.inverse(resample_output).shape == x.shape

        nearest = K.CenterCrop((4, 4), resample="nearest", align_corners=False, cropping_mode="resample")
        self.assert_close(nearest(x), expected)

    def test_wart_center_crop_slice_aliases_input_4413(self, device, dtype):
        # #4413: default slice mode returns its direct tensor slice. The write lands at the crop's centre offset.
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        output = K.CenterCrop(4, cropping_mode="slice")(x)
        assert output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        output[0, 0, 0, 0] = -99
        assert x[0, 0, 1, 2] == -99

    def test_convention_random_crop_padding_modes_and_inverse(self, device, dtype):
        x = torch.arange(9, device=device, dtype=dtype).reshape(1, 1, 3, 3)
        crop = K.RandomCrop((3, 3), padding=(1, 2, 3, 4), fill=9, p=1.0, cropping_mode="slice")
        assert crop.compute_padding((1, 1, 3, 3)) == [1, 3, 2, 4]
        torch.manual_seed(0)
        padded = crop(x)
        assert padded.shape == x.shape
        assert (padded == 9).any()

        needed = K.RandomCrop((4, 5), pad_if_needed=True, fill=9, p=1.0)
        assert needed.compute_padding((1, 1, 2, 3)) == [2, 2, 2, 2]
        prepared = needed.precrop_padding(torch.ones(1, 1, 2, 3, device=device, dtype=dtype))
        assert prepared.shape == (1, 1, 6, 7)
        assert (prepared[..., 0, :] == 9).all()

        resample = K.RandomCrop((3, 3), p=1.0, cropping_mode="resample")
        output = resample(x)
        assert resample.inverse(output).shape == x.shape
        slice_crop = K.RandomCrop((3, 3), p=1.0, cropping_mode="slice")
        slice_output = slice_crop(x)
        with pytest.raises(NotImplementedError, match="only applicable for resample"):
            slice_crop.inverse(slice_output)

        image = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        nearest = K.RandomCrop((3, 3), resample="nearest", align_corners=False, p=1.0, cropping_mode="resample")
        params = nearest.forward_parameters(image.shape)
        params["src"] = image.new_tensor([[[2, 1], [4, 1], [4, 3], [2, 3]]])
        self.assert_close(nearest(image, params=params), image[..., 1:4, 2:5])

    @pytest.mark.device_agnostic
    def test_convention_random_crop_merges_needed_padding_per_side(self):
        crop = K.RandomCrop((4, 5), padding=(5, 0, 1, 4), pad_if_needed=True)
        assert crop.compute_padding((1, 1, 2, 3)) == [5, 2, 2, 4]

    @pytest.mark.parametrize(
        "padding_mode, expected",
        [
            ("constant", [[0, 0, 0], [0, 0, 1], [0, 3, 4]]),
            ("replicate", [[0, 0, 1], [0, 0, 1], [3, 3, 4]]),
            ("reflect", [[4, 3, 4], [1, 0, 1], [4, 3, 4]]),
        ],
    )
    def test_convention_random_crop_sampler_padding(self, device, dtype, padding_mode, expected):
        if padding_mode == "replicate" and not supports_2d_border_padding(device):
            pytest.skip("2D border padding is unavailable on this backend")
        image = torch.arange(9, device=device, dtype=dtype).reshape(1, 1, 3, 3)
        crop = K.RandomCrop((3, 3), cropping_mode="resample", padding_mode=padding_mode, align_corners=True, p=1.0)
        params = crop.forward_parameters(image.shape)
        # Force sampling beyond the top/left edges, where zeros, border and reflection differ.
        params["src"] = image.new_tensor([[[-1, -1], [1, -1], [1, 1], [-1, 1]]])
        self.assert_close(crop(image, params=params), image.new_tensor(expected).reshape_as(image))

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("kind", ["crop", "resized_crop"])
    def test_convention_random_crops_gate_the_whole_batch(self, kind):
        crop = K.RandomCrop((3, 4), p=0.5) if kind == "crop" else K.RandomResizedCrop((3, 4), p=0.5)
        torch.manual_seed(0)
        gates = [crop.forward_parameters((16, 1, 6, 8))["batch_prob"] for _ in range(16)]
        assert all(torch.equal(gate, gate[:1].expand_as(gate)) for gate in gates)
        assert any(gate.all() for gate in gates)
        assert any(not gate.any() for gate in gates)

    @pytest.mark.parametrize("mode", ["slice", "resample"])
    def test_random_crop_explicit_padding_preserves_matrix_4542(self, device, dtype, mode):
        image = torch.arange(9, device=device, dtype=dtype).reshape(1, 1, 3, 3) / 8
        crop = K.RandomCrop((4, 4), padding=1, cropping_mode=mode, p=1.0)
        seq = K.AugmentationSequential(crop, data_keys=["input", "keypoints", "bbox"])
        params = seq.forward_parameters(image.shape)
        params[0].data["src"] = image.new_tensor([[[0, 0], [3, 0], [3, 3], [0, 3]]])
        points = image.new_tensor([[[2, 2]]])
        boxes = image.new_tensor([[[[1, 1], [2, 1], [2, 2], [1, 2]]]])
        output, out_points, out_boxes = seq(image, points, boxes, params=params)
        self.assert_close(crop.transform_matrix, image.new_tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))
        self.assert_close(out_points, image.new_tensor([[[3, 3]]]))
        self.assert_close(out_boxes, boxes + 1)
        expected = [[0, 0, 0, 0], [0, 0, 1, 2], [0, 3, 4, 5], [0, 6, 7, 8]]
        self.assert_close(output, image.new_tensor(expected).reshape(1, 1, 4, 4) / 8)

    @pytest.mark.parametrize("mode", ["slice", "resample"])
    def test_random_crop_explicit_asymmetric_padding_replays_and_skips_4542(self, device, dtype, mode):
        image = torch.arange(9, device=device, dtype=dtype).reshape(1, 1, 3, 3) / 8
        padding = (2, 1, 1, 2)
        crop = K.RandomCrop((4, 4), padding=padding, cropping_mode=mode, p=1.0)
        seq = K.AugmentationSequential(crop, data_keys=["input", "keypoints", "bbox"])
        params = seq.forward_parameters(image.shape)
        params[0].data["src"] = image.new_tensor([[[0, 0], [3, 0], [3, 3], [0, 3]]])
        points = image.new_tensor([[[2, 2]]])
        boxes = image.new_tensor([[[[1, 1], [2, 1], [2, 2], [1, 2]]]])
        output, out_points, out_boxes = seq(image, points, boxes, params=params)
        expected = image.new_tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 3, 4], [0, 0, 6, 7]]).reshape(1, 1, 4, 4) / 8
        self.assert_close(crop.transform_matrix, image.new_tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))
        self.assert_close(output, expected)
        self.assert_close(out_points, image.new_tensor([[[4, 3]]]))
        self.assert_close(out_boxes, boxes + image.new_tensor([2, 1]))
        replay, replay_points, replay_boxes = seq(image, points, boxes, params=params)
        self.assert_close(replay, output)
        self.assert_close(replay_points, out_points)
        self.assert_close(replay_boxes, out_boxes)

        skip = K.AugmentationSequential(
            K.RandomCrop((4, 4), padding=padding, cropping_mode=mode, p=0.0),
            data_keys=["input", "keypoints", "bbox"],
        )
        skipped, skipped_points, skipped_boxes = skip(image, points, boxes)
        self.assert_close(skipped, image)
        self.assert_close(skipped_points, points)
        self.assert_close(skipped_boxes, boxes)

    @pytest.mark.parametrize("mode", ["slice", "resample"])
    def test_random_crop_explicit_padding_replay_uses_stored_canvas_under_flag_overrides(self, device, dtype, mode):
        image = torch.arange(9, device=device, dtype=dtype).reshape(1, 1, 3, 3) / 8
        points = image.new_tensor([[[2, 2]]])
        boxes = image.new_tensor([[[[1, 1], [2, 1], [2, 2], [1, 2]]]])
        crop = K.RandomCrop((4, 4), padding=1, cropping_mode=mode, p=1.0)
        params = K.AugmentationSequential(crop, data_keys=["input", "keypoints", "bbox"]).forward_parameters(
            image.shape
        )
        params[0].data["src"] = image.new_tensor([[[0, 0], [3, 0], [3, 3], [0, 3]]])
        baseline = K.AugmentationSequential(crop, data_keys=["input", "keypoints", "bbox"])
        expected = baseline(image, points, boxes, params=params)
        self.assert_close(baseline.transform_matrix, image.new_tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))
        self.assert_close(
            expected[0], image.new_tensor([[[[0, 0, 0, 0], [0, 0, 1, 2], [0, 3, 4, 5], [0, 6, 7, 8]]]]) / 8
        )
        self.assert_close(expected[1], image.new_tensor([[[3, 3]]]))
        self.assert_close(expected[2], boxes + 1)
        override = K.AugmentationSequential(
            crop,
            data_keys=["input", "keypoints", "bbox"],
            extra_args={
                DataKey.INPUT: {"padding": 0, "pad_if_needed": False},
                DataKey.KEYPOINTS: {"padding": 0, "pad_if_needed": False},
                DataKey.BBOX: {"padding": 0, "pad_if_needed": False},
            },
        )
        actual = override(image, points, boxes, params=params)
        for got, want in zip(actual, expected):
            self.assert_close(got, want)

    @pytest.mark.parametrize("mode", ["slice", "resample"])
    def test_random_crop_explicit_padding_partial_fit_uses_padded_canvas_4542(self, device, dtype, mode):
        image = torch.arange(9, device=device, dtype=dtype).reshape(1, 1, 3, 3) / 8
        points = image.new_tensor([[[2, 2]]])
        boxes = image.new_tensor([[[[1, 1], [2, 1], [2, 2], [1, 2]]]])
        crop = K.RandomCrop((6, 4), padding=1, cropping_mode=mode, p=1.0)
        seq = K.AugmentationSequential(crop, data_keys=["input", "keypoints", "bbox"])
        params = seq.forward_parameters(image.shape)
        params[0].data["src"] = image.new_tensor([[[0, 0], [3, 0], [3, 5], [0, 5]]])
        params[0].data["dst"] = image.new_tensor([[[0, 0], [3, 0], [3, 5], [0, 5]]])
        output, out_points, out_boxes = seq(image, points, boxes, params=params)
        self.assert_close(seq.transform_matrix[..., 0, 0], image.new_tensor([4 / 5]))
        self.assert_close(seq.transform_matrix[..., 1, 1], image.new_tensor([6 / 5]))
        self.assert_close(out_points, image.new_tensor([[[2.4, 3.6]]]))
        self.assert_close(out_boxes, image.new_tensor([[[[1.6, 2.4], [2.4, 2.4], [2.4, 3.6], [1.6, 3.6]]]]))
        padded = torch.nn.functional.pad(image, [1, 1, 1, 1])
        matrix = image.new_tensor([[[4 / 5, 0, 0], [0, 6 / 5, 0], [0, 0, 1]]])
        if mode == "slice":
            expected = resize(padded[..., :5, :4], (6, 4), interpolation="bilinear", align_corners=None)
        else:
            expected = crop_by_transform_mat(
                padded,
                matrix[..., :2, :],
                (6, 4),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        self.assert_close(output, expected)
        override = K.AugmentationSequential(
            crop,
            data_keys=["input", "keypoints", "bbox"],
            extra_args={
                DataKey.INPUT: {"pad_if_needed": True},
                DataKey.KEYPOINTS: {"pad_if_needed": True},
                DataKey.BBOX: {"pad_if_needed": True},
            },
        )
        overridden = override(image, points, boxes, params=params)
        for got, want in zip(overridden, (output, out_points, out_boxes)):
            self.assert_close(got, want)
        self.assert_close(output[0, 0, 0], image.new_tensor([0, 0, 0, 0]))
        assert output.shape == (1, 1, 6, 4)

    @pytest.mark.parametrize("mode", ["slice", "resample"])
    @pytest.mark.parametrize("size", [(10, 10), (10, 4), (4, 10)])
    def test_wart_random_crop_oversized_rescales_both_matrix_axes_4414(self, device, dtype, mode, size):
        image = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8) / 48
        crop = K.RandomCrop(size, cropping_mode=mode, p=1.0)
        params = crop.forward_parameters(image.shape)
        h, w = size
        left, top = (2, 0) if size == (10, 4) else ((0, 1) if size == (4, 10) else (0, 0))
        params["src"] = image.new_tensor(
            [[[left, top], [left + w - 1, top], [left + w - 1, top + h - 1], [left, top + h - 1]]]
        )
        output = crop(image, params=params)
        self.assert_close(crop.transform_matrix, image.new_tensor([[[w / 8, 0, -left], [0, h / 6, -top], [0, 0, 1]]]))
        # Explicit ramp rows distinguish the slice resize from the mis-scaled, zero-padded warp.
        rows = {
            ((10, 10), "slice"): [40, 40.7, 41.5, 42.3, 43.1, 43.9, 44.7, 45.5, 46.3, 47],
            ((10, 10), "resample"): [24, 24.48, 24.96, 25.44, 25.92, 26.4, 26.88, 27.36, 27.84, 22.56],
            ((10, 4), "slice"): [2, 3, 4, 5],
            ((10, 4), "resample"): [4, 6, 0, 0],
            ((4, 10), "slice"): [8, 8.7, 9.5, 10.3, 11.1, 11.9, 12.7, 13.5, 14.3, 15],
            ((4, 10), "resample"): [12, 12.8, 13.6, 14.4, 15.2, 16, 16.8, 17.6, 18.4, 15.2],
        }
        row = -1 if size == (10, 10) else 0
        self.assert_close(output[0, 0, row], image.new_tensor(rows[size, mode]) / 48, low_tolerance=True)

    def test_convention_random_resized_crop_size_modes_and_inverse(self, device, dtype):
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        kwargs = {"size": (3, 4), "scale": (1.0, 1.0), "ratio": (0.75, 0.75), "p": 1.0}
        for mode in ("slice", "resample"):
            crop = K.RandomResizedCrop(cropping_mode=mode, **kwargs)
            output = crop(x)
            assert output.shape == (1, 1, 3, 4)
            if mode == "resample":
                assert crop.inverse(output).shape == x.shape
            else:
                with pytest.raises(NotImplementedError, match="only applicable for resample"):
                    crop.inverse(output)

        nearest = K.RandomResizedCrop(
            size=(6, 8),
            scale=(1.0, 1.0),
            ratio=(0.75, 0.75),
            resample="nearest",
            align_corners=True,
            cropping_mode="resample",
            p=1.0,
        )
        params = nearest.forward_parameters(x.shape)
        params["src"] = torch.tensor([[[0, 0], [7, 0], [7, 5], [0, 5]]], device=device, dtype=dtype)
        params["dst"] = torch.tensor([[[0, 0], [7, 0], [7, 5], [0, 5]]], device=device, dtype=dtype)
        self.assert_close(nearest(x, params=params), x)

    @pytest.mark.device_agnostic
    def test_convention_random_resized_crop_fallback_can_escape_scale_and_ratio(self):
        params = K.RandomResizedCrop((4, 4), scale=(1.0, 1.0), p=1.0).forward_parameters((1, 1, 8, 6))

        src = params["src"]
        crop_height = (src[0, 2, 1] - src[0, 1, 1]).item() + 1
        crop_width = (src[0, 1, 0] - src[0, 0, 0]).item() + 1
        assert (crop_height, crop_width) == (4, 6)
        assert crop_height * crop_width / (8 * 6) != 1.0
        assert crop_width / crop_height > 4 / 3

    @pytest.mark.device_agnostic
    def test_wart_crop_siblings_disagree_on_integer_size_4417(self):
        # #4417: CenterCrop accepts an int while its random siblings reject it through implementation details.
        assert K.CenterCrop(4).size == (4, 4)
        with pytest.raises(AssertionError):
            K.RandomCrop(4)(torch.ones(1, 1, 6, 8))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="not subscriptable"):
            K.RandomCrop(4, pad_if_needed=True)(torch.ones(1, 1, 6, 8))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            K.RandomResizedCrop(4)  # type: ignore[arg-type]

    def test_convention_resize_side_policies_and_inverse(self, device, dtype):
        x = torch.arange(70, device=device, dtype=dtype).reshape(1, 1, 7, 10)
        exact = K.Resize((3, 5), p=1.0)
        assert exact(x).shape == (1, 1, 3, 5)
        assert exact.inverse(exact(x)).shape == x.shape
        assert K.Resize(4, side="short", p=1.0)(x).shape == (1, 1, 4, 5)
        assert K.Resize(4, side="long", p=1.0)(x).shape == (1, 1, 2, 4)
        assert K.Resize(4, side="vert", p=1.0)(x).shape == (1, 1, 4, 5)
        assert K.Resize(4, side="horz", p=1.0)(x).shape == (1, 1, 2, 4)
        portrait = x.transpose(-1, -2)
        assert K.Resize(4, side="vert", p=1.0)(portrait).shape == (1, 1, 4, 2)
        assert K.Resize(4, side="horz", p=1.0)(portrait).shape == (1, 1, 5, 4)
        assert K.LongestMaxSize(4)(x).shape == (1, 1, 2, 4)
        assert K.SmallestMaxSize(4)(x).shape == (1, 1, 4, 5)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("kind", ["resize", "longest"])
    def test_convention_resize_rejects_a_truncated_zero_dimension(self, kind):
        aug = K.Resize(4, side="long") if kind == "resize" else K.LongestMaxSize(4)
        with pytest.raises(AssertionError, match="2 positive integers"):
            aug(torch.ones(1, 1, 1, 10))

    def test_wart_pad_to_smaller_target_crops_and_inverse_is_noop_4410(self, device, dtype):
        # #4410: negative right/bottom padding crops, and the identity matrix leaves inverse unable to restore it.
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        crop = K.PadTo((4, 5), pad_value=9)
        seq = K.AugmentationSequential(crop, data_keys=["input", "keypoints", "bbox"])
        points = x.new_tensor([[[7, 5]]])
        boxes = x.new_tensor([[[[6, 4], [7, 4], [7, 5], [6, 5]]]])
        output, out_points, out_boxes = seq(x, points, boxes)
        self.assert_close(out_points, points)
        self.assert_close(out_boxes, boxes)
        self.assert_close(output, x[..., :4, :5])
        self.assert_close(crop.inverse(output), output)
        self.assert_close(crop.transform_matrix, torch.eye(3, device=device, dtype=dtype)[None])

    def test_convention_pad_to_right_bottom_padding_round_trips(self, device, dtype):
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        pad = K.PadTo((10, 12), pad_value=9)
        output = pad(x)
        assert output.shape == (1, 1, 10, 12)
        self.assert_close(output[..., :6, :8], x)
        self.assert_close(output[..., 6:, :], torch.full((1, 1, 4, 12), 9, device=device, dtype=dtype))
        self.assert_close(pad.inverse(output), x)
