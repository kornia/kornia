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

import kornia.augmentation as K
from kornia.constants import DType

from testing.base import BaseTester


class TestMixConventions(BaseTester):
    def test_convention_mix_base_has_no_matrix_or_inverse(self):
        aug = K.RandomMixUpV2()
        with pytest.raises(RuntimeError, match="Transformation matrices"):
            _ = aug.transform_matrix
        with pytest.raises(RuntimeError, match="Inverse"):
            aug.inverse()

    @pytest.mark.parametrize(
        ("factory", "shape"),
        [
            (lambda: K.RandomMixUpV2(p=1.0), (2, 3, 8, 12)),
            (lambda: K.RandomCutMixV2(p=1.0, use_correct_lambda=True), (2, 3, 8, 12)),
            (lambda: K.PatchMix(p=1.0, patch_size=2), (2, 3, 8, 12)),
            (lambda: K.RandomJigsaw(p=1.0, grid=(2, 3)), (2, 3, 8, 12)),
            (lambda: K.RandomMosaic(p=1.0), (2, 3, 8, 12)),
        ],
    )
    def test_convention_image_rank_is_promoted_and_keepdim_restores_it(self, factory, shape, device, dtype):
        # MixAugmentationBaseV2 accepts CHW images, forwards an NCHW batch to the operation, and the same
        # operation restores CHW only when keepdim=True. The Jigsaw dimensions divide its 2x3 grid exactly.
        for input_shape, batch_shape in ((shape[2:], (1, 1, *shape[2:])), (shape[1:], (1, *shape[1:])), (shape, shape)):
            image = torch.rand(input_shape, device=device, dtype=dtype)
            torch.manual_seed(7)
            assert factory()(image).shape == batch_shape
            aug = factory()
            aug.keepdim = True
            torch.manual_seed(7)
            assert aug(image).shape == input_shape

    def test_convention_mixup_class_rows_describe_the_image_mix(self, device, dtype):
        image = torch.stack(
            [torch.zeros(1, 4, 4, device=device, dtype=dtype), torch.ones(1, 4, 4, device=device, dtype=dtype)]
        )
        labels = torch.tensor([4, 9], device=device)
        aug = K.RandomMixUpV2(lambda_val=(0.25, 0.25), p=1.0, data_keys=["input", "class"])
        # A fixed non-identity pairing: a random draw pairs each row with itself half the time at B=2.
        aug(image, labels)
        params = dict(aug._params)
        params["mixup_pairs"] = torch.tensor([1, 0])
        output, mixed_labels = aug(image, labels, params=params)

        pairs = aug._params["mixup_pairs"].to(device)
        expected = image * 0.75 + image.index_select(0, pairs) * 0.25
        self.assert_close(output, expected)
        assert mixed_labels.shape == (2, 3)
        self.assert_close(mixed_labels[:, 0], labels.to(dtype))
        self.assert_close(mixed_labels[:, 1], labels.index_select(0, pairs).to(dtype))
        self.assert_close(mixed_labels[:, 2], torch.full((2,), 0.25, device=device, dtype=dtype))

    @pytest.mark.parametrize("num_mix", [1, 2])
    def test_convention_cutmix_class_axis_is_num_mix_then_batch(self, num_mix, device, dtype):
        image = torch.rand(3, 1, 8, 8, device=device, dtype=dtype)
        labels = torch.tensor([2, 5, 7], device=device)
        aug = K.RandomCutMixV2(
            num_mix=num_mix, cut_size=(0.5, 0.5), p=1.0, data_keys=["input", "class"], use_correct_lambda=True
        )
        aug(image, labels)
        params = dict(aug._params)
        params["mix_pairs"] = torch.tensor([[1, 2, 0], [2, 0, 1]])[:num_mix]
        _, mixed_labels = aug(image, labels, params=params)

        assert mixed_labels.shape == (num_mix, 3, 3)
        self.assert_close(mixed_labels[:, :, 0], labels.to(dtype).expand(num_mix, -1))
        for mix, pairs in enumerate(aug._params["mix_pairs"].to(device)):
            self.assert_close(mixed_labels[mix, :, 1], labels.index_select(0, pairs).to(dtype))

    @pytest.mark.parametrize(("use_correct_lambda", "expected_lambda"), [(False, 0.25), (True, 0.75)])
    def test_convention_cutmix_lambda_is_the_configured_area_fraction(
        self, use_correct_lambda, expected_lambda, device, dtype
    ):
        image = torch.zeros(2, 1, 4, 4, device=device, dtype=dtype)
        labels = torch.tensor([2, 5], device=device)
        crop = torch.tensor([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        params = {
            "batch_prob": torch.ones(2),
            "mix_pairs": torch.tensor([[1, 0]]),
            "crop_src": crop.repeat(1, 2, 1, 1),
            "image_shape": torch.tensor([4, 4]),
            "dtype": torch.tensor(DType.get(dtype).value),
        }
        aug = K.RandomCutMixV2(data_keys=["input", "class"], use_correct_lambda=use_correct_lambda)
        _, mixed_labels = aug(image, labels, params=params)

        self.assert_close(mixed_labels[0, :, 2], torch.full((2,), expected_lambda, device=device, dtype=dtype))

    @pytest.mark.parametrize(
        ("factory", "label_shape"),
        [
            (lambda: K.RandomMixUpV2(p=0.0, data_keys=["input", "class"]), (3, 3)),
            (lambda: K.RandomCutMixV2(p=0.0, data_keys=["input", "class"], use_correct_lambda=True), (1, 3, 3)),
        ],
    )
    def test_convention_mix_class_batch_gate_returns_identity_labels(self, factory, label_shape, device, dtype):
        image = torch.rand(3, 1, 8, 8, device=device, dtype=dtype)
        labels = torch.tensor([2, 5, 7], device=device)
        output, mixed_labels = factory()(image, labels)

        self.assert_close(output, image)
        assert mixed_labels.shape == label_shape
        expected_labels = labels.to(dtype).expand(label_shape[:-1])
        self.assert_close(mixed_labels[..., 0], expected_labels)
        self.assert_close(mixed_labels[..., 1], expected_labels)
        self.assert_close(mixed_labels[..., 2], torch.zeros(label_shape[:-1], device=device, dtype=dtype))

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: K.RandomMixUpV2(p=0.5),
            lambda: K.RandomCutMixV2(p=0.5, use_correct_lambda=True),
            lambda: K.PatchMix(p=0.5, patch_size=2),
        ],
    )
    def test_convention_mix_batch_gate_is_all_or_nothing(self, factory):
        # An invariant, not a statistic: a batch-wide gate never selects a strict subset of the rows.
        aug = factory()
        for _ in range(20):
            batch_prob = aug.forward_parameters(torch.Size([8, 1, 8, 8]))["batch_prob"]
            assert batch_prob.numel() == 8
            assert bool((batch_prob == batch_prob[0]).all())

    def test_convention_per_sample_gate_keeps_unselected_rows(self, device, dtype):
        image = torch.arange(32, device=device, dtype=dtype).reshape(2, 1, 4, 4)
        params = {"batch_prob": torch.tensor([1.0, 0.0]), "permutation": torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]])}
        output = K.RandomJigsaw(grid=(2, 2), p=0.5)(image, params=params)
        assert not torch.equal(output[0], image[0])
        self.assert_close(output[1], image[1])

    def test_convention_jigsaw_same_on_batch_shares_the_patch_permutation(self, device, dtype):
        image = torch.rand(3, 1, 8, 12, device=device, dtype=dtype)
        aug = K.RandomJigsaw(grid=(2, 3), p=1.0, same_on_batch=True)
        aug(image)

        permutation = aug._params["permutation"]
        assert torch.equal(permutation, permutation[0].expand_as(permutation))

    def test_convention_jigsaw_permutation_is_laid_out_by_columns(self, device, dtype):
        image = torch.zeros(1, 1, 4, 4, device=device, dtype=dtype)
        image[:, :, :2, :2] = 1
        image[:, :, :2, 2:] = 2
        image[:, :, 2:, :2] = 3
        image[:, :, 2:, 2:] = 4
        params = {"batch_prob": torch.ones(1), "permutation": torch.tensor([[3, 2, 1, 0]])}
        output = K.RandomJigsaw(grid=(2, 2), p=1.0)(image, params=params)

        expected = torch.tensor(
            [[[[4, 4, 2, 2], [4, 4, 2, 2], [3, 3, 1, 1], [3, 3, 1, 1]]]], device=device, dtype=dtype
        )
        self.assert_close(output, expected)

    def test_convention_mosaic_uses_height_width_and_xy_start_ratio(self, device, dtype):
        image = torch.rand(3, 1, 6, 8, device=device, dtype=dtype)
        aug = K.RandomMosaic(output_size=(4, 10), start_ratio_range=(0.5, 0.5), p=1.0)
        output = aug(image)

        assert output.shape == (3, 1, 4, 10)
        top_left = aug._params["src"][0, 0]
        self.assert_close(top_left, torch.tensor([4.0, 3.0], device=top_left.device, dtype=top_left.dtype))

    def test_convention_mosaic_grid_first_axis_is_width(self, device, dtype):
        image = torch.arange(6, device=device, dtype=dtype).view(6, 1, 1, 1).expand(6, 1, 6, 4)
        aug = K.RandomMosaic(mosaic_grid=(2, 3), start_ratio_range=(0.5, 0.5), p=1.0)
        params = aug.forward_parameters(image.shape)
        params["permutation"] = torch.arange(6).expand(6, -1)
        output = aug(image, params=params)

        # Cropping from (x, y)=(2, 3) crosses the four tiles at the top-left of the 3-row, 2-column grid.
        expected = image.new_tensor([[0, 0, 3, 3]] * 3 + [[1, 1, 4, 4]] * 3)
        self.assert_close(output, expected.expand(6, 1, 6, 4))

    @pytest.mark.parametrize("data_key", ["bbox", "bbox_xyxy", "bbox_xywh"])
    def test_convention_mosaic_supports_each_documented_box_key(self, data_key, device, dtype):
        image = torch.rand(4, 1, 6, 8, device=device, dtype=dtype)
        if data_key == "bbox":
            boxes = image.new_tensor([[[[1, 1], [7, 1], [7, 5], [1, 5]]]] * 4)
            expected = image.new_tensor(
                [
                    [[0, 0], [3, 0], [3, 2], [0, 2]],
                    [[0, 4], [3, 4], [3, 6], [0, 6]],
                    [[5, 0], [8, 0], [8, 2], [5, 2]],
                    [[5, 4], [8, 4], [8, 6], [5, 6]],
                ]
            )
        elif data_key == "bbox_xyxy":
            boxes = image.new_tensor([[[1, 1, 7, 5]]] * 4)
            expected = image.new_tensor([[0, 0, 3, 2], [0, 4, 3, 6], [5, 0, 8, 2], [5, 4, 8, 6]])
        else:
            boxes = image.new_tensor([[[1, 1, 6, 4]]] * 4)
            expected = image.new_tensor([[0, 0, 3, 2], [0, 4, 3, 2], [5, 0, 3, 2], [5, 4, 3, 2]])
        # The crop starts at (4, 3); translate each tile's box and clip to the 8x6 output.
        output, output_boxes = K.RandomMosaic(start_ratio_range=(0.5, 0.5), p=1.0, data_keys=["input", data_key])(
            image, boxes
        )

        assert output.shape == image.shape
        self.assert_close(output_boxes, expected.unsqueeze(0).expand(4, *expected.shape))

    @pytest.mark.parametrize(
        ("factory", "extra", "error"),
        [
            (lambda: K.RandomMixUpV2(p=1.0), "mask", NotImplementedError),
            (lambda: K.RandomCutMixV2(p=1.0, use_correct_lambda=True), "mask", NotImplementedError),
            (lambda: K.PatchMix(p=1.0, patch_size=2), "class", NotImplementedError),
            (lambda: K.RandomJigsaw(p=1.0, grid=(2, 2)), "class", NotImplementedError),
            (lambda: K.RandomMosaic(p=1.0), "mask", NotImplementedError),
        ],
    )
    def test_convention_unsupported_data_keys_raise(self, factory, extra, error, device, dtype):
        image = torch.rand(2, 1, 8, 8, device=device, dtype=dtype)
        annotation = torch.ones_like(image) if extra == "mask" else torch.tensor([0, 1], device=device)
        with pytest.raises(error):
            factory()(image, annotation, data_keys=["input", extra])

    def test_convention_patchmix_lam_does_not_weight_the_replacement(self, device, dtype):
        # The generator records a Beta sample as `lam`, but PatchMix copies a full square patch. The fixed
        # coordinates and permutation make the written pixels independent of that recorded value.
        image = torch.stack(
            [torch.zeros(1, 4, 4, device=device, dtype=dtype), torch.ones(1, 4, 4, device=device, dtype=dtype)]
        )
        aug = K.PatchMix(patch_size=2, p=1.0)
        params = {
            "batch_prob": torch.ones(2),
            "mix_pairs": torch.tensor([1, 0]),
            "patch_coords": torch.tensor([[1, 1], [1, 1]]),
            "lam": torch.tensor([0.125, 0.875]),
            "dtype": torch.tensor(6),
        }
        output = aug(image, params=params)

        expected = image.clone()
        expected[0, :, 1:3, 1:3] = 1
        expected[1, :, 1:3, 1:3] = 0
        self.assert_close(output, expected)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int64])
    def test_convention_mix_rejects_integer_images(self, dtype):
        with pytest.raises(TypeError, match="float16"):
            K.RandomMixUpV2(p=1.0)(torch.ones(2, 1, 4, 4, dtype=dtype))

    @pytest.mark.device_agnostic
    def test_convention_mix_inverse_is_keyword_only(self):
        with pytest.raises(TypeError, match="positional"):
            K.RandomMixUpV2().inverse(torch.ones(2, 1, 4, 4))

    @pytest.mark.device_agnostic
    def test_convention_jigsaw_same_on_batch_shares_the_gate(self):
        aug = K.RandomJigsaw(grid=(2, 2), p=0.5, same_on_batch=True)
        for _ in range(20):
            batch_prob = aug.forward_parameters(torch.Size([8, 1, 4, 4]))["batch_prob"]
            assert bool((batch_prob == batch_prob[0]).all())

    @pytest.mark.device_agnostic
    def test_wart_unsupported_key_passes_silently_without_selection_4651(self):
        # #4651: the unsupported-key handlers and the Jigsaw reshape are only reached when a sample is selected.
        image = torch.rand(4, 1, 8, 8)
        boxes = torch.tensor([[[1.0, 1.0, 4.0, 4.0]]] * 4)
        with pytest.raises(NotImplementedError):
            K.RandomMixUpV2(p=1.0)(image, boxes, data_keys=["input", "bbox_xyxy"])
        _, returned = K.RandomMixUpV2(p=0.0)(image, boxes, data_keys=["input", "bbox_xyxy"])
        self.assert_close(returned, boxes)
        _, labels = K.RandomJigsaw(p=0.0, grid=(2, 2))(image, torch.arange(4), data_keys=["input", "class"])
        assert torch.equal(labels, torch.arange(4))
        with pytest.raises(NotImplementedError):
            K.RandomMixUpV2(p=0.0)(image, torch.ones_like(image), data_keys=["input", "mask"])
        indivisible = torch.rand(2, 1, 9, 13)
        self.assert_close(K.RandomJigsaw(grid=(2, 3), p=0.0)(indivisible), indivisible)
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            K.RandomJigsaw(grid=(2, 3), p=1.0)(indivisible)

    def test_wart_jigsaw_nondivisible_input_can_silently_lose_a_channel_4651(self, device, dtype):
        # The missing divisibility check can also let reshape infer fewer channels, instead of raising.
        image = torch.arange(36, device=device, dtype=dtype).reshape(1, 3, 3, 4)
        params = {"batch_prob": torch.ones(1), "permutation": torch.tensor([[0, 1, 2, 3]])}
        output = K.RandomJigsaw(grid=(2, 2), p=1.0)(image, params=params)
        assert output.shape == (1, 2, 3, 4)
        self.assert_close(K.RandomJigsaw(grid=(2, 2), p=0.0)(image), image)

    @pytest.mark.device_agnostic
    def test_wart_mixup_and_cutmix_apply_p_to_rows_a_second_time_4649(self):
        # #4649: inside a selected batch, rows are dropped again with probability 1 - p.
        mixup = K.RandomMixUpV2(p=0.5, lambda_val=(0.5, 0.5))
        cutmix = K.RandomCutMixV2(p=0.5, cut_size=(0.5, 0.5), use_correct_lambda=True)
        image = torch.rand(8, 1, 8, 8)
        mixup_rows, cutmix_rows = [], []
        for seed in range(40):
            torch.manual_seed(seed)
            params = mixup.forward_parameters(image.shape)
            if params["batch_prob"].all():
                mixup_rows.append(params["mixup_lambdas"] > 0)
            output = cutmix(image)
            if cutmix._params["batch_prob"].all():
                cutmix_rows.append((output != image).flatten(1).any(1))
        for rows in (torch.cat(mixup_rows), torch.cat(cutmix_rows)):
            assert rows.numel() >= 64
            assert 0.3 < rows.float().mean() < 0.7

    @pytest.mark.device_agnostic
    def test_wart_patchmix_oversized_patch_and_same_on_batch_4650(self):
        image = torch.rand(4, 1, 8, 10)
        aug = K.PatchMix(p=1.0)  # the default patch_size=16 exceeds both image sides
        output = aug(image)
        assert (aug._params["patch_coords"] < 0).all()
        changed = (output != image)[:, 0]
        assert changed.any(-1).sum(-1).max() < 8 and changed.any(-2).sum(-1).max() < 10
        shared = K.PatchMix(patch_size=4, p=1.0, same_on_batch=True)
        self.assert_close(shared(image), image, rtol=0, atol=0)
        assert shared._params["mix_pairs"].tolist() == [0, 1, 2, 3]

    @pytest.mark.device_agnostic
    def test_wart_mosaic_output_size_boxes_and_resample_default_4652(self):
        image = torch.rand(4, 1, 6, 8)
        boxes = torch.tensor([[[1.0, 1.0, 4.0, 4.0]]] * 4)
        assert K.RandomMosaic(output_size=(4, 10), p=0.0)(image).shape == (4, 1, 4, 10)
        # Replay one recorded draw under both settings, with a hand-set gate that selects rows 1 and 3 only.
        reference = K.RandomMosaic(p=1.0, data_keys=["input", "bbox_xyxy"])
        reference(image, boxes)
        params = dict(reference._params)
        params["batch_prob"] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        selected = params["batch_prob"] > 0
        results = [
            K.RandomMosaic(output_size=output_size, p=1.0, data_keys=["input", "bbox_xyxy"])(
                image, boxes, params=params
            )
            for output_size in (None, (4, 10))
        ]
        assert results[0][0].shape == (4, 1, 6, 8) and results[1][0].shape == (4, 1, 4, 10)
        self.assert_close(results[0][1], results[1][1])  # the boxes ignore output_size
        assert results[1][1][selected][..., 3].max() > 4  # a box bottom below the 4-pixel-high output
        placeholder = torch.tensor([[1.0, 1.0, 4.0, 4.0]] + [[0.0, 0.0, 1.0, 1.0]] * 3)
        self.assert_close(results[1][1][~selected], placeholder.expand(2, -1, -1))
        with pytest.raises(TypeError, match="NoneType"):
            K.RandomMosaic(p=1.0, cropping_mode="resample")(image)

    @pytest.mark.device_agnostic
    def test_wart_mix_labels_round_in_bfloat16_4657(self):
        image = torch.rand(2, 1, 4, 4, dtype=torch.bfloat16)
        labels = torch.tensor([257, 999])
        for aug in (K.RandomMixUpV2(p=1.0), K.RandomCutMixV2(p=1.0, use_correct_lambda=True)):
            _, mixed = aug(image, labels, data_keys=["input", "class"])
            assert mixed.dtype == torch.bfloat16
            assert mixed[..., 0].flatten().tolist() == [256.0, 1000.0]
