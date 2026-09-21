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

import warnings

import pytest
import torch

import kornia.augmentation as K
from kornia.constants import DType

from testing.base import BaseTester


def _label_dtype(image_dtype: torch.dtype) -> torch.dtype:
    # Half-precision images return float32 labels, so that integer class ids are not rounded.
    return image_dtype if image_dtype in (torch.float32, torch.float64) else torch.float32


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
            [torch.full((1, 4, 4), value, device=device, dtype=dtype) for value in (0.0, 1.0, 2.0, 3.0)]
        )
        labels = torch.tensor([4, 9, 12, 15], device=device)
        aug = K.RandomMixUpV2(lambda_val=(0.25, 0.25), p=1.0, data_keys=["input", "class"])
        aug(image, labels)
        params = dict(aug._params)
        # This is neither identity nor a one-row batch roll, so the recorded pairing is observable.
        params["mixup_pairs"] = torch.tensor([2, 0, 3, 1])
        output, mixed_labels = aug(image, labels, params=params)

        # Build the oracle from the literal set above, not from aug._params: forward() stores the supplied
        # dict by reference, so reading it back cannot tell whether the pairing was honoured.
        pairs = torch.tensor([2, 0, 3, 1], device=device)
        expected = image * 0.75 + image.index_select(0, pairs) * 0.25
        self.assert_close(output, expected)
        assert mixed_labels.shape == (4, 3)
        label_dtype = _label_dtype(dtype)
        assert output.dtype == dtype and mixed_labels.dtype == label_dtype
        self.assert_close(mixed_labels[:, 0], labels.to(label_dtype))
        self.assert_close(mixed_labels[:, 1], labels.index_select(0, pairs).to(label_dtype))
        self.assert_close(mixed_labels[:, 2], torch.full((4,), 0.25, device=device, dtype=label_dtype))

    @pytest.mark.parametrize("num_mix", [1, 2])
    def test_convention_cutmix_class_axis_is_num_mix_then_batch(self, num_mix, device, dtype):
        image = torch.stack(
            [torch.arange(42, device=device, dtype=dtype).reshape(1, 6, 7) + 100 * row for row in range(3)]
        )
        labels = torch.tensor([2, 5, 7], device=device)
        aug = K.RandomCutMixV2(num_mix=num_mix, p=1.0, data_keys=["input", "class"], use_correct_lambda=True)
        aug(image, labels)
        params = dict(aug._params)
        params["mix_pairs"] = torch.tensor([[1, 2, 0], [2, 0, 1]])[:num_mix]
        # The two rectangles have unequal x/y starts and dimensions.  They also do not overlap, so num_mix=2
        # checks both donor/crop stages independently.
        params["crop_src"] = image.new_tensor(
            [
                [[[1, 2], [3, 2], [3, 3], [1, 3]], [[1, 2], [3, 2], [3, 3], [1, 3]], [[1, 2], [3, 2], [3, 3], [1, 3]]],
                [[[4, 0], [5, 0], [5, 2], [4, 2]], [[4, 0], [5, 0], [5, 2], [4, 2]], [[4, 0], [5, 0], [5, 2], [4, 2]]],
            ]
        )[:num_mix]
        params["image_shape"] = image.new_tensor([6, 7])
        params["dtype"] = torch.tensor(DType.get(dtype).value)
        output, mixed_labels = aug(image, labels, params=params)

        expected = image.clone()
        expected[:, :, 2:4, 1:4] = image[torch.tensor([1, 2, 0], device=device), :, 2:4, 1:4]
        if num_mix == 2:
            expected[:, :, 0:3, 4:6] = image[torch.tensor([2, 0, 1], device=device), :, 0:3, 4:6]
        self.assert_close(output, expected)

        assert mixed_labels.shape == (num_mix, 3, 3)
        label_dtype = _label_dtype(dtype)
        assert output.dtype == dtype and mixed_labels.dtype == label_dtype
        self.assert_close(mixed_labels[:, :, 0], labels.to(label_dtype).expand(num_mix, -1))
        for mix, pairs in enumerate(aug._params["mix_pairs"].to(device)):
            self.assert_close(mixed_labels[mix, :, 1], labels.index_select(0, pairs).to(label_dtype))
        expected_lambdas = torch.tensor([1 - 6 / 42, 1 - 6 / 42], device=device, dtype=label_dtype)[:num_mix]
        self.assert_close(mixed_labels[:, :, 2], expected_lambdas[:, None].expand(-1, 3))

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

        expected = torch.full((2,), expected_lambda, device=device, dtype=_label_dtype(dtype))
        self.assert_close(mixed_labels[0, :, 2], expected)

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
        label_dtype = _label_dtype(dtype)
        expected_labels = labels.to(label_dtype).expand(label_shape[:-1])
        self.assert_close(mixed_labels[..., 0], expected_labels)
        self.assert_close(mixed_labels[..., 1], expected_labels)
        self.assert_close(mixed_labels[..., 2], torch.zeros(label_shape[:-1], device=device, dtype=label_dtype))

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
        # The second row carries different values and a different permutation: each image is partitioned from
        # its own pieces, never from another row's.
        image = torch.cat([image, image + 10])
        params = {"batch_prob": torch.ones(2), "permutation": torch.tensor([[3, 2, 1, 0], [1, 0, 3, 2]])}
        output = K.RandomJigsaw(grid=(2, 2), p=1.0)(image, params=params)

        expected = torch.tensor(
            [
                [[[4, 4, 2, 2], [4, 4, 2, 2], [3, 3, 1, 1], [3, 3, 1, 1]]],
                [[[12, 12, 14, 14], [12, 12, 14, 14], [11, 11, 13, 13], [11, 11, 13, 13]]],
            ],
            device=device,
            dtype=dtype,
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

    def test_convention_mosaic_uses_each_output_row_permutation(self, device, dtype):
        image = torch.arange(4, device=device, dtype=dtype).reshape(4, 1, 1, 1).expand(4, 1, 4, 4)
        aug = K.RandomMosaic(start_ratio_range=(0.5, 0.5), p=1.0)
        params = aug.forward_parameters(image.shape)
        params["batch_prob"] = torch.ones(4)
        params["permutation"] = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
        output = aug(image, params=params)

        expected = image.new_tensor(
            [
                [[0, 0, 2, 2], [0, 0, 2, 2], [1, 1, 3, 3], [1, 1, 3, 3]],
                [[1, 1, 3, 3], [1, 1, 3, 3], [0, 0, 2, 2], [0, 0, 2, 2]],
                [[2, 2, 0, 0], [2, 2, 0, 0], [3, 3, 1, 1], [3, 3, 1, 1]],
                [[3, 3, 1, 1], [3, 3, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]],
            ]
        ).unsqueeze(1)
        self.assert_close(output, expected)

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
            [torch.arange(35, device=device, dtype=dtype).reshape(1, 5, 7) + 100 * row for row in range(3)]
        )
        aug = K.PatchMix(patch_size=2, p=1.0)
        params = {
            "batch_prob": torch.ones(3),
            "mix_pairs": torch.tensor([2, 0, 1]),
            "patch_coords": torch.tensor([[0, 2], [3, 0], [1, 1]]),
            "lam": torch.tensor([0.125, 0.5, 0.875]),
            "dtype": torch.tensor(DType.get(dtype).value),
        }
        output = aug(image, params=params)

        expected = image.clone()
        expected[0, :, 2:4, 0:2] = image[2, :, 2:4, 0:2]
        expected[1, :, 0:2, 3:5] = image[0, :, 0:2, 3:5]
        expected[2, :, 1:3, 1:3] = image[1, :, 1:3, 1:3]
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
    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize(
        ("factory", "supported", "class_error"),
        [
            (lambda p: K.RandomMixUpV2(p=p), {"class"}, NotImplementedError),
            (lambda p: K.RandomCutMixV2(p=p, use_correct_lambda=True), {"class"}, NotImplementedError),
            (lambda p: K.RandomJigsaw(grid=(2, 2), p=p), set(), NotImplementedError),
            (lambda p: K.PatchMix(patch_size=4, p=p), set(), NotImplementedError),
            (lambda p: K.RandomMosaic(p=p), {"bbox_xyxy"}, RuntimeError),
        ],
    )
    def test_convention_unsupported_key_raises_whatever_the_gate(self, factory, supported, class_error, p):
        # Fixed by #4676: these used to pass through silently whenever the gate selected no sample (#4651).
        image = torch.rand(4, 1, 8, 8)
        annotations = {
            "bbox_xyxy": torch.tensor([[[1.0, 1.0, 4.0, 4.0]]] * 4),
            "keypoints": torch.rand(4, 2, 2),
            "mask": torch.ones_like(image),
            "class": torch.arange(4),
        }
        for key, annotation in annotations.items():
            if key in supported:
                factory(p)(image, annotation, data_keys=["input", key])
                continue
            # `None` rather than "": MixAugmentationBaseV2._validate_data_key raises a bare
            # NotImplementedError, so there is no message to match on that half.
            message = "does not support" if key == "class" and class_error is RuntimeError else None
            state = torch.get_rng_state().clone()
            with pytest.raises(class_error if key == "class" else NotImplementedError, match=message):
                factory(p)(image, annotation, data_keys=["input", key])
            # "Before anything is sampled": a handler fallback raises the same bare error, but only after the draw.
            assert torch.equal(state, torch.get_rng_state())

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_jigsaw_rejects_a_nondivisible_input_whatever_the_gate(self, p, device, dtype):
        # Fixed by #4676: (1, 3, 3, 4) with grid=(2, 2) used to come back as (1, 2, 3, 4), and p=0 returned the input.
        for shape, grid in (((1, 3, 3, 4), (2, 2)), ((2, 1, 9, 13), (2, 3)), ((2, 1, 8, 13), (2, 3))):
            image = torch.rand(shape, device=device, dtype=dtype)
            with pytest.raises(RuntimeError, match="must be divisible by grid"):
                K.RandomJigsaw(grid=grid, p=p)(image)

    @pytest.mark.device_agnostic
    def test_convention_mix_class_handlers_ignore_a_replayed_partial_gate(self):
        image = torch.arange(3.0).view(3, 1, 1, 1).expand(3, 1, 4, 4).clone()
        labels = torch.tensor([10, 20, 30])
        aug = K.RandomMixUpV2(p=1.0, lambda_val=(0.25, 0.25), data_keys=["input", "class"])
        aug(image, labels)
        params = dict(aug._params)
        params["mixup_pairs"] = torch.tensor([1, 2, 0])
        params["batch_prob"] = torch.tensor([1.0, 0.0, 1.0])
        output, mixed = aug(image, labels, params=params)
        self.assert_close(output[:, 0, 0, 0], torch.tensor([0.25, 1.0, 1.5]))  # row 1 keeps its image
        self.assert_close(mixed, torch.tensor([[10.0, 20.0, 0.25], [20.0, 30.0, 0.25], [30.0, 10.0, 0.25]]))

    @pytest.mark.device_agnostic
    def test_convention_cutmix_compatibility_lambda_warns(self):
        with pytest.warns(DeprecationWarning, match="use_correct_lambda"):
            K.RandomCutMixV2()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            K.RandomCutMixV2(use_correct_lambda=True)

    @pytest.mark.device_agnostic
    def test_convention_mixup_and_cutmix_apply_p_once_per_batch_4649(self):
        from kornia.geometry.bbox import infer_bbox_shape

        # ``p`` selects the whole batch; inside a selected batch no row (or mix) is dropped a second time.
        mixup = K.RandomMixUpV2(p=0.5, lambda_val=(0.5, 0.5))
        cutmix = K.RandomCutMixV2(p=0.5, cut_size=(0.5, 0.5), num_mix=2, use_correct_lambda=True)
        shape = torch.Size([8, 1, 8, 8])

        def mixup_rows(params):
            return params["mixup_lambdas"] > 0

        def cutmix_rows(params):  # one (B, 4, 2) box set per mix
            sides = [torch.stack(infer_bbox_shape(boxes)) for boxes in params["crop_src"]]
            return (torch.stack(sides) > 0).flatten()

        for aug, mixed in ((mixup, mixup_rows), (cutmix, cutmix_rows)):
            rows = []
            for seed in range(40):
                torch.manual_seed(seed)
                params = aug.forward_parameters(shape)
                gate = params["batch_prob"]
                assert bool((gate == gate[0]).all())  # all ones or all zeros
                if gate[0] > 0:
                    rows.append(mixed(params))
            assert 10 <= len(rows) <= 30  # the batch gate itself still fires at rate p
            assert bool(torch.cat(rows).all())

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_patchmix_rejects_a_patch_larger_than_the_image(self, p):
        # Fixed by #4680: the default patch_size=16 used to draw negative corners on this image (#4650).
        image = torch.rand(4, 1, 8, 10)
        for patch_size in (9, 16):
            with pytest.raises(ValueError, match="`patch_size` to fit the input"):
                K.PatchMix(patch_size=patch_size, p=p)(image)
        assert K.PatchMix(patch_size=8, p=p)(image).shape == image.shape  # min(H, W) itself is accepted

    @pytest.mark.device_agnostic
    def test_convention_patchmix_same_on_batch_shares_the_patch_but_not_the_pairing(self):
        # Fixed by #4680: a shared pairing draw used to argsort to the identity, so every image patched itself.
        # Each donor's pixel value is its batch index, providing an oracle independent of patch-copy slicing.
        image = torch.arange(64, dtype=torch.float32).reshape(64, 1, 1, 1).expand(-1, 1, 8, 10).clone()
        aug = K.PatchMix(patch_size=4, p=1.0, same_on_batch=True)
        output = aug(image)
        pairs = aug._params["mix_pairs"]
        assert torch.equal(pairs.sort().values, torch.arange(64))
        assert not torch.equal(pairs, torch.arange(64))
        coords = aug._params["patch_coords"]
        assert bool((coords == coords[0]).all())
        x, y = (int(value) for value in coords[0])
        expected = image.clone()
        expected[:, :, y : y + 4, x : x + 4] = pairs.to(image)[:, None, None, None]
        self.assert_close(output, expected, rtol=0, atol=0)
        # Without same_on_batch the coordinate is per sample as well.
        independent = K.PatchMix(patch_size=4, p=1.0)
        independent(image)
        assert independent._params["patch_coords"].unique(dim=0).shape[0] > 1

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
        # An unselected row keeps its image, yet its own boxes are clipped to the input extent and dropped to the
        # placeholder below min_bbox_size.
        loose = torch.tensor([[[2.0, 1.0, 30.0, 20.0], [1.0, 1.0, 5.0, 6.0]]] * 4)
        strict = K.RandomMosaic(p=1.0, min_bbox_size=19.0, data_keys=["input", "bbox_xyxy"])
        output, filtered = strict(image, loose, params=params)
        self.assert_close(output[~selected], image[~selected])
        own = torch.tensor([[2.0, 1.0, 8.0, 6.0], [0.0, 0.0, 1.0, 1.0]])  # clipped to W=8, H=6; the small one dropped
        self.assert_close(filtered[~selected][:, :2], own.expand(2, -1, -1))
        with pytest.raises(TypeError, match="NoneType"):
            K.RandomMosaic(p=1.0, cropping_mode="resample")(image)
        self.assert_close(K.RandomMosaic(p=0.0, cropping_mode="resample")(image), image)  # no selection, no raise

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("image_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_mix_labels_stay_exact_for_half_precision_images(self, image_dtype, p):
        # Fixed by #4661: labels used to be cast to the image dtype, so bfloat16 returned [256, 1000] here (#4657).
        image = torch.rand(2, 1, 4, 4, dtype=image_dtype)
        labels = torch.tensor([257, 999])
        cases = (
            (K.RandomMixUpV2(p=p), "mixup_pairs", torch.tensor([1, 0])),
            (K.RandomCutMixV2(p=p, use_correct_lambda=True), "mix_pairs", torch.tensor([[1, 0]])),
        )
        for aug, key, swapped in cases:
            # Supply the swap: with a sampled pairing both columns can legitimately hold [257, 999], and then
            # nothing shows that the paired label was used, let alone kept exact.
            aug(image, labels, data_keys=["input", "class"])
            params = dict(aug._params)
            params[key] = swapped
            output, mixed = aug(image, labels, params=params, data_keys=["input", "class"])
            assert output.dtype == image_dtype and mixed.dtype == torch.float32
            assert mixed[..., 0].flatten().tolist() == [257.0, 999.0]
            assert mixed[..., 1].flatten().tolist() == ([999.0, 257.0] if p == 1.0 else [257.0, 999.0])

    def test_convention_jigsaw_identity_permutation_transposes_the_grid(self, device, dtype):
        # The destination cell is chosen column-major by an entry's position; the entry's value indexes the
        # source patch row-major. The two orders differ, so the identity permutation is not a no-op.
        image = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], device=device, dtype=dtype)
        params = {"batch_prob": torch.ones(1), "permutation": torch.tensor([[0, 1, 2, 3]])}
        self.assert_close(K.RandomJigsaw(grid=(2, 2), p=1.0)(image, params=params), image.transpose(-1, -2))
        params["permutation"] = torch.tensor([[0, 2, 1, 3]])
        self.assert_close(K.RandomJigsaw(grid=(2, 2), p=1.0)(image, params=params), image)
        # Every permutation above is its own inverse, so "the value indexes the source" and "the value indexes
        # the destination" agree on all of them. A 3-cycle tells them apart.
        strip = torch.arange(6, device=device, dtype=dtype).view(1, 1, 2, 3)
        params["permutation"] = torch.tensor([[1, 2, 0, 3, 4, 5]])
        cycled = K.RandomJigsaw(grid=(2, 3), p=1.0)(strip, params=params)
        assert cycled[0, 0].tolist() == [[1.0, 0.0, 4.0], [2.0, 3.0, 5.0]]
        # "Transposes" is the square case. On any grid the image-preserving permutation is the transposed index
        # grid, and the identity permutation reproduces the image only when the grid has a single row or column.
        for grid in ((2, 3), (3, 2), (1, 3), (3, 1)):
            cells = torch.arange(grid[0] * grid[1], device=device, dtype=dtype).view(grid)
            picture = cells.repeat_interleave(2, 0).repeat_interleave(2, 1)[None, None]
            aug = K.RandomJigsaw(grid=grid, p=1.0)
            params["permutation"] = torch.arange(grid[0] * grid[1])[None]
            identity_is_a_noop = torch.equal(aug(picture, params=params), picture)
            assert identity_is_a_noop == (1 in grid)
            params["permutation"] = torch.arange(grid[0] * grid[1]).view(grid).T.flatten()[None]
            self.assert_close(aug(picture, params=params), picture)

    @pytest.mark.device_agnostic
    def test_wart_jigsaw_ensure_perm_rejects_the_wrong_permutation_4703(self):
        # ensure_perm rejects arange(N), which is a real shuffle on a 2 x 2 grid, and still draws [0, 2, 1, 3],
        # which is the no-op. 400 draws miss one of the 23 remaining permutations with probability 2e-8.
        aug = K.RandomJigsaw(grid=(2, 2), p=1.0, ensure_perm=True)
        drawn = set()
        for _ in range(50):
            permutation = aug.forward_parameters(torch.Size([8, 1, 4, 4]))["permutation"]
            drawn.update(tuple(row) for row in permutation.tolist())
        assert (0, 1, 2, 3) not in drawn
        assert (0, 2, 1, 3) in drawn

    def test_convention_transplantation3d_is_a_mix_class_over_volumes(self, device, dtype):
        # The base block's (B, C, H, W) working layout has one exception: the 3D transplantation class.
        volume = torch.rand(2, 3, 4, 5, 6, device=device, dtype=dtype)
        mask = torch.zeros(2, 4, 5, 6, device=device, dtype=torch.long)
        mask[:, 1:3, 1:3, 1:3] = 1
        output, _ = K.RandomTransplantation3D(p=1.0)(volume, mask, data_keys=["input", "mask"])
        assert output.shape == volume.shape
        # It is still a mix augmentation: no matrix and no inverse.
        with pytest.raises(RuntimeError, match="Transformation matrices"):
            _ = K.RandomTransplantation3D(p=1.0).transform_matrix
        # Both transplantation classes override forward: the spatial rank is free, so the 2D class takes the
        # same volume, and neither promotes an unbatched image.
        flat, _ = K.RandomTransplantation(p=1.0)(volume, mask, data_keys=["input", "mask"])
        assert flat.shape == volume.shape
        for cls in (K.RandomTransplantation, K.RandomTransplantation3D):
            with pytest.raises(Exception, match="must match except for the channel"):
                cls(p=1.0)(volume[0], mask[0], data_keys=["input", "mask"])

    def test_convention_mosaic_start_ratio_range_is_a_sampling_range(self, device, dtype):
        # Both entries are (low, high) bounds on the SAME ratio draw, not an (x, y) position.
        image = torch.rand(8, 1, 6, 10, device=device, dtype=dtype)
        aug = K.RandomMosaic(start_ratio_range=(0.3, 0.7), p=1.0)
        aug(image)
        top_left = aug._params["src"][:, 0]
        ratios = torch.stack([top_left[:, 0] / 10, top_left[:, 1] / 6], dim=-1)
        assert bool(((ratios >= 0.3 - 1e-5) & (ratios <= 0.7 + 1e-5)).all())
        # Both entries are bounds on one draw, so both axes vary across the batch.
        assert ratios[:, 0].unique().numel() > 1 and ratios[:, 1].unique().numel() > 1
        # "Draws a pair": the two ratios are separate draws, not one value used twice.
        assert not torch.allclose(ratios[:, 0], ratios[:, 1])

    @pytest.mark.device_agnostic
    def test_convention_jigsaw_gate_is_drawn_per_sample(self):
        # The gate pin above supplies `batch_prob`, so nothing measures which of p / p_batch it comes from.
        aug = K.RandomJigsaw(grid=(2, 2), p=0.5)
        assert aug.p == 0.5 and aug.p_batch == 1.0
        subsets = 0
        for _ in range(20):
            gate = aug.forward_parameters(torch.Size([8, 1, 4, 4]))["batch_prob"] > 0.5
            subsets += 0 < int(gate.sum()) < 8
        assert subsets > 0  # a batch-wide gate can never select a strict subset

    @pytest.mark.device_agnostic
    def test_convention_patchmix_patch_stays_inside_the_image(self):
        # The other patch pins supply or read back `patch_coords`, so the sampled range itself is unpinned.
        aug = K.PatchMix(patch_size=4, p=1.0)
        xs, ys = set(), set()
        for _ in range(20):
            coords = aug.forward_parameters(torch.Size([16, 1, 8, 10]))["patch_coords"]
            xs.update(coords[:, 0].tolist())
            ys.update(coords[:, 1].tolist())
        # Reach as well as containment: every inside corner is drawn, and nothing else.
        assert xs == set(range(10 - 4 + 1)) and ys == set(range(8 - 4 + 1))
