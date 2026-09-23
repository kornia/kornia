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
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        "factory",
        [
            K.RandomMixUpV2,
            lambda: K.RandomCutMixV2(use_correct_lambda=True),
            K.PatchMix,
            K.RandomJigsaw,
            K.RandomMosaic,
            K.RandomTransplantation,
            K.RandomTransplantation3D,
        ],
    )
    def test_convention_mix_has_no_matrix_or_inverse(self, factory):
        aug = factory()
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
        ("factory", "per_sample"),
        [
            (lambda: K.RandomJigsaw(grid=(2, 2), p=0.5, same_on_batch=True), False),
            (lambda: K.RandomTransplantation(p=0.5), True),
        ],
    )
    def test_convention_mix_gate_is_per_sample_or_batch_wide(self, factory, per_sample):
        # A batch-wide gate never selects a strict subset of the rows; a per-sample one misses doing so in 20
        # draws of 8 rows with probability (2 / 256) ** 20. Jigsaw and Mosaic's per-sample `p` and the batch-wide
        # `p` of MixUp, CutMix and PatchMix (#4425) are pinned in test_base.py.
        aug = factory()
        shape = torch.Size([8, 8, 8]) if isinstance(aug, K.RandomTransplantation) else torch.Size([8, 1, 8, 8])
        subsets = 0
        for _ in range(20):
            gate = aug.forward_parameters(shape)["batch_prob"] > 0.5
            assert gate.numel() == 8
            subsets += 0 < int(gate.sum()) < 8
        assert (subsets > 0) == per_sample

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
        # #4651 / #4676: the key is validated before the gate is consulted.
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
        # #4676: the divisibility check runs whatever the gate.
        for shape, grid in (((1, 3, 3, 4), (2, 2)), ((2, 1, 9, 13), (2, 3)), ((2, 1, 8, 13), (2, 3))):
            image = torch.rand(shape, device=device, dtype=dtype)
            with pytest.raises(RuntimeError, match="must be divisible by grid"):
                K.RandomJigsaw(grid=grid, p=p)(image)

    @pytest.mark.device_agnostic
    def test_wart_mix_class_handlers_ignore_a_replayed_partial_gate_4775(self):
        # #4775: flips when the class handler honours batch_prob, i.e. row 1 is labelled [20, 20, 0].
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
        # RandomCutMixV2 has the same handler split: a 2 x 2 cut, and row 1 is gated off.
        aug = K.RandomCutMixV2(p=1.0, use_correct_lambda=True, data_keys=["input", "class"])
        aug(image, labels)
        params = dict(aug._params)
        params["mix_pairs"] = torch.tensor([[1, 2, 0]])
        params["crop_src"] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]).expand(1, 3, 4, 2).clone()
        params["batch_prob"] = torch.tensor([1.0, 0.0, 1.0])
        output, mixed = aug(image, labels, params=params)
        self.assert_close(output[1], image[1])
        self.assert_close(mixed[0], torch.tensor([[10.0, 20.0, 0.75], [20.0, 30.0, 0.75], [30.0, 10.0, 0.75]]))

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
    def test_convention_cutmix_same_on_batch_shares_cut_geometry(self):
        from kornia.geometry.bbox import infer_bbox_shape

        # The default ``cut_size`` lets the sizes vary, so sharing is what makes the boxes equal: a fixed
        # ``cut_size`` would pass with ``same_on_batch=False`` as well.
        shape = torch.Size([8, 1, 64, 48])
        for seed in range(10):
            torch.manual_seed(seed)
            shared = K.RandomCutMixV2(p=1.0, num_mix=2, same_on_batch=True, use_correct_lambda=True)
            crop_src = shared.forward_parameters(shape)["crop_src"]
            self.assert_close(crop_src, crop_src[0, 0].expand_as(crop_src), rtol=0, atol=0)
            independent = K.RandomCutMixV2(p=1.0, num_mix=2, same_on_batch=False, use_correct_lambda=True)
            heights, widths = infer_bbox_shape(independent.forward_parameters(shape)["crop_src"].flatten(0, 1))
            assert heights.unique().numel() > 1 and widths.unique().numel() > 1

    @pytest.mark.device_agnostic
    def test_wart_cutmix_same_on_batch_repeats_one_cut_per_mix_4805(self):
        # #4805: with same_on_batch=True every mix repeats the first cut from the same donor, yet each mix is
        # labelled. Flips when the mixes differ or collapse to one, i.e. when the labels match the pixels.
        torch.manual_seed(0)
        image = torch.arange(4.0).view(4, 1, 1, 1).expand(4, 1, 8, 8).clone()  # row i is filled with i
        aug = K.RandomCutMixV2(
            num_mix=2, same_on_batch=True, p=1.0, use_correct_lambda=True, data_keys=["input", "class"]
        )
        output, mixed = aug(image, torch.tensor([0, 1, 2, 3]))
        assert mixed.shape[0] == 2 and torch.equal(mixed[0], mixed[1])
        replaced = (output[:, 0] != image[:, 0]).float().mean((-2, -1))
        self.assert_close(replaced, 1 - mixed[0, :, 2])  # one cut's area, while two mixes credit the donor

    @pytest.mark.device_agnostic
    def test_convention_cutmix_placement_is_drawn_per_cut_4712(self):
        from kornia.geometry.bbox import infer_bbox_shape

        # #4712: every cut draws its own placement. With the size fixed, only the placement can differ, so equal
        # boxes here would mean the placement is shared.
        shape = torch.Size([8, 1, 64, 48])
        for seed in range(10):
            torch.manual_seed(seed)
            aug = K.RandomCutMixV2(p=1.0, cut_size=(0.5, 0.5), num_mix=2, same_on_batch=False, use_correct_lambda=True)
            crop_src = aug.forward_parameters(shape)["crop_src"]  # (num_mix, B, 4, 2)
            heights, widths = infer_bbox_shape(crop_src.flatten(0, 1))
            assert heights.unique().numel() == 1 and widths.unique().numel() == 1  # the sizes are pinned...
            top_left = crop_src[:, :, 0, :]
            assert top_left[0].unique(dim=0).shape[0] > 1  # ...and rows of one mix land in different places,
            assert top_left[1].unique(dim=0).shape[0] > 1
            assert bool((top_left[0] != top_left[1]).any())  # as do the two mixes of one row.
            # Each axis is drawn on its own: sharing only the x or only the y draw would still vary the pairs.
            assert top_left[..., 0].unique().numel() > 1 and top_left[..., 1].unique().numel() > 1
            # Every cut fits inside the image (the far corner is inclusive).
            assert bool((crop_src >= 0).all())
            assert bool((crop_src[..., 0] <= shape[-1] - 1).all())
            assert bool((crop_src[..., 1] <= shape[-2] - 1).all())

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_patchmix_rejects_a_patch_larger_than_the_image(self, p):
        # #4650 / #4680: checked whatever the gate.
        image = torch.rand(4, 1, 8, 10)
        for patch_size in (9, 16):
            with pytest.raises(ValueError, match="`patch_size` to fit the input"):
                K.PatchMix(patch_size=patch_size, p=p)(image)
        assert K.PatchMix(patch_size=8, p=p)(image).shape == image.shape  # min(H, W) itself is accepted

    @pytest.mark.device_agnostic
    def test_convention_patchmix_same_on_batch_shares_the_patch_but_not_the_pairing(self):
        # #4680. Each donor's pixel value is its batch index, an oracle independent of patch-copy slicing.
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
        with pytest.raises(TypeError):
            K.RandomMosaic(p=1.0, cropping_mode="resample")(image)
        self.assert_close(K.RandomMosaic(p=0.0, cropping_mode="resample")(image), image)  # no selection, no raise

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("image_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_mix_labels_stay_exact_for_half_precision_images(self, image_dtype, p):
        # #4657: a half-precision image gives float32 labels, so class ids such as 257 and 999 stay exact.
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

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("image_dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_mix_forward_parameters_replay_with_class_4706(self, image_dtype, p):
        # A forward_parameters() dictionary has no "dtype"; forward takes it from the input, so replaying it with
        # the "class" key matches the call that sampled the same draws, instead of raising KeyError.
        image = torch.rand(2, 1, 4, 4, dtype=image_dtype)
        labels = torch.tensor([1, 2])
        for aug in (K.RandomMixUpV2(p=p), K.RandomCutMixV2(p=p, use_correct_lambda=True)):
            torch.manual_seed(0)
            sampled, sampled_labels = aug(image, labels, data_keys=["input", "class"])
            torch.manual_seed(0)
            params = aug.forward_parameters(image.shape)
            output, mixed = aug(image, labels, params=params, data_keys=["input", "class"])
            assert "dtype" not in params  # the caller's dictionary is left as it was
            assert torch.equal(output, sampled) and torch.equal(mixed, sampled_labels)
            assert mixed.dtype == _label_dtype(image_dtype)

    def test_convention_mix_replay_takes_dtype_from_the_input_not_the_dictionary(self):
        # A dictionary recorded on a float32 image and replayed on a float64 one gives float64 labels, like a fresh
        # draw on the float64 image: the recorded "dtype" is not reused.
        image = torch.rand(2, 1, 4, 4)
        labels = torch.tensor([1, 2])
        for aug in (K.RandomMixUpV2(p=1.0), K.RandomCutMixV2(p=1.0, use_correct_lambda=True)):
            aug(image, labels, data_keys=["input", "class"])
            params = dict(aug._params)
            _, mixed = aug(image.double(), labels, params=params, data_keys=["input", "class"])
            assert mixed.dtype == torch.float64

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("replay", [False, True])
    @pytest.mark.parametrize("image_dtype", [torch.uint8, torch.int32, torch.int64, torch.bool])
    def test_convention_mix_rejects_a_non_floating_image(self, image_dtype, replay):
        # A fresh draw and a replayed dictionary go through the same dtype guard.
        image = torch.rand(2, 1, 4, 4)
        for aug_cls in (K.RandomMixUpV2, K.RandomCutMixV2, K.RandomJigsaw, K.RandomMosaic):
            aug = aug_cls(p=1.0)
            params = aug.forward_parameters(image.shape) if replay else None
            with pytest.raises(TypeError, match="Expected input of"):
                aug((image * 255).to(image_dtype), params=params)

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
    def test_convention_jigsaw_ensure_perm_rejects_the_image_preserving_permutation_4703(self):
        # ensure_perm rejects [0, 2, 1, 3], the no-op on a 2 x 2 grid, and still draws arange(N), which is a real
        # shuffle there. 400 draws miss one of the 23 remaining permutations with probability 2e-8.
        aug = K.RandomJigsaw(grid=(2, 2), p=1.0, ensure_perm=True)
        drawn = set()
        for _ in range(50):
            permutation = aug.forward_parameters(torch.Size([8, 1, 4, 4]))["permutation"]
            drawn.update(tuple(row) for row in permutation.tolist())
        assert (0, 2, 1, 3) not in drawn
        assert (0, 1, 2, 3) in drawn
        assert len(drawn) == 23
        # On a single-row or single-column grid the image-preserving permutation is arange(N), so the only
        # permutation left for two patches is the swap.
        for grid in ((1, 2), (2, 1)):
            permutation = K.RandomJigsaw(grid=grid, p=1.0).forward_parameters(torch.Size([8, 1, 4, 4]))["permutation"]
            assert permutation.tolist() == [[1, 0]] * 8
        # The guarantee as stated: with ensure_perm the output never equals the input.
        cells = torch.arange(4.0).view(2, 2)
        picture = cells.repeat_interleave(2, 0).repeat_interleave(2, 1)[None, None].repeat(64, 1, 1, 1)
        output = aug(picture)
        assert not (output == picture).flatten(1).all(1).any()

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
