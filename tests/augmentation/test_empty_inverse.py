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

from testing.base import BaseTester, supports_bilinear_2d_grid_sample_backward


class TestEmptyGeometricInverse(BaseTester):
    @pytest.mark.parametrize("method", ["inverse_inputs", "inverse_masks"])
    @pytest.mark.parametrize(
        "make_aug",
        [
            lambda: K.Resize((4, 5)),
            lambda: K.Resize(4, side="long"),
            lambda: K.LongestMaxSize(4),
            lambda: K.SmallestMaxSize(4),
            lambda: K.CenterCrop((4, 5), cropping_mode="resample"),
            lambda: K.RandomResizedCrop((4, 5), cropping_mode="resample"),
        ],
        ids=["resize-tuple", "resize-int", "longest", "smallest", "center-crop", "resized-crop"],
    )
    def test_restores_spatial_shape_4429(self, make_aug, method, device, dtype):
        aug = make_aug()
        params = aug.forward_parameters((0, 3, 6, 8))
        # A mask can have fewer channels than the image whose parameters it replays.
        channels = 1 if method == "inverse_masks" else 3
        cropped = torch.empty(0, channels, 4, 5, device=device, dtype=dtype, requires_grad=True)
        restored = getattr(aug, method)(cropped, params, aug.flags)
        assert restored.shape == (0, channels, 6, 8)
        assert restored.device == device
        assert restored.dtype == dtype
        restored.sum().backward()
        self.assert_close(cropped.grad, torch.empty_like(cropped))

    @pytest.mark.parametrize("method", ["inverse_inputs", "inverse_masks"])
    def test_without_shape_metadata_preserves_input(self, method, device, dtype):
        aug = K.RandomHorizontalFlip()
        image = torch.empty(0, 3, 4, 5, device=device, dtype=dtype)
        output = getattr(aug, method)(image, {"batch_prob": torch.empty(0)}, aug.flags)
        self.assert_close(output, image)

    @pytest.mark.parametrize(
        "padding,pad_if_needed,size",
        [
            (None, False, (4, 5)),
            (2, False, (4, 5)),
            ((1, 2, 3, 4), False, (4, 5)),
            ((0, 1, 2, 3), True, (10, 12)),
        ],
    )
    @pytest.mark.parametrize("method", ["inverse_inputs", "inverse_masks", "inverse"])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_crop_replays_padding_history_4429(self, padding, pad_if_needed, size, method, p, device, dtype):
        aug = K.RandomCrop(size, padding=padding, pad_if_needed=pad_if_needed, cropping_mode="resample", p=p)
        params = deepcopy(aug.forward_parameters((0, 3, 6, 8)))
        assert params["padding_size"].shape == (0, 4)
        assert params["forward_input_shape"].tolist() == [0, 3, 6, 8]
        channels = 1 if method == "inverse_masks" else 3
        cropped = torch.empty(0, channels, *size, device=device, dtype=dtype, requires_grad=True)
        # Replay must use the original padding history, including after module flags change.
        aug.flags.update(padding=7, pad_if_needed=False, size=(2, 3))
        if method == "inverse":
            restored = aug.inverse(cropped, params=params)
        else:
            restored = getattr(aug, method)(cropped, params, aug.flags)
        assert restored.shape == (0, channels, 6, 8)
        assert restored.device == device
        assert restored.dtype == dtype
        restored.sum().backward()
        self.assert_close(cropped.grad, torch.empty_like(cropped))

    @pytest.mark.parametrize(
        "make_aug",
        [
            lambda: K.Resize((4, 5), p=0.0),
            lambda: K.RandomCrop((4, 5), padding=(1, 2, 3, 4), p=0.0, cropping_mode="resample"),
        ],
        ids=["resize", "crop"],
    )
    @pytest.mark.parametrize("method", ["inverse_inputs", "inverse_masks"])
    def test_nonempty_skipped_batch_is_unchanged(self, make_aug, method, device, dtype):
        aug = make_aug()
        image = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        params = aug.forward_parameters(image.shape)
        output = getattr(aug, method)(image, params, aug.flags)
        self.assert_close(output, image, rtol=0, atol=0)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("height,width", [(6, 8), (9, 13)])
    def test_sequential_round_trip_4429(self, p, height, width, device, dtype):
        if not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("The forward resampling kernel does not support this device/dtype's backward pass.")
        aug = K.AugmentationSequential(
            K.RandomCrop((4, 5), padding=(1, 2, 3, 4), cropping_mode="resample", p=p),
            K.Resize((2, 3)),
            data_keys=["input"],
        )
        image = torch.empty(0, 3, height, width, device=device, dtype=dtype, requires_grad=True)
        output = aug(image)
        mask = torch.empty(0, 1, *output.shape[-2:], device=device, dtype=dtype, requires_grad=True)
        restored_image, restored_mask = aug.inverse(output, mask, data_keys=["input", "mask"])
        assert restored_image.shape == image.shape
        assert restored_mask.shape == (0, 1, height, width)
        for restored, original in ((restored_image, image), (restored_mask, mask)):
            assert restored.device == original.device
            assert restored.dtype == original.dtype
            restored.sum().backward()
            self.assert_close(original.grad, torch.empty_like(original))

    @pytest.mark.parametrize("height,width", [(6, 8), (9, 13)])
    @pytest.mark.parametrize("replay_state", ["immediate", "after-forward", "fresh-instance"])
    def test_mask_only_saved_params_replay_4429(self, height, width, replay_state, device, dtype):
        if not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("The forward resampling kernel does not support this device/dtype's backward pass.")

        def make_sequence():
            return K.AugmentationSequential(
                K.RandomCrop(
                    (4, 5),
                    padding=(1, 2, 3, 4),
                    cropping_mode="resample",
                    p=1.0,
                ),
                data_keys=["input"],
            )

        sequence = make_sequence()
        output = sequence(torch.empty(0, 3, height, width, device=device, dtype=dtype))
        saved_params = deepcopy(sequence._params)
        assert output.shape == (0, 3, 4, 5)
        assert saved_params[0].data["forward_input_shape"].tolist() == [0, 3, height, width]

        if replay_state == "after-forward":
            next_height, next_width = (9, 13) if (height, width) == (6, 8) else (6, 8)
            nonempty = torch.arange(
                2 * 3 * next_height * next_width,
                device=device,
                dtype=dtype,
            ).reshape(2, 3, next_height, next_width)
            sequence(nonempty)
        elif replay_state == "fresh-instance":
            sequence = make_sequence()

        mask = torch.empty(0, 1, *output.shape[-2:], device=device, dtype=dtype, requires_grad=True)
        restored = sequence.inverse(mask, params=saved_params, data_keys=["mask"])
        assert restored.shape == (0, 1, height, width)
        assert restored.device == device
        assert restored.dtype == dtype
        restored.sum().backward()
        assert mask.grad is not None
        assert mask.grad.shape == mask.shape

    @pytest.mark.parametrize(
        "make_sequence",
        [
            lambda: K.AugmentationSequential(
                K.RandomCrop(
                    (4, 5),
                    padding=(1, 2, 3, 4),
                    cropping_mode="resample",
                    p=1.0,
                ),
                data_keys=["input"],
            ),
            lambda: K.AugmentationSequential(K.Resize((4, 5)), data_keys=["input"]),
        ],
        ids=["random-crop", "resize"],
    )
    def test_dynamo_mask_only_saved_params_replay_4429(
        self,
        make_sequence,
        torch_optimizer,
        device,
        dtype,
    ):
        sequence = make_sequence()
        output = sequence(torch.empty(0, 3, 9, 13, device=device, dtype=dtype))
        saved_params = deepcopy(sequence._params)
        mask = torch.empty(0, 1, *output.shape[-2:], device=device, dtype=dtype)

        def replay(saved_mask):
            return sequence.inverse(saved_mask, params=saved_params, data_keys=["mask"])

        eager = replay(mask)
        compiled = torch_optimizer(replay)
        actual = compiled(mask)
        self.assert_close(actual, eager)
        assert actual.shape == (0, 1, 9, 13)
        assert actual.device == device
        assert actual.dtype == dtype

    @pytest.mark.parametrize(
        "make_aug,output_size",
        [
            (lambda p: K.Resize((4, 5), p=p), (4, 5)),
            (lambda p: K.Resize(4, side="long", p=p), (3, 4)),
            (lambda p: K.LongestMaxSize(4, p=p), (3, 4)),
            (lambda p: K.SmallestMaxSize(4, p=p), (4, 5)),
            (lambda p: K.CenterCrop((4, 5), cropping_mode="resample", p=p), (4, 5)),
            (lambda p: K.RandomResizedCrop((4, 5), cropping_mode="resample", p=p), (4, 5)),
            (lambda p: K.RandomCrop((4, 5), padding=(1, 2, 3, 4), cropping_mode="resample", p=p), (4, 5)),
        ],
        ids=["resize-tuple", "resize-int", "longest", "smallest", "center-crop", "resized-crop", "crop"],
    )
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_public_round_trip_4429(self, make_aug, output_size, p, device, dtype):
        aug = make_aug(p)
        if not isinstance(aug, K.Resize) and not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("The forward resampling kernel does not support this device/dtype's backward pass.")
        image = torch.empty(0, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = aug(image)
        expected_size = output_size if p == 1.0 else (6, 8)
        assert output.shape == (0, 3, *expected_size)
        restored = aug.inverse(output)
        assert restored.shape == image.shape
        assert restored.device == image.device
        assert restored.dtype == image.dtype
        restored.sum().backward()
        self.assert_close(image.grad, torch.empty_like(image))

    def test_empty_slice_crop_inverse_is_unsupported(self, device, dtype):
        aug = K.RandomCrop((4, 5), cropping_mode="slice")
        image = torch.empty(0, 3, 6, 8, device=device, dtype=dtype)
        output = aug(image)
        with pytest.raises(NotImplementedError, match="only applicable for resample cropping mode"):
            aug.inverse(output)
