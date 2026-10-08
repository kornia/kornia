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

    @pytest.mark.parametrize(
        "crop_p,resize_p",
        [
            (0.0, 0.0),
            (0.0, 0.5),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.5, 1.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (1.0, 1.0),
        ],
    )
    @pytest.mark.parametrize("height,width", [(6, 8), (9, 13)])
    def test_empty_stage_shape_history_4429(self, crop_p, resize_p, height, width, device, dtype):
        crop = K.RandomCrop((4, 5), cropping_mode="resample", p=crop_p)
        resize = K.Resize((3, 4), p=resize_p)
        sequence = K.AugmentationSequential(crop, resize, data_keys=["input"])
        image = torch.empty(0, 3, height, width, device=device, dtype=dtype)
        output = sequence(image)
        saved_params = deepcopy(sequence._params)

        # With p < 1 an empty batch keeps its input size, because the blend falls back to the skipped
        # branch (tracked in #4429). This pins the current forward, which the tracked shapes must follow.
        crop_size = (4, 5) if crop_p == 1.0 else (height, width)
        output_size = (3, 4) if resize_p == 1.0 else crop_size
        assert saved_params[0].data["forward_input_shape"].tolist() == [0, 3, height, width]
        assert saved_params[1].data["forward_input_shape"].tolist() == [0, 3, *crop_size]
        assert output.shape == (0, 3, *output_size)
        assert output.dtype == image.dtype
        assert output.device == image.device

    @pytest.mark.parametrize(
        "height,width,intermediate_size",
        [(6, 8, (7, 9)), (9, 13, (7, 10))],
    )
    def test_empty_integer_resize_shape_history_4429(self, height, width, intermediate_size, device, dtype):
        first_resize = K.Resize(7, side="short")
        second_resize = K.Resize((3, 4))
        sequence = K.AugmentationSequential(first_resize, second_resize, data_keys=["input"])
        image = torch.empty(0, 3, height, width, device=device, dtype=dtype)
        direct_output = first_resize(image)
        assert direct_output.shape == (0, 3, *intermediate_size)
        assert direct_output.dtype == image.dtype
        assert direct_output.device == image.device

        output = sequence(image)
        saved_params = deepcopy(sequence._params)
        assert output.shape == (0, 3, 3, 4)
        assert output.dtype == image.dtype
        assert output.device == image.device
        assert saved_params[0].data["forward_input_shape"].tolist() == [0, 3, height, width]
        assert saved_params[1].data["forward_input_shape"].tolist() == [0, 3, *intermediate_size]

        for replay_state in ("immediate", "after-forward"):
            if replay_state == "after-forward":
                next_height, next_width = (9, 13) if (height, width) == (6, 8) else (6, 8)
                nonempty = torch.arange(
                    2 * 3 * next_height * next_width,
                    device=device,
                    dtype=dtype,
                ).reshape(2, 3, next_height, next_width)
                sequence(nonempty)

            mask = torch.empty(0, 1, *output.shape[-2:], device=device, dtype=dtype, requires_grad=True)
            intermediate = second_resize.inverse(mask, params=saved_params[1].data)
            assert intermediate.shape == (0, 1, *intermediate_size)
            restored = first_resize.inverse(intermediate, params=saved_params[0].data)
            assert restored.shape == (0, 1, height, width)

            replayed = sequence.inverse(mask, params=saved_params, data_keys=["mask"])
            assert replayed.shape == (0, 1, height, width)
            assert replayed.dtype == mask.dtype
            assert replayed.device == mask.device
            replayed.sum().backward()
            assert mask.grad is not None
            assert mask.grad.shape == mask.shape

    @pytest.mark.parametrize("height,width", [(6, 8), (9, 13)])
    def test_empty_nested_shape_history_4429(self, height, width, device, dtype):
        crop = K.RandomCrop((4, 5), padding=(1, 2, 3, 4), cropping_mode="resample", p=1.0)
        inner_resize = K.Resize((3, 4))
        inner = K.ImageSequential(crop, inner_resize)
        outer_resize = K.Resize((2, 3))
        sequence = K.AugmentationSequential(inner, outer_resize, data_keys=["input"])
        image = torch.empty(0, 3, height, width, device=device, dtype=dtype)
        output = sequence(image)
        saved_params = deepcopy(sequence._params)
        inner_params = saved_params[0].data
        assert output.shape == (0, 3, 2, 3)
        assert output.dtype == image.dtype
        assert output.device == image.device
        assert inner_params[0].data["forward_input_shape"].tolist() == [0, 3, height, width]
        assert inner_params[1].data["forward_input_shape"].tolist() == [0, 3, 4, 5]
        assert saved_params[1].data["forward_input_shape"].tolist() == [0, 3, 3, 4]

        mask = torch.empty(0, 1, *output.shape[-2:], device=device, dtype=dtype, requires_grad=True)
        outer_restored = outer_resize.inverse(mask, params=saved_params[1].data)
        assert outer_restored.shape == (0, 1, 3, 4)
        inner_restored = inner_resize.inverse(outer_restored, params=inner_params[1].data)
        assert inner_restored.shape == (0, 1, 4, 5)
        restored = crop.inverse(inner_restored, params=inner_params[0].data)
        assert restored.shape == (0, 1, height, width)

        replayed = sequence.inverse(mask, params=deepcopy(saved_params), data_keys=["mask"])
        assert replayed.shape == (0, 1, height, width)
        assert replayed.dtype == mask.dtype
        assert replayed.device == mask.device
        replayed.sum().backward()
        assert mask.grad is not None
        assert mask.grad.shape == mask.shape

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

    @pytest.mark.parametrize("height,width", [(6, 8), (9, 13)])
    def test_two_stage_mask_saved_params_replay_4429(self, height, width, device, dtype):
        if not supports_bilinear_2d_grid_sample_backward(device, dtype):
            pytest.skip("The forward resampling kernel does not support this device/dtype's backward pass.")
        crop = K.RandomCrop(
            (4, 5),
            padding=(1, 2, 3, 4),
            cropping_mode="resample",
            p=1.0,
        )
        resize = K.Resize((3, 4))
        sequence = K.AugmentationSequential(crop, resize, data_keys=["input"])
        output = sequence(torch.empty(0, 3, height, width, device=device, dtype=dtype))
        saved_params = deepcopy(sequence._params)
        mask = torch.empty(0, 1, *output.shape[-2:], device=device, dtype=dtype, requires_grad=True)

        intermediate = resize.inverse(mask, params=saved_params[1].data)
        assert intermediate.shape == (0, 1, 4, 5)
        restored = crop.inverse(intermediate, params=saved_params[0].data)
        assert restored.shape == (0, 1, height, width)

        replayed = sequence.inverse(mask, params=deepcopy(saved_params), data_keys=["mask"])
        assert replayed.shape == (0, 1, height, width)
        assert replayed.device == device
        assert replayed.dtype == dtype
        replayed.sum().backward()
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
        # With p < 1 an empty batch keeps its input size (blend fallback, tracked in #4429); the inverse
        # must undo whichever size the forward produced.
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

    @pytest.mark.parametrize(
        "make_first,intermediate_size",
        [
            (lambda: K.LongestMaxSize(12), (9, 12)),
            (lambda: K.Resize(12, side="long"), (9, 12)),
            (lambda: K.SmallestMaxSize(12), (12, 14)),
        ],
        ids=["longest", "resize-long", "smallest"],
    )
    def test_empty_max_size_shape_history_4429(self, make_first, intermediate_size, device, dtype):
        # The container must size an integer resize on the side the module selects, as the module does.
        first = make_first()
        second = K.Resize((3, 4))
        sequence = K.AugmentationSequential(first, second, data_keys=["input"])
        image = torch.empty(0, 3, 9, 11, device=device, dtype=dtype)
        assert first(image).shape == (0, 3, *intermediate_size)
        output = sequence(image)
        assert output.shape == (0, 3, 3, 4)
        params = sequence._params
        assert params[1].data["forward_input_shape"].tolist() == [0, 3, *intermediate_size]
        mask = torch.empty(0, 1, 3, 4, device=device, dtype=dtype)
        assert second.inverse(mask, params=params[1].data).shape == (0, 1, *intermediate_size)
        assert sequence.inverse(mask, params=params, data_keys=["mask"]).shape == (0, 1, 9, 11)

    @pytest.mark.parametrize("same_on_frame", [False, True])
    def test_empty_video_resize_4429(self, same_on_frame, device, dtype):
        # VideoSequential tracks shapes with the same helper; an empty batch used to raise IndexError there.
        sequence = K.VideoSequential(
            K.RandomCrop((6, 7), cropping_mode="resample"),
            K.LongestMaxSize(10),
            data_format="BTCHW",
            same_on_frame=same_on_frame,
        )
        video = torch.empty(0, 2, 3, 9, 11, device=device, dtype=dtype)
        output = sequence(video)
        assert output.shape == (0, 2, 3, 8, 10)
        assert sequence.inverse(output).shape == video.shape
