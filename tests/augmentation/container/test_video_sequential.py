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

import kornia
import kornia.augmentation as K

from testing.augmentation.utils import assert_close, reproducibility_test


class TestVideoSequential:
    def test_smoke(self, device, dtype):
        input_1 = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 3, 1, 1)
        input_2 = torch.randn(4, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 3, 1, 1)
        aug_list = K.VideoSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1))
        aug_list(input_1)
        aug_list(input_2)

    @pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (2, 3, 5, 6), (2, 3, 4, 5, 6, 7)])
    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    def test_exception(self, shape, data_format, device, dtype):
        aug_list = K.VideoSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1), data_format=data_format, same_on_frame=True)
        img = torch.randn(*shape, device=device, dtype=dtype)
        with pytest.raises(AssertionError):
            aug_list(img)

    @pytest.mark.parametrize(
        "augmentation",
        [
            K.RandomAffine(360, p=1.0),
            K.CenterCrop((3, 3), p=1.0),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomCrop((5, 5), p=1.0),
            K.RandomErasing(p=1.0),
            K.RandomGrayscale(p=1.0),
            K.RandomHorizontalFlip(p=1.0),
            K.RandomVerticalFlip(p=1.0),
            K.RandomPerspective(p=1.0),
            K.RandomResizedCrop((5, 5), p=1.0),
            K.RandomRotation(360.0, p=1.0),
            K.RandomSolarize(p=1.0),
            K.RandomPosterize(p=1.0),
            K.RandomSharpness(p=1.0),
            K.RandomEqualize(p=1.0),
            K.RandomMotionBlur(3, 35.0, 0.5, p=1.0),
            K.Normalize(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]), p=1.0),
            K.Denormalize(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]), p=1.0),
            K.RandomAutoContrast(p=1.0),
            K.RandomShear((10.0, 10.0), p=1.0),
            K.RandomTranslate((0.5, 0.5), p=1.0),
        ],
    )
    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    def test_augmentation(self, augmentation, data_format, device, dtype):
        input = torch.randint(255, (1, 3, 3, 5, 6), device=device, dtype=dtype).repeat(2, 1, 1, 1, 1) / 255.0
        torch.manual_seed(21)
        aug_list = K.VideoSequential(augmentation, data_format=data_format, same_on_frame=True)
        reproducibility_test(input, aug_list)

    @pytest.mark.parametrize(
        "augmentations",
        [
            [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)],
            [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)],
            [K.RandomAffine(360, p=1.0), kornia.color.BgrToRgb()],
            [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.0), K.RandomAffine(360, p=0.0)],
            [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.0)],
            [K.RandomAffine(360, p=0.0)],
            # NOTE: RandomMixUpV2 failed occasionally but always passed in the debugger. Unable to debug now.
            # [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0), K.RandomMixUpV2(p=1.0)],
        ],
    )
    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    @pytest.mark.parametrize("random_apply", [1, (1, 1), (1,), 10, True, False])
    def test_same_on_frame(self, augmentations, data_format, random_apply, device, dtype):
        aug_list = K.VideoSequential(
            *augmentations, data_format=data_format, same_on_frame=True, random_apply=random_apply
        )

        if data_format == "BCTHW":
            input = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
            output = aug_list(input)
            assert_close(output[:, :, 0], output[:, :, 1])
            assert_close(output[:, :, 1], output[:, :, 2])
            assert_close(output[:, :, 2], output[:, :, 3])
        if data_format == "BTCHW":
            input = torch.randn(2, 1, 3, 5, 6, device=device, dtype=dtype).repeat(1, 4, 1, 1, 1)
            output = aug_list(input)
            assert_close(output[:, 0], output[:, 1])
            assert_close(output[:, 1], output[:, 2])
            assert_close(output[:, 2], output[:, 3])
        reproducibility_test(input, aug_list)

    @pytest.mark.parametrize(
        "augmentations",
        [
            [K.RandomAffine(360, p=1.0)],
            [K.RandomCrop((2, 2), padding=2)],
            [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)],
            [K.RandomAffine(360, p=0.0), K.ImageSequential(K.RandomAffine(360, p=0.0))],
            [K.ImageSequential(K.RandomAffine(360, p=1.0))],
            [K.RandomAffine(360, p=1.0), K.ImageSequential(K.RandomAffine(360, p=0.5))],
        ],
    )
    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    def test_against_sequential(self, augmentations, data_format, device, dtype):
        aug_list_1 = K.VideoSequential(*augmentations, data_format=data_format, same_on_frame=False)
        aug_list_2 = torch.nn.Sequential(*augmentations)

        if data_format == "BCTHW":
            input = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
        if data_format == "BTCHW":
            input = torch.randn(2, 1, 3, 5, 6, device=device, dtype=dtype).repeat(1, 4, 1, 1, 1)

        torch.manual_seed(0)
        output_1 = aug_list_1(input)

        torch.manual_seed(0)
        if data_format == "BCTHW":
            input = input.transpose(1, 2)
        output_2 = aug_list_2(input.reshape(-1, 3, 5, 6))
        if any(isinstance(a, K.RandomCrop) for a in augmentations):
            output_2 = output_2.view(2, 4, 3, 2, 2)
        else:
            output_2 = output_2.view(2, 4, 3, 5, 6)
        if data_format == "BCTHW":
            output_2 = output_2.transpose(1, 2)
        assert_close(output_1, output_2)

    @pytest.mark.skip(reason="turn off due to Union Type")
    def test_jit(self, device, dtype):
        B, C, D, H, W = 2, 3, 5, 4, 4
        img = torch.ones(B, C, D, H, W, device=device, dtype=dtype)
        op = K.VideoSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1), same_on_frame=True)
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    def test_autocast(self, data_format, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        tfs = (K.RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0), K.RandomGaussianBlur((3, 3), (0.1, 3), p=1))
        aug = K.VideoSequential(*tfs, data_format=data_format, random_apply=True)
        if data_format == "BCTHW":
            imgs = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
        elif data_format == "BTCHW":
            imgs = torch.randn(2, 1, 3, 5, 6, device=device, dtype=dtype).repeat(1, 4, 1, 1, 1)

        with torch.autocast(device.type):
            output = aug(imgs)

        assert output.dtype == dtype, "Output image dtype should match the input dtype"

    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    @pytest.mark.parametrize(
        "same_on_frame,same_on_batch",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_all_frame_and_batch(self, data_format, same_on_frame, same_on_batch, device, dtype):
        aug = K.VideoSequential(
            K.RandomCrop(size=(4, 4), p=1.0, same_on_batch=same_on_batch),
            data_format=data_format,
            same_on_frame=same_on_frame,
        )

        if data_format == "BCTHW":
            if same_on_frame:
                start = torch.randn(1, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
                x = start.repeat(2, 1, 1, 1, 1)
            else:
                start = torch.randn(1, 3, 4, 5, 6, device=device, dtype=dtype)
                x = start.repeat(2, 1, 1, 1, 1)
        elif same_on_frame:
            start = torch.randn(1, 1, 3, 5, 6, device=device, dtype=dtype).repeat(1, 4, 1, 1, 1)
            x = start.repeat(2, 1, 1, 1, 1)
        else:
            start = torch.randn(1, 4, 3, 5, 6, device=device, dtype=dtype)
            x = start.repeat(2, 1, 1, 1, 1)

        torch.manual_seed(0)
        y = aug(x)

        if data_format == "BCTHW":
            same_frame = torch.allclose(y[:, :, 0], y[:, :, 1])
            same_batch = torch.allclose(y[0, :, 0], y[1, :, 0])
        else:
            same_frame = torch.allclose(y[:, 0], y[:, 1])
            same_batch = torch.allclose(y[0, 0], y[1, 0])

        assert same_frame == same_on_frame
        assert same_batch == same_on_batch

        reproducibility_test(x, aug)

    def test_tensor_annotation_methods_preserve_video_shape_and_accept_empty_or_noncontiguous(self, device, dtype):
        video = torch.rand(2, 3, 1, 6, 6, device=device, dtype=dtype)
        aug = K.VideoSequential(K.RandomHorizontalFlip(p=1.0), data_format="BTCHW")
        aug(video)
        params = aug._params

        # Integer coordinates keep the flip and its inverse exact in float16 and bfloat16.
        keypoints = torch.randint(0, 6, (2, 3, 4, 2), device=device).to(dtype)
        transformed = aug.transform_keypoints(keypoints, params)
        assert transformed.shape == keypoints.shape
        assert_close(aug.inverse_keypoints(transformed, params), keypoints)

        noncontiguous_keypoints = keypoints.transpose(0, 1).contiguous().transpose(0, 1)
        assert not noncontiguous_keypoints.is_contiguous()
        assert_close(aug.transform_keypoints(noncontiguous_keypoints, params), transformed)
        assert_close(aug.inverse_keypoints(noncontiguous_keypoints, params), aug.inverse_keypoints(keypoints, params))

        empty_keypoints = keypoints[:, :, :0]
        assert aug.transform_keypoints(empty_keypoints, params).shape == empty_keypoints.shape
        assert aug.inverse_keypoints(empty_keypoints, params).shape == empty_keypoints.shape

        boxes = torch.randint(0, 6, (2, 3, 2, 4, 2), device=device).to(dtype)
        transformed_boxes = aug.transform_boxes(boxes, params)
        assert transformed_boxes.shape == boxes.shape
        assert_close(aug.inverse_boxes(transformed_boxes, params), boxes)

        noncontiguous_boxes = boxes.transpose(0, 1).contiguous().transpose(0, 1)
        assert not noncontiguous_boxes.is_contiguous()
        assert_close(aug.transform_boxes(noncontiguous_boxes, params), transformed_boxes)
        assert_close(aug.inverse_boxes(noncontiguous_boxes, params), aug.inverse_boxes(boxes, params))

        empty_boxes = boxes[:, :, :0]
        assert aug.transform_boxes(empty_boxes, params).shape == empty_boxes.shape
        assert aug.inverse_boxes(empty_boxes, params).shape == empty_boxes.shape

    def test_tensor_annotation_methods_accept_an_empty_video_batch(self, device, dtype):
        aug = K.VideoSequential(K.RandomHorizontalFlip(p=1.0), data_format="BTCHW")
        aug(torch.rand(0, 3, 1, 6, 6, device=device, dtype=dtype))
        params = aug._params

        keypoints = torch.rand(0, 3, 4, 2, device=device, dtype=dtype)
        assert aug.transform_keypoints(keypoints, params).shape == keypoints.shape
        assert aug.inverse_keypoints(keypoints, params).shape == keypoints.shape

        boxes = torch.rand(0, 3, 2, 4, 2, device=device, dtype=dtype)
        assert aug.transform_boxes(boxes, params).shape == boxes.shape
        assert aug.inverse_boxes(boxes, params).shape == boxes.shape

    @pytest.mark.parametrize(
        "mix",
        [
            lambda sob: K.RandomMixUpV2(p=1.0, same_on_batch=sob),
            lambda sob: K.PatchMix(patch_size=2, p=1.0, same_on_batch=sob),
            lambda sob: K.RandomMosaic(p=1.0),
        ],
        ids=["mixup", "patchmix", "mosaic"],
    )
    @pytest.mark.parametrize("same_on_batch", [False, True])
    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    def test_mix_same_on_frame_pairs_matching_frames(self, mix, same_on_batch, data_format, device, dtype):
        batch_size, frame_num = 3, 4
        aug = mix(same_on_batch)
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)
        seq = K.VideoSequential(aug, data_format=data_format, same_on_frame=True)

        torch.manual_seed(0)
        out = seq(video.transpose(1, 2).contiguous() if data_format == "BCTHW" else video)
        out = out.transpose(1, 2) if data_format == "BCTHW" else out

        torch.manual_seed(0)
        params = aug.forward_parameters(torch.Size((batch_size, 3, 8, 8)))
        for frame in range(frame_num):
            assert_close(out[:, frame], aug(video[:, frame], params=params))

    @pytest.mark.parametrize(
        "mix",
        [
            lambda sob: K.RandomMixUpV2(p=1.0, same_on_batch=sob),
            lambda sob: K.PatchMix(patch_size=2, p=1.0, same_on_batch=sob),
            lambda sob: K.RandomMosaic(p=1.0),
        ],
        ids=["mixup", "patchmix", "mosaic"],
    )
    @pytest.mark.parametrize("same_on_batch", [False, True])
    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    def test_mix_not_same_on_frame_matches_flattened_frames(self, mix, same_on_batch, data_format, device, dtype):
        # With same_on_frame=False the B * T frames are one image batch for the mix, whose own generator applies
        # same_on_batch. One draw per frame index, repeated over the videos, took every partner from the first video.
        batch_size, frame_num = 3, 4
        aug = mix(same_on_batch)
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)
        seq = K.VideoSequential(aug, data_format=data_format, same_on_frame=False)

        torch.manual_seed(0)
        out = seq(video.transpose(1, 2).contiguous() if data_format == "BCTHW" else video)
        out = out.transpose(1, 2) if data_format == "BCTHW" else out

        torch.manual_seed(0)
        params = aug.forward_parameters(torch.Size((batch_size * frame_num, 3, 8, 8)))
        assert_close(out, aug(video.reshape(-1, 3, 8, 8), params=params).reshape(video.shape))

    @staticmethod
    def _tagged_boxes(batch_size, frame_num, device, dtype):
        """One box per frame, shifted by its video and frame, so a box names the frame it came from."""
        box = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]], device=device, dtype=dtype)
        shift = torch.stack(
            torch.meshgrid(
                torch.arange(batch_size, device=device, dtype=dtype),
                torch.arange(frame_num, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        )
        return (box + 0.5 * shift[:, :, None]).unsqueeze(2).contiguous()  # (B, T, 1, 4, 2)

    @staticmethod
    def _dense_boxes(boxes):
        from kornia.augmentation.utils.helpers import _boxes_to_padded_tensor

        return _boxes_to_padded_tensor(boxes, "vertices_plus")

    @pytest.mark.parametrize("p", [1.0, 0.5])
    def test_mosaic_boxes_same_on_frame_follow_each_frame_6173(self, p, device, dtype):
        # Frame t of the boxes is the mosaic of frame t of every video with the video-level parameters, as for the
        # images. VideoBoxes indexes whole videos, so the mosaic's frame permutation raised IndexError (#6173).
        batch_size, frame_num = 3, 2
        mosaic = K.RandomMosaic(p=p)
        aug = K.AugmentationSequential(K.VideoSequential(mosaic, same_on_frame=True), data_keys=["input", "bbox"])
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)
        boxes = self._tagged_boxes(batch_size, frame_num, device, dtype)
        partial = False
        for seed in range(6):
            torch.manual_seed(seed)
            out, out_boxes = aug(video, boxes)
            torch.manual_seed(seed)
            params = mosaic.forward_parameters(torch.Size((batch_size, 3, 8, 8)))
            applied = params["batch_prob"] > 0.5
            partial |= bool(applied.any()) and not bool(applied.all())
            for frame in range(frame_num):
                assert_close(out[:, frame], mosaic(video[:, frame], params=params))
                frame_boxes = kornia.geometry.boxes.Boxes.from_tensor(boxes[:, frame], mode="vertices_plus")
                expected = self._dense_boxes(mosaic.transform_boxes(frame_boxes, params, mosaic.flags))
                assert out_boxes.shape == (batch_size, frame_num, *expected.shape[1:])
                assert_close(out_boxes[:, frame], expected)
        assert partial == (p < 1.0)

    @pytest.mark.parametrize("p", [1.0, 0.5])
    def test_mosaic_boxes_not_same_on_frame_match_flattened_frames_6173(self, p, device, dtype):
        # With same_on_frame=False the mosaic runs on the B * T frames; the boxes follow the same frames (#6173).
        batch_size, frame_num = 2, 3
        mosaic = K.RandomMosaic(p=p)
        aug = K.AugmentationSequential(K.VideoSequential(mosaic, same_on_frame=False), data_keys=["input", "bbox"])
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)
        boxes = self._tagged_boxes(batch_size, frame_num, device, dtype)
        frames = video.reshape(-1, 3, 8, 8)
        frame_boxes = kornia.geometry.boxes.Boxes.from_tensor(boxes.flatten(0, 1), mode="vertices_plus")
        partial = False
        for seed in range(6):
            torch.manual_seed(seed)
            out, out_boxes = aug(video, boxes)
            torch.manual_seed(seed)
            params = mosaic.forward_parameters(torch.Size((batch_size * frame_num, 3, 8, 8)))
            applied = params["batch_prob"] > 0.5
            partial |= bool(applied.any()) and not bool(applied.all())
            assert_close(out, mosaic(frames, params=params).reshape(video.shape))
            expected = self._dense_boxes(mosaic.transform_boxes(frame_boxes.clone(), params, mosaic.flags))
            assert_close(out_boxes, expected.reshape(batch_size, frame_num, *expected.shape[1:]))
        assert partial == (p < 1.0)

    def test_mosaic_video_boxes_object_input_exports_padded_6173(self, device, dtype):
        # A VideoBoxes input comes back as VideoBoxes. After a partial gate its frames hold 4 or 1 boxes, and
        # to_tensor() returns them as one padded (B, T, N, 4, 2) tensor, equal to the dense tensor input's output.
        from kornia.geometry.boxes import VideoBoxes

        batch_size, frame_num = 3, 2
        mosaic = K.RandomMosaic(p=0.5)
        aug = K.AugmentationSequential(K.VideoSequential(mosaic, same_on_frame=True), data_keys=["input", "bbox"])
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)
        boxes = self._tagged_boxes(batch_size, frame_num, device, dtype)
        partial = False
        for seed in range(6):
            torch.manual_seed(seed)
            _, dense = aug(video, boxes)
            torch.manual_seed(seed)
            _, wrapped = aug(video, VideoBoxes.from_tensor(boxes))
            torch.manual_seed(seed)
            applied = mosaic.forward_parameters(torch.Size((batch_size, 3, 8, 8)))["batch_prob"] > 0.5
            partial |= bool(applied.any()) and not bool(applied.all())
            assert isinstance(wrapped, VideoBoxes)
            exported = wrapped.to_tensor()
            assert exported.shape == dense.shape
            assert_close(exported, dense)
            assert wrapped.to_tensor("xyxy").shape == (*dense.shape[:3], 4)
        assert partial

    @pytest.mark.parametrize("then_flip", [False, True])
    def test_mosaic_video_boxes_padding_exports_as_zero_6173(self, then_flip, device, dtype):
        # A video a partial gate skips keeps its one box per frame, followed by padding rows up to the mosaic box
        # count. A stored zero row exports as [0, 0, 1, 1] in the exclusive modes and a later flip moves it (to
        # [7, 0, 8, 1] on 8-pixel frames), so the padding has to export as zero in every mode and after indexing,
        # while the real rows export as their stored coordinates do.
        from kornia.geometry.boxes import Boxes, VideoBoxes

        batch_size, frame_num = 3, 2
        mosaic = K.RandomMosaic(p=0.5)
        children = [mosaic] + ([K.RandomHorizontalFlip(p=1.0)] if then_flip else [])
        aug = K.AugmentationSequential(K.VideoSequential(*children, same_on_frame=True), data_keys=["input", "bbox"])
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)
        boxes = self._tagged_boxes(batch_size, frame_num, device, dtype)
        partial = False
        for seed in range(6):
            torch.manual_seed(seed)
            _, dense = aug(video, boxes)
            torch.manual_seed(seed)
            _, wrapped = aug(video, VideoBoxes.from_tensor(boxes))
            torch.manual_seed(seed)
            applied = mosaic.forward_parameters(torch.Size((batch_size, 3, 8, 8)))["batch_prob"] > 0.5
            if bool(applied.all()):
                continue
            partial |= bool(applied.any())
            num_boxes = wrapped.data.shape[1]
            # In a skipped video, the rows after each frame's own box are padding.
            padding = ~applied.to(device)[:, None, None] & (torch.arange(num_boxes, device=device) >= boxes.shape[2])
            padding = padding.expand(batch_size, frame_num, num_boxes)
            assert_close(wrapped.to_tensor(), dense)
            for mode in ("xyxy", "xyxy_plus", "xywh", "vertices", "vertices_plus"):
                exported = wrapped.to_tensor(mode)
                assert (exported[padding] == 0).all(), mode
                frames = Boxes(wrapped.data, False).to_tensor(mode).reshape(exported.shape)
                assert_close(exported[~padding], frames[~padding])
                # Indexing whole videos keeps each frame's padding.
                assert_close(wrapped[1:].to_tensor(mode), exported[1:])
                assert_close(wrapped[0].to_tensor(mode), exported[:1])
        assert partial

    @staticmethod
    def _tagged_video(batch_size, frame_num, data_format, device, dtype):
        """Build a video whose pixel value is ``10 * (video + 1) + frame``.

        Every pixel then names the video and the frame it came from, so a mixed output can be decoded back into
        the donor video and donor frame of each pixel.
        """
        video = 10.0 * (torch.arange(batch_size, device=device, dtype=dtype) + 1)[:, None] + torch.arange(
            frame_num, device=device, dtype=dtype
        )
        video = video[:, :, None, None, None].expand(batch_size, frame_num, 3, 8, 8).contiguous()
        return video.transpose(1, 2).contiguous() if data_format == "BCTHW" else video

    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    @pytest.mark.parametrize("random_apply", [False, 1])
    @pytest.mark.parametrize("num_mix", [1, 2])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_cutmix_same_on_frame_shares_box_and_partner(
        self, data_format, random_apply, num_mix, same_on_batch, device, dtype
    ):
        # RandomCutMixV2 draws (num_mix, B, ...) parameters, so the frame repeat must not touch the mix axis.
        batch_size, frame_num = 4, 3
        aug = K.VideoSequential(
            K.RandomCutMixV2(num_mix=num_mix, p=1.0, same_on_batch=same_on_batch, use_correct_lambda=True),
            data_format=data_format,
            same_on_frame=True,
            random_apply=random_apply,
        )
        video = self._tagged_video(batch_size, frame_num, data_format, device, dtype)

        torch.manual_seed(0)
        out = aug(video)
        assert out.shape == video.shape

        out = out.transpose(1, 2) if data_format == "BCTHW" else out  # (B, T, C, H, W)
        frame_id = torch.arange(frame_num, device=device, dtype=dtype)[None, :, None, None, None]
        # A frame only ever takes pixels from the same frame index of some video: value = 10 * (donor + 1) + frame.
        donor = (out - frame_id) / 10.0 - 1.0
        assert_close(donor, donor.round())
        # Every frame of a video cuts the same box from the same partner video.
        for frame in range(1, frame_num):
            assert_close(donor[:, frame], donor[:, 0])
        # The cut is not a no-op: at least one video received pixels from another one.
        own = torch.arange(batch_size, device=device, dtype=dtype)[:, None, None, None, None]
        assert (donor != own).any()

    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_cutmix_same_on_frame_with_same_on_batch_and_p(self, data_format, same_on_batch, p, device, dtype):
        aug = K.VideoSequential(
            K.RandomCutMixV2(p=p, same_on_batch=same_on_batch, use_correct_lambda=True),
            data_format=data_format,
            same_on_frame=True,
        )
        video = self._tagged_video(3, 4, data_format, device, dtype)

        out = aug(video)

        assert out.shape == video.shape
        changed = out != video
        if p == 0.0:
            assert not changed.any()
        frame_dim = 2 if data_format == "BCTHW" else 1
        for frame in range(1, 4):
            assert torch.equal(changed.select(frame_dim, frame), changed.select(frame_dim, 0))

    def test_cutmix_video_params_follow_the_flattened_frames(self, device, dtype):
        batch_size, frame_num, num_mix = 3, 4, 2
        aug = K.VideoSequential(
            K.RandomCutMixV2(num_mix=num_mix, p=1.0, use_correct_lambda=True), data_format="BTCHW", same_on_frame=True
        )
        aug(torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype))
        params = aug._params[0].data

        flat = batch_size * frame_num
        assert params["mix_pairs"].shape == (num_mix, flat)
        assert params["crop_src"].shape == (num_mix, flat, 4, 2)
        assert params["image_shape"].shape == (2,)
        assert params["batch_prob"].shape == (flat,)

        pairs = params["mix_pairs"].view(num_mix, batch_size, frame_num)
        boxes = params["crop_src"].view(num_mix, batch_size, frame_num, 4, 2)
        for frame in range(frame_num):
            # Frame t of a video is mixed with frame t of the partner video, in the same box as frame 0.
            assert torch.equal(pairs[..., frame] % frame_num, torch.full_like(pairs[..., frame], frame))
            assert torch.equal(pairs[..., frame] // frame_num, pairs[..., 0] // frame_num)
            assert torch.equal(boxes[:, :, frame], boxes[:, :, 0])

    def test_cutmix_not_same_on_frame_still_runs(self, device, dtype):
        aug = K.VideoSequential(K.RandomCutMixV2(p=1.0, use_correct_lambda=True), same_on_frame=False)
        video = torch.rand(2, 4, 3, 8, 8, device=device, dtype=dtype)
        assert aug(video).shape == video.shape

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("frame_num", [1, 4])
    def test_cutmix_default_same_on_frame_replay(self, batch_size, frame_num, device, dtype):
        aug = K.VideoSequential(K.RandomCutMixV2(p=1.0, use_correct_lambda=True))
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)
        out = aug(video)

        assert out.shape == video.shape
        assert torch.equal(aug(video, params=aug._params), out)

    @pytest.mark.parametrize("data_format", ["BCTHW", "BTCHW"])
    @pytest.mark.parametrize("num_mix", [1, 2])
    def test_cutmix_not_same_on_frame_with_same_on_batch(self, data_format, num_mix, device, dtype):
        # With same_on_frame=False the B * T frames are one image batch for CutMix, whose own generator applies
        # same_on_batch. One draw per frame index, repeated over the videos, raised IndexError for B > 1.
        batch_size, frame_num = 2, 3
        mix = K.RandomCutMixV2(num_mix=num_mix, p=1.0, same_on_batch=True, use_correct_lambda=True)
        aug = K.VideoSequential(mix, data_format=data_format, same_on_frame=False)
        video = torch.rand(batch_size, frame_num, 3, 8, 8, device=device, dtype=dtype)

        torch.manual_seed(0)
        out = aug(video.transpose(1, 2).contiguous() if data_format == "BCTHW" else video)
        out = out.transpose(1, 2) if data_format == "BCTHW" else out

        torch.manual_seed(0)
        params = mix.forward_parameters(torch.Size((batch_size * frame_num, 3, 8, 8)))
        expected = mix(video.reshape(-1, 3, 8, 8), params=params).reshape(video.shape)
        assert torch.equal(out, expected)

    def test_cutmix_same_on_frame_empty_video_batch(self, device, dtype):
        aug = K.VideoSequential(K.RandomCutMixV2(p=1.0, use_correct_lambda=True))
        video = torch.rand(0, 4, 3, 8, 8, device=device, dtype=dtype)

        assert aug(video).shape == video.shape
        assert aug(video, params=aug._params).shape == video.shape

    @pytest.mark.parametrize("same_on_frame", [False, True])
    def test_partial_gate_keypoints_round_trip_6375(self, same_on_frame, device, dtype):
        # Keypoints reach VideoSequential as VideoKeypoints, which index whole videos (#6375), while a partially
        # applied child gates frames. VideoSequential hands its children frame-level Keypoints, as it does for boxes
        # (#6173), so the inverse of a partial gate restores every frame's points.
        affine = K.RandomAffine(30.0, translate=(0.1, 0.1), p=0.5)
        aug = K.AugmentationSequential(
            K.VideoSequential(affine, same_on_frame=same_on_frame), data_keys=["input", "keypoints"]
        )
        video = torch.rand(3, 4, 3, 16, 16, device=device, dtype=dtype)
        points = torch.rand(3, 4, 5, 2, device=device, dtype=dtype) * 15
        # A forward and inverse warp rounds 15 px coordinates to about the dtype's spacing there (0.125 in bfloat16).
        tol = {torch.float16: 1e-2, torch.bfloat16: 0.125}.get(dtype, 1e-4)
        partial = False
        for seed in range(4):
            torch.manual_seed(seed)
            out_video, out_points = aug(video, points)
            applied = aug._params[0].data[0].data["batch_prob"] > 0.5
            partial |= bool(applied.any()) and not bool(applied.all())
            # The gate is per frame: a skipped frame's points are unchanged and an applied frame's points move.
            moved = (out_points - points).abs().flatten(0, 1).amax((-1, -2)) > tol
            assert_close(moved, applied.to(moved.device))  # the gate is drawn on the CPU
            _, restored = aug.inverse(out_video, out_points)
            assert out_points.shape == points.shape
            assert_close(restored, points, atol=tol, rtol=0.0)
        assert partial

    def test_video_keypoints_object_input_6375(self, device, dtype):
        # A VideoKeypoints object goes through a VideoSequential pipeline and comes back as VideoKeypoints, equal to
        # the output for the same points as a (B, T, N, 2) tensor. It used to raise a TypeError in is_floating_point.
        # Plain Keypoints carry no frame count, so they are rejected with a message that says so.
        from kornia.geometry.keypoints import Keypoints, VideoKeypoints

        aug = K.AugmentationSequential(
            K.VideoSequential(K.RandomAffine(30.0, p=0.5), K.RandomHorizontalFlip(p=0.5)),
            data_keys=["input", "keypoints"],
        )
        video = torch.rand(3, 4, 3, 16, 16, device=device, dtype=dtype)
        points = torch.rand(3, 4, 5, 2, device=device, dtype=dtype) * 15
        torch.manual_seed(0)
        out_video, dense = aug(video, points)
        torch.manual_seed(0)
        _, wrapped = aug(video, VideoKeypoints.from_tensor(points.clone()))
        assert isinstance(wrapped, VideoKeypoints)
        assert_close(wrapped.to_tensor(), dense)
        _, restored = aug.inverse(out_video, wrapped)
        assert isinstance(restored, VideoKeypoints)
        assert_close(restored.to_tensor(), aug.inverse(out_video, dense)[1])

        with pytest.raises(TypeError, match="VideoKeypoints or a"):
            aug(video, Keypoints(points.flatten(0, 1)))
