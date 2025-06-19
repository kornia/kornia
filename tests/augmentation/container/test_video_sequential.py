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
        with pytest.raises(AssertionError):
            img = torch.randn(*shape, device=device, dtype=dtype)
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

    @pytest.mark.jit()
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
    class TestVideoSequential:
        def test_same_on_frame_true(self, data_format, device, dtype):
            B, C, T, H, W = 1, 3, 4, 5, 5
            if data_format == "BCTHW":
                input = torch.randn(B, C, 1, H, W, device=device, dtype=dtype).repeat(1, 1, T, 1, 1)
            else:
                input = torch.randn(B, 1, C, H, W, device=device, dtype=dtype).repeat(1, T, 1, 1, 1)

            torch.manual_seed(42)
            aug = K.VideoSequential(
                K.ColorJiggle(0.5, 0.5, 0.5, 0.5, p=1.0),
                data_format=data_format,
                same_on_frame=True,
            )
            out = aug(input)

            if data_format == "BCTHW":
                for t in range(1, T):
                    assert_close(out[:, :, t], out[:, :, 0])
            else:
                for t in range(1, T):
                    assert_close(out[:, t], out[:, 0])

        def test_same_on_frame_false(self, data_format, device, dtype):
            B, C, T, H, W = 1, 3, 4, 5, 5
            if data_format == "BCTHW":
                input = torch.randn(B, C, 1, H, W, device=device, dtype=dtype).repeat(1, 1, T, 1, 1)
            else:
                input = torch.randn(B, 1, C, H, W, device=device, dtype=dtype).repeat(1, T, 1, 1, 1)

            torch.manual_seed(42)
            aug = K.VideoSequential(
                K.ColorJiggle(0.5, 0.5, 0.5, 0.5, p=1.0),
                data_format=data_format,
                same_on_frame=False,
            )
            out = aug(input)

            if data_format == "BCTHW":
                has_diff = any(not torch.allclose(out[:, :, t], out[:, :, 0], atol=1e-5) for t in range(1, T))
            else:
<<<<<<< HEAD
                has_diff = any(not torch.allclose(out[:, t], out[:, 0], atol=1e-5) for t in range(1, T))
=======
                has_diff = any(
                    not torch.allclose(out[:, t], out[:, 0], atol=1e-5)
                    for t in range(1, T)
                ) 
                if not has_diff: 
                    for t in range(T): 
                        print(f"Frame {t} mean: {out [:, :, t].mean().item()}")
>>>>>>> 8967bbac (Implement per-frame param repetition for VideoSequential; add repeat_param_item_nested_list helper and test for same_on_frame=False)

            assert has_diff, "Expected different frames with same_on_frame=False"
