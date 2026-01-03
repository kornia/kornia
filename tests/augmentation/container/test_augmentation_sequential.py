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

from functools import partial
from unittest.mock import patch

import pytest
import torch

import kornia
import kornia.augmentation as K
from kornia.augmentation.container.base import ParamItem
from kornia.constants import BorderType
from kornia.geometry.bbox import bbox_to_mask

from testing.augmentation.utils import reproducibility_test
from testing.base import assert_close


class TestAugmentationSequential:
    @pytest.mark.parametrize(
        "data_keys", ["input", "image", ["mask", "input"], ["input", "bbox_yxyx"], [0, 10], [BorderType.REFLECT]]
    )
    @pytest.mark.parametrize("augmentation_list", [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)])
    def test_exception(self, augmentation_list, data_keys, device, dtype):
        with pytest.raises(Exception):  # AssertError and NotImplementedError
            K.AugmentationSequential(augmentation_list, data_keys=data_keys)

    @pytest.mark.slow
    @pytest.mark.parametrize("same_on_batch", [True, False])
    @pytest.mark.parametrize("random_apply", [1, (2, 2), (1, 2), (2,), 10, True, False])
    @pytest.mark.parametrize("inp", [torch.randn(1, 3, 1000, 500), torch.randn(3, 1000, 500)])
    def test_mixup(self, inp, random_apply, same_on_batch, device, dtype):
        inp = torch.as_tensor(inp, device=device, dtype=dtype)
        aug = K.AugmentationSequential(
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, p=1.0),
            K.RandomMixUpV2(p=1.0),
            data_keys=["input"],
            random_apply=random_apply,
            same_on_batch=same_on_batch,
        )
        out = aug(inp)
        assert out.shape[-3:] == inp.shape[-3:]
        reproducibility_test(inp, aug)

    def test_mixup_cutmix_only(self, device, dtype):

        mixup = K.RandomMixUpV2(p=1.0, data_keys=["input"])
        cutmix = K.RandomCutMixV2(p=1.0, data_keys=["input"])
        aug = K.AugmentationSequential(
            mixup,
            cutmix,
            data_keys=["input"],
            random_apply=1,
        )

        input = torch.randn(2, 3, 224, 224, device=device, dtype=dtype)


        out_input = aug(input)

        assert out_input.shape == input.shape


    def test_video(self, device, dtype):
        input = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)[None]
        bbox = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype).expand(
            2, 1, -1, -1
        )[None]
        points = torch.tensor([[[1.0, 1.0]]], device=device, dtype=dtype).expand(2, -1, -1)[None]
        aug_list = K.AugmentationSequential(
            K.VideoSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), kornia.augmentation.RandomAffine(360, p=1.0)
            ),
            data_keys=["input", "mask", "bbox", "keypoints"],
        )
        out = aug_list(input, input, bbox, points)
        assert out[0].shape == input.shape
        assert out[1].shape == input.shape
        assert out[2].shape == bbox.shape
        assert out[3].shape == points.shape

        out_inv = aug_list.inverse(*out)
        assert out_inv[0].shape == input.shape
        assert out_inv[1].shape == input.shape
        assert out_inv[2].shape == bbox.shape
        assert out_inv[3].shape == points.shape

    def test_3d_augmentations(self, device, dtype):
        input = torch.randn(2, 2, 3, 5, 6, device=device, dtype=dtype)
        aug_list = K.AugmentationSequential(
            K.RandomAffine3D(360.0, p=1.0), K.RandomHorizontalFlip3D(p=1.0), data_keys=["input"]
        )
        out = aug_list(input)
        assert out.shape == input.shape

    def test_random_flips(self, device, dtype):
        inp = torch.randn(1, 3, 510, 1020, device=device, dtype=dtype)
        bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)

        expected_bbox_vertical_flip = torch.tensor(
            [[[355, 259], [660, 259], [660, 499], [355, 499]]], device=device, dtype=dtype
        )
        expected_bbox_horizontal_flip = torch.tensor(
            [[[359, 10], [664, 10], [664, 250], [359, 250]]], device=device, dtype=dtype
        )

        aug_ver = K.AugmentationSequential(
            K.RandomVerticalFlip(p=1.0), data_keys=["input", "bbox"], same_on_batch=False
        )

        aug_hor = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=1.0), data_keys=["image", "bbox"], same_on_batch=False
        )

        out_ver = aug_ver(inp.clone(), bbox.clone())
        out_hor = aug_hor(inp.clone(), bbox.clone())

        assert_close(out_ver[1], expected_bbox_vertical_flip)
        assert_close(out_hor[1], expected_bbox_horizontal_flip)

    def test_with_mosaic(self, device, dtype):
        width, height = 100, 100
        crop_width, crop_height = 3, 3
        input = torch.randn(3, 3, width, height, device=device, dtype=dtype)
        bbox = torch.tensor(
            [[[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0]]], device=device, dtype=dtype
        ).expand(3, -1, -1)
        aug = K.AugmentationSequential(
            K.RandomCrop((crop_width, crop_height), padding=1, cropping_mode="resample", fill=0),
            K.RandomHorizontalFlip(p=1.0),
            K.RandomMosaic(p=1.0),
            data_keys=["input", "bbox_xyxy"],
        )

        reproducibility_test((input, bbox), aug)

    def test_random_crops_and_flips(self, device, dtype):
        width, height = 100, 100
        crop_width, crop_height = 3, 3
        input = torch.randn(3, 3, width, height, device=device, dtype=dtype)
        bbox = torch.tensor(
            [[[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0]]], device=device, dtype=dtype
        ).expand(3, -1, -1)
        aug = K.AugmentationSequential(
            K.RandomCrop((crop_width, crop_height), padding=1, cropping_mode="resample", fill=0),
            K.RandomHorizontalFlip(p=1.0),
            data_keys=["input", "bbox_xyxy"],
        )

        reproducibility_test((input, bbox), aug)

        _params = aug.forward_parameters(input.shape)
        # specifying the crop locations allows us to compute by hand the expected outputs
        crop_locations = torch.tensor(
            [[1.0, 2.0], [1.0, 1.0], [2.0, 0.0]],
            device=_params[0].data["src"].device,
            dtype=_params[0].data["src"].dtype,
        )
        crops = crop_locations.expand(4, -1, -1).permute(1, 0, 2).clone()
        crops[:, 1:3, 0] += crop_width - 1
        crops[:, 2:4, 1] += crop_height - 1
        _params[0].data["src"] = crops

        # expected output bboxes after crop for specified crop locations and crop size (3,3)
        expected_out_bbox = torch.tensor(
            [
                [[1.0, 0.0, 2.0, 1.0], [0.0, -1.0, 1.0, 1.0], [0.0, -1.0, 2.0, 0.0]],
                [[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0]],
                [[0.0, 2.0, 1.0, 3.0], [-1.0, 1.0, 0.0, 3.0], [-1.0, 1.0, 1.0, 2.0]],
            ],
            device=device,
            dtype=dtype,
        )
        # horizontally flip boxes based on crop width
        xmins = expected_out_bbox[..., 0].clone()
        xmaxs = expected_out_bbox[..., 2].clone()
        expected_out_bbox[..., 0] = crop_width - xmaxs - 1
        expected_out_bbox[..., 2] = crop_width - xmins - 1

        out = aug(input, bbox, params=_params)
        assert out[1].shape == bbox.shape
        assert_close(out[1], expected_out_bbox, atol=1e-4, rtol=1e-4)

        out_inv = aug.inverse(*out)
        assert out_inv[1].shape == bbox.shape
        assert_close(out_inv[1], bbox, atol=1e-4, rtol=1e-4)

    def test_random_erasing(self, device, dtype):
        fill_value = 0.5
        input = torch.randn(3, 3, 100, 100, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomErasing(p=1.0, value=fill_value), data_keys=["image", "mask"])

        reproducibility_test((input, input), aug)

        out = aug(input, input)
        assert torch.all(out[1][out[0] == fill_value] == 0.0)

    def test_resize(self, device, dtype):
        size = 50
        input = torch.randn(3, 3, 100, 100, device=device, dtype=dtype)
        mask = torch.randn(3, 1, 100, 100, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.Resize((size, size), p=1.0), data_keys=["input", "mask"])

        reproducibility_test((input, mask), aug)

        out = aug(input, mask)
        assert out[0].shape == (3, 3, size, size)
        assert out[1].shape == (3, 1, size, size)

    def test_random_crops(self, device, dtype):
        # Test with relaxed tolerance for platform-specific numerical precision
        torch.manual_seed(233)
        input = torch.randn(3, 3, 3, 3, device=device, dtype=dtype)
        bbox = torch.tensor(
            [[[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0]]], device=device, dtype=dtype
        ).expand(3, -1, -1)
        points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], device=device, dtype=dtype).expand(3, -1, -1)
        aug = K.AugmentationSequential(
            K.RandomCrop((3, 3), padding=1, cropping_mode="resample", fill=0),
            K.RandomAffine((360.0, 360.0), p=1.0),
            data_keys=["input", "mask", "bbox_xyxy", "keypoints"],
            extra_args={},
        )

        reproducibility_test((input, input, bbox, points), aug)

        _params = aug.forward_parameters(input.shape)
        # specifying the crops allows us to compute by hand the expected outputs
        _params[0].data["src"] = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
                [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
                [[2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0]],
            ],
            device=_params[0].data["src"].device,
            dtype=_params[0].data["src"].dtype,
        )

        expected_out_bbox = torch.tensor(
            [
                [[1.0, 0.0, 2.0, 1.0], [0.0, -1.0, 1.0, 1.0], [0.0, -1.0, 2.0, 0.0]],
                [[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0]],
                [[0.0, 2.0, 1.0, 3.0], [-1.0, 1.0, 0.0, 3.0], [-1.0, 1.0, 1.0, 2.0]],
            ],
            device=device,
            dtype=dtype,
        )
        expected_out_points = torch.tensor(
            [[[0.0, -1.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[-1.0, 1.0], [0.0, 2.0]]], device=device, dtype=dtype
        )

        out = aug(input, input, bbox, points, params=_params)
        assert out[0].shape == (3, 3, 3, 3)
        assert_close(out[0], out[1], atol=1e-4, rtol=1e-4)
        assert out[2].shape == bbox.shape
        assert_close(out[2], expected_out_bbox, atol=1e-3, rtol=1e-3)
        assert out[3].shape == points.shape
        assert_close(out[3], expected_out_points, atol=1e-4, rtol=1e-4)

        out_inv = aug.inverse(*out)
        assert out_inv[0].shape == input.shape
        assert_close(out_inv[0], out_inv[1], atol=1e-4, rtol=1e-4)
        assert out_inv[2].shape == bbox.shape
        assert_close(out_inv[2], bbox, atol=1e-3, rtol=1e-3)
        assert out_inv[3].shape == points.shape
        assert_close(out_inv[3], points, atol=1e-4, rtol=1e-4)

    def test_random_resized_crop(self, device, dtype):
        size = 50
        input = torch.randn(3, 3, 100, 100, device=device, dtype=dtype)
        mask = torch.randn(3, 1, 100, 100, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomResizedCrop((size, size), p=1.0), data_keys=["input", "mask"])

        reproducibility_test((input, mask), aug)

        out = aug(input, mask)
        assert out[0].shape == (3, 3, size, size)
        assert out[1].shape == (3, 1, size, size)

    @pytest.mark.parametrize(
        "bbox",
        [
            [
                torch.tensor([[1, 5, 2, 7], [0, 3, 9, 9]]),
                torch.tensor([[1, 5, 2, 7], [0, 3, 9, 9], [0, 5, 8, 7]]),
                torch.empty((0, 4)),
            ],
            torch.empty((3, 0, 4)),
            torch.tensor([[[1, 5, 2, 7], [0, 3, 9, 9]], [[1, 5, 2, 7], [0, 3, 9, 9]], [[0, 5, 8, 7], [0, 2, 5, 5]]]),
        ],
    )
    @pytest.mark.parametrize(
        "augmentation", [K.RandomCrop((30, 30), padding=1, cropping_mode="resample", fill=0), K.Resize((30, 30))]
    )
    def test_bbox(self, bbox, augmentation, device, dtype):
        img = torch.rand((3, 3, 10, 10), device=device, dtype=dtype)
        if isinstance(bbox, list):
            for i, b in enumerate(bbox):
                bbox[i] = b.to(device=device, dtype=dtype)
        else:
            bbox = bbox.to(device=device, dtype=dtype)

        inputs = [img, bbox]

        aug = K.AugmentationSequential(augmentation, data_keys=["input", "bbox_xyxy"])

        transformed = aug(*inputs)

        assert len(transformed) == len(inputs)
        bboxes_transformed = transformed[-1]
        assert len(bboxes_transformed) == len(bbox)
        assert bboxes_transformed.__class__ == bbox.__class__
        for i in range(len(bbox)):
            assert len(bboxes_transformed[i]) == len(bbox[i])

    def test_class(self, device, dtype):
        img = torch.zeros((5, 1, 5, 5))
        labels = torch.randint(0, 10, size=(5, 1))
        aug = K.AugmentationSequential(K.RandomCrop((3, 3), pad_if_needed=True), data_keys=["input", "class"])

        _, out_labels = aug(img, labels)
        assert labels is out_labels

    @pytest.mark.slow
    @pytest.mark.parametrize("random_apply", [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_forward_and_inverse(self, random_apply, device, dtype):
        inp = torch.randn(1, 3, 1000, 500, device=device, dtype=dtype)
        bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
        )[:, None]
        aug = K.AugmentationSequential(
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.AugmentationSequential(
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                K.RandomAffine(360, p=1.0),
                K.RandomAffine(360, p=1.0),
                data_keys=["input", "mask", "bbox", "keypoints"],
            ),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, p=1.0),
            data_keys=["input", "mask", "bbox", "keypoints"],
            random_apply=random_apply,
        )
        out = aug(inp, mask, bbox, keypoints)
        assert out[0].shape == inp.shape
        assert out[1].shape == mask.shape
        assert out[2].shape == bbox.shape
        assert out[3].shape == keypoints.shape
        assert set(out[1].unique().tolist()).issubset(set(mask.unique().tolist()))

        out_inv = aug.inverse(*out)
        assert out_inv[0].shape == inp.shape
        assert out_inv[1].shape == mask.shape
        assert out_inv[2].shape == bbox.shape
        assert out_inv[3].shape == keypoints.shape
        assert set(out_inv[1].unique().tolist()).issubset(set(mask.unique().tolist()))

        if random_apply is False:
            reproducibility_test((inp, mask, bbox, keypoints), aug)

    @pytest.mark.slow
    def test_individual_forward_and_inverse(self, device, dtype):
        inp = torch.randn(1, 3, 1000, 500, device=device, dtype=dtype)
        bbox = torch.tensor([[[[355, 10], [660, 10], [660, 250], [355, 250]]]], device=device, dtype=dtype)
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 500, 1000
        )[:, None]
        crop_size = (200, 200)

        aug = K.AugmentationSequential(
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.AugmentationSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.RandomAffine(360, p=1.0),
            K.RandomCrop(crop_size, padding=1, cropping_mode="resample", fill=0),
            data_keys=["input", "mask", "bbox", "keypoints"],
            extra_args={},
        )
        # NOTE: Mask data with nearest not passing reproducibility check under float64.
        reproducibility_test((inp, mask, bbox, keypoints), aug)

        out = aug(inp, mask, bbox, keypoints)
        assert out[0].shape == (*inp.shape[:2], *crop_size)
        assert out[1].shape == (*mask.shape[:2], *crop_size)
        assert out[2].shape == bbox.shape
        assert out[3].shape == keypoints.shape

        out_inv = aug.inverse(*out)
        assert out_inv[0].shape == inp.shape
        assert out_inv[1].shape == mask.shape
        assert out_inv[2].shape == bbox.shape
        assert out_inv[3].shape == keypoints.shape

        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0))
        assert aug(inp, data_keys=["input"]).shape == inp.shape
        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0))
        assert aug(inp, data_keys=["input"]).shape == inp.shape
        assert aug(mask, data_keys=["mask"], params=aug._params).shape == mask.shape

        assert aug.inverse(inp, data_keys=["input"]).shape == inp.shape
        assert aug.inverse(bbox, data_keys=["bbox"]).shape == bbox.shape
        assert aug.inverse(keypoints, data_keys=["keypoints"]).shape == keypoints.shape
        assert aug.inverse(mask, data_keys=["mask"]).shape == mask.shape

    @pytest.mark.slow
    @pytest.mark.parametrize("random_apply", [2, (1, 1), (2,), 10, True, False])
    def test_forward_and_inverse_return_transform(self, random_apply, device, dtype):
        inp = torch.randn(1, 3, 1000, 500, device=device, dtype=dtype)
        bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
        )[:, None]
        aug = K.AugmentationSequential(
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.AugmentationSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, p=1.0),
            data_keys=["input", "mask", "bbox", "keypoints"],
            random_apply=random_apply,
            extra_args={},
        )
        out = aug(inp, mask, bbox, keypoints)
        assert out[0].shape == inp.shape
        assert out[1].shape == mask.shape
        assert out[2].shape == bbox.shape
        assert out[3].shape == keypoints.shape

        reproducibility_test((inp, mask, bbox, keypoints), aug)

        out_inv = aug.inverse(*out)
        assert out_inv[0].shape == inp.shape
        assert out_inv[1].shape == mask.shape
        assert out_inv[2].shape == bbox.shape
        assert out_inv[3].shape == keypoints.shape

    @pytest.mark.slow
    @pytest.mark.parametrize("random_apply", [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_inverse_and_forward_return_transform(self, random_apply, device, dtype):
        inp = torch.randn(1, 3, 1000, 500, device=device, dtype=dtype)
        bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)
        bbox_2 = [
            # torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype),
            torch.tensor(
                [[[355, 10], [660, 10], [660, 250], [355, 250]], [[355, 10], [660, 10], [660, 250], [355, 250]]],
                device=device,
                dtype=dtype,
            )
        ]
        bbox_wh = torch.tensor([[[30, 40, 100, 100]]], device=device, dtype=dtype)
        bbox_wh_2 = [
            # torch.tensor([[30, 40, 100, 100]], device=device, dtype=dtype),
            torch.tensor([[30, 40, 100, 100], [30, 40, 100, 100]], device=device, dtype=dtype)
        ]
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
        )[:, None]
        aug = K.AugmentationSequential(
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.AugmentationSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, p=1.0),
            data_keys=["input", "mask", "bbox", "keypoints", "bbox", "BBOX_XYWH", "BBOX_XYWH"],
            random_apply=random_apply,
        )
        with pytest.raises(Exception):  # No parameters available for inversing.
            aug.inverse(inp, mask, bbox, keypoints, bbox_2, bbox_wh, bbox_wh_2)

        out = aug(inp, mask, bbox, keypoints, bbox_2, bbox_wh, bbox_wh_2)
        assert out[0].shape == inp.shape
        assert out[1].shape == mask.shape
        assert out[2].shape == bbox.shape
        assert out[3].shape == keypoints.shape

        if random_apply is False:
            reproducibility_test((inp, mask, bbox, keypoints, bbox_2, bbox_wh, bbox_wh_2), aug)

    @pytest.mark.jit()
    @pytest.mark.skip(reason="turn off due to Union Type")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = K.AugmentationSequential(
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0), same_on_batch=True
        )
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.parametrize("batch_prob", [[True, True], [False, True], [False, False]])
    @pytest.mark.parametrize("box", ["bbox", "bbox_xyxy", "bbox_xywh"])
    def test_autocast(self, batch_prob, box, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        def mock_forward_parameters_sequential(batch_shape, cls, batch_prob):
            named_modules = cls.get_forward_sequence()
            params = []
            for name, module in named_modules:
                if isinstance(module, (K.base._AugmentationBase, K.MixAugmentationBaseV2, K.ImageSequential)):
                    with patch.object(module, "__batch_prob_generator__", return_value=batch_prob):
                        mod_param = module.forward_parameters(batch_shape)

                    param = ParamItem(name, mod_param)
                else:
                    param = ParamItem(name, None)
                batch_shape = K.container.image._get_new_batch_shape(param, batch_shape)
                params.append(param)
            return params

        tfs = (K.RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0), K.RandomGaussianBlur((3, 3), (0.1, 3), p=1))
        data_keys = ["input", "mask", box, "keypoints"]
        aug = K.AugmentationSequential(*tfs, data_keys=data_keys, random_apply=True)
        bs = len(batch_prob)
        imgs = torch.rand(bs, 3, 7, 4, dtype=dtype, device=device)
        if box == "bbox":
            bb = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], dtype=dtype, device=device).expand(
                bs, 1, -1, -1
            )
        else:
            bb = torch.rand(bs, 1, 4, dtype=dtype, device=device)

        msk = torch.zeros_like(imgs)
        msk[..., 3:, 2] = 1.0
        points = torch.rand(bs, 1, 2, dtype=dtype, device=device)

        to_apply = torch.tensor(batch_prob, device=device)

        fwd_params = partial(mock_forward_parameters_sequential, cls=aug, batch_prob=to_apply)
        with patch.object(aug, "forward_parameters", fwd_params):
            params = aug.forward_parameters(imgs.shape)

        with torch.autocast(device.type):
            outputs = aug(imgs, msk, bb, points, params=params)

        assert outputs[0].dtype == dtype, "Output image dtype should match the input dtype"
        assert outputs[1].dtype == dtype, "Output mask dtype should match the input dtype"
        assert outputs[2].dtype == dtype, "Output box dtype should match the input dtype"
        assert outputs[3].dtype == dtype, "Output keypoints dtype should match the input dtype"
