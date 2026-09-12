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
from kornia.constants import BorderType, DataKey, Resample
from kornia.core._compat import torch_version_lt
from kornia.geometry.bbox import bbox_to_mask
from kornia.geometry.boxes import Boxes

from testing.augmentation.utils import reproducibility_test
from testing.base import BaseTester, assert_close


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

    @pytest.mark.parametrize("image_dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
    def test_mixed_image_bbox_dtypes(self, device, image_dtype):
        # Regression test for https://github.com/kornia/kornia/issues/3705 and #3706:
        # bbox stays in fp32 while the image uses a half/double compute dtype.
        if device.type == "mps" and image_dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        if device.type == "cpu" and image_dtype in (torch.float16, torch.bfloat16) and torch_version_lt(2, 6, 0):
            pytest.skip("PyTorch <2.6 has no CPU Half/BFloat16 grid_sample kernel")
        torch.manual_seed(0)
        img = torch.rand(2, 3, 32, 32, device=device, dtype=image_dtype)
        bb = torch.tensor(
            [[[[4.0, 4.0], [12.0, 4.0], [12.0, 12.0], [4.0, 12.0]]]] * 2,
            device=device,
            dtype=torch.float32,
        )
        aug = K.AugmentationSequential(K.RandomAffine(degrees=10, p=1.0), data_keys=["image", "bbox"])
        out_img, out_bb = aug(img, bb)
        assert out_img.dtype == image_dtype
        assert out_bb.dtype == torch.float32
        assert out_img.shape == img.shape
        assert out_bb.shape == bb.shape

    def test_random_flips(self, device, dtype):
        inp = torch.randn(1, 3, 255, 510, device=device, dtype=dtype)
        bbox = torch.tensor([[[177, 5], [330, 5], [330, 125], [177, 125]]], device=device, dtype=dtype)

        expected_bbox_vertical_flip = torch.tensor(
            [[[177, 129], [330, 129], [330, 249], [177, 249]]], device=device, dtype=dtype
        )
        expected_bbox_horizontal_flip = torch.tensor(
            [[[179, 5], [332, 5], [332, 125], [179, 125]]], device=device, dtype=dtype
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

    @pytest.mark.parametrize("num_boxes", [1, 2])
    def test_convention_padded_random_crop_accepts_rank2_bboxes_4244(self, num_boxes, device, dtype):
        # kornia#4244: RandomCrop's padded path routes bounding boxes through Boxes.pad and
        # Boxes.unpad. A rank-2 (N, 4) bbox for a single image builds an *unbatched* Boxes
        # container, which those methods crashed on with "output with shape [1, 4] doesn't match
        # the broadcast shape [1, 1, 4]". This is the reproducer from the issue's follow-up, and it
        # is the only public route to that container: a rank-3 (B, N, 4) bbox builds a batched one,
        # which never crashed. Both box counts are covered because clamp failed differently at N=1
        # (IndexError) and N=2 (RuntimeError) on the unbatched path.
        input = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)
        bbox = torch.tensor([[1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 5.0, 6.0]], device=device, dtype=dtype)[:num_boxes]
        aug = K.AugmentationSequential(K.RandomCrop((6, 6), padding=1, p=1.0), data_keys=["input", "bbox_xyxy"])

        out_input, out_bbox = aug(input, bbox)

        assert out_input.shape == (1, 3, 6, 6)
        # the rank of the caller's bbox is preserved on the way out
        assert out_bbox.shape == (num_boxes, 4)
        assert torch.isfinite(out_bbox).all()

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_padded_random_crop_batched_bboxes(self, batch_size, device, dtype):
        # The batched (B, N, 4) companion of the pin above. It builds a batched Boxes container,
        # which always worked, so it passes on either side of the fix and is kept only as a guard
        # that the rank-2 repair did not disturb the batched route.
        input = torch.rand(batch_size, 3, 8, 8, device=device, dtype=dtype)
        bbox = torch.tensor([[[1.0, 1.0, 4.0, 4.0]]], device=device, dtype=dtype).expand(batch_size, -1, -1)
        aug = K.AugmentationSequential(K.RandomCrop((6, 6), padding=1, p=1.0), data_keys=["input", "bbox_xyxy"])

        out_input, out_bbox = aug(input, bbox)

        assert out_input.shape == (batch_size, 3, 6, 6)
        assert out_bbox.shape == (batch_size, 1, 4)
        assert torch.isfinite(out_bbox).all()

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


@pytest.mark.usefixtures("restore_torch_rng")
class TestConventionAugmentationSequential(BaseTester):
    """Pins for the `AugmentationSequential` contract (batch-6 conventions, anchor 2).

    Every literal was generated by the body of the pin that carries it, executed on this worktree on
    2026-09-11 with `.venv/bin/python` (torch 2.14.0, python 3.11, cpu, float32). Fixtures are asymmetric
    (H != W, a hot pixel off both centre lines, a distinct value per box corner) and every draw is made
    deterministic with `p=1.0` plus a point range, or seeded in the test.
    """

    def test_convention_flip_is_integer_centre_inclusive_for_every_data_key(self, device, dtype):
        # Convention pin: a horizontal flip maps column x to W - 1 - x, and a vertical flip row y to H - 1 - y,
        # for the image, the mask, keypoints and boxes alike - inclusive pixel coordinates about the integer
        # centre, the same rule `kornia.geometry.transform.hflip` / `vflip` follow.
        # H = 3, W = 4 (asymmetric); hot pixel (row 0, col 1), off both centre lines; keypoint (x=1, y=2).
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu). hflip:
        # pixel -> (row 0, col 2), keypoint -> (2, 2), bbox_xyxy [0,0,1,1] -> [2, 0, 3, 1], bbox vertices ->
        # [[2,0],[3,0],[3,1],[2,1]]. vflip: pixel -> (row 2, col 1), keypoint -> (1, 0),
        # bbox_xyxy -> [0, 1, 1, 2]. An exclusive reading (x' = W - x) would give keypoint (3, 2) and
        # box [3, 0, 4, 1].
        img = torch.zeros(1, 1, 3, 4, device=device, dtype=dtype)
        img[0, 0, 0, 1] = 1.0
        kpts = torch.tensor([[[1.0, 2.0]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], device=device, dtype=dtype)
        vertices = torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]], device=device, dtype=dtype)
        keys = ["input", "mask", "keypoints", "bbox_xyxy", "bbox"]

        aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=keys)
        out_img, out_mask, out_kpts, out_boxes, out_vertices = aug(img, img.clone(), kpts, boxes, vertices)
        expected_img = torch.zeros(1, 1, 3, 4, device=device, dtype=dtype)
        expected_img[0, 0, 0, 2] = 1.0
        self.assert_close(out_img, expected_img)
        self.assert_close(out_mask, out_img)
        self.assert_close(out_kpts, torch.tensor([[[2.0, 2.0]]], device=device, dtype=dtype))
        self.assert_close(out_boxes, torch.tensor([[[2.0, 0.0, 3.0, 1.0]]], device=device, dtype=dtype))
        self.assert_close(
            out_vertices,
            torch.tensor([[[[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]]], device=device, dtype=dtype),
        )

        aug = K.AugmentationSequential(K.RandomVerticalFlip(p=1.0), data_keys=keys[:-1])
        out_img, out_mask, out_kpts, out_boxes = aug(img, img.clone(), kpts, boxes)
        expected_img = torch.zeros(1, 1, 3, 4, device=device, dtype=dtype)
        expected_img[0, 0, 2, 1] = 1.0
        self.assert_close(out_img, expected_img)
        self.assert_close(out_mask, out_img)
        self.assert_close(out_kpts, torch.tensor([[[1.0, 0.0]]], device=device, dtype=dtype))
        self.assert_close(out_boxes, torch.tensor([[[0.0, 1.0, 1.0, 2.0]]], device=device, dtype=dtype))

    def test_convention_data_key_shapes_are_validated(self, device, dtype):
        # Convention pin: each data key has one accepted layout - `bbox` is (B, N, 4, 2) vertices, `bbox_xyxy`
        # is (B, N, 4) corners, `keypoints` is (B, N, 2) - and feeding one layout under the other key raises
        # `ValueError` naming the expected shape and the box mode.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu):
        # "Boxes shape must be (N, 4, 2) or (B, N, 4, 2) when vertices_plus mode. Got torch.Size([1, 1, 4])."
        # "Boxes shape must be (N, 4) or (B, N, 4) when xyxy_plus mode. Got torch.Size([1, 1, 4, 2])."
        # "Keypoints shape must be (N, 2) or (B, N, 2). Got torch.Size([1, 2, 3])."
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        cases = [
            ("bbox", torch.rand(1, 1, 4, device=device, dtype=dtype), "must be .N, 4, 2."),
            ("bbox_xyxy", torch.rand(1, 1, 4, 2, device=device, dtype=dtype), "must be .N, 4."),
            ("keypoints", torch.rand(1, 2, 3, device=device, dtype=dtype), "Keypoints shape must be"),
        ]
        for key, wrong, message in cases:
            aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", key])
            with pytest.raises(ValueError, match=message):
                aug(img, wrong)
        # the accepted layouts pass, including the degenerate N = 0 case
        for key, shape in [("bbox", (2, 0, 4, 2)), ("bbox_xyxy", (2, 0, 4)), ("keypoints", (2, 0, 2))]:
            aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", key])
            empty = torch.rand(*shape, device=device, dtype=dtype)
            out = aug(torch.rand(2, 1, 3, 4, device=device, dtype=dtype), empty)
            assert out[1].shape == shape

    def test_convention_masks_keep_labels_and_add_padding_fill_through_a_rotation(self, device, dtype):
        # Convention pin: masks are resampled with nearest interpolation, so a {2, 3} mask still holds those
        # labels after a 45 degree affine without intermediate values. The zero padding fill is also present,
        # and the mask dtype is preserved, `bool` included.
        # The claim is checked one parameter away from the pin's fixture: B = 2 and B = 1 both hold.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): value set
        # [0.0, 2.0, 3.0], dtype torch.float32 preserved, bool mask stays torch.bool, max|mask - input mask|
        # 2.0 (the rotation really moved it).
        for batch in (2, 1):
            aug = K.AugmentationSequential(K.RandomAffine(degrees=(45.0, 45.0), p=1.0), data_keys=["input", "mask"])
            mask = torch.full((batch, 1, 6, 8), 2.0, device=device, dtype=dtype)
            mask[:, :, 1:4, 2:6] = 3.0
            out_mask = aug(torch.rand(batch, 3, 6, 8, device=device, dtype=dtype), mask)[1]
            assert sorted(out_mask.unique().tolist()) == [0.0, 2.0, 3.0]
            assert out_mask.dtype == dtype
            assert (out_mask - mask).abs().max().item() == 2.0  # the rotation moved the mask
        aug = K.AugmentationSequential(K.RandomAffine(degrees=(45.0, 45.0), p=1.0), data_keys=["input", "mask"])
        bool_mask = torch.zeros(2, 1, 6, 8, device=device, dtype=torch.bool)
        bool_mask[:, :, 1:4, 2:6] = True
        assert aug(torch.rand(2, 3, 6, 8, device=device, dtype=dtype), bool_mask)[1].dtype == torch.bool

    def test_convention_boxes_follow_the_xyxy_plus_convention(self, device, dtype):
        # Convention pin: the container's box arithmetic is `Boxes`' inclusive `xyxy_plus` mode, for a scaling
        # op as well as for a flip - `Boxes.from_tensor(..., mode="xyxy_plus").transform_boxes_(M)` reproduces
        # the container's output exactly, while `mode="xyxy"` (exclusive) does not.
        # Two asymmetric boxes on a 6x8 image, resized to (3, 4) => M = diag(4/8, 3/6).
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): container
        # [[0.0, 0.0, 1.2857143, 0.8]], [[0.85714293, 0.4, 2.1428573, 1.6]] == xyxy_plus;
        # xyxy gives [[0.0, 0.0, 1.8571429, 1.4]], [[0.85714293, 0.4, 2.7142859, 2.2]].
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the xyxy vs xyxy_plus gap is smaller than the half-precision box tolerance")
        boxes = torch.tensor([[[0.0, 0.0, 3.0, 2.0]], [[2.0, 1.0, 5.0, 4.0]]], device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.Resize((3, 4)), data_keys=["input", "bbox_xyxy"])
        out_boxes = aug(torch.rand(2, 3, 6, 8, device=device, dtype=dtype), boxes.clone())[1]
        matrix = aug.transform_matrix
        inclusive = Boxes.from_tensor(boxes.clone(), mode="xyxy_plus")
        inclusive.transform_boxes_(matrix)
        self.assert_close(out_boxes, inclusive.to_tensor(mode="xyxy_plus"))
        exclusive = Boxes.from_tensor(boxes.clone(), mode="xyxy")
        exclusive.transform_boxes_(matrix)
        assert not torch.allclose(out_boxes, exclusive.to_tensor(mode="xyxy"))

    def test_convention_inverse_restores_keypoints_and_loses_rotated_box_corners(self, device, dtype):
        # Convention pin: `.inverse()` restores keypoints moved by the geometric chain, while non-axis-aligned
        # rotations cannot restore tensor boxes. Forward turns both `bbox_xyxy` and vertex `bbox` formats into
        # axis-aligned enclosures, losing the original corners before inverse is called.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu): forward
        # max|move| 1.6213202476501465, inverse max|error| 7.152557373046875e-07.
        torch.manual_seed(0)
        aug = K.AugmentationSequential(K.RandomAffine(degrees=(45.0, 45.0), p=1.0), data_keys=["input", "keypoints"])
        img = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        kpts = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], device=device, dtype=dtype)
        out_img, out_kpts = aug(img, kpts)
        assert (out_kpts - kpts).abs().max().item() > 1.0  # the forward really moved them
        restored = aug.inverse(out_img, out_kpts)[1]
        self.assert_close(restored, kpts)
        boxes_by_key = {
            "bbox_xyxy": torch.tensor([[[1.0, 2.0, 4.0, 5.0]]], device=device, dtype=dtype),
            "bbox": torch.tensor([[[[1.0, 2.0], [4.0, 2.0], [4.0, 5.0], [1.0, 5.0]]]], device=device, dtype=dtype),
        }
        for key, boxes in boxes_by_key.items():
            box_aug = K.AugmentationSequential(K.RandomAffine(degrees=(45.0, 45.0), p=1.0), data_keys=["input", key])
            out_img, out_boxes = box_aug(torch.rand(1, 3, 8, 8, device=device, dtype=dtype), boxes)
            restored_boxes = box_aug.inverse(out_img, out_boxes)[1]
            assert not torch.allclose(restored_boxes, boxes)

    def test_wart_nested_non_rigid_silent_matrix_accumulation_is_order_sensitive(self, device, dtype):
        # Wart pin: `transformation_matrix_mode="silent"` only skips a direct non-rigid child. A nested
        # sequence containing `RandomElasticTransform` supplies `None` as its matrix: after a rigid child,
        # accessing the outer matrix raises TypeError, while putting the same nested sequence first returns
        # the rigid child's matrix. The image forward succeeds in both orders.
        # This records a current limitation; remove the pin when nested None-matrix handling is repaired.
        image = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)
        nested_non_rigid = lambda: K.AugmentationSequential(  # noqa: E731
            K.RandomElasticTransform(p=1.0), data_keys=["input"]
        )
        rigid_then_nested = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=1.0), nested_non_rigid(), data_keys=["input"]
        )
        assert rigid_then_nested(image).shape == image.shape
        with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for @: 'NoneType' and 'Tensor'"):
            _ = rigid_then_nested.transform_matrix

        nested_then_rigid = K.AugmentationSequential(
            nested_non_rigid(), K.RandomHorizontalFlip(p=1.0), data_keys=["input"]
        )
        assert nested_then_rigid(image).shape == image.shape
        self.assert_close(
            nested_then_rigid.transform_matrix,
            torch.tensor([[[-1.0, 0.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
        )

    def test_convention_same_on_batch_none_does_not_override_a_child(self, device, dtype):
        # Convention pin: `AugmentationSequential(same_on_batch=None)` - the default - keeps whatever each
        # child was built with, while `True` and `False` overwrite the child's own setting in both directions.
        # `keepdim` follows the same three-state rule.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu), seed 0, B = 4:
        # None over a same_on_batch=True child -> one distinct angle (36.5752067565918 four times);
        # False over the same child -> four distinct angles; True over a same_on_batch=False child -> one.
        # keepdim: None over a keepdim=True child -> (3, 6, 8), False -> (1, 3, 6, 8), True over False -> (3, 6, 8).
        def angles(container_flag, child_flag):
            torch.manual_seed(0)
            child = K.RandomAffine(degrees=(10.0, 90.0), p=1.0, same_on_batch=child_flag)
            seq = K.AugmentationSequential(child, data_keys=["input"], same_on_batch=container_flag)
            seq(torch.rand(4, 3, 6, 8, device=device, dtype=dtype))
            return child._params["angle"].unique().numel()

        assert angles(None, True) == 1
        assert angles(False, True) == 4
        assert angles(True, False) == 1
        assert angles(None, False) == 4

        def shape(container_flag, child_flag):
            seq = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=1.0, keepdim=child_flag), data_keys=["input"], keepdim=container_flag
            )
            return tuple(seq(torch.rand(3, 6, 8, device=device, dtype=dtype)).shape)

        assert shape(None, True) == (3, 6, 8)
        assert shape(False, True) == (1, 3, 6, 8)
        assert shape(True, False) == (3, 6, 8)

    def test_wart_extra_args_mask_resample_override_is_discarded_4419(self, device, dtype):
        # Wart pin (#4419): `extra_args[DataKey.MASK]` is documented as the way to control how masks are
        # handled, but on every class whose mask path is `RigidAffineAugmentationBase2D.apply_transform_mask`
        # the `resample` half is overwritten with NEAREST after the override was merged, so asking for
        # bilinear changes nothing. The `align_corners` half of the same dict IS honoured and reaches the
        # sampler, and `RandomElasticTransform`, which has its own mask path, honours both halves.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu), seed 0, a
        # (1, 1, 6, 8) mask with a 1-block: RandomAffine resample override max|delta| 0.0, RandomPerspective
        # align_corners override 1.0.
        # The elastic half uses its own fixture, reusing the #4420 pin's style below: `RandomElasticTransform
        # (alpha=(5.0, 5.0), sigma=(4.0, 4.0), p=1.0)` on a checkerboard mask, same (1, 1, 6, 8), H != W frame,
        # instead of the default alpha/sigma with a solid block. #4382 ("fix: respect align_corners in elastic
        # transform grid"), now in this branch's rebased base, shrank the displacement at the default fixture
        # until the bilinear-vs-nearest mask delta collapsed - 0.0093 (float32) / 0.0076 (float64) at seed 0,
        # both under the old > 0.1 threshold - and a solid block mask can also map onto itself under a small
        # warp regardless of alpha/sigma, so a checkerboard is used instead.
        # Sweep, seeds 0-9, this elastic fixture: min/max delta 0.461683/0.739558 on cpu float32, 0.292913/
        # 0.716883 on cpu float64, 0.462891/0.740234 on cpu float16, 0.472656/0.742188 on cpu bfloat16, and
        # 0.401424/0.749953 on mps float32 - never near the old default-fixture value at any seed or dtype
        # checked, so `> 0.1` stays a safe threshold. Seed-0 delta: 0.739558 (cpu float32), 0.588690 (cpu
        # float64), 0.740234 (cpu float16), 0.742188 (cpu bfloat16), 0.749953 (mps float32). The half-dtype
        # skip that used to guard this pin is dropped: affine and perspective are exact on float16/bfloat16
        # (0.0 and 1.0, matching float32/float64) and the elastic delta stays far above the threshold there
        # too.
        # The fix lands in the repair window and flips the first assertion; do not "correct" it here.

        def mask_of(aug_factory, extra):
            torch.manual_seed(0)
            aug = K.AugmentationSequential(aug_factory(), data_keys=["input", "mask"], extra_args=extra)
            img = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
            mask = torch.zeros(1, 1, 6, 8, device=device, dtype=dtype)
            mask[:, :, 1:4, 2:6] = 1.0
            return aug(img, mask)[1]

        bilinear = {DataKey.MASK: {"resample": Resample.BILINEAR, "align_corners": None}}
        align = {DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": True}}

        affine = lambda: K.RandomAffine(degrees=(45.0, 45.0), p=1.0)  # noqa: E731
        assert (mask_of(affine, bilinear) - mask_of(affine, None)).abs().max().item() == 0.0

        perspective = lambda: K.RandomPerspective(0.5, p=1.0)  # noqa: E731
        assert (mask_of(perspective, align) - mask_of(perspective, None)).abs().max().item() == 1.0

        def elastic_mask_of(extra):
            torch.manual_seed(0)
            aug = K.AugmentationSequential(
                K.RandomElasticTransform(alpha=(5.0, 5.0), sigma=(4.0, 4.0), p=1.0),
                data_keys=["input", "mask"],
                extra_args=extra,
            )
            img = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(torch.arange(6, device=device), torch.arange(8, device=device), indexing="ij")
            mask = ((yy + xx) % 2).to(dtype).expand(1, 1, 6, 8).clone()
            return aug(img, mask)[1]

        assert (elastic_mask_of(bilinear) - elastic_mask_of(None)).abs().max().item() > 0.1

        # a user dict replaces the container's default wholesale rather than merging into it
        default_args = K.AugmentationSequential(affine(), data_keys=["input", "mask"]).extra_args
        assert default_args == {DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}}
        replaced = K.AugmentationSequential(
            affine(), data_keys=["input", "mask"], extra_args={DataKey.MASK: {"align_corners": True}}
        ).extra_args
        assert replaced == {DataKey.MASK: {"align_corners": True}}

    @pytest.mark.xfail(strict=True, reason="Tracked in #4419")
    def test_convention_extra_args_mask_resample_override_is_honoured(self, device, dtype):
        # Strict xfail (#4419): the intended reading is the one `AugmentationSequential`'s own docstring
        # states - `extra_args[DataKey.MASK]` controls how masks are handled, so asking for a bilinear
        # `resample` must reach the sampler and change the mask, exactly as it already does on
        # `RandomElasticTransform`. Today the 2D geometric mask path overwrites `resample` with NEAREST after
        # merging the override, so this XFAILs; it turns XPASS when the repair lands, which is the signal to
        # delete the wart pin above. Executed 2026-09-11 (torch 2.14.0, cpu): the override moves the mask by
        # 0.0 on `RandomAffine` and by 0.4567949 on `RandomElasticTransform`.
        def mask_of(factory, extra):
            torch.manual_seed(0)
            aug = K.AugmentationSequential(factory(), data_keys=["input", "mask"], extra_args=extra)
            img = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
            mask = torch.zeros(1, 1, 6, 8, device=device, dtype=dtype)
            mask[:, :, 1:4, 2:6] = 1.0
            return aug(img, mask)[1]

        bilinear = {DataKey.MASK: {"resample": Resample.BILINEAR, "align_corners": None}}
        affine = lambda: K.RandomAffine(degrees=(45.0, 45.0), p=1.0)  # noqa: E731
        assert (mask_of(affine, bilinear) - mask_of(affine, None)).abs().max().item() > 0.0

    def test_wart_non_rigid_augmentation_desynchronizes_the_data_keys_4420(self, device, dtype):
        # Wart pin (#4420): the container promises that one call applies the same sampled transform to every
        # registered data type, but a non-rigid augmentation has no transform matrix, so the coordinate keys
        # pass through untouched instead of raising: `RandomElasticTransform` warps the image *and* the mask
        # - it carries its own mask path, which resamples the mask through the very same displacement field,
        # nearest - while the keypoints and the boxes stay exactly where they were, and
        # `RandomThinPlateSpline` / `RandomFisheye` raise a bare `NotImplementedError` (no message) on a mask
        # key.
        # Fixture: a checkerboard mask on a non-square 6x8 frame at `alpha=5`, `sigma=4`, chosen because the
        # mask move has to be visible at every seed, not on a lucky one. A spatially uniform mask cannot
        # witness the warp at all - every pixel lands on a pixel of its own value - and the default
        # `sigma=(32, 32)`, `alpha=(1, 1)` displacement is sub-pixel over a frame this small, so the two
        # together read as "the mask came back unchanged", which is how the mask half of this wart was first
        # mis-recorded.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0), B = 2, 96 mask
        # pixels, seeds 0-9. This fixture changes 47, 39, 17, 34, 50, 33, 63, 38, 35, 30 pixels on cpu
        # float32 - never 0, and never 0 either on float64 (16-58), float16 (16-63), bfloat16 (17-63) or mps
        # float32 (24-54). The old solid (1:4, 2:6) block mask changes 0 pixels at every one of those seeds
        # at the default alpha and sigma (4-13 at this alpha and sigma), and the checkerboard at the default
        # alpha and sigma changes 10, 7, 3, 0, 7, 4, 7, 8, 0, 14 - zero on two of ten seeds, a coin flip,
        # which is why both halves of the fixture are raised here. Image max|move| >= 0.64 and keypoint and
        # bbox_xyxy deltas exactly 0.0 at every seed and dtype.
        # The fix lands in the repair window; do not "correct" this pin here.
        torch.manual_seed(0)
        aug = K.AugmentationSequential(
            K.RandomElasticTransform(p=1.0, alpha=(5.0, 5.0), sigma=(4.0, 4.0)),
            data_keys=["input", "keypoints", "mask", "bbox_xyxy"],
        )
        img = torch.rand(2, 3, 6, 8, device=device, dtype=dtype)
        kpts = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], device=device, dtype=dtype)
        yy, xx = torch.meshgrid(torch.arange(6, device=device), torch.arange(8, device=device), indexing="ij")
        mask = ((yy + xx) % 2).to(dtype).expand(2, 1, 6, 8).clone()
        boxes = torch.tensor([[[0.0, 0.0, 3.0, 2.0]], [[2.0, 1.0, 5.0, 4.0]]], device=device, dtype=dtype)
        out_img, out_kpts, out_mask, out_boxes = aug(img, kpts, mask, boxes)
        assert (out_img - img).abs().max().item() > 0.1  # the image really moved
        assert (out_mask != mask).sum().item() > 0  # and so did the mask, through the same displacement field
        self.assert_close(out_kpts, kpts)  # the wart: the coordinate keys did not follow
        self.assert_close(out_boxes, boxes)
        for factory in (
            lambda: K.RandomThinPlateSpline(p=1.0),
            lambda: K.RandomFisheye(
                torch.tensor([-0.3, 0.3], device=device, dtype=dtype),
                torch.tensor([-0.3, 0.3], device=device, dtype=dtype),
                torch.tensor([0.9, 1.0], device=device, dtype=dtype),
                p=1.0,
            ),
        ):
            with pytest.raises(NotImplementedError):
                K.AugmentationSequential(factory(), data_keys=["input", "mask"])(img, mask)
