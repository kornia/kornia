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
from kornia.geometry.transform import resize

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

    def test_3d_augmentation_rejects_4d_input(self, device, dtype):
        input = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)
        aug_list = K.AugmentationSequential(K.RandomAffine3D(360.0, p=1.0), data_keys=["input"])

        with pytest.raises(
            RuntimeError,
            match=r"3D augmentations in AugmentationSequential expect input shape",
        ):
            aug_list(input)

    def test_3d_augmentation_rejects_4d_input_when_replaying_params(self, device, dtype):
        input_5d = torch.randn(1, 2, 3, 5, 6, device=device, dtype=dtype)
        aug_list = K.AugmentationSequential(K.RandomAffine3D(360.0, p=1.0), data_keys=["input"])

        aug_list(input_5d)
        params = aug_list._params

        input_4d = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)

        with pytest.raises(
            RuntimeError,
            match=r"3D augmentations in AugmentationSequential expect input shape",
        ):
            aug_list(input_4d, params=params)

    def test_identity_matrix_3d(self, device, dtype):
        input = torch.rand(2, 1, 3, 4, 5, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomDepthicalFlip3D(p=1.0))

        assert aug.contains_3d_augmentation

        matrix = aug.identity_matrix(input)

        assert matrix.shape == (2, 4, 4)
        expected = torch.eye(4, device=device, dtype=dtype).expand(2, -1, -1)
        assert_close(matrix, expected)

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
    """Convention checks and pins for documented `AugmentationSequential` limitations."""

    def test_convention_flip_is_integer_centre_inclusive_for_every_data_key(self, device, dtype):
        # Convention pin: a horizontal flip maps column x to W - 1 - x, and a vertical flip row y to H - 1 - y,
        # for the image, the mask, keypoints and boxes alike - inclusive pixel coordinates about the integer
        # centre, the same rule `kornia.geometry.transform.hflip` / `vflip` follow.
        # H = 3, W = 4 (asymmetric); hot pixel (row 0, col 1), off both centre lines; keypoint (x=1, y=2).
        # Snippet used to generate expected: this body. hflip:
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
        # Snippet used to generate expected: this body:
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
        # `mask` is the one key whose rank changes: (B, H, W) in, (B, 1, H, W) out
        aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask"])
        out = aug(img, torch.rand(1, 3, 4, device=device, dtype=dtype))
        assert out[1].shape == (1, 1, 3, 4)

    def test_wart_mask_lists_desynchronize_mixed_probability_batches_4477(self, device, dtype):
        # A list entry selects one sample's gate, even when that entry contains the whole batch.
        image = torch.arange(24, device=device, dtype=dtype).reshape(2, 1, 3, 4)
        masks = [image.clone(), image.expand(-1, 2, -1, -1).clone()]
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5), data_keys=["input", "mask"])
        params = [
            ParamItem(
                "RandomHorizontalFlip_0",
                {"batch_prob": torch.tensor([1.0, 0.0]), "forward_input_shape": torch.tensor(image.shape)},
            )
        ]
        out_image, out_masks = seq(image, masks, params=params)
        self.assert_close(out_image, torch.stack((image[0].flip(-1), image[1])))
        self.assert_close(out_masks[0], masks[0].flip(-1))
        self.assert_close(out_masks[1], masks[1])
        assert not torch.equal(out_masks[0][1], out_image[1])
        assert not torch.equal(out_masks[1][0, :1], out_image[0])

        # Separate tensor mask keys preserve the per-sample gates for both channel counts.
        aligned = K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5), data_keys=["input", "mask", "mask"])
        out_image, first, second = aligned(image, *masks, params=params)
        self.assert_close(first, out_image)
        self.assert_close(second, out_image.expand(-1, 2, -1, -1))

        # Keep a deliberate invalid-index probe on CPU to avoid accelerator context poisoning.
        cpu_image = image.cpu()
        with pytest.raises(IndexError):
            seq(cpu_image, [cpu_image.clone() for _ in range(3)], params=params)

    def test_wart_per_sample_mask_lists_fail_warps_and_reuse_crop_window_4477(self, device, dtype):
        image = torch.arange(16, device=device, dtype=dtype).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
        masks = [image[:1].clone(), image[1:].clone()]
        warp = K.AugmentationSequential(K.RandomAffine((30.0, 30.0), p=1.0), data_keys=["input", "mask"])
        with pytest.raises(RuntimeError):
            warp(image, masks)
        # Control: full-batch list entries warp, so the failure above is the per-sample entries.
        assert [mask.shape for mask in warp(image, [image.clone(), image.clone()])[1]] == [(2, 1, 4, 4)] * 2
        crop = K.AugmentationSequential(K.RandomCrop((2, 2), p=1.0), data_keys=["input", "mask"])
        params = crop.forward_parameters(image.shape)
        params[0].data["src"] = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]]],
            device=device,
            dtype=dtype,
        )
        output, out_masks = crop(image, masks, params=params)
        self.assert_close(output[0], image[0, :, :2, :2])
        self.assert_close(output[1], image[1, :, 2:, 2:])
        self.assert_close(out_masks[0], output[:1])
        self.assert_close(out_masks[1], output[:1])
        assert not torch.equal(out_masks[1], output[1:])

    def test_mix_augmentation_inverse_raises_4693(self, device, dtype):
        image = torch.stack([torch.full((2, 4, 6), float(i + 1), device=device, dtype=dtype) for i in range(3)])
        mask = torch.zeros(3, 4, 6, device=device, dtype=torch.long)

        for i in range(3):
            mask[i, :2, :3] = i + 1

        seq = K.AugmentationSequential(
            K.RandomTransplantation(p=1.0),
            data_keys=["input", "mask"],
        )

        torch.manual_seed(7)
        output = seq(image, mask)

        assert not torch.equal(output[0], image)

        with pytest.raises(RuntimeError, match="Inverse for RandomTransplantation is not supported"):
            seq.inverse(*output)

    def test_mix_augmentation_3d_inverse_raises_4693(self, device, dtype):
        image = torch.stack([torch.full((2, 3, 4, 5), float(i + 1), device=device, dtype=dtype) for i in range(3)])
        mask = torch.zeros(3, 3, 4, 5, device=device, dtype=torch.long)

        for i in range(3):
            mask[i, :2, :2, :3] = i + 1

        seq = K.AugmentationSequential(
            K.RandomTransplantation3D(p=1.0),
            data_keys=["input", "mask"],
        )

        torch.manual_seed(7)
        output = seq(image, mask)

        assert not torch.equal(output[0], image)

        with pytest.raises(RuntimeError, match="Inverse for RandomTransplantation3D is not supported"):
            seq.inverse(*output)

    def test_mix_children_dispatch_annotation_keys_4493(self, device, dtype):
        # #4493: the container dispatches a mix child's annotation keys to the child's own handlers: RandomMosaic
        # transforms boxes (container xyxy_plus path), and unsupported keys raise NotImplementedError as a direct
        # call does. A class key raises from the container.
        from kornia.geometry.boxes import Boxes

        image = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)
        boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]], [[1.0, 1.0, 3.0, 3.0]]], device=device, dtype=dtype)
        mask = torch.arange(2, device=device, dtype=dtype).reshape(2, 1, 1, 1).expand(2, 1, 16, 16).clone()
        keypoints = torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]], device=device, dtype=dtype)

        # RandomMosaic boxes are transformed inside the container.
        seq = K.AugmentationSequential(K.RandomMosaic(p=1.0), data_keys=["input", "bbox_xyxy"])
        torch.manual_seed(0)
        out_image, out_boxes = seq(image, boxes)
        assert not torch.equal(out_image, image)
        assert not torch.equal(out_boxes, boxes)
        params = seq._params[0].data
        mosaic = K.RandomMosaic(p=1.0)
        expected = mosaic.transform_boxes(Boxes.from_tensor(boxes, mode="xyxy_plus"), params, mosaic.flags).to_tensor(
            mode="xyxy_plus"
        )
        self.assert_close(out_boxes, expected, rtol=0, atol=0)

        # Unsupported annotation keys raise, matching a direct call (no silent passthrough).
        for factory, key, payload in (
            (lambda: K.RandomMixUpV2(p=1.0), "mask", mask),
            (lambda: K.RandomMixUpV2(p=1.0), "bbox_xyxy", boxes),
            (lambda: K.RandomMixUpV2(p=1.0), "keypoints", keypoints),
            (lambda: K.RandomMosaic(p=1.0), "mask", mask),
            (lambda: K.RandomMosaic(p=1.0), "keypoints", keypoints),
            (lambda: K.RandomCutMixV2(p=1.0), "mask", mask),
            (lambda: K.RandomJigsaw(p=1.0), "mask", mask),
        ):
            with pytest.raises(NotImplementedError):
                K.AugmentationSequential(factory(), data_keys=["input", key])(image, payload)

        with pytest.raises(NotImplementedError, match="class labels"):
            K.AugmentationSequential(K.RandomMixUpV2(p=1.0), data_keys=["input", "class"])(
                image, torch.tensor([0, 1], device=device)
            )

    def test_convention_dictionary_mask_before_image_keeps_float64_precision_4478(self):
        # Convention pin for kornia#4478: a mask is converted with this call's image dtype wherever it sits in the
        # dictionary, so these float64 values, which float32 cannot hold, come back bit for bit in both insertion
        # orders.
        image = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
        mask = torch.tensor([1.0 + 2**-30, 2.0 + 2**-29], dtype=torch.float64).reshape_as(image)
        early_mask = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=None)
        image_first = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=None)
        expected = mask.flip(-1)
        early_output = early_mask({"mask": mask, "image": image})["mask"]
        assert early_output.dtype == torch.float64
        assert torch.equal(early_output, expected)
        assert torch.equal(image_first({"image": image, "mask": mask})["mask"], expected)

    @pytest.mark.parametrize("mask_dtypes", [(torch.int64, torch.bool), (torch.bool, torch.int64)])
    def test_convention_each_mask_keeps_its_own_dtype_4478(self, mask_dtypes, device, dtype):
        # Convention pin for kornia#4478: each mask output comes back in the dtype of its own argument, so labels
        # 2, 3 and 5 survive next to a boolean mask. Both argument orders are covered.
        image = torch.arange(24, device=device, dtype=dtype).reshape(2, 1, 3, 4)
        labels = torch.tensor([0, 2, 3, 5], device=device).reshape(1, 1, 1, 4).expand(2, 1, 3, 4)
        first, second = (labels.to(mask_dtype) for mask_dtype in mask_dtypes)
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask", "mask"])
        _, out_first, out_second = seq(image, first, second)
        assert (out_first.dtype, out_second.dtype) == mask_dtypes
        assert torch.equal(out_first, first.flip(-1))
        assert torch.equal(out_second, second.flip(-1))
        integer_output = out_first if mask_dtypes[0] == torch.int64 else out_second
        assert integer_output.unique().tolist() == [0, 2, 3, 5]  # the labels survive next to a boolean mask
        assert seq.mask_dtype == mask_dtypes[1]  # the attribute still records the last mask argument's dtype

    @pytest.mark.parametrize("mask_dtypes", [(torch.int64, torch.bool), (torch.bool, torch.int64)])
    def test_convention_each_list_mask_element_keeps_its_own_dtype_4478(self, mask_dtypes, device, dtype):
        # Convention pin for kornia#4478: a list mask comes back per element in each entry's own dtype. Both
        # element orders are covered.
        image = torch.arange(16, device=device, dtype=dtype).reshape(2, 1, 2, 4)
        labels = torch.tensor([0, 2, 3, 5], device=device).reshape(1, 1, 1, 4).expand(1, 1, 2, 4)
        entries = [labels.to(mask_dtype) for mask_dtype in mask_dtypes]
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask"])
        _, out = seq(image, entries)
        assert [m.dtype for m in out] == list(mask_dtypes)
        for out_entry, entry in zip(out, entries):
            assert torch.equal(out_entry, entry.flip(-1))
        assert seq.mask_dtype == mask_dtypes[0]  # the attribute records the first element of the last list mask

    @pytest.mark.parametrize("key", ["bbox_xyxy", "bbox_xywh"])
    @pytest.mark.parametrize("suffix", ["", "_2", "-a"])
    def test_dictionary_coordinate_box_keys_4483(self, key, suffix, device, dtype):
        image = torch.arange(20, device=device, dtype=dtype).reshape(1, 1, 4, 5)
        boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], device=device, dtype=dtype)
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=None)
        output = seq({"image": image, key + suffix: boxes})
        expected = [2.0, 0.0, 4.0, 2.0] if key == "bbox_xyxy" else [3.0, 0.0, 2.0, 2.0]
        self.assert_close(output[key + suffix], torch.tensor([[expected]], device=device, dtype=dtype))
        restored = seq.inverse(output)
        self.assert_close(restored[key + suffix], boxes)

    def test_dictionary_class_alias_4483(self):
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=None)
        keys, metadata = seq._read_datakeys_from_dict(("input", "class", "class-a", "label", "label-a"))
        assert keys == [DataKey.INPUT, DataKey.LABEL, DataKey.LABEL, DataKey.LABEL, DataKey.LABEL]
        assert metadata == []

    @pytest.mark.parametrize("cropping_mode", ["slice", "resample"])
    @pytest.mark.parametrize("align_corners", [None, False, True])
    def test_wart_resized_crop_mask_align_corners_depends_on_mode_4802(
        self, cropping_mode, align_corners, device, dtype
    ):
        # #4802: the slice-mode raise flips when slice mode drops `align_corners` for nearest resampling.
        image = torch.arange(16, device=device, dtype=dtype).reshape(1, 1, 4, 4) / 16
        mask = torch.arange(16, device=device, dtype=dtype).reshape(1, 1, 4, 4).remainder(2)
        seq = K.AugmentationSequential(
            K.RandomResizedCrop((2, 2), scale=(1.0, 1.0), ratio=(1.0, 1.0), cropping_mode=cropping_mode, p=1.0),
            data_keys=["input", "mask"],
            extra_args={DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": align_corners}},
        )
        if cropping_mode == "slice" and align_corners is not None:
            with pytest.raises(ValueError):
                seq(image, mask)
        else:
            out_image, out_mask = seq(image, mask)
            assert out_image.shape == out_mask.shape == (1, 1, 2, 2)
            assert set(out_mask.unique().tolist()).issubset({0.0, 1.0})

    def test_convention_masks_keep_labels_and_add_padding_fill_through_a_rotation(self, device, dtype):
        # Convention pin: masks are resampled with nearest interpolation, so a {2, 3} mask still holds those
        # labels after a 45 degree affine without intermediate values. The zero padding fill is also present,
        # and the mask dtype is preserved, `bool` included.
        # The claim is checked one parameter away from the pin's fixture: B = 2 and B = 1 both hold.
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

    @pytest.mark.parametrize("mask_dtype", [torch.float32, torch.int64, torch.bool])
    def test_resize_antialias_preserves_mask_labels_4479(self, mask_dtype, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(12, device=device),
            torch.arange(16, device=device),
            indexing="ij",
        )
        blocks = ((yy // 2) + (xx // 2)) % 2

        if mask_dtype == torch.bool:
            mask = blocks.bool()
        else:
            mask = (2 + blocks).to(mask_dtype)

        mask = mask[None, None]
        image = torch.rand(1, 3, 12, 16, device=device, dtype=dtype)

        without_antialias = K.AugmentationSequential(
            K.Resize((6, 8), antialias=False),
            data_keys=["input", "mask"],
        )
        with_antialias = K.AugmentationSequential(
            K.Resize((6, 8), antialias=True),
            data_keys=["input", "mask"],
        )

        expected_mask = without_antialias(image, mask)[1]
        out_image, out_mask = with_antialias(image, mask)
        expected_image = resize(image, (6, 8), "bilinear", align_corners=True, antialias=True)

        assert out_mask.dtype == mask_dtype
        assert torch.equal(out_mask, expected_mask)
        assert set(out_mask.unique().tolist()) == set(mask.unique().tolist())
        self.assert_close(out_image, expected_image)

    def test_resize_mask_explicit_antialias_override_4479(self, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(12, device=device),
            torch.arange(16, device=device),
            indexing="ij",
        )
        mask = (((yy // 2) + (xx // 2)) % 2).to(dtype)[None, None]
        image = torch.rand(1, 3, 12, 16, device=device, dtype=dtype)

        aug = K.AugmentationSequential(
            K.Resize((6, 8), antialias=True),
            data_keys=["input", "mask"],
            extra_args={
                DataKey.MASK: {
                    "resample": Resample.BILINEAR,
                    "align_corners": True,
                    "antialias": True,
                }
            },
        )

        out_mask = aug(image, mask)[1]
        expected = resize(mask, (6, 8), "bilinear", align_corners=True, antialias=True)
        without_antialias = resize(mask, (6, 8), "bilinear", align_corners=True, antialias=False)

        self.assert_close(out_mask, expected)
        assert not torch.equal(out_mask, without_antialias)

    def test_convention_boxes_follow_the_xyxy_plus_convention(self, device, dtype):
        # Convention pin: the container's box arithmetic is `Boxes`' inclusive `xyxy_plus` mode, for a scaling
        # op as well as for a flip - `Boxes.from_tensor(..., mode="xyxy_plus").transform_boxes_(M)` reproduces
        # the container's output exactly, while `mode="xyxy"` (exclusive) does not.
        # Two asymmetric boxes on a 6x8 image, resized to (3, 4) => M = diag(3/7, 2/5) in xyxy_plus space.
        # Snippet used to generate expected: this body. Container
        # [[0.0, 0.0, 1.2857143, 0.8]], [[0.85714293, 0.4, 2.1428573, 1.6]] == xyxy_plus;
        # xyxy gives [[0.0, 0.0, 1.8571429, 1.4]], [[0.85714293, 0.4, 2.7142859, 2.2]].
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
            assert torch.isfinite(restored_boxes).all()
            if key == "bbox":
                restored_corners = restored_boxes[0, 0]
                original_corners = boxes[0, 0]
            else:
                restored_corners = Boxes.from_tensor(restored_boxes, mode="xyxy_plus").to_tensor(mode="vertices_plus")[
                    0, 0
                ]
                original_corners = Boxes.from_tensor(boxes, mode="xyxy_plus").to_tensor(mode="vertices_plus")[0, 0]
            assert (restored_corners.min(dim=0).values < original_corners.min(dim=0).values - 1).all()
            assert (restored_corners.max(dim=0).values > original_corners.max(dim=0).values + 1).all()

    def test_wart_nested_container_matrix_is_order_sensitive_4476(self, device, dtype):
        # Wart pin (#4476): nesting itself, not a non-rigid child, breaks outer matrix accumulation. A rigid
        # child followed by a nested rigid sequence raises, while the reverse ordering silently drops the
        # nested transform. A directly-called nested container can also inject stale state.
        image = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)
        nested_rigid = lambda: K.AugmentationSequential(  # noqa: E731
            K.RandomVerticalFlip(p=1.0), data_keys=["input"]
        )
        rigid_then_nested = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), nested_rigid(), data_keys=["input"])
        assert rigid_then_nested(image).shape == image.shape
        with pytest.raises(TypeError):
            _ = rigid_then_nested.transform_matrix

        nested_then_rigid = K.AugmentationSequential(nested_rigid(), K.RandomHorizontalFlip(p=1.0), data_keys=["input"])
        assert nested_then_rigid(image).shape == image.shape
        self.assert_close(
            nested_then_rigid.transform_matrix,
            torch.tensor([[[-1.0, 0.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
        )

        stale_child = K.AugmentationSequential(K.RandomVerticalFlip(p=1.0), data_keys=["input"])
        stale_child(image)
        stale_outer = K.AugmentationSequential(stale_child, data_keys=["input"])
        shorter = image[:, :, :6]
        stale_outer(shorter)
        stale = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 7.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        self.assert_close(stale_outer.transform_matrix, stale)
        assert stale_outer.transform_matrix[0, 1, 2].item() != shorter.shape[-2] - 1

    @pytest.mark.parametrize("crop_cls", [K.CenterCrop, K.RandomCrop, K.RandomResizedCrop])
    def test_convention_slice_crop_inverse_raises(self, crop_cls, device, dtype):
        seq = K.AugmentationSequential(crop_cls((4, 6), p=1.0), data_keys=["input"])
        out = seq(torch.rand(1, 3, 6, 8, device=device, dtype=dtype))
        with pytest.raises(NotImplementedError, match="resample cropping mode"):
            seq.inverse(out)

    def test_convention_3d_inverse_raises_for_geometry_and_skips_intensity(self, device, dtype):
        # Enough voxels in two populated histogram bins force an actual equalization in every dtype.
        image = torch.full((1, 1, 8, 8, 8), 0.2, device=device, dtype=dtype)
        image[:, :, :4] = 0.7
        intensity = K.AugmentationSequential(K.RandomEqualize3D(p=1.0), data_keys=["input"])
        output = intensity(image)
        assert not torch.equal(output, image)
        self.assert_close(intensity.inverse(output), output, atol=0, rtol=0)
        geometric = K.AugmentationSequential(K.RandomHorizontalFlip3D(p=1.0), data_keys=["input"])
        with pytest.raises(NotImplementedError, match="3d inverse"):
            geometric.inverse(geometric(image))

    def test_wart_integer_masks_round_through_image_dtype_4478(self, device, dtype):
        first_inexact = {torch.bfloat16: 257, torch.float16: 2049, torch.float32: 2**24 + 1, torch.float64: 2**53 + 1}
        label = first_inexact[dtype]
        image = torch.ones(1, 1, 2, 2, device=device, dtype=dtype)
        mask = torch.full_like(image, label, dtype=torch.int64)
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask"])
        output = seq(image, mask)[1]
        assert output.dtype == torch.int64
        assert torch.equal(output, torch.full_like(mask, label - 1))

    @pytest.mark.parametrize("mask_dtype", [None, torch.int64, torch.bool])
    def test_convention_empty_batch_with_mask_4478(self, mask_dtype, device, dtype):
        # Convention pin for kornia#4478: an empty batch with a mask returns an empty mask, forward and inverse,
        # in the mask's own dtype.
        image = torch.empty(0, 1, 2, 2, device=device, dtype=dtype)
        mask = image.clone() if mask_dtype is None else torch.empty(0, 1, 2, 2, device=device, dtype=mask_dtype)
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask"])
        out_image, out_mask = seq(image, mask)
        assert out_image.shape == out_mask.shape == (0, 1, 2, 2)
        assert out_mask.dtype == mask.dtype
        restored_image, restored_mask = seq.inverse(out_image, out_mask)
        assert restored_image.shape == restored_mask.shape == (0, 1, 2, 2)
        assert restored_mask.dtype == mask.dtype

    def test_dictionary_preserves_metadata_and_input_4483(self, device, dtype):
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=None)
        metadata = {"imagenet_id": 7, "maskrcnn_boxes": "unchanged", "labelled_image": None, "note": "retained"}
        image = torch.arange(4, device=device, dtype=dtype).reshape(1, 1, 2, 2)
        data = {"note": metadata["note"], "image": image, **metadata}
        original = dict(data)
        output = seq(data)
        assert data.keys() == original.keys()
        for key, value in original.items():
            assert data[key] is value
        self.assert_close(output["image"], image.flip(-1))
        for key, value in metadata.items():
            assert output[key] is value
        restored = seq.inverse(output)
        self.assert_close(restored["image"], image)
        for key, value in metadata.items():
            assert restored[key] is value
            assert output[key] is value

    def test_dictionary_key_boundaries_4483(self):
        seq = K.AugmentationSequential(data_keys=None)
        names = ("image", "IMAGE-left", "input_2", "mask_2", "keypoints-right", "bbox_xyxy_2", "bbox_xywh-a", "bbox_2")
        keys, metadata = seq._read_datakeys_from_dict(names)
        assert keys == [
            DataKey.INPUT,
            DataKey.INPUT,
            DataKey.INPUT,
            DataKey.MASK,
            DataKey.KEYPOINTS,
            DataKey.BBOX_XYXY,
            DataKey.BBOX_XYWH,
            DataKey.BBOX,
        ]
        assert metadata == []
        names = (
            "imagenet_id",
            "maskrcnn_boxes",
            "labelled_image",
            "keypoint",
            "classification",
            "inputsize",
            "images",
            "masks",
            "labels",
            "bboxes",
            "inputs",
            "image2",
            "imageLeft",
            "image.2",
            "keypoints2",
        )
        keys, metadata = seq._read_datakeys_from_dict(names)
        assert keys == []
        assert metadata == list(names)

        keys, metadata = seq._read_datakeys_from_dict(
            ("bbox_xyxy2", "bbox_xywh2", "class", "class_id", "class_weights", "class-names")
        )
        assert keys == [DataKey.BBOX, DataKey.BBOX, DataKey.LABEL, DataKey.LABEL, DataKey.LABEL, DataKey.LABEL]
        assert metadata == []

    def test_convention_same_on_batch_none_does_not_override_a_child(self, device, dtype):
        # Convention pin: `AugmentationSequential(same_on_batch=None)` - the default - keeps whatever each
        # child was built with, while `True` and `False` overwrite the child's own setting in both directions.
        # `keepdim` follows the same three-state rule.
        # Snippet used to generate expected: this body, seed 0, B = 4:
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

    @pytest.mark.parametrize("factory", [lambda: K.RandomAffine(30.0, p=1.0), lambda: K.RandomElasticTransform(p=1.0)])
    def test_wart_extra_args_mask_resample_must_be_a_resample_member_4815(self, factory, device, dtype):
        # #4815: flips when the container normalizes a string override as the constructors do, or rejects it with
        # a kornia error.
        image = torch.rand(1, 1, 6, 8, device=device, dtype=dtype)
        for resample, raises in ((Resample.NEAREST, False), ("nearest", True)):
            seq = K.AugmentationSequential(
                factory(), data_keys=["input", "mask"], extra_args={DataKey.MASK: {"resample": resample}}
            )
            if raises:
                with pytest.raises(AttributeError):
                    seq(image, image.clone())
            else:
                assert seq(image, image.clone())[1].shape == image.shape

    def test_extra_args_mask_override_reaches_the_sampler_4419(self, device, dtype):
        # #4419: both halves of `extra_args[DataKey.MASK]` reach the sampler; `RandomElasticTransform`, which has
        # its own mask path, honours both too. The align_corners fixture samples outside the frame with
        # padding_mode="reflection", where the two conventions differ (delta 1.0). The elastic fixture uses a
        # checkerboard mask and a displacement large enough to keep the bilinear-vs-nearest delta well above 0.1.

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
        assert (mask_of(affine, bilinear) - mask_of(affine, None)).abs().max().item() > 0.0

        reflected = lambda: K.RandomAffine(degrees=(45.0, 45.0), p=1.0, padding_mode="reflection")  # noqa: E731
        assert (mask_of(reflected, align) - mask_of(reflected, None)).abs().max().item() == 1.0

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

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: K.RandomAffine(degrees=(45.0, 45.0), p=1.0),
            lambda: K.RandomRotation(degrees=(45.0, 45.0), p=1.0),
            lambda: K.RandomPerspective(0.5, p=1.0),
            lambda: K.RandomShear((20.0, 20.0), p=1.0),
            lambda: K.Resize((9, 13)),
            lambda: K.RandomResizedCrop((9, 13), p=1.0),
        ],
        ids=["RandomAffine", "RandomRotation", "RandomPerspective", "RandomShear", "Resize", "RandomResizedCrop"],
    )
    def test_extra_args_mask_resample_is_honoured_and_nearest_stays_the_default_4419(self, factory, device, dtype):
        # #4419: a bilinear `resample` under `extra_args[DataKey.MASK]` reaches the sampler on the base 2D
        # geometric mask path, as it already did on `inverse` and on `RandomElasticTransform`. Without one --
        # the container's default dict, or a user dict that leaves `resample` out -- masks stay nearest, so a
        # binary mask stays binary even though these classes resample images bilinearly. A checkerboard mask,
        # so that any bilinear resample produces in-between values wherever the crop or warp lands.
        def mask_of(extra):
            torch.manual_seed(0)
            aug = K.AugmentationSequential(factory(), data_keys=["input", "mask"], extra_args=extra)
            img = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(torch.arange(6, device=device), torch.arange(8, device=device), indexing="ij")
            mask = ((yy + xx) % 2).to(dtype).expand(1, 1, 6, 8).clone()
            return aug(img, mask)[1]

        def is_binary(t):
            return bool(((t == 0) | (t == 1)).all())

        default = mask_of(None)
        assert is_binary(default)
        assert is_binary(mask_of({DataKey.MASK: {"align_corners": None}}))

        bilinear = mask_of({DataKey.MASK: {"resample": Resample.BILINEAR, "align_corners": None}})
        assert not is_binary(bilinear)
        assert (bilinear - default).abs().max().item() > 0.0

    def test_direct_transform_masks_call_without_resample_stays_nearest_4419(self, device, dtype):
        # The module's own `resample` is for images. A direct call that does not choose one must not pick it
        # up for the mask.
        torch.manual_seed(0)
        aug = K.RandomAffine(degrees=(45.0, 45.0), p=1.0)
        assert aug.flags["resample"] == Resample.BILINEAR
        img = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
        mask = torch.zeros(1, 1, 6, 8, device=device, dtype=dtype)
        mask[:, :, 1:4, 2:6] = 1.0
        aug(img)
        out = aug.transform_masks(mask, params=aug._params, flags=aug.flags, transform=aug.transform_matrix)
        assert bool(((out == 0) | (out == 1)).all())
        assert aug.flags["resample"] == Resample.BILINEAR

    def test_wart_non_rigid_augmentation_desynchronizes_the_data_keys_4420(self, device, dtype):
        # Wart pin (#4420): the container promises that one call applies the same sampled transform to every
        # registered data type, but a non-rigid augmentation has no transform matrix, so the coordinate keys
        # pass through untouched instead of raising: `RandomElasticTransform` warps the image *and* the mask
        # - it carries its own mask path, which resamples the mask through the very same displacement field,
        # nearest - while the keypoints and the boxes stay exactly where they were, and
        # `RandomThinPlateSpline` / `RandomFisheye` raise a bare `NotImplementedError` (no message) on a mask
        # key.
        # Seed zero and a checkerboard with a stronger displacement expose both the image and mask warp.
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
