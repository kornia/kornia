from functools import partial
from unittest.mock import patch

import pytest
import torch

import kornia
import kornia.augmentation as K
from kornia.augmentation.container.base import ParamItem
from kornia.constants import BorderType
from kornia.geometry.bbox import bbox_to_mask
from kornia.testing import assert_close


def reproducibility_test(input, seq):
    """Any tests failed here indicate the output cannot be reproduced by the same params."""
    if isinstance(input, (tuple, list)):
        output_1 = seq(*input)
        output_2 = seq(*input, params=seq._params)
    else:
        output_1 = seq(input)
        output_2 = seq(input, params=seq._params)

    if isinstance(output_1, (tuple, list)) and isinstance(output_2, (tuple, list)):
        [
            assert_close(o1, o2)
            for o1, o2 in zip(output_1, output_2)
            if isinstance(o1, (torch.Tensor,)) and isinstance(o2, (torch.Tensor,))
        ]
    elif isinstance(output_1, (tuple, list)) and isinstance(output_2, (torch.Tensor,)):
        assert_close(output_1[0], output_2)
    elif isinstance(output_2, (tuple, list)) and isinstance(output_1, (torch.Tensor,)):
        assert_close(output_1, output_2[0])
    elif isinstance(output_2, (torch.Tensor,)) and isinstance(output_1, (torch.Tensor,)):
        assert_close(output_1, output_2, msg=f"{seq._params}")
    else:
        assert False, ("cannot compare", type(output_1), type(output_2))


class TestVideoSequential:
    @pytest.mark.parametrize('shape', [(3, 4), (2, 3, 4), (2, 3, 5, 6), (2, 3, 4, 5, 6, 7)])
    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    def test_exception(self, shape, data_format, device, dtype):
        aug_list = K.VideoSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1), data_format=data_format, same_on_frame=True)
        with pytest.raises(AssertionError):
            img = torch.randn(*shape, device=device, dtype=dtype)
            aug_list(img)

    @pytest.mark.parametrize(
        'augmentation',
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
        ],
    )
    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    def test_augmentation(self, augmentation, data_format, device, dtype):
        input = torch.randint(255, (1, 3, 3, 5, 6), device=device, dtype=dtype).repeat(2, 1, 1, 1, 1) / 255.0
        torch.manual_seed(21)
        aug_list = K.VideoSequential(augmentation, data_format=data_format, same_on_frame=True)
        reproducibility_test(input, aug_list)

    @pytest.mark.parametrize(
        'augmentations',
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
    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    @pytest.mark.parametrize('random_apply', [1, (1, 1), (1,), 10, True, False])
    def test_same_on_frame(self, augmentations, data_format, random_apply, device, dtype):
        aug_list = K.VideoSequential(
            *augmentations, data_format=data_format, same_on_frame=True, random_apply=random_apply
        )

        if data_format == 'BCTHW':
            input = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
            output = aug_list(input)
            assert (output[:, :, 0] == output[:, :, 1]).all()
            assert (output[:, :, 1] == output[:, :, 2]).all()
            assert (output[:, :, 2] == output[:, :, 3]).all()
        if data_format == 'BTCHW':
            input = torch.randn(2, 1, 3, 5, 6, device=device, dtype=dtype).repeat(1, 4, 1, 1, 1)
            output = aug_list(input)
            assert (output[:, 0] == output[:, 1]).all()
            assert (output[:, 1] == output[:, 2]).all()
            assert (output[:, 2] == output[:, 3]).all()
        reproducibility_test(input, aug_list)

    @pytest.mark.parametrize(
        'augmentations',
        [
            [K.RandomAffine(360, p=1.0)],
            [K.RandomCrop((2, 2), padding=2)],
            [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)],
            [K.RandomAffine(360, p=0.0), K.ImageSequential(K.RandomAffine(360, p=0.0))],
        ],
    )
    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    def test_against_sequential(self, augmentations, data_format, device, dtype):
        aug_list_1 = K.VideoSequential(*augmentations, data_format=data_format, same_on_frame=False)
        aug_list_2 = torch.nn.Sequential(*augmentations)

        if data_format == 'BCTHW':
            input = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
        if data_format == 'BTCHW':
            input = torch.randn(2, 1, 3, 5, 6, device=device, dtype=dtype).repeat(1, 4, 1, 1, 1)

        torch.manual_seed(0)
        output_1 = aug_list_1(input)

        torch.manual_seed(0)
        if data_format == 'BCTHW':
            input = input.transpose(1, 2)
        output_2 = aug_list_2(input.reshape(-1, 3, 5, 6))
        if any(isinstance(a, K.RandomCrop) for a in augmentations):
            output_2 = output_2.view(2, 4, 3, 2, 2)
        else:
            output_2 = output_2.view(2, 4, 3, 5, 6)
        if data_format == 'BCTHW':
            output_2 = output_2.transpose(1, 2)
        assert (output_1 == output_2).all(), dict(aug_list_1._params)

    @pytest.mark.jit
    @pytest.mark.skip(reason="turn off due to Union Type")
    def test_jit(self, device, dtype):
        B, C, D, H, W = 2, 3, 5, 4, 4
        img = torch.ones(B, C, D, H, W, device=device, dtype=dtype)
        op = K.VideoSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1), same_on_frame=True)
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    def test_autocast(self, data_format, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        tfs = (K.RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0), K.RandomGaussianBlur((3, 3), (0.1, 3), p=1))
        aug = K.VideoSequential(*tfs, data_format=data_format, random_apply=True)
        if data_format == 'BCTHW':
            imgs = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
        elif data_format == 'BTCHW':
            imgs = torch.randn(2, 1, 3, 5, 6, device=device, dtype=dtype).repeat(1, 4, 1, 1, 1)

        with torch.autocast(device.type):
            output = aug(imgs)

        assert output.dtype == dtype, 'Output image dtype should match the input dtype'


class TestSequential:
    @pytest.mark.parametrize('random_apply_weights', [None, [0.8, 0.9]])
    def test_exception(self, random_apply_weights, device, dtype):
        inp = torch.randn(1, 3, 30, 30, device=device, dtype=dtype)
        with pytest.raises(Exception):  # AssertError and NotImplementedError
            K.ImageSequential(
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), random_apply_weights=random_apply_weights
            ).inverse(inp)

    @pytest.mark.parametrize('same_on_batch', [True, False, None])
    @pytest.mark.parametrize("keepdim", [True, False, None])
    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 20, True, False])
    def test_construction(self, same_on_batch, keepdim, random_apply):
        aug = K.ImageSequential(
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, p=1.0),
            K.RandomMixUpV2(p=1.0),
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            random_apply=random_apply,
        )
        aug.same_on_batch = True
        aug.keepdim = True
        for m in aug.children():
            assert m.same_on_batch is True, m.same_on_batch
            assert m.keepdim is True, m.keepdim

    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_forward(self, random_apply, device, dtype):
        inp = torch.randn(1, 3, 30, 30, device=device, dtype=dtype)
        aug = K.ImageSequential(
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            kornia.filters.MedianBlur((3, 3)),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)),
            K.ImageSequential(K.RandomAffine(360, p=1.0)),
            K.RandomAffine(360, p=1.0),
            K.RandomMixUpV2(p=1.0),
            random_apply=random_apply,
        )
        out = aug(inp)
        assert out.shape == inp.shape
        aug.inverse(inp)
        reproducibility_test(inp, aug)


class TestAugmentationSequential:
    @pytest.mark.parametrize(
        'data_keys', ["input", "image", ["mask", "input"], ["input", "bbox_yxyx"], [0, 10], [BorderType.REFLECT]]
    )
    @pytest.mark.parametrize("augmentation_list", [K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)])
    def test_exception(self, augmentation_list, data_keys, device, dtype):
        with pytest.raises(Exception):  # AssertError and NotImplementedError
            K.AugmentationSequential(augmentation_list, data_keys=data_keys)

    @pytest.mark.parametrize('same_on_batch', [True, False])
    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
    @pytest.mark.parametrize('inp', [torch.randn(1, 3, 1000, 500), torch.randn(3, 1000, 500)])
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
            K.RandomCrop((crop_width, crop_height), padding=1, cropping_mode='resample', fill=0),
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
            K.RandomCrop((crop_width, crop_height), padding=1, cropping_mode='resample', fill=0),
            K.RandomHorizontalFlip(p=1.0),
            data_keys=["input", "bbox_xyxy"],
        )

        reproducibility_test((input, bbox), aug)

        _params = aug.forward_parameters(input.shape)
        # specifying the crop locations allows us to compute by hand the expected outputs
        crop_locations = torch.tensor(
            [[1.0, 2.0], [1.0, 1.0], [2.0, 0.0]],
            device=_params[0].data['src'].device,
            dtype=_params[0].data['src'].dtype,
        )
        crops = crop_locations.expand(4, -1, -1).permute(1, 0, 2).clone()
        crops[:, 1:3, 0] += crop_width - 1
        crops[:, 2:4, 1] += crop_height - 1
        _params[0].data['src'] = crops

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
        torch.manual_seed(233)
        input = torch.randn(3, 3, 3, 3, device=device, dtype=dtype)
        bbox = torch.tensor(
            [[[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0]]], device=device, dtype=dtype
        ).expand(3, -1, -1)
        points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], device=device, dtype=dtype).expand(3, -1, -1)
        aug = K.AugmentationSequential(
            K.RandomCrop((3, 3), padding=1, cropping_mode='resample', fill=0),
            K.RandomAffine((360.0, 360.0), p=1.0),
            data_keys=["input", "mask", "bbox_xyxy", "keypoints"],
            extra_args={},
        )

        reproducibility_test((input, input, bbox, points), aug)

        _params = aug.forward_parameters(input.shape)
        # specifying the crops allows us to compute by hand the expected outputs
        _params[0].data['src'] = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
                [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
                [[2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0]],
            ],
            device=_params[0].data['src'].device,
            dtype=_params[0].data['src'].dtype,
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
        assert_close(out[2], expected_out_bbox, atol=1e-4, rtol=1e-4)
        assert out[3].shape == points.shape
        assert_close(out[3], expected_out_points, atol=1e-4, rtol=1e-4)

        out_inv = aug.inverse(*out)
        assert out_inv[0].shape == input.shape
        assert_close(out_inv[0], out_inv[1], atol=1e-4, rtol=1e-4)
        assert out_inv[2].shape == bbox.shape
        assert_close(out_inv[2], bbox, atol=1e-4, rtol=1e-4)
        assert out_inv[3].shape == points.shape
        assert_close(out_inv[3], points, atol=1e-4, rtol=1e-4)

    def test_bbox(self, device, dtype):
        img = torch.rand((3, 3, 10, 10), device=device, dtype=dtype)
        bbox = [
            torch.tensor([[1, 5, 2, 7], [0, 3, 9, 9]], device=device, dtype=dtype),
            torch.tensor([[1, 5, 2, 7], [0, 3, 9, 9], [0, 5, 8, 7]], device=device, dtype=dtype),
            torch.empty((0, 4), device=device, dtype=dtype),
        ]

        inputs = [img, bbox]

        aug = K.AugmentationSequential(K.Resize((30, 30)), data_keys=['input', 'bbox_xyxy'])

        transformed = aug(*inputs)

        assert len(transformed) == len(inputs)
        bboxes_transformed = transformed[-1]
        assert len(bboxes_transformed) == len(bbox) and isinstance(bboxes_transformed, (list,))
        assert len(bboxes_transformed[0]) == 2
        assert len(bboxes_transformed[1]) == 3, bboxes_transformed[1]
        assert len(bboxes_transformed[2]) == 0

    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
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
            K.RandomCrop(crop_size, padding=1, cropping_mode='resample', fill=0),
            data_keys=['input', 'mask', 'bbox', 'keypoints'],
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
        assert aug(inp, data_keys=['input']).shape == inp.shape
        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0))
        assert aug(inp, data_keys=['input']).shape == inp.shape
        assert aug(mask, data_keys=['mask'], params=aug._params).shape == mask.shape

        assert aug.inverse(inp, data_keys=['input']).shape == inp.shape
        assert aug.inverse(bbox, data_keys=['bbox']).shape == bbox.shape
        assert aug.inverse(keypoints, data_keys=['keypoints']).shape == keypoints.shape
        assert aug.inverse(mask, data_keys=['mask']).shape == mask.shape

    @pytest.mark.parametrize('random_apply', [2, (1, 1), (2,), 10, True, False])
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

    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
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

    @pytest.mark.jit
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
    @pytest.mark.parametrize("box", ['bbox', 'bbox_xyxy', 'bbox_xywh'])
    def test_autocast(self, batch_prob, box, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        def mock_forward_parameters_sequential(batch_shape, cls, batch_prob):
            named_modules = cls.get_forward_sequence()
            params = []
            for name, module in named_modules:
                if isinstance(module, (K.base._AugmentationBase, K.MixAugmentationBaseV2, K.ImageSequential)):
                    with patch.object(module, '__batch_prob_generator__', return_value=batch_prob):
                        mod_param = module.forward_parameters(batch_shape)

                    param = ParamItem(name, mod_param)
                else:
                    param = ParamItem(name, None)
                batch_shape = K.container.image._get_new_batch_shape(param, batch_shape)
                params.append(param)
            return params

        tfs = (K.RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0), K.RandomGaussianBlur((3, 3), (0.1, 3), p=1))
        data_keys = ['input', 'mask', box, 'keypoints']
        aug = K.AugmentationSequential(*tfs, data_keys=data_keys, random_apply=True)
        bs = len(batch_prob)
        imgs = torch.rand(bs, 3, 7, 4, dtype=dtype, device=device)
        if box == 'bbox':
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
        with patch.object(aug, 'forward_parameters', fwd_params):
            params = aug.forward_parameters(imgs.shape)

        with torch.autocast(device.type):
            outputs = aug(imgs, msk, bb, points, params=params)

        assert outputs[0].dtype == dtype, 'Output image dtype should match the input dtype'
        assert outputs[1].dtype == dtype, 'Output mask dtype should match the input dtype'
        assert outputs[2].dtype == dtype, 'Output box dtype should match the input dtype'
        assert outputs[3].dtype == dtype, 'Output keypoints dtype should match the input dtype'


class TestPatchSequential:
    @pytest.mark.parametrize(
        'error_param',
        [
            {"random_apply": False, "patchwise_apply": True, "grid_size": (2, 3)},
            {"random_apply": 2, "patchwise_apply": True},
            {"random_apply": (2, 3), "patchwise_apply": True},
        ],
    )
    def test_exception(self, error_param):
        with pytest.raises(Exception):  # AssertError and NotImplementedError
            K.PatchSequential(
                K.ImageSequential(
                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
                K.ImageSequential(
                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
                **error_param,
            )

    @pytest.mark.parametrize('shape', [(2, 3, 24, 24)])
    @pytest.mark.parametrize('padding', ["same", "valid"])
    @pytest.mark.parametrize('patchwise_apply', [True, False])
    @pytest.mark.parametrize('same_on_batch', [True, False, None])
    @pytest.mark.parametrize('keepdim', [True, False, None])
    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_forward(self, shape, padding, patchwise_apply, same_on_batch, keepdim, random_apply, device, dtype):
        torch.manual_seed(11)
        try:  # skip wrong param settings.
            seq = K.PatchSequential(
                K.color.RgbToBgr(),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
                K.ImageSequential(
                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.RandomMixUpV2(p=1.0),
                grid_size=(2, 2),
                padding=padding,
                patchwise_apply=patchwise_apply,
                same_on_batch=same_on_batch,
                keepdim=keepdim,
                random_apply=random_apply,
            )
        # TODO: improve me and remove the exception.
        except Exception:
            return

        input = torch.randn(*shape, device=device, dtype=dtype)
        out = seq(input)
        assert out.shape[-3:] == input.shape[-3:]

        reproducibility_test(input, seq)

    def test_intensity_only(self):
        seq = K.PatchSequential(
            K.ImageSequential(
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                K.RandomPerspective(0.2, p=0.5),
                K.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.ImageSequential(
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                K.RandomPerspective(0.2, p=0.5),
                K.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            grid_size=(2, 2),
        )
        assert not seq.is_intensity_only()

        seq = K.PatchSequential(
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5)),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            grid_size=(2, 2),
        )
        assert seq.is_intensity_only()

    def test_autocast(self, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        tfs = (K.RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0), K.RandomGaussianBlur((3, 3), (0.1, 3), p=1))
        aug = K.PatchSequential(*tfs, grid_size=(2, 2), random_apply=True)
        imgs = torch.rand(2, 3, 7, 4, dtype=dtype, device=device)

        with torch.autocast(device.type):
            output = aug(imgs)

        assert output.dtype == dtype, 'Output image dtype should match the input dtype'


class TestDispatcher:
    def test_many_to_many(self, device, dtype):
        input_1 = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)
        input_2 = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)
        mask_1 = torch.ones(2, 1, 5, 6, device=device, dtype=dtype)
        mask_2 = torch.ones(2, 1, 5, 6, device=device, dtype=dtype)
        aug_list = K.ManyToManyAugmentationDispather(
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
        )
        output = aug_list((input_1, mask_1), (input_2, mask_2))

        assert output[0][0].shape == input_1.shape
        assert output[1][0].shape == input_2.shape
        assert output[0][1].shape == mask_1.shape
        assert output[1][1].shape == mask_2.shape

    @pytest.mark.parametrize('strict', [True, False])
    def test_many_to_one(self, strict, device, dtype):
        input = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)
        mask = torch.ones(2, 1, 5, 6, device=device, dtype=dtype)
        aug_list = K.ManyToOneAugmentationDispather(
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
            strict=strict,
        )
        output = aug_list(input, mask)

        assert output[0][0].shape == input.shape
        assert output[1][0].shape == input.shape
        assert output[0][1].shape == mask.shape
        assert output[1][1].shape == mask.shape

    @pytest.mark.parametrize('strict', [True, False])
    def test_many_to_one_strict_mode(self, strict):
        def _init_many_to_one(strict):
            K.ManyToOneAugmentationDispather(
                K.AugmentationSequential(
                    kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                    kornia.augmentation.RandomAffine(360, p=1.0),
                    data_keys=["input"],
                ),
                K.AugmentationSequential(
                    kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                    kornia.augmentation.RandomAffine(360, p=1.0),
                    data_keys=["input", "mask"],
                ),
                strict=strict,
            )

        if strict:
            with pytest.raises(RuntimeError):
                _init_many_to_one(strict)  # fails
        else:
            _init_many_to_one(strict)  # passes
