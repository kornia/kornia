import pytest
import torch
from torch.testing import assert_allclose

import kornia
import kornia.augmentation as K
from kornia.constants import BorderType
from kornia.geometry.transform import bbox_to_mask


def reproducibility_test(input, seq):
    if isinstance(input, (tuple, list)):
        output_1 = seq(*input)
        output_2 = seq(*input, params=seq._params)
    else:
        output_1 = seq(input)
        output_2 = seq(input, params=seq._params)

    if isinstance(output_1, (tuple, list)) and isinstance(output_2, (tuple, list)):
        [
            assert_allclose(o1, o2)
            for o1, o2 in zip(output_1, output_2)
            if isinstance(o1, (torch.Tensor,)) and isinstance(o2, (torch.Tensor,))
        ]
    elif isinstance(output_1, (tuple, list)) and isinstance(output_2, (torch.Tensor,)):
        assert_allclose(output_1[0], output_2)
    elif isinstance(output_2, (tuple, list)) and isinstance(output_1, (torch.Tensor,)):
        assert_allclose(output_1, output_2[0])
    elif isinstance(output_2, (torch.Tensor,)) and isinstance(output_1, (torch.Tensor,)):
        assert_allclose(output_1, output_2, msg=f"{seq._params}")
    else:
        assert False, ("cannot compare", type(output_1), type(output_2))


class TestVideoSequential:
    @pytest.mark.parametrize('shape', [(3, 4), (2, 3, 4), (2, 3, 5, 6), (2, 3, 4, 5, 6, 7)])
    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    def test_exception(self, shape, data_format, device, dtype):
        aug_list = K.VideoSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1), data_format=data_format, same_on_frame=True)
        with pytest.raises(AssertionError):
            input = torch.randn(*shape, device=device, dtype=dtype)
            output = aug_list(input)

    @pytest.mark.parametrize(
        'augmentation',
        [
            K.RandomAffine(360, p=1.0),
            K.CenterCrop((3, 3), p=1.0),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
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
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)],
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0), K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0)],
            [K.RandomAffine(360, p=1.0), kornia.color.BgrToRgb()],
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.0), K.RandomAffine(360, p=0.0)],
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.0)],
            [K.RandomAffine(360, p=0.0)],
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
        'augmentations', [[K.RandomAffine(360, p=1.0)], [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0)]]
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
        output_2 = output_2.view(2, 4, 3, 5, 6)
        if data_format == 'BCTHW':
            output_2 = output_2.transpose(1, 2)
        assert (output_1 == output_2).all(), dict(aug_list_1._params)

    @pytest.mark.jit
    @pytest.mark.skip(reason="turn off due to Union Type")
    def test_jit(self, device, dtype):
        B, C, D, H, W = 2, 3, 5, 4, 4
        img = torch.ones(B, C, D, H, W, device=device, dtype=dtype)
        op = K.VideoSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1), same_on_frame=True)
        op_jit = torch.jit.script(op)
        assert_allclose(op(img), op_jit(img))


class TestSequential:
    @pytest.mark.parametrize('same_on_batch', [True, False, None])
    @pytest.mark.parametrize("return_transform", [True, False, None])
    @pytest.mark.parametrize("keepdim", [True, False, None])
    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_construction(self, same_on_batch, return_transform, keepdim, random_apply):
        K.ImageSequential(
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, p=1.0),
            same_on_batch=same_on_batch,
            return_transform=return_transform,
            keepdim=keepdim,
            random_apply=random_apply,
        )

    @pytest.mark.parametrize("return_transform", [True, False, None])
    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_forward(self, return_transform, random_apply, device, dtype):
        inp = torch.randn(1, 3, 30, 30, device=device, dtype=dtype)
        aug = K.ImageSequential(
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
            kornia.filters.MedianBlur((3, 3)),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0, return_transform=True),
            K.RandomAffine(360, p=1.0),
            return_transform=return_transform,
            random_apply=random_apply,
        )
        out = aug(inp)
        if isinstance(out, (tuple,)):
            assert out[0].shape == inp.shape
        else:
            assert out.shape == inp.shape
        reproducibility_test(inp, aug)


class TestAugmentationSequential:
    @pytest.mark.parametrize(
        'data_keys', ["input", ["mask", "input"], ["input", "bbox_yxyx"], [0, 10], [BorderType.REFLECT]]
    )
    @pytest.mark.parametrize("augmentation_list", [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0)])
    def test_exception(self, augmentation_list, data_keys, device, dtype):
        with pytest.raises(Exception):  # AssertError and NotImplementedError
            K.AugmentationSequential(augmentation_list, data_keys=data_keys)

    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_forward_and_inverse(self, random_apply, device, dtype):
        inp = torch.randn(1, 3, 1000, 500, device=device, dtype=dtype)
        bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
        )[:, None].float()
        aug = K.AugmentationSequential(
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, p=1.0),
            data_keys=["input", "mask", "bbox", "keypoints"],
            random_apply=random_apply,
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

    def test_individual_forward_and_inverse(self, device, dtype):
        inp = torch.randn(1, 3, 1000, 500, device=device, dtype=dtype)
        bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
        )[:, None].float()

        aug = K.AugmentationSequential(
            K.RandomAffine(360, p=1.0, return_transform=False), data_keys=['input', 'mask', 'bbox', 'keypoints']
        )
        reproducibility_test((inp, mask, bbox, keypoints), aug)

        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0, return_transform=True))
        assert aug(inp, data_keys=['input'])[0].shape == inp.shape
        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0, return_transform=False))
        assert aug(inp, data_keys=['input']).shape == inp.shape
        assert aug(mask, data_keys=['mask']).shape == mask.shape

        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0, return_transform=True))
        assert aug.inverse(inp, data_keys=['input']).shape == inp.shape
        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0, return_transform=True))
        assert aug.inverse(bbox, data_keys=['bbox']).shape == bbox.shape
        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0, return_transform=True))
        assert aug.inverse(keypoints, data_keys=['keypoints']).shape == keypoints.shape
        aug = K.AugmentationSequential(K.RandomAffine(360, p=1.0, return_transform=True))
        assert aug.inverse(mask, data_keys=['mask']).shape == mask.shape

    @pytest.mark.parametrize('random_apply', [2, (1, 1), (2,), 10, True, False])
    def test_forward_and_inverse_return_transform(self, random_apply, device, dtype):
        inp = torch.randn(1, 3, 1000, 500, device=device, dtype=dtype)
        bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
        )[:, None].float()
        aug = K.AugmentationSequential(
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0, return_transform=True),
            K.RandomAffine(360, p=1.0, return_transform=True),
            data_keys=["input", "mask", "bbox", "keypoints"],
            random_apply=random_apply,
        )
        out = aug(inp, mask, bbox, keypoints)
        assert out[0][0].shape == inp.shape
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
        keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
        mask = bbox_to_mask(
            torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
        )[:, None].float()
        aug = K.AugmentationSequential(
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0, return_transform=True),
            K.RandomAffine(360, p=1.0, return_transform=True),
            data_keys=["input", "mask", "bbox", "keypoints"],
            random_apply=random_apply,
        )

        out_inv = aug.inverse(inp, mask, bbox, keypoints)
        assert out_inv[0].shape == inp.shape
        assert out_inv[1].shape == mask.shape
        assert out_inv[2].shape == bbox.shape
        assert out_inv[3].shape == keypoints.shape

        out = aug(inp, mask, bbox, keypoints)
        assert out[0][0].shape == inp.shape
        assert out[1].shape == mask.shape
        assert out[2].shape == bbox.shape
        assert out[3].shape == keypoints.shape

        reproducibility_test((inp, mask, bbox, keypoints), aug)

    @pytest.mark.jit
    @pytest.mark.skip(reason="turn off due to Union Type")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = K.AugmentationSequential(
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0), same_on_batch=True
        )
        op_jit = torch.jit.script(op)
        assert_allclose(op(img), op_jit(img))


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
                    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJitter(0.1, 0.1, 0.1, 0.1),
                K.ImageSequential(
                    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJitter(0.1, 0.1, 0.1, 0.1),
                **error_param,
            )

    @pytest.mark.parametrize('shape', [(2, 3, 24, 24)])
    @pytest.mark.parametrize('padding', ["same", "valid"])
    @pytest.mark.parametrize('patchwise_apply', [True, False])
    @pytest.mark.parametrize('same_on_batch', [True, False, None])
    @pytest.mark.parametrize('keepdim', [True, False, None])
    @pytest.mark.parametrize('random_apply', [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_forward(self, shape, padding, patchwise_apply, same_on_batch, keepdim, random_apply, device, dtype):
        try:  # skip wrong param settings.
            seq = K.PatchSequential(
                K.ImageSequential(
                    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJitter(0.1, 0.1, 0.1, 0.1),
                K.ImageSequential(
                    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJitter(0.1, 0.1, 0.1, 0.1),
                grid_size=(2, 2),
                padding=padding,
                patchwise_apply=patchwise_apply,
                same_on_batch=same_on_batch,
                keepdim=keepdim,
                random_apply=random_apply,
            )
        except:
            return

        input = torch.randn(*shape, device=device, dtype=dtype)
        trans = torch.randn(shape[0], 3, 3, device=device, dtype=dtype)
        out = seq(input)
        assert out.shape[-3:] == input.shape[-3:]

        out = seq((input, trans))
        assert out[0].shape[-3:] == input.shape[-3:]
        assert out[1].shape == trans.shape

        reproducibility_test(input, seq)

    def test_intensity_only(self):
        seq = K.PatchSequential(
            K.ImageSequential(
                K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
                K.RandomPerspective(0.2, p=0.5),
                K.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            K.ImageSequential(
                K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
                K.RandomPerspective(0.2, p=0.5),
                K.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            grid_size=(2, 2),
        )
        assert not seq.is_intensity_only()

        seq = K.PatchSequential(
            K.ImageSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5)),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            grid_size=(2, 2),
        )
        assert seq.is_intensity_only()
