import pytest
import torch

import kornia.augmentation as K


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
        output = aug_list(input)

    @pytest.mark.parametrize('augmentations', [[K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5), K.RandomAffine(360, p=0.5)]])
    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    def test_p_half(self, augmentations, data_format, device, dtype):
        input = torch.randn(1, 3, 3, 5, 6, device=device, dtype=dtype).repeat(2, 1, 1, 1, 1)
        torch.manual_seed(21)
        aug_list = K.VideoSequential(*augmentations, data_format=data_format, same_on_frame=True)
        output = aug_list(input)

        assert not (output[0] == input[0]).all()
        assert (output[1] == input[1]).all()

    @pytest.mark.parametrize(
        'augmentations',
        [
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0), K.RandomAffine(360, p=1.0)],
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0)],
            [K.RandomAffine(360, p=1.0)],
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.0), K.RandomAffine(360, p=0.0)],
            [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.0)],
            [K.RandomAffine(360, p=0.0)],
        ],
    )
    @pytest.mark.parametrize('data_format', ["BCTHW", "BTCHW"])
    def test_same_on_frame(self, augmentations, data_format, device, dtype):
        aug_list = K.VideoSequential(*augmentations, data_format=data_format, same_on_frame=True)

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
        param_1 = list(aug_list_1.children())[0]._params

        torch.manual_seed(0)
        if data_format == 'BCTHW':
            input = input.transpose(1, 2)
        output_2 = aug_list_2(input.reshape(-1, 3, 5, 6))
        param_2 = list(aug_list_2.children())[0]._params
        output_2 = output_2.view(2, 4, 3, 5, 6)
        if data_format == 'BCTHW':
            output_2 = output_2.transpose(1, 2)
        assert (output_1 == output_2).all()

    @pytest.mark.jit
    @pytest.mark.skip(reason="turn off due to Union Type")
    def test_jit(self, device, dtype):
        B, C, D, H, W = 2, 3, 5, 4, 4
        img = torch.ones(B, C, D, H, W, device=device, dtype=dtype)
        op = K.VideoSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1), same_on_frame=True)
        op_jit = torch.jit.script(op)
        assert_allclose(op(img), op_jit(img))
