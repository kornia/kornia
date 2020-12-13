import torch
import pytest

import kornia.augmentation as K


class TestVideoSequential:

    @pytest.mark.parametrize('shape', [(3, 4), (2, 3, 4), (2, 3, 5, 6), (2, 3, 4, 5, 6, 7)])
    def test_exception(self, shape, device, dtype):
        aug_list = K.VideoSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1), same_on_frame=True)
        with pytest.raises(AssertionError):
            input = torch.randn(*shape, device=device, dtype=dtype)
            output = aug_list(input)

    @pytest.mark.parametrize('augmentations', [
        [
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=.5),
            K.RandomAffine(360, p=.5),
        ]
    ])
    def test_p_half(self, augmentations, device, dtype):
        input = torch.randn(1, 3, 4, 5, 6, device=device, dtype=dtype).repeat(2, 1, 1, 1, 1)
        torch.manual_seed(21)
        aug_list = K.VideoSequential(*augmentations, same_on_frame=True)
        output = aug_list(input)

        assert (output[0] == input[0]).all()
        assert (output[1] == input[1]).all()

    @pytest.mark.parametrize('augmentations', [
        [
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.),
            K.RandomAffine(360, p=1.),
        ],
        [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)],
        [K.RandomAffine(360, p=1.)],
        [
            K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.),
            K.RandomAffine(360, p=0.),
        ],
        [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.)],
        [K.RandomAffine(360, p=0.)],
    ])
    def test_same_on_frame(self, augmentations, device, dtype):
        input = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
        aug_list = K.VideoSequential(*augmentations, same_on_frame=True)
        output = aug_list(input)

        assert (output[:, :, 0] == output[:, :, 1]).all()
        assert (output[:, :, 1] == output[:, :, 2]).all()
        assert (output[:, :, 2] == output[:, :, 3]).all()

    @pytest.mark.parametrize('augmentations', [
        [K.RandomAffine(360, p=1.)],
        [K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)]
    ])
    def test_against_sequential(self, augmentations, device, dtype):
        input = torch.randn(2, 3, 1, 5, 6, device=device, dtype=dtype).repeat(1, 1, 4, 1, 1)
        aug_list_1 = K.VideoSequential(*augmentations, same_on_frame=False)
        aug_list_2 = torch.nn.Sequential(*augmentations)

        torch.manual_seed(0)
        output_1 = aug_list_1(input)
        param_1 = list(aug_list_1.children())[0]._params

        torch.manual_seed(0)
        input = input.transpose(1, 2).reshape(-1, 3, 5, 6)
        output_2 = aug_list_2(input)
        param_2 = list(aug_list_2.children())[0]._params
        output_2 = output_2.view(2, 4, 3, 5, 6).transpose(1, 2)
        assert (output_1 == output_2).all()

    @pytest.mark.jit
    @pytest.mark.skip(reason="turn off due to Union Type")
    def test_jit(self, device, dtype):
        B, C, D, H, W = 2, 3, 5, 4, 4
        img = torch.ones(B, C, D, H, W, device=device, dtype=dtype)
        op = K.VideoSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1), same_on_frame=True)
        op_jit = torch.jit.script(op)
        assert_allclose(op(img), op_jit(img))
