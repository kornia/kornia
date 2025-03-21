import pytest
import torch

import kornia
import kornia.augmentation as K

from reproducibility_test import reproducibility_test


class TestSequential:
    @pytest.mark.parametrize("random_apply_weights", [None, [0.8, 0.9]])
    def test_exception(self, random_apply_weights, device, dtype):
        inp = torch.randn(1, 3, 30, 30, device=device, dtype=dtype)
        with pytest.raises(Exception):  # AssertError and NotImplementedError
            K.ImageSequential(
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0), random_apply_weights=random_apply_weights
            ).inverse(inp)

    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    @pytest.mark.parametrize("keepdim", [True, False, None])
    @pytest.mark.parametrize("random_apply", [1, (2, 2), (1, 2), (2,), 20, True, False])
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

    @pytest.mark.parametrize("random_apply", [1, (2, 2), (1, 2), (2,), 10, True, False])
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