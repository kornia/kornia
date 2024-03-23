import torch

from kornia.augmentation import RandomHorizontalFlip, RandomVerticalFlip


def test_aug_2d_horizontal_flip(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomHorizontalFlip()
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_vertical_flip(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomVerticalFlip()
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape
