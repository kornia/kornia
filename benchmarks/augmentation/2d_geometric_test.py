import torch

from kornia.augmentation import (
    CenterCrop,
    PadTo,
    RandomAffine,
    RandomCrop,
    RandomElasticTransform,
    RandomErasing,
    RandomFisheye,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomResizedCrop,
    RandomRotation,
    RandomShear,
    RandomThinPlateSpline,
    RandomTranslate,
    RandomVerticalFlip,
    Resize,
)


def test_aug_2d_centercrop(benchmark, device, dtype, torch_optimizer, shape):
    w_target = h_target = 64
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = CenterCrop((h_target, w_target), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == (*shape[:-2], h_target, w_target)


def test_aug_2d_padto(benchmark, device, dtype, torch_optimizer, shape):
    w_target = h_target = 256
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = PadTo((h_target, w_target))
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == (*shape[:-2], h_target, w_target)


def test_aug_2d_affine(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomAffine(degrees=45, translate=0.25, scale=(0.9, 1.1), shear=1.25, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_crop(benchmark, device, dtype, torch_optimizer, shape):
    w_target = h_target = 64
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomCrop((h_target, w_target), pad_if_needed=True, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == (*shape[:-2], h_target, w_target)


def test_aug_2d_elastic_transform(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomElasticTransform(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_erasing(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomErasing(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_fisheye(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    center_x = center_y = torch.tensor([-0.3, 0.3], device=device, dtype=dtype)
    gamma = torch.tensor([0.9, 1.0], device=device, dtype=dtype)
    aug = RandomFisheye(center_x, center_y, gamma, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_horizontal_flip(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomHorizontalFlip(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_perspective(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomPerspective(0.5, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_resized_crop(benchmark, device, dtype, torch_optimizer, shape):
    w_target = h_target = 64
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomResizedCrop((h_target, w_target), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == (*shape[:-2], h_target, w_target)


def test_aug_2d_rotation(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomRotation(45, p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_shear(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomShear((-5.0, 2.0, 5.0, 10.0), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_thin_plate_spline(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomThinPlateSpline(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_translate(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomTranslate((-0.2, 0.2), (-0.1, 0.1), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_vertical_flip(benchmark, device, dtype, torch_optimizer, shape):
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = RandomVerticalFlip(p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == shape


def test_aug_2d_resize(benchmark, device, dtype, torch_optimizer, shape):
    w_target = h_target = 64
    data = torch.rand(*shape, device=device, dtype=dtype)
    aug = Resize((h_target, w_target), p=1.0)
    op = torch_optimizer(aug)

    actual = benchmark(op, input=data)

    assert actual.shape == (*shape[:-2], h_target, w_target)
