import torch
import kornia as K

from docs.generate_examples import handle_special_cases, apply_augmentation


def test_handle_special_cases_jigsaw():
    img = torch.randn(1, 3, 100, 100)
    out = handle_special_cases("RandomJigsaw", img)
    assert out.shape[-2:] == (1020, 500)


def test_handle_special_cases_jpeg():
    img = torch.randn(1, 3, 200, 200)
    out = handle_special_cases("RandomJPEG", img)
    assert out.shape[-2] == 176


def test_apply_augmentation_runs():
    img = torch.randn(2, 3, 64, 64)
    out = apply_augmentation(K.augmentation, "RandomHorizontalFlip", (), 42, img)
    assert out.shape == img.shape

if __name__ == "__main__":
    test_handle_special_cases_jigsaw()
    test_handle_special_cases_jpeg()
    test_apply_augmentation_runs()
    print("All tests passed!")