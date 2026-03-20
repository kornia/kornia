def test_randomjpeg_same_on_batch_true():
    import torch
    from kornia.augmentation import RandomJPEG

    torch.manual_seed(0)

    aug = RandomJPEG(p=1.0, same_on_batch=True)
    x = torch.rand(4, 3, 64, 64)

    y = aug(x)

    assert torch.allclose(y[0], y[1])
    assert torch.allclose(y[1], y[2])