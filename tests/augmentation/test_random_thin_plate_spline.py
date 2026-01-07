import torch
from kornia.augmentation import RandomThinPlateSpline


def test_smoke():
    x=torch.randn(2,3,64,64)
    aug=RandomThinPlateSpline()
    y=aug(x)
    assert y.shape==x.shape


def test_same_on_batch():
    torch.manual_seed(42)
    x=torch.randn(4,3,32,32)

    aug = RandomThinPlateSpline(p=1.0, same_on_batch=True)
    y = aug(x)

    params = aug._params

    # src and dst contain TPS control points
    for key in ["src", "dst"]:
        for j in range(1, 4):
            assert torch.allclose(params[key][0], params[key][j])


def test_device_cpu():
    x=torch.randn(1,3,16,16)
    aug=RandomThinPlateSpline(p=1.0)
    y=aug(x)
    assert y.device==x.device