import torch

from kornia.augmentation import CenterCrop3D
from testing.base import BaseTester


class TestCenterCrop3D(BaseTester):
    def test_no_transform(self, device, dtype):
        inp = torch.rand(1, 2, 4, 4, 4, device=device, dtype=dtype)
        out = CenterCrop3D(2)(inp)
        assert out.shape == (1, 2, 2, 2, 2)

    def test_transform(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, 8, device=device, dtype=dtype)
        aug = CenterCrop3D(2)
        out = aug(inp)
        assert out.shape == (1, 2, 2, 2, 2)
        assert aug.transform_matrix.shape == (1, 4, 4)

    def test_no_transform_tuple(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, 8, device=device, dtype=dtype)
        out = CenterCrop3D((3, 4, 5))(inp)
        assert out.shape == (1, 2, 3, 4, 5)

    def test_gradcheck(self, device):
        input_tensor = torch.rand(1, 2, 3, 4, 5, device=device, dtype=torch.float64)
        self.gradcheck(CenterCrop3D(3), (input_tensor,))