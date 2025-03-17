import pytest
import torch

from kornia.augmentation import (
    RandomVerticalFlip3D,
)
from kornia.augmentation.container.augment import AugmentationSequential

from testing.base import BaseTester


class TestRandomVerticalFlip3D(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomVerticalFlip3D(0.5)
        repr = "RandomVerticalFlip3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        assert str(f) == repr

    def test_random_vflip(self, device, dtype):
        f = RandomVerticalFlip3D(p=1.0)
        f1 = RandomVerticalFlip3D(p=0.0)

        input_tensor = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 2 x 3 x 3

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 2 x 3 x 3

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )  # 4 x 4

        identity = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )  # 1 x 4 x 4

        self.assert_close(f(input_tensor), expected)
        self.assert_close(f.transform_matrix, expected_transform)
        self.assert_close(f1(input_tensor), input_tensor)
        self.assert_close(f1.transform_matrix, identity)

    def test_batch_random_vflip(self, device):
        f = RandomVerticalFlip3D(p=1.0)
        f1 = RandomVerticalFlip3D(p=0.0)

        input_tensor = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]]])  # 1 x 1 x 1 x 3 x 3
        input_tensor = input_tensor.to(device)

        expected = torch.tensor([[[[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]])  # 1 x 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        )  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        identity = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        )  # 1 x 4 x 4
        identity = identity.to(device)

        input_tensor = input_tensor.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 4 x 4
        identity = identity.repeat(5, 1, 1)  # 5 x 4 x 4

        self.assert_close(f(input_tensor), expected)
        self.assert_close(f.transform_matrix, expected_transform)
        self.assert_close(f1(input_tensor), input_tensor)
        self.assert_close(f1.transform_matrix, identity)

    def test_same_on_batch(self, device):
        f = RandomVerticalFlip3D(p=0.5, same_on_batch=True)
        input_tensor = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1, 1)
        res = f(input_tensor)
        self.assert_close(res[0], res[1])

    def test_sequential(self, device):
        f = AugmentationSequential(RandomVerticalFlip3D(p=1.0), RandomVerticalFlip3D(p=1.0))

        input_tensor = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]]])  # 1 x 1 x 1 x 4 x 4
        input_tensor = input_tensor.to(device)

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        )  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        self.assert_close(f(input_tensor), input_tensor)
        self.assert_close(f.transform_matrix, expected_transform_1)

    def test_gradcheck(self, device):
        input_tensor = torch.rand((1, 3, 3)).to(device)  # 4 x 4
        self.gradcheck(RandomVerticalFlip3D(p=1.0), (input_tensor,))