import pytest
import torch

from kornia.augmentation import RandomAffine3D
from testing.base import BaseTester


class TestRandomAffine3D(BaseTester):
    def test_batch_random_affine_3d(self, device, dtype):
        # TODO(jian): cuda and fp64
        if "cuda" in str(device) and dtype == torch.float64:
            pytest.skip("AssertionError: assert tensor(False, device='cuda:0')")

        f = RandomAffine3D((0, 0, 0), p=1.0)  # No rotation
        tensor = torch.tensor(
            [[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]], device=device, dtype=dtype
        )  # 1 x 1 x 1 x 3 x 3

        expected = torch.tensor(
            [[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]], device=device, dtype=dtype
        )  # 1 x 1 x 1 x 3 x 3

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )  # 1 x 4 x 4

        tensor = tensor.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1, 1)  # 5 x 3 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 4 x 4

        self.assert_close(f(tensor), expected)
        self.assert_close(f.transform_matrix, expected_transform)