import pytest
import torch

from kornia.augmentation import RandomRotation3D
from kornia.augmentation.container.augment import AugmentationSequential

from testing.base import BaseTester


class TestRandomRotation3D(BaseTester):
    torch.manual_seed(0)  # for random reproductibility

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomRotation3D(degrees=45.5)
        repr = (
            """RandomRotation3D(degrees=tensor([[-45.5000, 45.5000],
        [-45.5000, 45.5000],
        [-45.5000, 45.5000]]), resample=BILINEAR, align_corners=False, p=0.5, """
            """p_batch=1.0, same_on_batch=False, return_transform=None)"""
        )
        assert str(f) == repr

    def test_random_rotation(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation3D(degrees=45.0)

        input_tensor = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
            ],
            device=device,
            dtype=dtype,
        )  # 3 x 4 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0000, 0.0000, 0.6810, 0.5250],
                            [0.5052, 0.0000, 0.0000, 0.0613],
                            [0.1159, 0.1072, 0.5324, 0.0870],
                            [0.0000, 0.0000, 0.1927, 0.0000],
                        ],
                        [
                            [0.0000, 0.1683, 0.6963, 0.1131],
                            [0.0566, 0.0000, 0.5215, 0.2796],
                            [0.0694, 0.6039, 1.4519, 1.1240],
                            [0.0000, 0.1325, 0.1542, 0.2510],
                        ],
                        [
                            [0.0000, 0.2054, 0.0000, 0.0000],
                            [0.0026, 0.6088, 0.7358, 0.2319],
                            [0.1261, 1.0830, 1.3687, 1.4940],
                            [0.0000, 0.0416, 0.2012, 0.3124],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_transform = torch.tensor(
            [
                [
                    [0.6523, 0.3666, -0.6635, 0.6352],
                    [-0.6185, 0.7634, -0.1862, 1.4689],
                    [0.4382, 0.5318, 0.7247, -1.1797],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        out = f(input_tensor)
        self.assert_close(out, expected, rtol=1e-6, atol=1e-4)
        self.assert_close(f.transform_matrix, expected_transform, rtol=1e-6, atol=1e-4)

    def test_batch_random_rotation(self, device, dtype):
        torch.manual_seed(24)  # for random reproductibility

        f = RandomRotation3D(degrees=45.0)

        input_tensor = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                    [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                    [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 4 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [1.0000, 0.0000, 0.0000, 2.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 1.0000, 2.0000, 0.0000],
                            [0.0000, 0.0000, 1.0000, 2.0000],
                        ],
                        [
                            [1.0000, 0.0000, 0.0000, 2.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 1.0000, 2.0000, 0.0000],
                            [0.0000, 0.0000, 1.0000, 2.0000],
                        ],
                        [
                            [1.0000, 0.0000, 0.0000, 2.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 1.0000, 2.0000, 0.0000],
                            [0.0000, 0.0000, 1.0000, 2.0000],
                        ],
                    ]
                ],
                [
                    [
                        [
                            [0.0000, 0.0726, 0.0000, 0.0000],
                            [0.1038, 1.0134, 0.5566, 0.1519],
                            [0.0000, 1.0849, 1.1068, 0.0000],
                            [0.1242, 1.1065, 0.9681, 0.0000],
                        ],
                        [
                            [0.0000, 0.0047, 0.0166, 0.0000],
                            [0.0579, 0.4459, 0.0000, 0.4728],
                            [0.1864, 1.3349, 0.7530, 0.3251],
                            [0.1431, 1.2481, 0.4471, 0.0000],
                        ],
                        [
                            [0.0000, 0.4840, 0.2314, 0.0000],
                            [0.0000, 0.0328, 0.0000, 0.1434],
                            [0.1899, 0.5580, 0.0000, 0.9170],
                            [0.0000, 0.2042, 0.1571, 0.0855],
                        ],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )

        expected_transform = torch.tensor(
            [
                [
                    [1.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 1.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 1.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ],
                [
                    [0.7522, -0.6326, -0.1841, 1.5047],
                    [0.6029, 0.5482, 0.5796, -0.8063],
                    [-0.2657, -0.5470, 0.7938, 1.4252],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        input_tensor = input_tensor.repeat(2, 1, 1, 1, 1)  # 5 x 4 x 4 x 3

        out = f(input_tensor)
        self.assert_close(out, expected, rtol=1e-6, atol=1e-4)
        self.assert_close(f.transform_matrix, expected_transform, rtol=1e-6, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation3D(degrees=40, same_on_batch=True)
        input_tensor = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 6, 1, 1)
        res = f(input_tensor)
        self.assert_close(res[0], res[1])

    def test_sequential(self, device, dtype):
        torch.manual_seed(24)  # for random reproductibility

        f = AugmentationSequential(RandomRotation3D(torch.tensor([-45.0, 90])), RandomRotation3D(10.4))
        input_tensor = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
            ],
            device=device,
            dtype=dtype,
        )  # 3 x 4 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.3431, 0.1239, 0.0000, 1.0348],
                            [0.0000, 0.2035, 0.1139, 0.1770],
                            [0.0789, 0.9057, 1.7780, 0.0000],
                            [0.0000, 0.2286, 1.2498, 1.2643],
                        ],
                        [
                            [0.5460, 0.2131, 0.0000, 1.1453],
                            [0.0000, 0.0899, 0.0000, 0.4293],
                            [0.0797, 1.0193, 1.6677, 0.0000],
                            [0.0000, 0.2458, 1.2765, 1.0920],
                        ],
                        [
                            [0.6322, 0.2614, 0.0000, 0.9207],
                            [0.0000, 0.0037, 0.0000, 0.6551],
                            [0.0689, 0.9251, 1.3442, 0.0000],
                            [0.0000, 0.2449, 0.9856, 0.6862],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_transform = torch.tensor(
            [
                [
                    [0.9857, -0.1686, -0.0019, 0.2762],
                    [0.1668, 0.9739, 0.1538, -0.3650],
                    [-0.0241, -0.1520, 0.9881, 0.2760],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        out = f(input_tensor)
        self.assert_close(out, expected, rtol=1e-6, atol=1e-4)
        self.assert_close(f.transform_matrix, expected_transform, rtol=1e-6, atol=1e-4)

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility

        input_tensor = torch.rand((3, 3, 3), device=device, dtype=torch.float64)  # 3 x 3 x 3
        self.gradcheck(RandomRotation3D(degrees=(15.0, 15.0), p=1.0), (input_tensor,))