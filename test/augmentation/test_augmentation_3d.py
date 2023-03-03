import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.augmentation import (
    CenterCrop3D,
    RandomAffine3D,
    RandomCrop,
    RandomCrop3D,
    RandomDepthicalFlip3D,
    RandomEqualize3D,
    RandomHorizontalFlip3D,
    RandomRotation3D,
    RandomVerticalFlip3D,
)
from kornia.augmentation.container.augment import AugmentationSequential
from kornia.testing import assert_close


class TestRandomHorizontalFlip3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomHorizontalFlip3D(0.5)
        repr = "RandomHorizontalFlip3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        assert str(f) == repr

    def test_random_hflip(self, device):
        f = RandomHorizontalFlip3D(p=1.0, keepdim=True)
        f1 = RandomHorizontalFlip3D(p=0.0, keepdim=True)

        input_tensor = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                ]
            ],
            device=device,
        )  # 1 x 2 x 3 x 4

        expected = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]],
                ]
            ],
            device=device,
        )  # 1 x 2 x 3 x 4

        expected_transform = torch.tensor(
            [[[-1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], device=device
        )  # 1 x 4 x 4

        identity = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], device=device
        )  # 1 x 4 x 4

        assert_close(f(input_tensor), expected)
        assert_close(f.transform_matrix, expected_transform)
        assert_close(f1(input_tensor), input_tensor)
        assert_close(f1.transform_matrix, identity)

    def test_batch_random_hflip(self, device):
        f = RandomHorizontalFlip3D(p=1.0)

        input_tensor = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]]])  # 1 x 1 x 1 x 3 x 3
        input_tensor = input_tensor.to(device)

        expected = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]]])  # 1 x 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor(
            [[[-1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
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

        assert_close(f(input_tensor), expected)
        assert_close(f.transform_matrix, expected_transform)

    def test_same_on_batch(self, device):
        f = RandomHorizontalFlip3D(p=0.5, same_on_batch=True)
        input_tensor = torch.eye(3, device=device).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1, 1)
        res = f(input_tensor)
        assert_close(res[0], res[1])

    def test_sequential(self, device):
        f = AugmentationSequential(RandomHorizontalFlip3D(p=1.0), RandomHorizontalFlip3D(p=1.0))

        input_tensor = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]]])  # 1 x 1 x 1 x 3 x 3
        input_tensor = input_tensor.to(device)

        expected_transform = torch.tensor(
            [[[-1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        )  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform
        expected_transform_1 = expected_transform_1.to(device)

        assert_close(f(input_tensor), input_tensor)
        assert_close(f.transform_matrix, expected_transform_1)

    def test_gradcheck(self, device):
        input_tensor = torch.rand((1, 3, 3)).to(device)  # 3 x 3
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(RandomHorizontalFlip3D(p=1.0), (input_tensor,), raise_exception=True, fast_mode=True)


class TestRandomVerticalFlip3D:
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

        assert_close(f(input_tensor), expected)
        assert_close(f.transform_matrix, expected_transform)
        assert_close(f1(input_tensor), input_tensor)
        assert_close(f1.transform_matrix, identity)

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

        assert_close(f(input_tensor), expected)
        assert_close(f.transform_matrix, expected_transform)
        assert_close(f1(input_tensor), input_tensor)
        assert_close(f1.transform_matrix, identity)

    def test_same_on_batch(self, device):
        f = RandomVerticalFlip3D(p=0.5, same_on_batch=True)
        input_tensor = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1, 1)
        res = f(input_tensor)
        assert_close(res[0], res[1])

    def test_sequential(self, device):
        f = AugmentationSequential(RandomVerticalFlip3D(p=1.0), RandomVerticalFlip3D(p=1.0))

        input_tensor = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]]])  # 1 x 1 x 1 x 4 x 4
        input_tensor = input_tensor.to(device)

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        )  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        assert_close(f(input_tensor), input_tensor)
        assert_close(f.transform_matrix, expected_transform_1)

    def test_gradcheck(self, device):
        input_tensor = torch.rand((1, 3, 3)).to(device)  # 4 x 4
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(RandomVerticalFlip3D(p=1.0), (input_tensor,), raise_exception=True, fast_mode=True)


class TestRandomDepthicalFlip3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomDepthicalFlip3D(0.5)
        repr = "RandomDepthicalFlip3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        assert str(f) == repr

    def test_random_dflip(self, device, dtype):
        f = RandomDepthicalFlip3D(p=1.0)
        f1 = RandomDepthicalFlip3D(p=0.0)

        input_tensor = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 2 x 3 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 2 x 3 x 4

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )  # 4 x 4

        identity = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )  # 4 x 4

        assert_close(f(input_tensor), expected)
        assert_close(f.transform_matrix, expected_transform)
        assert_close(f1(input_tensor), input_tensor)
        assert_close(f1.transform_matrix, identity)

    def test_batch_random_dflip(self, device):
        f = RandomDepthicalFlip3D(p=1.0)
        f1 = RandomDepthicalFlip3D(p=0.0)

        input_tensor = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
            ]
        )  # 2 x 3 x 4

        input_tensor = input_tensor.to(device)

        expected = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ]
        )  # 2 x 3 x 4
        expected = expected.to(device)

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
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

        assert_close(f(input_tensor), expected)
        assert_close(f.transform_matrix, expected_transform)
        assert_close(f1(input_tensor), input_tensor)
        assert_close(f1.transform_matrix, identity)

    def test_same_on_batch(self, device):
        f = RandomDepthicalFlip3D(p=0.5, same_on_batch=True)
        input_tensor = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 2, 1, 1)
        res = f(input_tensor)
        assert_close(res[0], res[1])

    def test_sequential(self, device):
        f = AugmentationSequential(RandomDepthicalFlip3D(p=1.0), RandomDepthicalFlip3D(p=1.0))

        input_tensor = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
                    ]
                ]
            ]
        )  # 2 x 3 x 4
        input_tensor = input_tensor.to(device)

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
        )  # 1 x 4 x 4
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        assert_close(f(input_tensor), input_tensor)
        assert_close(f.transform_matrix, expected_transform_1)

    def test_gradcheck(self, device):
        input_tensor = torch.rand((1, 3, 3)).to(device)  # 4 x 4
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(RandomDepthicalFlip3D(p=1.0), (input_tensor,), raise_exception=True, fast_mode=True)


class TestRandomRotation3D:
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
        assert_close(out, expected, rtol=1e-6, atol=1e-4)
        assert_close(f.transform_matrix, expected_transform, rtol=1e-6, atol=1e-4)

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
        assert_close(out, expected, rtol=1e-6, atol=1e-4)
        assert_close(f.transform_matrix, expected_transform, rtol=1e-6, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation3D(degrees=40, same_on_batch=True)
        input_tensor = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 6, 1, 1)
        res = f(input_tensor)
        assert_close(res[0], res[1])

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
        assert_close(out, expected, rtol=1e-6, atol=1e-4)
        assert_close(f.transform_matrix, expected_transform, rtol=1e-6, atol=1e-4)

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility

        input_tensor = torch.rand((3, 3, 3)).to(device)  # 3 x 3 x 3
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(RandomRotation3D(degrees=(15.0, 15.0), p=1.0), (input_tensor,), raise_exception=True, fast_mode=True)


class TestRandomCrop3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomCrop3D(size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, p=1.0)
        repr = (
            "RandomCrop3D(crop_size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, "
            "padding_mode=constant, resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, "
            "return_transform=None)"
        )
        assert str(f) == repr

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_no_padding(self, batch_size, device, dtype):
        torch.manual_seed(42)
        input_tensor = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10, 11, 12, 13, 14],
                            [15, 16, 17, 18, 19],
                            [20, 21, 22, 23, 24],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 5, 1, 1)
        f = RandomCrop3D(size=(2, 3, 4), padding=None, align_corners=True, p=1.0)
        out = f(input_tensor)
        if batch_size == 1:
            expected = torch.tensor(
                [[[[[11, 12, 13, 14], [16, 17, 18, 19], [21, 22, 23, 24]]]]], device=device, dtype=dtype
            ).repeat(batch_size, 1, 2, 1, 1)
        if batch_size == 2:
            expected = torch.tensor(
                [
                    [
                        [
                            [
                                [6.0000, 7.0000, 8.0000, 9.0000],
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                            ],
                            [
                                [6.0000, 7.0000, 8.0000, 9.0000],
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                                [21.0000, 22.0000, 23.0000, 24.0000],
                            ],
                            [
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                                [21.0000, 22.0000, 23.0000, 24.0000],
                            ],
                        ]
                    ],
                ],
                device=device,
                dtype=dtype,
            )

        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomCrop3D(size=(2, 3, 4), padding=None, align_corners=True, p=1.0, same_on_batch=True)
        input_tensor = (
            torch.eye(6, device=device, dtype=dtype)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .repeat(2, 3, 5, 1, 1)
        )
        res = f(input_tensor)
        assert_close(res[0], res[1])

    @pytest.mark.parametrize("padding", [1, (1, 1, 1), (1, 1, 1, 1, 1, 1)])
    def test_padding_batch(self, padding, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        input_tensor = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 3, 1, 1
        )
        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 1.0, 2.0, 10.0], [3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0]],
                        [[0.0, 1.0, 2.0, 10.0], [3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0]],
                    ]
                ],
                [
                    [
                        [[3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0], [10, 10, 10, 10.0]],
                        [[3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0], [10, 10, 10, 10.0]],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )
        f = RandomCrop3D(size=(2, 3, 4), fill=10.0, padding=padding, align_corners=True, p=1.0)
        out = f(input_tensor)

        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(42)
        input_tensor = torch.tensor([[[0.0, 1.0, 2.0]]], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [
                    [
                        [[9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]],
                        [[0.0, 1.0, 2.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        rc = RandomCrop3D(size=(2, 3, 4), pad_if_needed=True, fill=9, align_corners=True, p=1.0)
        out = rc(input_tensor)

        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        input_tensor = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(RandomCrop3D(size=(3, 3, 3), p=1.0), (input_tensor,), raise_exception=True, fast_mode=True)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.0).forward
        op_script = torch.jit.script(op)
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        actual = op_script(img)
        expected = kornia.geometry.transform.center_crop3d(img)
        assert_close(actual, expected)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit_trace(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.0).forward
        op_script = torch.jit.script(op)
        # 1. Trace op
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        op_trace = torch.jit.trace(op_script, (img,))

        # 2. Generate new input
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        # 3. Evaluate
        actual = op_trace(img)
        expected = op(img)
        assert_close(actual, expected)


class TestCenterCrop3D:
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

    def test_gradcheck(self, device, dtype):
        input_tensor = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(CenterCrop3D(3), (input_tensor,), raise_exception=True, fast_mode=True)


class TestRandomEqualize3D:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = RandomEqualize3D(p=0.5)
        repr = "RandomEqualize3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        assert str(f) == repr

    def test_random_equalize(self, device, dtype):
        f = RandomEqualize3D(p=1.0)
        f1 = RandomEqualize3D(p=0.0)

        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor(
            [0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000],
            device=device,
            dtype=dtype,
        )
        expected = self.build_input(channels, depth, height, width, bs=1, row=row_expected, device=device, dtype=dtype)

        identity = kornia.eye_like(4, expected)

        assert_close(f(inputs3d), expected, rtol=1e-4, atol=1e-4)
        assert_close(f.transform_matrix, identity, rtol=1e-4, atol=1e-4)
        assert_close(f1(inputs3d), inputs3d, rtol=1e-4, atol=1e-4)
        assert_close(f1.transform_matrix, identity, rtol=1e-4, atol=1e-4)

    def test_batch_random_equalize(self, device, dtype):
        f = RandomEqualize3D(p=1.0)
        f1 = RandomEqualize3D(p=0.0)

        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor([0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000])
        expected = self.build_input(channels, depth, height, width, bs, row=row_expected, device=device, dtype=dtype)

        identity = kornia.eye_like(4, expected)  # 2 x 4 x 4

        assert_close(f(inputs3d), expected, rtol=1e-4, atol=1e-4)
        assert_close(f.transform_matrix, identity, rtol=1e-4, atol=1e-4)
        assert_close(f1(inputs3d), inputs3d, rtol=1e-4, atol=1e-4)
        assert_close(f1.transform_matrix, identity, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomEqualize3D(p=0.5, same_on_batch=True)
        input_tensor = torch.eye(4, device=device, dtype=dtype)
        input_tensor = input_tensor.unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 2, 1, 1)
        res = f(input_tensor)
        assert_close(res[0], res[1])

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility

        inputs3d = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3 x 3
        inputs3d = utils.tensor_to_gradcheck_var(inputs3d)  # to var
        assert gradcheck(RandomEqualize3D(p=0.5), (inputs3d,), raise_exception=True, fast_mode=True)

    @staticmethod
    def build_input(channels, depth, height, width, bs=1, row=None, device='cpu', dtype=torch.float32):
        if row is None:
            row = torch.arange(width, device=device, dtype=dtype) / float(width)

        channel = torch.stack([row] * height)
        image = torch.stack([channel] * channels)
        image3d = torch.stack([image] * depth).transpose(0, 1)
        batch = torch.stack([image3d] * bs)

        return batch.to(device, dtype)


class TestRandomAffine3D:
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

        assert_close(f(tensor), expected)
        assert_close(f.transform_matrix, expected_transform)
