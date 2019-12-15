import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.augmentation import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRectangleErasing
from kornia.augmentation.random_erasing import get_random_rectangles_params, erase_rectangles


class TestRandomHorizontalFlip:

    def smoke_test(self):
        f = RandomHorizontalFlip(0.5)
        repr = "RandomHorizontalFlip(p=0.5, return_transform=False)"
        assert str(f) == repr

    def test_random_hflip(self):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip(p=0., return_transform=True)
        f2 = RandomHorizontalFlip(p=1.)
        f3 = RandomHorizontalFlip(p=0.)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[0., 0., 0.],
                                 [0., 0., 0.],
                                 [1., 1., 0.]])  # 3 x 3

        expected_transform = torch.tensor([[-1., 0., 3.],
                                           [0., 1., 0.],
                                           [0., 0., 1.]])  # 3 x 3

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()
        assert (f2(input) == expected).all()
        assert (f3(input) == input).all()

    def test_batch_random_hflip(self):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip(p=0.0, return_transform=True)

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3

        expected = torch.tensor([[[[0., 0., 0.],
                                   [0., 0., 0.],
                                   [1., 1., 0.]]]])  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor([[[-1., 0., 3.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()

    def test_sequential(self):

        f = nn.Sequential(
            RandomHorizontalFlip(1.0, return_transform=True),
            RandomHorizontalFlip(1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomHorizontalFlip(1.0, return_transform=True),
            RandomHorizontalFlip(1.0),
        )

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor([[[-1., 0., 3.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3

        expected_transform_1 = expected_transform @ expected_transform

        assert(f(input)[0] == input).all()
        assert(f(input)[1] == expected_transform_1).all()
        assert(f1(input)[0] == input).all()
        assert(f1(input)[1] == expected_transform).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.random_hflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 5., 5.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self):

        input = torch.rand((3, 3)).double()  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(RandomHorizontalFlip(p=1.), (input, ), raise_exception=True)


class TestRandomVerticalFlip:

    def smoke_test(self):
        f = RandomVerticalFlip(0.5)
        repr = "RandomVerticalFlip(p=0.5, return_transform=False)"
        assert str(f) == repr

    def test_random_vflip(self):

        f = RandomVerticalFlip(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip(p=0., return_transform=True)
        f2 = RandomVerticalFlip(p=1.)
        f3 = RandomVerticalFlip(p=0.)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3

        expected_transform = torch.tensor([[1., 0., 0.],
                                           [0., -1., 3.],
                                           [0., 0., 1.]])  # 3 x 3

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()
        assert (f2(input) == expected).all()
        assert (f3(input) == input).all()

    def test_batch_random_vflip(self):

        f = RandomVerticalFlip(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip(p=0.0, return_transform=True)

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3

        expected = torch.tensor([[[[0., 1., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]])  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 3.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()

    def test_sequential(self):

        f = nn.Sequential(
            RandomVerticalFlip(1.0, return_transform=True),
            RandomVerticalFlip(1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomVerticalFlip(1.0, return_transform=True),
            RandomVerticalFlip(1.0),
        )

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 3.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3

        expected_transform_1 = expected_transform @ expected_transform

        assert(f(input)[0] == input).all()
        assert(f(input)[1] == expected_transform_1).all()
        assert(f1(input)[0] == input).all()
        assert(f1(input)[1] == expected_transform).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.random_vflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [5., 5., 0.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self):

        input = torch.rand((3, 3)).double()  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(RandomVerticalFlip(p=1.), (input, ), raise_exception=True)


class TestColorJitter:

    def smoke_test(self):
        f = ColorJitter(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1)
        repr = "ColorJitter(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1, return_transform=False)"
        assert str(f) == repr

    def test_color_jitter(self):

        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(3, 5, 5)  # 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3).unsqueeze(0)  # 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_color_jitter_batch(self):
        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(2, 3, 5, 5)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_random_brightness(self):
        torch.manual_seed(42)
        f = ColorJitter(brightness=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]],

                                  [[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]],

                                  [[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]]],


                                 [[[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]],

                                  [[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]],

                                  [[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]]]])  # 1 x 1 x 3 x 3

        assert_allclose(f(input), expected)

    def test_random_brightness_tuple(self):
        torch.manual_seed(42)
        f = ColorJitter(brightness=(-0.2, 0.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]],

                                  [[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]],

                                  [[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]]],


                                 [[[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]],

                                  [[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]],

                                  [[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]]]])  # 1 x 1 x 3 x 3

        assert_allclose(f(input), expected)

    def test_random_contrast(self):
        torch.manual_seed(42)
        f = ColorJitter(contrast=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]],

                                  [[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]],

                                  [[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]]],


                                 [[[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]],

                                  [[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]],

                                  [[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]]]])

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def test_random_contrast_list(self):
        torch.manual_seed(42)
        f = ColorJitter(contrast=[0.8, 1.2])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]],

                                  [[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]],

                                  [[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]]],


                                 [[[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]],

                                  [[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]],

                                  [[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]]]])

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def test_random_saturation(self):
        torch.manual_seed(42)
        f = ColorJitter(saturation=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[1.8763e-01, 2.5842e-01, 3.3895e-01],
                                   [6.2921e-01, 5.0000e-01, 4.0000e-01],
                                   [7.0974e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 5.2921e-01, 6.0974e-01],
                                   [6.2921e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 1.6816e-01, 2.7790e-01]],

                                  [[6.3895e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 4.3895e-01, 5.4869e-01]]],


                                 [[[1.1921e-07, 1.2953e-01, 2.5302e-01],
                                   [5.6476e-01, 5.0000e-01, 4.0000e-01],
                                   [6.8825e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 4.6476e-01, 5.8825e-01],
                                   [5.6476e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 1.7781e-02, 1.0603e-01]],

                                  [[5.5556e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 3.5302e-01, 4.4127e-01]]]])

        assert_allclose(f(input), expected)

    def test_random_saturation_tensor(self):
        torch.manual_seed(42)
        f = ColorJitter(saturation=torch.tensor(0.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[1.8763e-01, 2.5842e-01, 3.3895e-01],
                                   [6.2921e-01, 5.0000e-01, 4.0000e-01],
                                   [7.0974e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 5.2921e-01, 6.0974e-01],
                                   [6.2921e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 1.6816e-01, 2.7790e-01]],

                                  [[6.3895e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 4.3895e-01, 5.4869e-01]]],


                                 [[[1.1921e-07, 1.2953e-01, 2.5302e-01],
                                   [5.6476e-01, 5.0000e-01, 4.0000e-01],
                                   [6.8825e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 4.6476e-01, 5.8825e-01],
                                   [5.6476e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 1.7781e-02, 1.0603e-01]],

                                  [[5.5556e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 3.5302e-01, 4.4127e-01]]]])

        assert_allclose(f(input), expected)

    def test_random_saturation_tuple(self):
        torch.manual_seed(42)
        f = ColorJitter(saturation=(0.8, 1.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[1.8763e-01, 2.5842e-01, 3.3895e-01],
                                   [6.2921e-01, 5.0000e-01, 4.0000e-01],
                                   [7.0974e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 5.2921e-01, 6.0974e-01],
                                   [6.2921e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 1.6816e-01, 2.7790e-01]],

                                  [[6.3895e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 4.3895e-01, 5.4869e-01]]],


                                 [[[1.1921e-07, 1.2953e-01, 2.5302e-01],
                                   [5.6476e-01, 5.0000e-01, 4.0000e-01],
                                   [6.8825e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 4.6476e-01, 5.8825e-01],
                                   [5.6476e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 1.7781e-02, 1.0603e-01]],

                                  [[5.5556e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 3.5302e-01, 4.4127e-01]]]])

        assert_allclose(f(input), expected)

    def test_random_hue(self):
        torch.manual_seed(42)
        f = ColorJitter(hue=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.1000, 0.2000, 0.3000],
                                   [0.6000, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.5251, 0.6167],
                                   [0.6126, 0.3000, 0.2000],
                                   [0.8000, 0.1000, 0.2000]],

                                  [[0.5623, 0.8000, 0.7000],
                                   [0.9000, 0.3084, 0.2084],
                                   [0.7958, 0.4293, 0.5335]]],

                                 [[[0.1000, 0.2000, 0.3000],
                                   [0.6116, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.4769, 0.5846],
                                   [0.6000, 0.3077, 0.2077],
                                   [0.7961, 0.1000, 0.2000]],

                                  [[0.6347, 0.8000, 0.7000],
                                   [0.9000, 0.3000, 0.2000],
                                   [0.8000, 0.3730, 0.4692]]]])

        assert_allclose(f(input), expected)

    def test_random_hue_list(self):
        torch.manual_seed(42)
        f = ColorJitter(hue=[-0.2, 0.2])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.1000, 0.2000, 0.3000],
                                   [0.6000, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.5251, 0.6167],
                                   [0.6126, 0.3000, 0.2000],
                                   [0.8000, 0.1000, 0.2000]],

                                  [[0.5623, 0.8000, 0.7000],
                                   [0.9000, 0.3084, 0.2084],
                                   [0.7958, 0.4293, 0.5335]]],


                                 [[[0.1000, 0.2000, 0.3000],
                                   [0.6116, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.4769, 0.5846],
                                   [0.6000, 0.3077, 0.2077],
                                   [0.7961, 0.1000, 0.2000]],

                                  [[0.6347, 0.8000, 0.7000],
                                   [0.9000, 0.3000, 0.2000],
                                   [0.8000, 0.3730, 0.4692]]]])

        assert_allclose(f(input), expected)

    def test_random_hue_tensor(self):
        torch.manual_seed(42)
        f = ColorJitter(hue=torch.tensor([-0.2, 0.2]))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.1000, 0.2000, 0.3000],
                                   [0.6000, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.5251, 0.6167],
                                   [0.6126, 0.3000, 0.2000],
                                   [0.8000, 0.1000, 0.2000]],

                                  [[0.5623, 0.8000, 0.7000],
                                   [0.9000, 0.3084, 0.2084],
                                   [0.7958, 0.4293, 0.5335]]],


                                 [[[0.1000, 0.2000, 0.3000],
                                   [0.6116, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.4769, 0.5846],
                                   [0.6000, 0.3077, 0.2077],
                                   [0.7961, 0.1000, 0.2000]],

                                  [[0.6347, 0.8000, 0.7000],
                                   [0.9000, 0.3000, 0.2000],
                                   [0.8000, 0.3730, 0.4692]]]])

        assert_allclose(f(input), expected)

    def test_sequential(self):

        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(3, 5, 5)  # 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3).unsqueeze(0)  # 3 x 3

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform)

    def test_color_jitter_batch(self):
        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(2, 3, 5, 5)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform)

    def test_gradcheck(self):

        input = torch.rand((3, 5, 5)).double()  # 3 x 3

        input = utils.tensor_to_gradcheck_var(input)  # to var

        assert gradcheck(kornia.color_jitter, (input, ), raise_exception=True)


class TestRectangleRandomErasing:
    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_random_rectangle_erasing_shape(
            self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomRectangleErasing(erase_scale_range, aspect_ratio_range)
        assert rand_rec(input).shape == batch_shape

    def test_rectangle_erasing1(self):
        inputs = torch.ones(1, 1, 10, 10)
        rect_params = (
            torch.tensor([5]), torch.tensor([5]),
            torch.tensor([5]), torch.tensor([5])
        )
        expected = torch.tensor([[[
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]
        ]]])
        assert_allclose(erase_rectangles(inputs, rect_params), expected)

    def test_rectangle_erasing2(self):
        inputs = torch.ones(3, 3, 3, 3)
        rect_params = (
            torch.tensor([3, 2, 1]), torch.tensor([3, 2, 1]),
            torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])
        )
        expected = torch.tensor(
            [[[[0., 0., 0.],
               [0., 0., 0.],
                [0., 0., 0.]],

                [[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.]],

                [[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.]]],

                [[[1., 1., 1.],
                  [1., 0., 0.],
                    [1., 0., 0.]],

                 [[1., 1., 1.],
                  [1., 0., 0.],
                    [1., 0., 0.]],

                 [[1., 1., 1.],
                  [1., 0., 0.],
                    [1., 0., 0.]]],

                [[[1., 1., 1.],
                  [1., 1., 1.],
                    [1., 1., 0.]],

                 [[1., 1., 1.],
                  [1., 1., 1.],
                    [1., 1., 0.]],

                 [[1., 1., 1.],
                  [1., 1., 1.],
                    [1., 1., 0.]]]]
        )

        assert_allclose(erase_rectangles(inputs, rect_params), expected)

    def test_gradcheck(self):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        erase_scale_range = (.2, .4)
        aspect_ratio_range = (.3, .5)
        rect_params = get_random_rectangles_params(
            (2,), 11, 7, erase_scale_range, aspect_ratio_range
        )

        # evaluate function gradient
        input = torch.rand(batch_shape, dtype=torch.double)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            erase_rectangles,
            (input, rect_params),
            raise_exception=True,
        )

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(img):
            return kornia.augmentation.random_rectangle_erase(img, (.2, .4), (.3, .5))

        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        expected = RandomRectangleErasing(
            (.2, .4), (.3, .5)
        )(img)
        actual = op_script(img)
        assert_allclose(actual, expected)
