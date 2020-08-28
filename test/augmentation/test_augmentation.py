from unittest.mock import patch
from typing import Union, Tuple

import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.constants import pi
from kornia.augmentation import AugmentationBase, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, \
    RandomErasing, RandomGrayscale, RandomRotation, RandomCrop, RandomResizedCrop, RandomMotionBlur


class TestAugmentationBase:

    def test_forward(self, device, dtype):
        torch.manual_seed(42)
        input = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        input_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        expected_output = torch.rand((2, 3, 4, 5), device=device, dtype=dtype)
        expected_transform = torch.rand((2, 3, 3), device=device, dtype=dtype)
        augmentation = AugmentationBase(return_transform=False)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, \
                patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters, \
                patch.object(augmentation, "compute_transformation", autospec=True) as compute_transformation:

            # Calling the augmentation with a single tensor shall return the expected tensor using the generated params.
            params = {"foo": 0}
            generate_parameters.return_value = params
            apply_transform.return_value = expected_output
            compute_transformation.return_value = expected_transform
            output = augmentation(input)
            apply_transform.assert_called_once_with(input, params)
            assert output is expected_output

            # Calling the augmentation with a tensor and set return_transform shall
            # return the expected tensor and transformation.
            output, transformation = augmentation(input, return_transform=True)
            assert output is expected_output
            assert transformation is expected_transform

            # Calling the augmentation with a tensor and params shall return the expected tensor using the given params.
            params = {"bar": 1}
            apply_transform.reset_mock()
            generate_parameters.return_value = None
            output = augmentation(input, params=params)
            apply_transform.assert_called_once_with(input, params)
            assert output is expected_output

            # Calling the augmentation with a tensor,a transformation and set
            # return_transform shall return the expected tensor and the proper
            # transformation matrix.
            expected_final_transformation = expected_transform @ input_transform
            output, transformation = augmentation((input, input_transform), return_transform=True)
            assert output is expected_output
            assert torch.allclose(expected_final_transformation, transformation)
            assert transformation.shape[0] == input.shape[0]

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(42)

        input = torch.rand((1, 1, 3, 3), device=device, dtype=dtype)
        output = torch.rand((1, 1, 3, 3), device=device, dtype=dtype)
        input_transform = torch.rand((1, 3, 3), device=device, dtype=dtype)
        other_transform = torch.rand((1, 3, 3), device=device, dtype=dtype)

        input = utils.tensor_to_gradcheck_var(input)  # to var
        input_transform = utils.tensor_to_gradcheck_var(input_transform)  # to var
        output = utils.tensor_to_gradcheck_var(output)  # to var
        other_transform = utils.tensor_to_gradcheck_var(other_transform)  # to var

        augmentation = AugmentationBase(return_transform=True)

        with patch.object(augmentation, "apply_transform", autospec=True) as apply_transform, \
                patch.object(augmentation, "generate_parameters", autospec=True) as generate_parameters, \
                patch.object(augmentation, "compute_transformation", autospec=True) as compute_transformation:

            apply_transform.return_value = output
            compute_transformation.return_value = other_transform

            assert gradcheck(augmentation, ((input, input_transform)), raise_exception=True)


class TestRandomHorizontalFlip:

    def smoke_test(self, device):
        f = RandomHorizontalFlip(0.5)
        repr = "RandomHorizontalFlip(p=0.5, return_transform=False)"
        assert str(f) == repr

    def test_random_hflip(self, device):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip(p=0., return_transform=True)
        f2 = RandomHorizontalFlip(p=1.)
        f3 = RandomHorizontalFlip(p=0.)

        input = torch.tensor([[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 1., 2.]])  # 3 x 4

        input = input.to(device)

        expected = torch.tensor([[0., 0., 0., 0.],
                                 [0., 0., 0., 0.],
                                 [2., 1., 0., 0.]])  # 3 x 4

        expected = expected.to(device)

        expected_transform = torch.tensor([[-1., 0., 4.],
                                           [0., 1., 0.],
                                           [0., 0., 1.]])  # 3 x 3

        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3
        identity = identity.to(device)

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()
        assert (f2(input) == expected).all()
        assert (f3(input) == input).all()

    def test_batch_random_hflip(self, device):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip(p=0.0, return_transform=True)

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected = torch.tensor([[[[0., 0., 0.],
                                   [0., 0., 0.],
                                   [1., 1., 0.]]]])  # 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[[-1., 0., 3.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3
        identity = identity.to(device)

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()

    def test_same_on_batch(self, device):
        f = RandomHorizontalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device):

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
        input = input.to(device)

        expected_transform = torch.tensor([[[-1., 0., 3.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform
        expected_transform_1 = expected_transform_1.to(device)

        assert(f(input)[0] == input).all()
        assert(f(input)[1] == expected_transform_1).all()
        assert(f1(input)[0] == input).all()
        assert(f1(input)[1] == expected_transform).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

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

    def test_gradcheck(self, device):
        input = torch.rand((3, 3)).to(device)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomHorizontalFlip(p=1.), (input, ), raise_exception=True)


class TestRandomVerticalFlip:

    def smoke_test(self, device):
        f = RandomVerticalFlip(0.5)
        repr = "RandomVerticalFlip(p=0.5, return_transform=False)"
        assert str(f) == repr

    def test_random_vflip(self, device):

        f = RandomVerticalFlip(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip(p=0., return_transform=True)
        f2 = RandomVerticalFlip(p=1.)
        f3 = RandomVerticalFlip(p=0.)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3
        input = input.to(device)

        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[1., 0., 0.],
                                           [0., -1., 3.],
                                           [0., 0., 1.]])  # 3 x 3
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3
        identity = identity.to(device)

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)
        assert_allclose(f2(input), expected)
        assert_allclose(f3(input), input)

    def test_batch_random_vflip(self, device):

        f = RandomVerticalFlip(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip(p=0.0, return_transform=True)

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected = torch.tensor([[[[0., 1., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]])  # 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 3.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3
        identity = identity.to(device)

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)

    def test_same_on_batch(self, device):
        f = RandomVerticalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device):

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
        input = input.to(device)

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 3.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        assert_allclose(f(input)[0], input.squeeze())
        assert_allclose(f(input)[1], expected_transform_1)
        assert_allclose(f1(input)[0], input.squeeze())
        assert_allclose(f1(input)[1], expected_transform)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def test_gradcheck(self, device):
        input = torch.rand((3, 3)).to(device)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomVerticalFlip(p=1.), (input, ), raise_exception=True)


class TestColorJitter:

    def smoke_test(self, device):
        f = ColorJitter(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1)
        repr = "ColorJitter(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1, return_transform=False)"
        assert str(f) == repr

    def test_color_jitter(self, device):

        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(3, 5, 5).to(device)  # 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3).unsqueeze(0).to(device)  # 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_color_jitter_batch(self, device):
        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(2, 3, 5, 5).to(device)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3).unsqueeze(0).expand((2, 3, 3)).to(device)  # 2 x 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_same_on_batch(self, device):
        f = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_random_brightness(self, device):
        torch.manual_seed(42)
        f = ColorJitter(brightness=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3
        input = input.to(device)

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
        expected = expected.to(device)

        assert_allclose(f(input), expected)

    def test_random_brightness_tuple(self, device):
        torch.manual_seed(42)
        f = ColorJitter(brightness=(0.8, 1.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3
        input = input.to(device)

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
        expected = expected.to(device)

        assert_allclose(f(input), expected)

    def test_random_contrast(self, device):
        torch.manual_seed(42)
        f = ColorJitter(contrast=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3
        input = input.to(device)

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
        expected = expected.to(device)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def test_random_contrast_list(self, device):
        torch.manual_seed(42)
        f = ColorJitter(contrast=[0.8, 1.2])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1).to(device)  # 2 x 3 x 3

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
        expected = expected.to(device)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def test_random_saturation(self, device):
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
        input = input.repeat(2, 1, 1, 1).to(device)  # 2 x 3 x 3

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
        expected = expected.to(device)
        assert_allclose(f(input), expected)

    def test_random_saturation_tensor(self, device):
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
        input = input.repeat(2, 1, 1, 1).to(device)  # 2 x 3 x 3

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
        expected = expected.to(device)

        assert_allclose(f(input), expected)

    def test_random_saturation_tuple(self, device):
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
        input = input.repeat(2, 1, 1, 1).to(device)  # 2 x 3 x 3

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
        expected = expected.to(device)

        assert_allclose(f(input), expected)

    def test_random_hue(self, device):
        torch.manual_seed(42)
        f = ColorJitter(hue=0.1 / pi.item())

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1).to(device)  # 2 x 3 x 3

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
        expected = expected.to(device)

        assert_allclose(f(input), expected)

    def test_random_hue_list(self, device):
        torch.manual_seed(42)
        f = ColorJitter(hue=[-0.1 / pi, 0.1 / pi])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1).to(device)  # 2 x 3 x 3

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
        expected = expected.to(device)

        assert_allclose(f(input), expected)

    def test_random_hue_list_batch(self, device):
        torch.manual_seed(42)
        f = ColorJitter(hue=[-0.1 / pi.item(), 0.1 / pi.item()])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1).to(device)  # 2 x 3 x 3

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
        expected = expected.to(device)

        assert_allclose(f(input), expected)

    def test_sequential(self, device):

        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(3, 5, 5).to(device)  # 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3).unsqueeze(0)  # 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform)

    def test_color_jitter_batch_sequential(self, device):
        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(2, 3, 5, 5).to(device)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)

    def test_gradcheck(self, device):
        input = torch.rand((3, 5, 5)).to(device)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.color_jitter, (input, ), raise_exception=True)


class TestRectangleRandomErasing:
    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_random_rectangle_erasing_shape(
            self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(1.0, erase_scale_range, aspect_ratio_range)
        assert rand_rec(input).shape == batch_shape

    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_no_rectangle_erasing_shape(
            self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(0., erase_scale_range, aspect_ratio_range)
        assert rand_rec(input).equal(input)

    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("shape", [(3, 11, 7)])
    def test_same_on_batch(self, shape, erase_scale_range, aspect_ratio_range):
        f = RandomErasing(0.5, erase_scale_range, aspect_ratio_range, same_on_batch=True)
        input = torch.rand(shape).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        print(f._params)
        assert (res[0] == res[1]).all()

    def test_gradcheck(self, device):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        erase_scale_range = (.2, .4)
        aspect_ratio_range = (.3, .5)

        rand_rec = RandomErasing(1.0, erase_scale_range, aspect_ratio_range)
        rect_params = rand_rec.generate_parameters(batch_shape)

        # evaluate function gradient
        input = torch.rand(batch_shape).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            rand_rec,
            (input, rect_params),
            raise_exception=True,
        )

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(img):
            return kornia.augmentation.random_rectangle_erase(img, (.2, .4), (.3, .5))

        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        expected = RandomErasing(
            1.0, (.2, .4), (.3, .5)
        )(img)
        actual = op_script(img)
        assert_allclose(actual, expected)


class TestRandomGrayscale:

    def smoke_test(self, device):
        f = RandomGrayscale()
        repr = "RandomGrayscale(p=0.5, return_transform=False)"
        assert str(f) == repr

    def test_random_grayscale(self, device):

        f = RandomGrayscale(return_transform=True)

        input = torch.rand(3, 5, 5).to(device)  # 3 x 5 x 5

        expected_transform = torch.eye(3).unsqueeze(0)  # 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[1], expected_transform)

    def test_same_on_batch(self, device):
        f = RandomGrayscale(p=0.5, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_opencv_true(self, device):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]])
        data = data.to(device)

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor([[[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]])
        expected = expected.to(device)

        img_gray = kornia.random_grayscale(data, p=1.)
        assert_allclose(img_gray, expected)

    def test_opencv_false(self, device):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]])
        data = data.to(device)

        expected = data

        img_gray = kornia.random_grayscale(data, p=0.)
        assert_allclose(img_gray, expected)

    def test_opencv_true_batch(self, device):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]])
        data = data.to(device)
        data = data.unsqueeze(0).repeat(4, 1, 1, 1)

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor([[[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]])
        expected = expected.to(device)
        expected = expected.unsqueeze(0).repeat(4, 1, 1, 1)

        img_gray = kornia.random_grayscale(data, p=1.)
        assert_allclose(img_gray, expected)

    def test_opencv_false_batch(self, device):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]])
        data = data.to(device)
        data = data.unsqueeze(0).repeat(4, 1, 1, 1)

        expected = data

        img_gray = kornia.random_grayscale(data, p=0.)
        assert_allclose(img_gray, expected)

    def test_random_grayscale_sequential_batch(self, device):
        f = nn.Sequential(
            RandomGrayscale(p=0., return_transform=True),
            RandomGrayscale(p=0., return_transform=True),
        )

        input = torch.rand(2, 3, 5, 5).to(device)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)

    def test_gradcheck(self, device):
        input = torch.rand((3, 5, 5)).to(device)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.random_grayscale, (input, 0.), raise_exception=True)
        assert gradcheck(kornia.random_grayscale, (input, 1.), raise_exception=True)


class TestCenterCrop:

    def test_no_transform(self, device):
        inp = torch.rand(1, 2, 4, 4).to(device)
        out = kornia.augmentation.CenterCrop(2)(inp)
        assert out.shape == (1, 2, 2, 2)

    def test_transform(self, device):
        inp = torch.rand(1, 2, 5, 4).to(device)
        out = kornia.augmentation.CenterCrop(2, return_transform=True)(inp)
        assert len(out) == 2
        assert out[0].shape == (1, 2, 2, 2)
        assert out[1].shape == (1, 3, 3)

    def test_no_transform_tuple(self, device):
        inp = torch.rand(1, 2, 5, 4).to(device)
        out = kornia.augmentation.CenterCrop((3, 4))(inp)
        assert out.shape == (1, 2, 3, 4)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.CenterCrop(3), (input,), raise_exception=True)


class TestRandomRotation:

    torch.manual_seed(0)  # for random reproductibility

    def smoke_test(self, device):
        f = RandomRotation(degrees=45.5)
        repr = "RandomHorizontalFlip(degrees=45.5, return_transform=False)"
        assert str(f) == repr

    def test_random_rotation(self, device):
        # This is included in doctest
        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, return_transform=True)
        f1 = RandomRotation(degrees=45.0)

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]])  # 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[0.9824, 0.0088, 0.0000, 1.9649],
                                  [0.0000, 0.0029, 0.0000, 0.0176],
                                  [0.0029, 1.0000, 1.9883, 0.0000],
                                  [0.0000, 0.0088, 1.0117, 1.9649]]])  # 1 x 4 x 4
        expected = expected.to(device)

        expected_transform = torch.tensor([[[1.0000, -0.0059, 0.0088],
                                            [0.0059, 1.0000, -0.0088],
                                            [0.0000, 0.0000, 1.0000]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_2 = torch.tensor([[0.1322, 0.0000, 0.7570, 0.2644],
                                   [0.3785, 0.0000, 0.4166, 0.0000],
                                   [0.0000, 0.6309, 1.5910, 1.2371],
                                   [0.0000, 0.1444, 0.3177, 0.6499]])  # 1 x 4 x 4
        expected_2 = expected_2.to(device)

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
        assert_allclose(f1(input), expected_2, rtol=1e-6, atol=1e-4)

    def test_batch_random_rotation(self, device):

        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, return_transform=True)

        input = torch.tensor([[[[1., 0., 0., 2.],
                                [0., 0., 0., 0.],
                                [0., 1., 2., 0.],
                                [0., 0., 1., 2.]]]])  # 1 x 1 x 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[[0.9824, 0.0088, 0.0000, 1.9649],
                                   [0.0000, 0.0029, 0.0000, 0.0176],
                                   [0.0029, 1.0000, 1.9883, 0.0000],
                                   [0.0000, 0.0088, 1.0117, 1.9649]]],
                                 [[[0.1322, 0.0000, 0.7570, 0.2644],
                                   [0.3785, 0.0000, 0.4166, 0.0000],
                                   [0.0000, 0.6309, 1.5910, 1.2371],
                                   [0.0000, 0.1444, 0.3177, 0.6499]]]])  # 2 x 1 x 4 x 4
        expected = expected.to(device)

        expected_transform = torch.tensor([[[1.0000, -0.0059, 0.0088],
                                            [0.0059, 1.0000, -0.0088],
                                            [0.0000, 0.0000, 1.0000]],

                                           [[0.9125, 0.4090, -0.4823],
                                            [-0.4090, 0.9125, 0.7446],
                                            [0.0000, 0.0000, 1.0000]]])  # 2 x 3 x 3
        expected_transform = expected_transform.to(device)

        input = input.repeat(2, 1, 1, 1)  # 5 x 3 x 3 x 3

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)

    def test_same_on_batch(self, device):
        f = RandomRotation(degrees=40, same_on_batch=True)
        input = torch.eye(6).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device):

        torch.manual_seed(0)  # for random reproductibility

        f = nn.Sequential(
            RandomRotation(torch.tensor([-45.0, 90]), return_transform=True),
            RandomRotation(10.4, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomRotation(torch.tensor([-45.0, 90]), return_transform=True),
            RandomRotation(10.4),
        )

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]])  # 4 x 4
        input = input.to(device)

        expected = torch.tensor([[[0.1314, 0.1050, 0.6649, 0.2628],
                                  [0.3234, 0.0202, 0.4256, 0.1671],
                                  [0.0525, 0.5976, 1.5199, 1.1306],
                                  [0.0000, 0.1453, 0.3224, 0.5796]]])  # 1 x 4 x 4
        expected = expected.to(device)

        expected_transform = torch.tensor([[[0.8864, 0.4629, -0.5240],
                                            [-0.4629, 0.8864, 0.8647],
                                            [0.0000, 0.0000, 1.0000]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_transform_2 = torch.tensor([[[0.8381, -0.5455, 1.0610],
                                              [0.5455, 0.8381, -0.5754],
                                              [0.0000, 0.0000, 1.0000]]])  # 1 x 3 x 3
        expected_transform_2 = expected_transform_2.to(device)

        out, mat = f(input)
        _, mat_2 = f1(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
        assert_allclose(mat_2, expected_transform_2, rtol=1e-6, atol=1e-4)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):

        torch.manual_seed(0)  # for random reproductibility

        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return kornia.random_rotation(data, degrees=45.0)

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]])  # 4 x 4

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        expected = torch.tensor([[[0.0000, 0.2584, 0.0000],
                                  [2.9552, 5.0000, 0.2584],
                                  [1.6841, 0.4373, 0.0000]]])

        actual = op_trace(input)

        assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)

    def test_gradcheck(self, device):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3)).to(device)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomRotation(degrees=(15.0, 15.0)), (input, ), raise_exception=True)


class TestRandomCrop:
    def smoke_test(self, device):
        f = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False)
        repr = "RandomCrop(crop_size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False,\
            return_transform=False)"
        assert str(f) == repr

    def test_no_padding(self, device):
        torch.manual_seed(0)
        inp = torch.tensor([[[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]]).to(device)
        expected = torch.tensor([[[
            [3., 4., 5.],
            [6., 7., 8.]
        ]]]).to(device)
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_no_padding_batch(self, device):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).repeat(batch_size, 1, 1, 1).to(device)
        expected = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
        ]]).repeat(batch_size, 1, 1, 1).to(device)
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_same_on_batch(self, device):
        f = RandomCrop(size=(2, 3), padding=1, same_on_batch=True, align_corners=True)
        input = torch.eye(6).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_padding_batch_1(self, device):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).repeat(batch_size, 1, 1, 1).to(device)
        expected = torch.tensor([[[
            [0., 0., 0.],
            [0., 1., 2.]
        ]], [[
            [0., 0., 0.],
            [1., 2., 0.]
        ]]]).to(device)
        rc = RandomCrop(size=(2, 3), padding=1, align_corners=True)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_padding_batch_2(self, device):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).repeat(batch_size, 1, 1, 1).to(device)
        expected = torch.tensor([[[
            [0., 1., 2.],
            [3., 4., 5.]
        ]], [[
            [1., 2., 10.],
            [4., 5., 10.]
        ]]]).to(device)
        rc = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, align_corners=True)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_padding_batch_3(self, device):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).repeat(batch_size, 1, 1, 1).to(device)
        expected = torch.tensor([[[
            [8., 8., 8.],
            [8., 0., 1.]
        ]], [[
            [8., 8., 8.],
            [1., 2., 8.]
        ]]]).to(device)
        rc = RandomCrop(size=(2, 3), padding=(0, 1, 2, 3), fill=8, align_corners=True)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_pad_if_needed(self, device):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
        ]]).repeat(batch_size, 1, 1, 1).to(device)
        expected = torch.tensor([[
            [9., 9., 9.],
            [0., 1., 2.]
        ]]).repeat(batch_size, 1, 1, 1).to(device)
        rc = RandomCrop(size=(2, 3), pad_if_needed=True, fill=9, align_corners=True)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((3, 3, 3)).to(device)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomCrop(size=(3, 3)), (inp, ), raise_exception=True)


class TestRandomResizedCrop:
    def smoke_test(self, device):
        f = RandomResizedCrop(size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0))
        repr = "RandomResizedCrop(size=(2, 3), resize_to=(1., 1.), resize_to=(1., 1.)\
            , return_transform=False)"
        assert str(f) == repr

    def test_no_resize(self, device):
        torch.manual_seed(0)
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).to(device)

        expected = torch.tensor(
            [[[[5.3750, 5.8750, 4.5938],
               [6.3437, 6.7812, 5.2500]]]]).to(device)
        rrc = RandomResizedCrop(size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected)

    def test_same_on_batch(self, device):
        f = RandomResizedCrop(size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0), same_on_batch=True)
        input = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).repeat(2, 1, 1, 1).to(device)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_crop_scale_ratio(self, device):
        # This is included in doctest
        torch.manual_seed(0)
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).to(device)

        expected = torch.tensor(
            [[[[3.7500, 4.7500, 5.7500],
               [5.2500, 6.2500, 7.2500],
               [4.5000, 5.2500, 6.0000]]]]).to(device)
        rrc = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected)

    def test_crop_scale_ratio_batch(self, device):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]).repeat(batch_size, 1, 1, 1).to(device)

        expected = torch. tensor(
            [[[[0.0000, 0.7500, 1.5000],
               [0.7500, 1.7500, 2.7500],
               [2.2500, 3.2500, 4.2500]]],
             [[[3.7500, 4.7500, 5.7500],
               [5.2500, 6.2500, 7.2500],
               [4.5000, 5.2500, 6.0000]]]]).to(device)
        rrc = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected)

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 3)).to(device)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomResizedCrop(size=(3, 3), scale=(1., 1.), ratio=(1., 1.)), (inp, ), raise_exception=True)


class TestRandomMotionBlur:
    def test_smoke(self, device):
        f = RandomMotionBlur(kernel_size=(3, 5), angle=(10, 30), direction=0.5)
        repr = "RandomMotionBlur(kernel_size=(3, 5), angle=tensor([10, 30]), direction=tensor([-0.5000,  0.5000]), "\
            "border_type='constant', return_transform=False)"
        assert str(f) == repr

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 11, 7)).to(device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        # TODO: Gradcheck for param random gen failed. Suspect get_motion_kernel2d issue.
        params = {
            'ksize_factor': torch.tensor(31),
            'angle_factor': torch.tensor(30.),
            'direction_factor': torch.tensor(-0.5),
            'border_type': torch.tensor([0]),
        }
        assert gradcheck(RandomMotionBlur(
            kernel_size=3, angle=(10, 30), direction=(-0.5, 0.5)), (inp, params), raise_exception=True)
