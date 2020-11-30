from unittest.mock import patch
from typing import Union, Tuple, Dict, Any, Optional, Type

import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.constants import pi, Resample
from kornia.augmentation import (
    CenterCrop,
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomErasing,
    RandomEqualize,
    RandomGrayscale,
    RandomRotation,
    RandomCrop,
    RandomResizedCrop
)


from kornia.testing import BaseTester, default_with_one_parameter_changed, cartesian_product_of_parameters
from kornia.augmentation.base import AugmentationBase2D

# TODO same_on_batch tests?


@pytest.mark.usefixtures("device", "dtype")
class CommonTests(BaseTester):
    fixture_names = ("device", "dtype")

    ############################################################################################################
    # Attribute variables to set
    ############################################################################################################
    _augmentation_cls: Optional[Type[AugmentationBase2D]] = None
    _default_param_set: Dict["str", Any] = {}
    ############################################################################################################
    # Fixtures
    ############################################################################################################

    @pytest.fixture(autouse=True)
    def auto_injector_fixture(self, request):
        for fixture_name in self.fixture_names:
            setattr(self, fixture_name, request.getfixturevalue(fixture_name))

    @pytest.fixture(scope="class")
    def param_set(self, request):
        raise NotImplementedError("param_set must be overriden in subclasses")

    ############################################################################################################
    # Test cases
    ############################################################################################################
    def test_smoke(self, param_set):
        self._test_smoke_implementation(params=param_set)
        self._test_smoke_call_implementation(params=param_set)
        self._test_smoke_return_transform_implementation(params=param_set)

    @pytest.mark.parametrize("input_shape,expected_output_shape",
                             [((4, 5), (1, 1, 4, 5)), ((3, 4, 5), (1, 3, 4, 5)), ((2, 3, 4, 5), (2, 3, 4, 5))])
    def test_cardinality(self, input_shape, expected_output_shape):
        self._test_cardinality_implementation(
            input_shape=input_shape, expected_output_shape=expected_output_shape, params=self._default_param_set)

    def test_random_p_0(self):
        self._test_random_p_0_implementation(params=self._default_param_set)

    def test_random_p_0_return_transform(self):
        self._test_random_p_0_return_transform_implementation(params=self._default_param_set)

    def test_random_p_1(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_random_p_1_return_transform(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_inverse_coordinate_check(self):
        self._test_inverse_coordinate_check_implementation(params=self._default_param_set)

    def test_exception(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_batch(self):
        raise NotImplementedError("Implement a stupid routine.")

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_module(self):
        self._test_module_implementation(params=self._default_param_set)

    def test_gradcheck(self):
        self._test_gradcheck_implementation(params=self._default_param_set)

# TODO Implement
# test_batch
# test_batch_return_transform
# test_coordinate check
# test_jit
# test_gradcheck

    def _create_augmentation_from_params(self, **params):
        return self._augmentation_cls(**params)

    ############################################################################################################
    # Test case implementations
    ############################################################################################################

    def _test_smoke_implementation(self, params):
        assert issubclass(self._augmentation_cls,
                          AugmentationBase2D), f"{self._augmentation_cls} is not a subclass of AugmentationBase2D"

        # Can be instatiated
        augmentation = self._create_augmentation_from_params(**params, return_transform=False)
        assert issubclass(
            type(augmentation), AugmentationBase2D), f"{type(augmentation)} is not a subclass of AugmentationBase2D"

        # generate_parameters can be called and returns the correct amount of parameters
        batch_shape = (4, 3, 5, 6)
        generated_params = augmentation.generate_parameters(batch_shape)
        assert isinstance(generated_params, dict)

        # compute_transformation can be called and returns the correct shaped transformation matrix
        expected_transformation_shape = torch.Size((batch_shape[0], 3, 3))
        test_input = torch.ones(batch_shape, device=self.device, dtype=self.dtype)
        transformation = augmentation.compute_transformation(test_input, generated_params)
        assert transformation.shape == expected_transformation_shape

        # apply_transform can be called and returns the correct batch sized output
        output = augmentation.apply_transform(test_input, generated_params)
        assert output.shape[0] == batch_shape[0]

    def _test_smoke_call_implementation(self, params):
        batch_shape = (4, 3, 5, 6)
        expected_transformation_shape = torch.Size((batch_shape[0], 3, 3))
        test_input = torch.ones(batch_shape, device=self.device, dtype=self.dtype)
        augmentation = self._create_augmentation_from_params(**params, return_transform=False)
        generated_params = augmentation.generate_parameters(batch_shape)
        test_transform = torch.rand(expected_transformation_shape, device=self.device, dtype=self.dtype)

        output = augmentation(test_input, params=generated_params)
        assert output.shape[0] == batch_shape[0]

        output, transformation = augmentation(test_input, params=generated_params, return_transform=True)
        assert output.shape[0] == batch_shape[0]
        assert transformation.shape == expected_transformation_shape

        output, final_transformation = augmentation(
            (test_input, test_transform), params=generated_params, return_transform=True)
        assert output.shape[0] == batch_shape[0]
        assert final_transformation.shape == expected_transformation_shape
        assert_allclose(final_transformation, transformation @ test_transform)

        output, transformation = augmentation((test_input, test_transform), params=generated_params)
        assert output.shape[0] == batch_shape[0]
        assert transformation.shape == expected_transformation_shape
        assert (transformation == test_transform).all()

    def _test_smoke_return_transform_implementation(self, params):
        batch_shape = (4, 3, 5, 6)
        expected_transformation_shape = torch.Size((batch_shape[0], 3, 3))
        test_input = torch.ones(batch_shape, device=self.device, dtype=self.dtype)
        augmentation = self._create_augmentation_from_params(**params, return_transform=True)
        generated_params = augmentation.generate_parameters(batch_shape)
        test_transform = torch.rand(expected_transformation_shape, device=self.device, dtype=self.dtype)

        output, transformation = augmentation(test_input, params=generated_params)
        assert output.shape[0] == batch_shape[0]
        assert transformation.shape == expected_transformation_shape

        output, final_transformation = augmentation((test_input, test_transform), params=generated_params)
        assert output.shape[0] == batch_shape[0]
        assert final_transformation.shape == expected_transformation_shape
        assert_allclose(final_transformation, transformation @ test_transform)

        output, final_transformation = augmentation(
            (test_input, test_transform), params=generated_params, return_transform=True)
        assert output.shape[0] == batch_shape[0]
        assert final_transformation.shape == expected_transformation_shape
        assert_allclose(final_transformation, transformation @ test_transform)

    def _test_cardinality_implementation(self, input_shape, expected_output_shape, params):

        # p==0.0
        augmentation = self._create_augmentation_from_params(**params, p=0.0)
        test_input = torch.rand(input_shape, device=self.device, dtype=self.dtype)
        output = augmentation(test_input)
        assert len(output.shape) == 4
        assert output.shape == torch.Size((1,) * (4 - len(input_shape)) + tuple(input_shape))

        # p==1.0
        augmentation = self._create_augmentation_from_params(**params, p=1.0)
        test_input = torch.rand(input_shape, device=self.device, dtype=self.dtype)
        output = augmentation(test_input)
        assert len(output.shape) == 4
        assert output.shape == expected_output_shape

    def _test_random_p_0_implementation(self, params):
        augmentation = self._create_augmentation_from_params(**params, p=0.0, return_transform=False)
        expected_output_shape = torch.Size((2, 3, 4, 5))
        test_input = torch.rand((2, 3, 4, 5), device=self.device, dtype=self.dtype)
        output = augmentation(test_input)
        assert (output == test_input).all()

    def _test_random_p_0_return_transform_implementation(self, params):
        augmentation = self._create_augmentation_from_params(**params, p=0.0, return_transform=True)
        expected_output_shape = torch.Size((2, 3, 4, 5))
        expected_transformation_shape = torch.Size((2, 3, 3))
        test_input = torch.rand((2, 3, 4, 5), device=self.device, dtype=self.dtype)
        output, transformation = augmentation(test_input)

        assert (output == test_input).all()
        assert transformation.shape == expected_transformation_shape
        assert (transformation == kornia.eye_like(3, transformation)).all()

    def _test_random_p_1_implementation(self, input_tensor, expected_output, params):
        augmentation = self._create_augmentation_from_params(**params, p=1.0, return_transform=False)
        output = augmentation(input_tensor.to(self.device).to(self.dtype))

        # Output should match
        assert output.shape == expected_output.shape
        assert_allclose(output, expected_output.to(self.device).to(self.dtype), atol=1e-4, rtol=1e-4)

    def _test_random_p_1_return_transform_implementation(
            self, input_tensor, expected_output, expected_transformation, params):
        augmentation = self._create_augmentation_from_params(**params, p=1.0, return_transform=True)
        output, transformation = augmentation(input_tensor.to(self.device).to(self.dtype))
        # Output should match
        assert output.shape == expected_output.shape
        assert_allclose(output, expected_output.to(self.device).to(self.dtype), atol=1e-4, rtol=1e-4)
        # Transformation should match
        assert transformation.shape == expected_transformation.shape
        assert_allclose(transformation, expected_transformation.to(
            self.device).to(self.dtype), atol=1e-4, rtol=1e-4)

    def _test_module_implementation(self, params):
        augmentation = self._create_augmentation_from_params(**params, p=0.5, return_transform=True)

        augmentation_sequence = nn.Sequential(augmentation, augmentation)

        input_tensor = torch.rand(3, 5, 5, device=self.device, dtype=self.dtype)  # 3 x 5 x 5

        torch.manual_seed(42)
        out1, transform1 = augmentation(input_tensor)
        out2, transform2 = augmentation(out1)
        transform = transform2 @ transform1

        torch.manual_seed(42)
        out_sequence, transform_sequence = augmentation_sequence(input_tensor)

        assert out2.shape == out_sequence.shape
        assert transform.shape == transform_sequence.shape
        assert_allclose(out2, out_sequence)
        assert_allclose(transform, transform_sequence)

    def _test_inverse_coordinate_check_implementation(self, params):
        torch.manual_seed(42)

        input_tensor = torch.zeros((1, 3, 50, 100), device=self.device, dtype=self.dtype)
        input_tensor[:, :, 20:30, 40:60] = 1.

        augmentation = self._create_augmentation_from_params(**params, p=1.0, return_transform=True)
        output, transform = augmentation(input_tensor)

        if (transform == kornia.eye_like(3, transform)).all():
            pytest.skip("Test not relevant for intensity augmentations.")

        indices = kornia.create_meshgrid(height=output.shape[-2],
                                         width=output.shape[-1],
                                         normalized_coordinates=False,
                                         device=self.device)
        output_indices = indices.reshape((1, -1, 2)).to(dtype=self.dtype)
        input_indices = kornia.geometry.transform_points(transform.to(self.dtype).inverse(), output_indices)

        output_indices = output_indices.round().long().squeeze(0)
        input_indices = input_indices.round().long().squeeze(0)
        output_values = output[0, 0, output_indices[:, 1], output_indices[:, 0]]
        value_mask = output_values > 0.9999

        output_values = output[0, :, output_indices[:, 1][value_mask], output_indices[:, 0][value_mask]]
        input_values = input_tensor[0, :, input_indices[:, 1][value_mask], input_indices[:, 0][value_mask]]

        assert_allclose(output_values, input_values)

    def _test_gradcheck_implementation(self, params):
        input_tensor = torch.rand((3, 5, 5), device=self.device, dtype=self.dtype)  # 3 x 3
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(
            self._create_augmentation_from_params(
                **params,
                p=1.,
                return_transform=False),
            (input_tensor,
             ),
            raise_exception=True)


class TestRandomEqualizeAlternative(CommonTests):

    possible_params: Dict["str", Tuple] = {}

    _augmentation_cls = RandomEqualize
    _default_param_set: Dict["str", Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        input_tensor = torch.arange(20., device=self.device, dtype=self.dtype) / 20
        input_tensor = input_tensor.repeat(1, 2, 20, 1)

        expected_output = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ], device=self.device, dtype=self.dtype)
        expected_output = expected_output.repeat(1, 2, 20, 1)

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            params=parameters)

    def test_random_p_1_return_transform(self):
        torch.manual_seed(42)

        input_tensor = torch.rand(1, 1, 3, 4, device=self.device, dtype=self.dtype)

        # Note: For small inputs it should return the input image
        expected_output = input_tensor

        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    def test_batch(self):
        input_tensor = torch.arange(20., device=self.device, dtype=self.dtype) / 20
        input_tensor = input_tensor.repeat(2, 3, 20, 1)

        expected_output = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ], device=self.device, dtype=self.dtype)
        expected_output = expected_output.repeat(2, 3, 20, 1)

        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    def test_exception(self):

        with pytest.raises(ValueError):
            self._create_augmentation_from_params(p=1.)(torch.ones(
                (1, 3, 4, 5) * 200, device=self.device, dtype=self.dtype))


class TestCenterCropAlternative(CommonTests):
    possible_params: Dict["str", Tuple] = {
        "size": (2, (2, 2)),
        "resample": (0, Resample.BILINEAR.name, Resample.BILINEAR),
        "align_corners": (False, True),
    }
    _augmentation_cls = CenterCrop
    _default_param_set: Dict["str", Any] = {"size": (2, 2), "align_corners": True}

    @pytest.fixture(params=default_with_one_parameter_changed(default=_default_param_set,
                                                              **possible_params), scope="class")
    def param_set(self, request):
        return request.param

    @pytest.mark.parametrize("input_shape,expected_output_shape",
                             [((4, 5), (1, 1, 2, 3)), ((3, 4, 5), (1, 3, 2, 3)), ((2, 3, 4, 5), (2, 3, 2, 3))])
    def test_cardinality(self, input_shape, expected_output_shape):
        self._test_cardinality_implementation(input_shape=input_shape,
                                              expected_output_shape=expected_output_shape,
                                              params={"size": (2, 3), "align_corners": True})

    @pytest.mark.xfail(reason="size=(1,2) results in RuntimeError: solve_cpu: For batch 0: U(3,3) is zero, singular U.")
    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 0.0, 0.1, 0.2]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[
            [0.6, 0.7, ],
        ]]], device=self.device, dtype=self.dtype)

        parameters = {"size": (1, 2), "align_corners": True, "resample": 0}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            params=parameters)

    def test_random_p_1_return_transform(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 0.0, 0.1, 0.2]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[
            [0.2, 0.3, ],
            [0.6, 0.7, ],
            [0.0, 0.1, ],
        ]]], device=self.device, dtype=self.dtype)
        expected_transformation = torch.tensor([[[1., 0., -1.],
                                                 [0., 1., 0.],
                                                 [0., 0., 1.]]], device=self.device, dtype=self.dtype)
        parameters = {"size": (3, 2), "align_corners": True, "resample": 0}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    def test_batch(self):
        torch.manual_seed(42)

        input_tensor = torch.rand((2, 3, 4, 4), device=self.device, dtype=self.dtype)
        expected_output = input_tensor[:, :, 1:3, 1:3]
        expected_transformation = torch.tensor([[[1., 0., -1.],
                                                 [0., 1., -1.],
                                                 [0., 0., 1.]]], device=self.device, dtype=self.dtype).repeat(2, 1, 1,)
        parameters = {"size": (2, 2), "align_corners": True, "resample": 0}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    @pytest.mark.xfail(reason="No input validation is implemented yet.")
    def test_exception(self):
        # Wrong type
        with pytest.raises(TypeError):
            self._create_augmentation_from_params(size=0.0)
        with pytest.raises(TypeError):
            self._create_augmentation_from_params(size=2, align_corners=0)
        with pytest.raises(TypeError):
            self._create_augmentation_from_params(size=2, resample=True)

        # Bound check
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(size=-1)
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(size=(-1, 2))
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(size=(2, -1))


class TestRandomHorizontalFlipAlternative(CommonTests):
    possible_params: Dict["str", Tuple] = {}
    _augmentation_cls = RandomHorizontalFlip
    _default_param_set: Dict["str", Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, ],
                                      [0.4, 0.5, 0.6, ],
                                      [0.7, 0.8, 0.9, ]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[[0.3, 0.2, 0.1, ],
                                          [0.6, 0.5, 0.4, ],
                                          [0.9, 0.8, 0.7, ]]]], device=self.device, dtype=self.dtype)

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            params=parameters)

    def test_random_p_1_return_transform(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, ],
                                      [0.4, 0.5, 0.6, ],
                                      [0.7, 0.8, 0.9, ]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[[0.3, 0.2, 0.1, ],
                                          [0.6, 0.5, 0.4, ],
                                          [0.9, 0.8, 0.7, ]]]], device=self.device, dtype=self.dtype)
        expected_transformation = torch.tensor([[[-1.0, 0.0, 2.0],
                                                 [0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype)
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    def test_batch(self):
        torch.manual_seed(12)

        input_tensor = torch.tensor([[[[0.1, 0.2, 0.3, ],
                                       [0.4, 0.5, 0.6, ],
                                       [0.7, 0.8, 0.9, ]]]], device=self.device, dtype=self.dtype).repeat((2, 1, 1, 1))
        expected_output = torch.tensor([[[[0.3, 0.2, 0.1, ],
                                          [0.6, 0.5, 0.4, ],
                                          [0.9, 0.8, 0.7, ]]]], device=self.device, dtype=self.dtype).repeat((2, 1, 1, 1))  # noqa: E501
        expected_transformation = torch.tensor([[[-1.0, 0.0, 2.0],
                                                 [0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype).repeat((2, 1, 1))  # noqa: E501
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    @pytest.mark.skip(reason="No special parameters to validate.")
    def test_exception(self):
        pass


class TestRandomVerticalFlipAlternative(CommonTests):
    possible_params: Dict["str", Tuple] = {}
    _augmentation_cls = RandomVerticalFlip
    _default_param_set: Dict["str", Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, ],
                                      [0.4, 0.5, 0.6, ],
                                      [0.7, 0.8, 0.9, ]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[[0.7, 0.8, 0.9, ],
                                          [0.4, 0.5, 0.6, ],
                                          [0.1, 0.2, 0.3, ]]]], device=self.device, dtype=self.dtype)

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            params=parameters)

    def test_random_p_1_return_transform(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, ],
                                      [0.4, 0.5, 0.6, ],
                                      [0.7, 0.8, 0.9, ]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[[0.7, 0.8, 0.9, ],
                                          [0.4, 0.5, 0.6, ],
                                          [0.1, 0.2, 0.3, ]]]], device=self.device, dtype=self.dtype)
        expected_transformation = torch.tensor([[[1.0, 0.0, 0.0],
                                                 [0.0, -1.0, 2.0],
                                                 [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype)
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    def test_batch(self):
        torch.manual_seed(12)

        input_tensor = torch.tensor([[[[0.1, 0.2, 0.3, ],
                                       [0.4, 0.5, 0.6, ],
                                       [0.7, 0.8, 0.9, ]]]], device=self.device, dtype=self.dtype).repeat((2, 1, 1, 1))
        expected_output = torch.tensor([[[[0.7, 0.8, 0.9, ],
                                          [0.4, 0.5, 0.6, ],
                                          [0.1, 0.2, 0.3, ]]]], device=self.device, dtype=self.dtype).repeat((2, 1, 1, 1))  # noqa: E501
        expected_transformation = torch.tensor([[[1.0, 0.0, 0.0],
                                                 [0.0, -1.0, 2.0],
                                                 [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype).repeat((2, 1, 1))  # noqa: E501
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    @pytest.mark.skip(reason="No special parameters to validate.")
    def test_exception(self):
        pass


class TestRandomRotationAlternative(CommonTests):
    possible_params: Dict["str", Tuple] = {
        "degrees": (0., (-360., 360.), [0., 0.], torch.Tensor((-180., 180))),
        "interpolation": (0, Resample.BILINEAR.name, Resample.BILINEAR, None),
        "resample": (0, Resample.BILINEAR.name, Resample.BILINEAR),
        "align_corners": (False, True),
    }
    _augmentation_cls = RandomRotation
    _default_param_set: Dict["str", Any] = {"degrees": (30., 30.), "align_corners": True}

    @pytest.fixture(params=default_with_one_parameter_changed(default=_default_param_set,
                                                              **possible_params), scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, ],
                                      [0.4, 0.5, 0.6, ],
                                      [0.7, 0.8, 0.9, ]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[[0.3, 0.6, 0.9, ],
                                          [0.2, 0.5, 0.8, ],
                                          [0.1, 0.4, 0.7, ]]]], device=self.device, dtype=self.dtype)

        parameters = {"degrees": (90., 90.), "align_corners": True}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            params=parameters)

    def test_random_p_1_return_transform(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[[0.1, 0.2, 0.3, ],
                                      [0.4, 0.5, 0.6, ],
                                      [0.7, 0.8, 0.9, ]]], device=self.device, dtype=self.dtype)
        expected_output = torch.tensor([[[[0.7, 0.4, 0.1, ],
                                          [0.8, 0.5, 0.2, ],
                                          [0.9, 0.6, 0.3, ]]]], device=self.device, dtype=self.dtype)
        expected_transformation = torch.tensor([[[0.0, -1.0, 2.0],
                                                 [1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype)
        parameters = {"degrees": (-90., -90.), "align_corners": True}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    def test_batch(self):
        torch.manual_seed(12)

        input_tensor = torch.tensor([[[[0.1, 0.2, 0.3, ],
                                       [0.4, 0.5, 0.6, ],
                                       [0.7, 0.8, 0.9, ]]]], device=self.device, dtype=self.dtype).repeat((2, 1, 1, 1))
        expected_output = input_tensor
        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {"degrees": (-360., -360.), "align_corners": True}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    @pytest.mark.xfail(reason="No input validation is implemented yet.")
    def test_exception(self):
        # Wrong type
        with pytest.raises(TypeError):
            self._create_augmentation_from_params(degrees="")
        with pytest.raises(TypeError):
            self._create_augmentation_from_params(degrees=(3, 3), align_corners=0)
        with pytest.raises(TypeError):
            self._create_augmentation_from_params(degrees=(3, 3), resample=True)

        # Bound check
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(degrees=-361.0)
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(degrees=(-361.0, 360.))
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(degrees=(-360.0, 361.))
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(degrees=(360.0, -360.))


class TestRandomGrayscaleAlternative(CommonTests):

    possible_params: Dict["str", Tuple] = {}

    _augmentation_cls = RandomGrayscale
    _default_param_set: Dict["str", Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    @pytest.mark.parametrize("input_shape,expected_output_shape",
                             [((3, 4, 5), (1, 3, 4, 5)), ((2, 3, 4, 5), (2, 3, 4, 5))])
    def test_cardinality(self, input_shape, expected_output_shape):
        self._test_cardinality_implementation(
            input_shape=input_shape, expected_output_shape=expected_output_shape, params=self._default_param_set)

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                     [0.5, 0.6, 0.7, 0.8],
                                     [0.9, 0.0, 0.1, 0.2]], device=self.device, dtype=self.dtype).repeat(1, 3, 1, 1)
        expected_output = (
            input_tensor *
            torch.tensor(
                [
                    0.299,
                    0.587,
                    0.114],
                device=self.device,
                dtype=self.dtype).view(
                1,
                3,
                1,
                1)).sum(
                    dim=1,
                    keepdim=True).repeat(
                        1,
                        3,
                        1,
            1)

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            params=parameters)

    def test_random_p_1_return_transform(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                     [0.5, 0.6, 0.7, 0.8],
                                     [0.9, 0.0, 0.1, 0.2]], device=self.device, dtype=self.dtype).repeat(1, 3, 1, 1)
        expected_output = (
            input_tensor *
            torch.tensor(
                [
                    0.299,
                    0.587,
                    0.114],
                device=self.device,
                dtype=self.dtype).view(
                1,
                3,
                1,
                1)).sum(
                    dim=1,
                    keepdim=True).repeat(
                        1,
                        3,
                        1,
            1)

        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    def test_batch(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                     [0.5, 0.6, 0.7, 0.8],
                                     [0.9, 0.0, 0.1, 0.2]], device=self.device, dtype=self.dtype).repeat(2, 3, 1, 1)
        expected_output = (
            input_tensor *
            torch.tensor(
                [
                    0.299,
                    0.587,
                    0.114],
                device=self.device,
                dtype=self.dtype).view(
                1,
                3,
                1,
                1)).sum(
                    dim=1,
                    keepdim=True).repeat(
                        1,
                        3,
                        1,
            1)

        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {}
        self._test_random_p_1_return_transform_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters)

    @pytest.mark.xfail(reason="No input validation is implemented yet when p=0.")
    def test_exception(self):
        torch.manual_seed(42)

        with pytest.raises(ValueError):
            self._create_augmentation_from_params(p=0.)(torch.rand((1, 1, 4, 5), device=self.device, dtype=self.dtype))
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(p=1.)(torch.rand((1, 4, 4, 5), device=self.device, dtype=self.dtype))


class TestRandomHorizontalFlip:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomHorizontalFlip(p=0.5)
        repr = "RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_hflip(self, device, dtype):

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

        expected_transform = torch.tensor([[-1., 0., 3.],
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

    def test_batch_random_hflip(self, device, dtype):

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

        expected_transform = torch.tensor([[[-1., 0., 2.],
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

    def test_same_on_batch(self, device, dtype):
        f = RandomHorizontalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        f = nn.Sequential(
            RandomHorizontalFlip(p=1.0, return_transform=True),
            RandomHorizontalFlip(p=1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomHorizontalFlip(p=1.0, return_transform=True),
            RandomHorizontalFlip(p=1.0),
        )

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected_transform = torch.tensor([[[-1., 0., 2.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform
        expected_transform_1 = expected_transform_1.to(device)

        assert(f(input)[0] == input).all()
        assert(f(input)[1] == expected_transform_1).all()
        assert(f1(input)[0] == input).all()
        assert(f1(input)[1] == expected_transform).all()

    def test_random_hflip_coord_check(self, device, dtype):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)

        input = torch.tensor([[[[1., 2., 3., 4.],
                                [5., 6., 7., 8.],
                                [9., 10., 11., 12.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        input_coordinates = torch.tensor([[
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],  # x coord
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # y coord
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_output = torch.tensor([[[[4., 3., 2., 1.],
                                          [8., 7., 6., 5.],
                                          [12., 11., 10., 9.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        output, transform = f(input)
        result_coordinates = transform @ input_coordinates
        # NOTE: without rounding it might produce unexpected results
        input_coordinates = input_coordinates.round().long()
        result_coordinates = result_coordinates.round().long()

        # Tensors must have the same shapes and values
        assert output.shape == expected_output.shape
        assert (output == expected_output).all()
        # Transformed indices must not be out of bound
        assert (torch.torch.logical_and(result_coordinates[0, 0, :] >= 0,
                                        result_coordinates[0, 0, :] < input.shape[-1])).all()
        assert (torch.torch.logical_and(result_coordinates[0, 1, :] >= 0,
                                        result_coordinates[0, 1, :] < input.shape[-2])).all()
        # Values in the output tensor at the places of transformed indices must
        # have the same value as the input tensor has at the corresponding
        # positions
        assert (output[..., result_coordinates[0, 1, :], result_coordinates[0, 0, :]] ==
                input[..., input_coordinates[0, 1, :], input_coordinates[0, 0, :]]).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
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

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomHorizontalFlip(p=1.), (input, ), raise_exception=True)


class TestRandomVerticalFlip:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomVerticalFlip(p=0.5)
        repr = "RandomVerticalFlip(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_vflip(self, device, dtype):

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
                                           [0., -1., 2.],
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

    def test_batch_random_vflip(self, device, dtype):

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
                                            [0., -1., 2.],
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

    def test_same_on_batch(self, device, dtype):
        f = RandomVerticalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        f = nn.Sequential(
            RandomVerticalFlip(p=1.0, return_transform=True),
            RandomVerticalFlip(p=1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomVerticalFlip(p=1.0, return_transform=True),
            RandomVerticalFlip(p=1.0),
        )

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 2.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        assert_allclose(f(input)[0], input.squeeze())
        assert_allclose(f(input)[1], expected_transform_1)
        assert_allclose(f1(input)[0], input.squeeze())
        assert_allclose(f1(input)[1], expected_transform)

    def test_random_vflip_coord_check(self, device, dtype):

        f = RandomVerticalFlip(p=1.0, return_transform=True)

        input = torch.tensor([[[[1., 2., 3., 4.],
                                [5., 6., 7., 8.],
                                [9., 10., 11., 12.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        input_coordinates = torch.tensor([[
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],  # x coord
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # y coord
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_output = torch.tensor([[[[9., 10., 11., 12.],
                                          [5., 6., 7., 8.],
                                          [1., 2., 3., 4.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        output, transform = f(input)
        result_coordinates = transform @ input_coordinates
        # NOTE: without rounding it might produce unexpected results
        input_coordinates = input_coordinates.round().long()
        result_coordinates = result_coordinates.round().long()

        # Tensors must have the same shapes and values
        assert output.shape == expected_output.shape
        assert (output == expected_output).all()
        # Transformed indices must not be out of bound
        assert (torch.torch.logical_and(result_coordinates[0, 0, :] >= 0,
                                        result_coordinates[0, 0, :] < input.shape[-1])).all()
        assert (torch.torch.logical_and(result_coordinates[0, 1, :] >= 0,
                                        result_coordinates[0, 1, :] < input.shape[-2])).all()
        # Values in the output tensor at the places of transformed indices must
        # have the same value as the input tensor has at the corresponding
        # positions
        assert (output[..., result_coordinates[0, 1, :], result_coordinates[0, 0, :]] ==
                input[..., input_coordinates[0, 1, :], input_coordinates[0, 0, :]]).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
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

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomVerticalFlip(p=1.), (input, ), raise_exception=True)


class TestColorJitter:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = ColorJitter(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1)
        repr = "ColorJitter(brightness=tensor([0.5000, 1.5000]), contrast=tensor([0.7000, 1.3000]), "\
               "saturation=tensor([0.2000, 1.2000]), hue=tensor([-0.1000,  0.1000]), "\
               "p=1.0, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_color_jitter(self, device, dtype):

        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(3, 5, 5, device=device, dtype=dtype)  # 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_color_jitter_batch(self, device, dtype):
        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_same_on_batch(self, device, dtype):
        f = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def _get_expected_brightness(self, device, dtype):
        return torch.tensor([
            [[[0.2529, 0.3529, 0.4529],
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
              [0.8660, 0.9660, 1.0000]]]], device=device, dtype=dtype)

    def test_random_brightness(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(brightness=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_brightness(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_brightness_tuple(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(brightness=(0.8, 1.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_brightness(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def _get_expected_contrast(self, device, dtype):
        return torch.tensor([
            [[[0.0953, 0.1906, 0.2859],
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
              [0.8286, 0.9470, 1.0000]]]], device=device, dtype=dtype)

    def test_random_contrast(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(contrast=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_contrast(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def test_random_contrast_list(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(contrast=[0.8, 1.2])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_contrast(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def _get_expected_saturation(self, device, dtype):
        return torch.tensor([
            [[[0.1876, 0.2584, 0.3389],
              [0.6292, 0.5000, 0.4000],
              [0.7097, 0.8000, 1.0000]],
             [[1.0000, 0.5292, 0.6097],
              [0.6292, 0.3195, 0.2195],
              [0.8000, 0.1682, 0.2779]],
             [[0.6389, 0.8000, 0.7000],
              [0.9000, 0.3195, 0.2195],
              [0.8000, 0.4389, 0.5487]]],
            [[[0.0000, 0.1295, 0.2530],
              [0.5648, 0.5000, 0.4000],
              [0.6883, 0.8000, 1.0000]],
             [[1.0000, 0.4648, 0.5883],
              [0.5648, 0.2765, 0.1765],
              [0.8000, 0.0178, 0.1060]],
             [[0.5556, 0.8000, 0.7000],
              [0.9000, 0.2765, 0.1765],
              [0.8000, 0.3530, 0.4413]]]], device=device, dtype=dtype)

    def test_random_saturation(self, device, dtype):
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
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)
        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_saturation_tensor(self, device, dtype):
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
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_saturation_tuple(self, device, dtype):
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
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def _get_expected_hue(self, device, dtype):
        return torch.tensor([
            [[[0.1000, 0.2000, 0.3000],
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
              [0.8000, 0.3730, 0.4692]]]], device=device, dtype=dtype)

    def test_random_hue(self, device, dtype):
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
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_hue(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_hue_list(self, device, dtype):
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
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_hue(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_hue_list_batch(self, device, dtype):
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
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_hue(device, dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_sequential(self, device, dtype):

        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(3, 5, 5, device=device, dtype=dtype)  # 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform, atol=1e-4, rtol=1e-5)

    def test_color_jitter_batch_sequential(self, device, dtype):
        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform, atol=1e-4, rtol=1e-5)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 5, 5), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.ColorJitter(p=1.), (input, ), raise_exception=True)


class TestRectangleRandomErasing:
    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_random_rectangle_erasing_shape(
            self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=1.)
        assert rand_rec(input).shape == batch_shape

    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_no_rectangle_erasing_shape(
            self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=0.)
        assert rand_rec(input).equal(input)

    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("shape", [(3, 11, 7)])
    def test_same_on_batch(self, shape, erase_scale_range, aspect_ratio_range):
        f = RandomErasing(erase_scale_range, aspect_ratio_range, same_on_batch=True, p=0.5)
        input = torch.rand(shape).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        erase_scale_range = (.2, .4)
        aspect_ratio_range = (.3, .5)

        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=1.0)
        rect_params = rand_rec.__forward_parameters__(batch_shape, p=1.0, p_batch=1., same_on_batch=False)

        # evaluate function gradient
        input = torch.rand(batch_shape, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            rand_rec,
            (input, rect_params),
            raise_exception=True,
        )

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
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

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomGrayscale()
        repr = "RandomGrayscale(p=0.1, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_grayscale(self, device, dtype):

        f = RandomGrayscale(return_transform=True)

        input = torch.rand(3, 5, 5, device=device, dtype=dtype)  # 3 x 5 x 5

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[1], expected_transform)

    def test_same_on_batch(self, device, dtype):
        f = RandomGrayscale(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_opencv_true(self, device, dtype):
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
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)

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
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]], device=device, dtype=dtype)

        img_gray = kornia.augmentation.RandomGrayscale(p=1.)(data)
        assert_allclose(img_gray, expected)

    def test_opencv_false(self, device, dtype):
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
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)

        expected = data

        img_gray = kornia.augmentation.RandomGrayscale(p=0.)(data)
        assert_allclose(img_gray, expected)

    def test_opencv_true_batch(self, device, dtype):
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
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)
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
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]], device=device, dtype=dtype)
        expected = expected.unsqueeze(0).repeat(4, 1, 1, 1)

        img_gray = kornia.augmentation.RandomGrayscale(p=1.)(data)
        assert_allclose(img_gray, expected)

    def test_opencv_false_batch(self, device, dtype):
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
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)
        data = data.unsqueeze(0).repeat(4, 1, 1, 1)

        expected = data

        img_gray = kornia.augmentation.RandomGrayscale(p=0.)(data)
        assert_allclose(img_gray, expected)

    def test_random_grayscale_sequential_batch(self, device, dtype):
        f = nn.Sequential(
            RandomGrayscale(p=0., return_transform=True),
            RandomGrayscale(p=0., return_transform=True),
        )

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 5, 5), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.RandomGrayscale(p=1.), (input,), raise_exception=True)
        assert gradcheck(kornia.augmentation.RandomGrayscale(p=0.), (input,), raise_exception=True)


class TestCenterCrop:

    def test_no_transform(self, device, dtype):
        inp = torch.rand(1, 2, 4, 4, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop(2)(inp)
        assert out.shape == (1, 2, 2, 2)

    def test_transform(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop(2, return_transform=True)(inp)
        assert len(out) == 2
        assert out[0].shape == (1, 2, 2, 2)
        assert out[1].shape == (1, 3, 3)

    def test_no_transform_tuple(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop((3, 4))(inp)
        assert out.shape == (1, 2, 3, 4)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.CenterCrop(3), (input,), raise_exception=True)


class TestRandomRotation:

    torch.manual_seed(0)  # for random reproductibility

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomRotation(degrees=45.5)
        repr = "RandomRotation(degrees=tensor([-45.5000,  45.5000]), interpolation=BILINEAR, p=0.5, "\
               "p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_rotation(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, return_transform=True, p=1.)
        f1 = RandomRotation(degrees=45.0, p=1.)

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]], device=device, dtype=dtype)  # 4 x 4

        expected = torch.tensor([[[[0.9824, 0.0088, 0.0000, 1.9649],
                                   [0.0000, 0.0029, 0.0000, 0.0176],
                                   [0.0029, 1.0000, 1.9883, 0.0000],
                                   [0.0000, 0.0088, 1.0117, 1.9649]]]], device=device, dtype=dtype)  # 1 x 4 x 4

        expected_transform = torch.tensor([[[1.0000, -0.0059, 0.0088],
                                            [0.0059, 1.0000, -0.0088],
                                            [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_2 = torch.tensor([[[[0.1322, 0.0000, 0.7570, 0.2644],
                                     [0.3785, 0.0000, 0.4166, 0.0000],
                                     [0.0000, 0.6309, 1.5910, 1.2371],
                                     [0.0000, 0.1444, 0.3177, 0.6499]]]], device=device, dtype=dtype)  # 1 x 4 x 4

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
        assert_allclose(f1(input), expected_2, rtol=1e-6, atol=1e-4)

    def test_batch_random_rotation(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, return_transform=True, p=1.)

        input = torch.tensor([[[[1., 0., 0., 2.],
                                [0., 0., 0., 0.],
                                [0., 1., 2., 0.],
                                [0., 0., 1., 2.]]]], device=device, dtype=dtype)  # 1 x 1 x 4 x 4

        expected = torch.tensor([[[[0.9824, 0.0088, 0.0000, 1.9649],
                                   [0.0000, 0.0029, 0.0000, 0.0176],
                                   [0.0029, 1.0000, 1.9883, 0.0000],
                                   [0.0000, 0.0088, 1.0117, 1.9649]]],
                                 [[[0.1322, 0.0000, 0.7570, 0.2644],
                                   [0.3785, 0.0000, 0.4166, 0.0000],
                                   [0.0000, 0.6309, 1.5910, 1.2371],
                                   [0.0000, 0.1444, 0.3177, 0.6499]]]], device=device, dtype=dtype)  # 2 x 1 x 4 x 4

        expected_transform = torch.tensor([[[1.0000, -0.0059, 0.0088],
                                            [0.0059, 1.0000, -0.0088],
                                            [0.0000, 0.0000, 1.0000]],
                                           [[0.9125, 0.4090, -0.4823],
                                            [-0.4090, 0.9125, 0.7446],
                                            [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 2 x 3 x 3

        input = input.repeat(2, 1, 1, 1)  # 5 x 3 x 3 x 3

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation(degrees=40, same_on_batch=True)
        input = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = nn.Sequential(
            RandomRotation(torch.tensor([-45.0, 90]), return_transform=True, p=1.),
            RandomRotation(10.4, return_transform=True, p=1.),
        )
        f1 = nn.Sequential(
            RandomRotation(torch.tensor([-45.0, 90]), return_transform=True, p=1.),
            RandomRotation(10.4, p=1.),
        )

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]], device=device, dtype=dtype)  # 4 x 4

        expected = torch.tensor([[[[0.1314, 0.1050, 0.6649, 0.2628],
                                   [0.3234, 0.0202, 0.4256, 0.1671],
                                   [0.0525, 0.5976, 1.5199, 1.1306],
                                   [0.0000, 0.1453, 0.3224, 0.5796]]]], device=device, dtype=dtype)  # 1 x 4 x 4

        expected_transform = torch.tensor([[[0.8864, 0.4629, -0.5240],
                                            [-0.4629, 0.8864, 0.8647],
                                            [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_transform_2 = torch.tensor([[[0.8381, -0.5455, 1.0610],
                                              [0.5455, 0.8381, -0.5754],
                                              [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 1 x 3 x 3

        out, mat = f(input)
        _, mat_2 = f1(input)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-4, atol=1e-4)
        assert_allclose(mat_2, expected_transform_2, rtol=1e-4, atol=1e-4)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):

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

    def test_gradcheck(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomRotation(degrees=(15.0, 15.0), p=1.), (input, ), raise_exception=True)


class TestRandomCrop:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False, p=1.)
        repr = "RandomCrop(crop_size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False, padding_mode=constant, "\
               "resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_no_padding(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]], device=device, dtype=dtype)
        expected = torch.tensor([[[
            [3., 4., 5.],
            [6., 7., 8.]
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_no_padding_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([
            [[[0., 1., 2.],
              [3., 4., 5.]]],
            [[[3., 4., 5.],
              [6., 7., 8.]]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_same_on_batch(self, device, dtype):
        f = RandomCrop(size=(2, 3), padding=1, same_on_batch=True, align_corners=True, p=1.)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_padding_batch_1(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([[[
            [1., 2., 0.],
            [4., 5., 0.]
        ]], [[
            [7., 8., 0.],
            [0., 0., 0.]
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=1, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_padding_batch_2(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([[[
            [1., 2., 10.],
            [4., 5., 10.]
        ]], [[
            [4., 5., 10.],
            [7., 8., 10.],
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_padding_batch_3(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([[[
            [8., 8., 8.],
            [8., 0., 1.]
        ]], [[
            [8., 8., 8.],
            [1., 2., 8.]
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=(0, 1, 2, 3), fill=8, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([
            [[[9., 9., 9.],
              [0., 1., 2.]]],
            [[[9., 9., 9.],
              [0., 1., 2.]]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), pad_if_needed=True, fill=9, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomCrop(size=(3, 3), p=1.), (inp, ), raise_exception=True)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.).forward
        op_script = torch.jit.script(op)
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        actual = op_script(img)
        expected = kornia.center_crop3d(img)
        assert_allclose(actual, expected)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit_trace(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.).forward
        op_script = torch.jit.script(op)
        # 1. Trace op
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        op_trace = torch.jit.trace(op_script, (img,))

        # 2. Generate new input
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        # 3. Evaluate
        actual = op_trace(img)
        expected = op(img)
        assert_allclose(actual, expected)


class TestRandomResizedCrop:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomResizedCrop(size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0))
        repr = "RandomResizedCrop(size=(2, 3), scale=tensor([1., 1.]), ratio=tensor([1., 1.]), "\
               "interpolation=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_no_resize(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[5.3750, 5.8750, 4.5938],
               [6.3437, 6.7812, 5.2500]]]], device=device, dtype=dtype)
        rrc = RandomResizedCrop(
            size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomResizedCrop(
            size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0), same_on_batch=True)
        input = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_crop_scale_ratio(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[1.2500, 1.7500, 1.5000],
               [4.2500, 4.7500, 3.7500],
               [7.2500, 7.7500, 6.0000]]]], device=device, dtype=dtype)
        rrc = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected)

    def test_crop_scale_ratio_batch(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)

        expected = torch. tensor([
            [[[1.2500, 1.7500, 1.5000],
              [4.2500, 4.7500, 3.7500],
              [7.2500, 7.7500, 6.0000]]],
            [[[0.0000, 0.2500, 0.7500],
              [2.2500, 3.2500, 3.7500],
              [4.5000, 6.2500, 6.7500]]]], device=device, dtype=dtype)
        rrc = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 3), device=device, dtype=dtype)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomResizedCrop(
            size=(3, 3), scale=(1., 1.), ratio=(1., 1.)), (inp, ), raise_exception=True)


class TestRandomEqualize:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = RandomEqualize(p=0.5)
        repr = "RandomEqualize(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_equalize(self, device, dtype):
        f = RandomEqualize(p=1.0, return_transform=True)
        f1 = RandomEqualize(p=0., return_transform=True)
        f2 = RandomEqualize(p=1.)
        f3 = RandomEqualize(p=0.)

        bs, channels, height, width = 1, 3, 20, 20

        inputs = self.build_input(channels, height, width, device=device, dtype=dtype).squeeze(dim=0)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ])
        expected = self.build_input(channels, height, width, bs=1, row=row_expected,
                                    device=device, dtype=dtype)
        identity = kornia.eye_like(3, expected)  # 3 x 3

        assert_allclose(f(inputs)[0], expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[0], inputs, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f2(inputs), expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f3(inputs), inputs, rtol=1e-4, atol=1e-4)

    def test_batch_random_equalize(self, device, dtype):
        f = RandomEqualize(p=1.0, return_transform=True)
        f1 = RandomEqualize(p=0., return_transform=True)
        f2 = RandomEqualize(p=1.)
        f3 = RandomEqualize(p=0.)

        bs, channels, height, width = 2, 3, 20, 20

        inputs = self.build_input(channels, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ])
        expected = self.build_input(channels, height, width, bs, row=row_expected,
                                    device=device, dtype=dtype)

        identity = kornia.eye_like(3, expected)  # 2 x 3 x 3

        assert_allclose(f(inputs)[0], expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[0], inputs, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f2(inputs), expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f3(inputs), inputs, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomEqualize(p=0.5, same_on_batch=True)
        input = torch.eye(4, device=device, dtype=dtype)
        input = input.unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_gradcheck(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomEqualize(p=0.5), (input,), raise_exception=True)

    @staticmethod
    def build_input(channels, height, width, bs=1, row=None, device='cpu', dtype=torch.float32):
        if row is None:
            row = torch.arange(width, device=device, dtype=dtype) / float(width)

        channel = torch.stack([row] * height)
        image = torch.stack([channel] * channels)
        batch = torch.stack([image] * bs)

        return batch.to(device, dtype)
