from typing import Any, Dict, Optional, Type

import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.augmentation import (
    RandomJitterPC,
    RandomRotatePC,
    RandomRGBJitterPC,
)
from kornia.augmentation._pc.base import AugmentationBasePC
from kornia.testing import BaseTester, assert_close


@pytest.mark.usefixtures("device", "dtype")
class CommonTests(BaseTester):
    fixture_names = ("device", "dtype")

    ############################################################################################################
    # Attribute variables to set
    ############################################################################################################
    _augmentation_cls: Optional[Type[AugmentationBasePC]] = None
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
        raise NotImplementedError("param_set must be overridden in subclasses")

    ############################################################################################################
    # Test cases
    ############################################################################################################
    def test_smoke(self, param_set):
        self._test_smoke_implementation(params=param_set)
        self._test_smoke_call_implementation(params=param_set)

    def test_random_p_0(self):
        self._test_random_p_0_implementation(params=self._default_param_set)

    def test_random_p_1(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_exception(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_batch(self):
        raise NotImplementedError("Implement a stupid routine.")

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        raise NotImplementedError("Implement a stupid routine.")

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass

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
        assert issubclass(
            self._augmentation_cls, AugmentationBasePC
        ), f"{self._augmentation_cls} is not a subclass of AugmentationBasePC"

        # Can be instatiated
        augmentation = self._create_augmentation_from_params(**params)
        assert issubclass(
            type(augmentation), AugmentationBasePC
        ), f"{type(augmentation)} is not a subclass of AugmentationBasePC"

        # generate_parameters can be called and returns the correct amount of parameters
        batch_shape = (4, 10, 9)
        generated_params = augmentation.forward_parameters(batch_shape)
        assert isinstance(generated_params, dict)

        # compute_transformation can be called and returns the correct shaped transformation matrix
        to_apply = generated_params['batch_prob'] > 0.5
        expected_transformation_shape = torch.Size((to_apply.sum(), 3, 3))
        test_input = torch.ones(batch_shape, device=self.device, dtype=self.dtype)
        transformation = augmentation.compute_transformation(test_input[to_apply], generated_params, augmentation.flags)
        assert transformation.shape == expected_transformation_shape

        # apply_transform can be called and returns the correct batch sized output
        if to_apply.sum() != 0:
            output = augmentation.apply_transform(
                test_input[to_apply], generated_params, augmentation.flags, transformation
            )
            assert output.shape[0] == to_apply.sum()
        else:
            # Re-generate parameters if 0 batch size
            self._test_smoke_implementation(params)

    def _test_smoke_call_implementation(self, params):
        batch_shape = (4, 10, 9)
        expected_transformation_shape = torch.Size((batch_shape[0], 3, 3))
        test_input = torch.ones(batch_shape, device=self.device, dtype=self.dtype)
        augmentation = self._create_augmentation_from_params(**params)
        generated_params = augmentation.forward_parameters(batch_shape)

        output = augmentation(test_input, params=generated_params)
        assert output.shape[0] == batch_shape[0]
        assert augmentation.transform_matrix.shape == expected_transformation_shape

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
        augmentation = self._create_augmentation_from_params(**params, p=0.0)
        test_input = torch.rand((2, 10, 9), device=self.device, dtype=self.dtype)
        output = augmentation(test_input)
        assert_close(output, test_input)

    def _test_random_p_1_implementation(self, input_tensor, expected_output, params, expected_transformation=None):
        augmentation = self._create_augmentation_from_params(**params, p=1.0)
        output = augmentation(input_tensor.to(self.device).to(self.dtype))

        # Output should match
        assert output.shape == expected_output.shape
        self.assert_close(output, expected_output.to(device=self.device, dtype=self.dtype), low_tolerance=True)
        if expected_transformation is not None:
            transform = augmentation.transform_matrix
            self.assert_close(transform, expected_transformation, low_tolerance=True)

    def _test_gradcheck_implementation(self, params):
        input_tensor = torch.rand((3, 30, 9), device=self.device, dtype=self.dtype)  # 3 x 3
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(
            self._create_augmentation_from_params(**params, p=1.0),
            (input_tensor,),
            raise_exception=True,
            fast_mode=True,
        )


class TestRandomJitterPC(CommonTests):
    possible_params: Dict["str", tuple] = {}

    _augmentation_cls = RandomJitterPC
    _default_param_set: Dict["str", Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        input = torch.randn((3, 10, 9))
        aug = RandomJitterPC(p=1.)
        output = aug(input)
        assert (input != output).any()

    def test_random_p_0(self):
        input = torch.randn((3, 10, 9))
        aug = RandomJitterPC(p=0.)
        output = aug(input)
        assert (input == output).all()

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self):
        raise NotImplementedError("Implement a stupid routine.")

    @pytest.mark.skip(reason="no need.")
    def test_batch(self):
        pass


class TestRandomRGBJitterPC(CommonTests):
    possible_params: Dict["str", tuple] = {}

    _augmentation_cls = RandomRGBJitterPC
    _default_param_set: Dict["str", Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        input = torch.randn((3, 10, 9))
        aug = RandomRGBJitterPC(p=1.)
        output = aug(input)
        assert (input != output).any()

    def test_random_p_0(self):
        input = torch.randn((3, 10, 9))
        aug = RandomRGBJitterPC(p=0.)
        output = aug(input)
        assert (input == output).all()

    def test_exception(self):
        input = torch.randn((3, 10, 8))
        aug = RandomRGBJitterPC(p=1.)
        with pytest.raises(RuntimeError):
            aug(input)

    @pytest.mark.skip(reason="no need.")
    def test_batch(self):
        pass


class TestRandomRotatePC(CommonTests):
    possible_params: Dict["str", tuple] = {}

    _augmentation_cls = RandomRotatePC
    _default_param_set: Dict["str", Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        input = torch.randn((3, 10, 9))
        aug = RandomRotatePC(p=1.)
        output = aug(input)
        assert (input != output).any()

    def test_random_p_0(self):
        input = torch.randn((3, 10, 9))
        aug = RandomRotatePC(p=0.)
        output = aug(input)
        assert (input == output).all()

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self):
        pass

    @pytest.mark.skip(reason="no need.")
    def test_batch(self):
        pass