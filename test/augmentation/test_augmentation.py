from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.augmentation import (
    AugmentationSequential,
    CenterCrop,
    ColorJiggle,
    ColorJitter,
    Denormalize,
    LongestMaxSize,
    Normalize,
    PadTo,
    RandomBoxBlur,
    RandomChannelShuffle,
    RandomCrop,
    RandomElasticTransform,
    RandomEqualize,
    RandomErasing,
    RandomFisheye,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomInvert,
    RandomPlanckianJitter,
    RandomPlasmaBrightness,
    RandomPlasmaContrast,
    RandomPlasmaShadow,
    RandomPosterize,
    RandomResizedCrop,
    RandomRGBShift,
    RandomRotation,
    RandomThinPlateSpline,
    RandomVerticalFlip,
    Resize,
    SmallestMaxSize,
)
from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.constants import Resample, pi
from kornia.geometry import transform_points
from kornia.testing import BaseTester, default_with_one_parameter_changed
from kornia.utils import create_meshgrid
from kornia.utils.helpers import _torch_inverse_cast

# TODO same_on_batch tests?


@pytest.mark.usefixtures("device", "dtype")
class CommonTests(BaseTester):
    fixture_names = ("device", "dtype")

    ############################################################################################################
    # Attribute variables to set
    ############################################################################################################
    _augmentation_cls: type[AugmentationBase2D] | None = None
    _default_param_set: dict[str, Any] = {}
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

    @pytest.mark.parametrize(
        "input_shape,expected_output_shape",
        [((4, 5), (1, 1, 4, 5)), ((3, 4, 5), (1, 3, 4, 5)), ((2, 3, 4, 5), (2, 3, 4, 5))],
    )
    def test_cardinality(self, input_shape, expected_output_shape):
        self._test_cardinality_implementation(
            input_shape=input_shape, expected_output_shape=expected_output_shape, params=self._default_param_set
        )

    def test_random_p_0(self):
        self._test_random_p_0_implementation(params=self._default_param_set)

    def test_random_p_1(self):
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
        assert issubclass(
            self._augmentation_cls, AugmentationBase2D
        ), f"{self._augmentation_cls} is not a subclass of AugmentationBase2D"

        # Can be instatiated
        augmentation = self._create_augmentation_from_params(**params)
        assert issubclass(
            type(augmentation), AugmentationBase2D
        ), f"{type(augmentation)} is not a subclass of AugmentationBase2D"

        # generate_parameters can be called and returns the correct amount of parameters
        batch_shape = (4, 3, 5, 6)
        generated_params = augmentation.forward_parameters(batch_shape)
        assert isinstance(generated_params, dict)

        # compute_transformation can be called and returns the correct shaped transformation matrix
        expected_transformation_shape = torch.Size((generated_params['batch_prob'].sum(), 3, 3))
        test_input = torch.ones(batch_shape, device=self.device, dtype=self.dtype)
        transformation = augmentation.compute_transformation(
            test_input[generated_params['batch_prob']], generated_params, augmentation.flags
        )
        assert transformation.shape == expected_transformation_shape

        # apply_transform can be called and returns the correct batch sized output
        if generated_params['batch_prob'].sum() != 0:
            output = augmentation.apply_transform(
                test_input[generated_params['batch_prob']], generated_params, augmentation.flags, transformation
            )
            assert output.shape[0] == generated_params['batch_prob'].sum()
        else:
            # Re-generate parameters if 0 batch size
            self._test_smoke_implementation(params)

    def _test_smoke_call_implementation(self, params):
        batch_shape = (4, 3, 5, 6)
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
        test_input = torch.rand((2, 3, 4, 5), device=self.device, dtype=self.dtype)
        output = augmentation(test_input)
        assert (output == test_input).all()

    def _test_random_p_1_implementation(self, input_tensor, expected_output, params, expected_transformation=None):
        augmentation = self._create_augmentation_from_params(**params, p=1.0)
        output = augmentation(input_tensor.to(self.device).to(self.dtype))

        # Output should match
        assert output.shape == expected_output.shape
        self.assert_close(output, expected_output.to(device=self.device, dtype=self.dtype), low_tolerance=True)
        if expected_transformation is not None:
            transform = augmentation.transform_matrix
            self.assert_close(transform, expected_transformation, low_tolerance=True)

    def _test_module_implementation(self, params):
        augmentation = self._create_augmentation_from_params(**params, p=0.5)

        augmentation_sequence = AugmentationSequential(augmentation, augmentation)

        input_tensor = torch.rand(3, 5, 5, device=self.device, dtype=self.dtype)  # 3 x 5 x 5

        torch.manual_seed(42)
        out1 = augmentation(input_tensor)
        transform1 = augmentation.transform_matrix
        out2 = augmentation(out1)
        transform = augmentation.transform_matrix @ transform1

        torch.manual_seed(42)
        out_sequence = augmentation_sequence(input_tensor)
        transform_sequence = augmentation_sequence.transform_matrix

        assert out1.shape == out_sequence.shape
        assert transform.shape == transform_sequence.shape
        self.assert_close(out2, out_sequence, low_tolerance=True)
        self.assert_close(transform, transform_sequence, low_tolerance=True)

    def _test_inverse_coordinate_check_implementation(self, params):
        torch.manual_seed(42)

        input_tensor = torch.zeros((1, 3, 50, 100), device=self.device, dtype=self.dtype)
        input_tensor[:, :, 20:30, 40:60] = 1.0

        augmentation = self._create_augmentation_from_params(**params, p=1.0)
        output = augmentation(input_tensor)
        transform = augmentation.transform_matrix

        if (transform == kornia.eye_like(3, transform)).all():
            pytest.skip("Test not relevant for intensity augmentations.")

        indices = create_meshgrid(
            height=output.shape[-2], width=output.shape[-1], normalized_coordinates=False, device=self.device
        )
        output_indices = indices.reshape((1, -1, 2)).to(dtype=self.dtype)
        input_indices = transform_points(_torch_inverse_cast(transform.to(self.dtype)), output_indices)

        output_indices = output_indices.round().long().squeeze(0)
        input_indices = input_indices.round().long().squeeze(0)
        output_values = output[0, 0, output_indices[:, 1], output_indices[:, 0]]
        value_mask = output_values > 0.9999

        output_values = output[0, :, output_indices[:, 1][value_mask], output_indices[:, 0][value_mask]]
        input_values = input_tensor[0, :, input_indices[:, 1][value_mask], input_indices[:, 0][value_mask]]

        self.assert_close(output_values, input_values, low_tolerance=True)

    def _test_gradcheck_implementation(self, params):
        input_tensor = torch.rand((3, 5, 5), device=self.device, dtype=self.dtype)  # 3 x 3
        input_tensor = utils.tensor_to_gradcheck_var(input_tensor)  # to var
        assert gradcheck(self._create_augmentation_from_params(**params, p=1.0), (input_tensor,), raise_exception=True)


class TestRandomEqualizeAlternative(CommonTests):

    possible_params: dict[str, tuple] = {}

    _augmentation_cls = RandomEqualize
    _default_param_set: dict[str, Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        input_tensor = torch.arange(20.0, device=self.device, dtype=self.dtype) / 20
        input_tensor = input_tensor.repeat(1, 2, 20, 1)

        expected_output = torch.tensor(
            [
                0.0000,
                0.07843,
                0.15686,
                0.2353,
                0.3137,
                0.3922,
                0.4706,
                0.5490,
                0.6275,
                0.7059,
                0.7843,
                0.8627,
                0.9412,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        expected_output = expected_output.repeat(1, 2, 20, 1)

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor, expected_output=expected_output, params=parameters
        )

    def test_batch(self):
        input_tensor = torch.arange(20.0, device=self.device, dtype=self.dtype) / 20
        input_tensor = input_tensor.repeat(2, 3, 20, 1)

        expected_output = torch.tensor(
            [
                0.0000,
                0.07843,
                0.15686,
                0.2353,
                0.3137,
                0.3922,
                0.4706,
                0.5490,
                0.6275,
                0.7059,
                0.7843,
                0.8627,
                0.9412,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        expected_output = expected_output.repeat(2, 3, 20, 1)

        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters,
        )

    def test_exception(self):

        with pytest.raises(ValueError):
            self._create_augmentation_from_params(p=1.0)(
                torch.ones((1, 3, 4, 5) * 3, device=self.device, dtype=self.dtype)
            )


class TestCenterCropAlternative(CommonTests):
    possible_params: dict[str, tuple] = {
        "size": (2, (2, 2)),
        "resample": (0, Resample.BILINEAR.name, Resample.BILINEAR),
        "align_corners": (False, True),
    }
    _augmentation_cls = CenterCrop
    _default_param_set: dict[str, Any] = {"size": (2, 2), "align_corners": True}

    @pytest.fixture(
        params=default_with_one_parameter_changed(default=_default_param_set, **possible_params), scope="class"
    )
    def param_set(self, request):
        return request.param

    @pytest.mark.parametrize(
        "input_shape,expected_output_shape",
        [((4, 5), (1, 1, 2, 3)), ((3, 4, 5), (1, 3, 2, 3)), ((2, 3, 4, 5), (2, 3, 2, 3))],
    )
    def test_cardinality(self, input_shape, expected_output_shape):
        self._test_cardinality_implementation(
            input_shape=input_shape,
            expected_output_shape=expected_output_shape,
            params={"size": (2, 3), "align_corners": True},
        )

    @pytest.mark.xfail(reason="size=(1,2) results in RuntimeError: solve_cpu: For batch 0: U(3,3) is zero, singular U.")
    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.0, 0.1, 0.2]]], device=self.device, dtype=self.dtype
        )
        expected_output = torch.tensor([[[[0.6, 0.7]]]], device=self.device, dtype=self.dtype)

        parameters = {"size": (1, 2), "align_corners": True, "resample": 0}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor, expected_output=expected_output, params=parameters
        )

    def test_batch(self):
        torch.manual_seed(42)

        input_tensor = torch.rand((2, 3, 4, 4), device=self.device, dtype=self.dtype)
        expected_output = input_tensor[:, :, 1:3, 1:3]
        expected_transformation = torch.tensor(
            [[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype
        ).repeat(2, 1, 1)
        parameters = {"size": (2, 2), "align_corners": True, "resample": 0, "cropping_mode": "resample"}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters,
        )

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
    possible_params: dict[str, tuple] = {}
    _augmentation_cls = RandomHorizontalFlip
    _default_param_set: dict[str, Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], device=self.device, dtype=self.dtype
        )
        expected_output = torch.tensor(
            [[[[0.3, 0.2, 0.1], [0.6, 0.5, 0.4], [0.9, 0.8, 0.7]]]], device=self.device, dtype=self.dtype
        )

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor, expected_output=expected_output, params=parameters
        )

    def test_batch(self):
        torch.manual_seed(12)

        input_tensor = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]], device=self.device, dtype=self.dtype
        ).repeat((2, 1, 1, 1))
        expected_output = torch.tensor(
            [[[[0.3, 0.2, 0.1], [0.6, 0.5, 0.4], [0.9, 0.8, 0.7]]]], device=self.device, dtype=self.dtype
        ).repeat((2, 1, 1, 1))
        expected_transformation = torch.tensor(
            [[[-1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype
        ).repeat((2, 1, 1))
        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters,
        )

    @pytest.mark.skip(reason="No special parameters to validate.")
    def test_exception(self):
        pass


class TestRandomVerticalFlipAlternative(CommonTests):
    possible_params: dict[str, tuple] = {}
    _augmentation_cls = RandomVerticalFlip
    _default_param_set: dict[str, Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], device=self.device, dtype=self.dtype
        )
        expected_output = torch.tensor(
            [[[[0.7, 0.8, 0.9], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]]], device=self.device, dtype=self.dtype
        )

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor, expected_output=expected_output, params=parameters
        )

    def test_batch(self):
        torch.manual_seed(12)

        input_tensor = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]], device=self.device, dtype=self.dtype
        ).repeat((2, 1, 1, 1))
        expected_output = torch.tensor(
            [[[[0.7, 0.8, 0.9], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]]], device=self.device, dtype=self.dtype
        ).repeat((2, 1, 1, 1))
        expected_transformation = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, -1.0, 2.0], [0.0, 0.0, 1.0]]], device=self.device, dtype=self.dtype
        ).repeat((2, 1, 1))
        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters,
        )

    @pytest.mark.skip(reason="No special parameters to validate.")
    def test_exception(self):
        pass


class TestRandomRotationAlternative(CommonTests):
    possible_params: dict[str, tuple] = {
        "degrees": (0.0, (-360.0, 360.0), [0.0, 0.0], torch.tensor((-180.0, 180))),
        "resample": (0, Resample.BILINEAR.name, Resample.BILINEAR),
        "align_corners": (False, True),
    }
    _augmentation_cls = RandomRotation
    _default_param_set: dict[str, Any] = {"degrees": (30.0, 30.0), "align_corners": True}

    @pytest.fixture(
        params=default_with_one_parameter_changed(default=_default_param_set, **possible_params), scope="class"
    )
    def param_set(self, request):
        return request.param

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], device=self.device, dtype=self.dtype
        )
        expected_output = torch.tensor(
            [[[[0.3, 0.6, 0.9], [0.2, 0.5, 0.8], [0.1, 0.4, 0.7]]]], device=self.device, dtype=self.dtype
        )

        parameters = {"degrees": (90.0, 90.0), "align_corners": True}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor, expected_output=expected_output, params=parameters
        )

    def test_batch(self):
        if self.dtype == torch.float16:
            pytest.skip('not work for half-precision')

        torch.manual_seed(12)

        input_tensor = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]], device=self.device, dtype=self.dtype
        ).repeat((2, 1, 1, 1))
        expected_output = input_tensor
        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {"degrees": (-360.0, -360.0), "align_corners": True}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters,
        )

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
            self._create_augmentation_from_params(degrees=(-361.0, 360.0))
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(degrees=(-360.0, 361.0))
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(degrees=(360.0, -360.0))


class TestRandomGrayscaleAlternative(CommonTests):

    possible_params: dict[str, tuple] = {}

    _augmentation_cls = RandomGrayscale
    _default_param_set: dict[str, Any] = {}

    @pytest.fixture(params=[_default_param_set], scope="class")
    def param_set(self, request):
        return request.param

    @pytest.mark.parametrize(
        "input_shape,expected_output_shape", [((3, 4, 5), (1, 3, 4, 5)), ((2, 3, 4, 5), (2, 3, 4, 5))]
    )
    def test_cardinality(self, input_shape, expected_output_shape):
        self._test_cardinality_implementation(
            input_shape=input_shape, expected_output_shape=expected_output_shape, params=self._default_param_set
        )

    def test_random_p_1(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.0, 0.1, 0.2]], device=self.device, dtype=self.dtype
        ).repeat(1, 3, 1, 1)
        expected_output = (
            (input_tensor * torch.tensor([0.299, 0.587, 0.114], device=self.device, dtype=self.dtype).view(1, 3, 1, 1))
            .sum(dim=1, keepdim=True)
            .repeat(1, 3, 1, 1)
        )

        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor, expected_output=expected_output, params=parameters
        )

    def test_batch(self):
        torch.manual_seed(42)

        input_tensor = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.0, 0.1, 0.2]], device=self.device, dtype=self.dtype
        ).repeat(2, 3, 1, 1)
        expected_output = (
            (input_tensor * torch.tensor([0.299, 0.587, 0.114], device=self.device, dtype=self.dtype).view(1, 3, 1, 1))
            .sum(dim=1, keepdim=True)
            .repeat(1, 3, 1, 1)
        )

        expected_transformation = kornia.eye_like(3, input_tensor)
        parameters = {}
        self._test_random_p_1_implementation(
            input_tensor=input_tensor,
            expected_output=expected_output,
            expected_transformation=expected_transformation,
            params=parameters,
        )

    @pytest.mark.xfail(reason="No input validation is implemented yet when p=0.")
    def test_exception(self):
        torch.manual_seed(42)

        with pytest.raises(ValueError):
            self._create_augmentation_from_params(p=0.0)(torch.rand((1, 1, 4, 5), device=self.device, dtype=self.dtype))
        with pytest.raises(ValueError):
            self._create_augmentation_from_params(p=1.0)(torch.rand((1, 4, 4, 5), device=self.device, dtype=self.dtype))


class TestRandomHorizontalFlip:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomHorizontalFlip(p=0.5)
        repr = "RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr

    def test_random_hflip(self, device, dtype):

        f = RandomHorizontalFlip(p=1.0)
        f1 = RandomHorizontalFlip(p=0.0)

        input = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]], device=device, dtype=dtype
        )  # 3 x 4

        expected = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype
        )  # 3 x 4

        expected = expected.to(device)

        expected_transform = torch.tensor(
            [[-1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )  # 3 x 3

        identity = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )  # 3 x 3

        assert (f(input) == expected).all()
        assert (f.transform_matrix == expected_transform).all()
        assert (f1(input) == input).all()
        assert (f1.transform_matrix == identity).all()
        assert (f.inverse(expected) == input).all()
        assert (f1.inverse(expected) == expected).all()

    def test_batch_random_hflip(self, device, dtype):

        f = RandomHorizontalFlip(p=1.0)
        f1 = RandomHorizontalFlip(p=0.0)

        input = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor(
            [[[-1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        identity = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        assert (f(input) == expected).all()
        assert (f.transform_matrix == expected_transform).all()
        assert (f1(input) == input).all()
        assert (f1.transform_matrix == identity).all()
        assert (f.inverse(expected) == input).all()
        assert (f1.inverse(expected) == expected).all()

    def test_same_on_batch(self, device, dtype):
        f = RandomHorizontalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()
        assert (f.inverse(res) == input).all()

    def test_sequential(self, device, dtype):

        f = AugmentationSequential(RandomHorizontalFlip(p=1.0), RandomHorizontalFlip(p=1.0))

        input = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor(
            [[[-1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        expected_transform_1 = expected_transform @ expected_transform

        out = f(input)
        assert (out == input).all()
        assert (f.transform_matrix == expected_transform_1).all()
        assert (f.inverse(out) == input).all()

    def test_random_hflip_coord_check(self, device, dtype):

        f = RandomHorizontalFlip(p=1.0)

        input = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 4

        input_coordinates = torch.tensor(
            [
                [
                    [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],  # x coord
                    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # y coord
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 3 x 3

        expected_output = torch.tensor(
            [[[[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0], [12.0, 11.0, 10.0, 9.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 4

        output = f(input)
        transform = f.transform_matrix
        result_coordinates = transform @ input_coordinates
        # NOTE: without rounding it might produce unexpected results
        input_coordinates = input_coordinates.round().long()
        result_coordinates = result_coordinates.round().long()

        # Tensors must have the same shapes and values
        assert output.shape == expected_output.shape
        assert (output == expected_output).all()
        # Transformed indices must not be out of bound
        assert (
            torch.torch.logical_and(result_coordinates[0, 0, :] >= 0, result_coordinates[0, 0, :] < input.shape[-1])
        ).all()
        assert (
            torch.torch.logical_and(result_coordinates[0, 1, :] >= 0, result_coordinates[0, 1, :] < input.shape[-2])
        ).all()
        # Values in the output tensor at the places of transformed indices must
        # have the same value as the input tensor has at the corresponding
        # positions
        assert (
            output[..., result_coordinates[0, 1, :], result_coordinates[0, 0, :]]
            == input[..., input_coordinates[0, 1, :], input_coordinates[0, 0, :]]
        ).all()

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomHorizontalFlip(p=1.0), (input,), raise_exception=True)


class TestRandomVerticalFlip(BaseTester):

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomVerticalFlip(p=0.5)
        repr = "RandomVerticalFlip(p=0.5, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr

    def test_random_vflip(self, device, dtype):

        f = RandomVerticalFlip(p=1.0)
        f1 = RandomVerticalFlip(p=0.0)

        input = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected = torch.tensor(
            [[[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, -1.0, 2.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 3 x 3

        identity = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 3 x 3

        self.assert_close(f(input), expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)
        self.assert_close(f1(input), input, low_tolerance=True)
        self.assert_close(f1.transform_matrix, identity, low_tolerance=True)

        self.assert_close(f.inverse(expected), input, low_tolerance=True)
        self.assert_close(f1.inverse(input), input, low_tolerance=True)

    def test_batch_random_vflip(self, device, dtype):

        f = RandomVerticalFlip(p=1.0)

        input = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected = torch.tensor(
            [[[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, -1.0, 2.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        identity = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        self.assert_close(f(input), expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)
        self.assert_close(f.inverse(expected), input, low_tolerance=True)

    def test_same_on_batch(self, device, dtype):
        f = RandomVerticalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()
        assert (f.inverse(res) == input).all()

    def test_sequential(self, device, dtype):

        f = AugmentationSequential(RandomVerticalFlip(p=1.0), RandomVerticalFlip(p=1.0))

        input = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3

        expected_transform = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, -1.0, 2.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )  # 1 x 3 x 3

        expected_transform_1 = expected_transform @ expected_transform

        self.assert_close(f(input), input, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform_1, low_tolerance=True)

    def test_random_vflip_coord_check(self, device, dtype):

        f = RandomVerticalFlip(p=1.0)

        input = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 4

        input_coordinates = torch.tensor(
            [
                [
                    [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],  # x coord
                    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # y coord
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 3 x 3

        expected_output = torch.tensor(
            [[[[9.0, 10.0, 11.0, 12.0], [5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 4

        output = f(input)
        transform = f.transform_matrix
        result_coordinates = transform @ input_coordinates
        # NOTE: without rounding it might produce unexpected results
        input_coordinates = input_coordinates.round().long()
        result_coordinates = result_coordinates.round().long()

        # Tensors must have the same shapes and values
        assert output.shape == expected_output.shape
        assert (output == expected_output).all()
        # Transformed indices must not be out of bound
        assert (
            torch.torch.logical_and(result_coordinates[0, 0, :] >= 0, result_coordinates[0, 0, :] < input.shape[-1])
        ).all()
        assert (
            torch.torch.logical_and(result_coordinates[0, 1, :] >= 0, result_coordinates[0, 1, :] < input.shape[-2])
        ).all()
        # Values in the output tensor at the places of transformed indices must
        # have the same value as the input tensor has at the corresponding
        # positions
        assert (
            output[..., result_coordinates[0, 1, :], result_coordinates[0, 0, :]]
            == input[..., input_coordinates[0, 1, :], input_coordinates[0, 0, :]]
        ).all()

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestColorJiggle(BaseTester):

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = ColorJiggle(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1)
        repr = (
            "ColorJiggle(brightness=tensor([0.5000, 1.5000]), contrast=tensor([0.7000, 1.3000]), "
            "saturation=tensor([0.2000, 1.2000]), hue=tensor([-0.1000,  0.1000]), "
            "p=1.0, p_batch=1.0, same_on_batch=False)"
        )
        assert str(f) == repr

    def test_color_jiggle(self, device, dtype):

        f = ColorJiggle()

        input = torch.rand(3, 5, 5, device=device, dtype=dtype).unsqueeze(0)  # 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        self.assert_close(f(input), expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)

    def test_color_jiggle_batch(self, device, dtype):
        f = ColorJiggle()

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        self.assert_close(f(input), expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)

    def test_same_on_batch(self, device, dtype):
        f = ColorJiggle(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def _get_expected_brightness(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.2529, 0.3529, 0.4529], [0.7529, 0.6529, 0.5529], [0.8529, 0.9529, 1.0000]],
                    [[0.2529, 0.3529, 0.4529], [0.7529, 0.6529, 0.5529], [0.8529, 0.9529, 1.0000]],
                    [[0.2529, 0.3529, 0.4529], [0.7529, 0.6529, 0.5529], [0.8529, 0.9529, 1.0000]],
                ],
                [
                    [[0.2660, 0.3660, 0.4660], [0.7660, 0.6660, 0.5660], [0.8660, 0.9660, 1.0000]],
                    [[0.2660, 0.3660, 0.4660], [0.7660, 0.6660, 0.5660], [0.8660, 0.9660, 1.0000]],
                    [[0.2660, 0.3660, 0.4660], [0.7660, 0.6660, 0.5660], [0.8660, 0.9660, 1.0000]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_brightness(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(brightness=0.2)

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_brightness(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_brightness_tuple(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(brightness=(0.8, 1.2))

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_brightness(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def _get_expected_contrast(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.0953, 0.1906, 0.2859], [0.5719, 0.4766, 0.3813], [0.6672, 0.7625, 0.9531]],
                    [[0.0953, 0.1906, 0.2859], [0.5719, 0.4766, 0.3813], [0.6672, 0.7625, 0.9531]],
                    [[0.0953, 0.1906, 0.2859], [0.5719, 0.4766, 0.3813], [0.6672, 0.7625, 0.9531]],
                ],
                [
                    [[0.1184, 0.2367, 0.3551], [0.7102, 0.5919, 0.4735], [0.8286, 0.9470, 1.0000]],
                    [[0.1184, 0.2367, 0.3551], [0.7102, 0.5919, 0.4735], [0.8286, 0.9470, 1.0000]],
                    [[0.1184, 0.2367, 0.3551], [0.7102, 0.5919, 0.4735], [0.8286, 0.9470, 1.0000]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_contrast(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(contrast=0.2)

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_contrast(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_contrast_list(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(contrast=[0.8, 1.2])

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_contrast(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def _get_expected_saturation(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.1876, 0.2584, 0.3389], [0.6292, 0.5000, 0.4000], [0.7097, 0.8000, 1.0000]],
                    [[1.0000, 0.5292, 0.6097], [0.6292, 0.3195, 0.2195], [0.8000, 0.1682, 0.2779]],
                    [[0.6389, 0.8000, 0.7000], [0.9000, 0.3195, 0.2195], [0.8000, 0.4389, 0.5487]],
                ],
                [
                    [[0.0000, 0.1295, 0.2530], [0.5648, 0.5000, 0.4000], [0.6883, 0.8000, 1.0000]],
                    [[1.0000, 0.4648, 0.5883], [0.5648, 0.2765, 0.1765], [0.8000, 0.0178, 0.1060]],
                    [[0.5556, 0.8000, 0.7000], [0.9000, 0.2765, 0.1765], [0.8000, 0.3530, 0.4413]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_saturation(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(saturation=0.2)

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)
        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_saturation_tensor(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(saturation=torch.tensor(0.2))

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_saturation_tuple(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(saturation=(0.8, 1.2))

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def _get_expected_hue(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.1000, 0.2000, 0.3000], [0.6000, 0.5000, 0.4000], [0.7000, 0.8000, 1.0000]],
                    [[1.0000, 0.5251, 0.6167], [0.6126, 0.3000, 0.2000], [0.8000, 0.1000, 0.2000]],
                    [[0.5623, 0.8000, 0.7000], [0.9000, 0.3084, 0.2084], [0.7958, 0.4293, 0.5335]],
                ],
                [
                    [[0.1000, 0.2000, 0.3000], [0.6116, 0.5000, 0.4000], [0.7000, 0.8000, 1.0000]],
                    [[1.0000, 0.4769, 0.5846], [0.6000, 0.3077, 0.2077], [0.7961, 0.1000, 0.2000]],
                    [[0.6347, 0.8000, 0.7000], [0.9000, 0.3000, 0.2000], [0.8000, 0.3730, 0.4692]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_hue(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(hue=0.1 / pi.item())

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_hue(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_hue_list(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(hue=[-0.1 / pi, 0.1 / pi])

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_hue(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_sequential(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        f = AugmentationSequential(ColorJiggle(), ColorJiggle())

        input = torch.rand(3, 5, 5, device=device, dtype=dtype).unsqueeze(0)  # 1 x 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        self.assert_close(f(input), expected)
        self.assert_close(f.transform_matrix, expected_transform)

    def test_color_jitter_batch_sequential(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        f = AugmentationSequential(ColorJiggle(), ColorJiggle())

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        self.assert_close(f(input), expected, low_tolerance=True)
        self.assert_close(f(input), expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 5, 5), device=device, dtype=dtype).unsqueeze(0)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(ColorJiggle(p=1.0), (input,), raise_exception=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestColorJitter(BaseTester):

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = ColorJitter(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1)
        repr = (
            "ColorJitter(brightness=tensor([0.5000, 1.5000]), contrast=tensor([0.7000, 1.3000]), "
            "saturation=tensor([0.2000, 1.2000]), hue=tensor([-0.1000,  0.1000]), "
            "p=1.0, p_batch=1.0, same_on_batch=False)"
        )
        assert str(f) == repr

    def test_color_jitter(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        f = ColorJitter()

        input = torch.rand(3, 5, 5, device=device, dtype=dtype).unsqueeze(0)  # 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        self.assert_close(f(input), expected)
        self.assert_close(f.transform_matrix, expected_transform)

    def test_color_jitter_batch(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        f = ColorJitter()

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        self.assert_close(f(input), expected)
        self.assert_close(f.transform_matrix, expected_transform)

    def test_same_on_batch(self, device, dtype):
        f = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def _get_expected_brightness(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.1153, 0.2306, 0.3459], [0.6917, 0.5764, 0.4612], [0.8070, 0.9223, 1.0000]],
                    [[0.1153, 0.2306, 0.3459], [0.6917, 0.5764, 0.4612], [0.8070, 0.9223, 1.0000]],
                    [[0.1153, 0.2306, 0.3459], [0.6917, 0.5764, 0.4612], [0.8070, 0.9223, 1.0000]],
                ],
                [
                    [[0.1166, 0.2332, 0.3498], [0.6996, 0.5830, 0.4664], [0.8162, 0.9328, 1.0000]],
                    [[0.1166, 0.2332, 0.3498], [0.6996, 0.5830, 0.4664], [0.8162, 0.9328, 1.0000]],
                    [[0.1166, 0.2332, 0.3498], [0.6996, 0.5830, 0.4664], [0.8162, 0.9328, 1.0000]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_brightness(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(brightness=0.2)

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_brightness(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_brightness_tuple(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(brightness=(0.8, 1.2))

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_brightness(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def _get_expected_contrast(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.1193, 0.2146, 0.3099], [0.5958, 0.5005, 0.4052], [0.6911, 0.7865, 0.9771]],
                    [[0.1193, 0.2146, 0.3099], [0.5958, 0.5005, 0.4052], [0.6911, 0.7865, 0.9771]],
                    [[0.1193, 0.2146, 0.3099], [0.5958, 0.5005, 0.4052], [0.6911, 0.7865, 0.9771]],
                ],
                [
                    [[0.0245, 0.1428, 0.2612], [0.6163, 0.4980, 0.3796], [0.7347, 0.8531, 1.0000]],
                    [[0.0245, 0.1428, 0.2612], [0.6163, 0.4980, 0.3796], [0.7347, 0.8531, 1.0000]],
                    [[0.0245, 0.1428, 0.2612], [0.6163, 0.4980, 0.3796], [0.7347, 0.8531, 1.0000]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_contrast(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(contrast=0.2)

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_contrast(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_contrast_list(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(contrast=[0.8, 1.2])

        input = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]]]], device=device, dtype=dtype
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_contrast(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def _get_expected_saturation(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.1570, 0.2238, 0.3216], [0.6033, 0.4863, 0.3863], [0.7068, 0.7555, 0.9487]],
                    [[0.9693, 0.4946, 0.5924], [0.6033, 0.3058, 0.2058], [0.7971, 0.1237, 0.2266]],
                    [[0.6083, 0.7654, 0.6826], [0.8741, 0.3058, 0.2058], [0.7971, 0.3945, 0.4974]],
                ],
                [
                    [[0.0312, 0.1713, 0.2740], [0.5960, 0.5165, 0.4165], [0.6918, 0.8536, 1.0000]],
                    [[1.0000, 0.5065, 0.6092], [0.5960, 0.2930, 0.1930], [0.8035, 0.0714, 0.1679]],
                    [[0.5900, 0.8418, 0.7210], [0.9312, 0.2930, 0.1930], [0.8035, 0.4066, 0.5031]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_saturation(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(saturation=0.2)

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)
        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_saturation_tensor(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(saturation=torch.tensor(0.2))

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_saturation_tuple(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(saturation=(0.8, 1.2))

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_saturation(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def _get_expected_hue(self, device, dtype):
        return torch.tensor(
            [
                [
                    [[0.1000, 0.2000, 0.3000], [0.6000, 0.5000, 0.4000], [0.7000, 0.8000, 1.0000]],
                    [[1.0000, 0.5251, 0.6167], [0.6126, 0.3000, 0.2000], [0.8000, 0.1000, 0.2000]],
                    [[0.5623, 0.8000, 0.7000], [0.9000, 0.3084, 0.2084], [0.7958, 0.4293, 0.5335]],
                ],
                [
                    [[0.1000, 0.2000, 0.3000], [0.6116, 0.5000, 0.4000], [0.7000, 0.8000, 1.0000]],
                    [[1.0000, 0.4769, 0.5846], [0.6000, 0.3077, 0.2077], [0.7961, 0.1000, 0.2000]],
                    [[0.6347, 0.8000, 0.7000], [0.9000, 0.3000, 0.2000], [0.8000, 0.3730, 0.4692]],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def test_random_hue(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(hue=0.1 / pi.item())

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_hue(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_random_hue_list(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJiggle(hue=[-0.1 / pi, 0.1 / pi])

        input = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 1.0]],
                    [[1.0, 0.5, 0.6], [0.6, 0.3, 0.2], [0.8, 0.1, 0.2]],
                    [[0.6, 0.8, 0.7], [0.9, 0.3, 0.2], [0.8, 0.4, 0.5]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = self._get_expected_hue(device, dtype)

        self.assert_close(f(input), expected, low_tolerance=True)

    def test_sequential(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        f = AugmentationSequential(ColorJiggle(), ColorJiggle())

        input = torch.rand(3, 5, 5, device=device, dtype=dtype).unsqueeze(0)  # 1 x 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        self.assert_close(f(input), expected)
        self.assert_close(f.transform_matrix, expected_transform)

    def test_color_jitter_batch_sequential(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        f = AugmentationSequential(ColorJitter(), ColorJitter())

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        self.assert_close(f(input), expected)
        self.assert_close(f(input), expected)
        self.assert_close(f.transform_matrix, expected_transform)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 5, 5), device=device, dtype=dtype).unsqueeze(0)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(ColorJitter(p=1.0), (input,), raise_exception=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRectangleRandomErasing(BaseTester):
    @pytest.mark.parametrize("erase_scale_range", [(0.001, 0.001), (1.0, 1.0)])
    @pytest.mark.parametrize("aspect_ratio_range", [(0.1, 0.1), (10.0, 10.0)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_random_rectangle_erasing_shape(self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=1.0)
        assert rand_rec(input).shape == batch_shape

    @pytest.mark.parametrize("erase_scale_range", [(0.001, 0.001), (1.0, 1.0)])
    @pytest.mark.parametrize("aspect_ratio_range", [(0.1, 0.1), (10.0, 10.0)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_no_rectangle_erasing_shape(self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=0.0)
        assert rand_rec(input).equal(input)

    @pytest.mark.parametrize("erase_scale_range", [(0.001, 0.001), (1.0, 1.0)])
    @pytest.mark.parametrize("aspect_ratio_range", [(0.1, 0.1), (10.0, 10.0)])
    @pytest.mark.parametrize("shape", [(3, 11, 7)])
    def test_same_on_batch(self, shape, erase_scale_range, aspect_ratio_range):
        f = RandomErasing(erase_scale_range, aspect_ratio_range, same_on_batch=True, p=0.5)
        input = torch.rand(shape).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        erase_scale_range = (0.2, 0.4)
        aspect_ratio_range = (0.3, 0.5)

        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=1.0)
        rect_params = rand_rec.forward_parameters(batch_shape)

        # evaluate function gradient
        input = torch.rand(batch_shape, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(rand_rec, (input, rect_params), raise_exception=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomGrayscale(BaseTester):

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomGrayscale()
        repr = "RandomGrayscale(p=0.1, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr

    def test_random_grayscale(self, device, dtype):

        f = RandomGrayscale()

        input = torch.rand(3, 5, 5, device=device, dtype=dtype)  # 3 x 5 x 5

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3
        f(input)
        self.assert_close(f.transform_matrix, expected_transform)

    def test_same_on_batch(self, device, dtype):
        f = RandomGrayscale(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_opencv_true(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [
                        [0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                        [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                        [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                        [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                        [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593],
                    ],
                    [
                        [0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                        [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                        [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                        [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                        [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426],
                    ],
                    [
                        [0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                        [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                        [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                        [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                        [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor(
            [
                [
                    [
                        [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                        [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                        [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                        [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                        [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                    ],
                    [
                        [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                        [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                        [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                        [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                        [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                    ],
                    [
                        [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                        [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                        [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                        [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                        [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        img_gray = RandomGrayscale(p=1.0)(data)
        self.assert_close(img_gray, expected)

    def test_opencv_false(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [
                        [0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                        [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                        [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                        [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                        [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593],
                    ],
                    [
                        [0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                        [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                        [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                        [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                        [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426],
                    ],
                    [
                        [0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                        [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                        [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                        [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                        [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = data

        img_gray = RandomGrayscale(p=0.0)(data)
        self.assert_close(img_gray, expected)

    def test_opencv_true_batch(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                    [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                    [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                    [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                    [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593],
                ],
                [
                    [0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                    [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                    [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                    [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                    [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426],
                ],
                [
                    [0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                    [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                    [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                    [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                    [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        data = data.unsqueeze(0).repeat(4, 1, 1, 1)

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor(
            [
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ],
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ],
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        expected = expected.unsqueeze(0).repeat(4, 1, 1, 1)

        img_gray = RandomGrayscale(p=1.0)(data)
        self.assert_close(img_gray, expected)

    def test_opencv_false_batch(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                    [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                    [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                    [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                    [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593],
                ],
                [
                    [0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                    [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                    [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                    [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                    [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426],
                ],
                [
                    [0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                    [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                    [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                    [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                    [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        data = data.unsqueeze(0).repeat(4, 1, 1, 1)

        expected = data

        img_gray = RandomGrayscale(p=0.0)(data)
        self.assert_close(img_gray, expected)

    def test_random_grayscale_sequential_batch(self, device, dtype):
        f = AugmentationSequential(RandomGrayscale(p=0.0), RandomGrayscale(p=0.0))

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3
        expected_transform = expected_transform.to(device)

        self.assert_close(f(input), expected)
        self.assert_close(f.transform_matrix, expected_transform)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 5, 5), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomGrayscale(p=1.0), (input,), raise_exception=True)
        assert gradcheck(RandomGrayscale(p=0.0), (input,), raise_exception=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestCenterCrop(BaseTester):
    def test_no_transform(self, device, dtype):
        inp = torch.rand(1, 2, 4, 4, device=device, dtype=dtype)
        out = CenterCrop(2)(inp)
        assert out.shape == (1, 2, 2, 2)
        aug = CenterCrop(2, cropping_mode="resample")
        out = aug(inp)
        assert out.shape == (1, 2, 2, 2)
        assert aug.inverse(out).shape == (1, 2, 4, 4)

    def test_transform(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        aug = CenterCrop(2)
        out = aug(inp)
        assert out.shape == (1, 2, 2, 2)
        assert aug.transform_matrix.shape == (1, 3, 3)
        aug = CenterCrop(2, cropping_mode="resample")
        out = aug(inp)
        assert out.shape == (1, 2, 2, 2)
        assert aug.transform_matrix.shape == (1, 3, 3)
        assert aug.inverse(out).shape == (1, 2, 5, 4)

    def test_no_transform_tuple(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        out = CenterCrop((3, 4))(inp)
        assert out.shape == (1, 2, 3, 4)
        aug = CenterCrop((3, 4), cropping_mode="resample")
        out = aug(inp)
        assert out.shape == (1, 2, 3, 4)
        assert aug.inverse(out).shape == (1, 2, 5, 4)

    def test_crop_modes(self, device, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)

        op1 = CenterCrop(size=(2, 2), cropping_mode='resample')
        out = op1(img)

        op2 = CenterCrop(size=(2, 2), cropping_mode='slice')

        self.assert_close(out, op2(img, op1._params))

    def test_gradcheck(self, device, dtype):
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(CenterCrop(3), (input,), raise_exception=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomRotation(BaseTester):

    torch.manual_seed(0)  # for random reproductibility

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomRotation(degrees=45.5)
        repr = (
            "RandomRotation(degrees=tensor([-45.5000,  45.5000]), interpolation=BILINEAR, p=0.5, "
            "p_batch=1.0, same_on_batch=False)"
        )
        assert str(f) == repr

    def test_random_rotation(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, p=1.0)

        input = torch.tensor(
            [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
            device=device,
            dtype=dtype,
        )  # 4 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [0.9824, 0.0088, 0.0000, 1.9649],
                        [0.0000, 0.0029, 0.0000, 0.0176],
                        [0.0029, 1.0000, 1.9883, 0.0000],
                        [0.0000, 0.0088, 1.0117, 1.9649],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 4 x 4

        expected_transform = torch.tensor(
            [[[1.0000, -0.0059, 0.0088], [0.0059, 1.0000, -0.0088], [0.0000, 0.0000, 1.0000]]],
            device=device,
            dtype=dtype,
        )  # 1 x 3 x 3

        out = f(input)
        self.assert_close(out, expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)

    def test_batch_random_rotation(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, p=1.0)

        input = torch.tensor(
            [[[[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]]]],
            device=device,
            dtype=dtype,
        )  # 1 x 1 x 4 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [0.9824, 0.0088, 0.0000, 1.9649],
                        [0.0000, 0.0029, 0.0000, 0.0176],
                        [0.0029, 1.0000, 1.9883, 0.0000],
                        [0.0000, 0.0088, 1.0117, 1.9649],
                    ]
                ],
                [
                    [
                        [0.1322, 0.0000, 0.7570, 0.2644],
                        [0.3785, 0.0000, 0.4166, 0.0000],
                        [0.0000, 0.6309, 1.5910, 1.2371],
                        [0.0000, 0.1444, 0.3177, 0.6499],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2 x 1 x 4 x 4

        expected_transform = torch.tensor(
            [
                [[1.0000, -0.0059, 0.0088], [0.0059, 1.0000, -0.0088], [0.0000, 0.0000, 1.0000]],
                [[0.9125, 0.4090, -0.4823], [-0.4090, 0.9125, 0.7446], [0.0000, 0.0000, 1.0000]],
            ],
            device=device,
            dtype=dtype,
        )  # 2 x 3 x 3

        input = input.repeat(2, 1, 1, 1)  # 5 x 3 x 3 x 3

        out = f(input)
        self.assert_close(out, expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation(degrees=40, same_on_batch=True)
        input = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = AugmentationSequential(RandomRotation(torch.tensor([-45.0, 90]), p=1.0), RandomRotation(10.4, p=1.0))

        input = torch.tensor(
            [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
            device=device,
            dtype=dtype,
        )  # 4 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [0.1314, 0.1050, 0.6649, 0.2628],
                        [0.3234, 0.0202, 0.4256, 0.1671],
                        [0.0525, 0.5976, 1.5199, 1.1306],
                        [0.0000, 0.1453, 0.3224, 0.5796],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )  # 1 x 4 x 4

        expected_transform = torch.tensor(
            [[[0.8864, 0.4629, -0.5240], [-0.4629, 0.8864, 0.8647], [0.0000, 0.0000, 1.0000]]],
            device=device,
            dtype=dtype,
        )  # 1 x 3 x 3

        out = f(input)
        self.assert_close(out, expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, expected_transform, low_tolerance=True)

    def test_gradcheck(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomRotation(degrees=(15.0, 15.0), p=1.0), (input,), raise_exception=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomCrop(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False, p=1.0)
        repr = (
            "RandomCrop(crop_size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False, padding_mode=constant, "
            "resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False)"
        )
        assert str(f) == repr

    def test_no_padding(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.0)
        out = rc(inp)

        torch.manual_seed(0)
        out2 = rc(inp.squeeze())

        self.assert_close(out, expected)
        self.assert_close(out2, expected)
        torch.manual_seed(0)
        inversed = torch.tensor([[[[0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        aug = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.0, cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inversed)

    def test_no_padding_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]], [[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype
        )
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.0)
        out = rc(inp)
        self.assert_close(out, expected)

        torch.manual_seed(42)
        inversed = torch.tensor(
            [
                [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]],
            ],
            device=device,
            dtype=dtype,
        )
        aug = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.0, cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inversed)

    def test_same_on_batch(self, device, dtype):
        f = RandomCrop(size=(2, 3), padding=1, same_on_batch=True, align_corners=True, p=1.0)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_padding(self, device, dtype):
        torch.manual_seed(42)
        inp = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[7.0, 8.0, 7.0], [4.0, 5.0, 4.0]]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=1, padding_mode='reflect', align_corners=True, p=1.0)
        out = rc(inp)

        torch.manual_seed(42)
        out2 = rc(inp.squeeze())

        self.assert_close(out, expected)
        self.assert_close(out2, expected)
        torch.manual_seed(42)
        inversed = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        aug = RandomCrop(
            size=(2, 3), padding=1, padding_mode='reflect', align_corners=True, p=1.0, cropping_mode="resample"
        )
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out, padding_mode="constant"), inversed)

    def test_padding_batch_1(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]], [[[7.0, 8.0, 0.0], [0.0, 0.0, 0.0]]]], device=device, dtype=dtype
        )
        rc = RandomCrop(size=(2, 3), padding=1, align_corners=True, p=1.0)
        out = rc(inp)

        self.assert_close(out, expected)

        torch.manual_seed(42)
        inversed = torch.tensor(
            [
                [[[0.0, 1.0, 2.0], [0.0, 4.0, 5.0], [0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0]]],
            ],
            device=device,
            dtype=dtype,
        )
        aug = RandomCrop(size=(2, 3), padding=1, align_corners=True, p=1.0, cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inversed)

    def test_padding_batch_2(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[1.0, 2.0, 10.0], [4.0, 5.0, 10.0]]], [[[4.0, 5.0, 10.0], [7.0, 8.0, 10.0]]]], device=device, dtype=dtype
        )
        rc = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, align_corners=True, p=1.0)
        out = rc(inp)

        self.assert_close(out, expected)
        torch.manual_seed(42)
        inversed = torch.tensor(
            [
                [[[0.0, 1.0, 2.0], [0.0, 4.0, 5.0], [0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0], [0.0, 4.0, 5.0], [0.0, 7.0, 8.0]]],
            ],
            device=device,
            dtype=dtype,
        )
        aug = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, align_corners=True, p=1.0, cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inversed)

    def test_padding_batch_3(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[8.0, 8.0, 8.0], [8.0, 0.0, 1.0]]], [[[8.0, 8.0, 8.0], [1.0, 2.0, 8.0]]]], device=device, dtype=dtype
        )
        rc = RandomCrop(size=(2, 3), padding=(0, 1, 2, 3), fill=8, align_corners=True, p=1.0)
        out = rc(inp)

        self.assert_close(out, expected, low_tolerance=True)

        torch.manual_seed(0)
        inversed = torch.tensor(
            [
                [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )
        aug = RandomCrop(size=(2, 3), padding=(0, 1, 2, 3), fill=8, align_corners=True, p=1.0, cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected, low_tolerance=True)
        self.assert_close(aug.inverse(out), inversed, low_tolerance=True)

    def test_padding_no_forward(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        trans = torch.eye(3, device=device, dtype=dtype)[None]
        # Not return transform
        rc = RandomCrop(size=(2, 3), padding=(0, 1, 2, 3), fill=9, align_corners=True, p=0.0, cropping_mode="resample")

        out = rc(inp)
        self.assert_close(out, inp)
        self.assert_close(rc.transform_matrix, trans)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[[0.0, 1.0, 2.0]]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor(
            [[[[9.0, 9.0, 9.0], [0.0, 1.0, 2.0]]], [[[9.0, 9.0, 9.0], [0.0, 1.0, 2.0]]]], device=device, dtype=dtype
        )
        rc = RandomCrop(size=(2, 3), pad_if_needed=True, fill=9, align_corners=True, p=1.0)
        out = rc(inp)

        self.assert_close(out, expected)

        torch.manual_seed(0)
        inversed = torch.tensor([[[[0.0, 1.0, 2.0]]], [[[0.0, 1.0, 2.0]]]], device=device, dtype=dtype)
        aug = RandomCrop(size=(2, 3), pad_if_needed=True, fill=9, align_corners=True, p=1.0, cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inversed)

    def test_crop_modes(self, device, dtype):
        torch.manual_seed(0)
        img = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype)

        op1 = RandomCrop(size=(2, 2), cropping_mode='resample')
        out = op1(img)

        op2 = RandomCrop(size=(2, 2), cropping_mode='slice')

        self.assert_close(out, op2(img, op1._params))

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomCrop(size=(3, 3), p=1.0), (inp,), raise_exception=True)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.0).forward
        op_script = torch.jit.script(op)
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        actual = op_script(img)
        expected = kornia.geometry.transform.center_crop3d(img)
        self.assert_close(actual, expected)

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
        self.assert_close(actual, expected)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomResizedCrop(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomResizedCrop(size=(2, 3), scale=(1.0, 1.0), ratio=(1.0, 1.0))
        repr = (
            "RandomResizedCrop(size=(2, 3), scale=tensor([1., 1.]), ratio=tensor([1., 1.]), "
            "interpolation=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False)"
        )
        assert str(f) == repr

    def test_no_resize(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[0.0000, 1.0000, 2.0000], [6.0000, 7.0000, 8.0000]]]], device=device, dtype=dtype)

        rrc = RandomResizedCrop(size=(2, 3), scale=(1.0, 1.0), ratio=(1.0, 1.0))
        # It will crop a size of (2, 3) from the aspect ratio implementation of torch
        out = rrc(inp)
        self.assert_close(out, expected)

        torch.manual_seed(0)
        aug = RandomResizedCrop(size=(2, 3), scale=(1.0, 1.0), ratio=(1.0, 1.0), cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inp[None])

    def test_same_on_batch(self, device, dtype):
        f = RandomResizedCrop(size=(2, 3), scale=(1.0, 1.0), ratio=(1.0, 1.0), same_on_batch=True)
        input = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        res = f(input)
        self.assert_close(res[0], res[1])

        torch.manual_seed(0)
        aug = RandomResizedCrop(
            size=(2, 3), scale=(1.0, 1.0), ratio=(1.0, 1.0), same_on_batch=True, cropping_mode="resample"
        )
        out = aug(input)
        inversed = aug.inverse(out)
        self.assert_close(inversed[0], inversed[1])

    def test_crop_scale_ratio(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[1.0000, 1.5000, 2.0000], [4.0000, 4.5000, 5.0000], [7.0000, 7.5000, 8.0000]]]],
            device=device,
            dtype=dtype,
        )
        rrc = RandomResizedCrop(size=(3, 3), scale=(3.0, 3.0), ratio=(2.0, 2.0))
        # It will crop a size of (3, 3) from the aspect ratio implementation of torch
        out = rrc(inp)
        self.assert_close(out, expected)

        torch.manual_seed(0)
        inversed = torch.tensor([[[[0.0, 1.0, 2.0], [0.0, 4.0, 5.0], [0.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        aug = RandomResizedCrop(size=(3, 3), scale=(3.0, 3.0), ratio=(2.0, 2.0), cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inversed)

    def test_crop_size_greater_than_input(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype)

        exp = torch.tensor(
            [
                [
                    [
                        [1.0000, 1.3333, 1.6667, 2.0000],
                        [3.0000, 3.3333, 3.6667, 4.0000],
                        [5.0000, 5.3333, 5.6667, 6.0000],
                        [7.0000, 7.3333, 7.6667, 8.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        rrc = RandomResizedCrop(size=(4, 4), scale=(3.0, 3.0), ratio=(2.0, 2.0))
        # It will crop a size of (3, 3) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert out.shape == torch.Size([1, 1, 4, 4])
        self.assert_close(out, exp, low_tolerance=True)

        torch.manual_seed(0)
        inversed = torch.tensor([[[[0.0, 1.0, 2.0], [0.0, 4.0, 5.0], [0.0, 7.0, 8.0]]]], device=device, dtype=dtype)
        aug = RandomResizedCrop(size=(4, 4), scale=(3.0, 3.0), ratio=(2.0, 2.0), cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, exp, low_tolerance=True)
        self.assert_close(aug.inverse(out), inversed, low_tolerance=True)

    def test_crop_scale_ratio_batch(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 1, 1
        )

        expected = torch.tensor(
            [
                [[[1.0000, 1.5000, 2.0000], [4.0000, 4.5000, 5.0000], [7.0000, 7.5000, 8.0000]]],
                [[[0.0000, 0.5000, 1.0000], [3.0000, 3.5000, 4.0000], [6.0000, 6.5000, 7.0000]]],
            ],
            device=device,
            dtype=dtype,
        )
        rrc = RandomResizedCrop(size=(3, 3), scale=(3.0, 3.0), ratio=(2.0, 2.0))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        self.assert_close(out, expected)

        torch.manual_seed(0)
        inversed = torch.tensor(
            [
                [[[0.0, 1.0, 2.0], [0.0, 4.0, 5.0], [0.0, 7.0, 8.0]]],
                [[[0.0, 1.0, 0.0], [3.0, 4.0, 0.0], [6.0, 7.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )
        aug = RandomResizedCrop(size=(3, 3), scale=(3.0, 3.0), ratio=(2.0, 2.0), cropping_mode="resample")
        out = aug(inp)
        self.assert_close(out, expected)
        self.assert_close(aug.inverse(out), inversed)

    def test_crop_modes(self, device, dtype):
        torch.manual_seed(0)
        img = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]], device=device, dtype=dtype)

        op1 = RandomResizedCrop(size=(4, 4), cropping_mode='resample')
        out = op1(img)

        op2 = RandomResizedCrop(size=(4, 4), cropping_mode='slice')

        self.assert_close(out, op2(img, op1._params))

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 3), device=device, dtype=dtype)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(
            RandomResizedCrop(size=(3, 3), scale=(1.0, 1.0), ratio=(1.0, 1.0)), (inp,), raise_exception=True
        )

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomEqualize(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing precision.")
    def test_smoke(self, device, dtype):
        f = RandomEqualize(p=0.5)
        repr = "RandomEqualize(p=0.5, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr

    def test_random_equalize(self, device, dtype):
        f = RandomEqualize(p=1.0)
        f1 = RandomEqualize(p=0.0)

        bs, channels, height, width = 1, 3, 20, 20

        inputs = self.build_input(channels, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor(
            [
                0.0000,
                0.07843,
                0.15686,
                0.2353,
                0.3137,
                0.3922,
                0.4706,
                0.5490,
                0.6275,
                0.7059,
                0.7843,
                0.8627,
                0.9412,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ]
        )
        expected = self.build_input(channels, height, width, bs=1, row=row_expected, device=device, dtype=dtype)
        identity = kornia.eye_like(3, expected)  # 3 x 3

        self.assert_close(f(inputs), expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, identity, low_tolerance=True)
        self.assert_close(f1(inputs), inputs, low_tolerance=True)
        self.assert_close(f1.transform_matrix, identity, low_tolerance=True)

    def test_batch_random_equalize(self, device, dtype):
        f = RandomEqualize(p=1.0)
        f1 = RandomEqualize(p=0.0)

        bs, channels, height, width = 2, 3, 20, 20

        inputs = self.build_input(channels, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor(
            [
                0.0000,
                0.07843,
                0.15686,
                0.2353,
                0.3137,
                0.3922,
                0.4706,
                0.5490,
                0.6275,
                0.7059,
                0.7843,
                0.8627,
                0.9412,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ]
        )
        expected = self.build_input(channels, height, width, bs, row=row_expected, device=device, dtype=dtype)

        identity = kornia.eye_like(3, expected)  # 2 x 3 x 3

        self.assert_close(f(inputs), expected, low_tolerance=True)
        self.assert_close(f.transform_matrix, identity, low_tolerance=True)
        self.assert_close(f1(inputs), inputs, low_tolerance=True)
        self.assert_close(f1.transform_matrix, identity, low_tolerance=True)

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

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestGaussianBlur:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0)
        repr = "RandomGaussianBlur(p=1.0, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr


class TestRandomInvert(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.ones(1, 3, 4, 5, device=device, dtype=dtype)
        self.assert_close(RandomInvert(p=1.0)(img), torch.zeros_like(img))

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomChannelShuffle(BaseTester):
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)
        img = torch.arange(1 * 3 * 2 * 2, device=device, dtype=dtype).view(1, 3, 2, 2)

        out_expected = torch.tensor(
            [[[[8.0, 9.0], [10.0, 11.0]], [[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]],
            device=device,
            dtype=dtype,
        )

        aug = RandomChannelShuffle(p=1.0)
        out = aug(img)
        self.assert_close(out, out_expected)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomGaussianNoise:
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        aug = RandomGaussianNoise(p=1.0)
        assert img.shape == aug(img).shape


class TestNormalize(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = Normalize(mean=torch.tensor([1.0]), std=torch.tensor([1.0]))
        repr = "Normalize(mean=torch.tensor([1.]), std=torch.tensor([1.]), p=1., p_batch=1.0, " "same_on_batch=False)"
        assert str(f) == repr

    @pytest.mark.parametrize(
        "mean, std", [((1.0, 1.0, 1.0), (0.5, 0.5, 0.5)), (1.0, 0.5), (torch.tensor([1.0]), torch.tensor([0.5]))]
    )
    def test_random_normalize_different_parameter_types(self, mean, std):
        f = Normalize(mean=mean, std=std, p=1)
        data = torch.ones(2, 3, 256, 313)
        if isinstance(mean, float):
            expected = (data - torch.as_tensor(mean)) / torch.as_tensor(std)
        else:
            expected = (data - torch.as_tensor(mean[0])) / torch.as_tensor(std[0])
        self.assert_close(f(data), expected)

    @staticmethod
    @pytest.mark.parametrize("mean, std", [((1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5, 0.5)), ((1.0, 1.0), (0.5, 0.5))])
    def test_random_normalize_invalid_parameter_shape(mean, std):
        f = Normalize(mean=mean, std=std, p=1.0)
        inputs = torch.arange(0.0, 16.0, step=1).reshape(1, 4, 4).unsqueeze(0)
        with pytest.raises(ValueError):
            f(inputs)

    def test_random_normalize(self, device, dtype):
        f = Normalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=1.0)
        f1 = Normalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=0.0)

        inputs = torch.arange(0.0, 16.0, step=1, device=device, dtype=dtype).reshape(1, 4, 4).unsqueeze(0)

        expected = (inputs - 1) * 2

        identity = kornia.eye_like(3, expected)

        self.assert_close(f(inputs), expected)
        self.assert_close(f.transform_matrix, identity)
        self.assert_close(f1(inputs), inputs)
        self.assert_close(f1.transform_matrix, identity)

    def test_batch_random_normalize(self, device, dtype):
        f = Normalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=1.0)
        f1 = Normalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=0.0)

        inputs = torch.arange(0.0, 16.0 * 2, step=1, device=device, dtype=dtype).reshape(2, 1, 4, 4)

        expected = (inputs - 1) * 2

        identity = kornia.eye_like(3, expected)

        self.assert_close(f(inputs), expected)
        self.assert_close(f.transform_matrix, identity)
        self.assert_close(f1(inputs), inputs)
        self.assert_close(f1.transform_matrix, identity)

    def test_gradcheck(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            Normalize(mean=torch.tensor([1.0]), std=torch.tensor([1.0]), p=1.0), (input,), raise_exception=True
        )

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestDenormalize(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = Denormalize(mean=torch.tensor([1.0]), std=torch.tensor([1.0]))
        repr = "Denormalize(mean=torch.tensor([1.]), std=torch.tensor([1.]), p=1., p_batch=1.0, " "same_on_batch=False)"
        assert str(f) == repr

    def test_random_denormalize(self, device, dtype):
        f = Denormalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=1.0)
        f1 = Denormalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=0.0)

        inputs = torch.arange(0.0, 16.0, step=1, device=device, dtype=dtype).reshape(1, 4, 4).unsqueeze(0)

        expected = inputs / 2 + 1

        identity = kornia.eye_like(3, expected)

        self.assert_close(f(inputs), expected)
        self.assert_close(f.transform_matrix, identity)
        self.assert_close(f1(inputs), inputs)
        self.assert_close(f1.transform_matrix, identity)

    def test_batch_random_denormalize(self, device, dtype):
        f = Denormalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=1.0)
        f1 = Denormalize(mean=torch.tensor([1.0]), std=torch.tensor([0.5]), p=0.0)

        inputs = torch.arange(0.0, 16.0 * 2, step=1, device=device, dtype=dtype).reshape(2, 1, 4, 4)

        expected = inputs / 2 + 1

        identity = kornia.eye_like(3, expected)

        self.assert_close(f(inputs), expected)
        self.assert_close(f.transform_matrix, identity)
        self.assert_close(f1(inputs), inputs)
        self.assert_close(f1.transform_matrix, identity)

    def test_gradcheck(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            Denormalize(mean=torch.tensor([1.0]), std=torch.tensor([1.0]), p=1.0), (input,), raise_exception=True
        )

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomFisheye:
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)
        center_x = torch.tensor([-0.3, 0.3])
        center_y = torch.tensor([-0.3, 0.3])
        gamma = torch.tensor([-1.0, 1.0])
        img = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        aug = RandomFisheye(center_x, center_y, gamma, p=1.0)
        assert img.shape == aug(img).shape

    @pytest.mark.skip(reason="RuntimeError: Jacobian mismatch for output 0 with respect to input 0")
    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)
        center_x = torch.tensor([-0.3, 0.3], device=device, dtype=dtype)
        center_y = torch.tensor([-0.3, 0.3], device=device, dtype=dtype)
        gamma = torch.tensor([-1.0, 1.0], device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        center_x = utils.tensor_to_gradcheck_var(center_x)  # to var
        center_y = utils.tensor_to_gradcheck_var(center_y)  # to var
        gamma = utils.tensor_to_gradcheck_var(gamma)  # to var
        assert gradcheck(RandomFisheye(center_x, center_y, gamma), (img,), raise_exception=True)


class TestRandomElasticTransform:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        aug = RandomElasticTransform(p=1.0)
        assert img.shape == aug(img).shape

    def test_same_on_batch(self, device, dtype):
        f = RandomElasticTransform(p=1.0, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()


class TestRandomThinPlateSpline:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        aug = RandomThinPlateSpline(p=1.0)
        assert img.shape == aug(img).shape


class TestRandomBoxBlur:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        aug = RandomBoxBlur(p=1.0)
        assert img.shape == aug(img).shape


class TestPadTo:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        aug = PadTo(size=(4, 5))
        out = aug(img)
        assert out.shape == (1, 1, 4, 5)
        assert (aug.inverse(out) == img).all()


class TestResize:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 4, 6, device=device, dtype=dtype)
        aug = Resize(size=(4, 5))
        out = aug(img)
        assert out.shape == (1, 1, 4, 5)


class TestSmallestMaxSize:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 4, 6, device=device, dtype=dtype)
        aug = SmallestMaxSize(max_size=2)
        out = aug(img)
        assert out.shape == (1, 1, 2, 3)


class TestLongestMaxSize:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 4, 6, device=device, dtype=dtype)
        aug = LongestMaxSize(max_size=3)
        out = aug(img)
        assert out.shape == (1, 1, 2, 3)


class TestRandomPosterize:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 4, 5, device=device, dtype=dtype)
        aug = RandomPosterize(bits=6, p=1.0).to(device)
        out = aug(img)
        assert out.shape == (1, 1, 4, 5)


class TestRandomPlasma:
    def test_plasma_shadow(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        aug = RandomPlasmaShadow(p=1.0).to(device)
        out = aug(img)
        assert out.shape == (2, 3, 4, 5)

    def test_plasma_brightness(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        aug = RandomPlasmaBrightness(p=1.0).to(device)
        out = aug(img)
        assert out.shape == (2, 3, 4, 5)

    def test_plasma_contrast(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        aug = RandomPlasmaContrast(p=1.0).to(device)
        out = aug(img)
        assert out.shape == (2, 3, 4, 5)


class TestPlanckianJitter(BaseTester):
    def _get_expected_output_blackbody(self, device, dtype):
        return torch.tensor(
            [
                [
                    [
                        [0.7350, 1.0000, 0.1311, 0.1955],
                        [0.4553, 0.9391, 0.7258, 1.0000],
                        [0.6748, 0.9364, 0.5167, 0.5949],
                        [0.0330, 0.2501, 0.4353, 0.7679],
                    ],
                    [
                        [0.6977, 0.8000, 0.1610, 0.2823],
                        [0.6816, 0.9152, 0.3971, 0.8742],
                        [0.4194, 0.5529, 0.9527, 0.0362],
                        [0.1852, 0.3734, 0.3051, 0.9320],
                    ],
                    [
                        [0.0691, 0.1059, 0.0592, 0.0124],
                        [0.0817, 0.3650, 0.2839, 0.2914],
                        [0.2066, 0.0957, 0.2295, 0.0130],
                        [0.0545, 0.0951, 0.3202, 0.3114],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

    def _get_expected_output_cied(self, device, dtype):
        return torch.tensor(
            [
                [
                    [
                        [0.6058, 0.9377, 0.1080, 0.1611],
                        [0.3752, 0.7740, 0.5982, 1.0000],
                        [0.5561, 0.7718, 0.4259, 0.4903],
                        [0.0272, 0.2062, 0.3587, 0.6329],
                    ],
                    [
                        [0.6977, 0.8000, 0.1610, 0.2823],
                        [0.6816, 0.9152, 0.3971, 0.8742],
                        [0.4194, 0.5529, 0.9527, 0.0362],
                        [0.1852, 0.3734, 0.3051, 0.9320],
                    ],
                    [
                        [0.1149, 0.1762, 0.0984, 0.0207],
                        [0.1359, 0.6072, 0.4722, 0.4848],
                        [0.3437, 0.1592, 0.3818, 0.0217],
                        [0.0906, 0.1582, 0.5326, 0.5180],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

    def _get_expected_output_batch(self, device, dtype):
        return torch.tensor(
            [
                [
                    [
                        [0.7350, 1.0000, 0.1311, 0.1955],
                        [0.4553, 0.9391, 0.7258, 1.0000],
                        [0.6748, 0.9364, 0.5167, 0.5949],
                        [0.0330, 0.2501, 0.4353, 0.7679],
                    ],
                    [
                        [0.6977, 0.8000, 0.1610, 0.2823],
                        [0.6816, 0.9152, 0.3971, 0.8742],
                        [0.4194, 0.5529, 0.9527, 0.0362],
                        [0.1852, 0.3734, 0.3051, 0.9320],
                    ],
                    [
                        [0.0691, 0.1059, 0.0592, 0.0124],
                        [0.0817, 0.3650, 0.2839, 0.2914],
                        [0.2066, 0.0957, 0.2295, 0.0130],
                        [0.0545, 0.0951, 0.3202, 0.3114],
                    ],
                ],
                [
                    [
                        [0.4963, 0.7682, 0.0885, 0.1320],
                        [0.3074, 0.6341, 0.4901, 0.8964],
                        [0.4556, 0.6323, 0.3489, 0.4017],
                        [0.0223, 0.1689, 0.2939, 0.5185],
                    ],
                    [
                        [0.6977, 0.8000, 0.1610, 0.2823],
                        [0.6816, 0.9152, 0.3971, 0.8742],
                        [0.4194, 0.5529, 0.9527, 0.0362],
                        [0.1852, 0.3734, 0.3051, 0.9320],
                    ],
                    [
                        [0.1759, 0.2698, 0.1507, 0.0317],
                        [0.2081, 0.9298, 0.7231, 0.7423],
                        [0.5263, 0.2437, 0.5846, 0.0332],
                        [0.1387, 0.2422, 0.8155, 0.7932],
                    ],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def _get_expected_output_same_on_batch(self, device, dtype):
        return torch.tensor(
            [
                [
                    [
                        [0.3736, 0.5783, 0.0666, 0.0994],
                        [0.2314, 0.4774, 0.3690, 0.6749],
                        [0.3430, 0.4760, 0.2627, 0.3024],
                        [0.0168, 0.1272, 0.2213, 0.3904],
                    ],
                    [
                        [0.6977, 0.8000, 0.1610, 0.2823],
                        [0.6816, 0.9152, 0.3971, 0.8742],
                        [0.4194, 0.5529, 0.9527, 0.0362],
                        [0.1852, 0.3734, 0.3051, 0.9320],
                    ],
                    [
                        [0.2621, 0.4020, 0.2245, 0.0472],
                        [0.3101, 1.0000, 1.0000, 1.0000],
                        [0.7842, 0.3631, 0.8711, 0.0495],
                        [0.2067, 0.3609, 1.0000, 1.0000],
                    ],
                ],
                [
                    [
                        [0.3736, 0.5783, 0.0666, 0.0994],
                        [0.2314, 0.4774, 0.3690, 0.6749],
                        [0.3430, 0.4760, 0.2627, 0.3024],
                        [0.0168, 0.1272, 0.2213, 0.3904],
                    ],
                    [
                        [0.6977, 0.8000, 0.1610, 0.2823],
                        [0.6816, 0.9152, 0.3971, 0.8742],
                        [0.4194, 0.5529, 0.9527, 0.0362],
                        [0.1852, 0.3734, 0.3051, 0.9320],
                    ],
                    [
                        [0.2621, 0.4020, 0.2245, 0.0472],
                        [0.3101, 1.0000, 1.0000, 1.0000],
                        [0.7842, 0.3631, 0.8711, 0.0495],
                        [0.2067, 0.3609, 1.0000, 1.0000],
                    ],
                ],
            ],
            device=device,
            dtype=dtype,
        )

    def _get_input(self, device, dtype):
        return torch.tensor(
            [
                [
                    [
                        [0.4963, 0.7682, 0.0885, 0.1320],
                        [0.3074, 0.6341, 0.4901, 0.8964],
                        [0.4556, 0.6323, 0.3489, 0.4017],
                        [0.0223, 0.1689, 0.2939, 0.5185],
                    ],
                    [
                        [0.6977, 0.8000, 0.1610, 0.2823],
                        [0.6816, 0.9152, 0.3971, 0.8742],
                        [0.4194, 0.5529, 0.9527, 0.0362],
                        [0.1852, 0.3734, 0.3051, 0.9320],
                    ],
                    [
                        [0.1759, 0.2698, 0.1507, 0.0317],
                        [0.2081, 0.9298, 0.7231, 0.7423],
                        [0.5263, 0.2437, 0.5846, 0.0332],
                        [0.1387, 0.2422, 0.8155, 0.7932],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

    def test_planckian_jitter_blackbody(self, device, dtype):
        torch.manual_seed(0)
        f = RandomPlanckianJitter(select_from=1).to(device, dtype)
        input = self._get_input(device, dtype)
        expected = self._get_expected_output_blackbody(device, dtype)
        self.assert_close(f(input), expected, low_tolerance=True)

    def test_planckian_jitter_cied(self, device, dtype):
        torch.manual_seed(0)
        f = RandomPlanckianJitter(mode='CIED', select_from=1).to(device, dtype)
        input = self._get_input(device, dtype)
        expected = self._get_expected_output_cied(device, dtype)
        self.assert_close(f(input), expected, low_tolerance=True)

    def test_planckian_jitter_batch(self, device, dtype):
        torch.manual_seed(0)
        input = self._get_input(device, dtype).repeat(2, 1, 1, 1)

        select_from = [1, 2, 24]
        f = RandomPlanckianJitter(select_from=select_from).to(device, dtype)
        expected = self._get_expected_output_batch(device, dtype)
        self.assert_close(f(input), expected, low_tolerance=True)

    def test_planckian_jitter_same_on_batch(self, device, dtype):
        torch.manual_seed(0)
        input = self._get_input(device, dtype).repeat(2, 1, 1, 1)

        select_from = [1, 2, 24, 3, 4, 5]
        f = RandomPlanckianJitter(select_from=select_from, same_on_batch=True, p=1.0).to(device, dtype)
        expected = self._get_expected_output_same_on_batch(device, dtype)
        self.assert_close(f(input), expected, low_tolerance=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass


class TestRandomRGBShift:
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        aug = RandomRGBShift(p=1.0).to(device)
        out = aug(img)
        assert out.shape == (2, 3, 4, 5)

    def test_onnx_export(self, device, dtype):
        img = torch.rand(1, 3, 4, 5, device=device, dtype=dtype)
        aug = RandomRGBShift(p=1.0).to(device)
        torch.onnx.export(aug, img, "temp.onnx", export_params=True)

    def test_random_rgb_shift(self, device, dtype):
        torch.manual_seed(0)
        input = torch.tensor(
            [[[[0.2, 0.0]], [[0.3, 0.5]], [[0.4, 0.7]]], [[[0.2, 0.7]], [[0.0, 0.8]], [[0.2, 0.3]]]],
            device=device,
            dtype=dtype,
        )

        f = RandomRGBShift(p=1.0).to(device)
        expected = torch.tensor(
            [
                [[[0.19625, 0.00000]], [[0.56822, 0.76822]], [[0.00000, 0.28847]]],
                [[[0.00000, 0.33203]], [[0.00000, 0.60742]], [[0.33407, 0.43407]]],
            ],
            device=device,
            dtype=dtype,
        )
        utils.assert_close(f(input), expected, rtol=1e-4, atol=1e-4)

    def test_random_rgb_shift_same_batch(self, device, dtype):
        torch.manual_seed(0)
        input = torch.tensor(
            [[[[0.2, 0.0]], [[0.3, 0.5]], [[0.4, 0.7]]], [[[0.2, 0.7]], [[0.0, 0.8]], [[0.2, 0.3]]]],
            device=device,
            dtype=dtype,
        )

        f = RandomRGBShift(p=1.0, same_on_batch=True).to(device)
        expected = torch.tensor(
            [
                [[[0.19626, 0.00000]], [[0.29626, 0.49626]], [[0.66822, 0.96822]]],
                [[[0.46822, 0.96822]], [[0.00000, 0.38848]], [[0.00000, 0.00000]]],
            ],
            device=device,
            dtype=dtype,
        )
        utils.assert_close(f(input), expected, rtol=1e-4, atol=1e-4)
