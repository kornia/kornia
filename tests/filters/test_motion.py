# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect

import pytest
import torch

from kornia.core.exceptions import BaseError
from kornia.filters import (
    MotionBlur,
    MotionBlur3D,
    box_blur,
    filter2d,
    filter3d,
    gaussian_blur2d,
    get_motion_kernel2d,
    get_motion_kernel3d,
    motion_blur,
    motion_blur3d,
)

from testing.base import (
    BaseTester,
    supports_arange,
    supports_bilinear_2d_grid_sample,
    supports_bilinear_3d_grid_sample,
    supports_nearest_2d_grid_sample,
    supports_nearest_3d_grid_sample,
    supports_reflect_padding,
    supports_replicate_padding_3d,
)


def _default_border_type(op) -> str:
    return inspect.signature(op).parameters["border_type"].default


# The motion filters share their sibling's default border: the 2-D ones that of ``filter2d``, ``gaussian_blur2d`` and
# ``box_blur`` ("reflect"), the 3-D ones that of ``filter3d`` ("replicate").
@pytest.mark.device_agnostic
@pytest.mark.parametrize(
    "op, siblings",
    [
        (motion_blur, (filter2d, gaussian_blur2d, box_blur)),
        (MotionBlur, (filter2d, gaussian_blur2d, box_blur)),
        (motion_blur3d, (filter3d,)),
        (MotionBlur3D, (filter3d,)),
    ],
)
def test_default_border_type_matches_the_sibling_filters(op, siblings):
    assert {_default_border_type(op)} == {_default_border_type(sibling) for sibling in siblings}


class TestMotionBlur(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("angle", [36.0, 200.0])
    @pytest.mark.parametrize("direction", [-0.9, 0.0, 0.9])
    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_smoke(self, shape, kernel_size, angle, direction, mode, params_as_tensor, device, dtype):
        B, _C, _H, _W = shape
        data = torch.rand(shape, device=device, dtype=dtype)

        if params_as_tensor is True:
            angle = torch.tensor([angle], device=device, dtype=dtype).repeat(B)
            direction = torch.tensor([direction], device=device, dtype=dtype).repeat(B)
        actual = motion_blur(data, kernel_size, angle, direction, "constant", mode)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_cardinality(self, shape, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        ksize = 5
        angle = 200.0
        direction = 0.3

        sample = torch.rand(shape, device=device, dtype=dtype)
        motion = MotionBlur(ksize, angle, direction)
        assert motion(sample).shape == shape

    @pytest.mark.skip(reason="nothing to test")
    def test_exception(self): ...

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("ksize", [3, 11])
    @pytest.mark.parametrize("angle", [0.0, 360.0])
    @pytest.mark.parametrize("direction", [-1.0, 1.0])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_get_motion_kernel2d(self, batch_size, ksize, angle, direction, params_as_tensor, device, dtype):
        if params_as_tensor is True:
            angle = torch.tensor([angle], device=device, dtype=dtype).repeat(batch_size)
            direction = torch.tensor([direction], device=device, dtype=dtype).repeat(batch_size)
        else:
            batch_size = 1
            device = None
            dtype = None

        actual = get_motion_kernel2d(ksize, angle, direction)
        expected = torch.ones(1, device=device, dtype=dtype) * batch_size
        assert actual.shape == (batch_size, ksize, ksize)
        self.assert_close(actual.sum(), expected.sum())

    def test_get_motion_kernel2d_directional_weights(self, device, dtype):
        direction = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
        actual = get_motion_kernel2d(5, torch.zeros_like(direction), direction)

        expected = torch.zeros((3, 5, 5), device=device, dtype=dtype)
        expected[0, 2] = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], device=device, dtype=dtype)
        expected[1, 2] = 0.2
        expected[2, 2] = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], device=device, dtype=dtype)
        self.assert_close(actual, expected)

    def test_get_motion_kernel2d_direction_gradcheck(self, device):
        direction = torch.tensor([-0.5, 0.5], device=device, dtype=torch.float64, requires_grad=True)
        angle = torch.zeros_like(direction)
        self.gradcheck(lambda direction: get_motion_kernel2d(5, angle, direction), (direction,))

    def test_get_motion_kernel2d_mismatched_batch_size(self, device, dtype):
        angle = torch.zeros(3, device=device, dtype=dtype)
        direction = torch.zeros(2, device=device, dtype=dtype)
        with pytest.raises(Exception, match=r"direction and angle must have the same length. Got 2 and 3."):
            get_motion_kernel2d(3, angle, direction)

    @pytest.mark.parametrize("kernel_size", [(5, 5), [5, 5], 5.0], ids=["tuple", "list", "float"])
    def test_convention_kernel_size_must_be_an_int_5169(self, kernel_size, device, dtype):
        # The motion kernel is square, so kernel_size is a single int. Anything else raises a kornia error that names
        # the argument, from the kernel builder, the function and the module alike.
        image = torch.rand(1, 1, 8, 9, device=device, dtype=dtype)
        angle = torch.tensor([30.0], device=device, dtype=dtype)
        direction = torch.tensor([0.5], device=device, dtype=dtype)
        calls = (
            lambda: get_motion_kernel2d(kernel_size, angle, direction),
            lambda: motion_blur(image, kernel_size, 30.0, 0.5),
            lambda: MotionBlur(kernel_size, 30.0, 0.5)(image),
        )
        for call in calls:
            with pytest.raises(BaseError, match=f"kernel_size must be an int. Got {type(kernel_size).__name__}"):
                call()

    def test_noncontiguous(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = 3
        angle = 200.0
        direction = 0.3
        actual = motion_blur(inp, kernel_size, angle, direction)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        batch_shape = (1, 3, 4, 5)
        ksize = 9
        angle = 34.0
        direction = -0.2

        sample = torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(motion_blur, (sample, ksize, angle, direction, "replicate"), nondet_tol=1e-8)

    def test_module(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        params = [3, 20.0, 0.5]
        op = motion_blur
        op_module = MotionBlur(*params)
        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)

        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("mode", ["nearest", "bilinear"])
    @pytest.mark.parametrize("params_as_tensor", [False, True])
    def test_module_forwards_mode(self, mode, params_as_tensor, device, dtype):
        image = torch.rand(2, 2, 7, 9, device=device, dtype=dtype)
        angle, direction = 30.0, 0.3
        if params_as_tensor:
            angle = torch.tensor([30.0, 60.0], device=device, dtype=dtype)
            direction = torch.tensor([0.3, -0.5], device=device, dtype=dtype)
        actual = MotionBlur(5, angle, direction, "constant", mode)(image)
        expected = motion_blur(image, 5, angle, direction, "constant", mode)
        self.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_repr_includes_mode(self):
        assert "mode=bilinear" in repr(MotionBlur(3, 30.0, 0.3, mode="bilinear"))

    def test_python_float_parameters_preserve_float64_precision(self):
        image = torch.ones(1, 1, 9, 9, dtype=torch.float64)
        results = [
            motion_blur(image, 5, 30.0, 0.3, "reflect"),
            MotionBlur(5, 30.0, 0.3, "reflect")(image),
        ]
        for result in results:
            assert result.dtype == torch.float64
            assert (result - image).abs().max() < 1e-15

    @pytest.mark.parametrize("angle", [60.0, 120.0, 150.0])
    def test_python_float_parameters_match_cpu_tensor_kernel(self, angle, device, dtype):
        # Python-number parameters build the kernel on the CPU in the input dtype, never below float32.
        kernel_dtype = torch.promote_types(dtype, torch.float32)
        img = torch.rand(1, 2, 9, 9, device=device, dtype=dtype)
        kernel = get_motion_kernel2d(
            7, torch.tensor([angle], dtype=kernel_dtype), torch.tensor([0.3], dtype=kernel_dtype)
        )
        expected = filter2d(img, kernel, "reflect")
        self.assert_close(motion_blur(img, 7, angle, 0.3, "reflect"), expected, rtol=0, atol=0)
        self.assert_close(MotionBlur(7, angle, 0.3, "reflect")(img), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("tensor_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("tensor_parameter", ["angle", "angle_0d", "direction"])
    def test_tensor_parameter_with_a_python_number_is_built_like_the_tensor(
        self, tensor_parameter, tensor_dtype, device, dtype
    ):
        # A tensor angle or direction with a Python number for the other builds the number on the tensor's device and
        # in its dtype, whatever the input's dtype, so the blur is the one of the two tensors.
        if device.type == "mps" and torch.float64 in (dtype, tensor_dtype):
            pytest.skip("MPS has no float64")
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        if not supports_nearest_2d_grid_sample(device, tensor_dtype):
            pytest.skip(f"the kernel is rotated with grid_sample, which this device lacks for {tensor_dtype}")
        if not supports_arange(device, tensor_dtype):
            pytest.skip(f"the kernel taps are built with arange, which this device lacks for {tensor_dtype}")
        torch.manual_seed(0)
        image = torch.rand(1, 2, 9, 11, device=device, dtype=dtype)
        angle = torch.tensor([30.0], device=device, dtype=tensor_dtype)
        direction = torch.tensor([0.5], device=device, dtype=tensor_dtype)
        expected = motion_blur(image, 5, angle, direction)
        if tensor_parameter == "angle":
            params = (angle, 0.5)
        elif tensor_parameter == "angle_0d":
            params = (angle[0], 0.5)
        else:
            params = (30.0, direction)
        actual = motion_blur(image, 5, *params)
        assert actual.dtype == dtype
        assert actual.device == image.device
        self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(MotionBlur(5, *params)(image), expected, rtol=0, atol=0)

    # A blur of a constant image is that constant. ``(1, 1, 3, 9)`` and ``(1, 1, 9, 3)`` are shorter than the
    # 5-tap kernel along one axis.
    @pytest.mark.parametrize("shape", [(1, 1, 5, 7), (2, 3, 8, 6), (1, 1, 3, 9), (1, 1, 9, 3)])
    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 30.0])
    @pytest.mark.parametrize("direction", [0.0, 0.7])
    def test_constant_image_stays_constant_at_the_default_border(self, shape, angle, direction, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        image = torch.full(shape, 0.75, device=device, dtype=dtype)

        self.assert_close(motion_blur(image, 5, angle, direction), image)
        self.assert_close(MotionBlur(5, angle, direction)(image), image)

    # "reflect" needs each spatial axis longer than the kernel radius (``kernel_size // 2 = 2`` here), as it does for
    # ``filter2d``, ``gaussian_blur2d`` and ``box_blur`` at their defaults; "constant" and "replicate" do not.
    @pytest.mark.parametrize("shape", [(1, 1, 2, 9), (1, 1, 9, 2), (1, 1, 1, 9)])
    def test_default_border_raises_on_an_axis_of_at_most_the_kernel_radius(self, shape, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        image = torch.ones(shape, device=device, dtype=dtype)

        with pytest.raises(RuntimeError, match="Padding size should be less than"):
            motion_blur(image, 5, 0.0, 0.0)
        with pytest.raises(RuntimeError, match="Padding size should be less than"):
            MotionBlur(5, 0.0, 0.0)(image)
        assert motion_blur(image, 5, 0.0, 0.0, border_type="constant").shape == shape

    # Snippet used to generate expected:
    #   import torch, kornia.filters as KF
    #   print(KF.motion_blur(torch.ones(1, 1, 5, 7), 5, 0.0, 0.0, "constant")[0, 0, 2].tolist())
    #   -> [0.6, 0.8, 1.0, 1.0, 1.0, 0.8, 0.6]   (zero padding darkens every pixel the kernel reaches past the edge)
    def test_explicit_constant_border_zero_pads(self, device, dtype):
        image = torch.ones(1, 1, 5, 7, device=device, dtype=dtype)
        row = torch.tensor([0.6, 0.8, 1.0, 1.0, 1.0, 0.8, 0.6], device=device, dtype=dtype)
        expected = row.expand(1, 1, 5, 7)

        self.assert_close(motion_blur(image, 5, 0.0, 0.0, border_type="constant"), expected)
        self.assert_close(MotionBlur(5, 0.0, 0.0, border_type="constant")(image), expected)

    def test_function_and_module_agree_at_their_defaults(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(0)
        image = torch.rand(2, 3, 9, 11, device=device, dtype=dtype)
        params = (5, 35.0, 0.5)

        from_function = motion_blur(image, *params)
        self.assert_close(MotionBlur(*params)(image), from_function)
        # the shared default is "reflect"
        self.assert_close(from_function, motion_blur(image, *params, border_type="reflect"))

    @pytest.mark.skip(reason="After the op be optimized the results are not the same")
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        # TODO: FIX op
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MotionBlur(3, 36.0, 0.5)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


class TestMotionBlur3D(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 3, 8, 15), (2, 2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("angle", [(36.0, 15.0, 200.0), (200.0, 10.0, 150.0)])
    @pytest.mark.parametrize("direction", [-0.9, 0.0, 0.9])
    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_smoke(self, shape, kernel_size, angle, direction, mode, params_as_tensor, device, dtype):
        if params_as_tensor:
            supports_mode = (
                supports_nearest_3d_grid_sample(device, dtype)
                if mode == "nearest"
                else supports_bilinear_3d_grid_sample(device, dtype)
            )
            if not supports_mode:
                pytest.skip(f"This device does not support {mode} interpolation for 3D grid_sample")
        B, _C, _D, _H, _W = shape
        data = torch.rand(shape, device=device, dtype=dtype)

        if params_as_tensor is True:
            angle = torch.tensor([angle], device=device, dtype=dtype).expand(B, 3)
            direction = torch.tensor([direction], device=device, dtype=dtype).repeat(B)
        actual = motion_blur3d(data, kernel_size, angle, direction, "constant", mode)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 1, 8, 15), (2, 3, 1, 11, 7)])
    def test_cardinality(self, shape, device, dtype):
        if not supports_replicate_padding_3d(device, dtype):
            pytest.skip("replication_pad3d is unavailable for this device/dtype")
        ksize = 5
        angle = (200.0, 15.0, 120.0)
        direction = 0.3

        sample = torch.rand(shape, device=device, dtype=dtype)
        motion = MotionBlur3D(ksize, angle, direction)
        assert motion(sample).shape == shape

    @pytest.mark.skip(reason="nothing to test")
    def test_exception(self): ...

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("ksize", [3, 11])
    @pytest.mark.parametrize("angle", [(0.0, 360.0, 150.0)])
    @pytest.mark.parametrize("direction", [-1.0, 1.0])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_get_motion_kernel3d(self, batch_size, ksize, angle, direction, params_as_tensor, device, dtype):
        mode = "nearest"
        if params_as_tensor is True:
            if not supports_nearest_3d_grid_sample(device, dtype):
                if not supports_bilinear_3d_grid_sample(device, dtype):
                    pytest.skip("This device does not support a viable interpolation mode for 3D grid_sample")
                # This test checks only the normalized kernel sum, which is independent of interpolation mode.
                mode = "bilinear"
            angle = torch.tensor([angle], device=device, dtype=dtype).repeat(batch_size, 1)
            direction = torch.tensor([direction], device=device, dtype=dtype).repeat(batch_size)
        else:
            batch_size = 1
            device = None
            dtype = None

        actual = get_motion_kernel3d(ksize, angle, direction, mode=mode)
        expected = torch.ones(1, device=device, dtype=dtype) * batch_size
        assert actual.shape == (batch_size, ksize, ksize, ksize)
        self.assert_close(actual.sum(), expected.sum())

    @pytest.mark.parametrize("kernel_size", [(5, 5, 5), [5, 5, 5], 5.0], ids=["tuple", "list", "float"])
    def test_convention_kernel_size_must_be_an_int_5169(self, kernel_size, device, dtype):
        # The motion kernel is cubic, so kernel_size is a single int. Anything else raises a kornia error that names
        # the argument, from the kernel builder, the function and the module alike.
        volume = torch.rand(1, 1, 6, 8, 9, device=device, dtype=dtype)
        angle = torch.tensor([[0.0, 90.0, 90.0]], device=device, dtype=dtype)
        direction = torch.tensor([0.5], device=device, dtype=dtype)
        calls = (
            lambda: get_motion_kernel3d(kernel_size, angle, direction),
            lambda: motion_blur3d(volume, kernel_size, (0.0, 90.0, 90.0), 0.5),
            lambda: MotionBlur3D(kernel_size, (0.0, 90.0, 90.0), 0.5)(volume),
        )
        for call in calls:
            with pytest.raises(BaseError, match=f"kernel_size must be an int. Got {type(kernel_size).__name__}"):
                call()

    def test_noncontiguous(self, device, dtype):
        if not supports_replicate_padding_3d(device, dtype):
            pytest.skip("replication_pad3d is unavailable for this device/dtype")
        batch_size = 3
        inp = torch.rand(3, 1, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1, -1)

        kernel_size = 3
        angle = (0.0, 360.0, 150.0)
        direction = 0.3
        actual = motion_blur3d(inp, kernel_size, angle, direction)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        batch_shape = (1, 3, 1, 4, 5)
        ksize = 9
        angle = (0.0, 360.0, 150.0)
        direction = -0.2

        sample = torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(motion_blur3d, (sample, ksize, angle, direction, "replicate"), nondet_tol=1e-8)

    def test_module(self, device, dtype):
        if not supports_replicate_padding_3d(device, dtype):
            pytest.skip("replication_pad3d is unavailable for this device/dtype")
        params = [3, (0.0, 360.0, 150.0), 0.5]
        op = motion_blur3d
        op_module = MotionBlur3D(*params)
        img = torch.ones(1, 3, 1, 5, 5, device=device, dtype=dtype)

        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("mode", ["nearest", "bilinear"])
    @pytest.mark.parametrize("angle_form", ["float", "int", "tuple", "list", "tensor"])
    def test_module_forwards_mode_and_angle(self, mode, angle_form, device, dtype):
        volume = torch.rand(2, 2, 5, 6, 7, device=device, dtype=dtype)
        direction = 0.3
        expected_angle = (10.0, 20.0, 30.0)
        if angle_form in ("float", "int"):
            angle = 35.0 if angle_form == "float" else 35
            expected_angle = (35.0, 35.0, 35.0)
        elif angle_form == "tensor":
            supports_mode = (
                supports_nearest_3d_grid_sample(device, dtype)
                if mode == "nearest"
                else supports_bilinear_3d_grid_sample(device, dtype)
            )
            if not supports_mode:
                pytest.skip(f"This device does not support {mode} interpolation for 3D grid_sample")
            angle = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device, dtype=dtype)
            expected_angle = angle
            direction = torch.tensor([0.3, -0.5], device=device, dtype=dtype)
        else:
            angle = list(expected_angle) if angle_form == "list" else expected_angle
        actual = MotionBlur3D(3, angle, direction, "constant", mode)(volume)
        expected = motion_blur3d(volume, 3, expected_angle, direction, "constant", mode)
        self.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("angle", [[], [10.0, 20.0], (10.0, 20.0, 30.0, 40.0)])
    def test_module_rejects_wrong_angle_length(self, angle):
        with pytest.raises(BaseError, match="Angle sequence must have length 3"):
            MotionBlur3D(3, angle, 0.3)

    @pytest.mark.device_agnostic
    def test_repr_includes_mode(self):
        assert "mode=bilinear" in repr(MotionBlur3D(3, (10.0, 20.0, 30.0), 0.3, mode="bilinear"))

    def test_module_tensor_parameter_gradients(self, device):
        dtype = torch.float32 if device.type == "mps" else torch.float64
        if not supports_bilinear_3d_grid_sample(device, dtype):
            pytest.skip("This device does not support bilinear interpolation for 3D grid_sample")
        volume = torch.rand(2, 1, 5, 6, 7, device=device, dtype=dtype, requires_grad=True)
        angle = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device, dtype=dtype)
        angle.requires_grad_()
        direction = torch.tensor([0.3, -0.5], device=device, dtype=dtype, requires_grad=True)
        actual = MotionBlur3D(3, angle, direction, "constant", "bilinear")(volume)
        expected = motion_blur3d(volume, 3, angle, direction, "constant", "bilinear")
        actual_grads = torch.autograd.grad(actual.square().sum(), (volume, angle, direction))
        expected_grads = torch.autograd.grad(expected.square().sum(), (volume, angle, direction))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert torch.isfinite(actual_grad).all()
            self.assert_close(actual_grad, expected_grad)

    def test_python_float_parameters_preserve_float64_precision(self):
        volume = torch.ones(1, 1, 5, 6, 7, dtype=torch.float64)
        results = [
            motion_blur3d(volume, 3, (10.0, 20.0, 30.0), 0.3, "replicate"),
            MotionBlur3D(3, (10.0, 20.0, 30.0), 0.3, "replicate")(volume),
        ]
        for result in results:
            assert result.dtype == torch.float64
            assert (result - volume).abs().max() < 1e-15

    @pytest.mark.parametrize("angle", [(0.0, 120.0, 35.0), (60.0, 150.0, 120.0)])
    def test_python_float_parameters_match_cpu_tensor_kernel(self, angle, device, dtype):
        # Python-number parameters build the kernel on the CPU in the input dtype, never below float32 (#5181).
        kernel_dtype = torch.promote_types(dtype, torch.float32)
        volume = torch.rand(1, 2, 6, 7, 8, device=device, dtype=dtype)
        kernel = get_motion_kernel3d(
            5, torch.tensor([angle], dtype=kernel_dtype), torch.tensor([0.3], dtype=kernel_dtype)
        )
        expected = filter3d(volume, kernel, "replicate")
        self.assert_close(motion_blur3d(volume, 5, angle, 0.3, "replicate"), expected, rtol=0, atol=0)
        self.assert_close(MotionBlur3D(5, angle, 0.3, "replicate")(volume), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("tensor_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("tensor_parameter", ["angle", "direction"])
    def test_tensor_parameter_with_a_python_number_is_built_like_the_tensor(
        self, tensor_parameter, tensor_dtype, device, dtype
    ):
        # A tensor angle or direction with a Python number, or a tuple, for the other builds that one on the tensor's
        # device and in its dtype, whatever the input's dtype, so the blur is the one of the two tensors.
        if device.type == "mps" and torch.float64 in (dtype, tensor_dtype):
            pytest.skip("MPS has no float64")
        if not supports_replicate_padding_3d(device, dtype):
            pytest.skip("replication_pad3d is unavailable for this device/dtype")
        if not supports_nearest_3d_grid_sample(device, tensor_dtype):
            pytest.skip(f"the kernel is rotated with grid_sample, which this device lacks for {tensor_dtype}")
        torch.manual_seed(0)
        volume = torch.rand(1, 2, 6, 7, 8, device=device, dtype=dtype)
        angle = torch.tensor([[10.0, 20.0, 30.0]], device=device, dtype=tensor_dtype)
        direction = torch.tensor([0.5], device=device, dtype=tensor_dtype)
        expected = motion_blur3d(volume, 3, angle, direction)
        params = (angle, 0.5) if tensor_parameter == "angle" else ((10.0, 20.0, 30.0), direction)
        actual = motion_blur3d(volume, 3, *params)
        assert actual.dtype == dtype
        assert actual.device == volume.device
        self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(MotionBlur3D(3, *params)(volume), expected, rtol=0, atol=0)

    @pytest.mark.skip(reason="After the op be optimized the results are not the same")
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        # TODO: Fix the operation to works after dynamo optimize
        data = torch.ones(batch_size, 3, 1, 10, 10, device=device, dtype=dtype)
        op = MotionBlur3D(3, (0.0, 360.0, 150.0), 0.5)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    # A blur of a constant volume is that constant, thin volumes included: ``D = 1`` is why the default is
    # "replicate" and not "reflect", which needs each axis longer than the kernel radius. "replicate" has no such
    # limit, down to a single voxel, so unlike the 2-D default there is no size at which this one raises.
    @pytest.mark.parametrize(
        "shape", [(1, 1, 5, 6, 7), (2, 2, 3, 4, 5), (1, 1, 1, 5, 7), (2, 3, 1, 8, 9), (1, 1, 1, 1, 1)]
    )
    @pytest.mark.parametrize("angle", [(0.0, 0.0, 0.0), (45.0, 0.0, 0.0), (0.0, 90.0, 0.0), (30.0, 60.0, 120.0)])
    @pytest.mark.parametrize("direction", [0.0, 0.7])
    def test_constant_volume_stays_constant_at_the_default_border(self, shape, angle, direction, device, dtype):
        if not supports_replicate_padding_3d(device, dtype):
            pytest.skip("replication_pad3d is unavailable for this device/dtype")
        volume = torch.full(shape, 0.75, device=device, dtype=dtype)

        self.assert_close(motion_blur3d(volume, 3, angle, direction), volume)
        self.assert_close(MotionBlur3D(3, angle, direction)(volume), volume)

    # Snippet used to generate expected:
    #   import torch, kornia.filters as KF
    #   print(KF.motion_blur3d(torch.ones(1, 1, 5, 6, 7), 3, (0.0, 0.0, 0.0), 0.0, "constant")[0, 0, 2, 2].tolist())
    #   -> [0.6667, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6667]   (the k = 3 kernel at angle 0 runs along the last axis)
    def test_explicit_constant_border_zero_pads(self, device, dtype):
        volume = torch.ones(1, 1, 5, 6, 7, device=device, dtype=dtype)
        row = torch.tensor([2.0 / 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0 / 3.0], device=device, dtype=dtype)
        expected = row.expand(1, 1, 5, 6, 7)

        self.assert_close(motion_blur3d(volume, 3, (0.0, 0.0, 0.0), 0.0, border_type="constant"), expected)
        self.assert_close(MotionBlur3D(3, (0.0, 0.0, 0.0), 0.0, border_type="constant")(volume), expected)

    def test_function_and_module_agree_at_their_defaults(self, device, dtype):
        if not supports_replicate_padding_3d(device, dtype):
            pytest.skip("replication_pad3d is unavailable for this device/dtype")
        torch.manual_seed(0)
        volume = torch.rand(2, 2, 4, 6, 7, device=device, dtype=dtype)
        params = (3, (30.0, 10.0, 20.0), 0.5)

        from_function = motion_blur3d(volume, *params)
        self.assert_close(MotionBlur3D(*params)(volume), from_function)
        # the shared default is "replicate"
        self.assert_close(from_function, motion_blur3d(volume, *params, border_type="replicate"))


class TestConventionsMotionBlur(BaseTester):
    """Pins for the angle and direction of the motion blurs, and for their filed defects."""

    @staticmethod
    def _heaviest_offset(response, centre):
        # Offset of the heaviest pixel (or voxel) of a delta's response from the delta, in (..., row, column) order.
        flat = int(response.detach().cpu().float().argmax())
        index = []
        for size in reversed(response.shape):
            index.append(flat % size)
            flat //= size
        return tuple(i - c for i, c in zip(reversed(index), centre))

    @staticmethod
    def _skip_without_default_border(device, dtype, volumetric):
        # The default border pads with reflect (2-D) or replicate (3-D), which some torch builds lack for half dtypes.
        if volumetric and not supports_replicate_padding_3d(device, dtype):
            pytest.skip(f"this torch build has no 3D replicate padding kernel for {dtype} on {device.type}")
        if not volumetric and not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")

    def test_convention_motion_blur_angle_is_counterclockwise_as_displayed(self, device, dtype):
        # angle is in degrees and turns the streak counter-clockwise as the image is displayed (row 0 on top), as
        # rotate() does: with direction=1 a point's streak is heaviest 2 px right of it at angle 0 and 2 px above
        # it (row - 2) at angle 90.
        self._skip_without_default_border(device, dtype, False)
        image = torch.zeros(1, 1, 9, 12, device=device, dtype=dtype)
        image[0, 0, 3, 7] = 1.0
        at_0 = motion_blur(image, 5, 0.0, 1.0)
        at_90 = motion_blur(image, 5, 90.0, 1.0)
        assert self._heaviest_offset(at_0[0, 0], (3, 7)) == (0, 2)
        assert at_0[0, 0].nonzero()[:, 0].unique().tolist() == [3]
        assert self._heaviest_offset(at_90[0, 0], (3, 7)) == (-2, 0)
        assert at_90[0, 0].nonzero()[:, 1].unique().tolist() == [7]

        # Relabelling check: a transpose mirrors the image and reverses the sense of rotation, so the transposed
        # response at angle 0 is the transposed image's response at angle -90, not at +90.
        transposed = image.transpose(-1, -2).contiguous()
        self.assert_close(motion_blur(transposed, 5, -90.0, 1.0), at_0.transpose(-1, -2))
        assert (motion_blur(transposed, 5, 90.0, 1.0) - at_0.transpose(-1, -2)).abs().max() > 0.3

    def test_convention_motion_blur_direction_weights_the_streak_linearly(self, device, dtype):
        # direction, clamped to [-1, 1], tilts the weights linearly along the streak: at angle 0 a point's streak
        # carries 0.1, 0.2, 0.3, 0.4 from 1 px left to 2 px right of it for +1 (heaviest ahead, at +x), the mirror
        # image for -1, and 0.2 on each of 5 pixels for 0.
        # Snippet used to generate expected:
        #   x = torch.zeros(1, 1, 9, 12); x[0, 0, 3, 7] = 1; print(motion_blur(x, 5, 0.0, d)[0, 0, 3, 5:10])
        self._skip_without_default_border(device, dtype, False)
        image = torch.zeros(1, 1, 9, 12, device=device, dtype=dtype)
        image[0, 0, 3, 7] = 1.0
        forward, backward, even = [0.0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0.0], [0.2] * 5
        for direction, expected in ((1.0, forward), (-1.0, backward), (0.0, even), (5.0, forward), (-3.0, backward)):
            out = motion_blur(image, 5, 0.0, direction)
            assert out[0, 0].nonzero()[:, 0].unique().tolist() == [3]
            self.assert_close(out[0, 0, 3, 5:10], torch.tensor(expected, device=device, dtype=dtype))

    def test_convention_motion_blur3d_positive_roll_turns_x_toward_y(self, device, dtype):
        # angle = (yaw, pitch, roll) is one axis-angle vector (rx, ry, rz) in degrees, and the streak starts along +x.
        # Each case sets one component, a turn about that axis. A positive roll turns +x toward +y: clockwise as
        # displayed, the opposite sense to motion_blur's angle. A positive pitch turns it toward -z, and yaw, a turn
        # about the streak's own axis, leaves it along +x.
        # Snippet used to generate expected (k = 3, direction = 1; the heaviest voxel's (depth, row, column) offset):
        #   v = torch.zeros(1, 1, 7, 9, 11); v[0, 0, 3, 5, 4] = 1; out = motion_blur3d(v, 3, angle, 1.0)
        self._skip_without_default_border(device, dtype, True)
        volume = torch.zeros(1, 1, 7, 9, 11, device=device, dtype=dtype)
        volume[0, 0, 3, 5, 4] = 1.0
        for angle, offset in (
            ((0.0, 0.0, 0.0), (0, 0, 1)),
            ((0.0, 0.0, 90.0), (0, 1, 0)),
            ((0.0, 0.0, -90.0), (0, -1, 0)),
            ((0.0, 90.0, 0.0), (-1, 0, 0)),
            ((90.0, 0.0, 0.0), (0, 0, 1)),
        ):
            assert self._heaviest_offset(motion_blur3d(volume, 3, angle, 1.0)[0, 0], (3, 5, 4)) == offset

    def test_convention_motion_blur3d_module_float_angle_is_used_for_all_three_axes(self, device, dtype):
        # MotionBlur3D takes a float angle as the vector (angle, angle, angle): yaw, pitch and roll alike. At 50
        # degrees every vector that leaves one or two of its components at 0 is a different kernel.
        self._skip_without_default_border(device, dtype, True)
        torch.manual_seed(0)
        volume = torch.rand(1, 1, 7, 8, 9).to(device=device, dtype=dtype)
        out = MotionBlur3D(3, 50.0, 0.5)(volume)
        self.assert_close(out, motion_blur3d(volume, 3, (50.0, 50.0, 50.0), 0.5))
        for yaw, pitch, roll in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)):
            partial = motion_blur3d(volume, 3, (50.0 * yaw, 50.0 * pitch, 50.0 * roll), 0.5)
            assert (out - partial).abs().max() > 0.3

    def test_convention_motion_blur_tensor_angle_and_direction_are_per_sample(self, device, dtype):
        # A tensor angle and direction are (B,): entry b drives sample b. A scalar direction is not broadcast
        # against a (B,) angle.
        self._skip_without_default_border(device, dtype, False)
        if not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip(f"this torch build has no 2D grid_sample kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        image = torch.rand(2, 1, 9, 12).to(device=device, dtype=dtype)
        angle = torch.tensor([0.0, 90.0], device=device, dtype=dtype)
        direction = torch.tensor([0.5, -0.5], device=device, dtype=dtype)
        out = motion_blur(image, 5, angle, direction)
        self.assert_close(out[:1], motion_blur(image[:1], 5, 0.0, 0.5))
        self.assert_close(out[1:], motion_blur(image[1:], 5, 90.0, -0.5))
        with pytest.raises(BaseError, match="same length"):
            motion_blur(image, 5, angle, 0.5)

    @pytest.mark.parametrize("volumetric", [False, True], ids=["MotionBlur", "MotionBlur3D"])
    def test_convention_motion_blur_modules_honour_mode_5164(self, volumetric, device, dtype):
        """MotionBlur and MotionBlur3D rotate the kernel with their mode argument, as the functions do (#5164)."""
        self._skip_without_default_border(device, dtype, volumetric)
        torch.manual_seed(0)
        if volumetric:
            image = torch.rand(1, 1, 5, 6, 7).to(device=device, dtype=dtype)
            out = MotionBlur3D(3, (10.0, 20.0, 30.0), 0.5, mode="bilinear")(image)
            nearest = motion_blur3d(image, 3, (10.0, 20.0, 30.0), 0.5, mode="nearest")
            bilinear = motion_blur3d(image, 3, (10.0, 20.0, 30.0), 0.5, mode="bilinear")
        else:
            image = torch.rand(1, 1, 9, 12).to(device=device, dtype=dtype)
            out = MotionBlur(5, 30.0, 0.3, mode="bilinear")(image)
            nearest = motion_blur(image, 5, 30.0, 0.3, mode="nearest")
            bilinear = motion_blur(image, 5, 30.0, 0.3, mode="bilinear")
        self.assert_close(out, bilinear)
        assert (out - nearest).abs().max() > 0.03

    def test_convention_motion_blur3d_module_takes_a_tensor_or_int_angle_5164(self, device, dtype):
        """MotionBlur3D takes a (B, 3) tensor angle as motion_blur3d does, and an int as a float (#5164)."""
        self._skip_without_default_border(device, dtype, True)
        torch.manual_seed(0)
        volume = torch.rand(1, 1, 5, 6, 7).to(device=device, dtype=dtype)
        # the tensor angle and direction share a device and a dtype, as get_motion_kernel3d requires
        angle, direction = torch.tensor([[10.0, 20.0, 30.0]]), torch.tensor([0.5])
        self.assert_close(MotionBlur3D(3, angle, direction)(volume), motion_blur3d(volume, 3, angle, direction))
        self.assert_close(MotionBlur3D(3, 35, 0.5)(volume), motion_blur3d(volume, 3, (35.0, 35.0, 35.0), 0.5))

    def test_convention_motion_blur_default_border_keeps_a_constant_image_5168(self, device, dtype):
        """The 2-D motion blurs default to border_type='reflect' and the 3-D ones to 'replicate' (#5168)."""
        # Both defaults keep a constant image constant at its edges, where 'constant' darkens it.
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        if not supports_replicate_padding_3d(device, dtype):
            pytest.skip(f"this torch build has no 3D replicate padding kernel for {dtype} on {device.type}")
        image = torch.ones(1, 1, 5, 7, device=device, dtype=dtype)
        self.assert_close(motion_blur(image, 5, 0.0, 0.0), image)
        self.assert_close(MotionBlur(5, 0.0, 0.0)(image), image)
        assert motion_blur(image, 5, 0.0, 0.0, "constant")[0, 0, 2, 0] < 0.9
        volume = torch.ones(1, 1, 5, 6, 7, device=device, dtype=dtype)
        self.assert_close(motion_blur3d(volume, 3, (0.0, 0.0, 0.0), 0.0), volume)
        self.assert_close(MotionBlur3D(3, (0.0, 0.0, 0.0), 0.0)(volume), volume)
        assert motion_blur3d(volume, 3, (0.0, 0.0, 0.0), 0.0, "constant")[0, 0, 2, 3, 0] < 0.9
        # the defaults are those two modes, not just modes that keep a constant
        torch.manual_seed(0)
        image, volume = torch.rand_like(image), torch.rand_like(volume)
        self.assert_close(motion_blur(image, 5, 30.0, 0.5), motion_blur(image, 5, 30.0, 0.5, "reflect"))
        self.assert_close(
            motion_blur3d(volume, 3, (10.0, 20.0, 30.0), 0.5),
            motion_blur3d(volume, 3, (10.0, 20.0, 30.0), 0.5, "replicate"),
        )

    def test_convention_motion_blur_kernel_size_must_be_an_int_5169(self, device, dtype):
        """motion_blur and motion_blur3d reject a tuple kernel_size with a kornia error that names it (#5169)."""
        with pytest.raises(BaseError, match="kernel_size"):
            motion_blur(torch.rand(1, 1, 9, 12, device=device, dtype=dtype), (5, 5), 30.0, 0.5)
        with pytest.raises(BaseError, match="kernel_size"):
            motion_blur3d(torch.rand(1, 1, 5, 9, 12, device=device, dtype=dtype), (3, 3, 3), (30.0, 0.0, 0.0), 0.5)
