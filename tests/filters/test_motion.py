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
    supports_bilinear_3d_grid_sample,
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
