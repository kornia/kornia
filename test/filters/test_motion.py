import pytest
import torch

from kornia.filters import (
    MotionBlur,
    MotionBlur3D,
    get_motion_kernel2d,
    get_motion_kernel3d,
    motion_blur,
    motion_blur3d,
)
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestMotionBlur(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("angle", [36.0, 200.0])
    @pytest.mark.parametrize("direction", [-0.9, 0.0, 0.9])
    @pytest.mark.parametrize("mode", ['bilinear', 'nearest'])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_smoke(self, shape, kernel_size, angle, direction, mode, params_as_tensor, device, dtype):
        B, C, H, W = shape
        inpt = torch.rand(shape, device=device, dtype=dtype)

        if params_as_tensor is True:
            angle = torch.tensor([angle], device=device, dtype=dtype).repeat(B)
            direction = torch.tensor([direction], device=device, dtype=dtype).repeat(B)
        actual = motion_blur(inpt, kernel_size, angle, direction, 'constant', mode)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_cardinality(self, shape, device, dtype):
        ksize = 5
        angle = 200.0
        direction = 0.3

        sample = torch.rand(shape, device=device, dtype=dtype)
        motion = MotionBlur(ksize, angle, direction)
        assert motion(sample).shape == shape

    @pytest.mark.skip(reason='nothing to test')
    def test_exception(self):
        ...

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

    def test_noncontiguous(self, device, dtype):
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

        sample = torch.rand(batch_shape, device=device)
        sample = tensor_to_gradcheck_var(sample)
        self.gradcheck(motion_blur, (sample, ksize, angle, direction, "replicate"), nondet_tol=1e-8)

    def test_module(self, device, dtype):
        params = [3, 20.0, 0.5]
        op = motion_blur
        op_module = MotionBlur(*params)
        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)

        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.skip(reason='After the op be optimized the results are not the same')
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        # TODO: FIX op
        inpt = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = MotionBlur(3, 36.0, 0.5)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))


class TestMotionBlur3D(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 3, 8, 15), (2, 2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("angle", [(36.0, 15.0, 200.0), (200.0, 10.0, 150.0)])
    @pytest.mark.parametrize("direction", [-0.9, 0.0, 0.9])
    @pytest.mark.parametrize("mode", ['bilinear', 'nearest'])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_smoke(self, shape, kernel_size, angle, direction, mode, params_as_tensor, device, dtype):
        B, C, D, H, W = shape
        inpt = torch.rand(shape, device=device, dtype=dtype)

        if params_as_tensor is True:
            angle = torch.tensor([angle], device=device, dtype=dtype).expand(B, 3)
            direction = torch.tensor([direction], device=device, dtype=dtype).repeat(B)
        actual = motion_blur3d(inpt, kernel_size, angle, direction, 'constant', mode)

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 1, 8, 15), (2, 3, 1, 11, 7)])
    def test_cardinality(self, shape, device, dtype):
        ksize = 5
        angle = (200.0, 15.0, 120.0)
        direction = 0.3

        sample = torch.rand(shape, device=device, dtype=dtype)
        motion = MotionBlur3D(ksize, angle, direction)
        assert motion(sample).shape == shape

    @pytest.mark.skip(reason='nothing to test')
    def test_exception(self):
        ...

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("ksize", [3, 11])
    @pytest.mark.parametrize("angle", [(0.0, 360.0, 150.0)])
    @pytest.mark.parametrize("direction", [-1.0, 1.0])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_get_motion_kernel3d(self, batch_size, ksize, angle, direction, params_as_tensor, device, dtype):
        if params_as_tensor is True:
            angle = torch.tensor([angle], device=device, dtype=dtype).repeat(batch_size, 1)
            direction = torch.tensor([direction], device=device, dtype=dtype).repeat(batch_size)
        else:
            batch_size = 1
            device = None
            dtype = None

        actual = get_motion_kernel3d(ksize, angle, direction)
        expected = torch.ones(1, device=device, dtype=dtype) * batch_size
        assert actual.shape == (batch_size, ksize, ksize, ksize)
        self.assert_close(actual.sum(), expected.sum())

    def test_noncontiguous(self, device, dtype):
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

        sample = torch.rand(batch_shape, device=device)
        sample = tensor_to_gradcheck_var(sample)
        self.gradcheck(motion_blur3d, (sample, ksize, angle, direction, "replicate"), nondet_tol=1e-8)

    def test_module(self, device, dtype):
        params = [3, (0.0, 360.0, 150.0), 0.5]
        op = motion_blur3d
        op_module = MotionBlur3D(*params)
        img = torch.ones(1, 3, 1, 5, 5, device=device, dtype=dtype)

        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.skip(reason='After the op be optimized the results are not the same')
    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        # TODO: Fix the operation to works after dynamo optimize
        inpt = torch.ones(batch_size, 3, 1, 10, 10, device=device, dtype=dtype)
        op = MotionBlur3D(3, (0.0, 360.0, 150.0), 0.5)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))
