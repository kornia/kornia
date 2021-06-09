import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia


def _sample_points(batch_size, device, dtype=torch.float32):
    src = torch.tensor([[[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0], [5.0, 5.0]]], device=device, dtype=dtype)
    src = src.repeat(batch_size, 1, 1)
    dst = src + torch.rand_like(src) * 2.5
    return src, dst


class TestTransformParameters:
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 4, 2, device=device)
        out = kornia.get_tps_transform(src, src)
        assert len(out) == 2

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_no_warp(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 5, 2, device=device)
        kernel, affine = kornia.get_tps_transform(src, src)
        target_kernel = torch.zeros(batch_size, 5, 2, device=device)
        target_affine = torch.zeros(batch_size, 3, 2, device=device)
        target_affine[:, [1, 2], [0, 1]] = 1.0
        assert_allclose(kernel, target_kernel, atol=1e-4, rtol=1e-4)
        assert_allclose(affine, target_affine, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_affine_only(self, batch_size, device, dtype):
        src = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]], device=device).repeat(
            batch_size, 1, 1
        )
        dst = src.clone() * 2.0
        kernel, affine = kornia.get_tps_transform(src, dst)
        assert_allclose(kernel, torch.zeros_like(kernel), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_exception(self, batch_size, device, dtype):
        with pytest.raises(TypeError):
            src = torch.rand(batch_size, 5, 2).numpy()
            assert kornia.get_tps_transform(src, src)

        with pytest.raises(ValueError):
            src = torch.rand(batch_size, 5)
            assert kornia.get_tps_transform(src, src)

    @pytest.mark.grad
    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_gradcheck(self, batch_size, device, dtype, requires_grad):
        opts = dict(device=device, dtype=torch.float64)
        src, dst = _sample_points(batch_size, **opts)
        src.requires_grad_(requires_grad)
        dst.requires_grad_(not requires_grad)
        assert gradcheck(kornia.get_tps_transform, (src, dst), raise_exception=True)

    @pytest.mark.jit
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        op = kornia.get_tps_transform
        op_jit = torch.jit.script(op)
        op_output = op(src, dst)
        jit_output = op_jit(src, dst)
        assert_allclose(op_output[0], jit_output[0])
        assert_allclose(op_output[1], jit_output[1])


class TestWarpPoints:
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.get_tps_transform(src, dst)
        warp = kornia.warp_points_tps(src, dst, kernel, affine)
        assert warp.shape == src.shape

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_warp(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.get_tps_transform(src, dst)
        warp = kornia.warp_points_tps(src, dst, kernel, affine)
        assert_allclose(warp, dst, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_exception(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 5, 2)
        kernel = torch.zeros_like(src)
        affine = torch.zeros(batch_size, 3, 2)

        with pytest.raises(TypeError):
            assert kornia.warp_points_tps(src.numpy(), src, kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.warp_points_tps(src, src.numpy(), kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.warp_points_tps(src, src, kernel.numpy(), affine)

        with pytest.raises(TypeError):
            assert kornia.warp_points_tps(src, src, kernel, affine.numpy())

        with pytest.raises(ValueError):
            src_bad = torch.rand(batch_size, 5)
            assert kornia.warp_points_tps(src_bad, src, kernel, affine)

        with pytest.raises(ValueError):
            src_bad = torch.rand(batch_size, 5)
            assert kornia.warp_points_tps(src, src_bad, kernel, affine)

        with pytest.raises(ValueError):
            kernel_bad = torch.rand(batch_size, 5)
            assert kornia.warp_points_tps(src, src, kernel_bad, affine)

        with pytest.raises(ValueError):
            affine_bad = torch.rand(batch_size, 3)
            assert kornia.warp_points_tps(src, src, kernel, affine_bad)

    @pytest.mark.grad
    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_gradcheck(self, batch_size, device, dtype, requires_grad):
        opts = dict(device=device, dtype=torch.float64)
        src, dst = _sample_points(batch_size, **opts)
        kernel, affine = kornia.get_tps_transform(src, dst)
        kernel.requires_grad_(requires_grad)
        affine.requires_grad_(not requires_grad)
        assert gradcheck(kornia.warp_points_tps, (src, dst, kernel, affine), raise_exception=True)

    @pytest.mark.jit
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.get_tps_transform(src, dst)
        op = kornia.warp_points_tps
        op_jit = torch.jit.script(op)
        assert_allclose(op(src, dst, kernel, affine), op_jit(src, dst, kernel, affine))


class TestWarpImage:
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        tensor = torch.rand(batch_size, 3, 32, 32, device=device)
        kernel, affine = kornia.get_tps_transform(src, dst)
        warp = kornia.warp_image_tps(tensor, dst, kernel, affine)
        assert warp.shape == tensor.shape

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_warp(self, batch_size, device, dtype):
        src = torch.tensor([[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0]]], device=device).repeat(
            batch_size, 1, 1
        )
        # zoom in by a factor of 2
        dst = src.clone() * 2.0
        tensor = torch.zeros(batch_size, 3, 8, 8, device=device)
        tensor[:, :, 2:6, 2:6] = 1.0

        expected = torch.ones_like(tensor)
        # nn.grid_sample interpolates the at the edges it seems, so the boundaries have values < 1
        expected[:, :, [0, -1], :] *= 0.5
        expected[:, :, :, [0, -1]] *= 0.5

        kernel, affine = kornia.get_tps_transform(dst, src)
        warp = kornia.warp_image_tps(tensor, src, kernel, affine)
        assert_allclose(warp, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_exception(self, batch_size, device, dtype):
        image = torch.rand(batch_size, 3, 32, 32)
        dst = torch.rand(batch_size, 5, 2)
        kernel = torch.zeros_like(dst)
        affine = torch.zeros(batch_size, 3, 2)

        with pytest.raises(TypeError):
            assert kornia.warp_image_tps(image.numpy(), dst, kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.warp_image_tps(image, dst.numpy(), kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.warp_image_tps(image, dst, kernel.numpy(), affine)

        with pytest.raises(TypeError):
            assert kornia.warp_image_tps(image, dst, kernel, affine.numpy())

        with pytest.raises(ValueError):
            image_bad = torch.rand(batch_size, 32, 32)
            assert kornia.warp_image_tps(image_bad, dst, kernel, affine)

        with pytest.raises(ValueError):
            dst_bad = torch.rand(batch_size, 5)
            assert kornia.warp_image_tps(image, dst_bad, kernel, affine)

        with pytest.raises(ValueError):
            kernel_bad = torch.rand(batch_size, 5)
            assert kornia.warp_image_tps(image, dst, kernel_bad, affine)

        with pytest.raises(ValueError):
            affine_bad = torch.rand(batch_size, 3)
            assert kornia.warp_image_tps(image, dst, kernel, affine_bad)

    @pytest.mark.grad
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_gradcheck(self, batch_size, device, dtype):
        opts = dict(device=device, dtype=torch.float64)
        src, dst = _sample_points(batch_size, **opts)
        kernel, affine = kornia.get_tps_transform(src, dst)
        image = torch.rand(batch_size, 3, 8, 8, requires_grad=True, **opts)
        assert gradcheck(
            kornia.warp_image_tps, (image, dst, kernel, affine), raise_exception=True, atol=1e-4, rtol=1e-4
        )

    @pytest.mark.jit
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.get_tps_transform(src, dst)
        image = torch.rand(batch_size, 3, 32, 32, device=device)
        op = kornia.warp_image_tps
        op_jit = torch.jit.script(op)
        assert_allclose(op(image, dst, kernel, affine), op_jit(image, dst, kernel, affine), rtol=1e-4, atol=1e-4)
