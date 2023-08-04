import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import assert_close
from kornia.utils._compat import torch_version_ge


def _sample_points(batch_size, device, dtype=torch.float32):
    src = torch.tensor([[[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0], [5.0, 5.0]]], device=device, dtype=dtype)
    src = src.repeat(batch_size, 1, 1)
    dst = src + torch.rand_like(src) * 2.5
    return src, dst


class TestTransformParameters:
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 4, 2, device=device)
        out = kornia.geometry.transform.get_tps_transform(src, src)
        assert len(out) == 2

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_no_warp(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 5, 2, device=device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, src)
        target_kernel = torch.zeros(batch_size, 5, 2, device=device)
        target_affine = torch.zeros(batch_size, 3, 2, device=device)
        target_affine[:, [1, 2], [0, 1]] = 1.0
        assert_close(kernel, target_kernel, atol=1e-4, rtol=1e-4)
        assert_close(affine, target_affine, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_affine_only(self, batch_size, device, dtype):
        src = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]], device=device).repeat(
            batch_size, 1, 1
        )
        dst = src.clone() * 2.0
        kernel, _ = kornia.geometry.transform.get_tps_transform(src, dst)
        assert_close(kernel, torch.zeros_like(kernel), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_exception(self, batch_size, device, dtype):
        with pytest.raises(TypeError):
            src = torch.rand(batch_size, 5, 2).numpy()
            assert kornia.geometry.transform.get_tps_transform(src, src)

        with pytest.raises(ValueError):
            src = torch.rand(batch_size, 5)
            assert kornia.geometry.transform.get_tps_transform(src, src)

    @pytest.mark.grad()
    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_gradcheck(self, batch_size, device, dtype, requires_grad):
        opts = {'device': device, 'dtype': torch.float64}
        src, dst = _sample_points(batch_size, **opts)
        src.requires_grad_(requires_grad)
        dst.requires_grad_(not requires_grad)
        assert gradcheck(kornia.geometry.transform.get_tps_transform, (src, dst), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        op = kornia.geometry.transform.get_tps_transform
        op_jit = torch.jit.script(op)
        op_output = op(src, dst)
        jit_output = op_jit(src, dst)
        assert_close(op_output[0], jit_output[0])
        assert_close(op_output[1], jit_output[1])


class TestWarpPoints:
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_points_tps(src, dst, kernel, affine)
        assert warp.shape == src.shape

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_warp(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_points_tps(src, dst, kernel, affine)
        assert_close(warp, dst, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_exception(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 5, 2)
        kernel = torch.zeros_like(src)
        affine = torch.zeros(batch_size, 3, 2)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_points_tps(src.numpy(), src, kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_points_tps(src, src.numpy(), kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_points_tps(src, src, kernel.numpy(), affine)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_points_tps(src, src, kernel, affine.numpy())

        with pytest.raises(ValueError):
            src_bad = torch.rand(batch_size, 5)
            assert kornia.geometry.transform.warp_points_tps(src_bad, src, kernel, affine)

        with pytest.raises(ValueError):
            src_bad = torch.rand(batch_size, 5)
            assert kornia.geometry.transform.warp_points_tps(src, src_bad, kernel, affine)

        with pytest.raises(ValueError):
            kernel_bad = torch.rand(batch_size, 5)
            assert kornia.geometry.transform.warp_points_tps(src, src, kernel_bad, affine)

        with pytest.raises(ValueError):
            affine_bad = torch.rand(batch_size, 3)
            assert kornia.geometry.transform.warp_points_tps(src, src, kernel, affine_bad)

    @pytest.mark.grad()
    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_gradcheck(self, batch_size, device, dtype, requires_grad):
        opts = {'device': device, 'dtype': torch.float64}
        src, dst = _sample_points(batch_size, **opts)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        kernel.requires_grad_(requires_grad)
        affine.requires_grad_(not requires_grad)
        assert gradcheck(
            kornia.geometry.transform.warp_points_tps, (src, dst, kernel, affine), raise_exception=True, fast_mode=True
        )

    @pytest.mark.jit()
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        op = kornia.geometry.transform.warp_points_tps
        op_jit = torch.jit.script(op)
        assert_close(op(src, dst, kernel, affine), op_jit(src, dst, kernel, affine))


class TestWarpImage:
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        tensor = torch.rand(batch_size, 3, 32, 32, device=device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_image_tps(tensor, dst, kernel, affine)
        assert warp.shape == tensor.shape

    @pytest.mark.skipif(
        torch_version_ge(1, 10), reason="for some reason the solver detects singular matrices in pytorch >=1.10."
    )
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

        kernel, affine = kornia.geometry.transform.get_tps_transform(dst, src)
        warp = kornia.geometry.transform.warp_image_tps(tensor, src, kernel, affine)
        assert_close(warp, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_exception(self, batch_size, device, dtype):
        image = torch.rand(batch_size, 3, 32, 32)
        dst = torch.rand(batch_size, 5, 2)
        kernel = torch.zeros_like(dst)
        affine = torch.zeros(batch_size, 3, 2)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_image_tps(image.numpy(), dst, kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_image_tps(image, dst.numpy(), kernel, affine)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_image_tps(image, dst, kernel.numpy(), affine)

        with pytest.raises(TypeError):
            assert kornia.geometry.transform.warp_image_tps(image, dst, kernel, affine.numpy())

        with pytest.raises(ValueError):
            image_bad = torch.rand(batch_size, 32, 32)
            assert kornia.geometry.transform.warp_image_tps(image_bad, dst, kernel, affine)

        with pytest.raises(ValueError):
            dst_bad = torch.rand(batch_size, 5)
            assert kornia.geometry.transform.warp_image_tps(image, dst_bad, kernel, affine)

        with pytest.raises(ValueError):
            kernel_bad = torch.rand(batch_size, 5)
            assert kornia.geometry.transform.warp_image_tps(image, dst, kernel_bad, affine)

        with pytest.raises(ValueError):
            affine_bad = torch.rand(batch_size, 3)
            assert kornia.geometry.transform.warp_image_tps(image, dst, kernel, affine_bad)

    @pytest.mark.grad()
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_gradcheck(self, batch_size, device, dtype):
        opts = {'device': device, 'dtype': torch.float64}
        src, dst = _sample_points(batch_size, **opts)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        image = torch.rand(batch_size, 3, 8, 8, requires_grad=True, **opts)
        assert gradcheck(
            kornia.geometry.transform.warp_image_tps,
            (image, dst, kernel, affine),
            raise_exception=True,
            atol=1e-4,
            rtol=1e-4,
            nondet_tol=1e-8,
            fast_mode=True,
        )

    @pytest.mark.jit()
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        image = torch.rand(batch_size, 3, 32, 32, device=device)
        op = kornia.geometry.transform.warp_image_tps
        op_jit = torch.jit.script(op)
        assert_close(op(image, dst, kernel, affine), op_jit(image, dst, kernel, affine), rtol=1e-4, atol=1e-4)
