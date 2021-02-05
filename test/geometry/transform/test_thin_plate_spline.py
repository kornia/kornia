import pytest
import kornia as kornia
import kornia.testing as utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestTpsParameters:
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("device", ('cpu', 'cuda'))
    def test_smoke(self, batch_size, device):
        src = torch.rand(batch_size, 4, 2, device=device)
        out = kornia.get_tps_parameters(src, src)
        assert len(out) == 2

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("device", ('cpu', 'cuda'))
    def test_affine_only(self, batch_size, device):
        src = torch.tensor([[
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
            [.5, .5],
        ]], device=device).repeat(batch_size, 1, 1)
        dst = src.clone() * 2.
        kernel, affine = kornia.get_tps_parameters(src, dst)
        assert_allclose(kernel, torch.zeros_like(kernel), atol=1e-4, rtol=1e-4)


class TestWarpPoints:
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("device", ('cpu', 'cuda'))
    def test_smoke(self, batch_size, device):
        src = torch.rand(batch_size, 5, 2, device=device)
        dst = torch.rand(batch_size, 5, 2, device=device)
        kernel, affine = kornia.get_tps_parameters(src, dst)
        warp = kornia.warp_points_tps(src, dst, kernel, affine)
        assert warp.shape == src.shape

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("device", ('cpu', 'cuda'))
    def test_warp(self, batch_size, device):
        src = torch.rand(batch_size, 5, 2, device=device)
        dst = torch.rand(batch_size, 5, 2, device=device)
        kernel, affine = kornia.get_tps_parameters(src, dst)
        warp = kornia.warp_points_tps(src, dst, kernel, affine)
        assert_allclose(warp, dst, atol=1e-4, rtol=1e-4)


class TestWarpImage:
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("device", ('cpu', 'cuda'))
    def test_smoke(self, batch_size, device):
        src = torch.rand(batch_size, 5, 2, device=device)
        dst = torch.rand(batch_size, 5, 2, device=device)
        tensor = torch.rand(batch_size, 3, 32, 32, device=device)
        kernel, affine = kornia.get_tps_parameters(src, dst)
        warp = kornia.warp_img_tensor_tps(tensor, dst, kernel, affine)
        assert warp.shape == tensor.shape

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("device", ('cpu', 'cuda'))
    def test_warp(self, batch_size, device):
        src = torch.tensor([[
            [-1., -1.],
            [-1., 1.],
            [1., -1.],
            [1., -1.],
            [0., 0.],
        ]], device=device).repeat(batch_size, 1, 1)
        dst = src.clone() * 2.
        tensor = torch.zeros(batch_size, 3, 8, 8, device=device)
        tensor[:, :, 2:6, 2:6] = 1.
        expected = torch.ones_like(tensor)
        kernel, affine = kornia.get_tps_parameters(dst, src)
        warp = kornia.warp_img_tensor_tps(tensor, src, kernel, affine)
        assert_allclose(warp, expected, atol=1e-4, rtol=1e-4)
