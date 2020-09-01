import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
import kornia.geometry.transform.projwarp as proj


@pytest.fixture
def dtype():
    return torch.float32


class TestWarpProjective:
    def test_smoke(self, device, dtype):
        input = torch.rand(1, 3, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(1, 3, 4, device=device, dtype=dtype)
        output = proj.warp_projective(input, P, (3, 4, 5))
        assert output.shape == (1, 3, 3, 4, 5)

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("num_channels", [1, 3, 5])
    @pytest.mark.parametrize("out_shape", [(3, 3, 3), (4, 5, 6)])
    def test_batch(self, batch_size, num_channels, out_shape, device, dtype):
        B, C = batch_size, num_channels
        input = torch.rand(B, C, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(B, 3, 4, device=device, dtype=dtype)
        output = proj.warp_projective(input, P, out_shape)
        assert list(output.shape) == [B, C] + list(out_shape)

    def test_gradcheck(self, device):
        # generate input data
        input = torch.rand(1, 3, 3, 4, 5, device=device, dtype=torch.float64, requires_grad=True)
        P = torch.rand(1, 3, 4, device=device, dtype=torch.float64)
        assert gradcheck(proj.warp_projective, (input, P, (3, 3, 3)), raise_exception=True)

    def test_forth_back(self, device, dtype):
        out_shape = (3, 4, 5)
        input = torch.rand(2, 5, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(2, 3, 4, device=device, dtype=dtype)
        P = kornia.geometry.convert_affinematrix_to_homography3d(P)
        P_hat = (P.inverse() @ P)[:, :3]
        output = proj.warp_projective(input, P_hat, out_shape)
        assert_allclose(output, input, rtol=1e-4, atol=1e-4)

    def test_rotate_x(self, device, dtype):
        input = torch.tensor([[[[
            [0., 0., 0.],
            [0., 2., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 2., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        _, _, D, H, W = input.shape
        center = torch.tensor([
            [(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[90., 0., 0.]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_projective(input, P, (3, 3, 3))
        assert_allclose(output, expected)

    def test_rotate_y(self, device, dtype):
        input = torch.tensor([[[[
            [0., 0., 0.],
            [0., 2., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [2., 1., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        _, _, D, H, W = input.shape
        center = torch.tensor([
            [(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[0., 90., 0.]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_projective(input, P, (3, 3, 3))
        assert_allclose(output, expected)

    def test_rotate_z(self, device, dtype):
        input = torch.tensor([[[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ], [
            [0., 2., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 1., 2.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        _, _, D, H, W = input.shape
        center = torch.tensor([
            [(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[0., 0., 90.]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_projective(input, P, (3, 3, 3))
        assert_allclose(output, expected)

    def test_rotate_y_large(self, device, dtype):
        """Rotates 90deg anti-clockwise."""
        input = torch.tensor([[[[
            [0., 4., 0.],
            [0., 3., 0.],
            [0., 0., 0.],
        ], [
            [0., 2., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]], [[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 9., 0.],
        ], [
            [0., 0., 0.],
            [0., 6., 7.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 8., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ], [
            [4., 2., 0.],
            [3., 1., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]], [[
            [0., 0., 0.],
            [0., 7., 0.],
            [0., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 6., 8.],
            [9., 0., 0.],
        ], [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]]]], device=device, dtype=dtype)

        _, _, D, H, W = input.shape
        center = torch.tensor([
            [(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[0., 90., 0.]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_projective(input, P, (3, 3, 3))
        assert_allclose(output, expected)


class TestGetRotationMatrix3d:
    def test_smoke(self, device, dtype):
        center = torch.rand(1, 3, device=device, dtype=dtype)
        angle = torch.rand(1, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle)
        P = proj.get_projective_transform(center, angle, scales)
        assert P.shape == (1, 3, 4)

    @pytest.mark.parametrize("batch_size", [1, 3, 6])
    def test_batch(self, batch_size, device, dtype):
        B: int = batch_size
        center = torch.rand(B, 3, device=device, dtype=dtype)
        angle = torch.rand(B, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle)
        P = proj.get_projective_transform(center, angle, scales)
        assert P.shape == (B, 3, 4)

    def test_identity(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.zeros(1, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
        ], device=device, dtype=dtype)
        assert_allclose(P, P_expected)

    def test_rot90x(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[90., 0., 0.]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([
            [1., 0., 0., 0.],
            [0., 0., -1., 0.],
            [0., 1., 0., 0.],
        ], device=device, dtype=dtype)
        assert_allclose(P, P_expected)

    def test_rot90y(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[0., 90., 0.]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 0.],
        ], device=device, dtype=dtype)
        assert_allclose(P, P_expected)

    def test_rot90z(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[0., 0., 90.]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([
            [0., -1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
        ], device=device, dtype=dtype)
        assert_allclose(P, P_expected)

    def test_gradcheck(self, device):
        # generate input data
        center = torch.rand(1, 3, device=device, dtype=torch.float64, requires_grad=True)
        angle = torch.rand(1, 3, device=device, dtype=torch.float64)
        scales: torch.Tensor = torch.ones_like(angle)
        assert gradcheck(proj.get_projective_transform, (center, angle, scales), raise_exception=True)
