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

import pytest
import torch

import kornia
import kornia.geometry.transform as proj
from kornia.core.utils import _torch_inverse_cast

from testing.base import BaseTester


@pytest.mark.parametrize("op_name", ["warp_affine3d", "warp_perspective3d"])
@pytest.mark.parametrize("dsize", [(0, 4, 5), (3, 0, 5), (3, 4, 0)])
@pytest.mark.parametrize("align_corners", [True, False])
def test_empty_destination_is_autograd_connected(op_name, dsize, align_corners, device, dtype):
    src = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype, requires_grad=True)
    if op_name == "warp_affine3d":
        transform = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).requires_grad_()
        out = proj.warp_affine3d(src, transform, dsize, align_corners=align_corners)
    else:
        transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).requires_grad_()
        out = proj.warp_perspective3d(src, transform, dsize, align_corners=align_corners)

    assert out.shape == (1, 2, *dsize)
    assert out.numel() == 0
    out.sum().backward()
    assert src.grad is not None and torch.count_nonzero(src.grad) == 0
    assert transform.grad is not None and torch.count_nonzero(transform.grad) == 0


@pytest.mark.parametrize("op_name", ["warp_affine3d", "warp_perspective3d"])
def test_empty_source_policy(op_name, device, dtype):
    src = torch.empty(1, 2, 0, 4, 5, device=device, dtype=dtype, requires_grad=True)
    if op_name == "warp_affine3d":
        transform = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).requires_grad_()
        op = proj.warp_affine3d
    else:
        transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).requires_grad_()
        op = proj.warp_perspective3d

    empty = op(src, transform, (0, 4, 5))
    assert empty.shape == (1, 2, 0, 4, 5)
    empty.sum().backward()
    assert src.grad is not None and transform.grad is not None

    with pytest.raises(ValueError, match="must be positive"):
        op(src, transform, (3, 4, 5))


@pytest.mark.parametrize("op_name", ["warp_affine3d", "warp_perspective3d"])
def test_negative_destination_raises(op_name, device, dtype):
    src = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
    if op_name == "warp_affine3d":
        transform = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0)
        op = proj.warp_affine3d
    else:
        transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        op = proj.warp_perspective3d
    with pytest.raises(ValueError, match="must be non-negative"):
        op(src, transform, (-1, 4, 5))


@pytest.mark.parametrize("op_name", ["warp_affine3d", "warp_perspective3d"])
def test_empty_destination_blames_an_integral_src_by_name(op_name, device):
    """``grid_sample`` rejects an integral volume on both paths; the message must name ``src``.

    Scoped away from MPS, whose ``grid_sample`` accepts an integral image and samples it back into
    ``int64`` instead of rejecting it, so the non-empty half of the pairing does not hold there.
    """
    if device.type == "mps":
        pytest.skip("MPS grid_sample accepts an integral image instead of rejecting it")
    op = getattr(proj, op_name)
    matrix_size = (3, 4) if op_name == "warp_affine3d" else (4, 4)
    src = torch.zeros(1, 2, 3, 4, 5, device=device, dtype=torch.int64)
    matrix = torch.eye(*matrix_size, device=device).unsqueeze(0)

    # The non-empty path rejects it too, so the empty guard is not stricter.
    with pytest.raises(RuntimeError) as full:
        op(src, matrix, (1, 4, 5))
    with pytest.raises(type(full.value), match="floating point src"):
        op(src, matrix, (0, 4, 5))


@pytest.mark.parametrize("op_name", ["warp_affine3d", "warp_perspective3d"])
def test_empty_destination_keeps_grid_sample_validation(op_name, device, dtype):
    src = torch.rand(2, 2, 3, 4, 5, device=device, dtype=dtype)
    matrix_size = (3, 4) if op_name == "warp_affine3d" else (4, 4)
    transform = torch.eye(*matrix_size, device=device, dtype=dtype).repeat(3, 1, 1)
    op = getattr(proj, op_name)

    with pytest.raises(RuntimeError, match="same batch size"):
        op(src, transform, (0, 4, 5))
    with pytest.raises(ValueError, match="expected mode"):
        op(src[:1], transform[:1], (0, 4, 5), flags="invalid")
    padding_arg = "padding_mode" if op_name == "warp_affine3d" else "border_mode"
    with pytest.raises(ValueError, match="expected padding_mode"):
        op(src[:1], transform[:1], (0, 4, 5), **{padding_arg: "invalid"})
    if device.type == "cpu":
        other_dtype = torch.float64 if dtype != torch.float64 else torch.float32
        with pytest.raises(RuntimeError):
            op(src[:1], transform[:1].to(other_dtype), (0, 4, 5))

    if op_name == "warp_perspective3d":
        with pytest.raises(ValueError, match="Bx4x4"):
            op(src[:1], torch.eye(3, device=device, dtype=dtype).unsqueeze(0), (0, 4, 5))


@pytest.mark.parametrize("dsize", [(3, 4, 5), (0, 4, 5)])
def test_warp_perspective3d_accepts_an_unbatched_matrix(dsize, device, dtype):
    """An unbatched 4x4 matrix has always warped correctly here; the shape guard must not reject it."""
    src = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
    transform = torch.eye(4, device=device, dtype=dtype)
    unbatched = proj.warp_perspective3d(src, transform, dsize)
    batched = proj.warp_perspective3d(src, transform.unsqueeze(0), dsize)
    assert unbatched.shape == batched.shape
    assert torch.equal(unbatched, batched)


def test_homography_warp3d_negative_destination_raises(device, dtype):
    src = torch.rand(1, 2, 3, 4, 5, device=device, dtype=dtype)
    transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    with pytest.raises(ValueError, match="must be non-negative"):
        proj.homography_warp3d(src, transform, (-1, 4, 5))


class TestWarpPerspective3d(BaseTester):
    # Before #4502 none of these could pass: homography_warp3d handed create_meshgrid3d's (d, x, y)
    # grid straight to warp_grid3d and grid_sample, which both read (x, y, z), so even an identity
    # homography sampled the wrong voxel. Measured on main with the arange volumes below:
    #   identity, (4, 4, 4): max|out - in| = 45.0 at align_corners=True, 55.125 at False
    #   identity, (2, 3, 4): 9.0 and 20.125;  (1, 5, 3): 10.5 at False
    # The only prior numerical test compared an unbatched call with a batched one, and both were
    # wrong the same way. After that fix the align_corners=False residual was still 55.125 on the
    # cube (20.125 on (2, 3, 4), 10.5 on (1, 5, 3)) because the 3D normalization was corner-aligned
    # whatever flag reached grid_sample (#4503); the tests below now pin both conventions.

    @staticmethod
    def _arange_volume(dsize, device, dtype):
        d, h, w = dsize
        return torch.arange(float(d * h * w), device=device, dtype=dtype).view(1, 1, d, h, w)

    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("dsize", [(4, 4, 4), (2, 3, 4), (1, 5, 3), (3, 1, 4)])
    def test_convention_identity_reproduces_input(self, dsize, align_corners, device, dtype):
        # Both conventions: the normalization now follows the flag handed to grid_sample (#4503).
        sample = self._arange_volume(dsize, device, dtype)
        identity = torch.eye(4, device=device, dtype=dtype)[None]
        out = proj.warp_perspective3d(sample, identity, dsize, align_corners=align_corners)
        self.assert_close(out, sample)

    @pytest.mark.parametrize("size", [8, 32])
    def test_identity_float64_precision(self, device, size):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        # the sampling grid is built in the input dtype, so a float64 identity warp is exact to
        # float64 roundoff like warp_affine3d, not to float32 grid precision
        dsize = (size // 2, size, size)
        sample = torch.rand(1, 1, *dsize, device=device, dtype=torch.float64)
        identity = torch.eye(4, device=device, dtype=torch.float64)[None]
        out = proj.warp_perspective3d(sample, identity, dsize, align_corners=True)
        assert out.dtype == torch.float64
        self.assert_close(out, sample, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("dsize", [(5, 5, 5), (2, 3, 5), (3, 2, 5)])
    def test_convention_agrees_with_warp_affine3d(self, dsize, device, dtype):
        # warp_affine3d builds its grid with F.affine_grid, which is (x, y, z) already, so it never
        # had the channel-order defect. Given the same affine matrix the two warps must agree.
        # Every size is one more than a power of two so the corner-aligned scale 2 / (size - 1) is
        # exact in every dtype; the two routes (affine_grid against meshgrid-then-matmul) then
        # agree bit for bit at float16 and bfloat16 as well, whereas a size-4 axis makes them round
        # differently, by up to 0.023 at bfloat16 over ten seeds, which would say nothing about the
        # channel order under test. No singleton axis: torch's CPU F.affine_grid has no
        # float16/bfloat16 kernel for a size-1 dimension, and that is the reference, not the
        # subject. The identity test above covers singleton depth and height.
        sample = torch.rand(1, 2, *dsize, device=device, dtype=dtype)
        _, _, D, H, W = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)
        angles = torch.tensor([[20.0, -15.0, 30.0]], device=device, dtype=dtype)
        scales = torch.ones_like(angles)
        P = proj.get_projective_transform(center, angles, scales)  # (1, 3, 4)
        M = kornia.geometry.convert_affinematrix_to_homography3d(P)  # (1, 4, 4)
        out_perspective = proj.warp_perspective3d(sample, M, dsize, align_corners=True)
        out_affine = proj.warp_affine3d(sample, P, dsize, align_corners=True)
        self.assert_close(out_perspective, out_affine)

    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_convention_whole_voxel_translation(self, axis, device, dtype):
        # A +1 translation along x must shift columns (W), along y rows (H), along z slices (D).
        # With the channels in the wrong order a shift along x moved slices instead. The three
        # sizes differ so that shifting the wrong axis cannot reproduce the expected volume, and
        # each is one more than a power of two so that the corner-aligned scale 2 / (size - 1) is
        # exact in every dtype: with W = 4 that scale is 2/3, and at float16/bfloat16 the shift
        # lands a fraction of a voxel off, exactly as the 2D test_translation manifest entries
        # record for the same reason.
        dsize = (2, 3, 5)
        sample = self._arange_volume(dsize, device, dtype)
        M = torch.eye(4, device=device, dtype=dtype)[None].clone()
        M[:, axis, 3] = 1.0
        expected = torch.zeros_like(sample)
        if axis == 0:
            expected[..., 1:] = sample[..., :-1]
        elif axis == 1:
            expected[..., 1:, :] = sample[..., :-1, :]
        else:
            expected[:, :, 1:] = sample[:, :, :-1]
        out = proj.warp_perspective3d(sample, M, dsize, align_corners=True)
        self.assert_close(out, expected)

    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_convention_whole_voxel_translation_align_corners_false(self, axis, device, dtype):
        # The align_corners=False twin of the test above (#4503). Sizes are powers of two here so
        # that the half-pixel scale 2 / size is exact in every dtype, the mirror image of the
        # 2^k + 1 sizes the corner-aligned test uses. Before #4503 the x shift on a (2, 3, 4)
        # volume was off by 6.6667 at this setting.
        dsize = (2, 4, 8)
        sample = self._arange_volume(dsize, device, dtype)
        M = torch.eye(4, device=device, dtype=dtype)[None].clone()
        M[:, axis, 3] = 1.0
        expected = torch.zeros_like(sample)
        if axis == 0:
            expected[..., 1:] = sample[..., :-1]
        elif axis == 1:
            expected[..., 1:, :] = sample[..., :-1, :]
        else:
            expected[:, :, 1:] = sample[:, :, :-1]
        out = proj.warp_perspective3d(sample, M, dsize, align_corners=False)
        self.assert_close(out, expected)

    def test_convention_projective_row_is_honoured(self, device, dtype):
        # A (B, 4, 4) matrix with a non-trivial last row must change the output; otherwise the
        # fix would only have made the function a slower warp_affine3d.
        dsize = (2, 3, 4)
        sample = self._arange_volume(dsize, device, dtype)
        M = torch.eye(4, device=device, dtype=dtype)[None].clone()
        M[:, 3, 0] = 0.05
        out = proj.warp_perspective3d(sample, M, dsize, align_corners=True)
        assert not torch.allclose(out, sample)

    @pytest.mark.parametrize("align_corners", [True, False])
    def test_homography_warp3d_identity(self, align_corners, device, dtype):
        # The functional entry point with an already-normalized identity, both grid conventions:
        # the sampling grid is built under the flag grid_sample is called with (#4503).
        dsize = (2, 3, 4)
        sample = self._arange_volume(dsize, device, dtype)
        identity = torch.eye(4, device=device, dtype=dtype)[None]
        out = proj.homography_warp3d(sample, identity, dsize, align_corners=align_corners)
        self.assert_close(out, sample)
        out_pix = proj.homography_warp3d(
            sample, identity, dsize, align_corners=align_corners, normalized_coordinates=False
        )
        assert out_pix.shape == sample.shape

    def test_gradcheck(self, device):
        sample = torch.rand(1, 1, 2, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        M = torch.eye(4, device=device, dtype=torch.float64)[None].clone()
        M[:, :3, 3] = 0.3
        self.gradcheck(proj.warp_perspective3d, (sample, M, (2, 3, 4)), requires_grad=(True, False, False))


class TestWarpAffine3d(BaseTester):
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_convention_whole_voxel_translation(self, align_corners, device, dtype):
        # A +1 voxel translation along x must shift columns by exactly one voxel under either
        # convention. Before #4503 normalize_homography3d was corner-aligned whatever flag went to
        # grid_sample, so at align_corners=False this was off by 6.6667 on a (2, 3, 4) arange
        # volume. Sizes are powers of two plus one on the corner-aligned side (2/(size-1) exact)
        # and powers of two on the half-pixel side (2/size exact), so the pin is exact in every
        # dtype rather than rounding at float16/bfloat16.
        dsize = (2, 3, 5) if align_corners else (2, 4, 8)
        d, h, w = dsize
        sample = torch.arange(float(d * h * w), device=device, dtype=dtype).view(1, 1, d, h, w)
        M = torch.eye(3, 4, device=device, dtype=dtype)[None].clone()
        M[:, 0, 3] = 1.0
        expected = torch.zeros_like(sample)
        expected[..., 1:] = sample[..., :-1]
        out = proj.warp_affine3d(sample, M, dsize, align_corners=align_corners)
        self.assert_close(out, expected)

    def test_smoke(self, device, dtype):
        sample = torch.rand(1, 3, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(1, 3, 4, device=device, dtype=dtype)
        output = proj.warp_affine3d(sample, P, (3, 4, 5))
        assert output.shape == (1, 3, 3, 4, 5)

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("num_channels", [1, 3, 5])
    @pytest.mark.parametrize("out_shape", [(3, 3, 3), (4, 5, 6)])
    def test_batch(self, batch_size, num_channels, out_shape, device, dtype):
        B, C = batch_size, num_channels
        sample = torch.rand(B, C, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(B, 3, 4, device=device, dtype=dtype)
        output = proj.warp_affine3d(sample, P, out_shape)
        assert list(output.shape) == [B, C, *list(out_shape)]

    def test_gradcheck(self, device):
        # generate input data
        sample = torch.rand(1, 3, 3, 4, 5, device=device, dtype=torch.float64, requires_grad=True)
        P = torch.rand(1, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(proj.warp_affine3d, (sample, P, (3, 3, 3)))

    def test_forth_back(self, device, dtype):
        out_shape = (3, 4, 5)
        sample = torch.rand(2, 5, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(2, 3, 4, device=device, dtype=dtype)
        P = kornia.geometry.convert_affinematrix_to_homography3d(P)
        P_hat = (_torch_inverse_cast(P) @ P)[:, :3]
        output = proj.warp_affine3d(sample, P_hat, out_shape, flags="nearest")
        self.assert_close(output, sample, rtol=1e-4, atol=1e-4)

    def test_rotate_x(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        _, _, D, H, W = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[90.0, 0.0, 0.0]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        self.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_rotate_y(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        _, _, D, H, W = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[0.0, 90.0, 0.0]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        self.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_rotate_z(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        _, _, D, H, W = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[0.0, 0.0, 90.0]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        self.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_rotate_y_large(self, device, dtype):
        """Rotates 90deg anti-clockwise."""
        sample = torch.tensor(
            [
                [
                    [
                        [[0.0, 4.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 9.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 6.0, 7.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[4.0, 2.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 6.0, 8.0], [9.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        _, _, D, H, W = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)

        angles = torch.tensor([[0.0, 90.0, 0.0]], device=device, dtype=dtype)

        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        self.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestGetRotationMatrix3d(BaseTester):
    def test_smoke(self, device, dtype):
        center = torch.rand(1, 3, device=device, dtype=dtype)
        angle = torch.rand(1, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        assert P.shape == (1, 3, 4)

    @pytest.mark.parametrize("batch_size", [1, 3, 6])
    def test_batch(self, batch_size, device, dtype):
        B: int = batch_size
        center = torch.rand(B, 3, device=device, dtype=dtype)
        angle = torch.rand(B, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        assert P.shape == (B, 3, 4)

    def test_identity(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.zeros(1, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device=device, dtype=dtype
        ).unsqueeze(0)
        self.assert_close(P, P_expected, atol=1e-4, rtol=1e-4)

    def test_rot90x(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[90.0, 0.0, 0.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype
        ).unsqueeze(0)
        self.assert_close(P, P_expected, atol=1e-4, rtol=1e-4)

    def test_rot90y(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[0.0, 90.0, 0.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype
        ).unsqueeze(0)
        self.assert_close(P, P_expected, atol=1e-4, rtol=1e-4)

    def test_rot90z(self, device, dtype):
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[0.0, 0.0, 90.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor(
            [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device=device, dtype=dtype
        ).unsqueeze(0)
        self.assert_close(P, P_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # generate input data
        center = torch.rand(1, 3, device=device, dtype=torch.float64, requires_grad=True)
        angle = torch.rand(1, 3, device=device, dtype=torch.float64)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=torch.float64)
        self.gradcheck(proj.get_projective_transform, (center, angle, scales))


class TestPerspectiveTransform3D(BaseTester):
    @pytest.mark.skip("Not working")
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_get_perspective_transform3d(self, batch_size, device, dtype):
        # generate input data
        # d_max, h_max, w_max = 16, 64, 32  # height, width
        # d = torch.ceil(d_max * torch.rand(batch_size, device=device, dtype=dtype))
        # h = torch.ceil(h_max * torch.rand(batch_size, device=device, dtype=dtype))
        # w = torch.ceil(w_max * torch.rand(batch_size, device=device, dtype=dtype))

        norm = torch.rand(batch_size, 8, 3, device=device, dtype=dtype)
        points_src = torch.rand_like(norm, device=device, dtype=dtype)
        points_dst = points_src + norm

        # compute transform from source to target
        dst_homo_src = kornia.geometry.transform.get_perspective_transform3d(points_src, points_dst)

        # TODO: get_perspective_transform3d seems to be correct since it would result in the
        # expected output for cropping volumes. Not sure what is going on here.
        self.assert_close(
            kornia.geometry.linalg.transform_points(dst_homo_src, points_src), points_dst, rtol=1e-4, atol=1e-4
        )

        # compute gradient check
        self.gradcheck(kornia.geometry.transform.get_perspective_transform3d, (points_src, points_dst), fast_mode=False)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_get_perspective_transform3d_2(self, batch_size, device, dtype):
        torch.manual_seed(0)
        src = kornia.geometry.bbox.bbox_generator3d(
            torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype),
            torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype),
            torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype),
            torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size),
            torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size),
            torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size),
        ).to(device=device, dtype=dtype)
        dst = kornia.geometry.bbox.bbox_generator3d(
            torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype),
            torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype),
            torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype),
            torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size),
            torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size),
            torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size),
        ).to(device=device, dtype=dtype)
        out = kornia.geometry.transform.get_perspective_transform3d(src, dst)
        if batch_size == 1:
            expected = torch.tensor(
                [
                    [
                        [3.3000, 0.0000, 0.0000, -118.2000],
                        [0.0000, 0.0769, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.5517, 28.7930],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ]
                ],
                device=device,
                dtype=dtype,
            )
        if batch_size == 2:
            expected = torch.tensor(
                [
                    [
                        [0.9630, 0.0000, 0.0000, -9.3702],
                        [0.0000, 2.0000, 0.0000, -49.9999],
                        [0.0000, 0.0000, 0.3830, 44.0213],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ],
                    [
                        [0.9630, 0.0000, 0.0000, -36.5555],
                        [0.0000, 2.0000, 0.0000, -14.0000],
                        [0.0000, 0.0000, 0.3830, 16.8940],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                    ],
                ],
                device=device,
                dtype=dtype,
            )

        self.assert_close(out, expected, rtol=1e-4, atol=1e-4)

        # compute gradient check
        self.gradcheck(kornia.geometry.transform.get_perspective_transform3d, (src, dst))
