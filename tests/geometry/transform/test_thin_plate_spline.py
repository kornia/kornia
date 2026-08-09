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

from testing.base import BaseTester, supports_2d_border_padding


def _sample_points(batch_size, device, dtype=torch.float32):
    src = torch.tensor([[[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0], [5.0, 5.0]]], device=device, dtype=dtype)
    src = src.repeat(batch_size, 1, 1)
    dst = src + torch.rand_like(src) * 2.5
    return src, dst


class TestTransformParameters(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 4, 2, device=device)
        out = kornia.geometry.transform.get_tps_transform(src, src)
        assert len(out) == 2

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_no_warp(self, batch_size, device, dtype):
        src = torch.rand(batch_size, 5, 2, device=device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, src)
        target_kernel = torch.zeros(batch_size, 5, 2, device=device)
        target_affine = torch.zeros(batch_size, 3, 2, device=device)
        target_affine[:, [1, 2], [0, 1]] = 1.0
        self.assert_close(kernel, target_kernel, atol=1e-4, rtol=1e-4)
        self.assert_close(affine, target_affine, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_affine_only(self, batch_size, device, dtype):
        src = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]], device=device).repeat(
            batch_size, 1, 1
        )
        dst = src.clone() * 2.0
        kernel, _ = kornia.geometry.transform.get_tps_transform(src, dst)
        self.assert_close(kernel, torch.zeros_like(kernel), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_exception(self, batch_size, device, dtype):
        with pytest.raises(TypeError):
            src = torch.rand(batch_size, 5, 2).numpy()
            assert kornia.geometry.transform.get_tps_transform(src, src)

        with pytest.raises(ValueError):
            src = torch.rand(batch_size, 5)
            assert kornia.geometry.transform.get_tps_transform(src, src)

    @pytest.mark.grad()
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_gradcheck(self, batch_size, device, dtype, requires_grad):
        opts = {"device": device, "dtype": torch.float64}
        src, dst = _sample_points(batch_size, **opts)
        src.requires_grad_(requires_grad)
        dst.requires_grad_(not requires_grad)
        assert self.gradcheck(
            kornia.geometry.transform.get_tps_transform, (src, dst), raise_exception=True, fast_mode=True
        )

    @pytest.mark.jit()
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        op = kornia.geometry.transform.get_tps_transform
        op_jit = torch.jit.script(op)
        op_output = op(src, dst)
        jit_output = op_jit(src, dst)
        self.assert_close(op_output[0], jit_output[0])
        self.assert_close(op_output[1], jit_output[1])


class TestWarpPoints(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_points_tps(src, dst, kernel, affine)
        assert warp.shape == src.shape

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_warp(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_points_tps(src, dst, kernel, affine)
        self.assert_close(warp, dst, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
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
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_gradcheck(self, batch_size, device, dtype, requires_grad):
        opts = {"device": device, "dtype": torch.float64}
        src, dst = _sample_points(batch_size, **opts)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        kernel.requires_grad_(requires_grad)
        affine.requires_grad_(not requires_grad)
        assert self.gradcheck(
            kornia.geometry.transform.warp_points_tps, (src, dst, kernel, affine), raise_exception=True, fast_mode=True
        )

    @pytest.mark.jit()
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        op = kornia.geometry.transform.warp_points_tps
        op_jit = torch.jit.script(op)
        self.assert_close(op(src, dst, kernel, affine), op_jit(src, dst, kernel, affine))


class TestWarpImage(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        tensor = torch.rand(batch_size, 3, 32, 32, device=device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_image_tps(tensor, dst, kernel, affine)
        assert warp.shape == tensor.shape

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_warp(self, batch_size, device, dtype):
        src = torch.tensor([[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]], device=device).repeat(
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
        self.assert_close(warp, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
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
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_gradcheck(self, batch_size, device, dtype):
        if device.type != "cpu":
            pytest.skip("gradcheck is unstable for warp_image_tps on CUDA")
        if dtype != torch.float64:
            pytest.skip("gradcheck requires float64")

        opts = {"device": device, "dtype": torch.float64}
        src, dst = _sample_points(batch_size, **opts)

        # Compute TPS params without tracking gradients
        with torch.no_grad():
            kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)

        image = torch.rand(batch_size, 3, 32, 32, requires_grad=True, **opts)

        assert self.gradcheck(
            kornia.geometry.transform.warp_image_tps,
            (image, dst, kernel, affine),
            requires_grad=[True, False, False, False],
            raise_exception=True,
            atol=1e-4,
            rtol=1e-4,
            nondet_tol=1e-8,
            fast_mode=True,
        )

    @pytest.mark.jit()
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        image = torch.rand(batch_size, 3, 32, 32, device=device)
        op = kornia.geometry.transform.warp_image_tps
        op_jit = torch.jit.script(op)
        self.assert_close(op(image, dst, kernel, affine), op_jit(image, dst, kernel, affine), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size", [1])
    def test_identity_warp_align_corners(self, batch_size, device, dtype):
        image = torch.arange(9.0, device=device, dtype=dtype).reshape(1, 1, 3, 3)
        dst = torch.tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1)
        kernel, affine = kornia.geometry.transform.get_tps_transform(dst, dst)
        warped = kornia.geometry.transform.warp_image_tps(image, dst, kernel, affine, align_corners=True)
        self.assert_close(warped, image, atol=1e-4, rtol=1e-4)

    def test_convention_default_padding_mode_zeros(self, device, dtype):
        # warp_image_tps's padding_mode default is 'zeros': grid_sample calls that sample
        # outside bounds fill with 0, not the edge value ('border' would). This is
        # independent of the align_corners identity-mismatch bug pinned separately by
        # test_convention_default_align_corners_reproduces_identity below.
        if dtype == torch.float16:
            # get_tps_transform's linear solve is numerically unstable in float16 (produces NaN
            # kernel/affine weights) -- matches this file's pre-existing
            # test_identity_warp_align_corners float16 failure, unrelated to this convention.
            pytest.skip("get_tps_transform is numerically unstable in float16 (produces NaN)")

        src = torch.tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]], device=device, dtype=dtype
        )
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, src)
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_default = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine)

        out_zeros = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine, padding_mode="zeros")
        self.assert_close(out_default, out_zeros)

    def test_convention_padding_mode_border_differs_from_zeros(self, device, dtype):
        # Companion to test_convention_default_padding_mode_zeros above: 'border' padding
        # must actually differ from the 'zeros' default. MPS's 2D grid_sample doesn't
        # support 'border' (probed at runtime), so this half is skipped visibly there
        # instead of silently no-op'ing inside an `if` guard.
        if dtype == torch.float16:
            pytest.skip("get_tps_transform is numerically unstable in float16 (produces NaN)")
        if not supports_2d_border_padding(device):
            pytest.skip("MPS 2D grid_sample lacks 'border' padding")

        src = torch.tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]], device=device, dtype=dtype
        )
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, src)
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_default = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine)
        out_border = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine, padding_mode="border")
        assert not torch.allclose(out_default, out_border, atol=1e-2, rtol=1e-2)

    def test_convention_control_points_normalized_coords(self, device, dtype):
        # warp_image_tps's destination/output lattice is always corner-aligned
        # (create_meshgrid(h, w, normalized_coordinates=True)), independent of the
        # align_corners argument -- see the Convention block. With align_corners=True
        # passed explicitly, grid_sample's own convention matches that lattice, so
        # corner-aligned control points on both sides reproduce the intended warp
        # exactly. Pinned here with a small translation expressed in corner-aligned
        # normalized coordinates against a hardcoded expected output.
        if dtype == torch.float16:
            pytest.skip("get_tps_transform is numerically unstable in float16 (produces NaN)")
        if dtype == torch.bfloat16:
            pytest.skip("bfloat16 rounding of near-zero boundary values exceeds this test's atol")

        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)
        # 4 corners + center, normalized [-1, 1] coordinates (align_corners=True mapping).
        src = torch.tensor(
            [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]]], device=device, dtype=dtype
        )
        dst = src.clone()
        dst[..., 0] += 2.0 / 3.0  # one-pixel shift in normalized coords for W=4 (2 / (4 - 1))

        kernel, affine = kornia.geometry.transform.get_tps_transform(dst, src)
        warped = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine, align_corners=True)

        # Snippet used to generate expected (requires only this module):
        # img = torch.arange(16.0).view(1, 1, 4, 4)
        # src = torch.tensor([[[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.],[0.,0.]]])
        # dst = src.clone(); dst[..., 0] += 2.0 / 3.0
        # kernel, affine = get_tps_transform(dst, src)
        # warp_image_tps(img, src, kernel, affine, align_corners=True)
        expected = torch.tensor(
            [[[[0.0, 0.0, 1.0, 2.0], [0.0, 4.0, 5.0, 6.0], [0.0, 8.0, 9.0, 10.0], [0.0, 12.0, 13.0, 14.0]]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(warped, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.xfail(reason="warp_image_tps default align_corners=False breaks identity — kornia#3928", strict=True)
    def test_convention_default_align_corners_reproduces_identity(self, device, dtype):
        # Intended/correct behavior: an identity TPS transform warped with warp_image_tps's
        # *default* align_corners should reproduce the input image, exactly like the
        # align_corners=True case already pinned by test_identity_warp_align_corners above.
        # It currently does not (internal create_meshgrid always builds the sampling grid
        # using the align_corners=True convention, mismatching grid_sample's default False
        # convention -- see the warning in warp_image_tps's Convention block, #3928). This
        # test is marked xfail(strict=True) so that once #3928 is fixed it XPASSes and fails
        # loudly, forcing the xfail mark to be removed instead of silently staying green.
        if dtype == torch.float16:
            pytest.skip("get_tps_transform is numerically unstable in float16 (produces NaN)")

        src = torch.tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]], device=device, dtype=dtype
        )
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, src)
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_default = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine)
        self.assert_close(out_default, img, atol=1e-2, rtol=1e-2)
