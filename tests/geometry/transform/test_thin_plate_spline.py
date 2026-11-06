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

import math

import pytest
import torch

import kornia
from kornia.geometry.transform.thin_plate_spline import _kernel_distance

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
        src = torch.rand(batch_size, 5, 2).numpy()
        with pytest.raises(TypeError):
            assert kornia.geometry.transform.get_tps_transform(src, src)

        src = torch.rand(batch_size, 5)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.get_tps_transform(src, src)

    def test_kernel_distance_values(self, device, dtype):
        d2 = torch.tensor([0.0, 1.0, math.e**2], device=device, dtype=dtype)
        expected = torch.tensor([0.0, 0.0, math.e**2], device=device, dtype=dtype)
        self.assert_close(_kernel_distance(d2), expected)

    @pytest.mark.parametrize("grad_dtype", [torch.float32, torch.float64])
    def test_kernel_distance_zero_gradient(self, device, grad_dtype):
        if device.type == "mps" and grad_dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        squared_distances = torch.tensor([0.0, 1.0], device=device, dtype=grad_dtype, requires_grad=True)
        grad = torch.autograd.grad(_kernel_distance(squared_distances).sum(), squared_distances)[0]
        assert torch.isfinite(grad).all()

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

        src_bad = torch.rand(batch_size, 5)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_points_tps(src_bad, src, kernel, affine)

        src_bad = torch.rand(batch_size, 5)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_points_tps(src, src_bad, kernel, affine)

        kernel_bad = torch.rand(batch_size, 5)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_points_tps(src, src, kernel_bad, affine)

        affine_bad = torch.rand(batch_size, 3)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_points_tps(src, src, kernel, affine_bad)

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

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        op = kornia.geometry.transform.warp_points_tps
        op_jit = torch.jit.script(op)
        self.assert_close(op(src, dst, kernel, affine), op_jit(src, dst, kernel, affine))


class TestThinPlateSplineAutocast(BaseTester):
    """A caller's autocast region does not reach the kernel distances of the thin plate spline (#6573)."""

    # Exact: in and out of the region run the same arithmetic on one device.
    # Without the guard, warp_points_tps differs by 6.5e-05 in autocast(float16), which passes the float32 default.

    @pytest.mark.parametrize(
        "entry", ["get_tps_transform", "warp_points_tps", "warp_image_tps", "RandomThinPlateSpline"]
    )
    @pytest.mark.parametrize("autocast_dtype", ["bfloat16", "float16"])
    def test_region_does_not_change_the_result_6573(self, entry, autocast_dtype, device):
        # The one matrix product of _pair_square_euclidean ran in the autocast dtype: in bfloat16, a float32
        # warp_image_tps of a 256 px [0, 1] image was 0.08 off and a seeded RandomThinPlateSpline 0.29.
        if device.type not in ("cpu", "cuda", "mps"):
            pytest.skip("CPU, CUDA and MPS autocast")
        if device.type == "cuda" and autocast_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            pytest.skip("this CUDA device does not support bfloat16")
        image = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(0)).to(device)
        src = torch.tensor([[[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8], [0.1, -0.2]]], device=device)
        dst = src + torch.tensor(
            [[[0.03, -0.02], [-0.02, 0.05], [0.04, 0.01], [-0.03, -0.04], [0.02, 0.03]]], device=device
        )
        transform = kornia.geometry.transform

        def run():
            if entry == "RandomThinPlateSpline":
                torch.manual_seed(3)
                return kornia.augmentation.RandomThinPlateSpline(p=1.0)(image)
            kernel, affine = transform.get_tps_transform(src, dst)
            if entry == "get_tps_transform":
                return torch.cat([kernel, affine], dim=1)
            if entry == "warp_points_tps":
                return transform.warp_points_tps(src, src, kernel, affine)
            return transform.warp_image_tps(image, src, kernel, affine, align_corners=True)

        expected = run()
        with torch.autocast(device.type, dtype=getattr(torch, autocast_dtype)):
            actual = run()
        assert actual.dtype == torch.float32
        self.assert_close(actual, expected, rtol=0, atol=0)


class TestThinPlateSplineMixedDtypesAutocast(BaseTester):
    """Half points with float32 ones meet in float32 in a caller's autocast region, which casts both sides (#6573)."""

    # Exact: the mixed call in the region and the float32 call outside it run the same arithmetic on one device.

    @staticmethod
    def _skip_unsupported(device, autocast_dtype):
        if device.type not in ("cpu", "cuda", "mps"):
            pytest.skip("CPU, CUDA and MPS autocast")
        if device.type == "cuda" and autocast_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            pytest.skip("this CUDA device does not support bfloat16")

    @staticmethod
    def _inputs(device):
        image = torch.rand(1, 3, 96, 96, generator=torch.Generator().manual_seed(0)).to(device)
        src = torch.tensor([[[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8], [0.1, -0.2]]], device=device)
        dst = src + torch.tensor(
            [[[0.03, -0.02], [-0.02, 0.05], [0.04, 0.01], [-0.03, -0.04], [0.02, 0.03]]], device=device
        )
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        return image, src, dst, kernel, affine

    @pytest.mark.parametrize("half", ["bfloat16", "float16"])
    @pytest.mark.parametrize("autocast_dtype", ["bfloat16", "float16"])
    def test_float32_image_half_kernel_centers_6573(self, half, autocast_dtype, device):
        # A float32 image with kernel centers and weights a model predicted under AMP: the base ran this in the autocast
        # dtype, and the guard around the matrix product of the distances made it raise.
        self._skip_unsupported(device, autocast_dtype)
        half = getattr(torch, half)
        image, src, _, kernel, affine = self._inputs(device)
        warp = kornia.geometry.transform.warp_image_tps
        src, kernel, affine = src.to(half), kernel.to(half), affine.to(half)
        expected = warp(image, src.float(), kernel.float(), affine.float(), align_corners=True)
        with torch.autocast(device.type, dtype=getattr(torch, autocast_dtype)):
            actual = warp(image, src, kernel, affine, align_corners=True)
        assert actual.dtype == torch.float32
        self.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("first,second", [("float32", "bfloat16"), ("bfloat16", "float32"), ("float32", "float16")])
    @pytest.mark.parametrize("autocast_dtype", ["bfloat16", "float16"])
    def test_mixed_points_6573(self, first, second, autocast_dtype, device):
        self._skip_unsupported(device, autocast_dtype)
        if device.type == "cpu" and autocast_dtype not in (first, second):
            pytest.skip("CPU autocast rejects a half tensor of the other half dtype in torch.cat")
        _, src, dst, kernel, affine = self._inputs(device)
        src, dst = src.to(getattr(torch, first)), dst.to(getattr(torch, second))
        transform = kornia.geometry.transform
        expected_weights = transform.get_tps_transform(src.float(), dst.float())
        expected_points = transform.warp_points_tps(src.float(), dst.float(), kernel, affine)
        with torch.autocast(device.type, dtype=getattr(torch, autocast_dtype)):
            weights = transform.get_tps_transform(src, dst)
            points = transform.warp_points_tps(src, dst, kernel, affine)
        assert points.dtype == torch.float32
        self.assert_close(points, expected_points, rtol=0, atol=0)
        self.assert_close(weights[0].float(), expected_weights[0], rtol=1e-3, atol=1e-3)
        self.assert_close(weights[1].float(), expected_weights[1], rtol=1e-3, atol=1e-3)


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
        # nn.grid_sample interpolates the at the edges it seems, so the boundaries have values < 1.
        # Under the default align_corners=False, the outermost output pixel maps back to input
        # pixel 1.75 -- three quarters of the way from the zero at pixel 1 to the one at pixel 2.
        # Pre-#3945 the same warp gave a border factor of 0.5 (input pixel 1.5): the corner-aligned
        # grid sampled under grid_sample's half-pixel convention shifted the zoom by a quarter pixel.
        expected[:, :, [0, -1], :] *= 0.75
        expected[:, :, :, [0, -1]] *= 0.75

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

        image_bad = torch.rand(batch_size, 32, 32)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_image_tps(image_bad, dst, kernel, affine)

        dst_bad = torch.rand(batch_size, 5)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_image_tps(image, dst_bad, kernel, affine)

        kernel_bad = torch.rand(batch_size, 5)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_image_tps(image, dst, kernel_bad, affine)

        affine_bad = torch.rand(batch_size, 3)
        with pytest.raises(ValueError):
            assert kornia.geometry.transform.warp_image_tps(image, dst, kernel, affine_bad)

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

    def test_convention_padding_mode_border_differs_from_zeros(self, device, dtype):
        # padding_mode is forwarded: 'border' differs from the 'zeros' default once the warp samples
        # out of bounds.
        if not supports_2d_border_padding(device):
            pytest.skip("MPS 2D grid_sample lacks 'border' padding")

        src = torch.tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]], device=device, dtype=dtype
        )
        # zoom out, so the warp samples beyond the input extent and padding_mode is observable --
        # an identity warp now stays strictly in bounds and the two modes would agree
        kernel, affine = kornia.geometry.transform.get_tps_transform(src * 0.5, src)
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_default = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine)
        out_border = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine, padding_mode="border")
        assert not torch.allclose(out_default, out_border, atol=1e-2, rtol=1e-2)

    def test_convention_control_points_normalized_coords(self, device, dtype):
        # Control points live in the normalized frame selected by align_corners: at True, a
        # one-pixel shift of a W=4 image is 2 / (4 - 1) in corner-aligned coordinates.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half-precision rounding of near-zero boundary values exceeds this test's atol")

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

    def test_convention_default_align_corners_reproduces_identity(self, device, dtype):
        # An identity TPS transform reproduces the input at the default align_corners too, as
        # test_identity_warp_align_corners does at True (#3928).
        src = torch.tensor(
            [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]], device=device, dtype=dtype
        )
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, src)
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_default = kornia.geometry.transform.warp_image_tps(img, src, kernel, affine)
        self.assert_close(out_default, img, atol=1e-2, rtol=1e-2)
