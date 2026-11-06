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


# Everything scipy-generated below comes from this snippet (scipy 1.17.1), which needs no kornia:
#
# import numpy as np
# from scipy.interpolate import RBFInterpolator as R
# np.set_printoptions(precision=16, floatmode="unique")
# src = np.array([[0.1, 0.2], [0.8, 0.15], [0.75, 0.9], [0.2, 0.7], [0.5, 0.45], [0.35, 0.95]])
# dst = np.array([[0.05, 0.25], [0.85, 0.1], [0.7, 0.95], [0.3, 0.65], [0.45, 0.55], [0.3, 0.85]])
# q = np.array([[0.25, 0.4], [0.6, 0.3], [0.65, 0.65], [0.4, 0.8], [0.15, 0.5], [0.9, 0.5]])
# far = np.array([[10.0, 0.0], [100.0, 0.0], [1000.0, 0.0]])
# print(repr(R(src, dst, kernel="thin_plate_spline", degree=1)(q)))  # _SCIPY_BETWEEN
# print(repr(R(src, dst, kernel="thin_plate_spline", degree=1)(far)))  # _SCIPY_FAR
# ctrl = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
# moved = np.array([[-1.1, -1.05], [-1.05, 1.1], [1.08, -1.1], [1.1, 1.06], [0.1, -0.05]])
# grid = np.stack(np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)), -1).reshape(-1, 2)
# out = R(moved, ctrl, kernel="thin_plate_spline", degree=1)(grid).reshape(5, 5, 2)
# print(repr(out[..., 0]), repr(out[..., 1]))  # _SCIPY_IMAGE_X, _SCIPY_IMAGE_Y
_SRC = [[0.1, 0.2], [0.8, 0.15], [0.75, 0.9], [0.2, 0.7], [0.5, 0.45], [0.35, 0.95]]
_DST = [[0.05, 0.25], [0.85, 0.1], [0.7, 0.95], [0.3, 0.65], [0.45, 0.55], [0.3, 0.85]]
_BETWEEN = [[0.25, 0.4], [0.6, 0.3], [0.65, 0.65], [0.4, 0.8], [0.15, 0.5], [0.9, 0.5]]
_SCIPY_BETWEEN = [
    [0.24505088479397882, 0.4591797017747875],
    [0.571115461907041, 0.356524944368666],
    [0.6109860322953142, 0.7249048934977661],
    [0.388242005804625, 0.7718466089474806],
    [0.20367216196378365, 0.512029193087662],
    [0.9075084819864213, 0.5386174494805783],
]
_FAR = [[10.0, 0.0], [100.0, 0.0], [1000.0, 0.0]]
_SCIPY_FAR = [
    [1.0583676128137233e01, 9.3576032245965945e-02],
    [1.0495470748604535e02, 3.3711746497845936e00],
    [1.0485273536704819e03, 3.8122912621494670e01],
]
_CONTROL = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]
_MOVED = [[-1.1, -1.05], [-1.05, 1.1], [1.08, -1.1], [1.1, 1.06], [0.1, -0.05]]
_SCIPY_IMAGE_X = [
    [-0.9153137119413088, -0.47806270765882075, -0.03010039261200464, 0.4364710984997171, 0.917115991235269],
    [-0.9448418795698329, -0.5121965156991598, -0.06851609515742198, 0.40204997867374226, 0.8900759169978865],
    [-0.9638208728874599, -0.5324548794894945, -0.09199588651359836, 0.38377467915395835, 0.877084227806611],
    [-0.9665427347724956, -0.529168568630143, -0.07988805572989602, 0.3939087864941878, 0.8835182619493236],
    [-0.9577455614096613, -0.5128959258890495, -0.05697115461316048, 0.41597101061301245, 0.9009963878372474],
]
_SCIPY_IMAGE_Y = [
    [-0.9480532362849312, -0.9257631150129504, -0.9094366386308564, -0.9032282434423441, -0.9045214853465195],
    [-0.47281672143105397, -0.4477107618097911, -0.42871058971244425, -0.4243278206339493, -0.42932515673349664],
    [-0.003499355181189506, 0.022652004271380446, 0.04386849031101582, 0.045882930969295854, 0.03835981289089567],
    [0.4569535835679169, 0.4801792691124006, 0.4970596150975856, 0.5006143388050784, 0.49545916722280736],
    [0.9112711352460217, 0.9307394080720803, 0.9443557598746105, 0.9487235455778804, 0.9463778407910982],
]


def _coordinate_ramp(device, dtype):
    """Return a 5 x 5 image whose channels are its own corner-aligned x and y, so bilinear sampling returns the grid."""
    ramp = torch.linspace(-1.0, 1.0, 5, device=device, dtype=dtype)
    y, x = torch.meshgrid(ramp, ramp, indexing="ij")
    return torch.stack([x, y])[None]


class TestBooksteinSpline(BaseTester):
    """The fit is Bookstein's thin-plate spline: kernels and side conditions both on the first argument (#6404)."""

    def test_convention_matches_scipy_between_landmarks(self, device, dtype):
        src = torch.tensor([_SRC], device=device, dtype=dtype)
        dst = torch.tensor([_DST], device=device, dtype=dtype)
        query = torch.tensor([_BETWEEN], device=device, dtype=dtype)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warped = kornia.geometry.transform.warp_points_tps(query, src, kernel, affine)
        self.assert_close(warped, torch.tensor([_SCIPY_BETWEEN], device=device, dtype=dtype))

    def test_convention_interpolates_landmarks(self, device, dtype):
        src = torch.tensor([_SRC], device=device, dtype=dtype)
        dst = torch.tensor([_DST], device=device, dtype=dtype)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        self.assert_close(kornia.geometry.transform.warp_points_tps(src, src, kernel, affine), dst)

    def test_convention_side_conditions_hold_on_the_fit_points(self, device, dtype):
        # sum_i w_i = 0 and sum_i w_i c_i = 0 are what make the spline approach its affine part far away. They hold on
        # the first argument, so a kernel_centers other than the first argument gets a warp without them.
        src = torch.tensor([_SRC], device=device, dtype=dtype)
        dst = torch.tensor([_DST], device=device, dtype=dtype)
        kernel, _ = kornia.geometry.transform.get_tps_transform(src, dst)
        self.assert_close(kernel.sum(dim=1), torch.zeros_like(kernel[:, 0]))
        self.assert_close(kernel.mT @ src, torch.zeros_like(kernel[:, :2]))

    def test_convention_far_field_matches_scipy(self, device):
        # float64: the far field cancels r^2 log r terms of 1e7, which float32 cannot resolve.
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        src = torch.tensor([_SRC], device=device, dtype=torch.float64)
        dst = torch.tensor([_DST], device=device, dtype=torch.float64)
        far = torch.tensor([_FAR], device=device, dtype=torch.float64)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warped = kornia.geometry.transform.warp_points_tps(far, src, kernel, affine)
        self.assert_close(warped, torch.tensor([_SCIPY_FAR], device=device, dtype=torch.float64), rtol=1e-9, atol=1e-9)
        # Away from the landmarks, the spline leaves its affine part only like log r: 100 times farther adds little.
        hom = torch.cat([torch.ones_like(far[..., :1]), far], -1)
        residual = (warped - hom @ affine).norm(dim=-1)[0]
        assert residual[2] < 3 * residual[0]

    def test_convention_affine_map_gives_affine_weights(self, device, dtype):
        # Dyadic coordinates, so that half precision represents the affine image exactly and the targets stay affine.
        src = torch.tensor(
            [[[0.125, 0.25], [0.75, 0.125], [0.75, 0.875], [0.25, 0.75], [0.5, 0.375], [0.375, 0.875]]],
            device=device,
            dtype=dtype,
        )
        linear = torch.tensor([[1.25, 0.25], [-0.25, 0.75]], device=device, dtype=dtype)
        shift = torch.tensor([[0.25, -0.5]], device=device, dtype=dtype)
        dst = src @ linear + shift
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        self.assert_close(kernel, torch.zeros_like(kernel))
        self.assert_close(affine, torch.cat([shift, linear])[None])

    def test_convention_constant_map(self, device, dtype):
        # Used to be singular: with the kernels on the target points, equal targets made their columns equal.
        src = torch.tensor([_SRC], device=device, dtype=dtype)
        value = torch.tensor([[0.25, -0.5]], device=device, dtype=dtype)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, value.expand(1, 6, 2))
        self.assert_close(kernel, torch.zeros_like(kernel))
        self.assert_close(affine, torch.cat([value, torch.zeros(2, 2, device=device, dtype=dtype)])[None])

    def test_convention_warp_image_tps_matches_scipy(self, device, dtype):
        # Sampling an image of its own coordinates returns the spline at every output pixel.
        control = torch.tensor([_CONTROL], device=device, dtype=dtype)
        moved = torch.tensor([_MOVED], device=device, dtype=dtype)
        kernel, affine = kornia.geometry.transform.get_tps_transform(moved, control)
        warped = kornia.geometry.transform.warp_image_tps(
            _coordinate_ramp(device, dtype), moved, kernel, affine, align_corners=True
        )
        expected = torch.tensor([[_SCIPY_IMAGE_X, _SCIPY_IMAGE_Y]], device=device, dtype=dtype)
        self.assert_close(warped, expected)

    def test_convention_random_thin_plate_spline_matches_scipy(self, device, dtype):
        control = torch.tensor([_CONTROL], device=device, dtype=dtype)
        moved = torch.tensor([_MOVED], device=device, dtype=dtype)
        aug = kornia.augmentation.RandomThinPlateSpline(align_corners=True, p=1.0)
        params = {
            "batch_prob": torch.tensor([True], device=device),
            "forward_input_shape": torch.tensor([1, 2, 5, 5], device=device),
            "src": control,
            "dst": moved,
        }
        warped = aug(_coordinate_ramp(device, dtype), params=params)
        self.assert_close(warped, torch.tensor([[_SCIPY_IMAGE_X, _SCIPY_IMAGE_Y]], device=device, dtype=dtype))

    def test_repeated_target_points(self, device, dtype):
        # Two landmarks that move to one point are a valid fold; they make no equal rows or columns in the system.
        src = torch.tensor([_SRC], device=device, dtype=dtype)
        dst = torch.tensor([_DST], device=device, dtype=dtype)
        dst[:, 1] = dst[:, 0]
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        self.assert_close(kornia.geometry.transform.warp_points_tps(src, src, kernel, affine), dst)

    def test_mismatched_dtypes_raise(self, device):
        # The two point sets no longer meet in a matrix product, which is what used to reject this.
        src = torch.tensor([_SRC], device=device, dtype=torch.float32)
        for other in (torch.bfloat16, torch.float16):
            with pytest.raises(RuntimeError, match="must have one dtype"):
                kornia.geometry.transform.get_tps_transform(src, src.to(other))
            with pytest.raises(RuntimeError, match="must have one dtype"):
                kornia.geometry.transform.get_tps_transform(src.to(other), src)

    def test_gradcheck_through_fit_and_warp(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        src = torch.tensor([_SRC], device=device, dtype=torch.float64)
        dst = torch.tensor([_DST], device=device, dtype=torch.float64)
        query = torch.tensor([_BETWEEN], device=device, dtype=torch.float64)

        def fit_and_warp(src, dst):
            kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
            return kornia.geometry.transform.warp_points_tps(query, src, kernel, affine)

        assert self.gradcheck(fit_and_warp, (src, dst), raise_exception=True)

    def test_landmark_gradient_is_exact(self, device):
        # The warp of a landmark is its target whatever the landmark is: the zero-distance kernels on the diagonal of
        # the fit must contribute no gradient of their own, though the kernel's derivative diverges there.
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        src = torch.tensor([_SRC], device=device, dtype=torch.float64, requires_grad=True)
        dst = torch.tensor([_DST], device=device, dtype=torch.float64)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warped = kornia.geometry.transform.warp_points_tps(src, src, kernel, affine)
        grad = torch.autograd.grad(warped.sum(), src)[0]
        self.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=1e-10)


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
        warp = kornia.geometry.transform.warp_points_tps(src, src, kernel, affine)
        assert warp.shape == src.shape

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_warp(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_points_tps(src, src, kernel, affine)
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
            kornia.geometry.transform.warp_points_tps, (dst, src, kernel, affine), raise_exception=True, fast_mode=True
        )

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_jit(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        op = kornia.geometry.transform.warp_points_tps
        op_jit = torch.jit.script(op)
        self.assert_close(op(dst, src, kernel, affine), op_jit(dst, src, kernel, affine))


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
        _, src, dst, kernel, affine = self._inputs(device)
        src, dst = src.to(getattr(torch, first)), dst.to(getattr(torch, second))
        transform = kornia.geometry.transform
        expected_weights = transform.get_tps_transform(src.float(), dst.float())
        expected_points = transform.warp_points_tps(dst.float(), src.float(), kernel, affine)
        with torch.autocast(device.type, dtype=getattr(torch, autocast_dtype)):
            weights = transform.get_tps_transform(src, dst)
            points = transform.warp_points_tps(dst, src, kernel, affine)
        assert points.dtype == torch.float32
        self.assert_close(points, expected_points, rtol=0, atol=0)
        self.assert_close(weights[0].float(), expected_weights[0])
        self.assert_close(weights[1].float(), expected_weights[1])


class TestWarpImage(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_smoke(self, batch_size, device, dtype):
        src, dst = _sample_points(batch_size, device)
        tensor = torch.rand(batch_size, 3, 32, 32, device=device)
        kernel, affine = kornia.geometry.transform.get_tps_transform(src, dst)
        warp = kornia.geometry.transform.warp_image_tps(tensor, src, kernel, affine)
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
        warp = kornia.geometry.transform.warp_image_tps(tensor, dst, kernel, affine)
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
            (image, src, kernel, affine),
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
        self.assert_close(op(image, src, kernel, affine), op_jit(image, src, kernel, affine), rtol=1e-4, atol=1e-4)

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
        centers = src * 0.5
        kernel, affine = kornia.geometry.transform.get_tps_transform(centers, src)
        img = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_default = kornia.geometry.transform.warp_image_tps(img, centers, kernel, affine)
        out_border = kornia.geometry.transform.warp_image_tps(img, centers, kernel, affine, padding_mode="border")
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
        warped = kornia.geometry.transform.warp_image_tps(img, dst, kernel, affine, align_corners=True)

        # Snippet used to generate expected (requires only this module):
        # img = torch.arange(16.0).view(1, 1, 4, 4)
        # src = torch.tensor([[[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.],[0.,0.]]])
        # dst = src.clone(); dst[..., 0] += 2.0 / 3.0
        # kernel, affine = get_tps_transform(dst, src)
        # warp_image_tps(img, dst, kernel, affine, align_corners=True)
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
