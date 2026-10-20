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
from kornia.core.ops import eye_like
from kornia.core.utils import _torch_grid_sample, _torch_inverse_cast

from testing.base import BaseTester
from testing.error import compute_patch_abs_error


class TestHomographyWarper(BaseTester):
    num_tests = 10
    threshold = 0.1

    def test_identity(self, device, dtype):
        # create input data
        height, width = 2, 5
        patch_src = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = eye_like(3, patch_src)

        # instantiate warper
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)

        # warp from source to destination
        patch_dst = warper(patch_src, dst_homo_src)
        self.assert_close(patch_src, patch_dst)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_normalize_homography_identity(self, batch_size, device, dtype):
        # create input data
        height, width = 2, 5
        res = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        dst_homo_src = eye_like(3, res.expand(batch_size, -1, -1, -1))

        self.assert_close(
            kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype), res
        )

        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        self.assert_close(norm_homo, dst_homo_src)

        # change output scale
        norm_homo = kornia.geometry.conversions.normalize_homography(
            dst_homo_src, (height, width), (height * 2, width // 2)
        )
        res = torch.tensor(
            [[[4.0, 0.0, 3.0], [0.0, 1 / 3, -2 / 3], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        ).repeat(batch_size, 1, 1)
        self.assert_close(norm_homo, res, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_denormalize_homography_identity(self, batch_size, device, dtype):
        # create input data
        height, width = 2, 5

        res = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        dst_homo_src = eye_like(3, res.expand(batch_size, -1, -1, -1))

        self.assert_close(
            kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype), res
        )

        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height, width))
        self.assert_close(denorm_homo, dst_homo_src)

        # change output scale
        denorm_homo = kornia.geometry.conversions.denormalize_homography(
            dst_homo_src, (height, width), (height * 2, width // 2)
        )
        res = torch.tensor([[[0.25, 0.0, 0.0], [0.0, 3.0, 0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )
        self.assert_close(denorm_homo, res, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_normalize_homography_general(self, batch_size, device, dtype):
        # create input data
        height, width = 2, 5
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 2.0
        dst_homo_src[..., 0, 2] = 1.0
        dst_homo_src[..., 1, 2] = 2.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)

        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        res = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).expand(
            batch_size, -1, -1
        )
        self.assert_close(norm_homo, res)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_denormalize_homography_general(self, batch_size, device, dtype):
        # create input data
        height, width = 2, 5
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 2.0
        dst_homo_src[..., 0, 2] = 1.0
        dst_homo_src[..., 1, 2] = 2.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)

        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height, width))
        res = torch.tensor([[[0.5, 0.0, 3.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).expand(
            batch_size, -1, -1
        )
        self.assert_close(denorm_homo, res)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_consistency(self, batch_size, device, dtype):
        # The same high-level normalize/denormalize mutual-inverse invariant is pinned bitwise at
        # dyadic sizes by TestNormalizeHomography::test_convention_normalize_and_denormalize_round_trip
        # in tests/geometry/test_conversions.py, on its own literal, sizes and tolerances.
        # create input data
        height, width = 2, 5
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 2.0
        dst_homo_src[..., 0, 2] = 1.0
        dst_homo_src[..., 1, 2] = 2.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)

        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height, width))
        norm_denorm_homo = kornia.geometry.conversions.normalize_homography(
            denorm_homo, (height, width), (height, width)
        )
        self.assert_close(dst_homo_src, norm_denorm_homo)
        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        denorm_norm_homo = kornia.geometry.conversions.denormalize_homography(
            norm_homo, (height, width), (height, width)
        )
        self.assert_close(dst_homo_src, denorm_norm_homo)

    @pytest.mark.parametrize("offset", [1, 3, 7])
    @pytest.mark.parametrize("shape", [(4, 5), (2, 6), (4, 3), (5, 7)])
    def test_warp_grid_translation(self, shape, offset, device, dtype):
        # create input data
        height, width = shape
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device, dtype=dtype)
        dst_homo_src = eye_like(3, grid)
        dst_homo_src[..., 0, 2] = offset  # apply offset in x
        flow = kornia.geometry.transform.warp_grid(grid, dst_homo_src)

        # the grid the src plus the offset should be equal to the flow
        # on the x-axis, y-axis remains the same.
        self.assert_close(grid[..., 0].to(device=device, dtype=dtype) + offset, flow[..., 0])
        self.assert_close(grid[..., 1].to(device=device, dtype=dtype), flow[..., 1])

    @pytest.mark.parametrize("batch_shape", [(1, 1, 4, 5), (2, 2, 4, 6), (3, 1, 5, 7)])
    def test_identity_resize(self, batch_shape, device, dtype):
        # create input data
        batch_size, channels, height, width = batch_shape
        patch_src = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        dst_homo_src = eye_like(3, patch_src)

        # instantiate warper warp from source to destination
        warper = kornia.geometry.transform.HomographyWarper(height // 2, width // 2, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)

        # check the corners
        self.assert_close(patch_src[..., 0, 0], patch_dst[..., 0, 0], atol=1e-4, rtol=1e-4)
        self.assert_close(patch_src[..., 0, -1], patch_dst[..., 0, -1], atol=1e-4, rtol=1e-4)
        self.assert_close(patch_src[..., -1, 0], patch_dst[..., -1, 0], atol=1e-4, rtol=1e-4)
        self.assert_close(patch_src[..., -1, -1], patch_dst[..., -1, -1], atol=1e-4, rtol=1e-4)

    def test_convention_align_corners_default_false(self, device, dtype):
        # At HomographyWarper's bare default align_corners=False, an identity homography reproduces
        # the input: the sampling grid is built under the same convention it samples with (#3904).
        height, width = 4, 5
        patch_src = torch.arange(float(height * width), device=device, dtype=dtype).view(1, 1, height, width)
        dst_homo_src = eye_like(3, patch_src)
        warper = kornia.geometry.transform.HomographyWarper(height, width)
        patch_dst = warper(patch_src, dst_homo_src)
        self.assert_close(patch_dst, patch_src, atol=1e-3, rtol=1e-3)
        # the identity holds under either setting, so the default itself is pinned on the signatures
        import inspect

        assert (
            inspect.signature(kornia.geometry.transform.HomographyWarper.__init__).parameters["align_corners"].default
            is False
        )
        assert inspect.signature(kornia.geometry.transform.homography_warp).parameters["align_corners"].default is False

    @pytest.mark.parametrize("shape", [(4, 5), (2, 6), (4, 3), (5, 7)])
    def test_translation(self, shape, device, dtype):
        # create input data
        offset = 2.0  # in pixel
        height, width = shape
        patch_src = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = eye_like(3, patch_src)

        dst_homo_src[..., 0, 2] = offset / (width - 1)  # apply offset in x

        # instantiate warper and from source to destination
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)
        self.assert_close(patch_src[..., 1:], patch_dst[..., :-1], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(1, 1, 3, 5), (2, 2, 4, 3), (3, 1, 2, 3)])
    def test_rotation(self, batch_shape, device, dtype):
        # create input data
        batch_size, channels, height, width = batch_shape
        patch_src = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        # rotation of 90deg
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.0
        dst_homo_src[..., 0, 1] = 1.0
        dst_homo_src[..., 1, 0] = -1.0
        dst_homo_src[..., 1, 1] = 0.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)

        # instantiate warper and warp from source to destination
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)

        # check the corners
        self.assert_close(patch_src[..., 0, 0], patch_dst[..., 0, -1], atol=1e-4, rtol=1e-4)
        self.assert_close(patch_src[..., 0, -1], patch_dst[..., -1, -1], atol=1e-4, rtol=1e-4)
        self.assert_close(patch_src[..., -1, 0], patch_dst[..., 0, 0], atol=1e-4, rtol=1e-4)
        self.assert_close(patch_src[..., -1, -1], patch_dst[..., -1, 0], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_homography_warper(self, batch_size, device, dtype):
        # generate input data
        height, width = 128, 64
        eye_size = 3  # identity 3x3

        patch_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

        # create base homography
        dst_homo_src = eye_like(eye_size, patch_src)

        # instantiate warper
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)

        for _ in range(self.num_tests):
            # generate homography noise
            homo_delta = torch.rand_like(dst_homo_src) * 0.3

            dst_homo_src_i = dst_homo_src + homo_delta

            # transform the points from dst to ref
            patch_dst = warper(patch_src, dst_homo_src_i)
            patch_dst_to_src = warper(patch_dst, _torch_inverse_cast(dst_homo_src_i))

            # same transform precomputing the grid
            warper.precompute_warp_grid(_torch_inverse_cast(dst_homo_src_i))
            patch_dst_to_src_precomputed = warper(patch_dst)
            self.assert_close(patch_dst_to_src_precomputed, patch_dst_to_src, atol=1e-4, rtol=1e-4)

            # projected should be equal as initial
            error = compute_patch_abs_error(patch_src, patch_dst_to_src, height, width)

            assert error.item() < self.threshold

            # check functional api
            patch_dst_to_src_functional = kornia.geometry.transform.homography_warp(
                patch_dst, _torch_inverse_cast(dst_homo_src_i), (height, width), align_corners=True
            )

            self.assert_close(patch_dst_to_src, patch_dst_to_src_functional, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("align_corners", [True, False])
    def test_precompute_warp_grid_matches_direct(self, device, dtype, align_corners):
        # the cached grid must be built under the convention forward() hands to grid_sample,
        # otherwise the precomputed path silently disagrees with the direct one (#3904)
        height, width = 8, 6
        patch_src = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = eye_like(3, patch_src)

        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=align_corners)
        direct = warper(patch_src, dst_homo_src)

        warper.precompute_warp_grid(dst_homo_src)
        precomputed = warper(patch_src)

        # the cached-vs-direct comparison is the actual subject and is bit-exact in every dtype; the
        # identity-reproduces-input half takes the dtype-aware default, because the [-1, 1] scale is
        # not exactly representable at half precision
        self.assert_close(precomputed, direct, atol=1e-4, rtol=1e-4)
        self.assert_close(direct, patch_src)

    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("scale", [1e-7, 1e-8, 1e-12, 1e-30, 1e12, -1e8])
    def test_warp_grid_ignores_the_homography_scale(self, scale, align_corners, device, dtype):
        # A homography and any nonzero multiple of it describe the same warp. An absolute eps inside
        # convert_points_from_homogeneous used to leave grid points with |w| <= eps undivided, so the
        # sampled pixels depended on the matrix's arbitrary scale (#5761).
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("these scales test the float32/float64 arithmetic range")
        height, width = 6, 6
        image = torch.arange(float(height * width), device=device, dtype=dtype).reshape(1, 1, height, width) / 36
        dst_homo_src = torch.tensor(
            [[[1.0, 0.1, 0.03], [0.05, 1.0, 0.02], [0.1, 0.2, 1.0]]], device=device, dtype=dtype
        )
        grid = kornia.geometry.create_meshgrid(
            height, width, normalized_coordinates=True, device=device, dtype=dtype, align_corners=align_corners
        )

        self.assert_close(
            kornia.geometry.transform.warp_grid(grid, scale * dst_homo_src),
            kornia.geometry.transform.warp_grid(grid, dst_homo_src),
        )

        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=align_corners)
        self.assert_close(warper(image, scale * dst_homo_src), warper(image, dst_homo_src))

        dst_homo_src_norm = _torch_inverse_cast(
            kornia.geometry.normalize_homography(
                dst_homo_src, (height, width), (height, width), align_corners=align_corners
            )
        )
        warp = kornia.geometry.homography_warp
        kwargs = {"align_corners": align_corners, "normalized_homography": True}
        scaled = warp(image, scale * dst_homo_src_norm, (height, width), **kwargs)
        assert torch.isfinite(scaled).all()
        self.assert_close(scaled, warp(image, dst_homo_src_norm, (height, width), **kwargs))

    @pytest.mark.parametrize("scale", [1e-8, 1e-12, 1e12])
    def test_homography_warp_gradient_follows_the_projective_gauge(self, scale, device, dtype):
        # The scale is divided out with a detached factor, so the image is unchanged and the gradient
        # obeys the projective rule dL/d(s*H) = (dL/dH)/s rather than being silently rescaled (#5761).
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("these scales test the float32/float64 arithmetic range")
        height, width = 6, 6
        image = torch.arange(float(height * width), device=device, dtype=dtype).reshape(1, 1, height, width) / 36
        dst_homo_src = torch.tensor(
            [[[1.0, 0.1, 0.03], [0.05, 1.0, 0.02], [0.1, 0.2, 1.0]]], device=device, dtype=dtype
        )
        normalized = _torch_inverse_cast(
            kornia.geometry.normalize_homography(dst_homo_src, (height, width), (height, width))
        )
        reference_matrix = normalized.clone().requires_grad_()
        scaled_matrix = (normalized * scale).requires_grad_()

        reference = kornia.geometry.homography_warp(image, reference_matrix, (height, width))
        actual = kornia.geometry.homography_warp(image, scaled_matrix, (height, width))
        reference_grad = torch.autograd.grad(reference.square().sum(), reference_matrix)[0]
        actual_grad = torch.autograd.grad(actual.square().sum(), scaled_matrix)[0]

        assert torch.isfinite(actual_grad).all()
        self.assert_close(actual, reference)
        self.assert_close(actual_grad * scale, reference_grad)

    def test_warp_grid_keeps_a_gradient_for_an_all_zero_matrix(self, device, dtype):
        # A zero matrix has no nonzero magnitude to scale by. The gauge must fall back to 1 instead of
        # dividing by the infinite "smallest magnitude", which would collapse every grid gradient to 0.
        grid = kornia.geometry.create_meshgrid(3, 3, normalized_coordinates=True, device=device, dtype=dtype)
        matrix = torch.zeros(1, 3, 3, device=device, dtype=dtype).requires_grad_(True)

        warped = kornia.geometry.transform.warp_grid(grid, matrix)
        grad = torch.autograd.grad(warped.sum(), matrix)[0]

        assert torch.isfinite(warped).all()
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0

    @pytest.mark.parametrize("batch_shape", [(1, 1, 7, 5), (2, 3, 8, 5), (1, 1, 7, 16)])
    def test_gradcheck(self, batch_shape, device):
        dtype = torch.float64
        # generate input data
        eye_size = 3  # identity 3x3

        # create checkerboard
        patch_src = torch.rand(batch_shape, device=device, dtype=dtype)

        # create base homography
        _batch_size, _, height, width = patch_src.shape
        dst_homo_src = eye_like(eye_size, patch_src)

        # instantiate warper
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        # evaluate function gradient
        self.gradcheck(warper, (patch_src, dst_homo_src), requires_grad=(True, False), nondet_tol=1e-8)

    @staticmethod
    def _pixel_warp(entry, image, src_homo_dst, dsize, **options):
        # The three ways to warp with a pixel homography: the function, the module, and the module's cached grid.
        if entry == "functional":
            return kornia.geometry.transform.homography_warp(
                image, src_homo_dst, dsize, normalized_coordinates=False, **options
            )
        warper = kornia.geometry.transform.HomographyWarper(*dsize, normalized_coordinates=False, **options)
        if entry == "module":
            return warper(image, src_homo_dst)
        warper.precompute_warp_grid(src_homo_dst)
        return warper(image)

    @pytest.mark.parametrize("entry", ["functional", "module", "precomputed"])
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_pixel_coordinates_identity(self, entry, align_corners, device, dtype):
        # With normalized_coordinates=False the homography is in pixels: an identity returns the input, and a
        # smaller destination is the top-left crop of the source rather than a resize of it (#6170).
        image = torch.arange(2 * 5 * 7, device=device, dtype=dtype).view(1, 2, 5, 7) / 70
        identity = eye_like(3, image)

        self.assert_close(self._pixel_warp(entry, image, identity, (5, 7), align_corners=align_corners), image)
        self.assert_close(
            self._pixel_warp(entry, image, identity, (3, 4), align_corners=align_corners), image[..., :3, :4]
        )

    @pytest.mark.parametrize("entry", ["functional", "module", "precomputed"])
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_pixel_coordinates_whole_pixel_translation(self, entry, align_corners, device, dtype):
        # src_homo_dst maps destination pixels to source pixels, so a translation (tx, ty) samples
        # out[y, x] = src[y + ty, x + tx], one homography per sample, and zeros where that leaves the source.
        image = torch.arange(2 * 6 * 8, device=device, dtype=dtype).view(2, 1, 6, 8) / 96
        src_homo_dst = eye_like(3, image).clone()
        src_homo_dst[0, 0, 2] = 2.0
        src_homo_dst[1, 0, 2] = -3.0
        src_homo_dst[1, 1, 2] = 1.0
        expected = torch.zeros_like(image)
        expected[0, :, :, :6] = image[0, :, :, 2:]
        expected[1, :, :5, 3:] = image[1, :, 1:, :5]

        actual = self._pixel_warp(entry, image, src_homo_dst, (6, 8), align_corners=align_corners)
        # a half-precision grid lands a few thousandths of a pixel off, which the samples one pixel past the edge
        # pick up from the last row or column
        self.assert_close(actual, expected, low_tolerance=dtype in (torch.float16, torch.bfloat16))

    @pytest.mark.parametrize("entry", ["functional", "module", "precomputed"])
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_pixel_coordinates_match_warp_perspective(self, entry, align_corners, padding_mode, device, dtype):
        # warp_perspective takes the source->destination pixel homography; its inverse is the destination->source
        # pixel homography the pixel mode reads. Both normalize under the same align_corners, so they agree for every
        # padding mode, here with a perspective term and a destination of another size than the source.
        image = (torch.arange(2 * 3 * 7 * 9, device=device, dtype=dtype).view(2, 3, 7, 9) % 11) / 11
        dst_trans_src = torch.tensor(
            [
                [[0.9, 0.1, 1.5], [-0.05, 1.1, -0.7], [0.01, -0.02, 1.0]],
                [[1.2, 0.0, -2.0], [0.0, 0.8, 1.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float64,
        )
        # inverted in float64 on the CPU, which every device can then take in its own dtype
        src_homo_dst = torch.linalg.inv(dst_trans_src).to(device=device, dtype=dtype)
        dsize = (5, 11)

        expected = kornia.geometry.transform.warp_perspective(
            image,
            dst_trans_src.to(device=device, dtype=dtype),
            dsize,
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        actual = self._pixel_warp(
            entry, image, src_homo_dst, dsize, align_corners=align_corners, padding_mode=padding_mode
        )
        self.assert_close(actual, expected, low_tolerance=dtype in (torch.float16, torch.bfloat16))

    @pytest.mark.parametrize("align_corners", [True, False])
    def test_normalized_coordinates_path_is_the_plain_grid_warp(self, align_corners, device, dtype):
        # The default normalized path is grid_sample over warp_grid of a normalized meshgrid, bit for bit; the
        # pixel-mode normalization must not reach it (#6170).
        image = (torch.arange(2 * 5 * 6, device=device, dtype=dtype).view(1, 2, 5, 6) % 7) / 7
        src_homo_dst = torch.tensor(
            [[[0.95, 0.05, 0.1], [-0.03, 1.05, -0.08], [0.02, -0.01, 1.0]]], device=device, dtype=dtype
        )
        grid = kornia.geometry.create_meshgrid(
            4, 3, normalized_coordinates=True, device=device, dtype=dtype, align_corners=align_corners
        )
        warped_grid = kornia.geometry.transform.warp_grid(grid, src_homo_dst)
        expected = _torch_grid_sample(image, warped_grid, align_corners=align_corners)

        actual = kornia.geometry.transform.homography_warp(image, src_homo_dst, (4, 3), align_corners=align_corners)
        warper = kornia.geometry.transform.HomographyWarper(4, 3, align_corners=align_corners)
        direct = warper(image, src_homo_dst)
        warper.precompute_warp_grid(src_homo_dst)
        precomputed = warper(image)
        # the cached path warps the module's own base grid, built once in the default dtype
        expected_precomputed = _torch_grid_sample(
            image, kornia.geometry.transform.warp_grid(warper.grid, src_homo_dst), align_corners=align_corners
        )

        self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(direct, expected, rtol=0, atol=0)
        self.assert_close(precomputed, expected_precomputed, rtol=0, atol=0)

    @pytest.mark.parametrize("align_corners", [True, False])
    def test_gradcheck_pixel_coordinates(self, align_corners, device):
        # Gradients reach both the image and the pixel homography; sub-pixel sample positions keep the bilinear
        # kinks away from the finite differences.
        image = torch.rand(1, 2, 5, 6, device=device, dtype=torch.float64)
        src_homo_dst = torch.tensor(
            [[[1.05, 0.02, 0.3], [0.01, 0.95, -0.4], [1e-3, -2e-3, 1.0]]], device=device, dtype=torch.float64
        )

        def warp(image, src_homo_dst):
            return kornia.geometry.transform.homography_warp(
                image, src_homo_dst, (4, 5), align_corners=align_corners, normalized_coordinates=False
            )

        warper = kornia.geometry.transform.HomographyWarper(
            4, 5, align_corners=align_corners, normalized_coordinates=False
        )
        self.gradcheck(warp, (image, src_homo_dst), nondet_tol=1e-8)
        self.gradcheck(warper, (image, src_homo_dst), nondet_tol=1e-8)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("normalized_coordinates", [True, False])
    def test_dynamo(self, batch_size, align_corners, normalized_coordinates, device, dtype, torch_optimizer):
        if (
            device == torch.device("cpu")
            and batch_size != 1
            and kornia.core._compat.torch_version() in {"2.3.0", "2.3.1"}
        ):
            pytest.skip("Failing to compile batched inputs see pytorch/pytorch#126617")

        # generate input data
        height, width = 128, 64
        eye_size = 3  # identity 3x3

        patch_src = torch.rand(batch_size, 1, height, width, device=device, dtype=dtype)

        # create base homography
        dst_homo_src = eye_like(eye_size, patch_src)

        # generate homography noise
        homo_delta = torch.rand_like(dst_homo_src) * 0.3

        dst_homo_src_i = dst_homo_src + homo_delta
        # the same warp as a destination->source pixel homography: N^-1 H N, with N this align_corners' normalization
        pixel_homo = kornia.geometry.conversions.denormalize_homography(
            dst_homo_src_i, (height, width), (height, width), align_corners=align_corners
        )
        if not normalized_coordinates:
            dst_homo_src_i = pixel_homo

        # transform the points with and without jit
        patch_dst = kornia.geometry.transform.homography_warp(
            patch_src,
            dst_homo_src_i,
            (height, width),
            align_corners=align_corners,
            normalized_coordinates=normalized_coordinates,
        )
        patch_dst_optimized = torch_optimizer(kornia.geometry.transform.homography_warp)(
            patch_src,
            dst_homo_src_i,
            (height, width),
            align_corners=align_corners,
            normalized_coordinates=normalized_coordinates,
        )

        self.assert_close(patch_dst, patch_dst_optimized, atol=1e-4, rtol=1e-4)
        # compiled-vs-eager passes on a wrong result too (#6170): also check the compiled output against an eager
        # reference that does not run this code, warp_perspective of the source->destination pixel homography. It
        # runs in float64 on the CPU: in float32 its own error reaches 5e-4 on torch 2.9.1 for these perspectives.
        expected = kornia.geometry.transform.warp_perspective(
            patch_src.cpu().double(),
            torch.linalg.inv(pixel_homo.cpu().double()),
            (height, width),
            align_corners=align_corners,
        ).to(device=device, dtype=dtype)
        self.assert_close(patch_dst_optimized, expected, atol=1e-4, rtol=1e-4)


class TestHomographyNormalTransform(BaseTester):
    expected_2d_0 = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]])

    expected_2d_1 = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    expected_3d_0 = expected = torch.tensor(
        [[[0.4, 0.0, 0.0, -1.0], [0.0, 2.0, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    expected_3d_1 = torch.tensor(
        [[[0.4, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    @pytest.mark.parametrize("height,width,expected", [(2, 5, expected_2d_0), (1, 5, expected_2d_1)])
    def test_transform2d(self, height, width, expected, device, dtype):
        output = kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype)

        self.assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("height", [1, 2, 5])
    @pytest.mark.parametrize("width", [1, 2, 5])
    def test_divide_by_zero2d(self, height, width, device, dtype):
        output = kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype)
        assert torch.isinf(output).sum().item() == 0

    def test_transform2d_apply(self, device, dtype):
        height, width = 2, 5
        input = torch.tensor([[0.0, 0.0], [width - 1, height - 1]], device=device, dtype=dtype)
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=device, dtype=dtype)
        transform = kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype)
        output = kornia.geometry.linalg.transform_points(transform, input)
        self.assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    def test_transform_apply_align_corners_false(self, device, dtype):
        # align_corners=False normalizes to grid_sample's half-pixel convention, where +/-1 are the
        # outer pixel EDGES: pixel centers 0 and W-1 land at -1 + 1/W and 1 - 1/W (#3904).
        height, width = 2, 5
        input = torch.tensor([[0.0, 0.0], [width - 1, height - 1]], device=device, dtype=dtype)
        expected = torch.tensor([[-0.8, -0.5], [0.8, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.conversions.normal_transform_pixel(
            height, width, device=device, dtype=dtype, align_corners=False
        )
        output = kornia.geometry.linalg.transform_points(transform, input)
        self.assert_close(output, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("height", [1, 2, 5])
    @pytest.mark.parametrize("width", [1, 2, 5])
    def test_normalize_denormalize_round_trip_align_corners_false(self, height, width, device, dtype):
        # denormalize_homography must invert normalize_homography for either convention. The
        # align_corners=False mapping is also non-singular at a size of 1, unlike align_corners=True.
        homo = torch.tensor([[[1.0, 0.2, 0.5], [-0.1, 1.0, -0.3], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        norm = kornia.geometry.conversions.normalize_homography(homo, (height, width), (height, width), False)
        assert torch.isinf(norm).sum().item() == 0
        restored = kornia.geometry.conversions.denormalize_homography(norm, (height, width), (height, width), False)
        # dtype-aware default: the round trip multiplies by 2 / size and back, and the True-path sibling
        # test_normalize_homography_identity is already a half-precision manifest entry for the same reason
        self.assert_close(restored, homo)

    @pytest.mark.parametrize("height,width,depth,expected", [(2, 6, 4, expected_3d_0), (1, 6, 4, expected_3d_1)])
    def test_transform3d(self, height, width, depth, expected, device, dtype):
        output = kornia.geometry.conversions.normal_transform_pixel3d(depth, height, width, device=device, dtype=dtype)
        self.assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("height", [1, 2, 5])
    @pytest.mark.parametrize("width", [1, 2, 5])
    @pytest.mark.parametrize("depth", [1, 2, 5])
    def test_divide_by_zero3d(self, height, width, depth, device, dtype):
        output = kornia.geometry.conversions.normal_transform_pixel3d(depth, height, width, device=device, dtype=dtype)
        assert torch.isinf(output).sum().item() == 0

    def test_transform3d_apply(self, device, dtype):
        depth, height, width = 3, 2, 5
        input = torch.tensor([[0.0, 0.0, 0.0], [width - 1, height - 1, depth - 1]], device=device, dtype=dtype)
        expected = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
        transform = kornia.geometry.conversions.normal_transform_pixel3d(
            depth, height, width, device=device, dtype=dtype
        )
        output = kornia.geometry.linalg.transform_points(transform, input)
        self.assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)


class TestHomographyNormalizePrecision(BaseTester):
    """The normalization matrices must be built in the caller's dtype, not in float32 and then cast.

    The oracle is the documented contract of each op, evaluated natively in float64 with the same
    public helper: ``normalize_homography`` chains ``normal_transform_pixel(dst)`` with the inverse
    of ``normal_transform_pixel(src)``. Every size is picked so that ``2 / (size - 1)`` is not
    exactly representable in binary -- the 2-D pins use 2/3 and 2/5, the 3-D pin 2/3, 2/5, 2/7 and
    2/9 across all six axes -- which is what exposes the float32 rounding. A size whose
    ``size - 1`` is a power of two (2/1, 2/2, 2/4, 2/8 ...) is exact in every dtype and pins
    nothing here, so do not "simplify" these numbers.
    """

    @staticmethod
    def _skip_if_no_float64(device):
        if device.type == "mps":
            pytest.skip("float64 is not supported on MPS")

    def test_normalize_homography_float64(self, device):
        self._skip_if_no_float64(device)
        f64 = torch.float64
        dst_pix_trans_src_pix = torch.eye(3, device=device, dtype=f64)[None]

        out = kornia.geometry.conversions.normalize_homography(dst_pix_trans_src_pix, (4, 4), (6, 6))

        src_norm = kornia.geometry.conversions.normal_transform_pixel(4, 4, device=device, dtype=f64)
        dst_norm = kornia.geometry.conversions.normal_transform_pixel(6, 6, device=device, dtype=f64)
        expected = dst_norm @ _torch_inverse_cast(src_norm)
        self.assert_close(out, expected, rtol=0.0, atol=1e-15)

    def test_denormalize_homography_float64(self, device):
        self._skip_if_no_float64(device)
        f64 = torch.float64
        dst_pix_trans_src_pix = torch.eye(3, device=device, dtype=f64)[None]

        out = kornia.geometry.conversions.denormalize_homography(dst_pix_trans_src_pix, (4, 4), (6, 6))

        src_norm = kornia.geometry.conversions.normal_transform_pixel(4, 4, device=device, dtype=f64)
        dst_norm = kornia.geometry.conversions.normal_transform_pixel(6, 6, device=device, dtype=f64)
        expected = _torch_inverse_cast(dst_norm) @ src_norm
        self.assert_close(out, expected, rtol=0.0, atol=1e-15)

    def test_normalize_homography3d_float64(self, device):
        self._skip_if_no_float64(device)
        f64 = torch.float64
        dst_pix_trans_src_pix = torch.eye(4, device=device, dtype=f64)[None]

        out = kornia.geometry.conversions.normalize_homography3d(dst_pix_trans_src_pix, (4, 4, 6), (6, 8, 10))

        src_norm = kornia.geometry.conversions.normal_transform_pixel3d(4, 4, 6, device=device, dtype=f64)
        dst_norm = kornia.geometry.conversions.normal_transform_pixel3d(6, 8, 10, device=device, dtype=f64)
        expected = dst_norm @ _torch_inverse_cast(src_norm)
        self.assert_close(out, expected, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("side", [3000, 11600])
    def test_normalize_homography_of_a_large_image(self, device, dtype, side):
        # #5197: normalize_homography inverts the source normalization matrix, whose determinant 4 / (side - 1)**2 is
        # subnormal in float16 above about 256 px and rounds to zero there from about 11.6k px. The inverse ran
        # in float16, so the result was 6.8 % off at 3000 px and NaN at 11600 px; it now runs in float32 for half
        # input. Oracle: the same call in float64 on the CPU, on the same dtype-rounded homography. The bound is a
        # few roundings in `dtype` at the largest entry: the normalization matrices are built in `dtype` and the
        # chain of two products is rounded there.
        homography = torch.tensor([[[1.0, 0.02, 3.0], [0.01, 0.98, -2.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = kornia.geometry.conversions.normalize_homography(homography, (side, side), (side, side))

        assert out.dtype == dtype
        expected = kornia.geometry.conversions.normalize_homography(
            homography.cpu().double(), (side, side), (side, side)
        )
        atol = 4 * torch.finfo(dtype).eps * expected.abs().max().item()
        self.assert_close(out.cpu().double(), expected, rtol=0.0, atol=atol)

    def test_dtype_device_preserved(self, device, dtype):
        homo_2d = torch.eye(3, device=device, dtype=dtype)[None]
        homo_3d = torch.eye(4, device=device, dtype=dtype)[None]

        for out in (
            kornia.geometry.conversions.normalize_homography(homo_2d, (4, 4), (6, 6)),
            kornia.geometry.conversions.denormalize_homography(homo_2d, (4, 4), (6, 6)),
            kornia.geometry.conversions.normalize_homography3d(homo_3d, (2, 4, 5), (3, 8, 9)),
        ):
            assert out.dtype == dtype
            assert out.device.type == device.type


class TestHomographyWarper3D(BaseTester):
    num_tests = 10
    threshold = 0.1

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_normalize_homography_identity(self, batch_size, device, dtype):
        # create input data
        input_shape = (4, 8, 5)

        res = torch.tensor(
            [[[0.5000, 0.0, 0.0, -1.0], [0.0, 0.2857, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        norm = kornia.geometry.conversions.normal_transform_pixel3d(
            input_shape[0],
            input_shape[1],
            input_shape[2],
        ).to(device=device, dtype=dtype)

        dst_homo_src = eye_like(4, norm.expand(batch_size, -1, -1, -1))

        _atol = 1e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-4
        self.assert_close(norm, res, rtol=1e-4, atol=_atol)

        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, input_shape, input_shape).to(
            device=device, dtype=dtype
        )
        self.assert_close(norm_homo, dst_homo_src, rtol=1e-4, atol=_atol)

        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, input_shape, input_shape).to(
            device=device, dtype=dtype
        )
        self.assert_close(norm_homo, dst_homo_src, rtol=1e-4, atol=_atol)

        # change output scale
        norm_homo = kornia.geometry.conversions.normalize_homography3d(
            dst_homo_src, input_shape, (input_shape[0] // 2, input_shape[1] * 2, input_shape[2] // 2)
        ).to(device=device, dtype=dtype)
        res = torch.tensor(
            [[[4.0, 0.0, 0.0, 3.0], [0.0, 0.4667, 0.0, -0.5333], [0.0, 0.0, 3.0, 2.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1)

        atol = 1e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-4
        self.assert_close(norm_homo, res, rtol=1e-4, atol=atol)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_normalize_homography_general(self, batch_size, device, dtype):
        # create input data
        dst_homo_src = torch.eye(4, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 0.5
        dst_homo_src[..., 2, 2] = 2.0
        dst_homo_src[..., 0, 3] = 1.0
        dst_homo_src[..., 1, 3] = 2.0
        dst_homo_src[..., 2, 3] = 3.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)

        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, (2, 2, 5), (2, 2, 5))
        res = torch.tensor(
            [[[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 3.5], [0.0, 0.0, 2.0, 7.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).expand(batch_size, -1, -1)
        self.assert_close(norm_homo, res)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_denormalize_homography_general(self, batch_size, device, dtype):
        # The inverse of test_normalize_homography_general: denormalizing its expected result gives
        # back its input. Every entry and normalization constant is dyadic, so the values are exact
        # in every dtype.
        norm_homo = torch.tensor(
            [[[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 3.5], [0.0, 0.0, 2.0, 7.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).expand(batch_size, -1, -1)

        dst_homo_src = kornia.geometry.conversions.denormalize_homography3d(norm_homo, (2, 2, 5), (2, 2, 5))
        res = torch.tensor(
            [[[0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 2.0], [0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).expand(batch_size, -1, -1)
        assert dst_homo_src.shape == (batch_size, 4, 4)
        self.assert_close(dst_homo_src, res)

    def test_denormalize_homography_invalid_input(self, device):
        with pytest.raises(TypeError, match=r"Input type is not a torch.Tensor"):
            kornia.geometry.conversions.denormalize_homography3d([[1.0]], (3, 5, 7), (6, 8, 11))

        with pytest.raises(ValueError, match="must be a Bx4x4 tensor"):
            kornia.geometry.conversions.denormalize_homography3d(torch.ones(2, 2, device=device), (3, 5, 7), (6, 8, 11))

    @pytest.mark.parametrize("offset", [1, 3, 7])
    @pytest.mark.parametrize("shape", [(4, 5, 6), (2, 4, 6), (4, 3, 9), (5, 7, 8)])
    def test_warp_grid_translation(self, shape, offset, device, dtype):
        # create input data
        depth, height, width = shape
        grid = kornia.geometry.create_meshgrid3d(
            depth, height, width, normalized_coordinates=False, dtype=dtype, device=device
        )
        dst_homo_src = eye_like(4, grid)
        dst_homo_src[..., 0, 3] = offset  # apply offset in x

        flow = kornia.geometry.transform.warp_grid3d(grid, dst_homo_src)

        # the grid the src plus the offset should be equal to the flow
        # on the x-axis, y-axis remains the same.
        self.assert_close(grid[..., 0].to(device=device, dtype=dtype) + offset, flow[..., 0], atol=1e-4, rtol=1e-4)
        self.assert_close(grid[..., 1].to(device=device, dtype=dtype), flow[..., 1], atol=1e-4, rtol=1e-4)
        self.assert_close(grid[..., 2].to(device=device, dtype=dtype), flow[..., 2], atol=1e-4, rtol=1e-4)
