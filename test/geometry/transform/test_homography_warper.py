import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close
from kornia.utils.helpers import _torch_inverse_cast


class TestHomographyWarper:
    num_tests = 10
    threshold = 0.1

    def test_identity(self, device, dtype):
        # create input data
        height, width = 2, 5
        patch_src = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=3, device=device, dtype=dtype)

        # instantiate warper
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)

        # warp from source to destination
        patch_dst = warper(patch_src, dst_homo_src)
        assert_close(patch_src, patch_dst)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_normalize_homography_identity(self, batch_size, device, dtype):
        # create input data
        height, width = 2, 5
        dst_homo_src = utils.create_eye_batch(batch_size=batch_size, eye_size=3, device=device, dtype=dtype)

        res = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        assert (
            kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype) == res
        ).all()

        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        assert (norm_homo == dst_homo_src).all()

        # change output scale
        norm_homo = kornia.geometry.conversions.normalize_homography(
            dst_homo_src, (height, width), (height * 2, width // 2)
        )
        res = torch.tensor(
            [[[4.0, 0.0, 3.0], [0.0, 1 / 3, -2 / 3], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        ).repeat(batch_size, 1, 1)
        assert_close(norm_homo, res, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_denormalize_homography_identity(self, batch_size, device, dtype):
        # create input data
        height, width = 2, 5
        dst_homo_src = utils.create_eye_batch(batch_size=batch_size, eye_size=3, device=device, dtype=dtype)

        res = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        assert (
            kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype) == res
        ).all()

        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height, width))
        assert (denorm_homo == dst_homo_src).all()

        # change output scale
        denorm_homo = kornia.geometry.conversions.denormalize_homography(
            dst_homo_src, (height, width), (height * 2, width // 2)
        )
        res = torch.tensor([[[0.25, 0.0, 0.0], [0.0, 3.0, 0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )
        assert_close(denorm_homo, res, atol=1e-4, rtol=1e-4)

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
        res = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        assert (norm_homo == res).all()

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
        res = torch.tensor([[[0.5, 0.0, 3.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        assert (denorm_homo == res).all()

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_consistency(self, batch_size, device, dtype):
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
        assert (dst_homo_src == norm_denorm_homo).all()
        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        denorm_norm_homo = kornia.geometry.conversions.denormalize_homography(
            norm_homo, (height, width), (height, width)
        )
        assert (dst_homo_src == denorm_norm_homo).all()

    @pytest.mark.parametrize("offset", [1, 3, 7])
    @pytest.mark.parametrize("shape", [(4, 5), (2, 6), (4, 3), (5, 7)])
    def test_warp_grid_translation(self, shape, offset, device, dtype):
        # create input data
        height, width = shape
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 2] = offset  # apply offset in x
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False)
        flow = kornia.geometry.transform.warp_grid(grid, dst_homo_src)

        # the grid the src plus the offset should be equal to the flow
        # on the x-axis, y-axis remains the same.
        assert_close(grid[..., 0].to(device=device, dtype=dtype) + offset, flow[..., 0])
        assert_close(grid[..., 1].to(device=device, dtype=dtype), flow[..., 1])

    @pytest.mark.parametrize("batch_shape", [(1, 1, 4, 5), (2, 2, 4, 6), (3, 1, 5, 7)])
    def test_identity_resize(self, batch_shape, device, dtype):
        # create input data
        batch_size, channels, height, width = batch_shape
        patch_src = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size=3, device=device, dtype=dtype)

        # instantiate warper warp from source to destination
        warper = kornia.geometry.transform.HomographyWarper(height // 2, width // 2, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)

        # check the corners
        assert_close(patch_src[..., 0, 0], patch_dst[..., 0, 0], atol=1e-4, rtol=1e-4)
        assert_close(patch_src[..., 0, -1], patch_dst[..., 0, -1], atol=1e-4, rtol=1e-4)
        assert_close(patch_src[..., -1, 0], patch_dst[..., -1, 0], atol=1e-4, rtol=1e-4)
        assert_close(patch_src[..., -1, -1], patch_dst[..., -1, -1], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("shape", [(4, 5), (2, 6), (4, 3), (5, 7)])
    def test_translation(self, shape, device, dtype):
        # create input data
        offset = 2.0  # in pixel
        height, width = shape
        patch_src = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 2] = offset / (width - 1)  # apply offset in x

        # instantiate warper and from source to destination
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)
        assert_close(patch_src[..., 1:], patch_dst[..., :-1], atol=1e-4, rtol=1e-4)

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
        assert_close(patch_src[..., 0, 0], patch_dst[..., 0, -1], atol=1e-4, rtol=1e-4)
        assert_close(patch_src[..., 0, -1], patch_dst[..., -1, -1], atol=1e-4, rtol=1e-4)
        assert_close(patch_src[..., -1, 0], patch_dst[..., 0, 0], atol=1e-4, rtol=1e-4)
        assert_close(patch_src[..., -1, -1], patch_dst[..., -1, 0], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_homography_warper(self, batch_size, device, dtype):
        # generate input data
        height, width = 128, 64
        eye_size = 3  # identity 3x3

        patch_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

        # create base homography
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size, device=device, dtype=dtype)

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
            assert_close(patch_dst_to_src_precomputed, patch_dst_to_src, atol=1e-4, rtol=1e-4)

            # projected should be equal as initial
            error = utils.compute_patch_error(patch_src, patch_dst_to_src, height, width)

            assert error.item() < self.threshold

            # check functional api
            patch_dst_to_src_functional = kornia.geometry.transform.homography_warp(
                patch_dst, _torch_inverse_cast(dst_homo_src_i), (height, width), align_corners=True
            )

            assert_close(patch_dst_to_src, patch_dst_to_src_functional, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(1, 1, 7, 5), (2, 3, 8, 5), (1, 1, 7, 16)])
    def test_gradcheck(self, batch_shape, device, dtype):
        # generate input data
        eye_size = 3  # identity 3x3

        # create checkerboard
        patch_src = torch.rand(batch_shape, device=device, dtype=dtype)
        patch_src = utils.tensor_to_gradcheck_var(patch_src)  # to var

        # create base homography
        batch_size, _, height, width = patch_src.shape
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size, device=device, dtype=dtype)
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src, requires_grad=False)  # to var

        # instantiate warper
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)

        # evaluate function gradient
        assert gradcheck(warper, (patch_src, dst_homo_src), raise_exception=True)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("normalized_coordinates", [True, False])
    def test_jit_warp_homography(self, batch_size, align_corners, normalized_coordinates, device, dtype):
        # generate input data
        height, width = 128, 64
        eye_size = 3  # identity 3x3

        patch_src = torch.rand(batch_size, 1, height, width, device=device, dtype=dtype)

        # create base homography
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size, device=device, dtype=dtype)

        for _ in range(self.num_tests):
            # generate homography noise
            homo_delta = torch.rand_like(dst_homo_src) * 0.3

            dst_homo_src_i = dst_homo_src + homo_delta

            # transform the points with and without jit
            patch_dst = kornia.geometry.transform.homography_warp(
                patch_src,
                dst_homo_src_i,
                (height, width),
                align_corners=align_corners,
                normalized_coordinates=normalized_coordinates,
            )
            patch_dst_jit = torch.jit.script(kornia.geometry.transform.homography_warp)(
                patch_src,
                dst_homo_src_i,
                (height, width),
                align_corners=align_corners,
                normalized_coordinates=normalized_coordinates,
            )

            assert_close(patch_dst, patch_dst_jit, atol=1e-4, rtol=1e-4)


class TestHomographyNormalTransform:
    expected_2d_0 = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]])

    expected_2d_1 = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2e14, -1.0], [0.0, 0.0, 1.0]]])

    expected_3d_0 = expected = torch.tensor(
        [[[0.4, 0.0, 0.0, -1.0], [0.0, 2.0, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    expected_3d_1 = torch.tensor(
        [[[0.4, 0.0, 0.0, -1.0], [0.0, 2e14, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    @pytest.mark.parametrize("height,width,expected", [(2, 5, expected_2d_0), (1, 5, expected_2d_1)])
    def test_transform2d(self, height, width, expected, device, dtype):
        output = kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype)

        assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

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
        assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("height,width,depth,expected", [(2, 6, 4, expected_3d_0), (1, 6, 4, expected_3d_1)])
    def test_transform3d(self, height, width, depth, expected, device, dtype):
        output = kornia.geometry.conversions.normal_transform_pixel3d(depth, height, width, device=device, dtype=dtype)
        assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

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
        assert_close(output, expected.to(device=device, dtype=dtype), atol=1e-4, rtol=1e-4)


class TestHomographyWarper3D:
    num_tests = 10
    threshold = 0.1

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_normalize_homography_identity(self, batch_size, device, dtype):
        # create input data
        input_shape = (4, 8, 5)
        dst_homo_src = utils.create_eye_batch(batch_size=batch_size, eye_size=4).to(device=device, dtype=dtype)

        res = torch.tensor(
            [[[0.5000, 0.0, 0.0, -1.0], [0.0, 0.2857, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        norm = kornia.geometry.conversions.normal_transform_pixel3d(input_shape[0], input_shape[1], input_shape[2]).to(
            device=device, dtype=dtype
        )
        assert_close(norm, res, rtol=1e-4, atol=1e-4)

        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, input_shape, input_shape).to(
            device=device, dtype=dtype
        )
        assert_close(norm_homo, dst_homo_src, rtol=1e-4, atol=1e-4)

        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, input_shape, input_shape).to(
            device=device, dtype=dtype
        )
        assert_close(norm_homo, dst_homo_src, rtol=1e-4, atol=1e-4)

        # change output scale
        norm_homo = kornia.geometry.conversions.normalize_homography3d(
            dst_homo_src, input_shape, (input_shape[0] // 2, input_shape[1] * 2, input_shape[2] // 2)
        ).to(device=device, dtype=dtype)
        res = torch.tensor(
            [[[4.0, 0.0, 0.0, 3.0], [0.0, 0.4667, 0.0, -0.5333], [0.0, 0.0, 3.0, 2.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1)
        assert_close(norm_homo, res, rtol=1e-4, atol=1e-4)

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
        )
        assert (norm_homo == res).all()

    @pytest.mark.parametrize("offset", [1, 3, 7])
    @pytest.mark.parametrize("shape", [(4, 5, 6), (2, 4, 6), (4, 3, 9), (5, 7, 8)])
    def test_warp_grid_translation(self, shape, offset, device, dtype):
        # create input data
        depth, height, width = shape
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=4, device=device, dtype=dtype)
        dst_homo_src[..., 0, 3] = offset  # apply offset in x
        grid = kornia.utils.create_meshgrid3d(depth, height, width, normalized_coordinates=False)
        flow = kornia.geometry.transform.warp_grid3d(grid, dst_homo_src)

        # the grid the src plus the offset should be equal to the flow
        # on the x-axis, y-axis remains the same.
        assert_close(grid[..., 0].to(device=device, dtype=dtype) + offset, flow[..., 0], atol=1e-4, rtol=1e-4)
        assert_close(grid[..., 1].to(device=device, dtype=dtype), flow[..., 1], atol=1e-4, rtol=1e-4)
        assert_close(grid[..., 2].to(device=device, dtype=dtype), flow[..., 2], atol=1e-4, rtol=1e-4)
