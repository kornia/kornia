import pytest

import kornia as kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


@pytest.mark.parametrize("batch_shape", [(1, 1, 7, 32), (2, 3, 16, 31)])
def test_warp_perspective_rotation(batch_shape, device_type):
    # generate input data
    batch_size, channels, height, width = batch_shape
    alpha = 0.5 * kornia.pi * torch.ones(batch_size)  # 90 deg rotation

    # create data patch
    device = torch.device(device_type)
    patch = torch.rand(batch_shape).to(device)

    # create transformation (rotation)
    M = torch.eye(3, device=device).repeat(batch_size, 1, 1)  # Bx3x3
    M[:, 0, 0] = torch.cos(alpha)
    M[:, 0, 1] = -torch.sin(alpha)
    M[:, 1, 0] = torch.sin(alpha)
    M[:, 1, 1] = torch.cos(alpha)

    # apply transformation and inverse
    _, _, h, w = patch.shape
    patch_warped = kornia.warp_perspective(patch, M, dsize=(height, width))
    patch_warped_inv = kornia.warp_perspective(
        patch_warped, torch.inverse(M), dsize=(height, width))

    # generate mask to compute error
    mask = torch.ones_like(patch)
    mask_warped_inv = kornia.warp_perspective(
        kornia.warp_perspective(patch, M, dsize=(height, width)),
        torch.inverse(M),
        dsize=(height, width))

    assert_allclose(mask_warped_inv * patch,
                                   mask_warped_inv * patch_warped_inv)

    # evaluate function gradient
    patch = utils.tensor_to_gradcheck_var(patch)  # to var
    M = utils.tensor_to_gradcheck_var(M, requires_grad=False)  # to var
    assert gradcheck(
        kornia.warp_perspective, (patch, M, (
            height,
            width,
        )),
        raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_get_perspective_transform(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)

    h_max, w_max = 64, 32  # height, width
    h = torch.ceil(h_max * torch.rand(batch_size)).to(device)
    w = torch.ceil(w_max * torch.rand(batch_size)).to(device)

    norm = torch.rand(batch_size, 4, 2).to(device)
    points_src = torch.zeros_like(norm)
    points_src[:, 1, 0] = h
    points_src[:, 2, 1] = w
    points_src[:, 3, 0] = h
    points_src[:, 3, 1] = w
    points_dst = points_src + norm

    # compute transform from source to target
    dst_homo_src = kornia.get_perspective_transform(points_src, points_dst)

    assert_allclose(
        kornia.transform_points(dst_homo_src, points_src), points_dst)

    # compute gradient check
    points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
    points_dst = utils.tensor_to_gradcheck_var(points_dst)  # to var
    assert gradcheck(
        kornia.get_perspective_transform, (
            points_src,
            points_dst,
        ),
        raise_exception=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rotation_matrix2d(batch_size, device_type):
    # generate input data
    device = torch.device(device_type)
    center_base = torch.zeros(batch_size, 2).to(device)
    angle_base = torch.ones(batch_size).to(device)
    scale_base = torch.ones(batch_size).to(device)

    # 90 deg rotation
    center = center_base
    angle = 90. * angle_base
    scale = scale_base
    M = kornia.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        pytest.approx(M[i, 0, 0].item(), 0.0)
        pytest.approx(M[i, 0, 1].item(), 1.0)
        pytest.approx(M[i, 1, 0].item(), -1.0)
        pytest.approx(M[i, 1, 1].item(), 0.0)

    # 90 deg rotation + 2x scale
    center = center_base
    angle = 90. * angle_base
    scale = 2. * scale_base
    M = kornia.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        pytest.approx(M[i, 0, 0].item(), 0.0)
        pytest.approx(M[i, 0, 1].item(), 2.0)
        pytest.approx(M[i, 1, 0].item(), -2.0)
        pytest.approx(M[i, 1, 1].item(), 0.0)

    # 45 deg rotation
    center = center_base
    angle = 45. * angle_base
    scale = scale_base
    M = kornia.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        pytest.approx(M[i, 0, 0].item(), 0.7071)
        pytest.approx(M[i, 0, 1].item(), 0.7071)
        pytest.approx(M[i, 1, 0].item(), -0.7071)
        pytest.approx(M[i, 1, 1].item(), 0.7071)

    # evaluate function gradient
    center = utils.tensor_to_gradcheck_var(center)  # to var
    angle = utils.tensor_to_gradcheck_var(angle)  # to var
    scale = utils.tensor_to_gradcheck_var(scale)  # to var
    assert gradcheck(
        kornia.get_rotation_matrix2d, (center, angle, scale),
        raise_exception=True)


class TestWarpPerspective:
    @pytest.mark.parametrize("batch_size", [1, 5])
    @pytest.mark.parametrize("channels", [1, 5])
    def test_crop(self, device_type, batch_size, channels):
        # generate input data
        src_h, src_w = 3, 3
        dst_h, dst_w = 3, 3
        device = torch.device(device_type)

        # [x, y] origin
        # top-left, top-right, bottom-right, bottom-left
        points_src = torch.FloatTensor([[
            [0, 0],
            [0, src_w - 1],
            [src_h - 1, src_w - 1],
            [src_h - 1, 0],
        ]])

        # [x, y] destination
        # top-left, top-right, bottom-right, bottom-left
        points_dst = torch.FloatTensor([[
            [0, 0],
            [0, dst_w - 1],
            [dst_h - 1, dst_w - 1],
            [dst_h - 1, 0],
        ]])

        # compute transformation between points
        dst_trans_src = kornia.get_perspective_transform(points_src,
                                                         points_dst).expand(
            batch_size, -1, -1)

        # warp tensor
        patch = torch.FloatTensor([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]]).expand(batch_size, channels, -1, -1)

        expected = torch.FloatTensor([[[
            [1, 2, 3],
            [5, 6, 7],
            [9, 10, 11],
        ]]])

        # warp and assert
        patch_warped = kornia.warp_perspective(patch, dst_trans_src,
                                               (dst_h, dst_w))
        assert_allclose(patch_warped, expected)

    def test_crop_center_resize(self, device_type):
        # generate input data
        dst_h, dst_w = 4, 4
        device = torch.device(device_type)

        # [x, y] origin
        # top-left, top-right, bottom-right, bottom-left
        points_src = torch.FloatTensor([[
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 1],
        ]])

        # [x, y] destination
        # top-left, top-right, bottom-right, bottom-left
        points_dst = torch.FloatTensor([[
            [0, 0],
            [0, dst_w - 1],
            [dst_h - 1, dst_w - 1],
            [dst_h - 1, 0],
        ]])

        # compute transformation between points
        dst_trans_src = kornia.get_perspective_transform(points_src, points_dst)

        # warp tensor
        patch = torch.FloatTensor([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]])

        expected = torch.FloatTensor([[[
            [6.000, 6.333, 6.666, 7.000],
            [7.333, 7.666, 8.000, 8.333],
            [8.666, 9.000, 9.333, 9.666],
            [10.000, 10.333, 10.666, 11.000],
        ]]])

        # warp and assert
        patch_warped = kornia.warp_perspective(patch, dst_trans_src,
                                               (dst_h, dst_w))
        assert_allclose(patch_warped, expected)


class TestWarpAffine:
    def test_smoke(self):
        batch_size, channels, height, width = 1, 2, 3, 4
        aff_ab = torch.eye(2, 3)[None]  # 1x2x3
        img_b = torch.rand(batch_size, channels, height, width)
        img_a = kornia.warp_affine(img_b, aff_ab, (height, width))
        assert img_b.shape == img_a.shape

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation(self, batch_size):
        offset = 1.
        channels, height, width = 1, 3, 4
        aff_ab = torch.eye(2, 3).repeat(batch_size, 1, 1)  # Bx2x3
        aff_ab[..., -1] += offset
        img_b = torch.arange(float(height * width)).view(
            1, channels, height, width).repeat(batch_size, 1, 1, 1)
        img_a = kornia.warp_affine(img_b, aff_ab, (height, width))
        assert_allclose(img_b[..., :2, :3], img_a[..., 1:, 1:])

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 3, 4
        aff_ab = torch.eye(2, 3)[None]  # 1x2x3
        img_b = torch.rand(batch_size, channels, height, width)
        aff_ab = utils.tensor_to_gradcheck_var(
            aff_ab, requires_grad=False)  # to var
        img_b = utils.tensor_to_gradcheck_var(img_b)  # to var
        assert gradcheck(
            kornia.warp_affine, (
                img_b,
                aff_ab,
                (height, width),
            ),
            raise_exception=True)


class TestRemap:
    def test_smoke(self):
        height, width = 3, 4
        input = torch.ones(1, 1, height, width)
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        input_warped = kornia.remap(input, grid[..., 0], grid[..., 1])
        assert_allclose(input, input_warped)

    def test_shift(self):
        height, width = 3, 4
        inp = torch.tensor([[[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
        ]]])
        expected = torch.tensor([[[
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [0., 0., 0., 0.],
        ]]])

        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid += 1.  # apply shift in both x/y direction

        input_warped = kornia.remap(inp, grid[..., 0], grid[..., 1])
        assert_allclose(input_warped, expected)

    def test_shift_batch(self):
        height, width = 3, 4
        inp = torch.tensor([[[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
        ]]]).repeat(2, 1, 1, 1)

        expected = torch.tensor([[[
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
        ]], [[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [0., 0., 0., 0.],
        ]]])

        # generate a batch of grids
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid = grid.repeat(2, 1, 1, 1)
        grid[0, ..., 0] += 1.  # apply shift in the x direction
        grid[1, ..., 1] += 1.  # apply shift in the y direction

        input_warped = kornia.remap(inp, grid[..., 0], grid[..., 1])
        assert_allclose(input_warped, expected)

    def test_shift_batch_broadcast(self):
        height, width = 3, 4
        inp = torch.tensor([[[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
        ]]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[[
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [0., 0., 0., 0.],
        ]]])

        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid += 1.  # apply shift in both x/y direction

        input_warped = kornia.remap(inp, grid[..., 0], grid[..., 1])
        assert_allclose(input_warped, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 3, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid = utils.tensor_to_gradcheck_var(
            grid, requires_grad=False)  # to var

        assert gradcheck(kornia.remap, (img, grid[..., 0], grid[..., 1],),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input, map1, map2):
            return kornia.remap(input, map1, map2)
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.ones(batch_size, channels, height, width)

        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid += 1.  # apply some shift

        input = (img, grid[..., 0], grid[..., 1],)
        actual = op_script(*input)
        expected = kornia.remap(*input)
        assert_allclose(actual, expected)

    def test_jit_trace(self):
        @torch.jit.script
        def op_script(input, map1, map2):
            return kornia.remap(input, map1, map2)
        # 1. Trace op
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.ones(batch_size, channels, height, width)
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid += 1.  # apply some shift
        input_tuple = (img, grid[..., 0], grid[..., 1])
        op_traced = torch.jit.trace(op_script, input_tuple)

        # 2. Generate different input
        batch_size, channels, height, width = 2, 2, 2, 5
        img = torch.ones(batch_size, channels, height, width)
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=False)
        grid += 2.  # apply some shift

        # 3. Apply to different input
        input_tuple = (img, grid[..., 0], grid[..., 1])
        actual = op_script(*input_tuple)
        expected = kornia.remap(*input_tuple)
        assert_allclose(actual, expected)


class TestInvertAffineTransform:
    def test_smoke(self):
        matrix = torch.eye(2, 3)
        matrix_inv = kornia.invert_affine_transform(matrix)
        assert_allclose(matrix, matrix_inv)

    def test_rot90(self):
        angle = torch.tensor([90.])
        scale = torch.tensor([1.])
        center = torch.tensor([[0., 0.]])
        expected = torch.tensor([[
            [0., -1., 0.],
            [1., 0., 0.],
        ]])
        matrix = kornia.get_rotation_matrix2d(center, angle, scale)
        matrix_inv = kornia.invert_affine_transform(matrix)
        assert_allclose(matrix_inv, expected)

    def test_rot90_batch(self):
        angle = torch.tensor([90.])
        scale = torch.tensor([1.])
        center = torch.tensor([[0., 0.]])
        expected = torch.tensor([[
            [0., -1., 0.],
            [1., 0., 0.],
        ]])
        matrix = kornia.get_rotation_matrix2d(
            center, angle, scale).repeat(2, 1, 1)
        matrix_inv = kornia.invert_affine_transform(matrix)
        assert_allclose(matrix_inv, expected)

    def test_gradcheck(self):
        matrix = torch.eye(2, 3)
        matrix = utils.tensor_to_gradcheck_var(matrix)  # to var
        assert gradcheck(kornia.invert_affine_transform, (matrix,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.invert_affine_transform(input)
        matrix = torch.eye(2, 3)
        op_traced = torch.jit.trace(op_script, matrix)
        actual = op_traced(matrix)
        expected = kornia.invert_affine_transform(matrix)
        assert_allclose(actual, expected)

    def test_jit_trace(self):
        @torch.jit.script
        def op_script(input):
            return kornia.invert_affine_transform(input)
        matrix = torch.eye(2, 3)
        matrix_2 = torch.eye(2, 3).repeat(2, 1, 1)
        op_traced = torch.jit.trace(op_script, matrix)
        actual = op_traced(matrix_2)
        expected = kornia.invert_affine_transform(matrix_2)
        assert_allclose(actual, expected)
