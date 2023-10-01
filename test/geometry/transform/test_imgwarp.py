import sys

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close
from kornia.utils._compat import torch_version, torch_version_lt
from kornia.utils.helpers import _torch_inverse_cast


class DummyNNModule(torch.nn.Module):
    def __init__(self, h: int, w: int, align_corners: bool, padding_mode: str):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x, y):
        return kornia.geometry.transform.warp_affine(x, y, dsize=(self.h, self.w), align_corners=False)


class TestGetPerspectiveTransform:
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_smoke(self, device, dtype, batch_size):
        points_src = torch.rand(batch_size, 4, 2, device=device, dtype=dtype)
        points_dst = torch.rand(batch_size, 4, 2, device=device, dtype=dtype)

        dst_trans_src = kornia.geometry.get_perspective_transform(points_src, points_dst)

        assert dst_trans_src.shape == (batch_size, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 5])
    def test_crop_src_dst_type_mismatch(self, device, dtype, batch_size):
        # generate input data
        src_h, src_w = 3, 3
        dst_h, dst_w = 3, 3

        # [x, y] origin
        # top-left, top-right, bottom-right, bottom-left
        points_src = torch.tensor(
            [[[0, 0], [0, src_w - 1], [src_h - 1, src_w - 1], [src_h - 1, 0]]], device=device, dtype=torch.int64
        )

        # [x, y] destination
        # top-left, top-right, bottom-right, bottom-left
        points_dst = torch.tensor(
            [[[0, 0], [0, dst_w - 1], [dst_h - 1, dst_w - 1], [dst_h - 1, 0]]], device=device, dtype=dtype
        )

        # compute transformation between points
        with pytest.raises(Exception):
            _ = kornia.geometry.get_perspective_transform(points_src, points_dst)

    def test_back_and_forth(self, device, dtype):
        # generate input data
        h_max, w_max = 64, 32  # height, width
        h = h_max * torch.rand(1, device=device, dtype=dtype)
        w = w_max * torch.rand(1, device=device, dtype=dtype)

        norm = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points_src = torch.zeros_like(norm, device=device, dtype=dtype)
        points_src[:, 1, 0] = h
        points_src[:, 2, 1] = w
        points_src[:, 3, 0] = h
        points_src[:, 3, 1] = w
        points_dst = points_src + norm

        # compute transform from source to target
        dst_trans_src = kornia.geometry.get_perspective_transform(points_src, points_dst)
        points_dst_hat = kornia.geometry.transform_points(dst_trans_src, points_src)
        assert_close(points_dst, points_dst_hat)

    def test_hflip(self, device, dtype):
        points_src = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)

        points_dst = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], device=device, dtype=dtype)

        dst_trans_src = kornia.geometry.get_perspective_transform(points_src, points_dst)

        point_left = torch.tensor([[[0.0, 0.0]]], device=device, dtype=dtype)
        point_right = torch.tensor([[[1.0, 0.0]]], device=device, dtype=dtype)

        assert_close(kornia.geometry.transform_points(dst_trans_src, point_left), point_right)

    def test_dynamo(self, device, dtype, torch_optimizer):
        points_src = torch.rand(1, 4, 2, device=device, dtype=dtype)
        points_dst = torch.rand(1, 4, 2, device=device, dtype=dtype)

        op = kornia.geometry.get_perspective_transform
        op_optimized = torch_optimizer(op)

        assert_close(op(points_src, points_dst), op_optimized(points_src, points_dst))

    @pytest.mark.skipif(torch_version_lt(1, 11, 0), reason="backward for LSTSQ not supported in pytorch < 1.11.0")
    def test_gradcheck(self, device):
        # compute gradient check
        points_src = torch.rand(1, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        points_dst = torch.rand(1, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            kornia.geometry.get_perspective_transform, (points_src, points_dst), raise_exception=True, fast_mode=True
        )


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rotation_matrix2d(batch_size, device, dtype):
    # generate input data
    center_base = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    angle_base = torch.ones(batch_size, device=device, dtype=dtype)
    scale_base = torch.ones(batch_size, 2, device=device, dtype=dtype)

    # 90 deg rotation
    center = center_base
    angle = 90.0 * angle_base
    scale = scale_base
    M = kornia.geometry.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        assert_close(M[i, 0, 0].item(), 0.0, rtol=1e-4, atol=1e-4)
        assert_close(M[i, 0, 1].item(), 1.0, rtol=1e-4, atol=1e-4)
        assert_close(M[i, 1, 0].item(), -1.0, rtol=1e-4, atol=1e-4)
        assert_close(M[i, 1, 1].item(), 0.0, rtol=1e-4, atol=1e-4)

    # 90 deg rotation + 2x scale
    center = center_base
    angle = 90.0 * angle_base
    scale = 2.0 * scale_base
    M = kornia.geometry.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        assert_close(M[i, 0, 0].item(), 0.0, rtol=1e-4, atol=1e-4)
        assert_close(M[i, 0, 1].item(), 2.0, rtol=1e-4, atol=1e-4)
        assert_close(M[i, 1, 0].item(), -2.0, rtol=1e-4, atol=1e-4)
        assert_close(M[i, 1, 1].item(), 0.0, rtol=1e-4, atol=1e-4)

    # 45 deg rotation
    center = center_base
    angle = 45.0 * angle_base
    scale = scale_base
    M = kornia.geometry.get_rotation_matrix2d(center, angle, scale)

    for i in range(batch_size):
        assert_close(M[i, 0, 0].item(), 0.7071)
        assert_close(M[i, 0, 1].item(), 0.7071)
        assert_close(M[i, 1, 0].item(), -0.7071)
        assert_close(M[i, 1, 1].item(), 0.7071)

    # evaluate function gradient
    center = utils.tensor_to_gradcheck_var(center)  # to var
    angle = utils.tensor_to_gradcheck_var(angle)  # to var
    scale = utils.tensor_to_gradcheck_var(scale)  # to var
    assert gradcheck(
        kornia.geometry.get_rotation_matrix2d, (center, angle, scale), raise_exception=True, fast_mode=True
    )


class TestWarpAffine:
    def test_smoke(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 4
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype)[None]  # 1x2x3
        img_b = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img_a = kornia.geometry.warp_affine(img_b, aff_ab, (height, width))
        assert_close(img_b, img_a)

    @pytest.mark.parametrize("batch_shape", ([1, 3, 2, 5], [2, 4, 3, 4], [3, 5, 6, 2]))
    @pytest.mark.parametrize("out_shape", ([2, 5], [3, 4], [6, 2]))
    def test_cardinality(self, device, dtype, batch_shape, out_shape):
        batch_size, channels, height, width = batch_shape
        h_out, w_out = out_shape
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # Bx2x3
        img_b = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img_a = kornia.geometry.warp_affine(img_b, aff_ab, (h_out, w_out))
        assert img_a.shape == (batch_size, channels, h_out, w_out)

    def test_exception(self, device, dtype):
        img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        aff = torch.eye(2, 3, device=device, dtype=dtype)[None]
        size = (4, 5)

        with pytest.raises(TypeError):
            assert kornia.geometry.warp_affine(0.0, aff, size)

        with pytest.raises(TypeError):
            assert kornia.geometry.warp_affine(img, 0.0, size)

        with pytest.raises(ValueError):
            img = torch.rand(2, 3, 4, device=device, dtype=dtype)
            assert kornia.geometry.warp_affine(img, aff, size)

        with pytest.raises(ValueError):
            aff = torch.eye(2, 2, device=device, dtype=dtype)[None]
            assert kornia.geometry.warp_affine(img, aff, size)

    def test_translation(self, device, dtype):
        offset = 1.0
        h, w = 3, 4
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype)[None]
        aff_ab[..., -1] += offset

        img_b = torch.arange(float(h * w), device=device, dtype=dtype).view(1, 1, h, w)

        expected = torch.zeros_like(img_b)
        expected[..., 1:, 1:] = img_b[..., :2, :3]

        # Same as opencv: cv2.warpAffine(kornia.tensor_to_image(img_b), aff_ab[0].numpy(), (w, h))
        img_a = kornia.geometry.warp_affine(img_b, aff_ab, (h, w))
        assert_close(img_a, expected)

    def test_rotation_inverse(self, device, dtype):
        h, w = 4, 4
        img_b = torch.rand(1, 1, h, w, device=device, dtype=dtype)

        # create rotation matrix of 90deg (anti-clockwise)
        center = torch.tensor([[w - 1, h - 1]], device=device, dtype=dtype) / 2
        scale = torch.ones((1, 2), device=device, dtype=dtype)
        angle = 90.0 * torch.ones(1, device=device, dtype=dtype)
        aff_ab_2x3 = kornia.geometry.get_rotation_matrix2d(center, angle, scale)
        # Same as opencv: cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2), 90., 1.)

        # warp the tensor
        # Same as opencv: cv2.warpAffine(kornia.tensor_to_image(img_b), aff_ab[0].numpy(), (w, h))
        img_a = kornia.geometry.warp_affine(img_b, aff_ab_2x3, (h, w))

        # invert the transform
        aff_ab_3x3 = kornia.geometry.conversions.convert_affinematrix_to_homography(aff_ab_2x3)
        aff_ba_2x3 = _torch_inverse_cast(aff_ab_3x3)[..., :2, :]
        img_b_hat = kornia.geometry.warp_affine(img_a, aff_ba_2x3, (h, w))
        assert_close(img_b_hat, img_b, atol=1e-3, rtol=1e-3)

    def test_dynamo(self, device, dtype, torch_optimizer):
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype)[None]
        img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        args = (img, aff_ab, (4, 5))
        op = kornia.geometry.warp_affine
        op_optimized = torch_optimizer(op)
        assert_close(op(*args), op_optimized(*args))

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 4
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype)[None] + 1e-6  # 1x2x3
        img_b = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        aff_ab = utils.tensor_to_gradcheck_var(aff_ab)  # to var
        img_b = utils.tensor_to_gradcheck_var(img_b)  # to var
        assert gradcheck(
            kornia.geometry.warp_affine, (img_b, aff_ab, (height, width)), raise_exception=True, fast_mode=True
        )

    def test_fill_padding_translation(self, device, dtype):
        offset = 1.0
        h, w = 3, 4
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype)[None]
        aff_ab[..., -1] += offset

        img_b = torch.arange(float(3 * h * w), device=device, dtype=dtype).view(1, 3, h, w)

        # normally fill_value will also be converted to the right device and type in warp_affine
        fill_value = torch.tensor([0.5, 0.2, 0.1], device=device, dtype=dtype)

        img_a = kornia.geometry.warp_affine(img_b, aff_ab, (h, w), padding_mode="fill", fill_value=fill_value)
        top_row_mean = img_a[..., :1, :].mean(dim=[0, 2, 3])
        first_col_mean = img_a[..., :1].mean(dim=[0, 2, 3])
        assert_close(top_row_mean, fill_value)
        assert_close(first_col_mean, fill_value)

    @pytest.mark.parametrize("num_channels", [1, 3, 5])
    def test_fill_padding_channels(self, device, dtype, num_channels):
        offset = 1.0
        h, w = 3, 4
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype)[None]
        aff_ab[..., -1] += offset

        img_b = torch.arange(float(num_channels * h * w), device=device, dtype=dtype).view(1, num_channels, h, w)

        fill_value = torch.zeros(num_channels, device=device, dtype=dtype)

        img_a = kornia.geometry.warp_affine(img_b, aff_ab, (h, w), padding_mode="fill", fill_value=fill_value)

        assert_close(img_a[:, :, :1, :1].squeeze(), fill_value.squeeze())

    @pytest.mark.parametrize("align_corners", (True, False))
    @pytest.mark.parametrize("padding_mode", ("zeros", "fill"))
    def test_jit_script(self, align_corners, padding_mode):
        net = DummyNNModule(3, 4, align_corners, padding_mode)
        net = torch.jit.script(net)
        # Assert compilation doesn't fail

    @pytest.mark.parametrize("align_corners", (True, False))
    @pytest.mark.parametrize("padding_mode", ("zeros", "fill"))
    @pytest.mark.xfail(reason="aten::linalg_inv is not yet supported in ONNX opset version 14.")
    def test_onnx_export(self, device, dtype, align_corners, padding_mode):
        offset = 1.0
        h, w = 3, 4
        aff_ab = torch.eye(2, 3, device=device, dtype=dtype)[None]
        aff_ab[..., -1] += offset

        img_b = torch.arange(float(3 * h * w), device=device, dtype=dtype).view(1, 3, h, w)

        net = DummyNNModule(h, w, align_corners, padding_mode).to(device)
        torch.onnx.export(net, (img_b, aff_ab), "temp.onnx", export_params=True)


class TestWarpPerspective:
    def test_smoke(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 4
        img_b = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        H_ab = kornia.eye_like(3, img_b)
        img_a = kornia.geometry.warp_perspective(img_b, H_ab, (height, width))
        assert_close(img_b, img_a)

    @pytest.mark.parametrize("batch_shape", ([1, 3, 2, 5], [2, 4, 3, 4], [3, 5, 6, 2]))
    @pytest.mark.parametrize("out_shape", ([2, 5], [3, 4], [6, 2]))
    def test_cardinality(self, device, dtype, batch_shape, out_shape):
        batch_size, channels, height, width = batch_shape
        h_out, w_out = out_shape
        img_b = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        H_ab = kornia.eye_like(3, img_b)
        img_a = kornia.geometry.warp_perspective(img_b, H_ab, (h_out, w_out))
        assert img_a.shape == (batch_size, channels, h_out, w_out)

    def test_exception(self, device, dtype):
        img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        homo = torch.eye(3, device=device, dtype=dtype)[None]
        size = (4, 5)

        with pytest.raises(TypeError):
            assert kornia.geometry.warp_perspective(0.0, homo, size)

        with pytest.raises(TypeError):
            assert kornia.geometry.warp_perspective(img, 0.0, size)

        with pytest.raises(ValueError):
            img = torch.rand(2, 3, 4, device=device, dtype=dtype)
            assert kornia.geometry.warp_perspective(img, homo, size)

        with pytest.raises(ValueError):
            homo = torch.eye(2, 2, device=device, dtype=dtype)[None]
            assert kornia.geometry.warp_perspective(img, homo, size)

    def test_translation(self, device, dtype):
        offset = 1.0
        h, w = 3, 4

        img_b = torch.arange(float(h * w), device=device, dtype=dtype).view(1, 1, h, w)
        homo_ab = kornia.eye_like(3, img_b)
        homo_ab[..., :2, -1] += offset

        expected = torch.zeros_like(img_b)
        expected[..., 1:, 1:] = img_b[..., :2, :3]

        # Same as opencv: cv2.warpPerspective(kornia.tensor_to_image(img_b), homo_ab[0].numpy(), (w, h))
        img_a = kornia.geometry.warp_perspective(img_b, homo_ab, (h, w))
        assert_close(img_a, expected, atol=1e-4, rtol=1e-4)

    def test_translation_normalized(self, device, dtype):
        offset = 1.0
        h, w = 3, 4

        img_b = torch.arange(float(h * w), device=device, dtype=dtype).view(1, 1, h, w)
        homo_ab = kornia.eye_like(3, img_b)
        homo_ab[..., :2, -1] += offset

        expected = torch.zeros_like(img_b)
        expected[..., 1:, 1:] = img_b[..., :2, :3]

        # Same as opencv: cv2.warpPerspective(kornia.tensor_to_image(img_b), homo_ab[0].numpy(), (w, h))
        img_a = kornia.geometry.transform.homography_warp(img_b, homo_ab, (h, w), normalized_homography=False)
        assert_close(img_a, expected, atol=1e-4, rtol=1e-4)

    def test_rotation_inverse(self, device, dtype):
        h, w = 4, 4
        img_b = torch.rand(1, 1, h, w, device=device, dtype=dtype)

        # create rotation matrix of 90deg (anti-clockwise)
        center = torch.tensor([[w - 1, h - 1]], device=device, dtype=dtype) / 2
        scale = torch.ones((1, 2), device=device, dtype=dtype)
        angle = 90.0 * torch.ones(1, device=device, dtype=dtype)
        aff_ab = kornia.geometry.get_rotation_matrix2d(center, angle, scale)
        # Same as opencv: cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2), 90., 1.)

        H_ab = kornia.geometry.convert_affinematrix_to_homography(aff_ab)  # Bx3x3

        # warp the tensor
        # Same as opencv: cv2.warpPerspecive(kornia.tensor_to_image(img_b), H_ab[0].numpy(), (w, h))
        img_a = kornia.geometry.warp_perspective(img_b, H_ab, (h, w))

        # invert the transform
        H_ba = _torch_inverse_cast(H_ab)
        img_b_hat = kornia.geometry.warp_perspective(img_a, H_ba, (h, w))
        assert_close(img_b_hat, img_b, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 5])
    @pytest.mark.parametrize("channels", [1, 5])
    def test_crop(self, batch_size, channels, device, dtype):
        # generate input data
        src_h, src_w = 3, 3
        dst_h, dst_w = 3, 3

        # [x, y] origin
        # top-left, top-right, bottom-right, bottom-left
        points_src = torch.tensor(
            [[[0, 0], [0, src_w - 1], [src_h - 1, src_w - 1], [src_h - 1, 0]]], device=device, dtype=dtype
        )

        # [x, y] destination
        # top-left, top-right, bottom-right, bottom-left
        points_dst = torch.tensor(
            [[[0, 0], [0, dst_w - 1], [dst_h - 1, dst_w - 1], [dst_h - 1, 0]]], device=device, dtype=dtype
        )

        # compute transformation between points
        dst_trans_src = kornia.geometry.get_perspective_transform(points_src, points_dst).expand(batch_size, -1, -1)

        # warp tensor
        patch = torch.tensor(
            [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], device=device, dtype=dtype
        ).expand(batch_size, channels, -1, -1)

        expected = patch[..., :3, :3]

        # warp and assert
        patch_warped = kornia.geometry.warp_perspective(patch, dst_trans_src, (dst_h, dst_w))
        assert_close(patch_warped, expected)

    def test_crop_center_resize(self, device, dtype):
        # generate input data
        dst_h, dst_w = 4, 4

        # [x, y] origin
        # top-left, top-right, bottom-right, bottom-left
        points_src = torch.tensor([[[1, 1], [1, 2], [2, 2], [2, 1]]], device=device, dtype=dtype)

        # [x, y] destination
        # top-left, top-right, bottom-right, bottom-left
        points_dst = torch.tensor(
            [[[0, 0], [0, dst_w - 1], [dst_h - 1, dst_w - 1], [dst_h - 1, 0]]], device=device, dtype=dtype
        )

        # compute transformation between points
        dst_trans_src = kornia.geometry.get_perspective_transform(points_src, points_dst)

        # warp tensor
        patch = torch.tensor(
            [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], device=device, dtype=dtype
        )

        expected = torch.tensor(
            [
                [
                    [
                        [6.0000, 6.3333, 6.6667, 7.0000],
                        [7.3333, 7.6667, 8.0000, 8.3333],
                        [8.6667, 9.0000, 9.3333, 9.6667],
                        [10.0000, 10.3333, 10.6667, 11.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # warp and assert
        patch_warped = kornia.geometry.warp_perspective(patch, dst_trans_src, (dst_h, dst_w))
        assert_close(patch_warped, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if dtype == torch.float64 and torch_version() in {'2.0.0', '2.0.1'} and sys.platform == 'linux':
            pytest.xfail('Failling on CI on ubuntu with torch 2.0.0 for float64')
        img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        H_ab = kornia.eye_like(3, img)
        args = (img, H_ab, (4, 5))
        op = kornia.geometry.warp_perspective
        op_optimized = torch_optimizer(op)
        assert_close(op(*args), op_optimized(*args))

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 4
        img_b = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        H_ab = kornia.eye_like(3, img_b)
        img_b = utils.tensor_to_gradcheck_var(img_b)  # to var
        # TODO(dmytro/edgar): firgure out why gradient don't propagate for the tranaform
        H_ab = utils.tensor_to_gradcheck_var(H_ab, requires_grad=False)  # to var
        assert gradcheck(
            kornia.geometry.warp_perspective, (img_b, H_ab, (height, width)), raise_exception=True, fast_mode=True
        )

    def test_fill_padding_translation(self, device, dtype):
        offset = 1.0
        h, w = 3, 4

        img_b = torch.arange(float(3 * h * w), device=device, dtype=dtype).view(1, 3, h, w)
        homo_ab = kornia.eye_like(3, img_b)
        homo_ab[..., :2, -1] += offset

        # normally fill_value will also be converted to the right device and type in warp_perspective
        fill_value = torch.tensor([0.5, 0.2, 0.1], device=device, dtype=dtype)

        img_a = kornia.geometry.warp_perspective(img_b, homo_ab, (h, w), padding_mode="fill", fill_value=fill_value)
        top_row_mean = img_a[..., :1, :].mean(dim=[0, 2, 3])
        first_col_mean = img_a[..., :1].mean(dim=[0, 2, 3])
        assert_close(top_row_mean, fill_value)
        assert_close(first_col_mean, fill_value)


class TestRemap:
    def test_smoke(self, device, dtype):
        height, width = 3, 4
        input_org = torch.ones(1, 1, height, width, device=device, dtype=dtype)
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device, dtype=dtype)
        input_warped = kornia.geometry.remap(
            input_org, grid[..., 0], grid[..., 1], normalized_coordinates=False, align_corners=True
        )
        assert_close(input_org, input_warped, rtol=1e-4, atol=1e-4)

    def test_different_size(self, device, dtype):
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, device=device, dtype=dtype)

        img = torch.rand(1, 2, 6, 5, device=device, dtype=dtype)
        img_warped = kornia.geometry.remap(img, grid[..., 0], grid[..., 1])
        assert img_warped.shape == (1, 2, height, width)

    def test_shift(self, device, dtype):
        height, width = 3, 4
        inp = torch.tensor(
            [[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )
        expected = torch.tensor(
            [[[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]], device=device, dtype=dtype
        )

        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype)
        grid += 1.0  # apply shift in both x/y direction

        input_warped = kornia.geometry.remap(inp, grid[..., 0], grid[..., 1], align_corners=True)
        assert_close(input_warped, expected, rtol=1e-4, atol=1e-4)

    def test_shift_batch(self, device, dtype):
        height, width = 3, 4
        inp = torch.tensor(
            [[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        ).repeat(2, 1, 1, 1)

        expected = torch.tensor(
            [
                [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]],
                [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        # generate a batch of grids
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype)
        grid = grid.repeat(2, 1, 1, 1)
        grid[0, ..., 0] += 1.0  # apply shift in the x direction
        grid[1, ..., 1] += 1.0  # apply shift in the y direction

        input_warped = kornia.geometry.remap(inp, grid[..., 0], grid[..., 1], align_corners=True)
        assert_close(input_warped, expected, rtol=1e-4, atol=1e-4)

    def test_shift_batch_broadcast(self, device, dtype):
        height, width = 3, 4
        inp = torch.tensor(
            [[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [[[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]], device=device, dtype=dtype
        ).repeat(2, 1, 1, 1)

        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype)
        grid += 1.0  # apply shift in both x/y direction

        input_warped = kornia.geometry.remap(inp, grid[..., 0], grid[..., 1], align_corners=True)
        assert_close(input_warped, expected, rtol=1e-4, atol=1e-4)

    def test_normalized_coordinates(self, device, dtype):
        height, width = 3, 4
        normalized_coordinates = True
        inp = torch.tensor(
            [[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        ).repeat(2, 1, 1, 1)

        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=normalized_coordinates, device=device
        ).to(dtype)

        # Normalized input coordinates
        input_warped = kornia.geometry.remap(
            inp, grid[..., 0], grid[..., 1], align_corners=True, normalized_coordinates=normalized_coordinates
        )
        assert_close(input_warped, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype)
        grid = utils.tensor_to_gradcheck_var(grid, requires_grad=False)  # to var

        assert gradcheck(
            kornia.geometry.remap,
            (img, grid[..., 0], grid[..., 1], 'bilinear', 'zeros', True),
            raise_exception=True,
            fast_mode=True,
        )

    @pytest.mark.skip(reason='Not fully support dynamo')
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: add dynamo support to create_meshgrid
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)

        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device).to(dtype)
        grid += 1.0  # apply some shift

        op = kornia.geometry.remap
        op_script = torch_optimizer(op)

        inputs = (img, grid[..., 0], grid[..., 1], 'bilinear', 'zeros', True)
        actual = op_script(*inputs)
        expected = op(*inputs)
        assert_close(actual, expected, rtol=1e-4, atol=1e-4)


class TestInvertAffineTransform:
    def test_smoke(self, device, dtype):
        matrix = torch.eye(2, 3, device=device, dtype=dtype)[None]
        matrix_inv = kornia.geometry.invert_affine_transform(matrix)
        assert_close(matrix, matrix_inv, rtol=1e-4, atol=1e-4)

    def test_rot90(self, device, dtype):
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        scale = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        center = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]], device=device, dtype=dtype)
        matrix = kornia.geometry.get_rotation_matrix2d(center, angle, scale)
        matrix_inv = kornia.geometry.invert_affine_transform(matrix)
        assert_close(matrix_inv, expected, rtol=1e-4, atol=1e-4)

    def test_rot90_batch(self, device, dtype):
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        scale = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        center = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]], device=device, dtype=dtype).repeat(2, 1, 1)
        matrix = kornia.geometry.get_rotation_matrix2d(center, angle, scale).repeat(2, 1, 1)
        matrix_inv = kornia.geometry.invert_affine_transform(matrix)
        assert_close(matrix_inv, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        matrix = torch.eye(2, 3, device=device, dtype=dtype)[None]
        matrix = utils.tensor_to_gradcheck_var(matrix)  # to var
        assert gradcheck(kornia.geometry.invert_affine_transform, (matrix,), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        op = kornia.geometry.invert_affine_transform
        op_script = torch_optimizer(op)

        matrix = torch.eye(2, 3, device=device, dtype=dtype)[None]
        actual = op_script(matrix)
        expected = op(matrix)
        assert_close(actual, expected, rtol=1e-4, atol=1e-4)
