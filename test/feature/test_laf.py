import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.geometry.transform.imgwarp
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestAngleToRotationMatrix:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4).to(device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(inp)
        assert rotmat.shape == (1, 3, 4, 4, 2, 2)

    def test_angles(self, device):
        ang_deg = torch.tensor([0, 90.0], device=device)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0, 1.0], [-1.0, 0]]], device=device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(ang_deg)
        assert_close(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(
            kornia.geometry.transform.imgwarp.angle_to_rotation_matrix, (img,), raise_exception=True, fast_mode=True
        )

    @pytest.mark.jit
    @pytest.mark.skip("Problems with kornia.pi")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix
        model_jit = torch.jit.script(kornia.geometry.transform.imgwarp.angle_to_rotation_matrix)
        assert_close(model(patches), model_jit(patches))


class TestGetLAFScale:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        rotmat = kornia.feature.get_laf_scale(inp)
        assert rotmat.shape == (1, 3, 1, 1)

    def test_scale(self, device):
        inp = torch.tensor([[5.0, 1, 0], [1, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2]]]], device=device).float()
        rotmat = kornia.feature.get_laf_scale(inp)
        assert_close(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.get_laf_scale, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_scale
        model_jit = torch.jit.script(kornia.feature.get_laf_scale)
        assert_close(model(img), model_jit(img))


class TestGetLAFCenter:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        xy = kornia.feature.get_laf_center(inp)
        assert xy.shape == (1, 3, 2)

    def test_center(self, device):
        inp = torch.tensor([[5.0, 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[2, 3]]], device=device).float()
        xy = kornia.feature.get_laf_center(inp)
        assert_close(xy, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.get_laf_center, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_center
        model_jit = torch.jit.script(kornia.feature.get_laf_center)
        assert_close(model(img), model_jit(img))


class TestGetLAFOri:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        ori = kornia.feature.get_laf_orientation(inp)
        assert ori.shape == (1, 3, 1)

    def test_ori(self, device):
        inp = torch.tensor([[1, 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[45.0]]], device=device).float()
        angle = kornia.feature.get_laf_orientation(inp)
        assert_close(angle, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.get_laf_orientation, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_orientation
        model_jit = torch.jit.script(kornia.feature.get_laf_orientation)
        assert_close(model(img), model_jit(img))


class TestScaleLAF:
    def test_shape_float(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = 23.0
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_shape_tensor(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = torch.zeros(7, 1, 1, 1, device=device).float()
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_scale(self, device):
        inp = torch.tensor([[5.0, 1, 0.8], [1, 1, -4.0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        scale = torch.tensor([[[[2.0]]]], device=device).float()
        out = kornia.feature.scale_laf(inp, scale)
        expected = torch.tensor([[[[10.0, 2, 0.8], [2, 2, -4.0]]]], device=device).float()
        assert_close(out, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        scale = torch.rand(batch_size, device=device)
        scale = utils.tensor_to_gradcheck_var(scale)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.scale_laf, (laf, scale), raise_exception=True, atol=1e-4, fast_mode=True)

    @pytest.mark.jit
    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        scale = torch.rand(batch_size, device=device)
        model = kornia.feature.scale_laf
        model_jit = torch.jit.script(kornia.feature.scale_laf)
        assert_close(model(laf, scale), model_jit(laf, scale))


class TestSetLAFOri:
    def test_shape_tensor(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        ori = torch.ones(7, 3, 1, 1, device=device).float()
        assert kornia.feature.set_laf_orientation(inp, ori).shape == inp.shape

    def test_ori(self, device):
        inp = torch.tensor([[0.0, 5.0, 0.8], [-5.0, 0, -4.0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        ori = torch.zeros(1, 1, 1, 1, device=device).float()
        out = kornia.feature.set_laf_orientation(inp, ori)
        expected = torch.tensor([[[[5.0, 0.0, 0.8], [0.0, 5.0, -4.0]]]], device=device).float()
        assert_close(out, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        ori = torch.rand(batch_size, channels, 1, 1, device=device)
        ori = utils.tensor_to_gradcheck_var(ori)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(
            kornia.feature.set_laf_orientation, (laf, ori), raise_exception=True, atol=1e-4, fast_mode=True
        )

    @pytest.mark.jit
    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        ori = torch.rand(batch_size, channels, 1, 1, device=device)
        model = kornia.feature.set_laf_orientation
        model_jit = torch.jit.script(kornia.feature.set_laf_orientation)
        assert_close(model(laf, ori), model_jit(laf, ori))


class TestMakeUpright:
    def test_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        rotmat = kornia.feature.make_upright(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_do_nothing(self, device):
        inp = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[1, 0, 0], [0, 1, 0]]]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        assert_close(laf, expected)

    def test_do_nothing_with_scalea(self, device):
        inp = torch.tensor([[2, 0, 0], [0, 2, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2, 0, 0], [0, 2, 0]]]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        assert_close(laf, expected)

    def test_check_zeros(self, device):
        inp = torch.rand(4, 5, 2, 3, device=device)
        laf = kornia.feature.make_upright(inp)
        must_be_zeros = laf[:, :, 0, 1]
        assert_close(must_be_zeros, torch.zeros_like(must_be_zeros))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 14, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.make_upright, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.make_upright
        model_jit = torch.jit.script(kornia.feature.make_upright)
        assert_close(model(img), model_jit(img))


class TestELL2LAF:
    def test_shape(self, device):
        inp = torch.ones(5, 3, 5, device=device)
        inp[:, :, 3] = 0
        rotmat = kornia.feature.ellipse_to_laf(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_conversion(self, device):
        inp = torch.tensor([[10, -20, 0.01, 0, 0.01]], device=device).float()
        inp = inp.view(1, 1, 5)
        expected = torch.tensor([[10, 0, 10.0], [0, 10, -20]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        laf = kornia.feature.ellipse_to_laf(inp)
        assert_close(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height, device=device).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.0
        # assure it is positive definite
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.ellipse_to_laf, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height, device=device).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.0
        model = kornia.feature.ellipse_to_laf
        model_jit = torch.jit.script(kornia.feature.ellipse_to_laf)
        assert_close(model(img), model_jit(img))


class TestNormalizeLAF:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        assert inp.shape == kornia.feature.normalize_laf(inp, img).shape

    def test_conversion(self, device):
        w, h = 9, 5
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]]).float()
        laf = laf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w)
        expected = torch.tensor([[[[0.25, 0, 0.125], [0, 0.25, 0.25]]]]).float()
        lafn = kornia.feature.normalize_laf(laf, img)
        assert_close(lafn, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.normalize_laf, (laf, img), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        model = kornia.feature.normalize_laf
        model_jit = torch.jit.script(kornia.feature.normalize_laf)
        assert_close(model(laf, img), model_jit(laf, img))


class TestLAF2pts:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        n_pts = 13
        assert kornia.feature.laf_to_boundary_points(inp, n_pts).shape == (5, 3, n_pts, 2)

    def test_conversion(self, device):
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        laf = laf.view(1, 1, 2, 3)
        n_pts = 6
        expected = torch.tensor([[[[1, 1], [1, 2], [2, 1], [1, 0], [0, 1], [1, 2]]]], device=device).float()
        pts = kornia.feature.laf_to_boundary_points(laf, n_pts)
        assert_close(pts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.laf_to_boundary_points, (laf), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_to_boundary_points
        model_jit = torch.jit.script(kornia.feature.laf_to_boundary_points)
        assert_close(model(laf), model_jit(laf))


class TestDenormalizeLAF:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert inp.shape == kornia.feature.denormalize_laf(inp, img).shape

    def test_conversion(self, device):
        w, h = 9, 5
        expected = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w, device=device)
        lafn = torch.tensor([[0.25, 0, 0.125], [0, 0.25, 0.25]], device=device).float()
        laf = kornia.feature.denormalize_laf(lafn.view(1, 1, 2, 3), img)
        assert_close(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width, device=device)
        img = torch.rand(batch_size, 3, 10, 32, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.denormalize_laf, (laf, img), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        model = kornia.feature.denormalize_laf
        model_jit = torch.jit.script(kornia.feature.denormalize_laf)
        assert_close(model(laf, img), model_jit(laf, img))


class TestGenPatchGrid:
    def test_shape(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF

        grid = generate_patch_grid_from_normalized_LAF(img, laf, PS)
        assert grid.shape == (15, 3, 3, 2)

    def test_gradcheck(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF

        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(generate_patch_grid_from_normalized_LAF, (img, laf, PS), raise_exception=True, fast_mode=True)


class TestExtractPatchesSimple:
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_non_zero(self, device):
        img = torch.zeros(1, 1, 24, 24, device=device)
        img[:, :, 10:, 20:] = 1.0
        laf = torch.tensor([[8.0, 0, 14.0], [0, 8.0, 8.0]], device=device).reshape(1, 1, 2, 3)

        PS = 32
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.mean().item() > 0.01
        assert patches.shape == (1, 1, 1, PS, PS)

    def test_same_odd(self, device, dtype):
        img = torch.arange(5)[None].repeat(5, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[2.0, 0, 2.0], [0, 2.0, 2.0]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_simple(img, laf, 5, 1.0)
        assert_close(img, patch[0])

    def test_same_even(self, device, dtype):
        img = torch.arange(4)[None].repeat(4, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[1.5, 0, 1.5], [0, 1.5, 1.5]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_simple(img, laf, 4, 1.0)
        assert_close(img, patch[0])

    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device).float()
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device)
        PS = 11
        img = utils.tensor_to_gradcheck_var(img)  # to var
        nlaf = utils.tensor_to_gradcheck_var(nlaf)  # to var
        assert gradcheck(kornia.feature.extract_patches_simple, (img, nlaf, PS, False), raise_exception=True)


class TestExtractPatchesPyr:
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_non_zero(self, device):
        img = torch.zeros(1, 1, 24, 24, device=device)
        img[:, :, 10:, 20:] = 1.0
        laf = torch.tensor([[8.0, 0, 14.0], [0, 8.0, 8.0]], device=device).reshape(1, 1, 2, 3)

        PS = 32
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.mean().item() > 0.01
        assert patches.shape == (1, 1, 1, PS, PS)

    def test_same_odd(self, device, dtype):
        img = torch.arange(5)[None].repeat(5, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[2.0, 0, 2.0], [0, 2.0, 2.0]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_from_pyramid(img, laf, 5, 1.0)
        assert_close(img, patch[0])

    def test_same_even(self, device, dtype):
        img = torch.arange(4)[None].repeat(4, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[1.5, 0, 1.5], [0, 1.5, 1.5]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_from_pyramid(img, laf, 4, 1.0)
        assert_close(img, patch[0])

    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device).float()
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device)
        PS = 11
        img = utils.tensor_to_gradcheck_var(img)  # to var
        nlaf = utils.tensor_to_gradcheck_var(nlaf)  # to var
        assert gradcheck(
            kornia.feature.extract_patches_from_pyramid,
            (img, nlaf, PS, False),
            nondet_tol=1e-8,
            raise_exception=True,
            fast_mode=True,
        )


class TestLAFIsTouchingBoundary:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert (5, 3) == kornia.feature.laf_is_inside_image(inp, img).shape

    def test_touch(self, device):
        w, h = 10, 5
        img = torch.rand(1, 3, h, w, device=device)
        laf = torch.tensor([[[[10, 0, 3], [0, 10, 3]], [[1, 0, 5], [0, 1, 2]]]], device=device).float()
        expected = torch.tensor([[False, True]], device=device)
        assert torch.all(kornia.feature.laf_is_inside_image(laf, img) == expected).item()

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        w, h = 10, 5
        img = torch.rand(1, 3, h, w, device=device)
        laf = torch.tensor([[[[10, 0, 3], [0, 10, 3]], [[1, 0, 5], [0, 1, 2]]]], device=device).float()
        model = kornia.feature.laf_is_inside_image
        model_jit = torch.jit.script(kornia.feature.laf_is_inside_image)
        assert_close(model(laf, img), model_jit(laf, img))


class TestGetCreateLAF:
    def test_shape(self, device):
        xy = torch.ones(1, 3, 2, device=device)
        ori = torch.ones(1, 3, 1, device=device)
        scale = torch.ones(1, 3, 1, 1, device=device)
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        assert laf.shape == (1, 3, 2, 3)

    def test_laf(self, device):
        xy = torch.ones(1, 1, 2, device=device)
        ori = torch.zeros(1, 1, 1, device=device)
        scale = 5 * torch.ones(1, 1, 1, 1, device=device)
        expected = torch.tensor([[[[5, 0, 1], [0, 5, 1]]]], device=device).float()
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        assert_close(laf, expected)

    def test_laf_def(self, device):
        xy = torch.ones(1, 1, 2, device=device)
        expected = torch.tensor([[[[1, 0, 1], [0, 1, 1]]]], device=device).float()
        laf = kornia.feature.laf_from_center_scale_ori(xy)
        assert_close(laf, expected)

    def test_cross_consistency(self, device):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device)
        ori = torch.rand(batch_size, channels, 1, device=device)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device))
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        scale2 = kornia.feature.get_laf_scale(laf)
        assert_close(scale, scale2)
        xy2 = kornia.feature.get_laf_center(laf)
        assert_close(xy2, xy)
        ori2 = kornia.feature.get_laf_orientation(laf)
        assert_close(ori2, ori)

    def test_gradcheck(self, device):
        batch_size, channels = 3, 2
        xy = utils.tensor_to_gradcheck_var(torch.rand(batch_size, channels, 2, device=device))
        ori = utils.tensor_to_gradcheck_var(torch.rand(batch_size, channels, 1, device=device))
        scale = utils.tensor_to_gradcheck_var(torch.abs(torch.rand(batch_size, channels, 1, 1, device=device)))
        assert gradcheck(
            kornia.feature.laf_from_center_scale_ori, (xy, scale, ori), raise_exception=True, fast_mode=True
        )

    @pytest.mark.skip("Depends on angle-to-rotation-matric")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device)
        ori = torch.rand(batch_size, channels, 1, device=device)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device))
        model = kornia.feature.laf_from_center_scale_ori
        model_jit = torch.jit.script(kornia.feature.laf_from_center_scale_ori)
        assert_close(model(xy, scale, ori), model_jit(xy, scale, ori))


class TestGetLAF3pts:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        out = kornia.feature.laf_to_three_points(inp)
        assert out.shape == inp.shape

    def test_batch_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        out = kornia.feature.laf_to_three_points(inp)
        assert out.shape == inp.shape

    def test_conversion(self, device):
        inp = torch.tensor([[1, 0, 2], [0, 1, 3]], device=device).float().view(1, 1, 2, 3)
        expected = torch.tensor([[3, 2, 2], [3, 4, 3]], device=device).float().view(1, 1, 2, 3)
        threepts = kornia.feature.laf_to_three_points(inp)
        assert_close(threepts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(kornia.feature.laf_to_three_points, (inp,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_to_three_points
        model_jit = torch.jit.script(kornia.feature.laf_to_three_points)
        assert_close(model(inp), model_jit(inp))


class TestGetLAFFrom3pts:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        out = kornia.feature.laf_from_three_points(inp)
        assert out.shape == inp.shape

    def test_batch_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        out = kornia.feature.laf_from_three_points(inp)
        assert out.shape == inp.shape

    def test_conversion(self, device):
        expected = torch.tensor([[1, 0, 2], [0, 1, 3]], device=device).float().view(1, 1, 2, 3)
        inp = torch.tensor([[3, 2, 2], [3, 4, 3]], device=device).float().view(1, 1, 2, 3)
        threepts = kornia.feature.laf_from_three_points(inp)
        assert_close(threepts, expected)

    def test_cross_consistency(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp_2 = kornia.feature.laf_from_three_points(inp)
        inp_2 = kornia.feature.laf_to_three_points(inp_2)
        assert_close(inp_2, inp)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(kornia.feature.laf_from_three_points, (inp,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_from_three_points
        model_jit = torch.jit.script(kornia.feature.laf_from_three_points)
        assert_close(model(inp), model_jit(inp))


class TestTransformLAFs:
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    def test_transform_points(self, batch_size, num_points, device, dtype):
        # generate input data
        eye_size = 3
        lafs_src = torch.rand(batch_size, num_points, 2, 3, device=device, dtype=dtype)

        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device=device, dtype=dtype)

        # transform the points from dst to ref
        lafs_dst = kornia.feature.perspective_transform_lafs(dst_homo_src, lafs_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        lafs_dst_to_src = kornia.feature.perspective_transform_lafs(src_homo_dst, lafs_dst)

        # projected should be equal as initial
        assert_close(lafs_src, lafs_dst_to_src)

    def test_gradcheck(self, device, dtype):
        # generate input data
        batch_size, num_points = 2, 3
        eye_size = 3
        points_src = torch.rand(batch_size, num_points, 2, 3, device=device, dtype=dtype)
        dst_homo_src = utils.create_random_homography(batch_size, eye_size).to(device=device, dtype=dtype)
        # evaluate function gradient
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src)  # to var
        assert gradcheck(
            kornia.feature.perspective_transform_lafs, (dst_homo_src, points_src), raise_exception=True, fast_mode=True
        )
