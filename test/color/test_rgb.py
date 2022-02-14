import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import BaseTester, assert_close


class TestRgbToBgr(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_bgr(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_bgr(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_bgr([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_bgr(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_bgr(img)

        with pytest.raises(TypeError):
            assert kornia.color.bgr_to_rgb([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.bgr_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.bgr_to_rgb(img)

    def test_back_and_forth(self, device, dtype):
        data_bgr = torch.rand(1, 3, 3, 2, device=device, dtype=dtype)
        data_rgb = kornia.color.bgr_to_rgb(data_bgr)
        data_bgr_new = kornia.color.rgb_to_bgr(data_rgb)
        assert_close(data_bgr, data_bgr_new)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]], device=device, dtype=dtype
        )  # 3x2x2

        expected = torch.tensor(
            [[[3.0, 3.0], [3.0, 3.0]], [[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]], device=device, dtype=dtype
        )  # 3x2x2

        f = kornia.color.rgb_to_bgr
        assert_close(f(data), expected)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_bgr, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_bgr
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToBgr().to(device, dtype)
        fcn = kornia.color.rgb_to_bgr
        assert_close(ops(img), fcn(img))

    @pytest.mark.nn
    def test_module_bgr(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.BgrToRgb().to(device, dtype)
        fcn = kornia.color.bgr_to_rgb
        assert_close(ops(img), fcn(img))


class TestRgbToRgba(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_rgba(img, 0.0), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        out_shape = list(shape)
        out_shape[-3] += 1
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_rgba(img, 0.0).shape == tuple(out_shape)

    def test_exception(self, device, dtype):
        # rgb to rgba
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_rgba([0.0], 0.0)

        with pytest.raises(TypeError):
            img = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_rgba(img)

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_rgba(img, 0.0)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_rgba(img, 0.0)

        with pytest.raises(TypeError):
            img = torch.ones(3, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_rgba(img, 'alpha_str')

        # rgba to rgb
        with pytest.raises(TypeError):
            assert kornia.color.rgba_to_rgb(0.0)

        with pytest.raises(ValueError):
            img = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgba_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgba_to_rgb(img)

    def test_back_and_forth_rgb(self, device, dtype):
        a_val: float = 1.0
        x_rgb = torch.ones(3, 4, 4, device=device, dtype=dtype)
        x_rgba = kornia.color.rgb_to_rgba(x_rgb, a_val)
        x_rgb_new = kornia.color.rgba_to_rgb(x_rgba)
        assert_close(x_rgb, x_rgb_new)

    def test_back_and_forth_bgr(self, device, dtype):
        a_val: float = 1.0
        x_bgr = torch.ones(3, 4, 4, device=device, dtype=dtype)
        x_rgba = kornia.color.bgr_to_rgba(x_bgr, a_val)
        x_bgr_new = kornia.color.rgba_to_bgr(x_rgba)
        assert_close(x_bgr, x_bgr_new)

    @pytest.mark.parametrize("aval", [0.4, 45.0])
    def test_unit(self, device, dtype, aval):
        data = torch.tensor(
            [[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]], device=device, dtype=dtype
        )  # Bx3x2x2

        expected = torch.tensor(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0]],
                    [[3.0, 3.0], [3.0, 3.0]],
                    [[aval, aval], [aval, aval]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # Bx4x2x2

        assert_close(kornia.color.rgb_to_rgba(data, aval), expected)

    @pytest.mark.parametrize("aval", [0.4, 45.0])
    def test_unit_aval_th(self, device, dtype, aval):
        data = torch.tensor(
            [[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]], device=device, dtype=dtype
        )  # Bx3x2x2

        expected = torch.tensor(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0]],
                    [[3.0, 3.0], [3.0, 3.0]],
                    [[aval, aval], [aval, aval]],
                ]
            ],
            device=device,
            dtype=dtype,
        )  # Bx4x2x2

        aval = torch.full_like(data[:, :1], aval)  # Bx1xHxW
        assert_close(kornia.color.rgb_to_rgba(data, aval), expected)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_rgba, (img, 1.0), raise_exception=True)

    @pytest.mark.grad
    def test_gradcheck_th(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        aval = torch.ones(B, 1, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_rgba, (img, aval), raise_exception=True)

    @pytest.mark.skip(reason="unsupported Union type")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_rgba
        op_jit = torch.jit.script(op)
        assert_close(op(img, 1.0), op_jit(img, 1.0))
        aval = torch.ones(B, 1, H, W, device=device, dtype=dtype)
        assert_close(op(img, aval), op_jit(img, aval))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToRgba(1.0).to(device, dtype)
        fcn = kornia.color.rgb_to_rgba
        assert_close(ops(img), fcn(img, 1.0))

    @pytest.mark.nn
    def test_module_bgr(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.BgrToRgba(1.0).to(device, dtype)
        fcn = kornia.color.bgr_to_rgba
        assert_close(ops(img), fcn(img, 1.0))

    @pytest.mark.nn
    def test_module_bgra2rgb(self, device, dtype):
        B, C, H, W = 2, 4, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbaToRgb().to(device, dtype)
        fcn = kornia.color.rgba_to_rgb
        assert_close(ops(img), fcn(img))

    @pytest.mark.nn
    def test_module_bgra2bgr(self, device, dtype):
        B, C, H, W = 2, 4, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbaToBgr().to(device, dtype)
        fcn = kornia.color.rgba_to_bgr
        assert_close(ops(img), fcn(img))


class TestLinearRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_linear_rgb(img), torch.Tensor)
        assert isinstance(kornia.color.linear_rgb_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_linear_rgb(img).shape == shape
        assert kornia.color.linear_rgb_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_linear_rgb([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_linear_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_linear_rgb(img)

        with pytest.raises(TypeError):
            assert kornia.color.linear_rgb_to_rgb([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.linear_rgb_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.linear_rgb_to_rgb(img)

    def test_back_and_forth(self, device, dtype):
        data_bgr = torch.rand(1, 3, 3, 2, device=device, dtype=dtype)
        data_rgb = kornia.color.rgb_to_linear_rgb(data_bgr)
        data_bgr_new = kornia.color.linear_rgb_to_rgb(data_rgb)
        assert_close(data_bgr, data_bgr_new)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 0.0], [0.5, 0.1]], [[1.0, 0.0], [0.5, 0.2]], [[1.0, 0.0], [0.5, 0.3]]], device=device, dtype=dtype
        )  # 3x2x2

        expected = torch.tensor(
            [
                [[1.00000000, 0.00000000], [0.21404116, 0.01002283]],
                [[1.00000000, 0.00000000], [0.21404116, 0.03310477]],
                [[1.00000000, 0.00000000], [0.21404116, 0.07323898]],
            ],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        f = kornia.color.rgb_to_linear_rgb
        assert_close(f(data), expected)

    def test_unit_linear(self, device, dtype):
        data = torch.tensor(
            [
                [[1.00000000, 0.00000000], [0.21404116, 0.01002283]],
                [[1.00000000, 0.00000000], [0.21404116, 0.03310477]],
                [[1.00000000, 0.00000000], [0.21404116, 0.07323898]],
            ],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [[[1.0, 0.0], [0.5, 0.1]], [[1.0, 0.0], [0.5, 0.2]], [[1.0, 0.0], [0.5, 0.3]]], device=device, dtype=dtype
        )  # 3x2x2

        f = kornia.color.linear_rgb_to_rgb
        assert_close(f(data), expected)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_linear_rgb, (img,), raise_exception=True)
        assert gradcheck(kornia.color.linear_rgb_to_rgb, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_linear_rgb
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.jit
    def test_jit_linear(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.linear_rgb_to_rgb
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToLinearRgb().to(device, dtype)
        fcn = kornia.color.rgb_to_linear_rgb
        assert_close(ops(img), fcn(img))

    @pytest.mark.nn
    def test_module_linear(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.LinearRgbToRgb().to(device, dtype)
        fcn = kornia.color.linear_rgb_to_rgb
        assert_close(ops(img), fcn(img))
