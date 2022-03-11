import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestResize:
    def test_smoke(self, device, dtype):
        inp = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        assert_close(inp, out, atol=1e-4, rtol=1e-4)

        # 2D
        inp = torch.rand(3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        assert_close(inp, out, atol=1e-4, rtol=1e-4)

        # 3D
        inp = torch.rand(3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        assert_close(inp, out, atol=1e-4, rtol=1e-4)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        assert_close(inp, out, atol=1e-4, rtol=1e-4)

    def test_upsize(self, device, dtype):
        inp = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (1, 3, 6, 8)

        # 2D
        inp = torch.rand(3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (6, 8)

        # 3D
        inp = torch.rand(3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (3, 6, 8)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (1, 2, 3, 2, 1, 3, 6, 8)

    def test_downsize(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (1, 3, 3, 1)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (3, 1)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (3, 3, 1)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (1, 2, 3, 2, 1, 3, 3, 1)

    def test_downsizeAA(self, device, dtype):
        inp = torch.rand(1, 3, 10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (1, 3, 5, 3)

        inp = torch.rand(1, 1, 20, 10, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (15, 8), align_corners=False, antialias=True)
        assert out.shape == (1, 1, 15, 8)

        # 2D
        inp = torch.rand(10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (5, 3)

        # 3D
        inp = torch.rand(3, 10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (3, 5, 3)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (1, 2, 3, 2, 1, 3, 5, 3)

    def test_one_param(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (1, 3, 25, 10)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (25, 10)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (3, 25, 10)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (1, 2, 3, 2, 1, 3, 25, 10)

    def test_one_param_long(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (1, 3, 10, 4)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (10, 4)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (3, 10, 4)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (1, 2, 3, 2, 1, 3, 10, 4)

    def test_one_param_vert(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (1, 3, 10, 4)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (10, 4)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (3, 10, 4)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (1, 2, 3, 2, 1, 3, 10, 4)

    def test_one_param_horz(self, device, dtype):
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

        # 2D
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

        # 3D
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

        # arbitrary dim
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

    def test_gradcheck(self, device, dtype):
        # test parameters
        new_size = 4
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            kornia.geometry.transform.Resize(new_size, align_corners=False), (input,), raise_exception=True
        )


class TestRescale:
    def test_smoke(self, device, dtype):
        input = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, (1.0, 1.0), align_corners=False)
        assert_close(input, output, atol=1e-4, rtol=1e-4)

    def test_upsize(self, device, dtype):
        input = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, (3.0, 2.0), align_corners=False)
        assert output.shape == (1, 3, 9, 8)

    def test_downsize(self, device, dtype):
        input = torch.rand(1, 3, 9, 8, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, (1.0 / 3.0, 1.0 / 2.0), align_corners=False)
        assert output.shape == (1, 3, 3, 4)

    def test_downscale_values(self, device, dtype):
        inp_x = torch.arange(20, device=device, dtype=dtype) / 20.0
        inp = inp_x[None].T @ inp_x[None]
        inp = inp[None, None]
        out = kornia.geometry.transform.rescale(inp, (0.25, 0.25), antialias=False, align_corners=False)
        expected = torch.tensor(
            [
                [
                    [
                        [0.0056, 0.0206, 0.0356, 0.0506, 0.0656],
                        [0.0206, 0.0756, 0.1306, 0.1856, 0.2406],
                        [0.0356, 0.1306, 0.2256, 0.3206, 0.4156],
                        [0.0506, 0.1856, 0.3206, 0.4556, 0.5906],
                        [0.0656, 0.2406, 0.4156, 0.5906, 0.7656],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_downscale_values_AA(self, device, dtype):
        inp_x = torch.arange(20, device=device, dtype=dtype) / 20.0
        inp = inp_x[None].T @ inp_x[None]
        inp = inp[None, None]
        out = kornia.geometry.transform.rescale(inp, (0.25, 0.25), antialias=True, align_corners=False)
        expected = torch.tensor(
            [
                [
                    [
                        [0.0074, 0.0237, 0.0409, 0.0581, 0.0743],
                        [0.0237, 0.0756, 0.1306, 0.1856, 0.2376],
                        [0.0409, 0.1306, 0.2256, 0.3206, 0.4104],
                        [0.0581, 0.1856, 0.3206, 0.4556, 0.5832],
                        [0.0743, 0.2376, 0.4104, 0.5832, 0.7464]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_one_param(self, device, dtype):
        input = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, 2.0, align_corners=False)
        assert output.shape == (1, 3, 6, 8)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)
        assert gradcheck(kornia.geometry.transform.Rescale(2.0, align_corners=False), (input,), raise_exception=True)


class TestRotate:
    def test_angle90(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]], device=device, dtype=dtype)
        # prepare transformation
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Rotate(angle, align_corners=True)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_angle90_batch2(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]], [[[0.0, 0.0], [5.0, 3.0], [6.0, 4.0], [0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        angle = torch.tensor([90.0, -90.0], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Rotate(angle, align_corners=True)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_angle90_batch2_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]], [[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Rotate(angle, align_corners=True)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # test parameters
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        angle = utils.tensor_to_gradcheck_var(angle, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.geometry.transform.rotate, (input, angle), raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        rot = kornia.geometry.transform.Rotate(angle)
        rot_traced = torch.jit.trace(kornia.geometry.transform.Rotate(angle), img)
        assert_close(rot(img), rot_traced(img))


class TestTranslate:
    def test_dxdy(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]], device=device, dtype=dtype)
        # prepare transformation
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Translate(translation, align_corners=True)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_dxdy_batch(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]], [[[0.0, 0.0], [0.0, 1.0], [0.0, 3.0], [0.0, 5.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        translation = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Translate(translation, align_corners=True)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_dxdy_batch_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]], [[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Translate(translation, align_corners=True)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # test parameters
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        translation = utils.tensor_to_gradcheck_var(translation, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.geometry.transform.translate, (input, translation), raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        trans = kornia.geometry.transform.Translate(translation)
        trans_traced = torch.jit.trace(kornia.geometry.transform.Translate(translation), img)
        assert_close(trans(img), trans_traced(img), atol=1e-4, rtol=1e-4)


class TestScale:
    def test_scale_factor_2(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        scale_factor = torch.tensor([[2.0, 2.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        assert_close(transform(inp).sum().item(), 12.25, atol=1e-4, rtol=1e-4)

    def test_scale_factor_05(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_scale_factor_05_batch2(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        # prepare transformation
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_scale_factor_05_batch2_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        # prepare transformation
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # test parameters
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        scale_factor = utils.tensor_to_gradcheck_var(scale_factor, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.geometry.transform.scale, (input, scale_factor), raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        trans = kornia.geometry.transform.Scale(scale_factor)
        trans_traced = torch.jit.trace(kornia.Scale(scale_factor), img)
        assert_close(trans(img), trans_traced(img), atol=1e-4, rtol=1e-4)


class TestShear:
    def test_shear_x(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [[[0.75, 1.0, 1.0, 1.0], [0.25, 1.0, 1.0, 1.0], [0.0, 0.75, 1.0, 1.0], [0.0, 0.25, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_shear_y(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [[[0.75, 0.25, 0.0, 0.0], [1.0, 1.0, 0.75, 0.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        # prepare transformation
        shear = torch.tensor([[0.0, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_shear_batch2(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        expected = torch.tensor(
            [
                [[[0.75, 1.0, 1.0, 1.0], [0.25, 1.0, 1.0, 1.0], [0.0, 0.75, 1.0, 1.0], [0.0, 0.25, 1.0, 1.0]]],
                [[[0.75, 0.25, 0.0, 0.0], [1.0, 1.0, 0.75, 0.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0], [0.0, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_shear_batch2_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        expected = torch.tensor(
            [[[[0.75, 1.0, 1.0, 1.0], [0.25, 1.0, 1.0, 1.0], [0.0, 0.75, 1.0, 1.0], [0.0, 0.25, 1.0, 1.0]]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # test parameters
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        shear = utils.tensor_to_gradcheck_var(shear, requires_grad=False)  # to var

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.geometry.transform.shear, (input, shear), raise_exception=True)

    @pytest.mark.skip('Need deep look into it since crashes everywhere.')
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        trans = kornia.geometry.transform.Shear(shear, align_corners=False)
        trans_traced = torch.jit.trace(kornia.geometry.transform.Shear(shear), img)
        assert_close(trans(img), trans_traced(img), atol=1e-4, rtol=1e-4)


class TestAffine2d:
    def test_affine_no_args(self):
        with pytest.raises(RuntimeError):
            kornia.geometry.transform.Affine()

    def test_affine_batch_size_mismatch(self, device, dtype):
        with pytest.raises(RuntimeError):
            angle = torch.rand(1, device=device, dtype=dtype)
            translation = torch.rand(2, 2, device=device, dtype=dtype)
            kornia.geometry.transform.Affine(angle, translation)

    def test_affine_rotate(self, device, dtype):
        # TODO: Remove when #666 is implemented
        if device.type == 'cuda':
            pytest.skip("Currently breaks in CUDA." "See https://github.com/kornia/kornia/issues/666")
        torch.manual_seed(0)
        angle = torch.rand(1, device=device, dtype=dtype) * 90.0
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        transform = kornia.geometry.transform.Affine(angle=angle).to(device=device, dtype=dtype)
        actual = transform(input)
        expected = kornia.geometry.transform.rotate(input, angle)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_affine_translate(self, device, dtype):
        # TODO: Remove when #666 is implemented
        if device.type == 'cuda':
            pytest.skip("Currently breaks in CUDA." "See https://github.com/kornia/kornia/issues/666")
        torch.manual_seed(0)
        translation = torch.rand(1, 2, device=device, dtype=dtype) * 2.0
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        transform = kornia.geometry.transform.Affine(translation=translation).to(device=device, dtype=dtype)
        actual = transform(input)
        expected = kornia.geometry.transform.translate(input, translation)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_affine_scale(self, device, dtype):
        # TODO: Remove when #666 is implemented
        if device.type == 'cuda':
            pytest.skip("Currently breaks in CUDA." "See https://github.com/kornia/kornia/issues/666")
        torch.manual_seed(0)
        _scale_factor = torch.rand(1, device=device, dtype=dtype) * 2.0
        scale_factor = torch.stack([_scale_factor, _scale_factor], dim=1)
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        transform = kornia.geometry.transform.Affine(scale_factor=scale_factor).to(device=device, dtype=dtype)
        actual = transform(input)
        expected = kornia.geometry.transform.scale(input, scale_factor)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.skip(
        "_compute_shear_matrix and get_affine_matrix2d yield different results. "
        "See https://github.com/kornia/kornia/issues/629 for details."
    )
    def test_affine_shear(self, device, dtype):
        torch.manual_seed(0)
        shear = torch.rand(1, 2, device=device, dtype=dtype)
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        transform = kornia.geometry.transform.Affine(shear=shear).to(device, dtype)
        actual = transform(input)
        expected = kornia.geometry.transform.shear(input, shear)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_affine_rotate_translate(self, device, dtype):
        # TODO: Remove when #666 is implemented
        if device.type == 'cuda':
            pytest.skip("Currently breaks in CUDA." "See https://github.com/kornia/kornia/issues/666")
        batch_size = 2

        input = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1, 1)

        angle = torch.tensor(180.0, device=device, dtype=dtype).repeat(batch_size)
        translation = torch.tensor([1.0, 0.0], device=device, dtype=dtype).repeat(batch_size, 1)

        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1, 1)

        transform = kornia.geometry.transform.Affine(angle=angle, translation=translation, align_corners=True).to(
            device=device, dtype=dtype
        )
        actual = transform(input)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_compose_affine_matrix_3x3(self, device, dtype):
        """To get parameters:
        import torchvision as tv
        from PIL import Image
        from torch import Tensor as T
        import math
        import random
        img_size = (96,96)
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        tfm = tv.transforms.RandomAffine(degrees=(-25.0,25.0),
                                        scale=(0.6, 1.4) ,
                                        translate=(0, 0.1),
                                        shear=(-25., 25., -20., 20.))
        angle, translations, scale, shear = tfm.get_params(tfm.degrees, tfm.translate,
                                                        tfm.scale, tfm.shear, img_size)
        print (angle, translations, scale, shear)
        output_size = img_size
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)

        matrix = tv.transforms.functional._get_inverse_affine_matrix(center, angle, translations, scale, shear)
        matrix = np.array(matrix).reshape(2,3)
        print (matrix)
        """
        import math

        from torch import Tensor as T

        batch_size, _, height, width = 1, 1, 96, 96
        angle, translations = 6.971339922894188, (0.0, -4.0)
        scale, shear = [0.7785685905190581, 0.7785685905190581], [11.8235607082617, 7.06797949691645]
        matrix_expected = T([[1.27536969, 4.26828945e-01, -3.2876e01], [2.18297196e-03, 1.29424165e00, -1.1717e01]])
        center = T([float(width), float(height)]).view(1, 2) / 2.0 + 0.5
        center = center.expand(batch_size, -1)
        matrix_kornia = kornia.geometry.transform.get_affine_matrix2d(
            T(translations).view(-1, 2),
            center,
            T([scale]).view(-1, 2),
            T([angle]).view(-1),
            T([math.radians(shear[0])]).view(-1, 1),
            T([math.radians(shear[1])]).view(-1, 1),
        )
        matrix_kornia = matrix_kornia.inverse()[0, :2].detach().cpu()
        assert_close(matrix_kornia, matrix_expected, atol=1e-4, rtol=1e-4)


class TestGetAffineMatrix:
    def test_smoke(self, device, dtype):
        H, W = 5, 5
        translation = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        # NOTE: ideally the center should be [W * 0.5, H * 0.5]
        center = torch.tensor([[W // 2, H // 2]], device=device, dtype=dtype)
        zoom1 = torch.ones([1, 1], device=device, dtype=dtype) * 0.5
        zoom2 = torch.ones([1, 1], device=device, dtype=dtype) * 1.0
        zoom = torch.cat([zoom1, zoom2], -1)
        angle = torch.zeros([1], device=device, dtype=dtype)
        affine_mat = kornia.geometry.get_affine_matrix2d(translation, center, zoom, angle)

        img = torch.ones(1, 1, H, W, device=device, dtype=dtype)
        expected = torch.zeros_like(img)
        expected[..., 1:4] = 1.0

        out = kornia.geometry.transform.warp_affine(img, affine_mat[:, :2], (H, W))
        assert_close(out, expected)
