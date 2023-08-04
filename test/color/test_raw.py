import warnings

import pytest
import torch
from packaging import version
from torch.autograd import gradcheck

import kornia
from kornia.testing import BaseTester, assert_close  # test utils


class TestRawToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 1, 4, 6
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.raw_to_rgb(img, kornia.color.CFA.BG), torch.Tensor)

    @pytest.mark.parametrize("batch_size, height, width", [(1, 6, 4), (2, 2, 4), (3, 4, 2)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        assert kornia.color.raw_to_rgb(img, kornia.color.CFA.BG).shape == (batch_size, 3, height, width)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.raw_to_rgb([0.0], kornia.color.CFA.BG)

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.raw_to_rgb(img, kornia.color.CFA.GB)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.raw_to_rgb(img, kornia.color.CFA.RG)

        with pytest.raises(ValueError):
            img = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert kornia.color.raw_to_rgb(img, kornia.color.CFA.GR)

        # dimensionality test
        with pytest.raises(ValueError):
            img = torch.ones(3, 2, 1, device=device, dtype=dtype)
            assert kornia.color.raw_to_rgb(img, kornia.color.CFA.GR)

        # dimensionality test
        with pytest.raises(ValueError):
            img = torch.ones(3, 1, 2, device=device, dtype=dtype)
            assert kornia.color.raw_to_rgb(img, kornia.color.CFA.GR)

    # With he current implementations we should get back an identical raw representation when doing raw -> rgb -> raw
    # Note that with more advanced implementations this may not necessarily be true or desirable
    def test_forth_and_back(self, device, dtype):  # skipcq: PYL-R0201
        data = torch.rand(1, 80, 80, device=device, dtype=dtype)
        raw = kornia.color.rgb_to_raw
        rgb = kornia.color.raw_to_rgb

        for x in kornia.color.CFA:
            data_out = raw(rgb(data, cfa=x), cfa=x)
            assert_close(data_out, data)

    # make sure different cfas are actually different
    def test_cfas_not_the_same(self, device, dtype):  # skipcq: PYL-R0201
        data = torch.rand(1, 16, 16, device=device, dtype=dtype)
        assert (
            torch.max(
                kornia.color.raw_to_rgb(data, kornia.color.CFA.BG) - kornia.color.raw_to_rgb(data, kornia.color.CFA.RG)
            )
            > 0.0
        )

    # The outcome will be very different for different implementations
    # Here we compare against a current baseline, it is safe to update this if the underlying algorithm changes
    def test_functional(self, device, dtype):  # skipcq: PYL-R0201
        data = torch.tensor(
            [[[1, 0.5, 0.2, 0.4], [0.75, 0.25, 0.8, 0.3], [0.65, 0.15, 0.7, 0.2], [0.55, 0.5, 0.6, 0.1]]],
            device=device,
            dtype=dtype,
        )
        # checked by hand as correct interpolation. Note the ugly replication that happens for Red on the last column
        # and row. We shall accept to live with that
        expected = torch.tensor(
            [
                [
                    [1.0000, 0.6000, 0.2000, 0.2000],
                    [0.8250, 0.6375, 0.4500, 0.4500],
                    [0.6500, 0.6750, 0.7000, 0.7000],
                    [0.6500, 0.6750, 0.7000, 0.7000],
                ],
                [
                    [0.6250, 0.5000, 0.6250, 0.4000],
                    [0.7500, 0.5500, 0.8000, 0.5500],
                    [0.4000, 0.1500, 0.4375, 0.2000],
                    [0.5500, 0.3625, 0.6000, 0.4000],
                ],
                [
                    [0.2500, 0.2500, 0.2750, 0.3000],
                    [0.2500, 0.2500, 0.2750, 0.3000],
                    [0.3750, 0.3750, 0.2875, 0.2000],
                    [0.5000, 0.5000, 0.3000, 0.1000],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        img_rgb = kornia.color.raw_to_rgb(data, kornia.color.raw.CFA.BG)
        assert_close(img_rgb, expected)

    # If we roll the data and the different CFAs they give the same result (expect on edges!)
    def test_cfa_on_rolled(self, device, dtype):  # skipcq: PYL-R0201
        data = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        bgres = kornia.color.raw_to_rgb(data, kornia.color.raw.CFA.BG)
        gbres = kornia.color.raw_to_rgb(data.roll((0, 1), (-2, -1)), kornia.color.raw.CFA.GB)
        grres = kornia.color.raw_to_rgb(data.roll((1, 0), (-2, -1)), kornia.color.raw.CFA.GR)
        rgres = kornia.color.raw_to_rgb(data.roll((1, 1), (-2, -1)), kornia.color.raw.CFA.RG)

        assert_close(bgres[:, :, 1:5, 1:5], gbres[:, :, 1:5, 2:6])
        assert_close(bgres[:, :, 1:5, 1:5], grres[:, :, 2:6, 1:5])
        assert_close(bgres[:, :, 1:5, 1:5], rgres[:, :, 2:6, 2:6])

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.raw_to_rgb, (img, kornia.color.raw.CFA.BG), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        if version.parse(torch.__version__) < version.parse('1.7.0'):
            warnings.warn(
                "This test is not compatible with pytorch < 1.7.0. This message will be removed as soon as we do not "
                "support pytorch 1.6.0. `rgb_to_hls()` method for pytorch < 1.7.0 version cannot be compiled with JIT.",
                DeprecationWarning,
                stacklevel=2,
            )
            return
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.raw_to_rgb
        op_jit = torch.jit.script(op)
        assert_close(op(img, kornia.color.raw.CFA.BG), op_jit(img, kornia.color.raw.CFA.BG))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        raw_ops = kornia.color.RawToRgb(kornia.color.raw.CFA.BG).to(device, dtype)
        raw_fcn = kornia.color.raw_to_rgb
        assert_close(raw_ops(img), raw_fcn(img, kornia.color.raw.CFA.BG))


class TestRgbToRaw(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 6
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_raw(img, kornia.color.raw.CFA.BG), torch.Tensor)

    @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 3, height, width, device=device, dtype=dtype)
        assert kornia.color.rgb_to_raw(img, kornia.color.raw.CFA.GR).shape == (batch_size, 1, height, width)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_raw([0.0], kornia.color.raw.CFA.RG)

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_raw(img, kornia.color.raw.CFA.BG)

        # Reverse test in rawtorgb is sufficient functional test

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_raw, (img, kornia.color.raw.CFA.BG), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        if version.parse(torch.__version__) < version.parse('1.7.0'):
            warnings.warn(
                "This test is not compatible with pytorch < 1.7.0. This message will be removed as soon as we do not "
                "support pytorch 1.6.0. `rgb_to_hls()` method for pytorch < 1.7.0 version cannot be compiled with JIT.",
                DeprecationWarning,
                stacklevel=2,
            )
            return
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_raw
        op_jit = torch.jit.script(op)
        assert_close(op(img, kornia.color.raw.CFA.BG), op_jit(img, kornia.color.raw.CFA.BG))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        raw_ops = kornia.color.RgbToRaw(kornia.color.raw.CFA.BG).to(device, dtype)
        raw_fcn = kornia.color.rgb_to_raw
        assert_close(raw_ops(img), raw_fcn(img, kornia.color.raw.CFA.BG))
