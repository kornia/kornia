import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestCornerHarris:
    def test_shape(self):
        inp = torch.ones(1, 3, 4, 4)
        harris = kornia.feature.CornerHarris(k=0.04)
        assert harris(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        harris = kornia.feature.CornerHarris(k=0.04)
        assert harris(inp).shape == (2, 6, 4, 4)

    def test_corners(self):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]]).float()

        expected = torch.tensor([[[
            [0.001233, 0.003920, 0.001985, 0.000000, 0.001985, 0.003920, 0.001233],
            [0.003920, 0.006507, 0.003976, 0.000000, 0.003976, 0.006507, 0.003920],
            [0.001985, 0.003976, 0.002943, 0.000000, 0.002943, 0.003976, 0.001985],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.001985, 0.003976, 0.002943, 0.000000, 0.002943, 0.003976, 0.001985],
            [0.003920, 0.006507, 0.003976, 0.000000, 0.003976, 0.006507, 0.003920],
            [0.001233, 0.003920, 0.001985, 0.000000, 0.001985, 0.003920, 0.001233]]]]).float()
        harris = kornia.feature.CornerHarris(k=0.04)
        scores = harris(inp)
        assert_allclose(scores, expected, atol=1e-4, rtol=1e-3)

    def test_corners_batch(self):
        inp = torch.tensor([[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ], [
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[
            [0.001233, 0.003920, 0.001985, 0.000000, 0.001985, 0.003920, 0.001233],
            [0.003920, 0.006507, 0.003976, 0.000000, 0.003976, 0.006507, 0.003920],
            [0.001985, 0.003976, 0.002943, 0.000000, 0.002943, 0.003976, 0.001985],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.001985, 0.003976, 0.002943, 0.000000, 0.002943, 0.003976, 0.001985],
            [0.003920, 0.006507, 0.003976, 0.000000, 0.003976, 0.006507, 0.003920],
            [0.001233, 0.003920, 0.001985, 0.000000, 0.001985, 0.003920, 0.001233]
        ], [
            [0.001233, 0.003920, 0.001985, 0.001985, 0.003920, 0.000589, 0.000000],
            [0.003920, 0.006507, 0.003976, 0.003976, 0.006507, 0.001526, 0.000008],
            [0.001985, 0.003976, 0.002943, 0.002943, 0.003976, 0.000542, 0.000000],
            [0.001985, 0.003976, 0.002943, 0.002943, 0.003976, 0.000542, 0.000000],
            [0.003920, 0.006507, 0.003976, 0.003976, 0.006507, 0.001526, 0.000008],
            [0.000589, 0.001526, 0.000542, 0.000542, 0.001526, 0.000277, 0.000000],
            [0.000000, 0.000008, 0.000000, 0.000000, 0.000008, 0.000000, 0.000000]
        ]]).repeat(2, 1, 1, 1)
        scores = kornia.feature.harris_response(inp, k=0.04)
        assert_allclose(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self):
        k = 0.04
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.harris_response, (img, k),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input, k):
            return kornia.feature.harris_response(input, k)
        k = torch.tensor(0.04)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img, k)
        expected = kornia.feature.harris_response(img, k)
        assert_allclose(actual, expected)


class TestCornerGFTT:
    def test_shape(self):
        inp = torch.ones(1, 3, 4, 4)
        shi_tomasi = kornia.feature.CornerGFTT()
        assert shi_tomasi(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        shi_tomasi = kornia.feature.CornerGFTT()
        assert shi_tomasi(inp).shape == (2, 6, 4, 4)

    def test_corners(self):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]]).float()

        expected = torch.tensor([[[
            [0.01548, 0.03340, 0.01944, 0.00000, 0.01944, 0.03340, 0.01548],
            [0.03340, 0.05748, 0.03388, 0.00000, 0.03388, 0.05748, 0.03340],
            [0.01944, 0.03388, 0.04974, 0.00000, 0.04974, 0.03388, 0.01944],
            [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
            [0.01944, 0.03388, 0.04974, 0.00000, 0.04974, 0.03388, 0.01944],
            [0.03340, 0.05748, 0.03388, 0.00000, 0.03388, 0.05748, 0.03340],
            [0.01548, 0.03340, 0.01944, 0.00000, 0.01944, 0.03340, 0.01548]]]]).float()
        shi_tomasi = kornia.feature.CornerGFTT()
        scores = shi_tomasi(inp)
        assert_allclose(scores, expected, atol=1e-4, rtol=1e-3)

    def test_corners_batch(self):
        inp = torch.tensor([[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ], [
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[
            [0.01548, 0.03340, 0.01944, 0.00000, 0.01944, 0.03340, 0.01548],
            [0.03340, 0.05748, 0.03388, 0.00000, 0.03388, 0.05748, 0.03340],
            [0.01944, 0.03388, 0.04974, 0.00000, 0.04974, 0.03388, 0.01944],
            [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
            [0.01944, 0.03388, 0.04974, 0.00000, 0.04974, 0.03388, 0.01944],
            [0.03340, 0.05748, 0.03388, 0.00000, 0.03388, 0.05748, 0.03340],
            [0.01548, 0.03340, 0.01944, 0.00000, 0.01944, 0.03340, 0.01548]
        ], [
            [0.01548, 0.03340, 0.01944, 0.01944, 0.03340, 0.01090, 0.00136],
            [0.03340, 0.05748, 0.03388, 0.03388, 0.05748, 0.01981, 0.00348],
            [0.01944, 0.03388, 0.04974, 0.04974, 0.03388, 0.01070, 0.00193],
            [0.01944, 0.03388, 0.04974, 0.04974, 0.03388, 0.01070, 0.00193],
            [0.03340, 0.05748, 0.03388, 0.03388, 0.05748, 0.01981, 0.00348],
            [0.01090, 0.01981, 0.01070, 0.01070, 0.01981, 0.00774, 0.00121],
            [0.00136, 0.00348, 0.00193, 0.00193, 0.00348, 0.00121, 0.00000]
        ]]).repeat(2, 1, 1, 1)
        shi_tomasi = kornia.feature.CornerGFTT()
        scores = shi_tomasi(inp)
        assert_allclose(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.gftt_response, (img),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.feature.gftt_response(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.feature.gftt_response(img)
        assert_allclose(actual, expected)


class TestBlobHessian:
    def test_shape(self):
        inp = torch.ones(1, 3, 4, 4)
        shi_tomasi = kornia.feature.BlobHessian()
        assert shi_tomasi(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        shi_tomasi = kornia.feature.BlobHessian()
        assert shi_tomasi(inp).shape == (2, 6, 4, 4)

    def test_blobs_batch(self):
        inp = torch.tensor([[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ], [
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 0., 0., 0., 0.],
            [0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[
            [0.0564, 0.0759, 0.0342, 0.0759, 0.0564, 0.0057, 0.0000],
            [0.0759, 0.0330, 0.0752, 0.0330, 0.0759, 0.0096, 0.0000],
            [0.0342, 0.0752, 0.1914, 0.0752, 0.0342, 0.0068, 0.0000],
            [0.0759, 0.0330, 0.0752, 0.0330, 0.0759, 0.0096, 0.0000],
            [0.0564, 0.0759, 0.0342, 0.0759, 0.0564, 0.0057, 0.0000],
            [0.0057, 0.0096, 0.0068, 0.0096, 0.0057, 0.0005, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        ], [
            [0.0564, 0.0522, 0.0522, 0.0564, 0.0057, 0.0000, 0.0000],
            [0.0522, 0.0688, 0.0688, 0.0123, 0.0033, 0.0057, 0.0005],
            [0.0522, 0.0688, 0.0755, 0.1111, 0.0123, 0.0564, 0.0057],
            [0.0564, 0.0123, 0.1111, 0.0755, 0.0688, 0.0522, 0.0080],
            [0.0057, 0.0033, 0.0123, 0.0688, 0.0688, 0.0522, 0.0080],
            [0.0000, 0.0057, 0.0564, 0.0522, 0.0522, 0.0564, 0.0057],
            [0.0000, 0.0005, 0.0057, 0.0080, 0.0080, 0.0057, 0.0005]
        ]]).repeat(2, 1, 1, 1)
        shi_tomasi = kornia.feature.BlobHessian()
        scores = shi_tomasi(inp)
        assert_allclose(scores, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.hessian_response, (img),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.feature.hessian_response(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.feature.hessian_response(img)
        assert_allclose(actual, expected)
