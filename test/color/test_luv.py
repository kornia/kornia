from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import BaseTester


class TestRgbToLuv(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_luv(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_luv(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_luv([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_luv(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_luv(img)

    def test_unit(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        data = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.69396782, 0.63581685, 0.09902618],
                    [0.59459005, 0.74215373, 0.89662376, 0.25920381, 0.89937686],
                    [0.29857584, 0.28139791, 0.16441015, 0.55507519, 0.06124221],
                    [0.40908658, 0.10261389, 0.01691456, 0.76006799, 0.32971736],
                ],
                [
                    [0.0, 1.0, 0.79009938, 0.91742945, 0.60044175],
                    [0.42812678, 0.18552390, 0.04186043, 0.38030245, 0.15420346],
                    [0.13552373, 0.53955473, 0.79102736, 0.49050815, 0.75271446],
                    [0.39861023, 0.80680277, 0.82823833, 0.54438462, 0.22063386],
                ],
                [
                    [0.0, 1.0, 0.84317145, 0.59529881, 0.15297393],
                    [0.59235313, 0.36617295, 0.34600773, 0.40304737, 0.61720451],
                    [0.46040250, 0.42006640, 0.54765106, 0.48982632, 0.13914755],
                    [0.58402964, 0.89597990, 0.98276161, 0.25019163, 0.69285921],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # Reference output generated using skimage: rgb2luv(data)
        expected = torch.tensor(
            [
                [
                    [0.0, 100.0, 79.75208282, 86.38912964, 55.25164032],
                    [51.66668701, 43.81214523, 48.93865585, 39.03804398, 52.55152512],
                    [23.71140671, 52.38661957, 72.54607391, 53.89587402, 67.94892883],
                    [45.02897263, 75.98315430, 78.25762177, 61.85069656, 33.77972794],
                ],
                [
                    [-0.0, -0.00054950, -13.54032803, -35.42317200, -49.27433014],
                    [21.34596062, 94.13956451, 137.11340332, -14.69241238, 102.94833374],
                    [9.55611229, -30.01761436, -58.94236755, 9.83261871, -62.96137619],
                    [-1.55336237, -55.22497559, -56.21067810, 43.76751328, 1.46367633],
                ],
                [
                    [-0.0, 0.00766720, -13.74480152, 52.17128372, 60.92724228],
                    [-27.01125526, -1.72837746, 6.57535267, -7.83582020, -38.45543289],
                    [-50.89970779, 17.65329361, 36.54148102, 2.25501800, 78.93702698],
                    [-38.39783859, -31.71204376, -46.63606644, 50.16629410, -84.74416351],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(kornia.color.rgb_to_luv(data), expected, low_tolerance=True)

    def test_forth_and_back(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        luv = kornia.color.rgb_to_luv
        rgb = kornia.color.luv_to_rgb

        data_out = luv(rgb(data))
        self.assert_close(data_out, data)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_luv, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_luv
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToLuv().to(device, dtype)
        fcn = kornia.color.rgb_to_luv
        self.assert_close(ops(img), fcn(img))


class TestLuvToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.luv_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.luv_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.luv_to_rgb([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.luv_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.luv_to_rgb(img)

    def test_unit(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')
        data = torch.tensor(
            [
                [
                    [
                        [50.21928787, 23.29810143, 14.98279190, 62.50927353, 72.78904724],
                        [70.86846924, 68.75330353, 52.81696701, 76.17090607, 88.63134003],
                        [46.87160873, 72.38699341, 37.71450806, 82.57386780, 74.79967499],
                        [77.33016968, 47.39180374, 61.76217651, 90.83254242, 86.96239471],
                    ],
                    [
                        [65.81327057, -3.69859719, 0.16971001, 14.86583614, -65.54960632],
                        [-41.03258133, -19.52661896, 64.16155243, -58.53935242, -71.78411102],
                        [112.05227661, -60.13330460, 43.07910538, -51.01456833, -58.25787354],
                        [-62.37575531, 50.88882065, -39.27450943, 17.00958824, -24.93779755],
                    ],
                    [
                        [-69.53346252, -73.34986877, -11.47461891, 66.73863220, 70.43983459],
                        [51.92737579, 58.77009583, 45.97863388, 24.44452858, 98.81991577],
                        [-7.60597992, 78.97976685, -69.31867218, 67.33953857, 14.28889370],
                        [92.31149292, -85.91405487, -32.83668518, -23.45091820, 69.99038696],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # Reference output generated using skimage: luv2rgb(data)

        expected = torch.tensor(
            [
                [
                    [
                        [0.78923208, 0.17048222, 0.14947766, 0.65528989, 0.07863078],
                        [0.41649094, 0.55222923, 0.72673196, 0.21939684, 0.34298307],
                        [0.82763243, 0.24021322, 0.58888060, 0.47255886, 0.16407511],
                        [0.30320778, 0.72233224, 0.21593384, 0.98893607, 0.71707106],
                    ],
                    [
                        [0.20532851, 0.13188709, 0.13879408, 0.59964627, 0.80721593],
                        [0.75411713, 0.70656943, 0.41770950, 0.82750136, 0.99659365],
                        [0.12436169, 0.79804462, 0.10958754, 0.89803618, 0.81000644],
                        [0.85726571, 0.17667055, 0.63285238, 0.85567462, 0.91538441],
                    ],
                    [
                        [0.73985511, 0.59308004, 0.21156698, 0.03804367, 0.32732114],
                        [0.42489606, 0.33011687, 0.12804756, 0.64905322, 0.25216782],
                        [0.41637793, 0.22158240, 0.63437861, 0.46121466, 0.68336427],
                        [0.06325728, 0.78878325, 0.74280596, 0.99514300, 0.47176042],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(kornia.color.luv_to_rgb(data), expected)

    def test_forth_and_back(self, device, dtype):
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        luv = kornia.color.rgb_to_luv
        rgb = kornia.color.luv_to_rgb

        data_out = rgb(luv(data))
        self.assert_close(data_out, data)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        img = kornia.color.rgb_to_luv(img)
        assert gradcheck(kornia.color.luv_to_rgb, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.luv_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.LuvToRgb().to(device, dtype)
        fcn = kornia.color.luv_to_rgb
        self.assert_close(ops(img), fcn(img))
