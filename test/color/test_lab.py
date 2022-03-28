import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils
from kornia.testing import BaseTester, assert_close


class TestRgbToLab(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_lab(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_lab(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_lab([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_lab(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_lab(img)

    def test_unit(self, device, dtype):
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

        # Reference output generated using skimage: rgb2lab(data)
        expected = torch.tensor(
            [
                [
                    [0.0, 100.0, 79.75208576, 86.38913217, 55.25164186],
                    [51.66668553, 43.81214392, 48.93865503, 39.03804484, 52.55152607],
                    [23.7114063, 52.38661792, 72.54607218, 53.89587489, 67.94892652],
                    [45.02897165, 75.98315061, 78.257619, 61.85069778, 33.77972627],
                ],
                [
                    [0.0, -0.002454937, -5.40909568, -37.74958445, -55.02172792],
                    [24.16049084, 58.53088654, 75.33566652, -9.65827726, 76.94753157],
                    [36.53113547, -28.57665427, -54.16269089, 6.2586262, -67.69290198],
                    [12.32708756, -33.04781428, -29.29282657, 13.46090338, 42.98737069],
                ],
                [
                    [0.0, 0.00465342, -9.49591204, 32.9931831, 47.80929165],
                    [-16.11189945, 7.72083678, 19.17820444, -6.90801653, -17.46468994],
                    [-39.99097133, 9.92432127, 19.90687976, 2.40429413, 61.24066709],
                    [-25.45166461, -22.94347485, -31.32259433, 47.2621717, -60.05694598],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(kornia.color.rgb_to_lab(data), expected, rtol=tol_val, atol=tol_val)

    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        lab = kornia.color.rgb_to_lab
        rgb = kornia.color.lab_to_rgb

        data_out = lab(rgb(data, clip=False))
        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(data_out, data, rtol=tol_val, atol=tol_val)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_lab, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_lab
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img), rtol=1e-3, atol=1e-3)

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToLab().to(device, dtype)
        fcn = kornia.color.rgb_to_lab
        assert_close(ops(img), fcn(img))


class TestLabToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.lab_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.lab_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.lab_to_rgb([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.lab_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.lab_to_rgb(img)

    def test_unit(self, device, dtype):
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

        # Reference output generated using skimage: lab2rgb(data)
        expected = torch.tensor(
            [
                [
                    [
                        [0.63513142, 0.0, 0.10660624, 0.79048697, 0.26823414],
                        [0.48903025, 0.64529494, 0.91140099, 0.15877841, 0.45987959],
                        [1.0, 0.36069696, 0.29236125, 0.55744393, 0.0],
                        [0.41710863, 0.3198324, 0.0, 0.94256868, 0.82748892],
                    ],
                    [
                        [0.28210726, 0.26080003, 0.15027717, 0.54540429, 0.80323837],
                        [0.748392, 0.68774842, 0.24204415, 0.83695682, 0.9902132],
                        [0.0, 0.79101603, 0.26633725, 0.89223337, 0.82301254],
                        [0.84857086, 0.34455393, 0.66555314, 0.86168397, 0.8948667],
                    ],
                    [
                        [0.94172458, 0.66390044, 0.21043296, 0.02453515, 0.04169043],
                        [0.28233233, 0.20235374, 0.19803933, 0.55069441, 0.0],
                        [0.50205101, 0.0, 0.79745394, 0.25376936, 0.6114783],
                        [0.0, 1.0, 0.80867314, 1.0, 0.28778443],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_unclipped = torch.tensor(
            [
                [
                    [
                        [0.63513142, -1.78708635, 0.10660624, 0.79048697, 0.26823414],
                        [0.48903025, 0.64529494, 0.91140099, 0.15877841, 0.45987959],
                        [1.01488435, 0.36069696, 0.29236125, 0.55744393, -0.28090181],
                        [0.41710863, 0.3198324, -1.81087917, 0.94256868, 0.82748892],
                    ],
                    [
                        [0.28210726, 0.26080003, 0.15027717, 0.54540429, 0.80323837],
                        [0.748392, 0.68774842, 0.24204415, 0.83695682, 0.9902132],
                        [-1.37862046, 0.79101603, 0.26633725, 0.89223337, 0.82301254],
                        [0.84857086, 0.34455393, 0.66555314, 0.86168397, 0.8948667],
                    ],
                    [
                        [0.94172458, 0.66390044, 0.21043296, 0.02453515, 0.04169043],
                        [0.28233233, 0.20235374, 0.19803933, 0.55069441, -0.62707704],
                        [0.50205101, -0.25005965, 0.79745394, 0.25376936, 0.6114783],
                        [-0.55802926, 1.0223477, 0.80867314, 1.07334156, 0.28778443],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(kornia.color.lab_to_rgb(data), expected, rtol=tol_val, atol=tol_val)
        assert_close(kornia.color.lab_to_rgb(data, clip=False), expected_unclipped, rtol=tol_val, atol=tol_val)

    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        lab = kornia.color.rgb_to_lab
        rgb = kornia.color.lab_to_rgb

        unclipped_data_out = rgb(lab(data), clip=False)
        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(unclipped_data_out, data, rtol=tol_val, atol=tol_val)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        img = kornia.color.rgb_to_lab(img)
        assert gradcheck(kornia.color.lab_to_rgb, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.lab_to_rgb
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.LabToRgb().to(device, dtype)
        fcn = kornia.color.lab_to_rgb
        assert_close(ops(img), fcn(img))
