from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import BaseTester


class TestRgbToYcbcr(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_ycbcr(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_ycbcr(img).shape == shape

    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4)])
    def test_rgb_to_y(self, device, dtype, shape):
        img = torch.rand(*shape, device=device, dtype=dtype)
        output_y = kornia.color.rgb_to_y(img)
        output_ycbcr = kornia.color.rgb_to_ycbcr(img)
        assert torch.equal(output_y, output_ycbcr[..., 0:1, :, :])

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_ycbcr([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_ycbcr(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_ycbcr(img)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [
                        [0.7925, 0.8704, 0.2771, 0.3279, 0.0193],
                        [0.2483, 0.6212, 0.9859, 0.5044, 0.5621],
                        [0.5762, 0.3959, 0.2931, 0.2669, 0.0243],
                        [0.6989, 0.0529, 0.8344, 0.6523, 0.8980],
                        [0.5181, 0.9341, 0.2172, 0.0520, 0.7266],
                    ],
                    [
                        [0.8413, 0.0284, 0.3625, 0.8864, 0.5595],
                        [0.3791, 0.0235, 0.4251, 0.0619, 0.5270],
                        [0.3516, 0.8005, 0.9571, 0.4113, 0.6119],
                        [0.0632, 0.8836, 0.0261, 0.1550, 0.4923],
                        [0.2332, 0.7044, 0.9514, 0.2443, 0.2818],
                    ],
                    [
                        [0.6899, 0.2063, 0.3179, 0.8989, 0.4378],
                        [0.0384, 0.5230, 0.6416, 0.9749, 0.7863],
                        [0.8577, 0.3115, 0.2375, 0.5446, 0.9837],
                        [0.3213, 0.6618, 0.5977, 0.3999, 0.4962],
                        [0.1385, 0.5831, 0.9756, 0.8714, 0.8017],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # Reference output generated using OpenCV: cv2.cvtColor(data, cv2.COLOR_RGB2XYZ)
        expected = torch.tensor(
            [
                [
                    [
                        [0.8094629, 0.30041942, 0.33188093, 0.7208117, 0.38409105],
                        [0.30114722, 0.2591345, 0.6174494, 0.29830942, 0.5670594],
                        [0.47643244, 0.623786, 0.6765313, 0.38332558, 0.47857922],
                        [0.2826805, 0.60994685, 0.33298355, 0.3316514, 0.6140681],
                        [0.30756146, 0.75926465, 0.7346588, 0.25828302, 0.4740529],
                    ],
                    [
                        [0.43258172, 0.4469004, 0.492101, 0.60042214, 0.53028214],
                        [0.35180065, 0.64882994, 0.51362985, 0.8815981, 0.62366176],
                        [0.71505237, 0.32387614, 0.2523859, 0.5909421, 0.78490996],
                        [0.52178127, 0.529264, 0.64931786, 0.53848785, 0.43353173],
                        [0.4046721, 0.40065458, 0.6358658, 0.8458253, 0.6847739],
                    ],
                    [
                        [0.48791525, 0.9063825, 0.46094114, 0.21983841, 0.23989305],
                        [0.46231723, 0.7581379, 0.7626976, 0.6469568, 0.496467],
                        [0.5711212, 0.33752257, 0.22661471, 0.4169921, 0.17608929],
                        [0.7967522, 0.10283372, 0.85753804, 0.72865105, 0.70245713],
                        [0.65009415, 0.6246666, 0.13107029, 0.35291404, 0.6800583],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(kornia.color.rgb_to_ycbcr(data), expected, low_tolerance=True)

    # TODO: investigate and implement me
    # def test_forth_and_back(self, device, dtype):
    #    pass

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_ycbcr, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_ycbcr
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToYcbcr().to(device, dtype)
        fcn = kornia.color.rgb_to_ycbcr
        self.assert_close(ops(img), fcn(img))


class TestYcbcrToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.ycbcr_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.ycbcr_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.ycbcr_to_rgb([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.ycbcr_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.ycbcr_to_rgb(img)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [
                        [0.9892, 0.2620, 0.4184, 0.5286, 0.2793],
                        [0.0722, 0.8828, 0.8714, 0.0657, 0.7798],
                        [0.8118, 0.7522, 0.0260, 0.8811, 0.5226],
                        [0.0644, 0.3648, 0.4448, 0.4202, 0.7316],
                        [0.9138, 0.1956, 0.4257, 0.6381, 0.1353],
                    ],
                    [
                        [0.7408, 0.8529, 0.5119, 0.0220, 0.0226],
                        [0.8963, 0.5652, 0.9568, 0.6977, 0.8221],
                        [0.4645, 0.0478, 0.4952, 0.5492, 0.4861],
                        [0.9980, 0.9978, 0.0281, 0.5283, 0.8146],
                        [0.7789, 0.2663, 0.6437, 0.6926, 0.5627],
                    ],
                    [
                        [0.7377, 0.7152, 0.3080, 0.8515, 0.4841],
                        [0.7192, 0.3297, 0.7337, 0.0230, 0.2464],
                        [0.6399, 0.8998, 0.3838, 0.3043, 0.3774],
                        [0.1281, 0.6731, 0.4218, 0.3963, 0.8541],
                        [0.2245, 0.2413, 0.2351, 0.9522, 0.8158],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # Reference output generated using OpenCV: cv2.cvtColor(data, cv2.COLOR_RGB2XYZ)
        expected = torch.tensor(
            [
                [
                    [
                        [1.3226931, 0.5639256, 0.14902398, 1.0217545, 0.2569923],
                        [0.37973762, 0.64386904, 1.1992811, -0.603531, 0.4239992],
                        [1.0080798, 1.3131194, -0.1370286, 0.60653293, 0.3505922],
                        [-0.4573757, 0.6076593, 0.33508536, 0.27470887, 1.2284023],
                        [0.52727354, -0.1673561, 0.05404532, 1.2725366, 0.5783674],
                    ],
                    [
                        [0.736647, -0.01305042, 0.55139434, 0.44206098, 0.4548782],
                        [-0.22063601, 0.98196536, 0.54739904, 0.33826917, 0.850068],
                        [0.72412336, 0.6222996, 0.110618, 1.0039049, 0.614918],
                        [0.15862459, 0.0699634, 0.66296846, 0.4845066, 0.3705502],
                        [1.0145653, 0.46070462, 0.5654058, 0.24897486, -0.11174999],
                    ],
                    [
                        [1.4161384, 0.88769174, 0.4394987, -0.31889397, -0.5671302],
                        [0.77483994, 0.99839956, 1.6813064, 0.41622213, 1.3508832],
                        [0.7488585, -0.04955059, 0.01748962, 0.9683316, 0.49795526],
                        [0.9473541, 1.2473994, -0.3918787, 0.47037587, 1.2893858],
                        [1.4082898, -0.21875012, 0.6804801, 0.9795798, 0.24646705],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(kornia.color.ycbcr_to_rgb(data), expected)

    # TODO: investigate and implement me
    # def test_forth_and_back(self, device, dtype):
    #    pass

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.ycbcr_to_rgb, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.ycbcr_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.YcbcrToRgb().to(device, dtype)
        fcn = kornia.color.ycbcr_to_rgb
        self.assert_close(ops(img), fcn(img))
