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
                    [1.0000, 0.5639, 0.1490, 1.0000, 0.2570],
                    [0.3797, 0.6439, 1.0000, 0.0000, 0.4240],
                    [1.0000, 1.0000, 0.0000, 0.6065, 0.3506],
                    [0.0000, 0.6077, 0.3351, 0.2747, 1.0000],
                    [0.5273, 0.0000, 0.0540, 1.0000, 0.5784]
                    ],
                    [
                    [0.7366, 0.0000, 0.5514, 0.4421, 0.4549],
                    [0.0000, 0.9820, 0.5474, 0.3383, 0.8501],
                    [0.7241, 0.6223, 0.1106, 1.0000, 0.6149],
                    [0.1586, 0.0700, 0.6630, 0.4845, 0.3706],
                    [1.0000, 0.4607, 0.5654, 0.2490, 0.0000]
                    ],
                    [
                    [1.0000, 0.8877, 0.4395, 0.0000, 0.0000],
                    [0.7748, 0.9984, 1.0000, 0.4162, 1.0000],
                    [0.7489, 0.0000, 0.0175, 0.9683, 0.4980],
                    [0.9474, 1.0000, 0.0000, 0.4704, 1.0000],
                    [1.0000, 0.0000, 0.6805, 0.9796, 0.2465]
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(kornia.color.ycbcr_to_rgb(data), expected, low_tolerance=True)

    # TODO: investigate and implement me
    # def test_forth_and_back(self, device, dtype):
    #    pass

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_ycbcr, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
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
                        [1.0000, 0.5639256, 0.14902398, 1.0000, 0.2569923],
                        [0.37973762, 0.64386904, 1.0000, 0.0000, 0.4239992],
                        [1.0000, 1.0000, 0.0000, 0.60653293, 0.3505922],
                        [0.0000, 0.6076593, 0.33508536, 0.27470887, 1.0000],
                        [0.52727354, 0.0000, 0.05404532, 1.0000, 0.5783674],
                    ],
                    [
                        [0.736647, 0.0000, 0.55139434, 0.44206098, 0.4548782],
                        [0.0000, 0.98196536, 0.54739904, 0.33826917, 0.850068],
                        [0.72412336, 0.6222996, 0.110618, 1.0000, 0.614918],
                        [0.15862459, 0.0699634, 0.66296846, 0.4845066, 0.3705502],
                        [1.0000, 0.46070462, 0.5654058, 0.24897486, 0.0000],
                    ],
                    [
                        [1.0000, 0.88769174, 0.4394987, 0.0000, 0.0000],
                        [0.77483994, 0.99839956, 1.0000, 0.41622213, 1.0000],
                        [0.7488585, 0.0000, 0.01748962, 0.9683316, 0.49795526],
                        [0.9473541, 1.0000, 0.0000, 0.47037587, 1.0000],
                        [1.0000, 0.0000, 0.6804801, 0.9795798, 0.24646705],
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

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.ycbcr_to_rgb, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
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
