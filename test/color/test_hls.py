import math
import warnings

import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import BaseTester
from packaging import version


class TestRgbToHls(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_hls(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_hls(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_hls([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_hls(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_hls(img)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [0.4237059, 0.1935902, 0.8585021, 0.3790484, 0.1389151],
                    [0.5933651, 0.0474544, 0.2801555, 0.1691061, 0.9221829],
                    [0.2351739, 0.5852075, 0.5789326, 0.8411915, 0.5960411],
                    [0.0290176, 0.6459382, 0.8581501, 0.4755400, 0.7735767],
                    [0.9497226, 0.0919441, 0.5462211, 0.7836787, 0.6403612],
                ],
                [
                    [0.2280025, 0.1352853, 0.7999730, 0.6658246, 0.4910861],
                    [0.3499791, 0.1250734, 0.6315800, 0.4785843, 0.8477826],
                    [0.3646359, 0.2415122, 0.5301932, 0.0782518, 0.8710389],
                    [0.6957581, 0.6162295, 0.6259052, 0.1753750, 0.6737530],
                    [0.7678874, 0.9825978, 0.0234877, 0.2485284, 0.8159551],
                ],
                [
                    [0.7330830, 0.9015747, 0.0229067, 0.4280063, 0.5400181],
                    [0.0037299, 0.3259412, 0.3467951, 0.9575506, 0.1525899],
                    [0.9660432, 0.5287710, 0.6654660, 0.3797526, 0.4981400],
                    [0.7422802, 0.9926301, 0.5334370, 0.7852844, 0.4397180],
                    [0.2281681, 0.2560037, 0.5134379, 0.5800887, 0.8685090],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # OpenCV
        expected = torch.tensor(
            [
                [
                    [4.59454770, 4.26846900, 0.97384680, 2.27317070, 3.26934400],
                    [0.61494170, 3.89691880, 2.29297200, 3.77774720, 0.94595980],
                    [4.00329600, 5.40794320, 4.56610100, 5.86935100, 1.81946310],
                    [3.20989560, 4.27144400, 0.29820946, 4.70416550, 0.73408560],
                    [0.78329855, 2.28729030, 5.30166340, 5.63437900, 3.38281500],
                ],
                [
                    [0.48054275, 0.51843000, 0.44070444, 0.52243650, 0.33946657],
                    [0.29854750, 0.18669781, 0.45586777, 0.56332830, 0.53738640],
                    [0.60060860, 0.41335985, 0.59782960, 0.45972168, 0.68458940],
                    [0.38564888, 0.80442977, 0.69579350, 0.48032972, 0.60664740],
                    [0.58894540, 0.53727096, 0.28485440, 0.51610350, 0.75443510],
                ],
                [
                    [0.52553130, 0.79561585, 0.94802250, 0.30024928, 0.59078425],
                    [0.98750657, 0.74582230, 0.38544560, 0.90278864, 0.83178820],
                    [0.91497860, 0.41573380, 0.16817844, 0.82978433, 0.59113250],
                    [0.92475650, 0.96231550, 0.53370523, 0.63488615, 0.42437580],
                    [0.87768690, 0.96239233, 0.91754496, 0.55295944, 0.46453667],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(kornia.color.rgb_to_hls(data), expected)

    def test_nan_rgb_to_hls(self, device, dtype):
        if device != torch.device('cpu') and version.parse(torch.__version__) < version.parse('1.7.0'):
            warnings.warn(
                "This test is not compatible with pytorch < 1.7.0. This message will be removed as soon as we do not "
                "support pytorch 1.6.0. `torch.max()` have a problem in pytorch < 1.7.0 then we cannot get the correct "
                "result. https://github.com/pytorch/pytorch/issues/41781",
                DeprecationWarning,
                stacklevel=2,
            )
            return
        data = torch.ones(2, 3, 5, 5, device=device, dtype=dtype)

        # OpenCV
        expected = torch.cat(
            [
                torch.zeros(2, 1, 5, 5, device=device, dtype=dtype),
                torch.ones(2, 1, 5, 5, device=device, dtype=dtype),
                torch.zeros(2, 1, 5, 5, device=device, dtype=dtype),
            ],
            dim=1,
        )
        self.assert_close(kornia.color.rgb_to_hls(data), expected)

    def test_nan_random_extreme_values(self, device, dtype):
        # generate extreme colors randomly
        ext_rand_slice = (torch.rand((1, 3, 32, 32), dtype=dtype, device=device) >= 0.5).float()
        assert not kornia.color.rgb_to_hls(ext_rand_slice).isnan().any()

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_hls, (img,), raise_exception=True)

    @pytest.mark.jit
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
        op = kornia.color.rgb_to_hls
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToHls().to(device, dtype)
        fcn = kornia.color.rgb_to_hls
        self.assert_close(ops(img), fcn(img))


class TestHlsToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.hls_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.hls_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.hls_to_rgb([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.hls_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.hls_to_rgb(img)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [
                        [0.5513626, 0.8487718, 0.1822479, 0.2851745, 0.2669488],
                        [0.7596772, 0.4565057, 0.6181599, 0.3852497, 0.7746902],
                        [0.5742747, 0.1957062, 0.7530835, 0.2104362, 0.9449323],
                        [0.9918052, 0.2437515, 0.4718738, 0.8502576, 0.1675640],
                        [0.9210159, 0.0538564, 0.5801026, 0.6110542, 0.3768399],
                    ],
                    [
                        [0.4111853, 0.0183454, 0.7832276, 0.2975794, 0.1139528],
                        [0.6207729, 0.1073406, 0.8335325, 0.5700451, 0.2594557],
                        [0.7520493, 0.5097187, 0.4719872, 0.9477938, 0.1640292],
                        [0.8973427, 0.6455371, 0.7567374, 0.3159562, 0.8135307],
                        [0.0855004, 0.6645504, 0.9923756, 0.6209313, 0.2356791],
                    ],
                    [
                        [0.4734681, 0.0422099, 0.7405791, 0.9671807, 0.1793800],
                        [0.8221875, 0.7219887, 0.3627397, 0.4403201, 0.0024084],
                        [0.0803350, 0.9432759, 0.0241543, 0.8292291, 0.7745832],
                        [0.3707901, 0.0851424, 0.5805428, 0.1098685, 0.4238486],
                        [0.1058410, 0.0816052, 0.5792874, 0.9578886, 0.6281684],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        data[:, 0] = 2 * math.pi * data[:, 0]

        # OpenCV
        expected = torch.tensor(
            [
                [
                    [
                        [0.21650219, 0.01911971, 0.91374826, 0.17609520, 0.10979544],
                        [0.65698080, 0.02984191, 0.77314806, 0.38072730, 0.25964087],
                        [0.73213010, 0.81102980, 0.47240910, 0.96834683, 0.29108350],
                        [0.93540700, 0.64780010, 0.61551300, 0.35066980, 0.89171433],
                        [0.09454980, 0.69192480, 0.98795897, 0.25782573, 0.08763295],
                    ],
                    [
                        [0.48587522, 0.01757100, 0.94376480, 0.58539250, 0.13439366],
                        [0.30897713, 0.18483935, 0.80829670, 0.75936294, 0.25883088],
                        [0.75421450, 0.97218925, 0.46058673, 0.99108470, 0.03697497],
                        [0.85927840, 0.67571700, 0.89796180, 0.28124255, 0.89256540],
                        [0.07645091, 0.65486740, 0.99254686, 0.50014400, 0.38372523],
                    ],
                    [
                        [0.60586834, 0.01897625, 0.62269044, 0.00976634, 0.09351197],
                        [0.93256867, 0.14439031, 0.89391685, 0.49867177, 0.26008060],
                        [0.77196840, 0.04724807, 0.48338777, 0.90450300, 0.12093388],
                        [0.86302150, 0.61535730, 0.85029656, 0.34361976, 0.73449594],
                        [0.08502806, 0.63717590, 0.99679226, 0.98403690, 0.16492467],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        f = kornia.color.hls_to_rgb
        self.assert_close(f(data), expected)

        data[:, 0] += 2 * math.pi
        self.assert_close(f(data), expected)

        data[:, 0] -= 4 * math.pi
        self.assert_close(f(data), expected)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.hls_to_rgb, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        if version.parse(torch.__version__) < version.parse('1.7.0'):
            warnings.warn(
                "This test is not compatible with pytorch < 1.7.0. This message will be removed as soon as we do not "
                "support pytorch 1.6.0. `hls_to_rgb()` method for pytorch < 1.7.0 version cannot be compiled with JIT.",
                DeprecationWarning,
                stacklevel=2,
            )
            return
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.hls_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.HlsToRgb().to(device, dtype)
        fcn = kornia.color.hls_to_rgb
        self.assert_close(ops(img), fcn(img))
