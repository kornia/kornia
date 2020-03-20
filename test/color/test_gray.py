import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToGrayscale:
    def test_rgb_to_grayscale(self, device):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width).to(device)
        assert kornia.rgb_to_grayscale(img).shape == (1, height, width)

    def test_rgb_to_grayscale_batch(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width).to(device)
        assert kornia.rgb_to_grayscale(img).shape == \
            (batch_size, 1, height, width)

    def test_opencv(self, device):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]])
        data = data.to(device)

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor([[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                 [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                 [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                 [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                 [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]])
        expected = expected.to(device)

        img_gray = kornia.rgb_to_grayscale(data)
        assert_allclose(img_gray, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.rgb_to_grayscale, (img,), raise_exception=True)

    def test_jit(self, device):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        op = kornia.rgb_to_grayscale
        op_jit = kornia.jit.rgb_to_grayscale
        assert_allclose(op(img), op_jit(img))

    def test_jit_trace(self, device):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        gray = kornia.color.RgbToGrayscale()
        gray_traced = torch.jit.trace(kornia.color.RgbToGrayscale(), img)
        assert_allclose(gray(img), gray_traced(img))


class TestBgrToGrayscale:
    def test_bgr_to_grayscale(self, device):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width).to(device)
        assert kornia.bgr_to_grayscale(img).shape == (1, height, width)

    def test_bgr_to_grayscale_batch(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width).to(device)
        assert kornia.bgr_to_grayscale(img).shape == \
            (batch_size, 1, height, width)

    # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    def test_opencv(self, device):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]])
        data = data.to(device)

        expected = torch.tensor([[0.4485849, 0.8233618, 0.6262833, 0.6218331, 0.6341921],
                                 [0.3200093, 0.4340172, 0.7107211, 0.5454938, 0.2801398],
                                 [0.6149265, 0.7018101, 0.3503231, 0.4891168, 0.5292346],
                                 [0.5096100, 0.4336508, 0.6704276, 0.4525143, 0.2134447],
                                 [0.7878902, 0.6494595, 0.5211386, 0.6623823, 0.6660464]])
        expected = expected.to(device)

        img_gray = kornia.bgr_to_grayscale(data)
        assert_allclose(img_gray, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.bgr_to_grayscale, (img,), raise_exception=True)

    def test_module(self, device):
        data = torch.tensor([[[[100., 73.],
                               [200., 22.]],

                              [[50., 10.],
                               [148, 14, ]],

                              [[225., 255.],
                               [48., 8.]]]])

        data = data.to(device)

        assert_allclose(kornia.bgr_to_grayscale(data / 255), kornia.color.BgrToGrayscale()(data / 255))

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width).to(device)
        gray = kornia.color.BgrToGrayscale()
        gray_traced = torch.jit.trace(kornia.color.BgrToGrayscale(), img)
        assert_allclose(gray(img), gray_traced(img))
