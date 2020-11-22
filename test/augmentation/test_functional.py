import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
import kornia.augmentation.functional as F
from kornia.constants import pi
from kornia.augmentation import ColorJitter


class TestHorizontalFlipFn:

    def test_random_hflip(self, device, dtype):
        input = torch.tensor([[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 1., 2.]], device=device, dtype=dtype)  # 3 x 4

        expected = torch.tensor([[0., 0., 0., 0.],
                                 [0., 0., 0., 0.],
                                 [2., 1., 0., 0.]], device=device, dtype=dtype)  # 3 x 4

        assert (F.apply_hflip(input[None, None]) == expected).all()

    def test_batch_random_hflip(self, device, dtype):
        batch_size = 5
        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3

        expected = torch.tensor([[[[0., 0., 0.],
                                   [0., 0., 0.],
                                   [1., 1., 0.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3

        input = input.repeat(batch_size, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(batch_size, 3, 1, 1)  # 5 x 3 x 3 x 3
        assert (F.apply_hflip(input) == expected).all()


class TestVerticalFlipFn:

    def test_random_vflip(self, device, dtype):
        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]], device=device, dtype=dtype)  # 3 x 3

        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]], device=device, dtype=dtype)  # 3 x 3

        assert (F.apply_vflip(input[None, None]) == expected).all()

    def test_batch_random_vflip(self, device, dtype):
        batch_size = 5

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3

        expected = torch.tensor([[[[0., 1., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3

        input = input.repeat(batch_size, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(batch_size, 3, 1, 1)  # 5 x 3 x 3 x 3

        assert (F.apply_vflip(input) == expected).all()


class TestColorJitter:

    def test_color_jitter(self):

        jitter_param = {
            'brightness_factor': torch.tensor(1.),
            'contrast_factor': torch.tensor(1.),
            'saturation_factor': torch.tensor(1.),
            'hue_factor': torch.tensor(0.),
            'order': torch.tensor([2, 3, 0, 1])
        }

        input = torch.rand(3, 5, 5)  # 3 x 5 x 5

        expected = input

        assert_allclose(F.apply_color_jitter(input[None], jitter_param), expected, atol=1e-4, rtol=1e-5)

    def test_color_jitter_batch(self):
        batch_size = 2
        jitter_param = {
            'brightness_factor': torch.tensor([1.] * batch_size),
            'contrast_factor': torch.tensor([1.] * batch_size),
            'saturation_factor': torch.tensor([1.] * batch_size),
            'hue_factor': torch.tensor([0.] * batch_size),
            'order': torch.tensor([2, 3, 0, 1])
        }

        input = torch.rand(batch_size, 3, 5, 5)  # 2 x 3 x 5 x 5
        expected = input

        assert_allclose(F.apply_color_jitter(input, jitter_param), expected, atol=1e-4, rtol=1e-5)

    def test_random_brightness(self):
        torch.manual_seed(42)
        jitter_param = {
            'brightness_factor': torch.tensor([1.1529, 1.1660]),
            'contrast_factor': torch.tensor([1., 1.]),
            'hue_factor': torch.tensor([0., 0.]),
            'saturation_factor': torch.tensor([1., 1.]),
            'order': torch.tensor([2, 3, 0, 1])
        }

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]],

                                  [[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]],

                                  [[0.2529, 0.3529, 0.4529],
                                   [0.7529, 0.6529, 0.5529],
                                   [0.8529, 0.9529, 1.0000]]],


                                 [[[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]],

                                  [[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]],

                                  [[0.2660, 0.3660, 0.4660],
                                   [0.7660, 0.6660, 0.5660],
                                   [0.8660, 0.9660, 1.0000]]]])  # 1 x 1 x 3 x 3

        assert_allclose(F.apply_color_jitter(input, jitter_param), expected)

    def test_random_contrast(self):
        torch.manual_seed(42)
        jitter_param = {
            'brightness_factor': torch.tensor([1., 1.]),
            'contrast_factor': torch.tensor([0.9531, 1.1837]),
            'hue_factor': torch.tensor([0., 0.]),
            'saturation_factor': torch.tensor([1., 1.]),
            'order': torch.tensor([2, 3, 0, 1])
        }

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]],

                                  [[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]],

                                  [[0.0953, 0.1906, 0.2859],
                                   [0.5719, 0.4766, 0.3813],
                                   [0.6672, 0.7625, 0.9531]]],


                                 [[[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]],

                                  [[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]],

                                  [[0.1184, 0.2367, 0.3551],
                                   [0.7102, 0.5919, 0.4735],
                                   [0.8286, 0.9470, 1.0000]]]])

        assert_allclose(F.apply_color_jitter(input, jitter_param), expected, atol=1e-4, rtol=1e-5)

    def test_random_saturation(self):
        torch.manual_seed(42)
        jitter_param = {
            'brightness_factor': torch.tensor([1., 1.]),
            'contrast_factor': torch.tensor([1., 1.]),
            'hue_factor': torch.tensor([0., 0.]),
            'saturation_factor': torch.tensor([0.9026, 1.1175]),
            'order': torch.tensor([2, 3, 0, 1])
        }

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[1.8763e-01, 2.5842e-01, 3.3895e-01],
                                   [6.2921e-01, 5.0000e-01, 4.0000e-01],
                                   [7.0974e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 5.2921e-01, 6.0974e-01],
                                   [6.2921e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 1.6816e-01, 2.7790e-01]],

                                  [[6.3895e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 3.1947e-01, 2.1947e-01],
                                   [8.0000e-01, 4.3895e-01, 5.4869e-01]]],


                                 [[[1.1921e-07, 1.2953e-01, 2.5302e-01],
                                   [5.6476e-01, 5.0000e-01, 4.0000e-01],
                                   [6.8825e-01, 8.0000e-01, 1.0000e+00]],

                                  [[1.0000e+00, 4.6476e-01, 5.8825e-01],
                                   [5.6476e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 1.7781e-02, 1.0603e-01]],

                                  [[5.5556e-01, 8.0000e-01, 7.0000e-01],
                                   [9.0000e-01, 2.7651e-01, 1.7651e-01],
                                   [8.0000e-01, 3.5302e-01, 4.4127e-01]]]])

        assert_allclose(F.apply_color_jitter(input, jitter_param), expected, atol=1e-4, rtol=1e-5)

    def test_random_hue(self):
        torch.manual_seed(42)
        jitter_param = {
            'brightness_factor': torch.tensor([1., 1.]),
            'contrast_factor': torch.tensor([1., 1.]),
            'hue_factor': torch.tensor([-0.0438 / 2 / pi, 0.0404 / 2 / pi]),
            'saturation_factor': torch.tensor([1., 1.]),
            'order': torch.tensor([2, 3, 0, 1])
        }
        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]])  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.1000, 0.2000, 0.3000],
                                   [0.6000, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.5251, 0.6167],
                                   [0.6126, 0.3000, 0.2000],
                                   [0.8000, 0.1000, 0.2000]],

                                  [[0.5623, 0.8000, 0.7000],
                                   [0.9000, 0.3084, 0.2084],
                                   [0.7958, 0.4293, 0.5335]]],

                                 [[[0.1000, 0.2000, 0.3000],
                                   [0.6116, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],

                                  [[1.0000, 0.4769, 0.5846],
                                   [0.6000, 0.3077, 0.2077],
                                   [0.7961, 0.1000, 0.2000]],

                                  [[0.6347, 0.8000, 0.7000],
                                   [0.9000, 0.3000, 0.2000],
                                   [0.8000, 0.3730, 0.4692]]]])

        assert_allclose(F.apply_color_jitter(input, jitter_param), expected, atol=1e-4, rtol=1e-5)


class TestRandomGrayscale:

    def test_opencv_true(self, device):
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

        expected = torch.tensor([[[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]])
        expected = expected.to(device)

        assert_allclose(F.apply_grayscale(data[None]), expected)

    def test_opencv_true_batch(self, device):
        batch_size = 4
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
        data = data.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor([[[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]])
        expected = expected.to(device)
        expected = expected.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        assert_allclose(F.apply_grayscale(data), expected)


class TestRandomRectangleEarasing:

    def test_rectangle_erasing1(self, device):
        inputs = torch.ones(1, 1, 10, 10).to(device)
        rect_params = {
            "widths": torch.tensor([5]),
            "heights": torch.tensor([5]),
            "xs": torch.tensor([5]),
            "ys": torch.tensor([5]),
            "values": torch.tensor([0.])
        }
        expected = torch.tensor([[[
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]
        ]]]).to(device)
        assert_allclose(F.apply_erase_rectangles(inputs, rect_params), expected)

    def test_rectangle_erasing2(self, device):
        inputs = torch.ones(3, 3, 3, 3).to(device)
        rect_params = {
            "widths": torch.tensor([3, 2, 1]),
            "heights": torch.tensor([3, 2, 1]),
            "xs": torch.tensor([0, 1, 2]),
            "ys": torch.tensor([0, 1, 2]),
            "values": torch.tensor([0., 0., 0.])
        }
        expected = torch.tensor(
            [[[[0., 0., 0.],
               [0., 0., 0.],
                [0., 0., 0.]],

                [[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.]],

                [[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.]]],

                [[[1., 1., 1.],
                  [1., 0., 0.],
                    [1., 0., 0.]],

                 [[1., 1., 1.],
                  [1., 0., 0.],
                    [1., 0., 0.]],

                 [[1., 1., 1.],
                  [1., 0., 0.],
                    [1., 0., 0.]]],

                [[[1., 1., 1.],
                  [1., 1., 1.],
                    [1., 1., 0.]],

                 [[1., 1., 1.],
                  [1., 1., 1.],
                    [1., 1., 0.]],

                 [[1., 1., 1.],
                  [1., 1., 1.],
                    [1., 1., 0.]]]]
        ).to(device)

        assert_allclose(F.apply_erase_rectangles(inputs, rect_params), expected)
