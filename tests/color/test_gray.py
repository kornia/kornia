# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.core._compat import torch_version

from testing.base import BaseTester, assert_close


class TestGrayscaleToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 1, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.grayscale_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        assert kornia.color.grayscale_to_rgb(img).shape == (batch_size, 3, height, width)

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError):
            assert kornia.color.grayscale_to_rgb([0.0])

        img = torch.ones(1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.grayscale_to_rgb(img)

        img = torch.ones(2, 1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.grayscale_to_rgb(img)

        img = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.grayscale_to_rgb(img)

    def test_opencv(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # Output data generated with OpenCV 4.5.2: cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        expected = torch.tensor(
            [
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ],
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ],
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        img_rgb = kornia.color.grayscale_to_rgb(data)
        assert_close(img_rgb, expected)

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.grayscale_to_rgb, (img,), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.grayscale_to_rgb
        op_optimized = torch_optimizer(op)
        assert_close(op(img), op_optimized(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        gray_ops = kornia.color.GrayscaleToRgb().to(device, dtype)
        gray_fcn = kornia.color.grayscale_to_rgb
        assert_close(gray_ops(img), gray_fcn(img))


class TestRgbToGrayscale(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        out = kornia.color.rgb_to_grayscale(img)
        assert out.device == img.device
        assert out.dtype == img.dtype

    def test_smoke_byte(self, device):
        C, H, W = 3, 4, 5
        img = torch.randint(0, 255, (C, H, W), device=device, dtype=torch.uint8)
        out = kornia.color.rgb_to_grayscale(img)
        assert out.device == img.device
        assert out.dtype == img.dtype

    def test_uint8_primaries(self, device):
        white = torch.full((3, 1, 1), 255, device=device, dtype=torch.uint8)
        red = torch.tensor([[[255]], [[0]], [[0]]], device=device, dtype=torch.uint8)
        green = torch.tensor([[[0]], [[255]], [[0]]], device=device, dtype=torch.uint8)
        blue = torch.tensor([[[0]], [[0]], [[255]]], device=device, dtype=torch.uint8)
        assert kornia.color.rgb_to_grayscale(white).item() == 255
        assert kornia.color.rgb_to_grayscale(red).item() == 76
        assert kornia.color.rgb_to_grayscale(green).item() == 150
        assert kornia.color.rgb_to_grayscale(blue).item() == 29

    def test_uint8_opencv(self, device):
        # Generated with OpenCV 4.10.0 and 5.0.0 (identical):
        #   cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY) on the HxWx3 rows below.
        # (12, 250, 9) is 151 in OpenCV but 152 with the 8-bit weights 76/150/29 over 255; (0, 1, 1) and
        # (100, 150, 50) need round-to-nearest rather than floor.
        rgb = [
            [[[255, 255, 255], [255, 0, 0], [0, 255, 0]], [[0, 0, 255], [0, 1, 1], [12, 250, 9]]],
            [[[100, 150, 50], [33, 66, 99], [12, 0, 8]], [[37, 14, 37], [0, 12, 4], [12, 37, 28]]],
        ]
        expected = [[[[255, 76, 150], [29, 1, 151]]], [[[124, 60, 5], [23, 8, 28]]]]
        img = torch.tensor(rgb, device=device, dtype=torch.uint8).permute(0, 3, 1, 2)
        out = kornia.color.rgb_to_grayscale(img)
        assert out.shape == (2, 1, 2, 3)
        assert out.dtype == torch.uint8
        assert torch.equal(out, torch.tensor(expected, device=device, dtype=torch.uint8))

    @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 3, height, width, device=device, dtype=dtype)
        assert kornia.color.rgb_to_grayscale(img).shape == (batch_size, 1, height, width)

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError):
            assert kornia.color.rgb_to_grayscale([0.0])

        img = torch.ones(1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.rgb_to_grayscale(img)

        img = torch.ones(2, 1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.rgb_to_grayscale(img)

        img = torch.ones(3, 1, 1, device=device, dtype=dtype)
        rgb_weights = torch.tensor([0.2, 0.8])
        with pytest.raises(ValueError):
            assert kornia.color.rgb_to_grayscale(img, rgb_weights=rgb_weights)

    def test_unsupported_dtype_raises(self, device):
        # int32 is not uint8 and not a float dtype -> TypeError
        img = torch.ones(3, 4, 4, device=device, dtype=torch.int32)
        with pytest.raises(TypeError, match="Unknown data type"):
            kornia.color.rgb_to_grayscale(img)

    def test_opencv(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                    [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                    [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                    [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                    [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593],
                ],
                [
                    [0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                    [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                    [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                    [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                    [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426],
                ],
                [
                    [0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                    [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                    [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                    [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                    [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor(
            [
                [
                    [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                    [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                    [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                    [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                    [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        img_gray = kornia.color.rgb_to_grayscale(data)
        assert_close(img_gray, expected)

    def test_custom_rgb_weights(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)

        rgb_weights = torch.tensor([0.5, 0.25, 0.25])
        img_gray = kornia.color.rgb_to_grayscale(img, rgb_weights=rgb_weights)
        assert img_gray.device == device
        assert img_gray.dtype == dtype

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_grayscale, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version() in {"2.0.0", "2.0.1"}, reason="Not working on 2.0")
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: investigate the problem with dynamo
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_grayscale
        op_optimized = torch_optimizer(op)
        assert_close(op(img), op_optimized(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        gray_ops = kornia.color.RgbToGrayscale().to(device, dtype)
        gray_fcn = kornia.color.rgb_to_grayscale
        assert_close(gray_ops(img), gray_fcn(img))

    def test_module_default_weights_are_the_functional_defaults_5109(self, device, dtype):
        # The module stored float32 default weights and passed them explicitly, so rgb_to_grayscale cast them to the
        # image: a float64 image got float32-rounded weights (up to 1.2e-8 off). It now leaves None to the functional,
        # so the two outputs are identical (#5109).
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        assert torch.equal(kornia.color.RgbToGrayscale()(img), kornia.color.rgb_to_grayscale(img))
        rgb_weights = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=dtype)
        out = kornia.color.RgbToGrayscale(rgb_weights)(img)
        assert torch.equal(out, kornia.color.rgb_to_grayscale(img, rgb_weights=rgb_weights))

    def test_module_uint8_is_not_all_zeros_5109(self, device):
        # The float32 default weights cast to uint8 were [0, 0, 0], so every uint8 image came back all zeros (#5109).
        img = torch.arange(120, device=device, dtype=torch.uint8).reshape(2, 3, 4, 5)
        out = kornia.color.RgbToGrayscale()(img)
        assert out.dtype == torch.uint8
        assert torch.equal(out, kornia.color.rgb_to_grayscale(img))
        assert out.count_nonzero() > 0


class TestBgrToGrayscale(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert kornia.color.bgr_to_grayscale(img) is not None

    @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 3, height, width, device=device, dtype=dtype)
        assert kornia.color.bgr_to_grayscale(img).shape == (batch_size, 1, height, width)

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError):
            assert kornia.color.bgr_to_grayscale([0.0])

        img = torch.ones(1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.bgr_to_grayscale(img)

        img = torch.ones(2, 1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.bgr_to_grayscale(img)

    def test_opencv(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                    [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                    [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                    [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                    [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593],
                ],
                [
                    [0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                    [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                    [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                    [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                    [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426],
                ],
                [
                    [0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                    [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                    [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                    [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                    [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        expected = torch.tensor(
            [
                [
                    [0.4485849, 0.8233618, 0.6262833, 0.6218331, 0.6341921],
                    [0.3200093, 0.4340172, 0.7107211, 0.5454938, 0.2801398],
                    [0.6149265, 0.7018101, 0.3503231, 0.4891168, 0.5292346],
                    [0.5096100, 0.4336508, 0.6704276, 0.4525143, 0.2134447],
                    [0.7878902, 0.6494595, 0.5211386, 0.6623823, 0.6660464],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        img_gray = kornia.color.bgr_to_grayscale(data)
        assert_close(img_gray, expected)

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.bgr_to_grayscale, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version() in {"2.0.0", "2.0.1"}, reason="Not working on 2.0")
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: investigate the problem with dynamo
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_grayscale
        op_optimized = torch_optimizer(op)
        assert_close(op(img), op_optimized(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        gray_ops = kornia.color.BgrToGrayscale().to(device, dtype)
        gray_fcn = kornia.color.bgr_to_grayscale
        assert_close(gray_ops(img), gray_fcn(img))
