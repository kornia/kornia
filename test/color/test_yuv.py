import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbYuvConversion:
    # Parameterize for CHW and NCHW shapes
    @pytest.mark.parametrize('shape', ((3, 4, 5), (2, 3, 4, 5)))
    # RGB to YUV and YUV to RGB should be inverse operations
    def test_inverse_operations(self, shape):
        input = torch.rand(*shape)
        yuv_to_rgb_converter = kornia.color.YuvToRgb()
        rgb_to_yuv_converter = kornia.color.RgbToYuv()

        assert_allclose(input, yuv_to_rgb_converter(rgb_to_yuv_converter(input)), rtol=0.005, atol=0.005)
        assert_allclose(input, rgb_to_yuv_converter(yuv_to_rgb_converter(input)), rtol=0.005, atol=0.005)

    def test_gradcheck(self):

        # prepare input data
        data = torch.tensor([[[0.1, 0.2],
                              [0.1, 0.1]],

                             [[0.2, 0.5],
                              [0.4, 0.2]],

                             [[0.3, 0.3],
                              [0.5, 0.5]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.YuvToRgb(), (data,),
                         raise_exception=True)
        assert gradcheck(kornia.color.RgbToYuv(), (data,),
                         raise_exception=True)

    def test_rgb_to_yuv_shape(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert kornia.rgb_to_yuv(img).shape == (channels, height, width)

    def test_rgb_to_yuv_batch_shape(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert kornia.rgb_to_yuv(img).shape == \
            (batch_size, channels, height, width)

    def test_yuv_to_rgb_shape(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert kornia.yuv_to_rgb(img).shape == (channels, height, width)

    def test_yuv_to_rgb_batch_shape(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert kornia.yuv_to_rgb(img).shape == \
            (batch_size, channels, height, width)

    def test_rgb_to_yuv_type(self):
        with pytest.raises(TypeError):
            out = kornia.rgb_to_yuv(1)

    def test_yuv_to_rbg_type(self):
        with pytest.raises(TypeError):
            out = kornia.yuv_to_rgb(1)

    @pytest.mark.parametrize("bad_input_shapes", [([2, 2],), ([3, 3, 3, 3, 3],), ([2, 2, 2],), ([2, 2, 2, 2],)])
    def test_rgb_to_yuv_shape(self, bad_input_shapes):
        with pytest.raises(ValueError):
            out = kornia.rgb_to_yuv(torch.ones(*bad_input_shapes))

    @pytest.mark.parametrize("bad_input_shapes", [([2, 2],), ([3, 3, 3, 3, 3],), ([2, 2, 2],), ([2, 2, 2, 2],)])
    def test_yuv_to_rbg_shape(self, bad_input_shapes):
        with pytest.raises(ValueError):
            out = kornia.yuv_to_rgb(torch.ones(*bad_input_shapes))

    # TODO add cv2 comparision test
