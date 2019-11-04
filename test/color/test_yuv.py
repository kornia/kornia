import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbYuvConversion:
    # Parameterize for CHW and NCHW shapes
    @pytest.mark.parametrize('shape', ((3,4,5), (2,3,4,5)))
    # RGB to YUV and YUV to RGB should be inverse operations
    def test_inverse_operations(self, shape):
        input = torch.rand(*shape)
        yuv_to_rgb_converter = kornia.color.YuvToRgb()
        rgb_to_yuv_converter = kornia.color.RgbToYuv()

        assert_allclose(input, yuv_to_rgb_converter(rgb_to_yuv_converter(input)))
        assert_allclose(input, rgb_to_yuv_converter(yuv_to_rgb_converter(input)))
