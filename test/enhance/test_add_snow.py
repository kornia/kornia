import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils


class TestAddSnow:
    def test_add_snow_no_snow(self, device, dtype):
        snow_coef, brightness_coef = -10, 1
        image = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        expected = image
        shifted = kornia.enhance.add_snow(image, snow_coef, brightness_coef)

        utils.assert_close(shifted, expected)

    def test_add_snow_all_zeros(self, device, dtype):
        snow_coef, brightness_coef = 2, 0
        image = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        expected = torch.zeros(2, 3, 5, 5, device=device, dtype=dtype)
        shifted = kornia.enhance.add_snow(image, snow_coef, brightness_coef)

        utils.assert_close(shifted, expected)

    def test_add_snow_all_ones(self, device, dtype):
        snow_coef, brightness_coef = 2, 1e5
        image = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        expected = torch.ones(2, 3, 5, 5, device=device, dtype=dtype)
        shifted = kornia.enhance.add_snow(image, snow_coef, brightness_coef)

        utils.assert_close(shifted, expected)

    def test_add_snow_invalid_parameter_shape(self, device, dtype):
        snow_coef, brightness_coef = 2, 1e5
        image = torch.randn(3, 3, device=device, dtype=dtype)
        with pytest.raises(TypeError):
            kornia.enhance.add_snow(image, snow_coef, brightness_coef)

    def test_add_snow_gradcheck(self, device, dtype):
        snow_coef, brightness_coef = 2, 1e5
        image = torch.randn(2, 3, 5, 5, device=device, dtype=dtype)
        image = utils.tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(kornia.enhance.add_snow, (image, snow_coef, brightness_coef), eps=1e-3, raise_exception=True)

    def test_add_snow(self, device, dtype):
        snow_coef, brightness_coef = 0, 2
        image = torch.tensor([[[[0.2, 0.0]], [[0.3, 0.5]], [[0.4, 0.7]]]], device=device, dtype=dtype)
        shifted = kornia.enhance.add_snow(image, snow_coef, brightness_coef)
        expected = torch.tensor([[[[0.4667, 0.]], [[0.6, 0.5]], [[0.7333, 0.7]]]], device=device, dtype=dtype)

        utils.assert_close(shifted, expected)
