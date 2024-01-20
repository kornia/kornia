import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils


class TestRGBShift:
    def test_rgb_shift_no_shift(self, device, dtype):
        r_shift, g_shift, b_shift = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        image = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        expected = image
        shifted = kornia.enhance.shift_rgb(image, r_shift, g_shift, b_shift)

        utils.assert_close(shifted, expected)

    def test_rgb_shift_all_zeros(self, device, dtype):
        r_shift, g_shift, b_shift = torch.Tensor([-0.1]), torch.Tensor([-0.1]), torch.Tensor([-0.1])
        image = torch.zeros(2, 3, 5, 5, device=device, dtype=dtype)
        expected = image
        shifted = kornia.enhance.shift_rgb(image, r_shift, g_shift, b_shift)

        utils.assert_close(shifted, expected)

    def test_rgb_shift_all_ones(self, device, dtype):
        r_shift, g_shift, b_shift = torch.Tensor([1]), torch.Tensor([1]), torch.Tensor([1])
        image = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        expected = torch.ones(2, 3, 5, 5, device=device, dtype=dtype)
        shifted = kornia.enhance.shift_rgb(image, r_shift, g_shift, b_shift)

        utils.assert_close(shifted, expected)

    def test_rgb_shift_invalid_parameter_shape(self, device, dtype):
        r_shift, g_shift, b_shift = torch.Tensor([0.5]), torch.Tensor([0.5]), torch.Tensor([0.5])
        image = torch.randn(3, 3, device=device, dtype=dtype)
        with pytest.raises(TypeError):
            kornia.enhance.shift_rgb(image, r_shift, g_shift, b_shift)

    def test_rgb_shift_gradcheck(self, device, dtype):
        r_shift, g_shift, b_shift = torch.Tensor([0.4]), torch.Tensor([0.5]), torch.Tensor([0.2])
        image = torch.randn(2, 3, 5, 5, device=device, dtype=dtype)
        image = utils.tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(
            kornia.enhance.shift_rgb, (image, r_shift, g_shift, b_shift), raise_exception=True, fast_mode=True
        )

    def test_rgb_shift(self, device, dtype):
        r_shift, g_shift, b_shift = torch.Tensor([0.1]), torch.Tensor([0.3]), torch.Tensor([-0.3])
        image = torch.tensor(
            [[[[0.2, 0.0]], [[0.3, 0.5]], [[0.4, 0.7]]], [[[0.2, 0.7]], [[0.0, 0.8]], [[0.2, 0.3]]]],
            device=device,
            dtype=dtype,
        )
        shifted = kornia.enhance.shift_rgb(image, r_shift, g_shift, b_shift)
        expected = torch.tensor(
            [[[[0.3, 0.1]], [[0.6, 0.8]], [[0.1, 0.4]]], [[[0.3, 0.8]], [[0.3, 1.0]], [[0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )

        utils.assert_close(shifted, expected)
