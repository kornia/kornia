import pytest
import numpy as np

import torch
import kornia as kornia
import kornia.testing as utils  # test utils
from test.common import device

@pytest.mark.parametrize("input_dtype, expected_dtype",
                         [(np.uint8, torch.uint8),
                          (np.float32, torch.float32),
                          (np.float64, torch.float64), ])
def test_image_to_tensor_keep_dtype(input_dtype, expected_dtype):
    image = np.ones((1, 3, 4, 5), dtype=input_dtype)
    tensor = kornia.image_to_tensor(image)
    assert tensor.dtype == expected_dtype

@pytest.mark.parametrize("input_shape, expected",
                         [((4, 4), (4, 4)),
                          ((1, 4, 4), (4, 4)),
                          ((1, 1, 4, 4), (4, 4)),
                          ((3, 4, 4), (4, 4, 3)),
                          ((2, 3, 4, 4), (2, 4, 4, 3)),
                          ((1, 3, 4, 4), (4, 4, 3)), ])
def test_tensor_to_image(device, input_shape, expected):
    tensor = torch.ones(input_shape).to(device)
    image = kornia.utils.tensor_to_image(tensor)
    assert image.shape == expected
    assert isinstance(image, np.ndarray)


@pytest.mark.parametrize("input_shape, expected",
                         [((4, 4), (1, 1, 4, 4)),
                          ((1, 4, 4), (1, 4, 1, 4)),
                          ((2, 3, 4), (1, 4, 2, 3)),
                          ((4, 4, 3), (1, 3, 4, 4)),
                          ((2, 4, 4, 3), (2, 3, 4, 4)),
                          ((1, 4, 4, 3), (1, 3, 4, 4)), ])
def test_image_to_tensor(input_shape, expected):
    image = np.ones(input_shape)
    tensor = kornia.utils.image_to_tensor(image, keepdim=False)
    assert tensor.shape == expected
    assert isinstance(tensor, torch.Tensor)


@pytest.mark.parametrize("input_shape, expected",
                         [((4, 4), (1, 4, 4)),
                          ((1, 4, 4), (4, 1, 4)),
                          ((2, 3, 4), (4, 2, 3)),
                          ((4, 4, 3), (3, 4, 4)),
                          ((2, 4, 4, 3), (2, 3, 4, 4)),
                          ((1, 4, 4, 3), (1, 3, 4, 4)), ])
def test_image_to_tensor_keepdim(input_shape, expected):
    image = np.ones(input_shape)
    tensor = kornia.utils.image_to_tensor(image, keepdim=True)
    assert tensor.shape == expected
    assert isinstance(tensor, torch.Tensor)
