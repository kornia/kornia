from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

import kornia
from kornia.testing import assert_close


@pytest.mark.parametrize(
    "input_dtype, expected_dtype", [(np.uint8, torch.uint8), (np.float32, torch.float32), (np.float64, torch.float64)]
)
def test_image_to_tensor_keep_dtype(input_dtype, expected_dtype):
    image = np.ones((1, 3, 4, 5), dtype=input_dtype)
    tensor = kornia.image_to_tensor(image)
    assert tensor.dtype == expected_dtype


@pytest.mark.parametrize("num_of_images, image_shape", [(2, (4, 3, 1)), (0, (1, 2, 3)), (5, (2, 3, 2, 5))])
def test_list_of_images_to_tensor(num_of_images, image_shape):
    images: list[np.array] = []
    if num_of_images == 0:
        with pytest.raises(ValueError):
            kornia.utils.image_list_to_tensor([])
        return
    for _ in range(num_of_images):
        images.append(np.ones(shape=image_shape))
    if len(image_shape) != 3:
        with pytest.raises(ValueError):
            kornia.utils.image_list_to_tensor(images)
        return
    tensor = kornia.utils.image_list_to_tensor(images)
    assert tensor.shape == (num_of_images, image_shape[-1], image_shape[-3], image_shape[-2])


@pytest.mark.parametrize(
    "input_shape, expected",
    [
        ((4, 4), (4, 4)),
        ((1, 4, 4), (4, 4)),
        ((1, 1, 4, 4), (4, 4)),
        ((3, 4, 4), (4, 4, 3)),
        ((2, 3, 4, 4), (2, 4, 4, 3)),
        ((1, 3, 4, 4), (4, 4, 3)),
    ],
)
def test_tensor_to_image(device, input_shape, expected):
    tensor = torch.ones(input_shape).to(device)
    image = kornia.utils.tensor_to_image(tensor)
    assert image.shape == expected
    assert isinstance(image, np.ndarray)


@pytest.mark.parametrize(
    "input_shape, expected",
    [
        ((4, 4), (4, 4)),
        ((1, 4, 4), (4, 4)),
        ((1, 1, 4, 4), (1, 4, 4)),
        ((3, 4, 4), (4, 4, 3)),
        ((2, 3, 4, 4), (2, 4, 4, 3)),
        ((1, 3, 4, 4), (1, 4, 4, 3)),
    ],
)
def test_tensor_to_image_keepdim(device, input_shape, expected):
    tensor = torch.ones(input_shape).to(device)
    image = kornia.utils.tensor_to_image(tensor, keepdim=True)
    assert image.shape == expected
    assert isinstance(image, np.ndarray)


@pytest.mark.parametrize(
    "input_shape, expected",
    [
        ((4, 4), (1, 1, 4, 4)),
        ((1, 4, 4), (1, 4, 1, 4)),
        ((2, 3, 4), (1, 4, 2, 3)),
        ((4, 4, 3), (1, 3, 4, 4)),
        ((2, 4, 4, 3), (2, 3, 4, 4)),
        ((1, 4, 4, 3), (1, 3, 4, 4)),
    ],
)
def test_image_to_tensor(input_shape, expected):
    image = np.ones(input_shape)
    tensor = kornia.utils.image_to_tensor(image, keepdim=False)
    assert tensor.shape == expected
    assert isinstance(tensor, torch.Tensor)

    to_tensor = kornia.utils.ImageToTensor(keepdim=False)
    assert_close(tensor, to_tensor(image))


@pytest.mark.parametrize(
    "input_shape, expected",
    [
        ((4, 4), (1, 4, 4)),
        ((1, 4, 4), (4, 1, 4)),
        ((2, 3, 4), (4, 2, 3)),
        ((4, 4, 3), (3, 4, 4)),
        ((2, 4, 4, 3), (2, 3, 4, 4)),
        ((1, 4, 4, 3), (1, 3, 4, 4)),
    ],
)
def test_image_to_tensor_keepdim(input_shape, expected):
    image = np.ones(input_shape)
    tensor = kornia.utils.image_to_tensor(image, keepdim=True)
    assert tensor.shape == expected
    assert isinstance(tensor, torch.Tensor)
