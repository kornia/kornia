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

from typing import List

import numpy as np
import pytest
import torch

import kornia

from testing.base import assert_close


@pytest.mark.parametrize(
    "input_dtype, expected_dtype", [(np.uint8, torch.uint8), (np.float32, torch.float32), (np.float64, torch.float64)]
)
def test_image_to_tensor_keep_dtype(input_dtype, expected_dtype):
    image = np.ones((1, 3, 4, 5), dtype=input_dtype)
    tensor = kornia.image.image_to_tensor(image)
    assert tensor.dtype == expected_dtype


@pytest.mark.parametrize("num_of_images, image_shape", [(2, (4, 3, 1)), (0, (1, 2, 3)), (5, (2, 3, 2, 5))])
def test_list_of_images_to_tensor(num_of_images, image_shape):
    images: List[np.array] = []
    if num_of_images == 0:
        with pytest.raises(ValueError):
            kornia.image.image_list_to_tensor([])
        return
    for _ in range(num_of_images):
        images.append(np.ones(shape=image_shape))
    if len(image_shape) != 3:
        with pytest.raises(ValueError):
            kornia.image.image_list_to_tensor(images)
        return
    tensor = kornia.image.image_list_to_tensor(images)
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
    image = kornia.image.tensor_to_image(tensor)
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
    image = kornia.image.tensor_to_image(tensor, keepdim=True)
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
    tensor = kornia.image.image_to_tensor(image, keepdim=False)
    assert tensor.shape == expected
    assert isinstance(tensor, torch.Tensor)

    to_tensor = kornia.image.ImageToTensor(keepdim=False)
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
    tensor = kornia.image.image_to_tensor(image, keepdim=True)
    assert tensor.shape == expected
    assert isinstance(tensor, torch.Tensor)


def test_tensor_to_image_contiguous(device, dtype):
    tensor = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)

    image = kornia.image.tensor_to_image(tensor)
    assert not image.flags["C_CONTIGUOUS"]

    image = kornia.image.tensor_to_image(tensor, force_contiguous=True)
    assert image.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    "op, kwargs",
    [
        (kornia.filters.in_range, {"lower": (0.2, 0.3, 0.4), "upper": (0.8, 0.7, 0.9)}),
        (kornia.enhance.normalize_min_max, {}),
        (kornia.enhance.posterize, {"bits": 4}),
        (kornia.enhance.sharpness, {"factor": 1.0}),
        (kornia.enhance.equalize, {}),
        (kornia.enhance.equalize_clahe, {"clip_limit": 40.0, "grid_size": (8, 8)}),
        (kornia.enhance.jpeg_codec_differentiable, {}),
    ],
)
def test_perform_keep_shape_image_non_contiguous(device, dtype, op, kwargs):
    tensor = torch.rand(2, 3, 3, 16, 24, device=device, dtype=dtype).transpose(0, 1)
    assert not tensor.is_contiguous()

    if op is kornia.enhance.jpeg_codec_differentiable:
        batch_size = tensor.shape[0] * tensor.shape[1]
        jpeg_quality = torch.linspace(30.0, 90.0, batch_size, device=device, dtype=dtype)

        result = op(tensor, jpeg_quality)

        expected = torch.stack(
            [
                torch.stack(
                    [
                        op(
                            tensor[i, j],
                            jpeg_quality[i * tensor.shape[1] + j : i * tensor.shape[1] + j + 1],
                        )
                        for j in range(tensor.shape[1])
                    ]
                )
                for i in range(tensor.shape[0])
            ]
        )
    else:
        result = op(tensor, **kwargs)

        expected = torch.stack(
            [torch.stack([op(tensor[i, j], **kwargs) for j in range(tensor.shape[1])]) for i in range(tensor.shape[0])]
        )

    assert result.shape == expected.shape
    assert_close(result, expected)


def test_perform_keep_shape_video_non_contiguous(device, dtype):
    tensor = torch.rand(2, 2, 3, 4, 32, 32, device=device, dtype=dtype).transpose(0, 1)
    assert not tensor.is_contiguous()

    result = kornia.enhance.equalize3d(tensor)

    expected = torch.stack(
        [
            torch.stack([kornia.enhance.equalize3d(tensor[i, j]) for j in range(tensor.shape[1])])
            for i in range(tensor.shape[0])
        ]
    )

    assert result.shape == expected.shape
    assert_close(result, expected)
