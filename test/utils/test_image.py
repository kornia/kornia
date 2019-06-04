import pytest
import numpy as np

import torch
import kornia as kornia
import kornia.testing as utils  # test utils


@pytest.mark.parametrize("batch_shape",
                         [(4, 4), (1, 4, 4), (3, 4, 4), ])
def test_tensor_to_image(batch_shape):
    tensor = torch.ones(batch_shape)
    image = kornia.utils.tensor_to_image(tensor)
    assert image.shape[:2] == batch_shape[-2:]
    assert isinstance(image, np.ndarray)


@pytest.mark.parametrize("batch_shape",
                         [(4, 4), (4, 4, 1), (4, 4, 3), ])
def test_image_to_tensor(batch_shape):
    image = np.ones(batch_shape)
    tensor = kornia.utils.image_to_tensor(image)
    assert tensor.shape[-2:] == batch_shape[:2]
    assert isinstance(tensor, torch.Tensor)
