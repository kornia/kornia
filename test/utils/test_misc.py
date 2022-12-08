import kornia
import pytest
import torch


@pytest.mark.parametrize(
    "input_shape, keep_last_dims, expected_shape",
    [
        ((2, 2, 2), 1, (4, 2)),
        ((2, 2, 2), 2, (2, 2, 2)),
        ((2, 2, 2, 2), 1, (8, 2)),
        ((2, 2, 2, 2), 2, (4, 2, 2)),
        ((2, 2, 2, 2), 3, (2, 2, 2, 2)),
        ((2, 2, 2, 2, 2), 1, (16, 2)),
        ((2, 2, 2, 2, 2), 2, (8, 2, 2)),
        ((2, 2, 2, 2, 2), 3, (4, 2, 2, 2)),
        ((2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2)),
    ],
)
def test_reduce_first_dims(input_shape, keep_last_dims, expected_shape):
    image = torch.rand(*input_shape)
    tensor = kornia.utils.misc.reduce_first_dims(image, keep_last_dims=keep_last_dims, return_shape=False)
    assert tuple(tensor.shape) == tuple(expected_shape)
