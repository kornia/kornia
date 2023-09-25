import jax
import keras_core as keras
import numpy as np
import pytest
import tensorflow as tf
import torch

from kornia.color.gray import grayscale_from_rgb


# TODO: make this a fixture
def genrate_image_data(channel_first: bool = False):
    if channel_first:
        shape = (3, 4, 5)
    else:
        shape = (4, 5, 3)

    backend = keras.backend.backend()

    if backend == "torch":
        return torch.randint(0, 1, shape, dtype=torch.float32)
    if backend == "numpy":
        key = np.random.RandomState(42)
        return key.uniform(size=shape).astype(np.float32)
    if backend == "jax":
        key = jax.random.PRNGKey(42)
        return jax.random.uniform(key, shape).astype(jax.numpy.float32)
    if backend == "tensorflow":
        return tf.random.uniform(shape, dtype=tf.float32)

    return None


class TestColorGray:
    @pytest.mark.parametrize("channel_first", [False, True])
    def test_channels(self, channel_first):
        image_rgb = genrate_image_data(channel_first)
        channels_axis = -3 if channel_first else -1
        image_gray = grayscale_from_rgb(image_rgb, channels_axis=channels_axis)
        assert image_gray.shape == (1, 4, 5) if channel_first else (4, 5, 1)
