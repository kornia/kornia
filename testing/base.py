from __future__ import annotations

import jax
import keras_core as keras
import numpy as np
import tensorflow as tf
import torch
from keras_core import KerasTensor as Tensor

from testing.dtypes import load_dtype


class BaseTester:
    """Base class for all testers."""

    @staticmethod
    def load_shape(B: int | None, W: int, H: int, C: int) -> tuple[int, ...]:
        return load_shape(B, W, H, C)

    @staticmethod
    def generate_random_image(
        B: int | None, W: int, H: int, C: int = 3, seed: int = 42, dtype: str = 'float32'
    ) -> Tensor:
        shape = load_shape(B, W, H, C)
        return generate_random_tensor(shape, seed, dtype)


def generate_random_tensor(shape: tuple[int, ...], seed: int = 42, dtype: str = 'float32') -> Tensor:
    """Generate random tensor data based on the backend.

    Args:
        shape: Shape of the desired tensor.

    Returns:
        Random tensor data.
    """
    backend = keras.backend.backend()

    dtype = load_dtype(dtype)

    if backend == "torch":
        torch.manual_seed(seed)
        return torch.rand(shape, dtype=dtype)

    elif backend == "numpy":
        key = np.random.RandomState(seed)
        return key.uniform(size=shape).astype(dtype)

    elif backend == "jax":
        key = jax.random.PRNGKey(seed)
        return jax.random.uniform(key, shape).astype(dtype)

    elif backend == "tensorflow":
        tf.random.set_seed(seed)
        return tf.random.uniform(shape, dtype=dtype)

    else:
        raise NotImplementedError(f"Backend {backend} is not supported.")


def load_shape(B: int | None, W: int, H: int, C: int) -> tuple[int, ...]:
    # TODO: add support to prefix dims
    image_data_format = keras.backend.image_data_format()
    if image_data_format == 'channels_last':
        shape = (B, W, H, C) if isinstance(B, int) else (W, H, C)
    elif image_data_format == 'channels_first':
        shape = (B, C, W, H) if isinstance(B, int) else (C, W, H)
    else:
        raise NotImplementedError('image data format is not channels last or first.')

    return shape
