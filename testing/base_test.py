from __future__ import annotations

import jax
import keras_core as keras
import numpy as np
import tensorflow as tf
import torch
from keras_core import KerasTensor as Tensor


class BaseTester:
    """Base class for all testers."""

    @staticmethod
    def generate_random_image_data(shape: tuple[int, ...]) -> Tensor:
        """Generate random image data based on the backend.

        Args:
            shape: Shape of the desired image.

        Returns:
            Random image data.
        """
        backend = keras.backend.backend()

        if backend not in ["torch", "numpy", "jax", "tensorflow"]:
            raise NotImplementedError(f"Backend {backend} is not supported.")

        if backend == "numpy":
            key = np.random.RandomState(42)
            return key.uniform(size=shape).astype(np.float32)

        if backend == "jax":
            key = jax.random.PRNGKey(42)
            return jax.random.uniform(key, shape).astype(jax.numpy.float32)

        if backend == "tensorflow":
            return tf.random.uniform(shape, dtype=tf.float32)

        # default backend is torch
        return torch.randint(0, 1, shape, dtype=torch.float32)
