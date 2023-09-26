from __future__ import annotations

from typing import Any, Literal

import jax
import keras_core as keras
import numpy as np
import tensorflow as tf
import torch

DTYPES = {'uint8', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'}


def load_dtype(
    dtype: Literal['uint8', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'] | Any
) -> Any:
    if isinstance(dtype, str):
        if dtype not in DTYPES:
            raise NotImplementedError(f'Dtype `{dtype}` is not supported!')
    else:
        return dtype

    backend = keras.backend.backend()
    if backend == "torch":
        dtypes = {
            'uint8': torch.uint8,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
        }
    elif backend == "numpy":
        dtypes = {
            'uint8': np.uint8,
            'int8': np.int8,
            'int16': np.int16,
            'int32': np.int32,
            'int64': np.int64,
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
        }
    elif backend == "jax":
        dtypes = {
            'uint8': jax.numpy.uint8,
            'int8': jax.numpy.int8,
            'int16': jax.numpy.int16,
            'int32': jax.numpy.int32,
            'int64': jax.numpy.int64,
            'float16': jax.numpy.float16,
            'float32': jax.numpy.float32,
            'float64': jax.numpy.float64,
        }
    elif backend == "tensorflow":
        dtypes = {
            'uint8': tf.uint8,
            'int8': tf.int8,
            'int16': tf.int16,
            'int32': tf.int32,
            'int64': tf.int64,
            'float16': tf.float16,
            'float32': tf.float32,
            'float64': tf.float64,
        }

    else:
        raise NotImplementedError(f'backend `{backend}` is not supported')

    return dtypes[dtype]
