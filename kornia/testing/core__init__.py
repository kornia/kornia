"""The testing package contains testing-specific utilities."""
import importlib
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, TypeVar, Union

# from kornia.core import Device, Dtype, IntegratedTensor, eye, tensor
from kornia.core import IntegratedTensor, eye

import keras_core as keras

__all__ = ["tensor_to_gradcheck_var", "create_eye_batch", "xla_is_available", "assert_close"]

def is_mps_tensor_safe(x: IntegratedTensor) -> bool:
    """Return whether tensor is on MPS device."""
    return 'mps' in str(x.device)

def create_eye_batch(batch_size: int, eye_size_N: int, eye_size_M: Union[None, int], k: int = 0) -> IntegratedTensor:
    """Create a batch of identity matrices of shape Bx3x3."""
    tensor = eye(N=eye_size_N, M=eye_size_M, k=k)
    tensor = keras.ops.expand_dims(tensor, axis=0)
    return keras.ops.tile(tensor, (batch_size, 1, 1))
