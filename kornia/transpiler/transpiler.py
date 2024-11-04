"""Module for transpiling Kornia to other frameworks."""

from types import ModuleType

import kornia
from kornia.core.external import ivy


def to_jax() -> ModuleType:
    """Convert Kornia to JAX.

    Transpiles the Kornia library to JAX using [ivy](https://github.com/ivy-llc/ivy). The transpilation process
    occurs lazily, so the transpilation on a given kornia function/class will only occur when it's called or
    instantiated for the first time. This will make any functions/classes slow when being used for the first time,
    but any subsequent uses should be as fast as expected.

    Return:
        The Kornia library transpiled to JAX

    Example:

    .. highlight:: python
    .. code-block:: python

        import kornia
        jax_kornia = kornia.to_jax()
        import jax
        input = jax.random.normal(jax.random.key(42), shape=(2, 3, 4, 5))
        gray = jax_kornia.color.gray.rgb_to_grayscale(input)
    """
    return ivy.transpile(
        kornia,
        source="torch",
        target="jax",
    )  # type: ignore


def to_numpy() -> ModuleType:
    """Convert Kornia to NumPy.

    Transpiles the Kornia library to NumPy using [ivy](https://github.com/ivy-llc/ivy). The transpilation process
    occurs lazily, so the transpilation on a given kornia function/class will only occur when it's called or
    instantiated for the first time. This will make any functions/classes slow when being used for the first time,
    but any subsequent uses should be as fast as expected.

    Return:
        The Kornia library transpiled to NumPy

    Example:

    .. highlight:: python
    .. code-block:: python

        import kornia
        np_kornia = kornia.to_numpy()
        import numpy as np
        input = np.random.normal(size=(2, 3, 4, 5))
        gray = np_kornia.color.gray.rgb_to_grayscale(input)

    Note:
        Ivy does not currently support transpiling trainable modules to NumPy.
    """
    return ivy.transpile(
        kornia,
        source="torch",
        target="numpy",
    )  # type: ignore


def to_tensorflow() -> ModuleType:
    """Convert Kornia to TensorFlow.

    Transpiles the Kornia library to TensorFlow using [ivy](https://github.com/ivy-llc/ivy). The transpilation process
    occurs lazily, so the transpilation on a given kornia function/class will only occur when it's called or
    instantiated for the first time. This will make any functions/classes slow when being used for the first time,
    but any subsequent uses should be as fast as expected.

    Return:
        The Kornia library transpiled to TensorFlow

    Example:

    .. highlight:: python
    .. code-block:: python

        import kornia
        tf_kornia = kornia.to_tensorflow()
        import tensorflow as tf
        input = tf.random.normal((2, 3, 4, 5))
        gray = tf_kornia.color.gray.rgb_to_grayscale(input)
    """
    return ivy.transpile(
        kornia,
        source="torch",
        target="tensorflow",
    )  # type: ignore
