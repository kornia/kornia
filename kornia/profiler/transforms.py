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

from typing import Callable, Union

import torch
from torch import Tensor, nn


TransformType = Union[nn.Module, Callable[[Tensor], Tensor], str]


def _build_transform(transform: TransformType) -> Callable[[Tensor], Tensor]:
    """Convert a supported transform specification into a callable.

    Args:
        transform: An ``nn.Module``, a callable accepting and returning a
            tensor, or the string ``"flatten"``.

    Returns:
        A callable that applies the specified transformation.

    Raises:
        TypeError: If the transform is not supported.
    """
    if transform == "flatten":
        return lambda x: torch.flatten(x, start_dim=1)

    if isinstance(transform, nn.Module) or callable(transform):
        return transform

    raise TypeError(f"Unsupported transform: {transform!r}")


class Transform:
    """Composable wrapper for tensor representation transformations.

    A transform can be composed with another supported transform using the
    ``|`` operator. The resulting transform applies the existing operation
    first and the newly supplied operation second.
    """

    def __init__(self, transform: TransformType) -> None:
        """Initialize a transform from a supported specification.

        Args:
            transform: An ``nn.Module``, a callable accepting and returning
                a tensor, or the string ``"flatten"``.

        Raises:
            TypeError: If the transform is not supported.
        """
        self._transform: Callable[[Tensor], Tensor] = _build_transform(transform)

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the configured transformation to an input tensor.

        Args:
            x: Input tensor.

        Returns:
            The transformed tensor.
        """
        return self._transform(x)

    def __or__(self, other: TransformType) -> "Transform":
        """Compose this transform with another transform.

        The current transform is applied first, followed by ``other``.

        Args:
            other: The transform to apply after the current transform.

        Returns:
            A new ``Transform`` representing the composed operations.

        Raises:
            TypeError: If ``other`` is not a supported transform.
        """
        next_transform = _build_transform(other)
        previous_transform = self._transform

        return Transform(lambda x: next_transform(previous_transform(x)))


class Flatten(Transform):
    """Flatten each sample while preserving the batch dimension."""

    def __init__(self) -> None:
        """Initialize a transform that flattens each input sample."""
        super().__init__("flatten")
