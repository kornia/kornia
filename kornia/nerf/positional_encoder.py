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

from functools import partial

import torch
from torch import nn

from kornia.core import Tensor


def _torch_sin(x: Tensor, freq: Tensor) -> Tensor:
    return (x * freq).sin()  # FIXME: PI?


def _torch_cos(x: Tensor, freq: Tensor) -> Tensor:
    return (x * freq).cos()


class PositionalEncoder(nn.Module):
    """Sine-cosine positional encoder for input points."""

    def __init__(self, num_dims: int, num_freqs: int, log_space: bool = False) -> None:
        """Initialize positional encoder.

        Args:
            num_dims: Number of input dimensions (channels): int
            num_freqs: Number of frequency bands for encoding span: int
            log_space: Whether frequency sampling should be log spaced: bool

        """
        super().__init__()
        self._num_dims = num_dims
        self._embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        freq_bands: Tensor
        if log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0 ** (num_freqs - 1), num_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self._embed_fns.append(partial(_torch_sin, freq=freq))
            self._embed_fns.append(partial(_torch_cos, freq=freq))

        self._num_encoded_dims = self._num_dims * len(self._embed_fns)

    @property
    def num_encoded_dims(self) -> int:
        """Number of encoded dimensions."""
        return self._num_encoded_dims

    def forward(self, x: Tensor) -> Tensor:
        """Apply positional encoding to input.

        Args:
            x: Positionsl (or directional) tensor to encode: Tensor

        Returns:
            Tensor with encoded position/direction: Tensor

        """
        if x.ndim < 1:
            raise ValueError("Input tensor represents a scalar")
        if x.shape[-1] != self._num_dims:
            raise ValueError(
                f"Input tensor number of dimensions {x.shape[-1]} does not match instantiated dimensionality "
                f"{self._num_dims}"
            )
        return torch.cat([fn(x) for fn in self._embed_fns], dim=-1)
