"""Based from the original code from Meta Platforms, Inc. and affiliates.

https://github.com/facebookresearch/segment-
anything/blob/3518c86b78b3bc9cf4fbe3d18e682fad1c79dc51/segment_anything/modeling/common.py
"""

from __future__ import annotations

from typing import Any

from torch import nn

from kornia.core import Module, Tensor


class MLPBlock(Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: type[Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 1e-6
