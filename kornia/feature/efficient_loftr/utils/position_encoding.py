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

import math
from typing import List, Optional, Tuple

import torch

from kornia.core import Module, Tensor


class RoPEPositionEncodingSine(Module):
    """This is a sinusoidal position encoding that generalized to 2-dimensional images."""

    def __init__(
        self,
        d_model: Tensor,
        max_shape: Tuple[int] = (256, 256),
        npe: Optional[List[int]] = None,
        ropefp16: bool = True,
    ) -> None:
        """Sinusoidal positional encoding.

        Args:
        d_model : model dimension
        max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        npe : number of position encoding
        ropefp16 : dtype
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1)  # [H, 1]
        j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1)  # [W, 1]

        if npe is None:
            raise AssertionError(npe)
        train_res_H, train_res_W, test_res_H, test_res_W = (
            npe[0],
            npe[1],
            npe[2],
            npe[3],
        )  # train_res_H, train_res_W, test_res_H, test_res_W
        i_position, j_position = i_position * train_res_H / test_res_H, j_position * train_res_W / test_res_W

        div_term = torch.exp(torch.arange(0, d_model // 4, 1).float() * (-math.log(10000.0) / (d_model // 4)))
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(*max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos = torch.zeros(*max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin[:, :, 0::2] = torch.sin(i_position * div_term).half() if ropefp16 else torch.sin(i_position * div_term)
        sin[:, :, 1::2] = torch.sin(j_position * div_term).half() if ropefp16 else torch.sin(j_position * div_term)
        cos[:, :, 0::2] = torch.cos(i_position * div_term).half() if ropefp16 else torch.cos(i_position * div_term)
        cos[:, :, 1::2] = torch.cos(j_position * div_term).half() if ropefp16 else torch.cos(j_position * div_term)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer("sin", sin.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer("cos", cos.unsqueeze(0), persistent=False)  # [1, H, W, C//2]

    def forward(self, x: Tensor, ratio: int = 1) -> Tensor:
        """Forward run.

        Args:
            x: [N, H, W, C]
            ratio : Not applicable.
        """
        return (x * self.cos[:, : x.size(1), : x.size(2), :]) + (
            self.rotate_half(x) * self.sin[:, : x.size(1), : x.size(2), :]
        )

    def rotate_half(self, x: Tensor) -> Tensor:
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
