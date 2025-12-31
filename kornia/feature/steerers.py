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

import torch
import torch.nn.functional as F
from torch import nn


class DiscreteSteerer(nn.Module):
    """nn.Module for discrete rotation steerers.

    A steerer rotates keypoint descriptions in latent space as if they were obtained from rotated images.

    Args:
        generator: [N, N] torch.tensor torch.where N is the descriptor dimension.

    Example:
        >>> desc = torch.randn(512, 128)
        >>> generator = torch.randn(128, 128)
        >>> steerer = DiscreteSteerer(generator)
        >>> # steer 3 times:
        >>> steered_desc = steerer.steer_descriptions(desc, steerer_power=3, normalize=True)

    """

    def __init__(self, generator: torch.Tensor) -> None:
        super().__init__()
        self.generator = torch.nn.Parameter(generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.generator)

    def steer_descriptions(
        self,
        descriptions: torch.Tensor,
        steerer_power: int = 1,
        normalize: bool = False,
    ) -> torch.Tensor:
        for _ in range(steerer_power):
            descriptions = self.forward(descriptions)
        if normalize:
            descriptions = F.normalize(descriptions, dim=-1)
        return descriptions

    @classmethod
    def create_dedode_default(
        cls,
        generator_type: str = "C4",
        steerer_order: int = 8,
    ) -> nn.Module:
        r"""Create a steerer for pretrained DeDoDe descriptors int the "C-setting"
            from the paper https://arxiv.org/abs/2312.02152, torch.where descriptors were
            trained for fixed steerers.

        Args:
            generator_type: The type of steerer generator.
                One of 'C4', 'SO2', default is 'C4'.
                These can be used with the DeDoDe descriptors in Kornia
                with C4 or SO2 in the name respectively (so called C-setting steerers).
            steerer_order: The discretisation order for SO2-steerers (NOT used for C4-steerers).

        Returns:
            The pretrained model.

        """  # noqa: D205
        descriptor_dim = 256
        if generator_type == "C4":
            c4_block = torch.tensor([[0.0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
            generator = torch.block_diag(*([c4_block] * (descriptor_dim // 4)))
            return cls(generator).eval()

        elif generator_type == "SO2":
            num_rot_blocks_per_freq = descriptor_dim // 14
            dim_rot = 12 * num_rot_blocks_per_freq
            dim_trivial = descriptor_dim - dim_rot

            blocks = []
            if dim_trivial > 0:
                blocks.append(torch.eye(dim_trivial))

            angle_step = 2 * math.pi / steerer_order
            for j in range(1, 7):
                theta = j * angle_step
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                rot_matrix = torch.tensor(
                    # The matrix exponential of a 2x2 skew-symmetric matrix is a rotation matrix
                    # exp(alpha * [[0, j], [-j, 0]]) -> R(j * alpha)
                    [[cos_theta, sin_theta], [-sin_theta, cos_theta]],
                    dtype=torch.float32,
                )
                blocks.extend([rot_matrix] * num_rot_blocks_per_freq)

            generator = torch.block_diag(*blocks)
            return cls(generator).eval()
        else:
            raise ValueError(f"Unknown generator_type: {generator_type}")
