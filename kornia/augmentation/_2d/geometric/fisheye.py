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

from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.geometry.grid import create_meshgrid
from kornia.geometry.transform import remap


class RandomFisheye(AugmentationBase2D):
    r"""Add random camera radial distortion.

    .. image:: _static/img/RandomFisheye.png

    Args:
        center_x: Ranges to sample respect to x-coordinate center with shape (2,).
        center_y: Ranges to sample respect to y-coordinate center with shape (2,).
        gamma: Ranges to sample for the gamma values respect to optical center with shape (2,).
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Examples:
        >>> import torch
        >>> img = torch.ones(1, 1, 2, 2)
        >>> center_x = torch.tensor([-.3, .3])
        >>> center_y = torch.tensor([-.3, .3])
        >>> gamma = torch.tensor([.9, 1.])
        >>> out = RandomFisheye(center_x, center_y, gamma)(img)
        >>> out.shape
        torch.Size([1, 1, 2, 2])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomFisheye(center_x, center_y, gamma, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        gamma: torch.Tensor,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._check_tensor(center_x)
        self._check_tensor(center_y)
        self._check_tensor(gamma)
        self.center_x_bound = center_x
        self.center_y_bound = center_y
        self.gamma_bound = gamma

    def _check_tensor(self, data: torch.Tensor) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Invalid input type. Expected torch.Tensor - got: {type(data)}")

        if len(data.shape) != 1 and data.shape[0] != 2:
            raise ValueError(f"torch.Tensor must be of shape (2,). Got: {data.shape}.")

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        if self.same_on_batch:
            center_x = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.center_x_bound[0].item(), self.center_x_bound[1].item())
                .expand(batch_size)
            )
            center_y = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.center_y_bound[0].item(), self.center_y_bound[1].item())
                .expand(batch_size)
            )
            gamma = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.gamma_bound[0].item(), self.gamma_bound[1].item())
                .expand(batch_size)
            )
        else:
            center_x = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                self.center_x_bound[0].item(), self.center_x_bound[1].item()
            )
            center_y = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                self.center_y_bound[0].item(), self.center_y_bound[1].item()
            )
            gamma = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                self.gamma_bound[0].item(), self.gamma_bound[1].item()
            )
        return {"center_x": center_x, "center_y": center_y, "gamma": gamma}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # create the initial sampling fields
        B, _, H, W = input.shape
        grid = create_meshgrid(H, W, normalized_coordinates=True)
        field_x = grid[..., 0].to(input)  # 1xHxW
        field_y = grid[..., 1].to(input)  # 1xHxW
        # vectorize the random parameters
        center_x = params["center_x"].view(B, 1, 1).to(input)
        center_y = params["center_y"].view(B, 1, 1).to(input)
        gamma = params["gamma"].view(B, 1, 1).to(input)
        # compute and apply the distances respect to the camera optical center
        distance = ((center_x - field_x) ** 2 + (center_y - field_y) ** 2) ** 0.5
        field_x = field_x + field_x * distance**gamma  # BxHxw
        field_y = field_y + field_y * distance**gamma  # BxHxW
        return remap(input, field_x, field_y, normalized_coordinates=True, align_corners=True)
