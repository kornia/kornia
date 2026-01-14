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

from typing import Any, Dict, Optional, Tuple, Union

import torch

import kornia
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.augmentation.utils import _tuple_range_reader
from kornia.constants import Resample
from kornia.geometry import affine3d
from kornia.geometry.transform.affwarp import _compute_rotation_matrix3d, _compute_tensor_center3d


class RandomRotation3D(GeometricAugmentationBase3D):
    r"""Apply random rotations to 3D volumes (5D torch.tensor).

    Input should be a torch.tensor of shape (C, D, H, W) or a batch of tensors :math:`(B, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will rotate the tensors and torch.cat the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        degrees: Range of degrees to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, D, H, W)`

    Note:
        Input torch.tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation torch.tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation torch.tensor and returned.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomRotation3D((15., 20., 20.), p=1.0)
        >>> aug(input), aug.transform_matrix
        (tensor([[[[[0.3819, 0.4886, 0.2111],
                   [0.1196, 0.3833, 0.4722],
                   [0.3432, 0.5951, 0.4223]],
        <BLANKLINE>
                  [[0.5553, 0.4374, 0.2780],
                   [0.2423, 0.1689, 0.4009],
                   [0.4516, 0.6376, 0.7327]],
        <BLANKLINE>
                  [[0.1605, 0.3112, 0.3673],
                   [0.4931, 0.4620, 0.5700],
                   [0.3505, 0.4685, 0.8092]]]]]), tensor([[[ 0.9722,  0.1131, -0.2049,  0.1196],
                 [-0.0603,  0.9669,  0.2478, -0.1545],
                 [ 0.2262, -0.2286,  0.9469,  0.0556],
                 [ 0.0000,  0.0000,  0.0000,  1.0000]]]))

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomRotation3D((15., 20., 20.), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        degrees: Union[
            torch.Tensor,
            float,
            Tuple[float, float, float],
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        ],
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.flags = {"resample": Resample.get(resample), "align_corners": align_corners}
        self.degrees = degrees

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _device, _dtype = self.device, self.dtype

        degrees = _tuple_range_reader(self.degrees, 3, _device, _dtype)

        if self.same_on_batch:
            yaw = (
                torch.empty(1, device=_device, dtype=_dtype)
                .uniform_(degrees[0][0].item(), degrees[0][1].item())
                .expand(batch_size)
            )
            pitch = (
                torch.empty(1, device=_device, dtype=_dtype)
                .uniform_(degrees[1][0].item(), degrees[1][1].item())
                .expand(batch_size)
            )
            roll = (
                torch.empty(1, device=_device, dtype=_dtype)
                .uniform_(degrees[2][0].item(), degrees[2][1].item())
                .expand(batch_size)
            )
        else:
            yaw = torch.empty(batch_size, device=_device, dtype=_dtype).uniform_(
                degrees[0][0].item(), degrees[0][1].item()
            )
            pitch = torch.empty(batch_size, device=_device, dtype=_dtype).uniform_(
                degrees[1][0].item(), degrees[1][1].item()
            )
            roll = torch.empty(batch_size, device=_device, dtype=_dtype).uniform_(
                degrees[2][0].item(), degrees[2][1].item()
            )

        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        yaw: torch.Tensor = params["yaw"].to(input)
        pitch: torch.Tensor = params["pitch"].to(input)
        roll: torch.Tensor = params["roll"].to(input)

        center: torch.Tensor = _compute_tensor_center3d(input)
        rotation_mat: torch.Tensor = _compute_rotation_matrix3d(yaw, pitch, roll, center.expand(yaw.shape[0], -1))

        # rotation_mat is B x 3 x 4 and we need a B x 4 x 4 matrix
        trans_mat: torch.Tensor = kornia.core.ops.eye_like(4, input)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]
        trans_mat[:, 2] = rotation_mat[:, 2]

        return trans_mat

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the transform to be a torch.Tensor. Gotcha {type(transform)}")

        return affine3d(input, transform[..., :3, :4], flags["resample"].name.lower(), "zeros", flags["align_corners"])
