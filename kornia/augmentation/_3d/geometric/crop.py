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

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.constants import Resample
from kornia.geometry import crop_by_transform_mat3d, get_perspective_transform3d


class RandomCrop3D(GeometricAugmentationBase3D):
    r"""Apply random crop on 3D volumes (5D torch.Tensor).

    Crops random sub-volumes on a given size.

    Args:
        p: probability of applying the transformation for the whole batch.
            When skipped, the input is returned without padding or cropping.
        size: Desired output size (out_d, out_h, out_w) of the crop.
            Must be Tuple[int, int, int], then out_d = size[0], out_h = size[1], out_w = size[2].
        padding: Optional padding on each border of the image.
            Default is None, i.e no padding. If a sequence of length 6 is provided, it is used to F.pad
            left, top, right, bottom, front, back borders respectively.
            If a sequence of length 3 is provided, it is used to F.pad left/right,
            top/bottom, front/back borders, respectively.
        pad_if_needed: It will F.pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0.
            This value is only used when the padding_mode is constant.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, , out_d, out_h, out_w)`

    Note:
        Input torch.Tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation torch.Tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation torch.Tensor and returned.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 3, 3, 3)
        >>> aug = RandomCrop3D((2, 2, 2), p=1.)
        >>> aug(inputs)
        tensor([[[[[-1.1258, -1.1524],
                   [-0.4339,  0.8487]],
        <BLANKLINE>
                  [[-1.2633,  0.3500],
                   [ 0.1665,  0.8744]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomCrop3D((24, 24, 24), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        size: Tuple[int, int, int],
        padding: Optional[Union[int, Tuple[int, int, int], Tuple[int, int, int, int, int, int]]] = None,
        pad_if_needed: Optional[bool] = False,
        fill: int = 0,
        padding_mode: str = "constant",
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        # Since PyTorch does not support ragged torch.Tensor. So cropping function happens batch-wisely.
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self.flags = {
            "size": size,
            "padding": padding,
            "pad_if_needed": pad_if_needed,
            "padding_mode": padding_mode,
            "fill": fill,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
        }
        self._param_generator = rg.CropGenerator3D(size, None)

    def _compute_padding(self, shape: Tuple[int, ...], flags: Dict[str, Any]) -> List[List[int]]:
        """Compute successive padding steps without modifying the input volume."""
        padding_steps: List[List[int]] = []
        depth, height, width = shape[-3:]
        padding = flags["padding"]
        if padding is not None:
            if isinstance(padding, int):
                padding = [padding, padding, padding, padding, padding, padding]
            elif isinstance(padding, (tuple, list)) and len(padding) == 3:
                padding = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
            elif isinstance(padding, (tuple, list)) and len(padding) == 6:
                padding = [padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]]
            else:
                raise ValueError(f"`padding` must be an integer, 3-element-list or 6-element-list. Got {padding}.")
            padding_steps.append(padding)
            depth += padding[4] + padding[5]
            height += padding[2] + padding[3]
            width += padding[0] + padding[1]

        if flags["pad_if_needed"] and depth < flags["size"][0]:
            padding_steps.append([0, 0, 0, 0, flags["size"][0] - depth, flags["size"][0] - depth])

        if flags["pad_if_needed"] and height < flags["size"][1]:
            padding_steps.append([0, 0, flags["size"][1] - height, flags["size"][1] - height, 0, 0])

        if flags["pad_if_needed"] and width < flags["size"][2]:
            padding_steps.append([flags["size"][2] - width, flags["size"][2] - width, 0, 0, 0, 0])

        return padding_steps

    def precrop_padding(self, input: torch.Tensor, flags: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        flags = self.flags if flags is None else flags
        # Keep fixed and automatic padding separate to preserve non-constant boundary values.
        for padding in self._compute_padding(tuple(input.shape), flags):
            input = F.pad(input, padding, value=flags["fill"], mode=flags["padding_mode"])

        return input

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        transform: torch.Tensor = get_perspective_transform3d(params["src"].to(input), params["dst"].to(input))
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the transform to be a torch.Tensor. Gotcha {type(transform)}")

        input = self.precrop_padding(input, flags)
        return crop_by_transform_mat3d(
            input, transform, flags["size"], mode=flags["resample"].name.lower(), align_corners=flags["align_corners"]
        )

    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        # Sample crop coordinates on the padded canvas, while keeping the skip path unpadded.
        padded_shape = list(batch_shape)
        for padding in self._compute_padding(batch_shape, self.flags):
            padded_shape[-3] += padding[4] + padding[5]
            padded_shape[-2] += padding[2] + padding[3]
            padded_shape[-1] += padding[0] + padding[1]
        return super().forward_parameters(tuple(padded_shape))
