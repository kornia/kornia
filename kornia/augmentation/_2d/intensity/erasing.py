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

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.geometry.bbox import bbox_generator, bbox_to_mask


class RandomErasing(IntensityAugmentationBase2D):
    r"""Erase a random rectangle of a torch.tensor image according to a probability p value.

    .. image:: _static/img/RandomErasing.png

    The operator removes image parts and fills them with zero values at a selected rectangle
    for each of the images in the batch.

    The rectangle will have an area equal to the original image area multiplied by a value uniformly
    sampled between the range [scale[0], scale[1]) and an aspect ratio sampled
    between [ratio[0], ratio[1])

    Args:
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        value: the value to fill the erased area.
        same_on_batch: apply the same transformation across the batch.
        p: probability that the random erasing operation will be performed.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input torch.tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation torch.tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation torch.tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 1, 3, 3)
        >>> aug = RandomErasing((.4, .8), (.3, 1/.3), p=0.5)
        >>> aug(inputs)
        tensor([[[[1., 0., 0.],
                  [1., 0., 0.],
                  [1., 0., 0.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomErasing((.4, .8), (.3, 1/.3), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.scale = torch.as_tensor(scale)
        self.ratio = torch.as_tensor(ratio)
        self.value = value

        # Validate
        if not (isinstance(value, (int, float)) and 0 <= value <= 1):
            raise AssertionError(f"'value' must be a number between 0 - 1. Got {value}.")

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        if not (isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")

        images_area = height * width

        # Sample scale (area proportion)
        if self.same_on_batch:
            target_areas = (
                torch.empty(1, device=self.device, dtype=self.dtype)
                .uniform_(self.scale[0].item(), self.scale[1].item())
                .expand(batch_size)
                * images_area
            )
        else:
            target_areas = (
                torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                    self.scale[0].item(), self.scale[1].item()
                )
                * images_area
            )

        # Sample aspect ratio (special handling for range crossing 1.0)
        ratio_low, ratio_high = self.ratio[0].item(), self.ratio[1].item()
        if ratio_low < 1.0 and ratio_high > 1.0:
            # Sample from two sub-ranges and randomly pick
            if self.same_on_batch:
                aspect_ratios1 = torch.empty(1, device=self.device, dtype=self.dtype).uniform_(ratio_low, 1.0)
                aspect_ratios2 = torch.empty(1, device=self.device, dtype=self.dtype).uniform_(1.0, ratio_high)
                rand_idx = torch.rand(1) > 0.5
                aspect_ratios = (aspect_ratios1 if rand_idx else aspect_ratios2).expand(batch_size)
            else:
                aspect_ratios1 = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(ratio_low, 1.0)
                aspect_ratios2 = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(1.0, ratio_high)
                rand_idxs = torch.rand(batch_size) > 0.5
                aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
        elif self.same_on_batch:
            aspect_ratios = (
                torch.empty(1, device=self.device, dtype=self.dtype).uniform_(ratio_low, ratio_high).expand(batch_size)
            )
        else:
            aspect_ratios = torch.empty(batch_size, device=self.device, dtype=self.dtype).uniform_(
                ratio_low, ratio_high
            )

        # Compute rectangle dimensions
        heights = torch.min(
            torch.max(torch.round((target_areas * aspect_ratios) ** 0.5), torch.tensor(1.0)),
            torch.tensor(float(height)),
        )
        widths = torch.min(
            torch.max(torch.round((target_areas / aspect_ratios) ** 0.5), torch.tensor(1.0)), torch.tensor(float(width))
        )

        # Sample position
        if self.same_on_batch:
            xs_ratio = torch.rand(1, device=self.device, dtype=self.dtype).expand(batch_size)
            ys_ratio = torch.rand(1, device=self.device, dtype=self.dtype).expand(batch_size)
        else:
            xs_ratio = torch.rand(batch_size, device=self.device, dtype=self.dtype)
            ys_ratio = torch.rand(batch_size, device=self.device, dtype=self.dtype)

        xs = xs_ratio * (width - widths + 1)
        ys = ys_ratio * (height - heights + 1)

        return {
            "widths": widths.floor(),
            "heights": heights.floor(),
            "xs": xs.floor(),
            "ys": ys.floor(),
            "values": torch.full((batch_size,), self.value, device=self.device, dtype=self.dtype),
        }

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, c, h, w = input.size()
        values = params["values"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, *input.shape[1:]).to(input)

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        transformed = torch.where(mask == 1.0, values, input)
        return transformed

    def apply_transform_mask(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, c, h, w = input.size()

        values = params["values"][..., None, None, None].repeat(1, *input.shape[1:]).to(input)
        # Erase the corresponding areas on masks.
        values = values.zero_()

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        transformed = torch.where(mask == 1.0, values, input)
        return transformed
