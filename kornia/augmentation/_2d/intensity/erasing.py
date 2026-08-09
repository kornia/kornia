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

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.geometry.bbox import bbox_generator, bbox_to_mask


class RandomErasing(IntensityAugmentationBase2D):
    r"""Erase a random rectangle of a torch.Tensor image according to a probability p value.

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
        Input torch.Tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation torch.Tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation torch.Tensor and returned.

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
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self._param_generator = rg.RectangleEraseGenerator(scale, ratio, value)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, h, w = input.size()
        values = params["values"].view(-1, 1, 1, 1).to(input)

        if self.same_on_batch:
            # Every sample shares one rectangle, so write that single region directly instead of
            # building a full-image mask and `where`-blending the whole tensor. Equivalent to the
            # masked path (the mask covers [x, x+width) x [y, y+height), clipped to the image), but
            # touches only the erased box — much cheaper for a batch (matches torchvision's slice).
            # Read the four shared box coords in a single device sync (one ``tolist``) rather than
            # four separate ``int(...)`` calls.
            xs, ys, box_w, box_h = (
                torch.stack([params["xs"][0], params["ys"][0], params["widths"][0], params["heights"][0]])
                .long()
                .tolist()
            )
            transformed = input.clone()
            transformed[:, :, ys : ys + box_h, xs : xs + box_w] = values
            return transformed

        # Broadcast the per-sample fill value (B, 1, 1, 1) and the single-channel mask (B, 1, H, W)
        # directly in `torch.where` instead of `repeat`-materializing both to full (B, C, H, W).
        # `where` broadcasts to the input shape, so the result is identical while allocating two
        # small tensors instead of two image-sized ones.
        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h).unsqueeze(1).to(input)  # (B, 1, H, W)
        transformed = torch.where(mask == 1.0, values, input)
        return transformed

    def apply_transform_mask(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, h, w = input.size()

        # Erase the corresponding areas on masks (fill with zeros). Broadcast a scalar zero and the
        # single-channel mask (B, 1, H, W) in `torch.where` rather than `repeat`-materializing full
        # (B, C, H, W) tensors — same result, two small allocations instead of two image-sized ones.
        values = torch.zeros((), dtype=input.dtype, device=input.device)

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h).unsqueeze(1).to(input)  # (B, 1, H, W)
        transformed = torch.where(mask == 1.0, values, input)
        return transformed
