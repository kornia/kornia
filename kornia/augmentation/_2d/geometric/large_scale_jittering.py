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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.geometry.transform import resize


class RandomLargeScaleJittering(GeometricAugmentationBase2D):
    """Random scale jitter with crop/pad to output size — the SAM/MaskRCNN+ recipe.

    Implements the Large-Scale Jittering (LSJ) augmentation used in SAM, ViTDet, and MaskFormer:

    1. Sample a scale uniformly from ``[scale_range[0], scale_range[1]]``.
    2. Resize the image to ``(round(H * scale), round(W * scale))``.
    3. If the resized image is larger than ``output_size``, take a random crop.
    4. If the resized image is smaller than ``output_size``, pad to ``output_size``.

    The output shape is always exactly ``output_size`` regardless of input shape.

    .. note::
        Mask and bounding-box support (``apply_to_mask`` / ``apply_to_boxes``) is not implemented
        in this V1 release and is planned as a follow-up.

    Args:
        output_size: ``(H, W)`` of the final output.
        scale_range: ``(min_scale, max_scale)`` jitter range. Default ``(0.1, 2.0)`` per Masked R-CNN+.
        pad_value: fill value used when the resized image is smaller than ``output_size``. Default ``0.0``.
        resample: interpolation mode. Default ``"BILINEAR"``.
        same_on_batch: apply the same scale to all samples in a batch. Default ``False``.
        p: probability of applying the augmentation. Default ``0.5``.
        keepdim: standard kornia keepdim semantics.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, out_H, out_W)`

    Example:
        >>> import torch
        >>> aug = RandomLargeScaleJittering(output_size=(640, 640), scale_range=(0.5, 1.5), p=1.0)
        >>> x = torch.randn(2, 3, 480, 640)
        >>> aug(x).shape
        torch.Size([2, 3, 640, 640])
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        pad_value: float = 0.0,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        # p=1.0 at element level; p_batch=p controls batch-level probability.
        # We always apply (the output must be output_size), so route through p_batch.
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)

        if scale_range[0] <= 0:
            raise ValueError(f"scale_range min must be > 0, got {scale_range[0]}")
        if scale_range[0] > scale_range[1]:
            raise ValueError(f"scale_range min must be <= max, got min={scale_range[0]}, max={scale_range[1]}")

        self.output_size = output_size
        self.flags = {
            "output_size": output_size,
            "scale_range": scale_range,
            "pad_value": pad_value,
            "resample": Resample.get(resample),
        }

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """Generate per-sample scale and crop-offset parameters.

        Args:
            shape: ``(B, C, H, W)`` of the input batch.

        Returns:
            Dict with keys:

            - ``"scale"``: shape ``(B,)`` scale factors.
            - ``"crop_top"``: shape ``(B,)`` top-left row for the crop (only relevant when resized > output).
            - ``"crop_left"``: shape ``(B,)`` top-left col for the crop (only relevant when resized > output).
        """
        B, _, H, W = shape
        out_h, out_w = self.flags["output_size"]
        scale_min, scale_max = self.flags["scale_range"]

        if self.same_on_batch:
            scale = torch.empty(1, device=self.device, dtype=torch.float32).uniform_(scale_min, scale_max)
            scale = scale.expand(B)
        else:
            scale = torch.empty(B, device=self.device, dtype=torch.float32).uniform_(scale_min, scale_max)

        # Compute resized dimensions for each sample (integer)
        resized_h = (scale * H).round().long().clamp(min=1)
        resized_w = (scale * W).round().long().clamp(min=1)

        # Random crop offsets: clamped to 0 when resized < output (no crop needed)
        max_top = (resized_h - out_h).clamp(min=0)
        max_left = (resized_w - out_w).clamp(min=0)

        if self.same_on_batch:
            crop_top = (torch.rand(1, device=self.device) * (max_top[0].float() + 1)).long().clamp(max=max_top[0])
            crop_top = crop_top.expand(B)
            crop_left = (torch.rand(1, device=self.device) * (max_left[0].float() + 1)).long().clamp(max=max_left[0])
            crop_left = crop_left.expand(B)
        else:
            crop_top = (torch.rand(B, device=self.device) * (max_top.float() + 1)).long()
            crop_top = crop_top.clamp(max=max_top)
            crop_left = (torch.rand(B, device=self.device) * (max_left.float() + 1)).long()
            crop_left = crop_left.clamp(max=max_left)

        return {
            "scale": scale,
            "crop_top": crop_top,
            "crop_left": crop_left,
        }

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        """Return identity matrix (LSJ does not have a simple affine representation)."""
        return self.identity_matrix(input)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply LSJ to the batch.

        Args:
            input: ``(B, C, H, W)`` input tensor.
            params: output of :meth:`generate_parameters`.
            flags: static flags dict (``output_size``, ``scale_range``, ``pad_value``, ``resample``).
            transform: unused (kept for API compatibility).

        Returns:
            ``(B, C, out_H, out_W)`` tensor.
        """
        B, C, H, W = input.shape
        out_h, out_w = flags["output_size"]
        pad_value: float = flags["pad_value"]
        interp_mode: str = flags["resample"].name.lower()

        output = torch.full(
            (B, C, out_h, out_w),
            fill_value=pad_value,
            dtype=input.dtype,
            device=input.device,
        )

        scales = params["scale"]
        crop_tops = params["crop_top"]
        crop_lefts = params["crop_left"]

        for i in range(B):
            rh = max(1, round(float(scales[i]) * H))
            rw = max(1, round(float(scales[i]) * W))

            resized = resize(
                input[i : i + 1],
                (rh, rw),
                interpolation=interp_mode,
                align_corners=True if flags["resample"] in (Resample.BILINEAR, Resample.BICUBIC) else None,
            )  # (1, C, rh, rw)

            t = int(crop_tops[i])
            l = int(crop_lefts[i])  # noqa: E741

            # Effective crop window (may be smaller than output if resized < output)
            crop_h = min(rh - t, out_h)
            crop_w = min(rw - l, out_w)

            output[i, :, :crop_h, :crop_w] = resized[0, :, t : t + crop_h, l : l + crop_w]

        return output

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Inverse is not implemented for LSJ (crop+pad is not uniquely invertible)."""
        raise NotImplementedError(
            "RandomLargeScaleJittering does not support inverse transforms because the "
            "crop/pad operation is not uniquely invertible."
        )
