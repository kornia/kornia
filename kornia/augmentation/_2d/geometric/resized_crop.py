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

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.geometry.bbox import bbox_generator
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform


class RandomResizedCrop(GeometricAugmentationBase2D):
    r"""Crop random patches in an image torch.tensor and resizes to a given size.

    .. image:: _static/img/RandomResizedCrop.png

    Args:
        size: Desired output size (out_h, out_w) of each edge.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        scale: range of size of the origin size cropped.
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        resample: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of the augmentation been applied.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the torch.tensor based
                       on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                       to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                       differentiability.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input torch.tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation torch.tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation torch.tensor and returned.

    Example:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.tensor([[[0., 1., 2.],
        ...                         [3., 4., 5.],
        ...                         [6., 7., 8.]]])
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[1.0000, 1.5000, 2.0000],
                  [4.0000, 4.5000, 5.0000],
                  [7.0000, 7.5000, 8.0000]]]])
        >>> aug.inverse(out, padding_mode="border")
        tensor([[[[1., 1., 2.],
                  [4., 4., 5.],
                  [7., 7., 8.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.08, 1.0),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (3.0 / 4.0, 4.0 / 3.0),
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ) -> None:
        # Since PyTorch does not support ragged torch.tensor. So cropping function happens all the time.
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)

        if not (
            len(size) == 2
            and isinstance(size[0], (int,))
            and isinstance(size[1], (int,))
            and size[0] > 0
            and size[1] > 0
        ):
            raise AssertionError(f"`output_size` must be a tuple of 2 positive integers. Got {size}.")

        self.output_size = size
        self.scale = torch.as_tensor(scale)
        self.ratio = torch.as_tensor(ratio)
        self.flags = {
            "size": size,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
            "cropping_mode": cropping_mode,
            "padding_mode": "zeros",
        }

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height, width = batch_shape[-2], batch_shape[-1]
        _device, _dtype = self.device, self.dtype

        if batch_size == 0:
            return {
                "src": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                "dst": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                "size": torch.zeros([0, 2], device=_device, dtype=_dtype),
            }

        # Sample scale and compute areas
        if self.same_on_batch:
            rand_tensor = torch.rand(1, 10, device=_device, dtype=_dtype).expand(batch_size, -1).contiguous()
            log_ratio = (
                torch.empty(1, 10, device=_device, dtype=_dtype)
                .uniform_(torch.log(self.ratio[0]).item(), torch.log(self.ratio[1]).item())
                .expand(batch_size, -1)
                .contiguous()
            )
        else:
            rand_tensor = torch.rand(batch_size, 10, device=_device, dtype=_dtype)
            log_ratio = torch.empty(batch_size, 10, device=_device, dtype=_dtype).uniform_(
                torch.log(self.ratio[0]).item(), torch.log(self.ratio[1]).item()
            )

        scale_tensor = self.scale.to(device=_device, dtype=_dtype)
        area = (rand_tensor * (scale_tensor[1] - scale_tensor[0]) + scale_tensor[0]) * height * width
        aspect_ratio = torch.exp(log_ratio)

        w = torch.sqrt(area * aspect_ratio).round().floor()
        h = torch.sqrt(area / aspect_ratio).round().floor()

        # Element-wise w, h condition
        cond = ((0 < w) * (w < width) * (0 < h) * (h < height)).int()

        # Select first valid crop size
        cond_bool, argmax_dim1 = ((cond.cumsum(1) == 1) & cond.bool()).max(1)
        h_out = h[torch.arange(0, batch_size, device=_device, dtype=torch.long), argmax_dim1]
        w_out = w[torch.arange(0, batch_size, device=_device, dtype=torch.long), argmax_dim1]

        if not cond_bool.all():
            # Fallback to center crop
            in_ratio = float(height) / float(width)
            _min = float(self.ratio.min()) if isinstance(self.ratio, torch.Tensor) else min(self.ratio)
            if in_ratio < _min:
                h_ct = torch.tensor(height, device=_device, dtype=_dtype)
                w_ct = torch.round(h_ct / _min)
            elif in_ratio > _min:
                w_ct = torch.tensor(width, device=_device, dtype=_dtype)
                h_ct = torch.round(w_ct * _min)
            else:  # whole image
                h_ct = torch.tensor(height, device=_device, dtype=_dtype)
                w_ct = torch.tensor(width, device=_device, dtype=_dtype)
            h_ct = h_ct.floor()
            w_ct = w_ct.floor()

            h_out = torch.where(cond_bool, h_out, h_ct)
            w_out = torch.where(cond_bool, w_out, w_ct)

        # Clamp crop size to input size to prevent out-of-bounds crops
        h_out = torch.clamp(h_out, min=1, max=height)
        w_out = torch.clamp(w_out, min=1, max=width)

        # Now generate crop parameters using the computed sizes
        crop_size = torch.stack([h_out, w_out], dim=1)

        input_size = (height, width)
        size = crop_size.floor()

        x_diff = input_size[1] - size[:, 1] + 1
        y_diff = input_size[0] - size[:, 0] + 1

        # Start point will be 0 if diff < 0
        x_diff = x_diff.clamp(0)
        y_diff = y_diff.clamp(0)

        if self.same_on_batch:
            x_start = (torch.rand(1, device=_device, dtype=_dtype).expand(batch_size) * x_diff[0]).floor()
            y_start = (torch.rand(1, device=_device, dtype=_dtype).expand(batch_size) * y_diff[0]).floor()
        else:
            x_start = (torch.rand(batch_size, device=_device, dtype=_dtype) * x_diff).floor()
            y_start = (torch.rand(batch_size, device=_device, dtype=_dtype) * y_diff).floor()

        crop_src = bbox_generator(
            x_start.view(-1),
            y_start.view(-1),
            torch.where(size[:, 1] == 0, torch.tensor(input_size[1], device=_device, dtype=_dtype), size[:, 1]),
            torch.where(size[:, 0] == 0, torch.tensor(input_size[0], device=_device, dtype=_dtype), size[:, 0]),
        )

        crop_dst = torch.tensor(
            [
                [
                    [0, 0],
                    [self.output_size[1] - 1, 0],
                    [self.output_size[1] - 1, self.output_size[0] - 1],
                    [0, self.output_size[0] - 1],
                ]
            ],
            device=_device,
            dtype=_dtype,
        ).repeat(batch_size, 1, 1)

        _output_size = torch.tensor(self.output_size, device=_device, dtype=torch.long).expand(batch_size, -1)
        _input_size = torch.tensor(input_size, device=_device, dtype=torch.long).expand(batch_size, -1)

        return {"src": crop_src, "dst": crop_dst, "input_size": _input_size, "output_size": _output_size}

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        if flags["cropping_mode"] in ("resample", "slice"):
            transform: torch.Tensor = get_perspective_transform(params["src"].to(input), params["dst"].to(input))
            transform = transform.expand(input.shape[0], -1, -1)
            return transform
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            if not isinstance(transform, torch.Tensor):
                raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

            return crop_by_transform_mat(
                input,
                transform,
                flags["size"],
                mode=flags["resample"].name.lower(),
                padding_mode="zeros",
                align_corners=flags["align_corners"],
            )
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            return crop_by_indices(
                input,
                params["src"],
                flags["size"],
                interpolation=flags["resample"].name.lower(),
                align_corners=flags["align_corners"],
            )
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        if not isinstance(size, tuple):
            raise TypeError(f"Expected the size be a tuple. Gotcha {type(size)}")

        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

        return crop_by_transform_mat(
            input,
            transform[:, :2, :],
            size,
            flags["resample"].name.lower(),
            flags["padding_mode"],
            flags["align_corners"],
        )
