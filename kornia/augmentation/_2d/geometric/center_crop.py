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


class CenterCrop(GeometricAugmentationBase2D):
    r"""Crop a given image torch.tensor at the center.

    .. image:: _static/img/CenterCrop.png

    Args:
        size: Desired output size (out_h, out_w) of the crop.
            If integer,  out_h = out_w = size.
            If Tuple[int, int], out_h = size[0], out_w = size[1].
        align_corners: interpolation flag.
        resample: The interpolation mode.
        p: probability of applying the transformation for the whole batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the torch.tensor based
                       on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                       to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                       differentiability.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.crop_by_boxes`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 4, 4)
        >>> inputs
        tensor([[[[-1.1258, -1.1524, -0.2506, -0.4339],
                  [ 0.8487,  0.6920, -0.3160, -2.1152],
                  [ 0.3223, -1.2633,  0.3500,  0.3081],
                  [ 0.1198,  1.2377,  1.1168, -0.2473]]]])
        >>> aug = CenterCrop(2, p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[ 0.6920, -0.3160],
                  [-1.2633,  0.3500]]]])
        >>> aug.inverse(out, padding_mode="border")
        tensor([[[[ 0.6920,  0.6920, -0.3160, -0.3160],
                  [ 0.6920,  0.6920, -0.3160, -0.3160],
                  [-1.2633, -1.2633,  0.3500,  0.3500],
                  [-1.2633, -1.2633,  0.3500,  0.3500]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = CenterCrop(2, p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        align_corners: bool = True,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ) -> None:
        # same_on_batch is always True for CenterCrop
        # Since PyTorch does not support ragged torch.tensor. So cropping function happens batch-wisely.
        super().__init__(p=1.0, same_on_batch=True, p_batch=p, keepdim=keepdim)
        if isinstance(size, tuple):
            self.size = (size[0], size[1])
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise Exception(f"Invalid size type. Expected (int, tuple(int, int). Got: {type(size)}.")

        self.flags = {
            "resample": Resample.get(resample),
            "cropping_mode": cropping_mode,
            "align_corners": align_corners,
            "size": self.size,
            "padding_mode": "zeros",
        }

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        _device = self.device

        dst_h, dst_w = self.size

        if not (isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0):
            raise AssertionError(f"'height' and 'width' must be positive integers. Got {height}, {width}.")
        if not (height >= dst_h and width >= dst_w):
            raise AssertionError(f"Crop size must be smaller than input size. Got ({height}, {width}) and {self.size}.")

        if batch_size == 0:
            return {"src": torch.zeros([0, 4, 2], device=_device), "dst": torch.zeros([0, 4, 2], device=_device)}

        # compute start/end offsets
        dst_h_half = dst_h / 2
        dst_w_half = dst_w / 2
        src_h_half = height / 2
        src_w_half = width / 2

        start_x = src_w_half - dst_w_half
        start_y = src_h_half - dst_h_half

        # [y, x] origin
        points_src = bbox_generator(
            torch.tensor([start_x] * batch_size, device=_device, dtype=torch.long),
            torch.tensor([start_y] * batch_size, device=_device, dtype=torch.long),
            torch.tensor([dst_w] * batch_size, device=_device, dtype=torch.long),
            torch.tensor([dst_h] * batch_size, device=_device, dtype=torch.long),
        )

        # [y, x] destination
        points_dst = bbox_generator(
            torch.tensor([0] * batch_size, device=_device, dtype=torch.long),
            torch.tensor([0] * batch_size, device=_device, dtype=torch.long),
            torch.tensor([dst_w] * batch_size, device=_device, dtype=torch.long),
            torch.tensor([dst_h] * batch_size, device=_device, dtype=torch.long),
        )

        return {"src": points_src, "dst": points_dst}

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
                transform[:, :2, :],
                self.size,
                flags["resample"].name.lower(),
                "zeros",
                flags["align_corners"],
            )
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            return crop_by_indices(input, params["src"], flags["size"])
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
        if size is None:
            size = self.size
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
