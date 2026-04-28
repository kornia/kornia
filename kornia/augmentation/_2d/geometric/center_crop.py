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
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform


class CenterCrop(GeometricAugmentationBase2D):
    r"""Crop a given image torch.Tensor at the center.

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
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the torch.Tensor based
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
        # Since PyTorch does not support ragged torch.Tensor. So cropping function happens batch-wisely.
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

    # The legacy ``_fast_image_only_apply`` opt-in is disabled — the aggressive
    # forward override below is strictly faster.
    _supports_fast_image_only_path: bool = False

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Aggressive fast path: completely bypass the framework chain for the
        # ``slice`` cropping_mode + simple "single image tensor" call.  Slice
        # center crops are a pure ``input[..., y0:y1, x0:x1]`` view op.  The
        # ``resample`` mode requires ``warp_affine`` so we fall through.
        if (
            len(args) == 1
            and isinstance(args[0], torch.Tensor)
            and not kwargs
            and self.p_batch == 1.0
            and not self.keepdim
            and self.flags.get("cropping_mode") == "slice"
        ):
            x = args[0]
            d = x.dim()
            if d == 3:
                x = x.unsqueeze(0)
                d = 4
            if d == 4:
                b = x.shape[0]
                crop_h, crop_w = self.size
                in_h, in_w = x.shape[-2], x.shape[-1]
                start_y = int(in_h / 2 - crop_h / 2)
                start_x = int(in_w / 2 - crop_w / 2)
                self._params = {
                    "batch_prob": torch.full((b,), True, dtype=torch.bool),
                    "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                }
                # Mirror what ``get_perspective_transform`` would produce: a
                # pure translation matrix.
                mat = torch.tensor(
                    [[1.0, 0.0, -float(start_x)], [0.0, 1.0, -float(start_y)], [0.0, 0.0, 1.0]],
                    device=x.device,
                    dtype=x.dtype,
                )
                self._transform_matrix = mat.unsqueeze(0).expand(b, 3, 3)
                return x[..., start_y : start_y + crop_h, start_x : start_x + crop_w]
        return super().forward(*args, **kwargs)

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        return rg.center_crop_generator(batch_shape[0], batch_shape[-2], batch_shape[-1], self.size, self.device)

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
            # Under torch.export the params["src"] tensor carries symbolic
            # values that make int() / .item() calls data-dependent.  For
            # CenterCrop the crop box is fully determined by the static output
            # size (flags["size"]) and the concrete input shape, so we can
            # compute the slice indices directly in Python without going
            # through the params tensor.
            if torch.compiler.is_compiling():
                crop_h, crop_w = flags["size"]
                in_h, in_w = input.shape[-2], input.shape[-1]
                start_y = int(in_h / 2 - crop_h / 2)
                start_x = int(in_w / 2 - crop_w / 2)
                return input[..., start_y : start_y + crop_h, start_x : start_x + crop_w]
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
