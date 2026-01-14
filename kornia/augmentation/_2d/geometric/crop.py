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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.geometry.bbox import bbox_generator
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform


class RandomCrop(GeometricAugmentationBase2D):
    r"""Crop random patches of a torch.tensor image on a given size.

    .. image:: _static/img/RandomCrop.png

    Args:
        size: Desired output size (out_h, out_w) of the crop.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        padding: Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to F.pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            F.pad left/right, top/bottom borders, respectively.
        pad_if_needed: It will F.pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
        padding_mode: Type of padding. Should be: constant, reflect, replicate.
        resample: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
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

    Note:
        Input torch.tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation torch.tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation torch.tensor and returned.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> inputs = torch.arange(1*1*3*3.).view(1, 1, 3, 3)
        >>> aug = RandomCrop((2, 2), p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[3., 4.],
                  [6., 7.]]]])
        >>> aug.inverse(out, padding_mode="replicate")
        tensor([[[[3., 4., 4.],
                  [3., 4., 4.],
                  [6., 7., 7.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomCrop((2, 2), p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        size: Tuple[int, int],
        padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None,
        pad_if_needed: Optional[bool] = False,
        fill: int = 0,
        padding_mode: str = "constant",
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ) -> None:
        # Since PyTorch does not support ragged torch.tensor. So cropping function happens batch-wisely.
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self.crop_size = size
        self.flags = {
            "size": size,
            "padding": padding,
            "pad_if_needed": pad_if_needed,
            "fill": fill,
            "padding_mode": padding_mode,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
            "cropping_mode": cropping_mode,
        }

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _device, _dtype = self.device, self.dtype

        if batch_size == 0:
            return {
                "src": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                "dst": torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
            }

        input_size = (batch_shape[-2], batch_shape[-1])
        if not isinstance(self.crop_size, torch.Tensor):
            size = torch.tensor(self.crop_size, device=_device, dtype=_dtype).repeat(batch_size, 1)
        else:
            size = self.crop_size.to(device=_device, dtype=_dtype)
        if size.shape != torch.Size([batch_size, 2]):
            raise AssertionError(
                "If `size` is a torch.tensor, it must be shaped as (B, 2). "
                f"Got {size.shape} while expecting {torch.Size([batch_size, 2])}."
            )
        if not (input_size[0] > 0 and input_size[1] > 0 and (size > 0).all()):
            raise AssertionError(f"Got non-positive input size or size. {input_size}, {size}.")
        size = size.floor()

        x_diff = input_size[1] - size[:, 1] + 1
        y_diff = input_size[0] - size[:, 0] + 1

        # Start point will be 0 if diff < 0
        x_diff = x_diff.clamp(0)
        y_diff = y_diff.clamp(0)

        if self.same_on_batch:
            # If same_on_batch, select the first then repeat.
            x_start = (torch.rand(1, device=_device, dtype=_dtype).expand(batch_size) * x_diff[0]).floor()
            y_start = (torch.rand(1, device=_device, dtype=_dtype).expand(batch_size) * y_diff[0]).floor()
        else:
            x_start = (torch.rand(batch_size, device=_device, dtype=_dtype) * x_diff).floor()
            y_start = (torch.rand(batch_size, device=_device, dtype=_dtype) * y_diff).floor()

        crop_src = bbox_generator(
            x_start.view(-1).to(device=_device, dtype=_dtype),
            y_start.view(-1).to(device=_device, dtype=_dtype),
            torch.where(size[:, 1] == 0, torch.tensor(input_size[1], device=_device, dtype=_dtype), size[:, 1]),
            torch.where(size[:, 0] == 0, torch.tensor(input_size[0], device=_device, dtype=_dtype), size[:, 0]),
        )

        crop_dst = bbox_generator(
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            size[:, 1],
            size[:, 0],
        )
        _output_size = size.to(dtype=torch.long)
        _input_size = torch.tensor(input_size, device=_device, dtype=torch.long).expand(batch_size, -1)

        return {"src": crop_src, "dst": crop_dst, "input_size": _input_size, "output_size": _output_size}

    def compute_padding(self, shape: Tuple[int, ...], flags: Optional[Dict[str, Any]] = None) -> List[int]:
        flags = self.flags if flags is None else flags
        if len(shape) != 4:
            raise AssertionError(f"Expected BCHW. Got {shape}.")
        padding = [0, 0, 0, 0]  # left, right, top, bottom
        if flags["padding"] is not None:
            if isinstance(flags["padding"], int):
                padding = [flags["padding"]] * 4
            elif isinstance(flags["padding"], tuple) and len(flags["padding"]) == 2:
                padding = [flags["padding"][0], flags["padding"][0], flags["padding"][1], flags["padding"][1]]
            elif isinstance(flags["padding"], tuple) and len(flags["padding"]) == 4:
                padding = [flags["padding"][0], flags["padding"][2], flags["padding"][1], flags["padding"][3]]
            else:
                raise RuntimeError(f"Expect `padding` to be a scalar, or length 2/4 list. Got {flags['padding']}.")

        if flags["pad_if_needed"]:
            needed_padding: Tuple[int, int] = (flags["size"][0] - shape[-2], flags["size"][1] - shape[-1])  # HW
            # If crop width is larger than input width F.pad equally left and right
            if needed_padding[1] > 0:
                # Only use the extra padding if actually needed after possible fixed padding
                padding[0] = max(needed_padding[1], padding[0])
                padding[1] = max(needed_padding[1], padding[1])
            # If crop height is larger than input height F.pad equally top and bottom
            if needed_padding[0] > 0:
                # Only use the extra padding if actually needed after possible fixed padding
                padding[2] = max(needed_padding[0], padding[2])
                padding[3] = max(needed_padding[0], padding[3])
        return padding

    def precrop_padding(
        self, input: torch.Tensor, padding: Optional[List[int]] = None, flags: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        flags = self.flags if flags is None else flags
        if padding is None:
            padding = self.compute_padding(input.shape)

        if any(padding):
            input = F.pad(input, padding, value=flags["fill"], mode=flags["padding_mode"])

        return input

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        if flags["cropping_mode"] in ("resample", "slice"):
            src = params["src"].to(input)
            dst = params["dst"].to(input)
            transform: torch.Tensor = get_perspective_transform(src, dst)

            # Fast scaling correction when output exceeds input and padding disabled
            if not flags.get("pad_if_needed", False):
                h, w = input.shape[-2:]
                h_out, w_out = flags["size"]
                if h_out > h or w_out > w:
                    transform[:, 0, 0] *= w_out / w
                    transform[:, 1, 1] *= h_out / h

            return transform
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_transform_keypoint(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        # For F.pad the keypoints properly.
        padding_size = params["padding_size"].to(device=input.device)
        input = input.pad(padding_size)
        return super().apply_transform_keypoint(input=input, params=params, flags=flags, transform=transform)

    def apply_transform_box(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Boxes:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        # For F.pad the boxes properly.
        padding_size = params["padding_size"]
        input = input.pad(padding_size)
        return super().apply_transform_box(input=input, params=params, flags=flags, transform=transform)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        padding_size: Optional[List[int]] = None
        if "padding_size" in params and isinstance(params["padding_size"], torch.Tensor):
            padding_size = params["padding_size"].unique(dim=0).cpu().squeeze().tolist()
        input = self.precrop_padding(input, padding_size, flags)

        flags = self.flags if flags is None else flags
        if flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            if not isinstance(transform, torch.Tensor):
                raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")
            # Fit the arg to F.F.pad
            if flags["padding_mode"] == "constant":
                padding_mode = "zeros"
            elif flags["padding_mode"] == "replicate":
                padding_mode = "border"
            elif flags["padding_mode"] == "reflect":
                padding_mode = "reflection"
            else:
                padding_mode = flags["padding_mode"]

            return crop_by_transform_mat(
                input,
                transform,
                flags["size"],
                mode=flags["resample"].name.lower(),
                padding_mode=padding_mode,
                align_corners=flags["align_corners"],
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
            raise RuntimeError("`size` has to be a tuple. Got None.")
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")
        # Fit the arg to F.F.pad
        if flags["padding_mode"] == "constant":
            padding_mode = "zeros"
        elif flags["padding_mode"] == "replicate":
            padding_mode = "border"
        elif flags["padding_mode"] == "reflect":
            padding_mode = "reflection"
        else:
            padding_mode = flags["padding_mode"]
        return crop_by_transform_mat(
            input,
            transform[:, :2, :],
            size,
            flags["resample"].name.lower(),
            padding_mode=padding_mode,
            align_corners=flags["align_corners"],
        )

    def inverse_inputs(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        out = super().inverse_inputs(input, params, flags, transform, **kwargs)
        if not params["batch_prob"].all():
            return out
        padding_size = params["padding_size"].unique(dim=0).cpu().squeeze().tolist()
        padding_size = [-padding_size[0], -padding_size[1], -padding_size[2], -padding_size[3]]
        return self.precrop_padding(out, padding_size)

    def inverse_boxes(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Boxes:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        output = super().inverse_boxes(input, params, flags, transform, **kwargs)
        if not params["batch_prob"].all():
            return output

        return output.unpad(params["padding_size"])

    def inverse_keypoints(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Keypoints:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        output = super().inverse_keypoints(input, params, flags, transform, **kwargs)
        if not params["batch_prob"].all():
            return output

        return output.unpad(params["padding_size"].to(device=input.device))

    # Override parameters for precrop
    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        input_pad = self.compute_padding(batch_shape)
        batch_shape_new = torch.Size(
            (
                *batch_shape[:2],
                batch_shape[2] + input_pad[2] + input_pad[3],  # original height + top + bottom padding
                batch_shape[3] + input_pad[0] + input_pad[1],  # original width + left + right padding
            )
        )
        padding_size = torch.tensor(tuple(input_pad), dtype=torch.long).expand(batch_shape[0], -1)
        _params = super().forward_parameters(batch_shape_new)
        _params.update({"padding_size": padding_size})
        return _params
