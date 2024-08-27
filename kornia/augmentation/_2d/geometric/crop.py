from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.core import Tensor, pad, tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform


class RandomCrop(GeometricAugmentationBase2D):
    r"""Crop random patches of a tensor image on a given size.

    .. image:: _static/img/RandomCrop.png

    Args:
        size: Desired output size (out_h, out_w) of the crop.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        padding: Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed: It will pad the image if smaller than the
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
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
                       on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                       to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                       differentiability.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

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
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self._param_generator = rg.CropGenerator(size)
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
            # If crop width is larger than input width pad equally left and right
            if needed_padding[1] > 0:
                # Only use the extra padding if actually needed after possible fixed padding
                padding[0] = max(needed_padding[1], padding[0])
                padding[1] = max(needed_padding[1], padding[1])
            # If crop height is larger than input height pad equally top and bottom
            if needed_padding[0] > 0:
                # Only use the extra padding if actually needed after possible fixed padding
                padding[2] = max(needed_padding[0], padding[2])
                padding[3] = max(needed_padding[0], padding[3])
        return padding

    def precrop_padding(
        self, input: Tensor, padding: Optional[List[int]] = None, flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        flags = self.flags if flags is None else flags
        if padding is None:
            padding = self.compute_padding(input.shape)

        if any(padding):
            input = pad(input, padding, value=flags["fill"], mode=flags["padding_mode"])

        return input

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        if flags["cropping_mode"] in ("resample", "slice"):
            transform: Tensor = get_perspective_transform(params["src"].to(input), params["dst"].to(input))
            return transform
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_transform_keypoint(
        self, input: Keypoints, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        # For pad the keypoints properly.
        padding_size = params["padding_size"].to(device=input.device)
        input = input.pad(padding_size)
        return super().apply_transform_keypoint(input=input, params=params, flags=flags, transform=transform)

    def apply_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        # For pad the boxes properly.
        padding_size = params["padding_size"]
        input = input.pad(padding_size)
        return super().apply_transform_box(input=input, params=params, flags=flags, transform=transform)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        padding_size: Optional[List[int]] = None
        if "padding_size" in params and isinstance(params["padding_size"], Tensor):
            padding_size = params["padding_size"].unique(dim=0).cpu().squeeze().tolist()
        input = self.precrop_padding(input, padding_size, flags)

        flags = self.flags if flags is None else flags
        if flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            if not isinstance(transform, Tensor):
                raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")
            # Fit the arg to F.pad
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
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        if size is None:
            raise RuntimeError("`size` has to be a tuple. Got None.")
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")
        # Fit the arg to F.pad
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
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
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
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
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
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
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
    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        input_pad = self.compute_padding(batch_shape)
        batch_shape_new = torch.Size(
            (
                *batch_shape[:2],
                batch_shape[2] + input_pad[2] + input_pad[3],  # original height + top + bottom padding
                batch_shape[3] + input_pad[0] + input_pad[1],  # original width + left + right padding
            )
        )
        padding_size = tensor(tuple(input_pad), dtype=torch.long).expand(batch_shape[0], -1)
        _params = super().forward_parameters(batch_shape_new)
        _params.update({"padding_size": padding_size})
        return _params
