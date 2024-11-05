from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey, Resample
from kornia.core import Tensor, as_tensor, concatenate, pad, zeros
from kornia.core.check import KORNIA_UNWRAP
from kornia.geometry.boxes import Boxes
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform
from kornia.utils import eye_like

__all__ = ["RandomMosaic"]


class RandomMosaic(MixAugmentationBaseV2):
    r"""Mosaic augmentation.

    .. image:: https://raw.githubusercontent.com/kornia/data/main/random_mosaic.png

    Given a certain number of images, mosaic transform combines them into one output image.
    The output image is composed of the parts from each sub-image. To mess up each image individually,
    referring to :class:`kornia.augmentation.RandomJigsaw`.

    The mosaic transform steps are as follows:

         1. Concate selected images into a super-image.
         2. Crop out the outcome image according to the top-left corner and crop size.

    Args:
        output_size: the output tensor width and height after mosaicing.
        start_ratio_range: top-left (x, y) position for cropping the mosaic images.
        mosaic_grid: the number of images and image arrangement. e.g. (2, 2) means
            each output will mix 4 images in a 2x2 grid.
        min_bbox_size: minimum area of bounding boxes. Default to 0.
        data_keys: the input type sequential for applying augmentations.
            Accepts "input", "image", "mask", "bbox", "bbox_xyxy", "bbox_xywh", "keypoints",
            "class", "label".
        p: probability of applying the transformation for the whole batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
            to the batch form ``False``.
        padding_mode: Type of padding. Should be: constant, reflect, replicate.
        resample: the interpolation mode.
        align_corners: interpolation flag.
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
            on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
            to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
            differentiability.

    Examples:
        >>> mosaic = RandomMosaic((300, 300), data_keys=["input", "bbox_xyxy"])
        >>> boxes = torch.tensor([[
        ...     [70, 5, 150, 100],
        ...     [60, 180, 175, 220],
        ... ]]).repeat(8, 1, 1)
        >>> input = torch.randn(8, 3, 224, 224)
        >>> out = mosaic(input, boxes)
        >>> out[0].shape, out[1].shape
        (torch.Size([8, 3, 300, 300]), torch.Size([8, 8, 4]))
    """

    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        mosaic_grid: Tuple[int, int] = (2, 2),
        start_ratio_range: Tuple[float, float] = (0.3, 0.7),
        min_bbox_size: float = 0.0,
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        p: float = 0.7,
        keepdim: bool = False,
        padding_mode: str = "constant",
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        align_corners: bool = True,
        cropping_mode: str = "slice",
    ) -> None:
        super().__init__(p=p, p_batch=1.0, same_on_batch=False, keepdim=keepdim, data_keys=data_keys)
        self.start_ratio_range = start_ratio_range
        self._param_generator = rg.MosaicGenerator(output_size, mosaic_grid, start_ratio_range)

        self.flags = {
            "mosaic_grid": mosaic_grid,
            "output_size": output_size,
            "min_bbox_size": min_bbox_size,
            "padding_mode": padding_mode,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
            "cropping_mode": cropping_mode,
        }

    def apply_transform_mask(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def apply_transform_boxes(self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Boxes:
        to_apply = params["batch_prob"] > 0.5
        src_box = as_tensor(params["src"], device=input.device, dtype=input.dtype)
        dst_box = as_tensor(params["dst"], device=input.device, dtype=input.dtype)
        # Boxes is BxNx4x2 only.
        batch_shapes = as_tensor(params["batch_shapes"], device=input.device, dtype=input.dtype)
        offset = zeros((len(to_apply), 2), device=input.device, dtype=input.dtype)  # Bx2
        # NOTE: not a pretty good line I think.
        offset_end = dst_box[0, 2].repeat(input.data.shape[0], 1)
        idx = torch.arange(0, input.data.shape[0], device=input.device, dtype=torch.long)[to_apply]

        maybe_out_boxes: Optional[Boxes] = None
        for i in range(flags["mosaic_grid"][0]):
            for j in range(flags["mosaic_grid"][1]):
                _offset = offset.clone()
                _offset[idx, 0] = batch_shapes[:, -2] * i - src_box[:, 0, 0]
                _offset[idx, 1] = batch_shapes[:, -1] * j - src_box[:, 0, 1]
                _box = input.clone()
                _idx = i * flags["mosaic_grid"][1] + j
                _box._data[params["permutation"][:, 0]] = _box._data[params["permutation"][:, _idx]]
                _box.translate(_offset, inplace=True)
                # zero-out unrelated batch elements.
                _box._data[~to_apply] = 0
                if maybe_out_boxes is None:
                    _box._data[~to_apply] = input._data[~to_apply]
                    maybe_out_boxes = _box
                else:
                    KORNIA_UNWRAP(maybe_out_boxes, Boxes).merge(_box, inplace=True)
        out_boxes: Boxes = KORNIA_UNWRAP(maybe_out_boxes, Boxes)
        out_boxes.clamp(offset, offset_end, inplace=True)
        out_boxes.filter_boxes_by_area(flags["min_bbox_size"], inplace=True)
        return out_boxes

    def apply_transform_keypoint(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def apply_transform_class(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise RuntimeError(f"{self.__class__.__name__} does not support `TAG` types.")

    @torch.no_grad()
    def _compose_images(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        out = []
        for i in range(flags["mosaic_grid"][0]):
            out_row = []
            for j in range(flags["mosaic_grid"][1]):
                img_idx = flags["mosaic_grid"][1] * i + j
                image = input[params["permutation"][:, img_idx]]
                out_row.append(image)
            out.append(concatenate(out_row, -2))
        return concatenate(out, -1)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        if flags["cropping_mode"] == "resample":
            transform: Tensor = get_perspective_transform(params["src"].to(input), params["dst"].to(input))
            return transform
        if flags["cropping_mode"] == "slice":  # Skip the computation for slicing.
            return eye_like(3, input)
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def _crop_images(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        flags = self.flags if flags is None else flags
        if flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            if not isinstance(transform, Tensor):
                raise TypeError(f"Expected the transform to be a Tensor. Gotcha {type(transform)}")

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
                flags["output_size"],
                mode=flags["resample"].name.lower(),
                padding_mode=padding_mode,
                align_corners=flags["align_corners"],
            )
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            return crop_by_indices(input, params["src"], flags["output_size"], shape_compensation="pad")
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_non_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        if flags is not None and flags["output_size"] is not None:
            output_size = KORNIA_UNWRAP(flags["output_size"], Tuple[int, int])
            return pad(input, [0, output_size[1] - input.shape[-1], 0, output_size[0] - input.shape[-2]])
            # NOTE: resize is not suitable for being consistent with bounding boxes.
            # return resize(
            #     input,
            #     size=flags["output_size"],
            #     interpolation=flags["resample"].name.lower(),
            #     align_corners=flags["align_corners"]
            # )
        return input

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        flags = KORNIA_UNWRAP(maybe_flags, Dict[str, Any])
        output = self._compose_images(input, params, flags=flags)
        transform = self.compute_transformation(output, params, flags=flags)
        output = self._crop_images(output, params, flags=flags, transform=transform)
        return output
