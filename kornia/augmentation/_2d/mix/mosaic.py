from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from kornia.core import Tensor
from torch.nn.functional import interpolate

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey, Resample
from kornia.geometry.boxes import Boxes
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform
from kornia.utils import eye_like


class RandomMosaic(MixAugmentationBaseV2):
    r"""RandomMosaic.

    Args:
        min_bbox_size: minimum area of bounding boxes. Default to 0.
        p: probability for applying an augmentation. This param controls if to apply the augmentation for the batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Examples:
        >>> mosaic = RandomMosaic()
        >>> input = torch.randn(8, 3, 16, 16)
        >>> mosaic(input).shape
    """

    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        start_corner_range: Tuple[float, float] = (0.3, 0.7),
        mosaic_grid: Tuple[int, int] = (2, 2),
        min_bbox_size: float = 0.,
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        p: float = .7,
        keepdim: bool = False,
        padding_mode: str = "constant",
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        align_corners: bool = True,
        cropping_mode: str = "slice",
    ) -> None:
        super().__init__(p=p, p_batch=1., same_on_batch=False, keepdim=keepdim, data_keys=data_keys)
        self.start_corner_range = start_corner_range
        self._param_generator = cast(
            rg.MosaicGenerator, rg.MosaicGenerator(output_size, mosaic_grid, start_corner_range))
        self.flags = dict(
            mosaic_grid=mosaic_grid,
            output_size=output_size,
            min_bbox_size=min_bbox_size,
            padding_mode=padding_mode,
            resample=Resample.get(resample),
            align_corners=align_corners,
            cropping_mode=cropping_mode,
        )

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def apply_transform_boxes(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Boxes:
        # Boxes is BxNx4x2 only.
        batch_shapes = params["batch_shapes"]
        offset = torch.zeros((input.data.shape[0], 2), device=batch_shapes.device, dtype=params["src"].dtype)  # Bx2
        # NOTE: not a pretty good line I think.
        offset_end = params["dst"][0, 2].repeat(input.data.shape[0], 1)
        idx = torch.arange(0, input.data.shape[0], device=offset.device, dtype=torch.long)[params["batch_prob"]]

        out_boxes: Optional[Boxes] = None
        for i in range(flags['mosaic_grid'][1]):
            for j in range(flags['mosaic_grid'][0]):
                _offset = offset.clone()
                _offset[idx, 0] = batch_shapes[:, -2] * i - params["src"][:, 0, 0]
                _offset[idx, 1] = batch_shapes[:, -1] * j - params["src"][:, 0, 1]
                _box = input.clone()
                _box._data = _box._data[params["permutation"][:, i * flags['mosaic_grid'][1] + j]]
                _box.translate(_offset, inplace=True)
                # zero-out unrelated batch elements.
                _box._data[~params["batch_prob"]] = 0
                if out_boxes is None:
                    out_boxes = _box
                out_boxes.merge(_box, inplace=True)
        out_boxes.clamp(offset, offset_end, inplace=True)
        out_boxes.filter_boxes_by_area(flags["min_bbox_size"], inplace=True)
        return out_boxes

    def apply_transform_keypoint(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        raise NotImplementedError

    def apply_transform_tag(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        raise RuntimeError(f"{__class__.__name__} does not support `TAG` types.")

    @torch.no_grad()
    def _compose_images(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        base = input[params["permutation"][:, 0]]
        for i in range(1, flags['mosaic_grid'][0]):
            base = torch.cat([
                base,
                input[params["permutation"][:, i]]
            ], dim=-2)
        if flags['mosaic_grid'][1] == 1:
            # No need to concatenate later
            return base

        base_2 = input[params["permutation"][:, flags['mosaic_grid'][0]]]
        for i in range(1, flags['mosaic_grid'][1]):
            base_2 = torch.cat([
                base_2,
                input[params["permutation"][:, flags['mosaic_grid'][0] + i]]
            ], dim=-2)
        return torch.cat([base, base_2], dim=-1)

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
            transform = cast(Tensor, transform)
            # Fit the arg to F.pad
            if flags['padding_mode'] == "constant":
                padding_mode = "zeros"
            elif flags['padding_mode'] == "replicate":
                padding_mode = "border"
            elif flags['padding_mode'] == "reflect":
                padding_mode = "reflection"
            else:
                padding_mode = flags['padding_mode']

            return crop_by_transform_mat(
                input,
                transform,
                flags["output_size"],
                mode=flags["resample"].name.lower(),
                padding_mode=padding_mode,
                align_corners=flags["align_corners"],
            )
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            return crop_by_indices(input, params["src"], flags["output_size"])
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_non_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        if flags["output_size"] is not None:
            return interpolate(
                input,
                size=flags["output_size"],
                mode=flags["resample"].name.lower(),
                align_corners=flags["align_corners"]
            )
        return input

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        input = self._compose_images(input, params, flags=flags)
        transform = self.compute_transformation(input, params, flags=flags)
        input = self._crop_images(input, params, flags=flags, transform=transform)
        return input
