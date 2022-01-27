from typing import Dict, Optional, Tuple, Union, cast

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.geometry.transform import crop_by_transform_mat, get_perspective_transform, resize


class Resize(GeometricAugmentationBase2D):
    """Resize to size.

    Parameters
    ----------
    size : Union[int, Tuple[int, int]]
        Size (h, w) in pixels of the resized region or just one side.
    side: str
        Which side to resize, if size is only of type int.
    resample : Union[str, int, Resample], optional
        Resampling mode.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        side: str = "short",
        cropping_mode: str = "slice",
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim
        )
        self._param_generator = cast(rg.ResizeGenerator, rg.ResizeGenerator(resize_to=size, side=side))
        self.flags = dict(
            size=size,
            side=side,
            resample=Resample.get(resample),
            align_corners=align_corners,
            cropping_mode=cropping_mode,
        )

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_size = h, w = input.shape[-2:]
        if params["resize_to"] == input_size:
            return torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)

        transform: torch.Tensor = get_perspective_transform(params["src"], params["dst"])
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            transform = cast(torch.Tensor, transform)
            return crop_by_transform_mat(
                input,
                transform,
                params["resize_to"],
                mode=self.flags["resample"].name.lower(),
                padding_mode="zeros",
                align_corners=self.flags["align_corners"],
            )
        if self.flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            B, C, _, _ = input.shape
            out = torch.empty(B, C, *params["resize_to"], device=input.device, dtype=input.dtype)
            for i in range(B):
                x1 = int(params["src"][i, 0, 0])
                x2 = int(params["src"][i, 1, 0]) + 1
                y1 = int(params["src"][i, 0, 1])
                y2 = int(params["src"][i, 3, 1]) + 1
                out[i] = resize(
                    input[i : i + 1, :, y1:y2, x1:x2],
                    params["resize_to"],
                    interpolation=(self.flags["resample"].name).lower(),
                    align_corners=self.flags["align_corners"],
                )
            return out
        raise NotImplementedError(f"Not supported type: {self.flags['cropping_mode']}.")

    def inverse_transform(
        self,
        input: torch.Tensor,
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {self.flags['cropping_mode']}."
            )
        size = cast(Tuple[int, int], size)
        mode = self.flags["resample"].name.lower() if "mode" not in kwargs else kwargs["mode"]
        align_corners = self.flags["align_corners"] if "align_corners" not in kwargs else kwargs["align_corners"]
        padding_mode = "zeros" if "padding_mode" not in kwargs else kwargs["padding_mode"]
        transform = cast(torch.Tensor, transform)
        return crop_by_transform_mat(input, transform[:, :2, :], size, mode, padding_mode, align_corners)
