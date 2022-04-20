from typing import Dict, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.geometry.transform import crop_by_transform_mat, get_perspective_transform, resize
from kornia.utils import eye_like


class Resize(GeometricAugmentationBase2D):
    """Resize to size.

    Args:
        size: Size (h, w) in pixels of the resized region or just one side.
        side: Which side to resize, if size is only of type int.
        resample: Resampling mode.
        align_corners: interpolation flag.
        antialias: if True, then image will be filtered with Gaussian before downscaling. No effect for upscaling.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False).
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        side: str = "short",
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        align_corners: bool = True,
        antialias: bool = False,
        p: float = 1.0,
        return_transform: Optional[bool] = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=1., return_transform=return_transform, same_on_batch=True, p_batch=p, keepdim=keepdim)
        self._param_generator = cast(rg.ResizeGenerator, rg.ResizeGenerator(resize_to=size, side=side))
        self.flags = dict(
            size=size,
            side=side,
            resample=Resample.get(resample),
            align_corners=align_corners,
            antialias=antialias
        )

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor]) -> Tensor:
        if params["output_size"] == input.shape[-2:]:
            return eye_like(3, input)

        transform: Tensor = get_perspective_transform(params["src"], params["dst"])
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        B, C, _, _ = input.shape
        out_size = tuple(params["output_size"][0].tolist())
        out = torch.empty(B, C, *out_size, device=input.device, dtype=input.dtype)
        for i in range(B):
            x1 = int(params["src"][i, 0, 0])
            x2 = int(params["src"][i, 1, 0]) + 1
            y1 = int(params["src"][i, 0, 1])
            y2 = int(params["src"][i, 3, 1]) + 1
            out[i] = resize(
                input[i : i + 1, :, y1:y2, x1:x2],
                out_size,
                interpolation=(self.flags["resample"].name).lower(),
                align_corners=self.flags["align_corners"],
                antialias=self.flags["antialias"]
            )
        return out

    def inverse_transform(
        self,
        input: Tensor,
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tensor:
        size = cast(Tuple[int, int], size)
        mode = self.flags["resample"].name.lower() if "mode" not in kwargs else kwargs["mode"]
        align_corners = self.flags["align_corners"] if "align_corners" not in kwargs else kwargs["align_corners"]
        padding_mode = "zeros" if "padding_mode" not in kwargs else kwargs["padding_mode"]
        transform = cast(Tensor, transform)
        return crop_by_transform_mat(input, transform[:, :2, :], size, mode, padding_mode, align_corners)


class LongestMaxSize(Resize):
    """Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size: maximum size of the image after the transformation.
    """

    def __init__(
        self,
        max_size: int,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        align_corners: bool = True,
        p: float = 1.0,
        return_transform: Optional[bool] = None,
    ) -> None:
        # TODO: Support max_size list input to randomly select from
        super().__init__(
            size=max_size,
            side="long",
            resample=resample,
            return_transform=return_transform,
            align_corners=align_corners,
            p=p,
        )


class SmallestMaxSize(Resize):
    """Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size: maximum size of the image after the transformation.
    """

    def __init__(
        self,
        max_size: int,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        align_corners: bool = True,
        p: float = 1.0,
        return_transform: Optional[bool] = None,
    ) -> None:
        # TODO: Support max_size list input to randomly select from
        super().__init__(
            size=max_size,
            side="short",
            resample=resample,
            return_transform=return_transform,
            align_corners=align_corners,
            p=p,
        )
