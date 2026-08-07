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
from kornia.geometry.transform import crop_by_transform_mat, get_perspective_transform, resize


class Resize(GeometricAugmentationBase2D):
    """Resize to size.

    Args:
        size: Size (h, w) in pixels of the resized region or just one side.
        side: Which side to resize, if size is only of type int.
        resample: Resampling mode.
        align_corners: interpolation flag.
        antialias: if True, then image will be filtered with Gaussian before downscaling. No effect for upscaling.
        p: probability of the augmentation been applied.
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
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=1.0, same_on_batch=True, p_batch=p, keepdim=keepdim)
        self._param_generator = rg.ResizeGenerator(resize_to=size, side=side)
        self.flags = {
            "size": size,
            "side": side,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
            "antialias": antialias,
        }

    # apply_transform does a batched resize and ignores the transform matrix, so defer the
    # matrix build (which needs a linalg solve) until `.transform_matrix` is read.
    _compute_matrix_lazily = True

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        # NOTE: a former `if params["output_size"] == input.shape[-2:]: return eye_like(...)`
        # short-circuit was dead code — comparing a tensor to a ``torch.Size`` falls back to
        # identity ``==`` and is *always* Python ``False``, so the branch never ran. It also
        # graph-broke torch.compile (a Python ``if`` on a would-be tensor). Dropped; the
        # perspective transform below already yields identity when src == dst, so behaviour is
        # byte-identical.
        transform: torch.Tensor = torch.as_tensor(
            get_perspective_transform(params["src"], params["dst"]), dtype=input.dtype, device=input.device
        )
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The generator always sets `src` to the full input box, so the per-sample crop is a
        # no-op and this reduces to a single batched resize of the whole input to `output_size`.
        # Using the static `flags["size"]` for a tuple size keeps this torch.compile
        # fullgraph-safe (an int side depends on the input aspect ratio, so it falls back to the
        # data-dependent `output_size` param).
        if isinstance(flags["size"], (tuple, list)):
            out_size: Tuple[int, int] = (int(flags["size"][0]), int(flags["size"][1]))
        else:
            out_size = tuple(params["output_size"][0].tolist())
        return resize(
            input,
            out_size,
            interpolation=flags["resample"].name.lower(),
            align_corners=(
                flags["align_corners"] if flags["resample"] in [Resample.BILINEAR, Resample.BICUBIC] else None
            ),
            antialias=flags["antialias"],
        )

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not isinstance(size, tuple):
            raise TypeError(f"Expected the size be a tuple. Gotcha {type(size)}")

        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")

        return crop_by_transform_mat(
            input, transform[:, :2, :], size, flags["resample"].name.lower(), "zeros", flags["align_corners"]
        )


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
    ) -> None:
        # TODO: Support max_size list input to randomly select from
        super().__init__(size=max_size, side="long", resample=resample, align_corners=align_corners, p=p)


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
    ) -> None:
        # TODO: Support max_size list input to randomly select from
        super().__init__(size=max_size, side="short", resample=resample, align_corners=align_corners, p=p)
