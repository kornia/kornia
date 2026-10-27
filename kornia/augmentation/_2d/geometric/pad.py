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

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D


class PadTo(GeometricAugmentationBase2D):
    r"""Pad the given sample to a specific size. Always occurs (p=1.0).

    .. image:: _static/img/PadTo.png

    Args:
        size: a tuple of ints in the format (height, width) that give the spatial
            dimensions to pad inputs to.
        pad_mode: the type of padding to perform on the image (valid values
            are those accepted by torch.nn.functional.pad)
        pad_value: fill value for 'constant' padding applied to the image
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
        crop_if_exceeds: if True (default), an input dimension larger than ``size`` is cropped to ``size``. If
            False, that dimension passes through unchanged, so the output is at least ``size`` on each axis and its
            shape depends on the input.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`torch.nn.functional.pad`.

    Convention:
        See :class:`~kornia.augmentation.GeometricAugmentationBase2D` for coordinates, defaults and inverse.
        This operation always runs and pads on the right and bottom, using ``pad_mode`` and ``pad_value``. With the
        default ``crop_if_exceeds=True``, a target smaller than the input crops that axis instead, so a target can pad
        one axis while cropping the other. With ``crop_if_exceeds=False``, that axis passes through unchanged.

        :meth:`inverse` restores the prior input shape on every axis. It exactly removes right/bottom padding, and it
        regrows a cropped axis at the right/bottom with zeros, as the crop augmentations' inverses do: the discarded
        pixels cannot be recovered. The transformation matrix is identity, because every kept pixel keeps its
        coordinates, so keypoints and boxes are neither moved nor clipped when cropping puts them outside the canvas.

    Examples:
        >>> import torch
        >>> img = torch.tensor([[[[0., 0., 0.],
        ...                       [0., 0., 0.],
        ...                       [0., 0., 0.]]]])
        >>> pad = PadTo((4, 5), pad_value=1.)
        >>> out = pad(img)
        >>> out
        tensor([[[[0., 0., 0., 1., 1.],
                  [0., 0., 0., 1., 1.],
                  [0., 0., 0., 1., 1.],
                  [1., 1., 1., 1., 1.]]]])
        >>> pad.inverse(out)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])

        >>> crop = PadTo((2, 4))
        >>> crop.inverse(crop(torch.ones(1, 1, 3, 3)))
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [0., 0., 0.]]]])

        >>> img_large = torch.zeros(1, 1, 4, 4)
        >>> PadTo((2, 2), crop_if_exceeds=False)(img_large).shape
        torch.Size([1, 1, 4, 4])

    """

    def __init__(
        self,
        size: Tuple[int, int],
        pad_mode: str = "constant",
        pad_value: float = 0,
        keepdim: bool = False,
        crop_if_exceeds: bool = True,
    ) -> None:
        super().__init__(p=1.0, same_on_batch=True, p_batch=1.0, keepdim=keepdim)
        self.flags = {
            "size": size,
            "pad_mode": pad_mode,
            "pad_value": pad_value,
            "crop_if_exceeds": crop_if_exceeds,
        }

    # PadTo pads and crops only at the right and bottom, so every pixel it keeps stays at the same coordinates: the
    # identity is the exact transformation for keypoints and boxes in both directions.
    def compute_transformation(self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.identity_matrix(image)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        height_pad: int = flags["size"][0] - height
        width_pad: int = flags["size"][1] - width
        # A PadTo pickled before ``crop_if_exceeds`` existed has no such flag and keeps cropping.
        if not flags.get("crop_if_exceeds", True):
            height_pad = max(height_pad, 0)
            width_pad = max(width_pad, 0)
        return torch.nn.functional.pad(
            input, [0, width_pad, 0, height_pad], mode=flags["pad_mode"], value=flags["pad_value"]
        )

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if size is None:
            raise RuntimeError("`size` has to be a tuple. Got None.")
        # Negative padding removes the right/bottom padding; positive padding regrows a cropped axis with zeros
        # rather than ``pad_mode``, which would invent content from the pixels that were kept.
        height, width = input.shape[-2:]
        return torch.nn.functional.pad(input, [0, size[1] - width, 0, size[0] - height], mode="constant", value=0.0)
