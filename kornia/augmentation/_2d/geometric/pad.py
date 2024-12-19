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

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`torch.nn.functional.pad`.

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

    """

    def __init__(
        self, size: Tuple[int, int], pad_mode: str = "constant", pad_value: float = 0, keepdim: bool = False
    ) -> None:
        super().__init__(p=1.0, same_on_batch=True, p_batch=1.0, keepdim=keepdim)
        self.flags = {"size": size, "pad_mode": pad_mode, "pad_value": pad_value}

    # TODO: It is incorrect to return identity
    # TODO: Having a resampled version with ``warp_affine``
    def compute_transformation(self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.identity_matrix(image)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        height_pad: int = flags["size"][0] - height
        width_pad: int = flags["size"][1] - width
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
        return input[..., : size[0], : size[1]]
