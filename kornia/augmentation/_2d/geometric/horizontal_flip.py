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

# from torch import Tensor (use torch.Tensor instead)
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.geometry.transform import hflip


class RandomHorizontalFlip(GeometricAugmentationBase2D):
    r"""Apply a random horizontal flip to a torch.tensor image or a batch of torch.tensor images.

    The flip is applied with a given probability.

    .. image:: _static/img/RandomHorizontalFlip.png

    Input should be a torch.tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and torch.cat the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p: probability of the image being flipped.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.hflip`.

    Examples:
        >>> import torch
        >>> input = torch.tensor([[[[0., 0., 0.],
        ...                         [0., 0., 0.],
        ...                         [0., 1., 1.]]]])
        >>> seq = RandomHorizontalFlip(p=1.0)
        >>> seq(input), seq.transform_matrix
        (torch.tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]]), torch.tensor([[[-1.,  0.,  2.],
                 [ 0.,  1.,  0.],
                 [ 0.,  0.,  1.]]]))
        >>> seq.inverse(seq(input)).equal(input)
        True

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> seq = RandomHorizontalFlip(p=1.0)
        >>> (seq(input) == seq(input, params=seq._params)).all()
        torch.tensor(True)

    """

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        w: int = int(params["forward_input_shape"][-1])
        flip_mat: torch.Tensor = torch.tensor(
            [[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], device=input.device, dtype=input.dtype
        )

        return flip_mat.expand(input.shape[0], 3, 3)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return hflip(input)

    def inverse_transform(
        self,
        input: torch.Tensor,
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not isinstance(transform, torch.Tensor):
            raise TypeError(f"Expected the `transform` be a torch.Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=torch.as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
