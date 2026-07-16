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
    r"""Apply a random horizontal flip to a torch.Tensor image or a batch of torch.Tensor images.

    The flip is applied with a given probability.

    .. image:: _static/img/RandomHorizontalFlip.png

    Input should be a torch.Tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
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
        (tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]]), tensor([[[-1.,  0.,  2.],
                 [ 0.,  1.,  0.],
                 [ 0.,  0.,  1.]]]))
        >>> seq.inverse(seq(input)).equal(input)
        True

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> seq = RandomHorizontalFlip(p=1.0)
        >>> (seq(input) == seq(input, params=seq._params)).all()
        tensor(True)

    """

    # apply_transform is a pure flip that ignores the transform matrix, so defer building
    # it until `.transform_matrix` is read (see RigidAffineAugmentationBase2D).
    _compute_matrix_lazily = True

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        # Build the 3x3 flip matrix as a constant literal plus the (tensor) width, rather than
        # `int(w)`. `int(tensor)` lowers to `.item()`, which breaks torch.compile fullgraph on
        # some torch versions; a Python-float literal is fine (constant-folded) and adding the
        # width tensor into entry [0, 2] gives `w - 1` — numerically identical to
        # `[[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]]`, and ~1.8x cheaper than stacking 9 scalars.
        w = params["forward_input_shape"][-1].to(device=input.device, dtype=input.dtype)
        flip_mat: torch.Tensor = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=input.device, dtype=input.dtype
        )
        flip_mat[0, 2] = flip_mat[0, 2] + w

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
