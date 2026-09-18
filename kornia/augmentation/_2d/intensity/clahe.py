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

from __future__ import annotations

from typing import Any, Optional

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import equalize_clahe


class RandomClahe(IntensityAugmentationBase2D):
    r"""Apply CLAHE equalization on the input torch.Tensor randomly.

    .. image:: _static/img/equalize_clahe.png

    See the Convention block on :class:`~kornia.augmentation.IntensityAugmentationBase2D`. This class raises
    instead of transforming an out-of-range input, and its error is a raw one.

    Args:
        clip_limit: the ``(low, high)`` range the per-sample contrast-limiting threshold is drawn from.
            Unlike :func:`kornia.enhance.equalize_clahe`'s scalar argument of the same name this must be a
            two-element tuple -- a scalar raises ``ValueError: `center` and `bounds` cannot be None for
            single number`` at construction -- and ``(0.0, 0.0)`` is what disables clipping. The bound is
            unvalidated below zero: ``(-1.0, -1.0)`` is accepted and behaves exactly like ``(0.0, 0.0)``.
        grid_size: number of tiles to be cropped in each direction (GH, GW).
        slow_and_differentiable: selects the implementation. At the default ``False`` the fast path breaks the
            autograd graph -- the output has ``requires_grad=False`` and no ``grad_fn``, which no other 2D
            intensity augmentation does -- so set it to ``True`` to keep the class differentiable.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Convention:
        - an input outside ``[0, 1]`` raises a ``RuntimeError`` naming
          :func:`kornia.enhance.equalize_clahe` and that range, the way :class:`RandomEqualize` raises for
          :func:`kornia.enhance.equalize`. As for :class:`RandomEqualize`, the rejection is not exactly at
          the boundary: the check guards the 256-entry lookup indexed with ``(input * 255).long()``, so a
          value less than one 8-bit code outside ``[0, 1]``, at either end, is still admitted, up to the
          rounding of ``input * 255`` in the input's dtype. The check runs on the CPU and on CUDA. **It is
          skipped on MPS by design** -- materializing the condition there would drain the queued stream on
          every call -- so an MPS image keeps the pre-existing behaviour: torch ``2.14`` raises the raw
          ``gather`` error (`#4600 <https://github.com/kornia/kornia/issues/4600>`_), while ``2.5.1``
          leaves the gather unchecked and returns an in-range image as if the input had been valid.

    .. warning::
        ``clip_limit`` is drawn per sample, but the first sample's value is applied to the whole batch.
        Tracked in `#4572 <https://github.com/kornia/kornia/issues/4572>`_.

    .. warning::
        ``grid_size`` is unvalidated past its positivity check. Its two
        entries tile the two axes independently, and a grid that does not tile the image is padded instead, so
        ``grid_size=(3, 3)`` works on a ``10 x 10`` image. Because :func:`kornia.enhance.equalize_clahe` rounds
        the tile up to an even size along each axis, an exactly dividing grid can still pad: ``(4, 5)`` pads 4
        rows and no columns of a ``20 x 20`` image. An image too small for the grid raises a raw
        ``RuntimeError`` from the padding; at the default ``grid_size=(8, 8)`` the smallest admissible square
        image is ``9 x 9``, and ``8 x 8`` raises. A grid larger than the image gets the named ``ValueError``
        instead.

    .. note::
        This function internally uses :func:`kornia.enhance.equalize_clahe`, which expects the input in
        :math:`[0, 1]` and raises a ``RuntimeError`` naming that range for values its 256-bin lookup
        cannot index.

    Examples:
        >>> img = torch.rand(1, 10, 20)
        >>> aug = RandomClahe()
        >>> res = aug(img)
        >>> res.shape
        torch.Size([1, 1, 10, 20])

        >>> img = torch.rand(2, 3, 10, 20)
        >>> aug = RandomClahe()
        >>> res = aug(img)
        >>> res.shape
        torch.Size([2, 3, 10, 20])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomClahe(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        clip_limit: tuple[float, float] = (40.0, 40.0),
        grid_size: tuple[int, int] = (8, 8),
        slow_and_differentiable: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.clip_limit = clip_limit
        self._param_generator = rg.PlainUniformGenerator((self.clip_limit, "clip_limit_factor", None, None))
        self.flags = {"grid_size": grid_size, "slow_and_differentiable": slow_and_differentiable}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        clip_limit = float(params["clip_limit_factor"][0])
        return equalize_clahe(input, clip_limit, flags["grid_size"], flags["slow_and_differentiable"])
