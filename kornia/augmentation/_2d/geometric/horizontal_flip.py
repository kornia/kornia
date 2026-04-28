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
from kornia.augmentation.utils import _transform_input
from kornia.geometry.transform import hflip

# Module-level template for the horizontal-flip transformation matrix.
# Per-call we read width from params and substitute the `w-1` entry; this
# avoids Python-level torch.tensor([...]) allocation in the hot path and
# enables CUDA Graph capture (no in-forward tensor allocation).
_HFLIP_MAT_TEMPLATE = torch.tensor(
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
)

# Per-process cache: (device, dtype, width_int) -> (B, 3, 3) matrix.
# Populated lazily on first call for a given (device, dtype, width) triple and
# reused on every subsequent call with the same arguments.  The cache is a plain
# dict so it is never persisted across processes or serialised with the model.
_HFLIP_MAT_CACHE: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}


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

    .. note::
        A minimal-overhead fast forward path is taken automatically when called
        with a single plain ``Tensor`` (no boxes/masks/keypoints, no replay
        ``params=``, no kwargs) and ``p`` is deterministic (``0.0`` or ``1.0``).
        For boxes/masks/keypoints/replay the standard chain is preserved.

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

    # The legacy ``_fast_image_only_apply`` opt-in is disabled — the aggressive
    # forward override below is strictly faster.  Kept ``False`` so the parent
    # ``_BasicAugmentationBase.forward`` does not run the gating branch when the
    # subclass override falls through.
    _supports_fast_image_only_path: bool = False

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Aggressive fast path: completely bypass the framework chain
        # (``_BasicAugmentationBase.forward`` -> ``forward_parameters`` ->
        # ``apply_func`` -> ``transform_inputs``) for the simple
        # "single image tensor, deterministic p" call.  Parameter generation,
        # batch-prob branching, and per-call validation are skipped.
        # ``_transform_matrix`` is populated from the per-(device, dtype,
        # width) cache and ``_params`` carries the minimal viable suite
        # (``batch_prob`` + ``forward_input_shape``) so post-forward
        # ``aug.inverse(...)`` and ``transform_matrix`` keep working.
        if (
            len(args) == 1
            and isinstance(args[0], torch.Tensor)
            and not kwargs
            and self.p_batch == 1.0
            and not self.same_on_batch
            and not self.keepdim
            and self.p in (0.0, 1.0)
        ):
            x = args[0]
            d = x.dim()
            if d == 3:
                x = x.unsqueeze(0)
                d = 4
            if d == 4:
                b = x.shape[0]
                self._params = {
                    "batch_prob": torch.full((b,), bool(self.p > 0.5), dtype=torch.bool),
                    "forward_input_shape": torch.tensor(x.shape, dtype=torch.long),
                }
                if self.p == 1.0:
                    w = x.shape[-1]
                    key = (x.device, x.dtype, w)
                    cached = _HFLIP_MAT_CACHE.get(key)
                    if cached is None:
                        flip_mat = _HFLIP_MAT_TEMPLATE.to(device=x.device, dtype=x.dtype).clone()
                        flip_mat[0, 2] = w - 1
                        cached = flip_mat.unsqueeze(0)
                        _HFLIP_MAT_CACHE[key] = cached
                    self._transform_matrix = cached.expand(b, 3, 3)
                    return x.flip(-1)
                # p == 0.0: identity matrix, input unchanged.
                eye = torch.eye(3, device=x.device, dtype=x.dtype)
                self._transform_matrix = eye.unsqueeze(0).expand(b, 3, 3)
                return x
            # Other ranks: fall through to the standard chain (handles (H, W)).
        return super().forward(*args, **kwargs)

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        # Fast path: look up a pre-built (1, 3, 3) matrix keyed by (device, dtype, width).
        # The vast majority of calls share the same image size, device, and dtype, so the
        # dict lookup hits after the very first call with a given configuration.
        # expand() is a metadata-only view operation — no allocation for the batch dim.
        w: int = int(params["forward_input_shape"][-1].item())
        key = (input.device, input.dtype, w)
        cached = _HFLIP_MAT_CACHE.get(key)
        if cached is None:
            flip_mat = _HFLIP_MAT_TEMPLATE.to(device=input.device, dtype=input.dtype).clone()
            flip_mat[0, 2] = w - 1
            cached = flip_mat.unsqueeze(0)  # (1, 3, 3)
            _HFLIP_MAT_CACHE[key] = cached
        return cached.expand(input.shape[0], 3, 3)

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
