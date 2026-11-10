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

import inspect
import math
import operator
from typing import Any, Callable

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SAME_DEVICE
from kornia.core.ops import _is_numpy_bool
from kornia.core.utils import is_compiling


class _HausdorffERLossBase(nn.Module):
    """Base class for the Hausdorff losses based on morphological erosion, one binary map per class channel.

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion,which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
    blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: the exponent of the weight ``(i + 1) ** alpha`` of erosion ``i = 0, ..., k - 1``.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the mean of the output is taken,
            'sum': the output will be summed.

    Returns:
        Estimated Hausdorff Loss.

    """

    conv: Callable[..., torch.Tensor]
    max_pool: Callable[..., torch.Tensor]
    # Whether ``get_kernel`` takes ``dtype``; an override written before it did keeps eroding with its buffer.
    _get_kernel_takes_dtype: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._get_kernel_takes_dtype = "dtype" in inspect.signature(cls.get_kernel).parameters

    def __init__(self, alpha: float = 2.0, k: int = 10, reduction: str = "mean") -> None:
        super().__init__()
        # NaN and +inf weights make the loss NaN. -inf weights only the first erosion (the limit of the weights), and
        # an alpha tensor of shape (B, 1, 1, 1) gives each sample its own exponent, so neither is rejected.
        if not torch.as_tensor(alpha < math.inf).all():
            raise ValueError(f"alpha must not be NaN or +inf. Got {alpha}.")
        # operator.index reads a bool, a boolean tensor and, on NumPy 1.x, a NumPy bool as 0 or 1.
        if isinstance(k, bool) or (isinstance(k, torch.Tensor) and k.dtype == torch.bool) or _is_numpy_bool(k):
            raise TypeError("k must be an integer. Got bool.")
        try:
            # NumPy integers and integer 0-d tensors are integers too.
            k = operator.index(k)
        except TypeError:
            raise TypeError(f"k must be an integer. Got {type(k).__name__}.") from None
        if k < 0:
            raise ValueError(f"k must be non-negative. Got {k}.")
        if reduction not in ("mean", "sum", "none"):
            raise NotImplementedError(f"reduction `{reduction}` has not been implemented yet.")
        self.alpha = alpha
        self.k = k
        self.reduction = reduction
        self.register_buffer("kernel", self.get_kernel())

    def get_kernel(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Get kernel for image morphology convolution.

        Args:
            dtype: dtype of the kernel; ``None`` gives the default dtype. ``forward`` calls
                ``get_kernel(torch.float64)`` for a float64 prediction; an override without ``dtype`` keeps
                eroding with the ``kernel`` buffer.
        """
        raise NotImplementedError

    def perform_erosion(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bound = (pred - target) ** 2

        # The buffer holds the kernel built in the default dtype, normally float32, and ``.double()`` keeps that
        # rounding, so a float64 prediction gets a kernel built in float64 instead.
        if pred.dtype == torch.float64 and self._get_kernel_takes_dtype:
            kernel = self.get_kernel(torch.float64)
        else:
            kernel = self.kernel
        kernel = torch.as_tensor(kernel, device=pred.device, dtype=pred.dtype)
        eroded = torch.zeros_like(bound, device=pred.device, dtype=pred.dtype)

        # Same padding, assuming kernel is odd and square (cube) shaped.
        padding = (kernel.size(-1) - 1) // 2
        for k in range(self.k):
            # compute convolution with kernel
            dilation = self.conv(bound, weight=kernel, padding=padding, groups=1)
            # apply soft thresholding at 0.5 and F.normalize. masked_fill zeroes the negatives without
            # a data-dependent index_put_, and keeps its gradient of 1 at dilation == 0.5 exactly;
            # clamp_min's derivative at the bound is 0 on torch >= 2.14 and 1 before.
            erosion = (dilation - 0.5).masked_fill(dilation < 0.5, 0)

            # image-wise differences for 2D images
            erosion_max = self.max_pool(erosion)
            erosion_min = -self.max_pool(-erosion)
            # No normalization needed if `max - min = 0`: those lanes keep ``erosion`` through the
            # ``where`` below and divide by a safe 1 so they stay finite (a NaN in the unselected
            # branch would still poison the gradient). Branch-free, so the loss also traces for export.
            # NOTE: avoid in-place ops like below, which will not pass gradcheck:
            #       erosion[to_norm] = (erosion[to_norm] - erosion_min[to_norm]) / (
            #           erosion_max[to_norm] - erosion_min[to_norm])
            _range = erosion_max - erosion_min
            _to_norm = _range != 0
            _erosion_to_fill = (erosion - erosion_min) / torch.where(_to_norm, _range, 1.0)
            erosion = torch.where(_to_norm, _erosion_to_fill, erosion)

            # save erosion and add to loss
            eroded = eroded + erosion * (k + 1) ** self.alpha
            bound = erosion

        return eroded

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Hausdorff loss.

        Args:
            pred: predicted torch.Tensor with a shape of ``(B, C, H, W)`` or ``(B, C, D, H, W)``.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target torch.Tensor with a shape of ``(B, 1, H, W)`` or ``(B, 1, D, H, W)``.

        Returns:
            Estimated Hausdorff Loss.

        """
        KORNIA_CHECK_SAME_DEVICE(pred, target)

        if not (pred.shape[2:] == target.shape[2:] and pred.size(0) == target.size(0) and target.size(1) == 1):
            raise ValueError(
                "Prediction and target need to be of same size, and target should not be one-hot."
                f"Got {pred.shape} and {target.shape}."
            )

        if target.dtype != torch.long:
            raise ValueError(f"Expect long type target value in range [0, {pred.size(1)}). Got {target.dtype}.")
        # The range check reads the data, which graph capture cannot do; skip it under any capture.
        # Neither Hausdorff loss scripts on main (its ``super().forward``), so no TorchScript guard is needed.
        if not is_compiling() and target.numel() > 0 and not (target.max() < pred.size(1) and target.min() >= 0):
            raise ValueError(
                f"Expect long type target value in range [0, {pred.size(1)}). ({target.min()}, {target.max()})"
            )

        out = torch.stack(
            [
                self.perform_erosion(
                    pred[:, i : i + 1],
                    torch.where(
                        target == i,
                        torch.tensor(1, device=target.device, dtype=target.dtype),
                        torch.tensor(0, device=target.device, dtype=target.dtype),
                    ),
                )
                for i in range(pred.size(1))
            ]
        )

        if self.reduction == "mean":
            out = out.mean()
        elif self.reduction == "sum":
            out = out.sum()
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError(f"reduction `{self.reduction}` has not been implemented yet.")

        return out


class HausdorffERLoss(_HausdorffERLossBase):
    r"""Compute the erosion-based Hausdorff loss of ``(B, C, H, W)`` probabilities against ``(B, 1, H, W)`` labels.

    ``criterion = HausdorffERLoss(alpha, k, reduction)`` stores its arguments, and ``loss = criterion(pred, target)``
    scores, for each class channel, a surrogate of the Hausdorff distance between the predicted and the labelled
    segmentation from ``k`` erosions of their squared difference :cite:`karimi2019reducing`.

    .. math::

        HD(X, Y) = \max(hd(X, Y), hd(Y, X)), \qquad hd(X, Y) = \max_{x \in X} \min_{y \in Y} \|x - y\|_2

    Reference:
        Based on https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py

    Convention:
        - ``pred`` holds per-class probabilities ``(B, C, H, W)`` and nothing is applied to it, unlike the logits of
          :func:`~kornia.losses.focal_loss`: pass a softmax. ``target`` holds int64 labels ``(B, 1, H, W)`` in
          ``[0, C)``. Channel ``c`` is compared with ``target == c``, class 0 included.
        - The ``[0, C)`` check reads the data, so it is skipped wherever the call captures a graph
          (``torch.compile`` or ``torch.export``). A captured call whose target holds an out-of-range label scores
          those voxels in no class and returns a finite loss instead of raising; an eager call still raises.
        - The default ``'mean'`` is the plain mean of the ``'none'`` tensor.
        - Each erosion keeps only the part of the neighbourhood-averaged squared error above 0.5. Where every pixel's
          true-class probability is above :math:`1 - 1/\sqrt{2} \approx 0.29`, the loss and gradient are zero: a
          uniform softmax over two or three classes gives zero, and so does a two-class prediction that is 70 %
          confident and wrong (`#6676 <https://github.com/kornia/kornia/issues/6676>`_).
        - Each of the ``k`` erosions is min-max normalised per image and class and weighted by
          :math:`(i + 1)^\alpha` for :math:`i = 0, \dots, k - 1`: the value is a unitless surrogate, not a distance
          in pixels.

    Args:
        alpha: exponent of the weight :math:`(i + 1)^\alpha` of erosion :math:`i = 0, \dots, k - 1`.
        k: number of erosions.
        reduction: ``'mean'`` (default), ``'sum'`` or ``'none'``.

    Shape:
        - pred: probabilities ``(B, C, H, W)``.
        - target: int64 class labels ``(B, 1, H, W)``.
        - Output: a scalar for ``'mean'`` and ``'sum'``, ``(C, B, 1, H, W)`` for ``'none'``.

    See Also:
        - :class:`~kornia.losses.HausdorffERLoss3D`: the same loss for volumes.
        - :class:`~kornia.losses.DiceLoss`: an overlap loss that takes logits, not probabilities.

    Examples:
        >>> hdloss = HausdorffERLoss()
        >>> input = torch.randn(5, 3, 20, 20).softmax(1)
        >>> target = (torch.rand(5, 1, 20, 20) * 2).long()
        >>> res = hdloss(input, target)

    """

    conv = torch.conv2d

    @staticmethod
    def max_pool(x: torch.Tensor) -> torch.Tensor:
        """Take the global maximum over the spatial dimensions, keeping them with size one."""
        # Equals ``nn.AdaptiveMaxPool2d(1)`` but has an ONNX lowering.
        return x.amax(dim=(-2, -1), keepdim=True)

    def get_kernel(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Get kernel for image morphology convolution.

        Args:
            dtype: dtype of the kernel; ``None`` gives the default dtype.
        """
        cross = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=dtype)
        kernel = cross * torch.tensor(0.2, dtype=dtype)
        return kernel[None]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Hausdorff loss.

        Args:
            pred: predicted torch.Tensor with a shape of ``(B, C, H, W)``.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target torch.Tensor with a shape of ``(B, 1, H, W)``.

        Returns:
            Estimated Hausdorff Loss.

        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"pred type is not a torch.Tensor. Got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"target type is not a torch.Tensor. Got {type(target)}.")
        if pred.dim() != 4:
            raise ValueError(f"Only 2D images supported. Got {pred.dim()}.")

        return super().forward(pred, target)


class HausdorffERLoss3D(_HausdorffERLossBase):
    r"""Compute the erosion-based Hausdorff loss of ``(B, C, D, H, W)`` probabilities against volume labels.

    ``criterion = HausdorffERLoss3D(alpha, k, reduction)`` stores its arguments, and
    ``loss = criterion(pred, target)`` is :class:`~kornia.losses.HausdorffERLoss` on volumes :cite:`karimi2019reducing`.

    Reference:
        Based on https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py

    Convention:
        - See the Convention block of :class:`~kornia.losses.HausdorffERLoss`. The erosion kernel is the voxel and its
          six face neighbours, so the erosion also runs along ``D``.

    Args:
        alpha: exponent of the weight :math:`(i + 1)^\alpha` of erosion :math:`i = 0, \dots, k - 1`.
        k: number of erosions.
        reduction: ``'mean'`` (default), ``'sum'`` or ``'none'``.

    Shape:
        - pred: probabilities ``(B, C, D, H, W)``.
        - target: int64 class labels ``(B, 1, D, H, W)``.
        - Output: a scalar for ``'mean'`` and ``'sum'``, ``(C, B, 1, D, H, W)`` for ``'none'``.

    See Also:
        - :class:`~kornia.losses.HausdorffERLoss`: the same loss for images.

    Examples:
        >>> hdloss = HausdorffERLoss3D()
        >>> input = torch.randn(5, 3, 20, 20, 20).softmax(1)
        >>> target = (torch.rand(5, 1, 20, 20, 20) * 2).long()
        >>> res = hdloss(input, target)

    """

    conv = torch.conv3d

    @staticmethod
    def max_pool(x: torch.Tensor) -> torch.Tensor:
        """Take the global maximum over the volume dimensions, keeping them with size one."""
        # Equals ``nn.AdaptiveMaxPool3d(1)`` but has an ONNX lowering.
        return x.amax(dim=(-3, -2, -1), keepdim=True)

    def get_kernel(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Get kernel for image morphology convolution.

        Args:
            dtype: dtype of the kernel; ``None`` gives the default dtype.
        """
        cross = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=dtype)
        bound = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=dtype)
        # NOTE: The original repo claimed it shaped as (3, 1, 3, 3)
        #    which Jian suspect it is wrongly implemented.
        # https://github.com/PatRyg99/HausdorffLoss/blob/9f580acd421af648e74b45d46555ccb7a876c27c/hausdorff_loss.py#L94
        kernel = torch.stack([bound, cross, bound], 1) * torch.tensor(1 / 7, dtype=dtype)
        return kernel[None]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D Hausdorff loss.

        Args:
            pred: predicted torch.Tensor with a shape of ``(B, C, D, H, W)``.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target torch.Tensor with a shape of ``(B, 1, D, H, W)``.

        Returns:
            Estimated Hausdorff Loss.

        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"pred type is not a torch.Tensor. Got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"target type is not a torch.Tensor. Got {type(target)}.")
        if pred.dim() != 5:
            raise ValueError(f"Only 3D images supported. Got {pred.dim()}.")

        return super().forward(pred, target)
