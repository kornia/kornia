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

import math
import operator

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK_SAME_DEVICE
from kornia.core.ops import _is_numpy_bool
from kornia.core.utils import is_compiling

# Squared error of a probability whose 0.5 threshold gets the pixel wrong: |p - q| > 1/2.
_MISCLASSIFIED_SQUARED_ERROR = 0.25


class _HausdorffERLossBase(nn.Module):
    """Base class for the Hausdorff losses based on morphological erosion, one binary map per class channel.

    The loss follows ``Loss_ER`` of :cite:`karimi2019reducing` (Eq. 12). The error of a pixel is its squared error
    above the value where thresholding the probability at 0.5 starts to misclassify it, and it is weighted by
    ``sum_i i ** alpha`` over the ``k`` successive erosions of the misclassified set that keep the pixel. The erosion
    takes the minimum over the nonzero entries of :meth:`get_kernel`.

    Args:
        alpha: the exponent of the weight ``(i + 1) ** alpha`` of erosion ``i = 0, ..., k - 1``.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the mean of the output is taken,
            'sum': the output will be summed.

    Returns:
        Estimated Hausdorff Loss.

    """

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
        kernel = self.get_kernel()
        # Kept as a buffer so that a state_dict saved with it still loads; the erosion reads only the offsets below.
        self.register_buffer("kernel", kernel)
        # The structuring element as offsets from the kernel centre, read once here so that the erosion is a fixed
        # sequence of slices: reading the buffer in ``forward`` would be a data-dependent ``nonzero``.
        weights = kernel.reshape(kernel.shape[2:])
        self._radius = tuple((size - 1) // 2 for size in weights.shape)
        self._offsets = tuple(
            tuple(int(i) - r for i, r in zip(index, self._radius)) for index in torch.nonzero(weights).tolist()
        )

    def get_kernel(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Get the structuring element of the erosion as a ``(1, 1, *window)`` kernel with odd sizes.

        Only the nonzero entries matter: the erosion takes the minimum over them, centred on each pixel, and ignores
        their values. ``__init__`` calls ``get_kernel()`` once, so an override may take no arguments.

        Args:
            dtype: dtype of the kernel; ``None`` gives the default dtype.
        """
        raise NotImplementedError

    def _erode(self, x: torch.Tensor) -> torch.Tensor:
        """Erode ``x`` once: the minimum over the structuring element, centred on each pixel.

        Positions outside the image count as ``+inf``, so the image border is not an edge of the eroded set.

        Args:
            x: floating-point map with a shape of ``(B, 1, H, W)`` or ``(B, 1, D, H, W)``.

        Returns:
            The eroded map, with the shape of ``x``.
        """
        spatial = x.shape[2:]
        pad = [p for r in reversed(self._radius) for p in (r, r)]
        padded = F.pad(x, pad, value=math.inf)
        # One slice per element of the structuring element.
        windows = [
            padded[(..., *[slice(r + o, r + o + n) for r, o, n in zip(self._radius, offset, spatial)])]
            for offset in self._offsets
        ]
        return torch.stack(windows).amin(0)

    def perform_erosion(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the per-pixel loss of one class channel.

        Args:
            pred: probabilities of the class with a shape of ``(B, 1, H, W)`` or ``(B, 1, D, H, W)``.
            target: the binary map of the class, of the same shape.

        Returns:
            ``error * sum_i (i + 1) ** alpha * erode^(i + 1)(error > 0)`` for ``i = 0, ..., k - 1``, of the shape of
            ``pred``.
        """
        # The soft threshold sits where thresholding ``pred`` at 0.5 starts to get the pixel wrong, so every
        # misclassified pixel counts and no correct one does; a fully wrong pixel gives 1.
        error = ((pred - target) ** 2 - _MISCLASSIFIED_SQUARED_ERROR) / (1 - _MISCLASSIFIED_SQUARED_ERROR)
        error = error.clamp_min(0)

        # The depth weight is a constant, like the distance maps of the paper's HD-DT loss (Eq. 8): the gradient
        # reaches each misclassified pixel through ``error``, in proportion to its depth weight. Eroding ``error``
        # itself gives the same value on a binary prediction, but a minimum hands each window's gradient to one pixel.
        survivors = (error > 0).to(error.dtype)
        weight = torch.zeros_like(error)
        for i in range(self.k):
            survivors = self._erode(survivors)
            weight = weight + survivors * (i + 1) ** self.alpha

        return error * weight

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
    scores, for each class channel ``c``, a surrogate of the Hausdorff distance between the predicted and the labelled
    segmentation, after ``Loss_ER`` of :cite:`karimi2019reducing` (Eq. 12): the per-pixel error :math:`e`, weighted
    by the ``k`` successive erosions :math:`\varepsilon` of the misclassified set that keep the pixel:

    .. math::

        \mathcal{L} = e \sum_{i=1}^{k} i^\alpha \, \varepsilon^i([e > 0]), \qquad
        e = \max\left(\frac{(p_c - [y = c])^2 - 1/4}{3/4},\ 0\right)

    Convention:
        - ``pred`` holds per-class probabilities ``(B, C, H, W)`` and nothing is applied to it, unlike the logits of
          :func:`~kornia.losses.focal_loss`: pass a softmax. ``target`` holds int64 labels ``(B, 1, H, W)`` in
          ``[0, C)``. Channel ``c`` is compared with ``target == c``, class 0 included.
        - The ``[0, C)`` check reads the data, so it is skipped wherever the call captures a graph
          (``torch.compile`` or ``torch.export``). A captured call whose target holds an out-of-range label scores
          those voxels in no class and returns a finite loss instead of raising; an eager call still raises.
        - The default ``'mean'`` is the plain mean of the ``'none'`` tensor.
        - :math:`e` is nonzero exactly where thresholding :math:`p_c` at 0.5 gets the pixel wrong,
          :math:`|p_c - [y = c]| > 1/2`, and is 1 for a fully wrong pixel. A correct pixel adds nothing and gets no
          gradient, however uncertain; a probability of exactly 0.5 counts as correct. With two classes this is the
          argmax decision. With three or more it is a one-vs-rest margin, stricter than argmax: a pixel whose true
          class wins the argmax with :math:`p_c < 1/2` still counts in channel :math:`c`.
        - :math:`\varepsilon` takes the minimum over the pixel and its four neighbours. Pixels outside the image count
          as :math:`+\infty`, so the image border is not an edge of an error.
        - The weight of a misclassified pixel is :math:`\sum_{i=1}^{\min(k, d - 1)} i^\alpha`, for its city-block
          distance :math:`d` to the nearest correctly classified pixel. It is a constant: the gradient reaches each
          misclassified pixel through :math:`e`, in proportion to that weight. The pixels on the edge of an error
          (:math:`d = 1`) get none, and errors at most two pixels wide add nothing.
        - The weight measures how thick an error is, not how far it lies from the target, so a thin or small error
          far from the object, which can set the Hausdorff distance, adds little or nothing
          (`#6680 <https://github.com/kornia/kornia/issues/6680>`_).
        - Use it as an auxiliary term next to an overlap loss, never alone, as the paper does:
          ``region_loss + lam * hausdorff_loss``, with ``lam`` set so that both terms weigh the same (the paper
          recomputes it from their means after each epoch) or ramped up from a small value. Choose ``k`` from the
          expected error radius in pixels: an error deeper than ``k`` gets the same weight as one ``k`` deep.
        - The value is a unitless surrogate, at most :math:`\sum_{i=1}^{k} i^\alpha` per pixel (385 at the
          defaults), not a distance in pixels.
        - The paper erodes :math:`(p_c - [y = c])^2` itself, by a convolution followed by a soft threshold at 0.5,
          which leaves nothing of a solid error after two erosions and ignores misclassified pixels whose
          neighbourhood-averaged squared error is below 1/2. On a binary prediction this loss equals the exact
          erosion of :math:`e` (`#6676 <https://github.com/kornia/kornia/issues/6676>`_).

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

        Next to :class:`~kornia.losses.DiceLoss`, which takes the logits and ``(B, H, W)`` labels:

        >>> logits = torch.randn(5, 3, 20, 20)
        >>> loss = kornia.losses.DiceLoss()(logits, target[:, 0]) + 0.1 * hdloss(logits.softmax(1), target)

    """

    def get_kernel(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Get the structuring element of the erosion: the pixel and its four neighbours.

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

    Convention:
        - See the Convention block of :class:`~kornia.losses.HausdorffERLoss`. The erosion takes the minimum over the
          voxel and its six face neighbours, so it also runs along ``D``, and the distance :math:`d` is the city-block
          distance in the volume.

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

    def get_kernel(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Get the structuring element of the erosion: the voxel and its six face neighbours.

        Args:
            dtype: dtype of the kernel; ``None`` gives the default dtype.
        """
        cross = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=dtype)
        bound = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=dtype)
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
