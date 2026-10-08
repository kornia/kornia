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

r"""Losses based on the divergence between probability distributions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


def _kl_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # D_KL(P || Q) with the convention 0 * log 0 = 0. Cells with p == 0 and a finite q >= 0 contribute nothing to the
    # value or the gradient: both arguments are replaced by 1 there, so a q == 0 in such a cell cannot turn the term
    # into NaN. Every other cell keeps the value of F.kl_div, so a NaN or a negative entry in either input, or an
    # infinite q where p == 0, still gives NaN.
    batch, chans, height, width = p.shape
    p = p.reshape(batch * chans, height * width)
    q = q.reshape(batch * chans, height * width)
    empty = (p == 0) & (q >= 0) & torch.isfinite(q)
    p = p.masked_fill(empty, 1.0)
    q = q.masked_fill(empty, 1.0)
    unsummed_kl = F.kl_div(q.log(), p, reduction="none")

    return unsummed_kl.sum(-1).view(batch, chans)


def _js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)


# TODO: add this to the main module
def _reduce_loss(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return losses
    if reduction == "mean":
        return torch.mean(losses)
    if reduction == "sum":
        return torch.sum(losses)
    raise NotImplementedError(f"Invalid reduction mode: {reduction}")


def _check_heatmaps(pred: torch.Tensor, target: torch.Tensor) -> None:
    KORNIA_CHECK_SHAPE(pred, ["B", "N", "H", "W"])
    KORNIA_CHECK(
        pred.shape == target.shape, f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}"
    )


def js_div_loss_2d(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Calculate the Jensen-Shannon divergence loss between heatmaps.

    Convention:
        - The divergence is :math:`\frac{1}{2} \mathrm{KL}(P \,\|\, M) + \frac{1}{2} \mathrm{KL}(Q \,\|\, M)`
          with :math:`M = (P + Q) / 2`, in nats: symmetric in ``pred`` and ``target``, and at most :math:`\ln 2`. The
          inputs and the reductions are those of :func:`~kornia.losses.kl_div_loss_2d`; see its Convention block.
        - A cell that is zero in both inputs contributes zero, including to the gradients. Both inputs must have
          the same shape; unsupported reductions raise ``NotImplementedError``.

    Args:
        pred: the input torch.Tensor with shape :math:`(B, N, H, W)`.
        target: the target torch.Tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> pred = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(pred, pred)
        >>> loss.item()
        0.0

    """
    _check_heatmaps(pred, target)
    return _reduce_loss(_js_div_2d(target, pred), reduction)


def kl_div_loss_2d(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Calculate the Kullback-Leibler divergence loss between heatmaps.

    Convention:
        - ``kl_div_loss_2d(pred, target)`` is :math:`\mathrm{KL}(\text{target} \,\|\, \text{pred})`, the sum of
          ``target * (log(target) - log(pred))`` over :math:`H \times W` for every :math:`(b, n)`: the order of torch's
          ``F.kl_div(pred.log(), target)``. :ref:`Losses and metrics <losses-metrics-conventions>` maps it onto torch,
          scipy and torchmetrics.
        - The inputs are probabilities, each :math:`(b, n)` slice a distribution over :math:`H \times W`. The log is
          taken inside, so log-probabilities give NaN, and nothing is normalised. A zero in ``pred`` where ``target``
          is positive gives ``inf``.
        - ``reduction='none'`` returns :math:`(B, N)`; the default ``'mean'`` averages over the :math:`B N`
          distributions and ``'sum'`` adds them.
        - Both inputs must have the same shape. Unsupported reductions, including torch's ``'batchmean'``, raise
          ``NotImplementedError``. A zero in ``target`` with a finite nonnegative ``pred`` contributes zero to the
          value and gradients, including when both entries are zero.

    Args:
        pred: the input torch.Tensor with shape :math:`(B, N, H, W)`.
        target: the target torch.Tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> pred = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = kl_div_loss_2d(pred, pred)
        >>> loss.item()
        0.0

    """
    _check_heatmaps(pred, target)
    return _reduce_loss(_kl_div_2d(target, pred), reduction)
