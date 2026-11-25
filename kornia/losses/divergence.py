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

import math

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
    # CPU autocast runs kl_div in float32 but leaves log in half precision, so the logarithm of a half q would be
    # rounded before kl_div promotes it. CUDA and MPS autocast already run log in float32.
    if torch.jit.is_scripting():
        # Keep the original scripted graph; its autocast state queries have no device argument.
        log_q = q.log()
    else:
        if q.dtype in (torch.float16, torch.bfloat16) and q.device.type == "cpu" and torch.is_autocast_enabled("cpu"):
            q = q.float()
        log_q = q.log()
    unsummed_kl = F.kl_div(log_q, p, reduction="none")

    return unsummed_kl.sum(-1).view(batch, chans)


def _js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    output_dtype = torch.promote_types(p.dtype, q.dtype)
    # Keep half inputs in float32 for the logarithms and spatial reductions.
    if p.dtype in (torch.float16, torch.bfloat16):
        p = p.float()
    if q.dtype in (torch.float16, torch.bfloat16):
        q = q.float()
    # Form log(p + q) instead of halving a probability: even native float32/float64 Gaussian tails can
    # round (p + q) / 2 to zero (#6723). Substitute shared zeros before log to keep their backward finite.
    log_sum = (p + q).masked_fill((p == 0) & (q == 0), 1.0).log()
    terms = []
    for a, b in ((p, q), (q, p)):
        # Match KL's empty-cell convention without hiding invalid probabilities in either input.
        empty = (a == 0) & (b >= 0) & torch.isfinite(b)
        log_ratio = (a.masked_fill(empty, 1.0).log() - log_sum + math.log(2.0)).masked_fill(empty, 0.0)
        terms.append((a * log_ratio).sum((-2, -1)))
    return (0.5 * (terms[0] + terms[1])).to(output_dtype)


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
    r"""Compute the Jensen-Shannon divergence between heatmaps.

    ``loss = js_div_loss_2d(pred, target)`` is the symmetric divergence of each of the ``B N`` pairs of ``(H, W)``
    probability heatmaps.

    .. math::

        \text{JSD}(P, Q) = \frac{1}{2} \mathrm{KL}(P \,\|\, M) + \frac{1}{2} \mathrm{KL}(Q \,\|\, M), \quad
        M = \frac{P + Q}{2}

    Convention:
        - The divergence is in nats and at most :math:`\ln 2`. The inputs and the reductions are those of
          :func:`~kornia.losses.kl_div_loss_2d`; see its Convention block.
        - A cell that is zero in both inputs contributes zero, including to the gradients. Both inputs must have
          the same shape; unsupported reductions raise ``NotImplementedError``.
        - The mixture is evaluated in log space so halving a tiny probability cannot round it to zero.
          Float16 and bfloat16 inputs are evaluated in float32. The returned loss retains the promoted input dtype.

    Args:
        pred: predicted heatmaps with shape ``(B, N, H, W)``.
        target: target heatmaps with the same shape.
        reduction: ``'mean'`` (default) averages the ``B N`` divergences, ``'sum'`` adds them, ``'none'`` returns them
          with shape ``(B, N)``.

    Returns:
        the loss, a scalar or, for ``reduction='none'``, with shape ``(B, N)``.

    See Also:
        - :func:`~kornia.losses.kl_div_loss_2d`: not symmetric, and ``inf`` where ``pred`` is zero and ``target`` is
          positive.

    Examples:
        >>> pred = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(pred, pred)
        >>> loss.item()
        0.0

    """
    _check_heatmaps(pred, target)
    return _reduce_loss(_js_div_2d(target, pred), reduction)


def kl_div_loss_2d(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Compute the Kullback-Leibler divergence KL(target || pred) between heatmaps.

    ``loss = kl_div_loss_2d(pred, target)`` is the divergence of each of the ``B N`` pairs of ``(H, W)`` probability
    heatmaps, in nats.

    .. math::

        \mathrm{KL}(\text{target} \,\|\, \text{pred}) = \sum_{H \times W} \text{target}
        \left(\log \text{target} - \log \text{pred}\right)

    Convention:
        - The order is that of torch's ``F.kl_div(pred.log(), target)``.
          :ref:`Losses and metrics <losses-metrics-conventions>` maps it onto torch, scipy and torchmetrics.
        - The inputs are probabilities, each :math:`(b, n)` slice a distribution over :math:`H \times W`. The log is
          taken inside, so log-probabilities give NaN, and nothing is normalised. A zero in ``pred`` where ``target``
          is positive gives ``inf``.
        - The default ``'mean'`` averages over the :math:`B N` distributions, not over every element as torch's does.
        - Both inputs must have the same shape. Unsupported reductions, including torch's ``'batchmean'``, raise
          ``NotImplementedError``. A zero in ``target`` with a finite nonnegative ``pred`` contributes zero to the
          value and gradients, including when both entries are zero.
        - Under CPU autocast, the logarithm of a float16/bfloat16 ``pred`` is taken in float32, as CUDA and MPS
          autocast already do; ``F.kl_div`` alone would promote to float32 only after that logarithm had been rounded.

    Args:
        pred: predicted heatmaps with shape ``(B, N, H, W)``.
        target: target heatmaps with the same shape.
        reduction: ``'mean'`` (default) averages the ``B N`` divergences, ``'sum'`` adds them, ``'none'`` returns them
          with shape ``(B, N)``.

    Returns:
        the loss, a scalar or, for ``reduction='none'``, with shape ``(B, N)``.

    See Also:
        - :func:`~kornia.losses.js_div_loss_2d`: symmetric and at most :math:`\ln 2`, finite where ``pred`` is zero.

    Examples:
        >>> pred = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = kl_div_loss_2d(pred, pred)
        >>> loss.item()
        0.0

    """
    _check_heatmaps(pred, target)
    return _reduce_loss(_kl_div_2d(target, pred), reduction)
