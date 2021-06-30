r"""Losses based on the divergence between probability distributions."""

import torch
import torch.nn.functional as F


def _kl_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # D_KL(P || Q)
    batch, chans, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch * chans, height * width).log(), p.reshape(batch * chans, height * width), reduction='none'
    )
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values


def _js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)


# TODO: add this to the main module


def _reduce_loss(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'none':
        return losses
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


def js_div_loss_2d(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
    r"""Calculates the Jensen-Shannon divergence loss between heatmaps.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(_js_div_2d(target, input), reduction)


def kl_div_loss_2d(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
    r"""Calculates the Kullback-Leibler divergence loss between heatmaps.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(_kl_div_2d(target, input), reduction)
