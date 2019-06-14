r"""Losses based on the divergence between probability distributions."""


import torch
import torch.nn.functional as F


def _kl_div_2d(p, q):
    # D_KL(P || Q)
    batch, chans, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.view(batch * chans, height * width).log(),
        p.view(batch * chans, height * width),
        reduction='none',
    )
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values


def _js_div_2d(p, q):
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)


def _reduce_loss(losses, reduction):
    if reduction == 'none':
        return losses
    else:
        return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


def js_div_loss_2d(
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
):
    r"""Calculates the Jensen-Shannon divergence loss between heatmaps.

    Arguments:
        input (torch.Tensor): the input tensor.
        target (torch.Tensor): the target tensor.
        reduction (string, optional): Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed. Default: ``'mean'``.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Target: :math:`(B, N, H, W)`, same shape as the input
    """
    return _reduce_loss(_js_div_2d(target, input), reduction)


def kl_div_loss_2d(
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
):
    r"""Calculates the Kullback-Leibler divergence loss between heatmaps.

    Arguments:
        input (torch.Tensor): the input tensor.
        target (torch.Tensor): the target tensor.
        reduction (string, optional): Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed. Default: ``'mean'``.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Target: :math:`(B, N, H, W)`, same shape as the input
    """
    return _reduce_loss(_kl_div_2d(target, input), reduction)
