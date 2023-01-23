from typing import List

import torch


def accuracy(input: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        input: the input tensor with the logits to evaluate.
        target: the tensor containing the ground truth.
        topk: the expected topk ranking.

    Example:
        >>> logits = torch.tensor([[0, 1, 0]])
        >>> target = torch.tensor([[1]])
        >>> accuracy(logits, target)
        [tensor(100.)]
    """
    maxk = min(max(topk), input.size()[1])
    batch_size = target.size(0)
    _, pred = input.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
