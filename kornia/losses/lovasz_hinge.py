import torch
import torch.nn as nn
from torch import Tensor

from kornia.testing import KORNIA_CHECK_SHAPE

# based on:
# https://github.com/bermanmaxim/LovaszSoftmax


def lovasz_hinge_loss(input: Tensor, target: Tensor) -> Tensor:
    r"""Criterion that computes a surrogate binary intersection-over-union (IoU) loss.

    According to [2], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the binary tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] http://proceedings.mlr.press/v37/yub15.pdf
        [2] https://arxiv.org/pdf/1705.08790.pdf

    . note::
        This loss function only supports binary labels. For multi-class labels please
        use the Lovasz-Softmax loss.

    Args:
        input: logits tensor with shape :math:`(N, 1, H, W)`.
        labels: labels tensor with shape :math:`(N, H, W)` with binary values.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 1  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = lovasz_hinge_loss(input, target)
        >>> output.backward()
    """
    KORNIA_CHECK_SHAPE(input, ["B", "1", "H", "W"])

    KORNIA_CHECK_SHAPE(target, ["B", "H", "W"])

    if not input.shape[1] == 1:
        raise ValueError(f"Invalid input shape, we expect Bx1xHxW. Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # flatten input and target [B, -1] and to float
    input_flatten: Tensor = input.reshape(input.shape[0], -1)
    target_flatten: Tensor = target.reshape(target.shape[0], -1)

    # get shapes
    B, N = input_flatten.shape

    # compute actual loss
    signs = 2. * target_flatten - 1.
    errors = 1. - input_flatten * signs
    errors_sorted, permutation = errors.sort(dim=1, descending=True)
    batch_index: Tensor = torch.arange(B, device=input.device).reshape(-1, 1).repeat(1, N).reshape(-1)
    target_sorted: Tensor = target_flatten[batch_index, permutation.view(-1)]
    target_sorted: Tensor = target_sorted.view(B, N)
    target_sorted_sum: Tensor = target_sorted.sum(1, keepdim=True)
    intersection: Tensor = target_sorted_sum - target_sorted.cumsum(1)
    union: Tensor = target_sorted_sum + (1. - target_sorted).cumsum(1)
    gradient: Tensor = 1. - intersection / union
    if N > 1:
        gradient[..., 1:] = gradient[..., 1:] - gradient[..., :-1]
    loss: Tensor = (errors_sorted.relu() * gradient).sum(1).mean()
    return loss


class LovaszHingeLoss(nn.Module):
    r"""Criterion that computes a surrogate binary intersection-over-union (IoU) loss.

    According to [2], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the binary tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] http://proceedings.mlr.press/v37/yub15.pdf
        [2] https://arxiv.org/pdf/1705.08790.pdf

    . note::
        This loss function only supports binary labels. For multi-class labels please
        use the Lovasz-Softmax loss.

    Args:
        input: logits tensor with shape :math:`(N, 1, H, W)`.
        labels: labels tensor with shape :math:`(N, H, W)` with binary values.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 1  # num_classes
        >>> criterion = LovaszHingeLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(LovaszHingeLoss).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return lovasz_hinge_loss(input=input, target=target)
