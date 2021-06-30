import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.utils.one_hot import one_hot

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py


def tversky_loss(
    input: torch.Tensor, target: torch.Tensor, alpha: float, beta: float, eps: float = 1e-8
) -> torch.Tensor:
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = tversky_loss(input, target, alpha=0.5, beta=0.5)
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}".format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}".format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".format(input.device, target.device)
        )

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    fps = torch.sum(input_soft * (-target_one_hot + 1.0), dims)
    fns = torch.sum((-input_soft + 1.0) * target_one_hot, dims)

    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns
    tversky_loss = numerator / (denominator + eps)

    return torch.mean(-tversky_loss + 1.0)


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> criterion = TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, beta: float, eps: float = 1e-8) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tversky_loss(input, target, self.alpha, self.beta, self.eps)
