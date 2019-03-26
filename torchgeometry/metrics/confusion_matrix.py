from typing import Optional

import torch

# Inspired by:
# https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py#L68-L73  # noqa

def confusion_matrix(
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int) -> torch.Tensor:
    r"""Compute confusion matrix to evaluate the accuracy of a classification.

    Computes the confusion matrix of K x K size where K is no of classes.

    Args:
        input (torch.Tensor) : tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target (torch.Tensor) : tensor with ground truth (correct) target
          values. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1, whete targets are assumed to be provided as
          one-hot vectors.
        num_classes (int): total possible number of classes in target.

    Returns:
        torch.Tensor: a tensor with the confusion matrix with shape
        :math:`(B, K, K)` where K is the number of classes.
    """
    if not isinstance(input, torch.LongTensor):
        raise TypeError("Input input type is not a torch.LongTensor. Got {}"
                        .format(type(input)))
    if not isinstance(target, torch.LongTensor):
        raise TypeError("Input target type is not a torch.LongTensor. Got {}"
                        .format(type(target)))
    if not input.shape == target.shape:
        raise ValueError("Inputs input and target must have the same shape. "
                         "Got: {}".format(input.shape, target.shape))
    if not input.device == target.device:
        raise ValueError("Inputs must be in the same device. "
                         "Got: {} - {}".format(input.device, target.device))
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("The number of classes must be an intenger bigger "
                         "than two. Got: {}".format(num_classes))
    batch_size: int = input.shape[0]

    # hack for bitcounting 2 arrays together
    # NOTE: torch.bincount does not implement batched version
    pre_bincount: torch.Tensor = input + target * num_classes
    pre_bincount_vec: torch.Tensor = pre_bincount.view(batch_size, -1)
    confusion_vec: torch.Tensor = torch.stack([
        torch.bincount(pb, minlength=num_classes**2) for pb in pre_bincount_vec
    ])
    return confusion_vec.view(batch_size, num_classes, num_classes)  # BxKxK
