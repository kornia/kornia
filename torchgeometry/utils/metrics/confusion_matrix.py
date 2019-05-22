from typing import Optional

import torch

# Inspired by:
# https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py#L68-L73  # noqa


def confusion_matrix(
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        normalized: Optional[bool] = False) -> torch.Tensor:
    r"""Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        input (torch.Tensor) : tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target (torch.Tensor) : tensor with ground truth (correct) target
          values. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1, whete targets are assumed to be provided as
          one-hot vectors.
        num_classes (int): total possible number of classes in target.
        normalized: (Optional[bool]): wether to return the confusion matrix
          normalized. Default: False.

    Returns:
        torch.Tensor: a tensor containing the confusion matrix with shape
        :math:`(B, K, K)` where K is the number of classes.
    """
    if not torch.is_tensor(input) and input.dtype is not torch.int64:
        raise TypeError("Input input type is not a torch.Tensor with "
                        "torch.int64 dtype. Got {}".format(type(input)))
    if not torch.is_tensor(target) and target.dtype is not torch.int64:
        raise TypeError("Input target type is not a torch.Tensor with "
                        "torch.int64 dtype. Got {}".format(type(target)))
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

    confusion_list = []
    for iter_id in range(batch_size):
        pb: torch.Tensor = pre_bincount_vec[iter_id]
        bin_count: torch.Tensor = torch.bincount(pb, minlength=num_classes**2)
        confusion_list.append(bin_count)

    confusion_vec: torch.Tensor = torch.stack(confusion_list)
    confusion_mat: torch.Tensor = confusion_vec.view(
        batch_size, num_classes, num_classes).to(torch.float32)  # BxKxK

    if normalized:
        norm_val: torch.Tensor = torch.sum(confusion_mat, dim=1, keepdim=True)
        confusion_mat = confusion_mat / (norm_val + 1e-6)

    return confusion_mat
