import torch

# Inspired by:
# https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py#L68-L73


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, normalized: bool = False
) -> torch.Tensor:
    r"""Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        pred: tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target: tensor with ground truth (correct) target
          values. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1, where targets are assumed to be provided as
          one-hot vectors.
        num_classes: total possible number of classes in target.
        normalized: whether to return the confusion matrix normalized.

    Returns:
        a tensor containing the confusion matrix with shape
        :math:`(B, K, K)` where K is the number of classes.

    Example:
        >>> logits = torch.tensor([[0, 1, 0]])
        >>> target = torch.tensor([[0, 1, 0]])
        >>> confusion_matrix(logits, target, num_classes=3)
        tensor([[[2., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 0.]]])
    """
    if not torch.is_tensor(pred) and pred.dtype is not torch.int64:
        raise TypeError(f"Input pred type is not a torch.Tensor with torch.int64 dtype. Got {type(pred)}")

    if not torch.is_tensor(target) and target.dtype is not torch.int64:
        raise TypeError(f"Input target type is not a torch.Tensor with torch.int64 dtype. Got {type(target)}")
    if not pred.shape == target.shape:
        raise ValueError(f"Inputs pred and target must have the same shape. Got: {pred.shape} and {target.shape}")
    if not pred.device == target.device:
        raise ValueError(f"Inputs must be in the same device. Got: {pred.device} - {target.device}")

    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"The number of classes must be an integer bigger than two. Got: {num_classes}")

    batch_size: int = pred.shape[0]

    # hack for bitcounting 2 arrays together
    # NOTE: torch.bincount does not implement batched version
    pre_bincount: torch.Tensor = pred + target * num_classes
    pre_bincount_vec: torch.Tensor = pre_bincount.view(batch_size, -1)

    confusion_list = []
    for iter_id in range(batch_size):
        pb: torch.Tensor = pre_bincount_vec[iter_id]
        bin_count: torch.Tensor = torch.bincount(pb, minlength=num_classes**2)
        confusion_list.append(bin_count)

    confusion_vec: torch.Tensor = torch.stack(confusion_list)
    confusion_mat: torch.Tensor = confusion_vec.view(batch_size, num_classes, num_classes).to(torch.float32)  # BxKxK

    if normalized:
        norm_val: torch.Tensor = torch.sum(confusion_mat, dim=1, keepdim=True)
        confusion_mat = confusion_mat / (norm_val + 1e-6)

    return confusion_mat
