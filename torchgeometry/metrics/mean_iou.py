import torch

from torchgeometry.metrics import confusion_matrix


def mean_iou(
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int) -> torch.Tensor:
    r"""Calculate mean Intersection-Over-Union (mIOU).

    The function internally computes the confusion matrix.

    Args:
        input (torch.Tensor) : tensor with estimated targets returned by a
          classifier. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1.
        target (torch.Tensor) : tensor with ground truth (correct) target
          values. The shape can be :math:`(B, *)` and must contain integer
          values between 0 and K-1, whete targets are assumed to be provided as
          one-hot vectors.
        num_classes (int): total possible number of classes in target.
        labels: torch.Tensor,
        predictions: torch.Tensor,
        num_classes: int) -> torch.Tensor:

    Returns:
        torch.Tensor: a tensor representing the mean intersection-over union
        with shape :math:`(B, C)` where C is the number of classes.
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
    # we first compute the confusion matrix
    conf_mat: torch.Tensor = confusion_matrix(input, target, num_classes)

    # allocate output tensor
    batch_size: int = conf_mat.shape[0]
    ious: torch.Tensor = torch.zeros(
        batch_size, num_classes, device=conf_mat.device)

    # TODO: is it possible to vectorize this ?
    # iterate over classes
    for class_id in range(num_classes):
        tp: torch.Tensor = conf_mat[..., None, class_id, class_id]
        total = torch.sum(conf_mat[..., class_id, :], dim=-1, keepdim=True) + \
            torch.sum(conf_mat[..., :, class_id], dim=-1, keepdim=True)
        iou_val: torch.Tensor = tp / (total.float() - tp + 1e-6)
        ious[..., class_id:class_id + 1] += iou_val
    return ious
