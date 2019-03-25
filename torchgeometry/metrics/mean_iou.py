import torch

from torchgeometry.metrics import confusion_matrix


def mean_iou(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        num_classes: int) -> torch.Tensor:
    r"""Calculate mean Intersection-Over-Union (mIOU).

    Args:
        labels (torch.Tensor) : tensor with ground truth (correct) target
          values. The shape can be :math:`(N, *)`.
        predictions (torch.Tensor) : tensor with estimated targets returned by
          a classifier. The shape can be :math:`(N, *)`.
        num_classes (int): total possible number of classes in labels.

    Returns:
        torch.Tensor: a tensor representing the mean intersection-over union.
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not torch.is_tensor(predictions):
        raise TypeError("Input predictions type is not a torch.Tensor. Got {}"
                        .format(type(predictions)))
    if not labels.shape == predictions.shape:
        raise ValueError("Inputs labels and predictions must have the same "
                         "shape. Got: {}".format(labels.shape,
                                                 predictions.shape))
    if not labels.device == predictions.device:
        raise ValueError("Inputs must be in the same device. "
                         "Got: {} - {}".format(labels.device,
                                               predictions.device))
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("The number of classes must be an intenger bigger "
                         "than two. Got: {}".format(num_classes))
    # we first compute the confusion matrix
    conf_mat: torch.Tensor = confusion_matrix(labels, predictions, num_classes)

    # allocate output tensor
    batch_size: int = conf_mat.shape[0]
    ious: torch.Tensor = torch.zeros(
        batch_size, num_classes, device=conf_mat.device)

    # TODO: is it possible to vectorize this ?
    # iterate over classes
    for class_id in range(num_classes):
        tp: torch.Tensor = conf_mat[..., class_id, class_id].float()
        total = torch.sum(conf_mat[..., class_id, :], dim=-1, keepdim=True) + \
            torch.sum(conf_mat[..., :, class_id], dim=-1, keepdim=True)
        iou_val: torch.Tensor = tp / (total.float() - tp + 1e-6)
        ious[..., class_id:class_id + 1] += iou_val
    return ious
