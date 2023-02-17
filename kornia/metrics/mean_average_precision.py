from typing import Dict, List, Tuple

import torch

from kornia.core import Tensor, concatenate, tensor, zeros

from .mean_iou import mean_iou_bbox


def mean_average_precision(
    pred_boxes: List[Tensor],
    pred_labels: List[Tensor],
    pred_scores: List[Tensor],
    gt_boxes: List[Tensor],
    gt_labels: List[Tensor],
    n_classes: int,
    threshold: float = 0.5,
) -> Tuple[Tensor, Dict[int, float]]:
    """Calculate the Mean Average Precision (mAP) of detected objects.

    Code altered from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py#L271.
    Background class (0 index) is excluded.

    Args:
        pred_boxes: a tensor list of predicted bounding boxes.
        pred_labels: a tensor list of predicted labels.
        pred_scores: a tensor list of predicted labels' scores.
        gt_boxes: a tensor list of ground truth bounding boxes.
        gt_labels: a tensor list of ground truth labels.
        n_classes: the number of classes.
        threshold: count as a positive if the overlap is greater than the threshold.

    Returns:
        mean average precision (mAP), list of average precisions for each class.

    Examples:
        >>> boxes, labels, scores = torch.tensor([[100, 50, 150, 100.]]), torch.tensor([1]), torch.tensor([.7])
        >>> gt_boxes, gt_labels = torch.tensor([[100, 50, 150, 100.]]), torch.tensor([1])
        >>> mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)
        (tensor(1.), {1: 1.0})
    """
    # these are all lists of tensors of the same length, i.e. number of images
    if not len(pred_boxes) == len(pred_labels) == len(pred_scores) == len(gt_boxes) == len(gt_labels):
        raise AssertionError

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    gt_images = []
    for i, labels in enumerate(gt_labels):
        gt_images.extend([i] * labels.size(0))
    # (n_objects), n_objects is the total no. of objects across all images
    _gt_boxes = concatenate(gt_boxes, 0)  # (n_objects, 4)
    _gt_labels = concatenate(gt_labels, 0)  # (n_objects)
    _gt_images = tensor(gt_images, device=_gt_boxes.device, dtype=torch.long)

    if not _gt_images.size(0) == _gt_boxes.size(0) == _gt_labels.size(0):
        raise AssertionError

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    pred_images = []
    for i, labels in enumerate(pred_labels):
        pred_images.extend([i] * labels.size(0))
    _pred_boxes = concatenate(pred_boxes, 0)  # (n_detections, 4)
    _pred_labels = concatenate(pred_labels, 0)  # (n_detections)
    _pred_scores = concatenate(pred_scores, 0)  # (n_detections)
    _pred_images = tensor(pred_images, device=_pred_boxes.device, dtype=torch.long)  # (n_detections)

    if not _pred_images.size(0) == _pred_boxes.size(0) == _pred_labels.size(0) == _pred_scores.size(0):
        raise AssertionError

    # Calculate APs for each class (except background)
    average_precisions = zeros((n_classes - 1), device=_pred_boxes.device, dtype=_pred_boxes.dtype)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        gt_class_images = _gt_images[_gt_labels == c]  # (n_class_objects)
        gt_class_boxes = _gt_boxes[_gt_labels == c]  # (n_class_objects, 4)

        # Keep track of which true objects with this class have already been 'detected'
        # (n_class_objects)
        gt_class_boxes_detected = zeros((gt_class_images.size(0)), dtype=torch.uint8, device=gt_class_images.device)

        # Extract only detections with this class
        pred_class_images = _pred_images[_pred_labels == c]  # (n_class_detections)
        pred_class_boxes = _pred_boxes[_pred_labels == c]  # (n_class_detections, 4)
        pred_class_scores = _pred_scores[_pred_labels == c]  # (n_class_detections)
        n_class_detections = pred_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        pred_class_scores, sort_ind = torch.sort(pred_class_scores, dim=0, descending=True)  # (n_class_detections)
        pred_class_images = pred_class_images[sort_ind]  # (n_class_detections)
        pred_class_boxes = pred_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        gt_positives = zeros(
            (n_class_detections,), dtype=pred_class_boxes.dtype, device=pred_class_boxes.device
        )  # (n_class_detections)
        false_positives = zeros(
            (n_class_detections,), dtype=pred_class_boxes.dtype, device=pred_class_boxes.device
        )  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = pred_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = pred_class_images[d]  # (), scalar

            # Find objects in the image with this class, their difficulties, and whether they have been detected before
            object_boxes = gt_class_boxes[gt_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = mean_iou_bbox(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'gt_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = tensor(
                range(gt_class_boxes.size(0)), device=gt_class_boxes_detected.device, dtype=torch.long
            )[gt_class_images == this_image][ind]
            # We need 'original_ind' to update 'gt_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > threshold:
                # If this object has already not been detected, it's a true positive
                if gt_class_boxes_detected[original_ind] == 0:
                    gt_positives[d] = 1
                    gt_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_gt_positives = torch.cumsum(gt_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_gt_positives / (
            cumul_gt_positives + cumul_false_positives + 1e-10
        )  # (n_class_detections)
        cumul_recall = cumul_gt_positives / _gt_boxes.size(0)  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).tolist()  # (11)
        precisions = zeros((len(recall_thresholds)), device=_gt_boxes.device, dtype=_gt_boxes.dtype)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.0
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_ap = average_precisions.mean()

    # Keep class-wise average precisions in a dictionary
    ap_dict = {c + 1: float(v) for c, v in enumerate(average_precisions.tolist())}

    return mean_ap, ap_dict
