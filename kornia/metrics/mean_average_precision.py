# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Dict, List, Tuple

import torch

from kornia.core.check import KORNIA_CHECK

from .mean_iou import mean_iou_bbox


def mean_average_precision(
    pred_boxes: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    n_classes: int,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Calculate the Mean Average Precision (mAP) of detected objects.

    Code altered from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py#L271.

    Convention:
        - Every argument but ``n_classes`` and ``threshold`` is a list with one tensor per image. The call raises when
          the five lists differ in length, when the boxes, labels and scores of an image differ in their number of
          rows, or when a label lies outside ``[0, n_classes)``. Boxes are exclusive ``xyxy``, the default format of
          :func:`~kornia.metrics.mean_iou_bbox`, which computes the overlaps.
        - Class 0 is background: its objects and detections are never scored, and ``n_classes`` counts it, so the
          classes ``1`` to ``n_classes - 1`` are scored.
        - The detections of a class are ranked by score over all images at once, so the AP is not a mean of
          per-image APs. In that order, a detection takes the object of its image and class that it overlaps most;
          it is a true positive when that IoU is strictly greater than ``threshold`` and the object is not taken
          yet, and a false positive otherwise. A class with objects and no detection has AP 0.
        - A class without objects in any image has no AP: its entry is ``-1.0`` and it stays out of the mean, so
          neither a larger ``n_classes`` nor detections of such a class change the mAP. Without any foreground
          object the mAP is ``-1.0`` too.
        - The AP is the 11-point interpolated AP of PASCAL VOC2007: the mean, over the recall levels
          ``0, 0.1, ..., 1``, of the highest precision at a recall of at least that level.
        - The result is a 0-d tensor and a dict of Python floats, fractions in :math:`[0, 1]` apart from that
          ``-1.0``. The tensor takes the floating dtype of the boxes: integer boxes count as float32, and predicted
          and ground-truth boxes of different dtypes give the promotion of those two floating dtypes (float32 when one
          set is integer and the other float16). :ref:`Losses and metrics <losses-metrics-conventions>` compares the
          match rule with the VOC devkit and COCO.
        - Known defect: labels are checked for their range only, so a fractional label such as ``1.5`` matches no
          class and its detections and objects are dropped without an error
          (`#5629 <https://github.com/kornia/kornia/issues/5629>`_).

    Args:
        pred_boxes: a torch.Tensor list of predicted bounding boxes.
        pred_labels: a torch.Tensor list of predicted labels in ``[0, n_classes)``.
        pred_scores: a torch.Tensor list of predicted labels' scores.
        gt_boxes: a torch.Tensor list of ground truth bounding boxes.
        gt_labels: a torch.Tensor list of ground truth labels in ``[0, n_classes)``.
        n_classes: the number of classes, background (class 0) included.
        threshold: count as a positive if the overlap is greater than the threshold.

    Returns:
        the mAP, and a dict mapping each scored class id to its average precision.

    Examples:
        >>> boxes, labels, scores = torch.tensor([[100, 50, 150, 100.]]), torch.tensor([1]), torch.tensor([.7])
        >>> gt_boxes, gt_labels = torch.tensor([[100, 50, 150, 100.]]), torch.tensor([1])
        >>> mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)
        (tensor(1.), {1: 1.0})

    """
    # these are all lists of tensors of the same length, i.e. number of images
    KORNIA_CHECK(
        len(pred_boxes) == len(pred_labels) == len(pred_scores) == len(gt_boxes) == len(gt_labels),
        "The five per-image lists must have the same length. Got: "
        f"pred_boxes {len(pred_boxes)}, pred_labels {len(pred_labels)}, pred_scores {len(pred_scores)}, "
        f"gt_boxes {len(gt_boxes)}, gt_labels {len(gt_labels)}",
    )

    # Store all (true) objects in a single continuous torch.Tensor while keeping track of the image it is from.
    # The counts are checked per image: totals that agree can still pair the rows of one image with another's.
    gt_images = []
    for i, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
        KORNIA_CHECK(
            boxes.size(0) == labels.size(0),
            f"gt_boxes and gt_labels must have one row per object in every image. Got image {i}: {boxes.size(0)} "
            f"boxes and {labels.size(0)} labels",
        )
        gt_images.extend([i] * labels.size(0))
    # (n_objects), n_objects is the total no. of objects across all images
    _gt_boxes = torch.cat(gt_boxes, 0)  # (n_objects, 4)
    _gt_labels = torch.cat(gt_labels, 0)  # (n_objects)
    _gt_images = torch.tensor(gt_images, device=_gt_boxes.device, dtype=torch.long)

    # Store all detections in a single continuous torch.Tensor while keeping track of the image it is from
    pred_images = []
    for i, (boxes, labels, scores) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        KORNIA_CHECK(
            boxes.size(0) == labels.size(0) == scores.size(0),
            f"pred_boxes, pred_labels and pred_scores must have one row per detection in every image. Got image {i}: "
            f"{boxes.size(0)} boxes, {labels.size(0)} labels and {scores.size(0)} scores",
        )
        pred_images.extend([i] * labels.size(0))
    _pred_boxes = torch.cat(pred_boxes, 0)  # (n_detections, 4)
    _pred_labels = torch.cat(pred_labels, 0)  # (n_detections)
    _pred_scores = torch.cat(pred_scores, 0)  # (n_detections)
    _pred_images = torch.tensor(pred_images, device=_pred_boxes.device, dtype=torch.long)  # (n_detections)

    # The precisions need a floating dtype. Integer boxes count as float32, as mean_iou_bbox computes their overlaps,
    # and the two sets meet in their promoted dtype: integer and float16 boxes give float32, not float16.
    ap_dtype = torch.promote_types(
        _pred_boxes.dtype if _pred_boxes.is_floating_point() else torch.float32,
        _gt_boxes.dtype if _gt_boxes.is_floating_point() else torch.float32,
    )

    for name, labels in (("pred_labels", _pred_labels), ("gt_labels", _gt_labels)):
        KORNIA_CHECK(
            bool(((labels >= 0) & (labels < n_classes)).all()),
            f"{name} must satisfy 0 <= label < n_classes ({n_classes}).",
        )
        if labels.is_floating_point():
            KORNIA_CHECK(bool((labels == labels.round()).all()), f"{name} must contain integer-valued labels.")

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), device=_pred_boxes.device, dtype=ap_dtype)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        gt_class_images = _gt_images[_gt_labels == c]  # (n_class_objects)
        gt_class_boxes = _gt_boxes[_gt_labels == c]  # (n_class_objects, 4)

        if gt_class_images.size(0) == 0:
            average_precisions[c - 1] = -1.0
            continue

        # Keep track of which true objects with this class have already been 'detected'
        # (n_class_objects)
        gt_class_boxes_detected = torch.zeros(
            (gt_class_images.size(0)), dtype=torch.uint8, device=gt_class_images.device
        )

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
        gt_positives = torch.zeros((n_class_detections,), dtype=ap_dtype, device=pred_class_boxes.device)
        false_positives = torch.zeros((n_class_detections,), dtype=ap_dtype, device=pred_class_boxes.device)
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
            original_ind = torch.tensor(
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
        cumul_recall = cumul_gt_positives / gt_class_images.size(0)  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        # Exact tenths as Python floats: each comparison below casts them to the recall dtype. A float32 arange widened
        # to Python floats gives 0.10000000149..., which a float64 recall of exactly 1/10 never reaches (#5083).
        recall_thresholds = [i / 10 for i in range(11)]  # (11)
        precisions = torch.zeros((len(recall_thresholds)), device=_gt_boxes.device, dtype=ap_dtype)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.0
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    defined_precisions = average_precisions[average_precisions >= 0]
    mean_ap = defined_precisions.mean() if defined_precisions.numel() else average_precisions.new_tensor(-1.0)

    # Keep class-wise average precisions in a dictionary
    ap_dict = {c + 1: float(v) for c, v in enumerate(average_precisions.tolist())}

    return mean_ap, ap_dict
