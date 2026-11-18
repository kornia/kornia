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

"""Score predictions against ground truth for classification, segmentation, detection, images, flow, stereo and pose.

A metric that compares a prediction with a target takes the prediction first. Only the flow and disparity metrics take
a ``reduction``; the scale and the pooling of every metric are in
:ref:`Losses and metrics <losses-metrics-conventions>` and in its own Convention block.

Which function to use, with the module form after the function:

- Classification: top-k accuracy in percent from class scores, :func:`accuracy`.
- Segmentation: the counts of target against predicted labels per sample, :func:`confusion_matrix`; the IoU of every
  class in every sample from label maps, :func:`mean_iou`.
- Detection: the IoU of every pair of boxes from two sets, :func:`mean_iou_bbox`; the PASCAL VOC mean average
  precision, :func:`mean_average_precision`.
- Image quality: the PSNR of two images in dB, :func:`psnr`; the SSIM map of two images, :func:`ssim`,
  :class:`SSIM`; of two volumes, :func:`ssim3d`, :class:`SSIM3D`.
- Optical flow: the average endpoint error of two flows, :func:`aepe`, :class:`AEPE`; also exported as
  :func:`average_endpoint_error`.
- Stereo: the mean absolute error of two disparity maps, :func:`mean_absolute_disparity_error`; the root mean squared
  error, :func:`root_mean_squared_disparity_error`; the fraction of pixels above an error threshold,
  :func:`mean_bad_pixel_error`; the KITTI D1 outlier fraction, :func:`kitti_d1_error`.
- Pose: the angle between two rotations, :func:`angle_error_mat`; between two vectors, :func:`angle_error_vec`; the
  distance between two translations, :func:`translation_ate`; the rotation angle and translation-direction angle of
  two poses, :func:`pose_errors`; the area under the error curve at thresholds, :func:`auc_from_errors`.
- Monitoring: the running average over batches, :class:`AverageMeter`.
"""

from .accuracy import accuracy
from .average_meter import AverageMeter
from .confusion_matrix import confusion_matrix
from .disparity import (
    kitti_d1_error,
    mean_absolute_disparity_error,
    mean_bad_pixel_error,
    root_mean_squared_disparity_error,
)
from .endpoint_error import AEPE, aepe, average_endpoint_error
from .mean_average_precision import mean_average_precision
from .mean_iou import mean_iou, mean_iou_bbox
from .pose import angle_error_mat, angle_error_vec, auc_from_errors, pose_errors, translation_ate
from .psnr import psnr
from .ssim import SSIM, ssim
from .ssim3d import SSIM3D, ssim3d

__all__ = [
    "AEPE",
    "SSIM",
    "SSIM3D",
    "AverageMeter",
    "accuracy",
    "aepe",
    "angle_error_mat",
    "angle_error_vec",
    "auc_from_errors",
    "average_endpoint_error",
    "confusion_matrix",
    "kitti_d1_error",
    "mean_absolute_disparity_error",
    "mean_average_precision",
    "mean_bad_pixel_error",
    "mean_iou",
    "mean_iou_bbox",
    "pose_errors",
    "psnr",
    "root_mean_squared_disparity_error",
    "ssim",
    "ssim3d",
    "translation_ate",
]
