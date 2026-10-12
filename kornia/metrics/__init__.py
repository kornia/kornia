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

"""Metrics submodule for Kornia.

This package provides evaluation metrics for computer vision, such as accuracy, mean IoU,
PSNR, SSIM, endpoint error, and stereo disparity error.
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
