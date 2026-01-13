# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Alignment utilities for depth estimation and metric scaling.
"""

from typing import Tuple
import torch


def least_squares_scale_scalar(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute least squares scale factor s such that a ≈ s * b.

    Args:
        a: First tensor
        b: Second tensor
        eps: Small epsilon for numerical stability

    Returns:
        Scalar tensor containing the scale factor

    Raises:
        ValueError: If tensors have mismatched shapes or devices
        TypeError: If tensors are not floating point
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.device != b.device:
        raise ValueError(f"Device mismatch: {a.device} vs {b.device}")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Tensors must be floating point type")

    # Compute dot products for least squares solution
    num = torch.dot(a.reshape(-1), b.reshape(-1))
    den = torch.dot(b.reshape(-1), b.reshape(-1)).clamp_min(eps)
    return num / den


def compute_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    Compute non-sky mask from sky prediction.

    Args:
        sky_prediction: Sky prediction tensor
        threshold: Threshold for sky classification

    Returns:
        Boolean mask where True indicates non-sky regions
    """
    return sky_prediction < threshold


def compute_alignment_mask(
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    depth: torch.Tensor,
    metric_depth: torch.Tensor,
    median_conf: torch.Tensor,
    min_depth_threshold: float = 1e-3,
    min_metric_depth_threshold: float = 1e-2,
) -> torch.Tensor:
    """
    Compute mask for depth alignment based on confidence and depth thresholds.

    Args:
        depth_conf: Depth confidence tensor
        non_sky_mask: Non-sky region mask
        depth: Predicted depth tensor
        metric_depth: Metric depth tensor
        median_conf: Median confidence threshold
        min_depth_threshold: Minimum depth threshold
        min_metric_depth_threshold: Minimum metric depth threshold

    Returns:
        Boolean mask for valid alignment regions
    """
    return (
        (depth_conf >= median_conf)
        & non_sky_mask
        & (metric_depth > min_metric_depth_threshold)
        & (depth > min_depth_threshold)
    )


def sample_tensor_for_quantile(tensor: torch.Tensor, max_samples: int = 100000) -> torch.Tensor:
    """
    Sample tensor elements for quantile computation to reduce memory usage.

    Args:
        tensor: Input tensor to sample
        max_samples: Maximum number of samples to take

    Returns:
        Sampled tensor
    """
    if tensor.numel() <= max_samples:
        return tensor

    idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
    return tensor.flatten()[idx]


def apply_metric_scaling(
    depth: torch.Tensor, intrinsics: torch.Tensor, scale_factor: float = 300.0
) -> torch.Tensor:
    """
    Apply metric scaling to depth based on camera intrinsics.

    Args:
        depth: Input depth tensor
        intrinsics: Camera intrinsics tensor
        scale_factor: Scaling factor for metric conversion

    Returns:
        Scaled depth tensor
    """
    focal_length = (intrinsics[:, :, 0, 0] + intrinsics[:, :, 1, 1]) / 2
    return depth * (focal_length[:, :, None, None] / scale_factor)


def set_sky_regions_to_max_depth(
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    max_depth: float = 200.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Set sky regions to maximum depth and high confidence.

    Args:
        depth: Depth tensor
        depth_conf: Depth confidence tensor
        non_sky_mask: Non-sky region mask
        max_depth: Maximum depth value for sky regions

    Returns:
        Tuple of (updated_depth, updated_depth_conf)
    """
    depth = depth.clone()

    # Set sky regions to max depth and high confidence
    depth[~non_sky_mask] = max_depth
    if depth_conf is not None:
        depth_conf = depth_conf.clone()
        depth_conf[~non_sky_mask] = 1.0
        return depth, depth_conf
    else:
        return depth, None
