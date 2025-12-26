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

import torch


def xu_kernel(x: torch.Tensor, window_radius: float = 1.0) -> torch.Tensor:
    """Implementation of a 2nd-order polynomial kernel (Xu et al., 2008).

    Support: [-window_radius, window_radius]. Returns 0 outside this range.

    Ref: "Parzen-Window Based Normalized Mutual Information for Medical Image Registration", Eq. 22.
    """
    x = torch.abs(x) / window_radius

    kernel_val = torch.zeros_like(x)

    mask1 = x < 0.5
    kernel_val[mask1] = -1.8 * (x[mask1] ** 2) - 0.1 * x[mask1] + 1.0
    mask2 = (x >= 0.5) & (x <= 1.0)
    kernel_val[mask2] = 1.8 * (x[mask2] ** 2) - 3.7 * x[mask2] + 1.9

    return kernel_val


def _normalize_signal(data: torch.Tensor, num_bins: int):
    min_val, _ = data.min(axis=-1)
    max_val, _ = data.max(axis=-1)
    return (data - min_val.unsqueeze(-1)) / (max_val - min_val).unsqueeze(-1) * num_bins


def compute_joint_histogram(
    img_1: torch.Tensor,
    img_2: torch.Tensor,
    kernel_function=xu_kernel,
    num_bins: int = 64,
    window_radius: float = 2,
) -> torch.Tensor:
    """Computes the differentiable Joint Histogram using Parzen Window estimation.

    Input shapes: (B,N) or (N,)
    Output shape: (num_bins, num_bins)
    """
    img_1 = _normalize_signal(img_1, num_bins=num_bins)
    img_2 = _normalize_signal(img_2, num_bins=num_bins)

    bin_centers = torch.arange(num_bins, device=img_1.device)

    diff_1 = bin_centers.unsqueeze(-1) - img_1.unsqueeze(-2)
    diff_2 = bin_centers.unsqueeze(-1) - img_2.unsqueeze(-2)

    vals_1 = kernel_function(diff_1, window_radius=window_radius)
    vals_2 = kernel_function(diff_2, window_radius=window_radius)

    joint_histogram = torch.einsum("...in,...jn->...ij", vals_1, vals_2)

    return joint_histogram


def _joint_histogram_to_entropies(joint_histogram):
    P_xy = joint_histogram
    eps = torch.finfo(P_xy.dtype).eps
    # clamp for numerical stability
    P_xy = P_xy.clamp(eps)
    # divide by sum to get a density
    P_xy /= P_xy.sum(dim=(-1, -2), keepdim=True)

    P_x = P_xy.sum(dim=-2)
    P_y = P_xy.sum(dim=-1)
    eps = torch.finfo(P_xy.dtype).eps
    H_xy = torch.sum(-P_xy * torch.log(P_xy), dim=(-1, -2))
    H_x = torch.sum(-P_x * torch.log(P_x), dim=-1)
    H_y = torch.sum(-P_y * torch.log(P_y), dim=-1)

    return H_x, H_y, H_xy


def normalized_mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function=xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable normalized mutual information for for flat tensors.

    nmi = (H(X) + H(Y)) / H(X,Y)
    To have a loss function, the opposite is returned.
    Can also handle two batches of flat tensors, then a batch of loss values is returned.

    :param input: Batch of flat tensors shape (B,N) where B is any batch dimensions tuple, possibly empty
    :type input: torch.Tensor
    :param target: Batch of flat tensors, same shape as input.
    :type target: torch.Tensor
    :param kernel_function: The kernel function used for KDE, defaults to built-in xu_kernel.
    :param num_bins:The number of bins used for KDE, defaults to 64.
    :type num_bins: int
    :param window_radius: The smoothing window radius in KDE, in terms of bin width units, defaults to 1.
    :type window_radius: float
    :return: tensor of losses, shape B (common batch dims tuple of input and target)
    :rtype: torch.Tensor
    """
    if input.shape != target.shape:
        raise ValueError(f"Shape mismatch: {input.shape} != {target.shape}")

    P_xy = compute_joint_histogram(
        input,
        target,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )

    H_x, H_y, H_xy = _joint_histogram_to_entropies(P_xy)
    nmi = (H_x + H_y) / H_xy

    return -nmi


def mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function=xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable mutual information for for flat tensors.

    mi = (H(X) + H(Y) - H(X,Y))
    To have a loss function, the opposite is returned.
    Can also handle two batches of flat tensors, then a batch of loss values is returned.

    :param input: Batch of flat tensors shape (B,N) where B is any batch dimensions tuple, possibly empty
    :type input: torch.Tensor
    :param target: Batch of flat tensors, same shape as input.
    :type target: torch.Tensor
    :param kernel_function: The kernel function used for KDE, defaults to built-in xu_kernel.
    :param num_bins:The number of bins used for KDE, defaults to 64.
    :type num_bins: int
    :param window_radius: The smoothing window radius in KDE, in terms of bin width units, defaults to 1.
    :type window_radius: float
    :return: tensor of losses, shape B (common batch dims tuple of input and target)
    :rtype: torch.Tensor
    """
    if input.shape != target.shape:
        raise ValueError(f"Shape mismatch: {input.shape} != {target.shape}")

    P_xy = compute_joint_histogram(
        input,
        target,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )

    H_x, H_y, H_xy = _joint_histogram_to_entropies(P_xy)
    mi = H_x + H_y - H_xy

    return -mi
