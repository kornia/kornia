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


def parzen_window_kernel(x: torch.Tensor, window_radius: float = 1.0) -> torch.Tensor:
    """Implementation of the 2nd-order polynomial kernel (Xu et al., 2008).

    Range: [-1, 1]. Returns 0 outside this range.

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
    kernel_function=parzen_window_kernel,
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

    # density_1 = vals_1.sum(axis=-2)
    # density_1 /= density_1.sum()
    # density_2 = vals_2.sum(axis=-2)
    # density_2 /= density_2.sum()

    joint_histogram = torch.einsum("...ni,...nj->...ij", vals_1, vals_2)
    joint_density = joint_histogram / (joint_histogram.sum(dim=(-1, -2)))

    return joint_density


def normalized_mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function=parzen_window_kernel,
    num_bins: int = 64,
    window_radius: float = 2.0,
) -> torch.Tensor:
    """Calculates the Negative Normalized Mutual Information Loss.

    loss = - (H(X) + H(Y)) / H(X,Y)
    """
    if input.shape != target.shape:
        raise ValueError(f"Shape mismatch: {input.shape} != {target.shape}")

    P_xy = compute_joint_histogram(
        input, target, kernel_function=kernel_function, num_bins=num_bins, window_radius=window_radius
    )

    P_x = P_xy.sum(dim=-2)
    P_y = P_xy.sum(dim=-1)

    eps = 1e-8
    H_x = -torch.sum(P_x * torch.log(P_x + eps))
    H_y = -torch.sum(P_y * torch.log(P_y + eps))
    H_xy = -torch.sum(P_xy * torch.log(P_xy + eps))

    nmi = (H_x + H_y) / (H_xy + eps)

    return -nmi


def mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function=parzen_window_kernel,
    num_bins: int = 64,
    window_radius: float = 2.0,
) -> torch.Tensor:
    """Calculates the Negative Normalized Mutual Information Loss.

    loss = - (H(X) + H(Y)) / H(X,Y)
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

    P_x = P_xy.sum(dim=-2)
    P_y = P_xy.sum(dim=-1)

    eps = 1e-8
    H_x = -torch.sum(P_x * torch.log(P_x + eps))
    H_y = -torch.sum(P_y * torch.log(P_y + eps))
    H_xy = -torch.sum(P_xy * torch.log(P_xy + eps))

    mi = H_x + H_y - H_xy

    return -mi
