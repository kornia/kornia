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
from __future__ import annotations

from typing import Callable

import torch


def xu_kernel(x: torch.Tensor, window_radius: float = 1.0) -> torch.Tensor:
    """Implementation of a 2nd-order polynomial kernel for Kernel density estimate (Xu et al., 2008).

    Support: [-window_radius, window_radius]. Returns 0 outside this range.
    Ref: "Parzen-Window Based Normalized Mutual Information for Medical Image Registration", Eq. 22.

    Args:
        x (torch.Tensor): signal, any shape
        window_radius (float): radius of window for the kernel

    Returns:
        torch.Tensor: transformed signal
    """
    x = torch.abs(x) / window_radius

    kernel_val = torch.zeros_like(x)

    mask1 = x < 0.5
    x_mask1 = x[mask1]
    kernel_val[mask1] = -1.8 * (x_mask1**2) - 0.1 * x_mask1 + 1.0
    mask2 = (x >= 0.5) & (x <= 1.0)
    x_mask2 = x[mask2]
    kernel_val[mask2] = 1.8 * (x_mask2**2) - 3.7 * x_mask2 + 1.9

    return kernel_val


def _normalize_signal(data: torch.Tensor, num_bins: int, eps: float = 1e-8) -> torch.Tensor:
    min_val, _ = data.min(axis=-1)
    max_val, _ = data.max(axis=-1)
    diff = (max_val - min_val).unsqueeze(-1)
    # signal is considered trivial if too low variation
    return torch.where(diff > eps, (data - min_val.unsqueeze(-1)) / diff * num_bins, 0)


def _joint_histogram_to_entropies(joint_histogram: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    P_xy = joint_histogram
    # clamp for numerical stability
    P_xy = P_xy.clamp(eps)
    # divide by sum to get a density
    P_xy /= P_xy.sum(dim=(-1, -2), keepdim=True)

    P_x = P_xy.sum(dim=-2)
    P_y = P_xy.sum(dim=-1)
    H_xy = torch.sum(-P_xy * torch.log(P_xy), dim=(-1, -2))
    H_x = torch.sum(-P_x * torch.log(P_x), dim=-1)
    H_y = torch.sum(-P_y * torch.log(P_y), dim=-1)

    return H_x, H_y, H_xy


class EntropyBasedLossBase(torch.nn.Module):
    """A base class for entropy based losses."""

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: Callable = xu_kernel,
        num_bins: int = 64,
        window_radius: float = 1.0,
    ):
        """Instantiation.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method
            kernel_function (Callable): Used kernel function for kernel
                density estimate, by default xu_kernel
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__()
        self.eps = torch.finfo(reference_signal.dtype).eps
        self.register_buffer("signal", _normalize_signal(reference_signal, num_bins, self.eps))
        self.num_bins = num_bins
        self.kernel_function = kernel_function
        self.window_radius = window_radius
        self.bin_centers = torch.arange(self.num_bins, device=self.signal.device)

    def _compute_joint_histogram(self, other_signal: torch.Tensor, eps: float) -> torch.Tensor:
        if other_signal.shape != self.signal.shape:
            raise ValueError(f"The two signals have incompatible shapes: {other_signal.shape} and {self.signal.shape}.")
        other_signal = _normalize_signal(other_signal, num_bins=self.num_bins, eps=eps)

        diff_1 = self.bin_centers.unsqueeze(-1) - self.signal.unsqueeze(-2)
        diff_2 = self.bin_centers.unsqueeze(-1) - other_signal.unsqueeze(-2)

        vals_1 = self.kernel_function(diff_1, window_radius=self.window_radius)
        vals_2 = self.kernel_function(diff_2, window_radius=self.window_radius)

        joint_histogram = torch.einsum("...in,...jn->...ij", vals_1, vals_2)

        return joint_histogram

    def entropies(self, other_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_histogram = self._compute_joint_histogram(other_signal, self.eps)
        return _joint_histogram_to_entropies(joint_histogram, eps=self.eps)


class MILossFromRef(EntropyBasedLossBase):
    def forward(self, other_signal: torch.Tensor) -> torch.Tensor:
        """Compute differentiable mutual information for self.signal and other_signal.

        mi = (H(X) + H(Y) - H(X,Y))
        To have a loss function, the opposite is returned.
        Can also handle two batches of flat tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of flat tensors shape (B,N) where B is
                the tuple of batch dimensions for self.signal, possibly empty.

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        H_x, H_y, H_xy = self.entropies(other_signal)
        mi = H_x + H_y - H_xy

        return -mi


class NMILossFromRef(EntropyBasedLossBase):
    def forward(self, other_signal: torch.Tensor) -> torch.Tensor:
        """Compute differentiable normalized mutual information for self.signal and other_signal.

        nmi = (H(X) + H(Y)) / H(X,Y)
        To have a loss function, the opposite is returned.
        Can also handle two batches of flat tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of flat tensors shape (B,N) where B is
                the tuple of batch dimensions for self.signal, possibly empty.

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        H_x, H_y, H_xy = self.entropies(other_signal)
        nmi = (H_x + H_y) / H_xy

        return -nmi


class MILossFromRef2D(MILossFromRef):
    """MILossFromRef for 2D images."""

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: Callable = xu_kernel,
        num_bins: int = 64,
        window_radius: float = 1,
    ):
        """Instantiation.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 2D images, shape (B,H,W) where B are the batch
                dimensions.
            kernel_function (Callable): Used kernel function for kernel
                density estimate, by default xu_kernel
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(self.arrange_shape(reference_signal), kernel_function, num_bins, window_radius)

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[:-2] + (-1,))

    def forward(self, other_signal: torch.Tensor) -> torch.Tensor:
        """Compute differentiable mutual information for reference_signal and other_signal (both supposed 2D).

        mi = (H(X) + H(Y) - H(X,Y))
        To have a loss function, the opposite is returned.
        Can also handle two batches of flat tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of flat tensors same shape (B,H,W) as
                reference_signal passed for instantiation

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        return super().forward(self.arrange_shape(other_signal))


class MILossFromRef3D(MILossFromRef):
    """MILossFromRef for 3D images."""

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: Callable = xu_kernel,
        num_bins: int = 64,
        window_radius: float = 1,
    ):
        """Instantiation.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 3D images, shape (B,D,H,W) where B are the
                batch dimensions.
            kernel_function (Callable): Used kernel function for kernel
                density estimate, by default xu_kernel
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(self.arrange_shape(reference_signal), kernel_function, num_bins, window_radius)

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[:-3] + (-1,))

    def forward(self, other_signal: torch.Tensor) -> torch.Tensor:
        """Compute differentiable mutual information for reference_signal and other_signal (both supposed 3D).

        mi = (H(X) + H(Y) - H(X,Y))
        To have a loss function, the opposite is returned.
        Can also handle two batches of flat tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of flat tensors same shape (B,D,H,W) as
                reference_signal passed for instantiation

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        return super().forward(self.arrange_shape(other_signal))


class NMILossFromRef2D(NMILossFromRef):
    """NMILossFromRef for 2D images."""

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: Callable = xu_kernel,
        num_bins: int = 64,
        window_radius: float = 1,
    ):
        """Instantiation.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 2D images, shape (B,H,W) where B are the batch
                dimensions.
            kernel_function (Callable): Used kernel function for kernel
                density estimate, by default xu_kernel
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(self.arrange_shape(reference_signal), kernel_function, num_bins, window_radius)

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[:-2] + (-1,))

    def forward(self, other_signal: torch.Tensor) -> torch.Tensor:
        """Compute differentiable mutual information for reference_signal and other_signal (both supposed 2D).

        nmi = (H(X) + H(Y)) / H(X,Y)
        To have a loss function, the opposite is returned.
        Can also handle two batches of tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of tensors same shape (B,H,W) as
                reference_signal passed for instantiation

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        return super().forward(self.arrange_shape(other_signal))


class NMILossFromRef3D(NMILossFromRef):
    """NMILossFromRef for 3D images. Check details there."""

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: Callable = xu_kernel,
        num_bins: int = 64,
        window_radius: float = 1,
    ):
        """Instantiation.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 3D images, shape (B,D,H,W) where B are the
                batch dimensions.
            kernel_function (Callable): Used kernel function for kernel
                density estimate, by default xu_kernel
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(self.arrange_shape(reference_signal), kernel_function, num_bins, window_radius)

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[:-3] + (-1,))

    def forward(self, other_signal: torch.Tensor) -> torch.Tensor:
        """Compute differentiable mutual information for reference_signal and other_signal (both supposed 3D).

        nmi = (H(X) + H(Y)) / H(X,Y)
        To have a loss function, the opposite is returned.
        Can also handle two batches of flat tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of tensors, same shape (B,D,H,W) as
                reference_signal passed for instantiation

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        return super().forward(self.arrange_shape(other_signal))


def mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function: Callable = xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable mutual information for flat tensors.

    mi = (H(X) + H(Y) - H(X,Y))
    To have a loss function, the opposite is returned.
    Can also handle two batches of flat tensors, then a batch of loss values is returned.

    Args:
        input (torch.Tensor): Batch of flat tensors shape (B,N) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of flat tensors, same shape as
            input.
        kernel_function: The kernel function used for KDE, defaults to
            built-in xu_kernel.
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    module = MILossFromRef(
        reference_signal=target, kernel_function=kernel_function, num_bins=num_bins, window_radius=window_radius
    )
    return module.forward(input)


def mutual_information_loss_2d(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function: Callable = xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable mutual information for 2d tensors.

    nmi = (H(X) + H(Y)) / H(X,Y)
    To have a loss function, the opposite is returned.
    Can also handle two batches of 2d tensors, then a batch of loss values is returned.

    Args:
        input (torch.Tensor): Batch of 2d tensors shape (B,H,W) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 2d tensors, same shape as input.
        kernel_function: The kernel function used for KDE, defaults to
            built-in xu_kernel.
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    module = MILossFromRef2D(
        reference_signal=target, kernel_function=kernel_function, num_bins=num_bins, window_radius=window_radius
    )
    return module.forward(input)


def mutual_information_loss_3d(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function: Callable = xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable mutual information for 3d tensors.

    nmi = (H(X) + H(Y)) / H(X,Y)
    To have a loss function, the opposite is returned.
    Can also handle two batches of 3d tensors, then a batch of loss values is returned.

    Args:
        input (torch.Tensor): Batch of 3d tensors shape (B,D,H,W) where
            B is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 3d tensors, same shape as input.
        kernel_function: The kernel function used for KDE, defaults to
            built-in xu_kernel.
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    module = MILossFromRef3D(
        reference_signal=target, kernel_function=kernel_function, num_bins=num_bins, window_radius=window_radius
    )
    return module.forward(input)


def normalized_mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function: Callable = xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable normalized mutual information for flat tensors.

    nmi = (H(X) + H(Y)) / H(X,Y)
    To have a loss function, the opposite is returned.
    Can also handle two batches of flat tensors, then a batch of loss values is returned.

    Args:
        input (torch.Tensor): Batch of flat tensors shape (B,N) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of flat tensors, same shape as
            input.
        kernel_function: The kernel function used for KDE, defaults to
            built-in xu_kernel.
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    module = NMILossFromRef(
        reference_signal=target,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )
    return module.forward(input)


def normalized_mutual_information_loss_2d(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function: Callable = xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable normalized mutual information for 2d tensors.

    mi = (H(X) + H(Y) - H(X,Y))
    To have a loss function, the opposite is returned.
    Can also handle two batches of 2d tensors, then a batch of loss values is returned.

    Args:
        input (torch.Tensor): Batch of 2d tensors shape (B,H,W) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 2d tensors, same shape as input.
        kernel_function: The kernel function used for KDE, defaults to
            built-in xu_kernel.
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    module = NMILossFromRef2D(
        reference_signal=target, kernel_function=kernel_function, num_bins=num_bins, window_radius=window_radius
    )
    return module.forward(input)


def normalized_mutual_information_loss_3d(
    input: torch.Tensor,
    target: torch.Tensor,
    kernel_function: Callable = xu_kernel,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable normalized mutual information for 3d tensors.

    mi = (H(X) + H(Y) - H(X,Y))
    To have a loss function, the opposite is returned.
    Can also handle two batches of 3d tensors, then a batch of loss values is returned.

    Args:
        input (torch.Tensor): Batch of 3d tensors shape (B,D,H,W) where
            B is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 3d tensors, same shape as input.
        kernel_function: The kernel function used for KDE, defaults to
            built-in xu_kernel.
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    module = NMILossFromRef3D(
        reference_signal=target, kernel_function=kernel_function, num_bins=num_bins, window_radius=window_radius
    )
    return module.forward(input)
