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

try:
    from enum import Enum, member
except ImportError:
    from enum import Enum

    # Polyfill for Python < 3.11
    def member(obj):
        return obj


from functools import partial

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
    x_abs = x.abs().mul(1.0 / window_radius)

    poly1 = x_abs * (-1.8 * x_abs - 0.1) + 1.0
    poly2 = x_abs * (1.8 * x_abs - 3.7) + 1.9

    return torch.where(
        x_abs < 0.5, poly1, torch.where(x_abs <= 1.0, poly2, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    )


def rectangular_kernel(x: torch.Tensor, window_radius: float = 1.0) -> torch.Tensor:
    """Implementation of a rectangular kernel.

    Support: [-window_radius, window_radius]. Returns 1.0 inside this range, 0.0 otherwise.

    Args:
        x (torch.Tensor): signal, any shape
        window_radius (float): radius of window for the kernel

    Returns:
        torch.Tensor: transformed signal
    """
    x = torch.abs(x)
    return torch.where(x <= window_radius, 1.0, 0.0)


def truncated_gaussian_kernel(x: torch.Tensor, window_radius: float = 1.0) -> torch.Tensor:
    """Implementation of a truncated Gaussian kernel.

    Support: [-window_radius, window_radius]. Returns Gaussian value inside this range, 0.0 otherwise.
    Sigma is set to window_radius.

    Args:
        x (torch.Tensor): signal, any shape
        window_radius (float): radius of window for the kernel (used as sigma)

    Returns:
        torch.Tensor: transformed signal
    """
    sigma = window_radius
    mask = torch.abs(x) <= window_radius

    gaussian_val = torch.exp(-0.5 * (x / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)

    return torch.where(mask, gaussian_val, 0.0)


class MIKernel(Enum):
    xu = member(xu_kernel)
    rectangular = member(rectangular_kernel)
    truncated_gaussian = member(truncated_gaussian_kernel)


def _normalize_signal(data: torch.Tensor, num_bins: int, eps: float = 1e-8) -> torch.Tensor:
    min_val, _ = data.min(dim=-1)
    max_val, _ = data.max(dim=-1)
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
    """A base class for entropy-based loss functions using kernel density estimation.

    This class provides the foundation for computing entropy-based losses between signals
    by estimating probability distributions using kernel density estimation (KDE). It
    computes joint histograms and derives entropy measures that can be used to quantify
    the similarity or dissimilarity between signals.

    The class pre-processes a reference signal and provides methods to compute joint
    histograms with other signals, from which various entropy measures (joint, marginal,
    conditional, mutual information) can be derived in subclasses.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: MIKernel = MIKernel.xu,
        num_bins: int = 64,
        window_radius: float = 1.0,
    ) -> None:
        """Initialize the entropy-based loss base module.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0

        Raises:
            ValueError: If kernel_function is not a valid MIKernel member.
        """
        super().__init__()
        self.eps = torch.finfo(reference_signal.dtype).eps
        self.register_buffer("signal", _normalize_signal(reference_signal, num_bins, self.eps))
        self.num_bins = num_bins
        if kernel_function not in MIKernel:
            raise ValueError(
                f"The passed_kernel_function is not an accepted MIKernel, the available options are {list(MIKernel)}."
            )
        self.kernel_function = partial(kernel_function.value, window_radius=window_radius)
        self.window_radius = window_radius
        self.bin_centers = torch.arange(self.num_bins, device=self.signal.device)

    def _compute_joint_histogram(self, other_signal: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute the joint histogram between the reference signal and another signal.

        Uses kernel density estimation to estimate the joint probability distribution
        between the reference signal and the provided other signal. The histogram is
        computed by evaluating kernel functions at discretized signal values.

        Args:
            other_signal (torch.Tensor): Signal to compare with the reference signal.
                Must have the same shape as the reference signal.
            eps (float): Epsilon value for numerical stability in computations.

        Returns:
            torch.Tensor: Joint histogram tensor with shape [..., num_bins, num_bins]
                representing the estimated joint probability distribution.

        Raises:
            ValueError: If other_signal has incompatible shape with reference signal.
        """
        if other_signal.shape != self.signal.shape:
            raise ValueError(f"The two signals have incompatible shapes: {other_signal.shape} and {self.signal.shape}.")
        other_signal = _normalize_signal(other_signal, num_bins=self.num_bins, eps=eps)

        diff_1 = self.bin_centers.unsqueeze(-1) - self.signal.unsqueeze(-2)
        diff_2 = self.bin_centers.unsqueeze(-1) - other_signal.unsqueeze(-2)

        vals_1 = self.kernel_function(diff_1)
        vals_2 = self.kernel_function(diff_2)

        joint_histogram = torch.einsum("...in,...jn->...ij", vals_1, vals_2)

        return joint_histogram

    def entropies(self, other_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute entropy measures between the reference signal and another signal.

        Calculates joint entropy and marginal entropies based on the joint histogram of the two signals.

        Args:
            other_signal (torch.Tensor): Signal to compare with the reference signal.
                Must have the same shape as the reference_signal passed at instantiation.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Marginal entropy H(X) of reference signal
                - Marginal entropy H(Y) of other signal
                - Joint entropy H(X,Y)

            All tensors have the same batch dimensions as the input signals.

        Note:
            Subclasses should implement specific loss functions based on these entropy
            measures (e.g., mutual information).
        """
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
    """Mutual Information loss module specifically designed for 2D image data.

    This class extends MILossFromRef to handle 2D image inputs by automatically
    reshaping batched 2D images into the format expected by the base entropy
    computation methods. It computes mutual information between reference 2D images
    and other images using kernel density estimation.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: MIKernel = MIKernel.xu,
        num_bins: int = 64,
        window_radius: float = 1,
    ) -> None:
        """Initialize the 2D Mutual Information loss module.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 2D images, shape (B,H,W) where B are the batch
                dimensions.
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
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
    """Mutual Information loss module specifically designed for 3D image data.

    This class extends MILossFromRef to handle 3D image inputs by automatically
    reshaping batched 3D images into the format expected by the base entropy
    computation methods. It computes normalized mutual information between reference 3D images
    and other images using kernel density estimation.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: MIKernel = MIKernel.xu,
        num_bins: int = 64,
        window_radius: float = 1,
    ) -> None:
        """Initialize the 3D Mutual Information loss module.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 3D images, shape (B,D,H,W) where B are the
                batch dimensions.
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
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
    """Normalized Mutual Information loss module specifically designed for 2D image data.

    This class extends NMILossFromRef to handle 2D image inputs by automatically
    reshaping batched 2D images into the format expected by the base entropy
    computation methods. It computes normalized mutual information between reference 2D images
    and other images using kernel density estimation.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: MIKernel = MIKernel.xu,
        num_bins: int = 64,
        window_radius: float = 1,
    ) -> None:
        """Initialize the 2D Normalized Mutual Information loss module.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 2D images, shape (B,H,W) where B are the batch
                dimensions.
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
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
    """Normalized Mutual Information loss module specifically designed for 3D image data.

    This class extends NMILossFromRef to handle 3D image inputs by automatically
    reshaping batched 3D images into the format expected by the base entropy
    computation methods. It computes mutual information between reference 3D images
    and other images using kernel density estimation.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        kernel_function: MIKernel = MIKernel.xu,
        num_bins: int = 64,
        window_radius: float = 1,
    ) -> None:
        """Initialize the 3D Normalized Mutual Information loss module.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method.
                batch of 3D images, shape (B,D,H,W) where B are the
                batch dimensions.
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
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
    kernel_function: MIKernel = MIKernel.xu,
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
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
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
    kernel_function: MIKernel = MIKernel.xu,
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
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
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
    kernel_function: MIKernel = MIKernel.xu,
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
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
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
    kernel_function: MIKernel = MIKernel.xu,
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
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
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
    kernel_function: MIKernel = MIKernel.xu,
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
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
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
    kernel_function: MIKernel = MIKernel.xu,
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
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
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
