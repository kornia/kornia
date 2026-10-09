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

from enum import Enum, member
from functools import partial

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_SHAPE
from kornia.core.utils import is_exporting


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
    return (x <= window_radius).to(x.dtype)


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
    """Available kernels for mutual-information density estimation.

    Pass a member as the ``kernel_function`` of the mutual-information losses. The members are not callable: a
    member's ``value`` is the kernel ``f(d, window_radius=1.0)``, the weight that a sample at distance ``d`` from a bin
    centre, in bin widths, gives that bin.

    Convention:
        - Every kernel is 0 outside ``|d| <= window_radius``. With ``u = |d| / window_radius``:

          - ``xu`` (Xu et al., 2008), the default: :math:`1 - 0.1 u - 1.8 u^2` for :math:`u < 0.5` and
            :math:`1.9 - 3.7 u + 1.8 u^2` up to :math:`u = 1`. At ``window_radius=1`` the weights of a sample
            between the first and the last bin centre sum to 1.
          - ``rectangular``: 1 on the closed support, so a sample counts fully in every bin within
            ``window_radius``. Its loss has no gradient (evaluation only, see
            :func:`~kornia.losses.mutual_information_loss`).
          - ``truncated_gaussian``: the normal density with standard deviation ``window_radius``, cut at one
            standard deviation.
    """

    xu = member(xu_kernel)
    rectangular = member(rectangular_kernel)
    truncated_gaussian = member(truncated_gaussian_kernel)


def _validate_mask(mask: torch.Tensor | None, shape: torch.Size) -> None:
    if mask is not None:
        KORNIA_CHECK_IS_TENSOR(mask, "mask must be a boolean tensor or None.")
        KORNIA_CHECK(mask.dtype in (torch.bool, torch.uint8), "mask must have boolean dtype (torch.bool or uint8).")
        KORNIA_CHECK(
            mask.shape == shape,
            f"mask must have one-sample shape {shape}, common to the batch. Got {mask.shape}.",
        )


def _flatten_mask(mask: torch.Tensor | None, shape: torch.Size) -> torch.Tensor | None:
    _validate_mask(mask, shape)
    return None if mask is None else mask.reshape(-1)


def _mask_is_full(mask: torch.Tensor | None) -> bool:
    """Whether ``mask`` selects everything: ``None`` or ``True`` for a one-element signal.

    A one-element mask is decided structurally under graph capture (a single ``False`` would mask
    the whole signal, which is never meaningful), so exported graphs carry no data-dependent gather.
    """
    if mask is None:
        return True
    if mask.numel() != 1:
        return False
    return True if is_exporting() else bool(mask.reshape(()))


def _normalize_signal(data: torch.Tensor, num_bins: int, eps: float = 1e-8) -> torch.Tensor:
    min_val, _ = data.min(dim=-1)
    max_val, _ = data.max(dim=-1)
    diff = (max_val - min_val).unsqueeze(-1)
    # Map the range onto the bin centres 0 ... num_bins - 1, so that the maximum lands on the last centre and gets
    # full histogram weight like every other sample. The signal is considered trivial if too low variation; its range
    # is replaced by one inside the division, so that the discarded branch does not backpropagate 0 / 0.
    nontrivial = diff > eps
    safe_diff = torch.where(nontrivial, diff, torch.ones_like(diff))
    return torch.where(nontrivial, (data - min_val.unsqueeze(-1)) / safe_diff * (num_bins - 1), 0)


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
        mask: torch.Tensor | None = None,
        kernel_function: MIKernel = MIKernel.xu,
        num_bins: int = 64,
        window_radius: float = 1.0,
    ) -> None:
        """Initialize the entropy-based loss base module.

        Args:
            reference_signal (torch.Tensor): reference signal to which
                other signals will be compared by the forward method
            mask (torch.Tensor | None): mask of roi in reference_signal, by default None
                boolean with shape (N,), equal to reference_signal.shape[-1:]. It is common to all the samples
                in reference_signal.
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
        KORNIA_CHECK(
            isinstance(num_bins, int) and not isinstance(num_bins, bool) and num_bins >= 2,
            "num_bins must be an integer >= 2.",
        )
        KORNIA_CHECK(window_radius > 0, "window_radius must be > 0.")
        if not isinstance(kernel_function, MIKernel):
            raise ValueError(f"kernel_function must be a MIKernel member, the available options are {list(MIKernel)}.")
        _validate_mask(mask, reference_signal.shape[-1:])
        self._ref_mask_is_full = _mask_is_full(mask)
        mask = self.fix_mask(mask, reference_signal)
        eps = torch.finfo(reference_signal.dtype).eps
        self.initial_shape = reference_signal.shape
        signal = reference_signal[..., mask]
        # Without a mask the gather keeps every sample: checking its size would guard on a data-dependent size
        # and break ``fullgraph`` capture.
        KORNIA_CHECK(self._ref_mask_is_full or signal.shape[-1] > 0, "mask must select at least one sample.")
        self.register_buffer("signal", _normalize_signal(signal, num_bins, eps))
        # Keep the mask consistent with the cached reference if the caller edits its tensor.
        self.register_buffer("mask", mask.clone())
        self.num_bins = num_bins
        self.kernel_function = partial(kernel_function.value, window_radius=window_radius)
        self.window_radius = window_radius
        # A non-persistent buffer follows ``Module.to(device)`` and keeps the ``state_dict`` keys as they are.
        self.register_buffer("bin_centers", torch.arange(self.num_bins, device=self.signal.device), persistent=False)

    @property
    def eps(self) -> float:
        """Machine epsilon of the cached reference signal, so that it follows ``Module.to(dtype)``."""
        return torch.finfo(self.signal.dtype).eps

    @staticmethod
    def fix_mask(mask: torch.Tensor, masked_guy: torch.Tensor) -> torch.Tensor:
        """Convert mask to correct full mask if it is None.

        Args:
            mask (torch.Tensor | None): input mask
            masked_guy (torch.Tensor): the tensor to be masked by mask

        Returns:
            torch.Tensor: normalized mask
        """
        _validate_mask(mask, masked_guy.shape[-1:])
        if mask is None:
            return torch.ones(masked_guy.shape[-1], dtype=torch.bool, device=masked_guy.device)
        return mask.to(masked_guy.device, torch.bool)

    # TODO: optimize method below, maybe with ihdex coordinates conversion
    def trace_in_ref_mask(self, other_signal, other_mask):
        """Align another masked signal with the reference mask positions.

        The reference signal can be stored with a mask, while the compared
        signal may use a different mask over the same flattened coordinate
        space. This helper restores the compared signal to the original flat
        shape when needed, then selects the elements that correspond to the
        stored reference mask.

        Args:
            other_signal: Flattened signal values after applying ``other_mask``.
            other_mask: Boolean mask describing which flattened positions are
                present in ``other_signal``.

        Returns:
            Signal values traced onto the reference mask used by this loss
            instance.
        """
        if other_mask.all():
            return other_signal[..., self.mask]

        intermediate = torch.zeros(self.initial_shape).to(other_signal)
        intermediate[..., other_mask] = other_signal
        return intermediate[..., self.mask]

    def _compute_joint_histogram(
        self, other_signal: torch.Tensor, eps: float, other_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the joint histogram between the reference signal and another signal.

        Uses kernel density estimation to estimate the joint probability distribution
        between the reference signal and the provided other signal. The histogram is
        computed by evaluating kernel functions at discretized signal values.

        Args:
            other_signal (torch.Tensor): Signal to compare with the reference signal.
                Must have the same shape as the reference signal.
            eps (float): Epsilon value for numerical stability in computations.
            other_mask (torch.Tensor): common mask of roi for all the samples in other_signal, defaults to None

        Returns:
            torch.Tensor: Joint histogram tensor with shape [..., num_bins, num_bins]
                representing the estimated joint probability distribution of the passed signals in the intersection
                of rois.

        Raises:
            ValueError: If other_signal has incompatible shape with reference signal.
        """
        if other_signal.shape != self.initial_shape:
            raise ValueError(
                f"The two signals have incompatible shapes: {other_signal.shape} and {self.initial_shape}."
            )
        _validate_mask(other_mask, other_signal.shape[-1:])
        if self._ref_mask_is_full and _mask_is_full(other_mask):
            # No roi on either side: skip the boolean-mask gathers, whose output sizes depend on the
            # mask values and therefore cannot be captured by ``torch.export``/ONNX.
            ref_signal = self.signal
            other_signal = _normalize_signal(other_signal, num_bins=self.num_bins, eps=eps)
        else:
            # normalize in restriction to mask and recast in self.signal coords
            other_mask = self.fix_mask(other_mask, other_signal)
            common_mask = other_mask[self.mask]
            ref_signal = self.signal[..., common_mask]
            KORNIA_CHECK(ref_signal.shape[-1] > 0, "mask intersection must select at least one sample.")
            other_signal = other_signal[..., other_mask]
            other_signal = _normalize_signal(other_signal, num_bins=self.num_bins, eps=eps)
            other_signal = self.trace_in_ref_mask(other_signal, other_mask)
            other_signal = other_signal[..., common_mask]

        diff_1 = self.bin_centers.unsqueeze(-1) - ref_signal.unsqueeze(-2)
        diff_2 = self.bin_centers.unsqueeze(-1) - other_signal.unsqueeze(-2)

        vals_1 = self.kernel_function(diff_1)
        vals_2 = self.kernel_function(diff_2)

        return torch.einsum("...in,...jn->...ij", vals_1, vals_2)

    def entropies(
        self, other_signal: torch.Tensor, other_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute entropy measures between the reference signal and another signal.

        Calculates joint entropy and marginal entropies based on the joint histogram of the two signals.

        Args:
            other_signal (torch.Tensor): Signal to compare with the reference signal.
                Must have the same shape as the reference_signal passed at instantiation.
            other_mask (torch.Tensor): common mask of roi for all the samples in other_signal, defaults to None

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the entropies of the joint
            histogram's marginals and of the histogram itself, in this order:
                - Marginal entropy of other_signal
                - Marginal entropy of the reference signal
                - Joint entropy

            All tensors have the same batch dimensions as the input signals.

        Note:
            Subclasses should implement specific loss functions based on these entropy
            measures (e.g., mutual information).
        """
        joint_histogram = self._compute_joint_histogram(other_signal, self.eps, other_mask)
        return _joint_histogram_to_entropies(joint_histogram, eps=self.eps)


class MILossFromRef(EntropyBasedLossBase):
    """Mutual-information loss against a stored reference signal.

    The module form of :func:`~kornia.losses.mutual_information_loss`, built on the target:
    ``MILossFromRef(target, target_mask)(input, input_mask)`` equals
    ``mutual_information_loss(input, target, input_mask, target_mask)``.

    Convention:
        - See the Convention block of :func:`~kornia.losses.mutual_information_loss`. ``reference_signal`` is
          ``(*, N)`` with a boolean ``(N,)`` ``mask``, and the module is called with an ``other_signal`` of the same
          shape and its own ``(N,)`` mask.
        - At construction the reference is restricted to its mask, min-max normalised and stored as the buffer
          ``signal``, next to the buffer ``mask``; later in-place changes to the reference tensor are not seen. The
          cache is not detached: a reference that requires grad receives a gradient from the first backward pass,
          and a second backward pass through the same module raises.
        - Known defect: a boolean ``mask`` on the reference's device is stored as the buffer ``mask`` itself, not a
          copy, while ``signal`` was restricted with it at construction: editing that mask in place afterwards
          silently changes the loss, or raises when the number of selected positions changes
          (`#5630 <https://github.com/kornia/kornia/issues/5630>`_).
    """

    def forward(self, other_signal: torch.Tensor, other_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the mutual-information loss of other_signal against the stored reference.

        mi = (H(X) + H(Y) - H(X,Y))
        To have a loss function, the opposite is returned.
        Can also handle two batches of flat tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of flat tensors shape (B,N) where B is
                the tuple of batch dimensions for self.signal, possibly empty.
            other_mask (torch.Tensor): common boolean (N,) mask of roi for all the samples in other_signal, defaults
                to None

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        H_x, H_y, H_xy = self.entropies(other_signal, other_mask)
        mi = H_x + H_y - H_xy

        return -mi


class NMILossFromRef(EntropyBasedLossBase):
    """Normalized mutual-information loss against a stored reference signal.

    This variant divides the sum of marginal entropies by joint entropy before
    negating the value. It is the module form of :func:`~kornia.losses.normalized_mutual_information_loss`,
    built on the target.

    Convention:
        See the Convention blocks of :func:`~kornia.losses.normalized_mutual_information_loss` for the loss and of
        :class:`~kornia.losses.MILossFromRef` for the stored reference.
    """

    def forward(self, other_signal: torch.Tensor, other_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the normalized mutual-information loss of other_signal against the stored reference.

        nmi = (H(X) + H(Y)) / H(X,Y)
        To have a loss function, the opposite is returned.
        Can also handle two batches of flat tensors, then a batch of loss values is returned.

        Args:
            other_signal: Batch of flat tensors shape (B,N) where B is
                the tuple of batch dimensions for self.signal, possibly empty.
            other_mask (torch.Tensor): common boolean (N,) mask of roi for all the samples in other_signal, defaults
                to None

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        H_x, H_y, H_xy = self.entropies(other_signal, other_mask)
        nmi = (H_x + H_y) / H_xy

        return -nmi


class MILossFromRef2D(MILossFromRef):
    """Mutual Information loss module specifically designed for 2D image data.

    This class extends MILossFromRef to handle 2D image inputs by automatically
    reshaping batched 2D images into the format expected by the base entropy
    computation methods. It computes mutual information between reference 2D images
    and other images using kernel density estimation.

    Convention:
        See the Convention block of :class:`~kornia.losses.MILossFromRef`. This is the module form of
        :func:`~kornia.losses.mutual_information_loss_2d`, built on the target: the reference and ``other_signal`` are
        ``(*, H, W)`` images, flattened as there, and both masks are ``(H, W)``.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        mask: torch.Tensor | None = None,
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
            mask (torch.Tensor | None): common boolean (H, W) mask of roi for all the samples in reference_signal,
                by default None
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(
            self.arrange_shape(reference_signal),
            _flatten_mask(mask, reference_signal.shape[-2:]),
            kernel_function,
            num_bins,
            window_radius,
        )
        self._reference_shape = reference_signal.shape

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        """Flatten the spatial dimensions of a 2D signal batch.

        Args:
            tensor: Signal tensor with shape :math:`(*, H, W)`, where ``*`` is
                any leading batch shape.

        Returns:
            Tensor with shape :math:`(*, H * W)` suitable for histogram-based
            mutual-information computation.
        """
        return tensor.reshape(tensor.shape[:-2] + (-1,))

    def forward(
        self,
        other_signal: torch.Tensor,
        other_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the mutual-information loss of other_signal against the stored reference (both 2D).

        mi = (H(X) + H(Y) - H(X,Y))
        To have a loss function, the opposite is returned.
        Can also handle two batches of 2D images, then a batch of loss values is returned.

        Args:
            other_signal: Batch of 2D images, same shape (B,H,W) as
                reference_signal passed for instantiation
            other_mask (torch.Tensor): common boolean (H, W) mask of roi for all the samples in other_signal,
                defaults to None

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        KORNIA_CHECK(
            other_signal.shape == self._reference_shape,
            f"input and reference must have the same shape. Got {other_signal.shape} and {self._reference_shape}.",
        )
        return super().forward(self.arrange_shape(other_signal), _flatten_mask(other_mask, other_signal.shape[-2:]))


class MILossFromRef3D(MILossFromRef):
    """Mutual Information loss module specifically designed for 3D image data.

    This class extends MILossFromRef to handle 3D image inputs by automatically
    reshaping batched 3D images into the format expected by the base entropy
    computation methods. It computes mutual information between reference 3D images
    and other images using kernel density estimation.

    Convention:
        See the Convention block of :class:`~kornia.losses.MILossFromRef`. This is the module form of
        :func:`~kornia.losses.mutual_information_loss_3d`, built on the target: the reference and ``other_signal`` are
        ``(*, D, H, W)`` volumes, flattened as there, and both masks are ``(D, H, W)``.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        mask: torch.Tensor | None = None,
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
            mask (torch.Tensor | None): common boolean (D, H, W) mask of roi for all the samples in
                reference_signal, by default None
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(
            self.arrange_shape(reference_signal),
            _flatten_mask(mask, reference_signal.shape[-3:]),
            kernel_function,
            num_bins,
            window_radius,
        )
        self._reference_shape = reference_signal.shape

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        """Flatten the spatial dimensions of a 3D signal batch.

        Args:
            tensor: Signal tensor with shape :math:`(*, D, H, W)`, where
                :math:`D` is depth, :math:`H` is height, and :math:`W` is
                width.

        Returns:
            Tensor with shape :math:`(*, D * H * W)` for entropy estimation.
        """
        return tensor.reshape(tensor.shape[:-3] + (-1,))

    def forward(
        self,
        other_signal: torch.Tensor,
        other_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the mutual-information loss of other_signal against the stored reference (both 3D).

        mi = (H(X) + H(Y) - H(X,Y))
        To have a loss function, the opposite is returned.
        Can also handle two batches of 3D volumes, then a batch of loss values is returned.

        Args:
            other_signal: Batch of 3D volumes, same shape (B,D,H,W) as
                reference_signal passed for instantiation
            other_mask (torch.Tensor): common boolean (D, H, W) mask of roi for all the samples in other_signal,
                defaults to None

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        KORNIA_CHECK(
            other_signal.shape == self._reference_shape,
            f"input and reference must have the same shape. Got {other_signal.shape} and {self._reference_shape}.",
        )
        return super().forward(self.arrange_shape(other_signal), _flatten_mask(other_mask, other_signal.shape[-3:]))


class NMILossFromRef2D(NMILossFromRef):
    """Normalized Mutual Information loss module specifically designed for 2D image data.

    This class extends NMILossFromRef to handle 2D image inputs by automatically
    reshaping batched 2D images into the format expected by the base entropy
    computation methods. It computes normalized mutual information between reference 2D images
    and other images using kernel density estimation.

    Convention:
        See the Convention block of :class:`~kornia.losses.NMILossFromRef`. This is the module form of
        :func:`~kornia.losses.normalized_mutual_information_loss_2d`, built on the target: the reference and
        ``other_signal`` are ``(*, H, W)`` images, flattened as there, and both masks are ``(H, W)``.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        mask: torch.Tensor | None = None,
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
            mask (torch.Tensor | None): common boolean (H, W) mask of roi for all the samples in reference_signal,
                by default None
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(
            self.arrange_shape(reference_signal),
            _flatten_mask(mask, reference_signal.shape[-2:]),
            kernel_function,
            num_bins,
            window_radius,
        )
        self._reference_shape = reference_signal.shape

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        """Flatten the spatial dimensions of a 2D signal batch.

        Args:
            tensor: Signal tensor with shape :math:`(*, H, W)`.

        Returns:
            Tensor with shape :math:`(*, H * W)` used by the normalized
            mutual-information base implementation.
        """
        return tensor.reshape(tensor.shape[:-2] + (-1,))

    def forward(
        self,
        other_signal: torch.Tensor,
        other_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the normalized mutual-information loss of other_signal against the stored reference (both 2D).

        nmi = (H(X) + H(Y)) / H(X,Y)
        To have a loss function, the opposite is returned.
        Can also handle two batches of 2D images, then a batch of loss values is returned.

        Args:
            other_signal: Batch of 2D images, same shape (B,H,W) as
                reference_signal passed for instantiation
            other_mask (torch.Tensor): common boolean (H, W) mask of roi for all the samples in other_signal,
                defaults to None

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        KORNIA_CHECK(
            other_signal.shape == self._reference_shape,
            f"input and reference must have the same shape. Got {other_signal.shape} and {self._reference_shape}.",
        )
        return super().forward(self.arrange_shape(other_signal), _flatten_mask(other_mask, other_signal.shape[-2:]))


class NMILossFromRef3D(NMILossFromRef):
    """Normalized Mutual Information loss module specifically designed for 3D image data.

    This class extends NMILossFromRef to handle 3D image inputs by automatically
    reshaping batched 3D images into the format expected by the base entropy
    computation methods. It computes normalized mutual information between reference 3D images
    and other images using kernel density estimation.

    Convention:
        See the Convention block of :class:`~kornia.losses.NMILossFromRef`. This is the module form of
        :func:`~kornia.losses.normalized_mutual_information_loss_3d`, built on the target: the reference and
        ``other_signal`` are ``(*, D, H, W)`` volumes, flattened as there, and both masks are ``(D, H, W)``.
    """

    def __init__(
        self,
        reference_signal: torch.Tensor,
        mask: torch.Tensor | None = None,
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
            mask (torch.Tensor | None): common boolean (D, H, W) mask of roi for all the samples in
                reference_signal, by default None
            kernel_function (MIKernel): Used kernel function for kernel
                density estimate, by default MIKernel.xu
            num_bins (int): number of signal value bins in kernel
                density estimate, by default 64
            window_radius (float): radius of the kernel's support
                interval, by default 1.0
        """
        super().__init__(
            self.arrange_shape(reference_signal),
            _flatten_mask(mask, reference_signal.shape[-3:]),
            kernel_function,
            num_bins,
            window_radius,
        )
        self._reference_shape = reference_signal.shape

    @staticmethod
    def arrange_shape(tensor: torch.Tensor) -> torch.Tensor:
        """Flatten the spatial dimensions of a 3D signal batch.

        Args:
            tensor: Signal tensor with shape :math:`(*, D, H, W)`.

        Returns:
            Tensor with shape :math:`(*, D * H * W)` used by the normalized
            mutual-information base implementation.
        """
        return tensor.reshape(tensor.shape[:-3] + (-1,))

    def forward(
        self,
        other_signal: torch.Tensor,
        other_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the normalized mutual-information loss of other_signal against the stored reference (both 3D).

        nmi = (H(X) + H(Y)) / H(X,Y)
        To have a loss function, the opposite is returned.
        Can also handle two batches of 3D volumes, then a batch of loss values is returned.

        Args:
            other_signal: Batch of 3D volumes, same shape (B,D,H,W) as
                reference_signal passed for instantiation
            other_mask (torch.Tensor): common boolean (D, H, W) mask of roi for all the samples in other_signal,
                defaults to None

        Returns:
            torch.Tensor: tensor of losses, shape B as above
        """
        KORNIA_CHECK(
            other_signal.shape == self._reference_shape,
            f"input and reference must have the same shape. Got {other_signal.shape} and {self._reference_shape}.",
        )
        return super().forward(self.arrange_shape(other_signal), _flatten_mask(other_mask, other_signal.shape[-3:]))


def mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    kernel_function: MIKernel = MIKernel.xu,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute the mutual-information loss of flat tensors.

    Convention:
        - The loss is the negative mutual information, :math:`-(H(X) + H(Y) - H(X, Y))`, in nats: minimising it
          maximises the mutual information. It is symmetric in ``input`` and ``target``.
          :ref:`Registration <mutual-information-porting>` maps the losses onto scikit-learn and scikit-image.
        - ``input`` and ``target`` are ``(*, N)`` tensors of one shape and one floating dtype; a single ``target`` is
          not broadcast over a batch of inputs. There is no ``reduction``: the result holds one loss per leading
          index, shape ``*``, in the input dtype (:ref:`Losses and metrics <losses-metrics-conventions>` lists the
          defaults of the other losses).
        - Each signal is min-max normalised per sample, over its mask, so no data range is assumed. A positive
          rescaling and a shift of one sample leave its loss unchanged up to roundoff when both its original and
          transformed ranges exceed the absolute threshold ``torch.finfo(dtype).eps``. One outlier squeezes the
          other values of its sample into a few bins. A sample whose range is at most that threshold counts as
          constant: its MI is 0 and it gets a zero gradient.
        - The joint histogram has ``num_bins`` bins per signal, and each value spreads over the bins within
          ``window_radius`` bin widths of it, weighted by ``kernel_function``, a :class:`~kornia.losses.MIKernel`
          member. With continuous values and the default ``window_radius``, this soft histogram scores an image
          against itself below its entropy. The estimate depends on ``num_bins`` and ``window_radius``, in no fixed
          direction, so values compare only at equal ``num_bins``, ``window_radius`` and kernel.
        - With ``MIKernel.xu`` (the default) and ``MIKernel.truncated_gaussian``, ``input`` and ``target`` both get
          gradients. ``MIKernel.rectangular`` is piecewise constant: its loss has no gradient and serves for
          evaluation only.
        - ``input_mask`` and ``target_mask`` are boolean ``(N,)`` masks, common to the batch. The histogram counts the
          positions inside both masks, while each signal is normalised over its own mask: a value inside one mask
          only is not counted, but it changes the loss when it is that signal's minimum or maximum. Values outside
          both masks have no effect. Masks of another shape, or with no position in common, raise an error.
        - Known defect: in float16 and bfloat16 the empty bins are floored at ``torch.finfo(dtype).eps`` counts
          before the histogram is normalised: both dtypes bias the loss of small images, and float16 returns NaN for
          large ones, from about ``2**15 / window_radius**2`` pixels with the default ``MIKernel.xu``, from a
          quarter of that with ``MIKernel.rectangular`` and from about ``2**16`` pixels, whatever the radius, with
          ``MIKernel.truncated_gaussian`` (`#4153 <https://github.com/kornia/kornia/issues/4153>`_).

    Args:
        input (torch.Tensor): Batch of flat tensors shape (B,N) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of flat tensors, same shape as
            input.
        input_mask (torch.Tensor): boolean roi mask of shape (N,), common to the batch. Defaults to None.
        target_mask (torch.Tensor): boolean roi mask of shape (N,), common to the batch. Defaults to None.
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    KORNIA_CHECK_SAME_SHAPE(input, target)
    module = MILossFromRef(
        reference_signal=target,
        mask=target_mask,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )
    return module.forward(input, other_mask=input_mask)


def mutual_information_loss_2d(
    input: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    kernel_function: MIKernel = MIKernel.xu,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute the mutual-information loss of 2d tensors.

    mi = (H(X) + H(Y) - H(X,Y))
    To have a loss function, the opposite is returned.
    Can also handle two batches of 2d tensors, then a batch of loss values is returned.

    Convention:
        See the Convention block of :func:`~kornia.losses.mutual_information_loss`; this function applies it to
        ``input.flatten(-2)`` and ``target.flatten(-2)``. Every axis before the last two is a batch axis, so a
        ``(B, C, H, W)`` input gives one loss per image and channel, ``(B, C)``. The masks are ``(H, W)``.

    Args:
        input (torch.Tensor): Batch of 2d tensors shape (B,H,W) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 2d tensors, same shape as input.
        input_mask (torch.Tensor): boolean roi mask of shape (H, W), common to the batch. Defaults to None.
        target_mask (torch.Tensor): boolean roi mask of shape (H, W), common to the batch. Defaults to None.
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    KORNIA_CHECK_SAME_SHAPE(input, target)
    module = MILossFromRef2D(
        reference_signal=target,
        mask=target_mask,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )
    return module.forward(input, other_mask=input_mask)


def mutual_information_loss_3d(
    input: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    kernel_function: MIKernel = MIKernel.xu,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute the mutual-information loss of 3d tensors.

    mi = (H(X) + H(Y) - H(X,Y))
    To have a loss function, the opposite is returned.
    Can also handle two batches of 3d tensors, then a batch of loss values is returned.

    Convention:
        See the Convention block of :func:`~kornia.losses.mutual_information_loss`; this function applies it to
        ``input.flatten(-3)`` and ``target.flatten(-3)``, with ``(D, H, W)`` masks. Every axis before the last three
        is a batch axis, so a ``(B, C, H, W)`` tensor is read as ``B`` volumes of depth ``C`` and gives ``(B,)``.

    Args:
        input (torch.Tensor): Batch of 3d tensors shape (B,D,H,W) where
            B is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 3d tensors, same shape as input.
        input_mask (torch.Tensor): boolean roi mask of shape (D, H, W), common to the batch. Defaults to None.
        target_mask (torch.Tensor): boolean roi mask of shape (D, H, W), common to the batch. Defaults to None.
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    KORNIA_CHECK_SAME_SHAPE(input, target)
    module = MILossFromRef3D(
        reference_signal=target,
        mask=target_mask,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )
    return module.forward(input, other_mask=input_mask)


def normalized_mutual_information_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    kernel_function: MIKernel = MIKernel.xu,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute the normalized mutual-information loss of flat tensors.

    Convention:
        - The loss is the negative of Studholme's normalized mutual information,
          :math:`(H(X) + H(Y)) / H(X, Y)`, which lies in :math:`[1, 2]`: the loss lies in ``[-2, -1]`` and is ``-1``
          when one signal is constant and the other is not; for two constant signals its value is undefined. With
          continuous values and the default ``window_radius``, even an image against itself scores above ``-2``.
          :ref:`Registration <mutual-information-porting>` maps this normalisation onto scikit-image and
          scikit-learn.
        - Everything else, the known defect included, is as in the Convention block of
          :func:`~kornia.losses.mutual_information_loss`.

    Args:
        input (torch.Tensor): Batch of flat tensors shape (B,N) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of flat tensors, same shape as
            input.
        input_mask (torch.Tensor): boolean roi mask of shape (N,), common to the batch. Defaults to None.
        target_mask (torch.Tensor): boolean roi mask of shape (N,), common to the batch. Defaults to None.
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    KORNIA_CHECK_SAME_SHAPE(input, target)
    module = NMILossFromRef(
        reference_signal=target,
        kernel_function=kernel_function,
        mask=target_mask,
        num_bins=num_bins,
        window_radius=window_radius,
    )
    return module.forward(input, other_mask=input_mask)


def normalized_mutual_information_loss_2d(
    input: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    kernel_function: MIKernel = MIKernel.xu,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute the normalized mutual-information loss of 2d tensors.

    nmi = (H(X) + H(Y)) / H(X,Y)
    To have a loss function, the opposite is returned.
    Can also handle two batches of 2d tensors, then a batch of loss values is returned.

    Convention:
        See the Convention block of :func:`~kornia.losses.normalized_mutual_information_loss`; the flattening, the
        batch axes and the ``(H, W)`` masks are those of :func:`~kornia.losses.mutual_information_loss_2d`.

    Args:
        input (torch.Tensor): Batch of 2d tensors shape (B,H,W) where B
            is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 2d tensors, same shape as input.
        input_mask (torch.Tensor): boolean roi mask of shape (H, W), common to the batch. Defaults to None.
        target_mask (torch.Tensor): boolean roi mask of shape (H, W), common to the batch. Defaults to None.
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    KORNIA_CHECK_SAME_SHAPE(input, target)
    module = NMILossFromRef2D(
        reference_signal=target,
        mask=target_mask,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )
    return module.forward(input, other_mask=input_mask)


def normalized_mutual_information_loss_3d(
    input: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    kernel_function: MIKernel = MIKernel.xu,
    num_bins: int = 64,
    window_radius: float = 1.0,
) -> torch.Tensor:
    """Compute the normalized mutual-information loss of 3d tensors.

    nmi = (H(X) + H(Y)) / H(X,Y)
    To have a loss function, the opposite is returned.
    Can also handle two batches of 3d tensors, then a batch of loss values is returned.

    Convention:
        See the Convention block of :func:`~kornia.losses.normalized_mutual_information_loss`; the flattening, the
        batch axes and the ``(D, H, W)`` masks are those of :func:`~kornia.losses.mutual_information_loss_3d`.

    Args:
        input (torch.Tensor): Batch of 3d tensors shape (B,D,H,W) where
            B is any batch dimensions tuple, possibly empty.
        target (torch.Tensor): Batch of 3d tensors, same shape as input.
        input_mask (torch.Tensor): boolean roi mask of shape (D, H, W), common to the batch. Defaults to None.
        target_mask (torch.Tensor): boolean roi mask of shape (D, H, W), common to the batch. Defaults to None.
        kernel_function (MIKernel): Used kernel function for kernel
            density estimate, by default MIKernel.xu
        num_bins (int): The number of bins used for KDE, defaults to 64.
        window_radius (float): The smoothing window radius in KDE, in
            terms of bin width units, defaults to 1.

    Returns:
        torch.Tensor: tensor of losses, shape B (common batch dims tuple
        of input and target)
    """
    KORNIA_CHECK_SAME_SHAPE(input, target)
    module = NMILossFromRef3D(
        reference_signal=target,
        mask=target_mask,
        kernel_function=kernel_function,
        num_bins=num_bins,
        window_radius=window_radius,
    )
    return module.forward(input, other_mask=input_mask)
