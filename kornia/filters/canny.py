from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kornia.color import rgb_to_grayscale
from kornia.core import ImageModule as Module
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .gaussian import gaussian_blur2d
from .kernels import get_canny_nms_kernel, get_hysteresis_kernel
from .sobel import spatial_gradient


def canny(
    input: Tensor,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    kernel_size: tuple[int, int] | int = (5, 5),
    sigma: tuple[float, float] | Tensor = (1, 1),
    hysteresis: bool = True,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    r"""Find edges of the input image and filters them using the Canny algorithm.

    .. image:: _static/img/canny.png

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/canny.html>`__.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = canny(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        low_threshold <= high_threshold,
        "Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: "
        f"{low_threshold}>{high_threshold}",
    )
    KORNIA_CHECK(0 < low_threshold < 1, f"Invalid low threshold. Should be in range (0, 1). Got: {low_threshold}")
    KORNIA_CHECK(0 < high_threshold < 1, f"Invalid high threshold. Should be in range (0, 1). Got: {high_threshold}")

    device = input.device
    dtype = input.dtype

    # To Grayscale
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)

    # Gaussian filter
    blurred: Tensor = gaussian_blur2d(input, kernel_size, sigma)

    # Compute the gradients
    gradients: Tensor = spatial_gradient(blurred, normalized=False)

    # Unpack the edges
    gx: Tensor = gradients[:, :, 0]
    gy: Tensor = gradients[:, :, 1]

    # Compute gradient magnitude and angle
    magnitude: Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    angle: Tensor = torch.atan2(gy, gx)

    # Radians to degrees and round to nearest 45 degree
    # degrees = angle * (180.0 / math.pi)
    # angle = torch.round(degrees / 45) * 45
    angle_45 = (angle * (4 / math.pi)).round()

    # Non-maximal suppression
    nms_kernels: Tensor = get_canny_nms_kernel(device, dtype)
    nms_magnitude: Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

    # Get the indices for both directions
    positive_idx: Tensor = angle_45 % 8
    positive_idx = positive_idx.long()

    negative_idx: Tensor = (angle_45 + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive: Tensor = torch.gather(nms_magnitude, 1, positive_idx)
    channel_select_filtered_negative: Tensor = torch.gather(nms_magnitude, 1, negative_idx)

    channel_select_filtered: Tensor = torch.stack(
        [channel_select_filtered_positive, channel_select_filtered_negative], 1
    )

    is_max: Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

    magnitude = magnitude * is_max

    # Threshold
    edges: Tensor = F.threshold(magnitude, low_threshold, 0.0)

    low: Tensor = magnitude > low_threshold
    high: Tensor = magnitude > high_threshold

    edges = low * 0.5 + high * 0.5
    edges = edges.to(dtype)

    # Hysteresis
    if hysteresis:
        edges_old: Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
        hysteresis_kernels: Tensor = get_hysteresis_kernel(device, dtype)

        while ((edges_old - edges).abs() != 0).any():
            weak: Tensor = (edges == 0.5).float()
            strong: Tensor = (edges == 1).float()

            hysteresis_magnitude: Tensor = F.conv2d(
                edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2
            )
            hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
            hysteresis_magnitude = hysteresis_magnitude * weak + strong

            edges_old = edges.clone()
            edges = hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5

        edges = hysteresis_magnitude

    return magnitude, edges


class Canny(Module):
    r"""Module that finds edges of the input image and filters them using the Canny algorithm.

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """

    # TODO: Handle multiple inputs and outputs models later
    ONNX_EXPORTABLE = False

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: tuple[int, int] | int = (5, 5),
        sigma: tuple[float, float] | Tensor = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        KORNIA_CHECK(
            low_threshold <= high_threshold,
            "Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: "
            f"{low_threshold}>{high_threshold}",
        )
        KORNIA_CHECK(0 < low_threshold < 1, f"Invalid low threshold. Should be in range (0, 1). Got: {low_threshold}")
        KORNIA_CHECK(
            0 < high_threshold < 1, f"Invalid high threshold. Should be in range (0, 1). Got: {high_threshold}"
        )

        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Double threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Hysteresis
        self.hysteresis = hysteresis

        self.eps: float = eps

    def __repr__(self) -> str:
        return "".join(
            (
                f"{type(self).__name__}(",
                ", ".join(
                    f"{name}={getattr(self, name)}" for name in sorted(self.__dict__) if not name.startswith("_")
                ),
                ")",
            )
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        return canny(
            input, self.low_threshold, self.high_threshold, self.kernel_size, self.sigma, self.hysteresis, self.eps
        )
