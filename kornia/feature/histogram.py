from typing import Tuple, List

import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

EPSILON = 1e-10


def marginalPdf(values: torch.Tensor, bins: torch.Tensor, sigma: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Args:
        values: (torch.Tensor), shape [BxN]
        bins: (torch.Tensor), shape [NUM_BINS]
        sigma: (torch.Tensor), scalar, gaussian smoothing factor
    Returns:
        pdf: (torch.Tensor), shape [BxN]
        kernel_values: (torch.Tensor), shape [BxNxNUM_BINS]"""

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + EPSILON
    pdf = pdf / normalization

    return (pdf, kernel_values)


def jointPdf(kernel_values1: torch.Tensor, kernel_values2: torch.Tensor) -> torch.Tensor:
    """Args:
        kernel_values1: (torch.Tensor), shape [BxNxNUM_BINS]
        kernel_values2: (torch.Tensor), shape [BxNxNUM_BINS]
    Returns:
        pdf: (torch.Tensor), shape [BxNUM_BINSxNUM_BINS]"""

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + EPSILON
    pdf = joint_kernel_values / normalization

    return pdf


def histogram(x: torch.Tensor, bins: torch.Tensor, bandwidth: float = 0.7) -> torch.Tensor:
    """
    Function that estimates the histogram of the input tensor. The calculation uses kernel
    density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x: (torch.Tensor), shape [BxN]
        bins: (torch.Tensor), shape [NUM_BINS]
        bandwidth: (torch.Tensor), scalar, gaussian smoothing factor
    Returns:
        pdf: (torch.Tensor), shape [BxNUM_BINS]"""

    x = x * 255
    pdf, _ = marginalPdf(x.unsqueeze(2), bins, bandwidth)

    return pdf


def histogram2d(x1: torch.Tensor, x2: torch.Tensor, bins: torch.Tensor, bandwidth: float = 0.7) -> torch.Tensor:
    """
    Function that estimates the histogram of the input tensor. The calculation uses kernel
    density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x1: (torch.Tensor), shape [BxN1]
        x2: (torch.Tensor), shape [BxN2]
        bins: (torch.Tensor), shape [NUM_BINS]
        bandwidth: (torch.Tensor), scalar, gaussian smoothing factor
    Returns:
        pdf: (torch.Tensor), shape [BxNUM_BINSxNUM_BINS]"""

    x1 = x1 * 255
    x2 = x2 * 255

    pdf1, kernel_values1 = marginalPdf(x1.unsqueeze(2), bins, bandwidth)
    pdf2, kernel_values2 = marginalPdf(x2.unsqueeze(2), bins, bandwidth)

    joint_pdf = jointPdf(kernel_values1, kernel_values2)

    return joint_pdf
