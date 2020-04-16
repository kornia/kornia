from typing import Tuple, List

import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

EPSILON = 1e-10


def marginalPdf(values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, epsilon: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Args:
        values: (torch.Tensor), shape [BxN]
        bins: (torch.Tensor), shape [NUM_BINS]
        sigma: (torch.Tensor), scalar, gaussian smoothing factor
    Returns:
        pdf: (torch.Tensor), shape [BxN]
        kernel_values: (torch.Tensor), shape [BxNxNUM_BINS]"""

    if not torch.is_tensor(values):
        raise TypeError("Input values type is not a torch.Tensor. Got {}"
                        .format(type(values)))
    if not torch.is_tensor(bins):
        raise TypeError("Input bins type is not a torch.Tensor. Got {}"
                        .format(type(bins)))
    if not torch.is_tensor(sigma):
        raise TypeError("Input sigma type is not a torch.Tensor. Got {}"
                        .format(type(sigma)))
    print(values.dim(), bins.dim(), sigma.dim())
    exit(0)
    if not values.dim() in (2, 3):
        raise ValueError("Input values must be a of the shape BxN."
                         " Got {}".format(values.shape))
    if not bins.dim() in (2, 3):
        raise ValueError("Input bins must be a of the shape NUM_BINS"
                         " Got {}".format(bins.shape))
    if not sigma.dim() in (2, 3):
        raise ValueError("Input sigma must be a of the shape 1"
                         " Got {}".format(sigma.shape))



    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return (pdf, kernel_values)


def jointPdf(kernel_values1: torch.Tensor, kernel_values2: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Args:
        kernel_values1: (torch.Tensor), shape [BxNxNUM_BINS]
        kernel_values2: (torch.Tensor), shape [BxNxNUM_BINS]
    Returns:
        pdf: (torch.Tensor), shape [BxNUM_BINSxNUM_BINS]"""

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + epsilon
    pdf = joint_kernel_values / normalization

    return pdf


def histogram(x: torch.Tensor, bins: torch.Tensor, bandwidth: float = 0.7, epsilon: float = 1e-10) -> torch.Tensor:
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
    pdf, _ = marginalPdf(x.unsqueeze(2), bins, bandwidth, epsilon)

    return pdf


def histogram2d(x1: torch.Tensor, x2: torch.Tensor, bins: torch.Tensor, bandwidth: float = 0.7, epsilon: float = 1e-10) -> torch.Tensor:
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

    pdf1, kernel_values1 = marginalPdf(x1.unsqueeze(2), bins, bandwidth, epsilon)
    pdf2, kernel_values2 = marginalPdf(x2.unsqueeze(2), bins, bandwidth, epsilon)

    joint_pdf = jointPdf(kernel_values1, kernel_values2)

    return joint_pdf
