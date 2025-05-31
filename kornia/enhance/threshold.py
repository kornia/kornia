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

from typing import Union

import torch

from kornia.core import ImageModule as Module
from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR


def _differentiable_binary_thresh(
    src: Tensor, thresh: Union[float, Tensor], beta: Union[float, Tensor], inv: bool = False
):
    """Perform differentiable binary thresholding.

    Args:
        src: Image tensor to threshold.
        thresh: Threshold value.
        beta: Sharpness, where larger beta approaches hard threshold.
        inv: If true, threshold when src <= thresh.

    Returns:
        Thresholded tensor of same shape/dtype/device as src.
    """
    KORNIA_CHECK_IS_TENSOR(src)

    if not isinstance(thresh, Tensor):
        thresh = tensor(thresh, dtype=src.dtype, device=src.device)
    if not isinstance(beta, Tensor):
        beta = tensor(beta, dtype=src.dtype, device=src.device)

    if inv:
        x = thresh - src
    else:
        x = src - thresh

    return torch.sigmoid(beta * x)


def thresh_binary(
    src: Tensor, thresh: Union[float, Tensor], maxval: Union[float, Tensor], beta: Union[float, Tensor] = 50.0
):
    r"""Apply differentiable binary thresholding to image.

    .. image:: _static/img/thresh_binary.png

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        src: Image tensor to threshold.
        thresh: Threshold value.
        maxval: Value to set pixel if value > thresh.
        beta: Sharpness, where larger beta approaches hard threshold.

    Returns:
        Thresholded tensor of same shape/dtype/device as src.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = thresh_binary(x, thresh=0.5, maxval=255.)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/maxval/beta have to be with shape broadcastable to src shape.
    """
    if not isinstance(maxval, Tensor):
        maxval = tensor(maxval, dtype=src.dtype, device=src.device)

    return maxval * _differentiable_binary_thresh(src, thresh, beta)


def thresh_binary_inv(
    src: Tensor, thresh: Union[float, Tensor], maxval: Union[float, Tensor], beta: Union[float, Tensor] = 50.0
):
    r"""Apply differentiable inverse binary thresholding to image.

    .. image:: _static/img/thresh_binary_inv.png

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        src: Image tensor to threshold.
        thresh: Threshold value.
        maxval: Value to set pixel if value <= thresh.
        beta: Sharpness, where larger beta approaches hard threshold.

    Returns:
        Thresholded tensor of same shape/dtype/device as src.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = thresh_binary_inv(x, thresh=0.5, maxval=255.)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/maxval/beta have to be with shape broadcastable to src shape.
    """
    if not isinstance(maxval, Tensor):
        maxval = tensor(maxval, dtype=src.dtype, device=src.device)

    return maxval * _differentiable_binary_thresh(src, thresh, beta, inv=True)


def thresh_trunc(src: Tensor, thresh: Union[float, Tensor], beta: Union[float, Tensor] = 50.0):
    r"""Apply differentiable truncate thresholding to image.

    .. image:: _static/img/thresh_trunc.png

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        src: Image tensor to threshold.
        thresh: Threshold value.
        beta: Sharpness, where larger beta approaches hard threshold.

    Returns:
        Thresholded tensor of same shape/dtype/device as src.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = thresh_trunc(x, thresh=0.5)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/beta have to be with shape broadcastable to src shape.
    """
    return src - torch.nn.functional.softplus(src - thresh, beta)


def thresh_tozero(src: Tensor, thresh: Union[float, Tensor], beta: Union[float, Tensor] = 50.0):
    r"""Apply differentiable to-zero thresholding to image.

    .. image:: _static/img/thresh_tozero.png

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        src: Image tensor to threshold.
        thresh: Threshold value.
        beta: Sharpness, where larger beta approaches hard threshold.

    Returns:
        Thresholded tensor of same shape/dtype/device as src.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = thresh_tozero(x, thresh=0.5)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/beta have to be with shape broadcastable to src shape.
    """
    return src * _differentiable_binary_thresh(src, thresh, beta)


def thresh_tozero_inv(src: Tensor, thresh: Union[float, Tensor], beta: Union[float, Tensor] = 50.0):
    r"""Apply differentiable inverse to-zero thresholding to image.

    .. image:: _static/img/thresh_tozero_inv.png

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        src: Image tensor to threshold.
        thresh: Threshold value.
        beta: Sharpness, where larger beta approaches hard threshold.

    Returns:
        Thresholded tensor of same shape/dtype/device as src.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = thresh_tozero_inv(x, thresh=0.5)
        >>> torch.unique(out[out != x])
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/beta have to be with shape broadcastable to src shape.
    """
    return src * _differentiable_binary_thresh(src, thresh, beta, inv=True)


class ThreshBinary(Module):
    r"""Apply differentiable binary thresholding to image.

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        thresh: Threshold value.
        maxval: Value to set pixel if value > thresh.
        beta: Sharpness, where larger beta approaches hard threshold.

    Shape:
        - Input: The input tensor to threshold with an arbitrary shape.
        - Output: Thresholded tensor of same shape/dtype/device as Input.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = ThreshBinary(thresh=0.5, maxval=255.)(x)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/maxval/beta have to be with shape broadcastable to Input shape.
    """

    def __init__(
        self, thresh: Union[float, Tensor], maxval: Union[float, Tensor], beta: Union[float, Tensor] = 50.0
    ) -> None:
        super().__init__()
        self.thresh: Union[float, Tensor] = thresh
        self.maxval: Union[float, Tensor] = maxval
        self.beta: Union[float, Tensor] = beta

    def forward(self, input: Tensor) -> Tensor:
        return thresh_binary(input, self.thresh, self.maxval, self.beta)


class ThreshBinaryInv(Module):
    r"""Apply differentiable inverse binary thresholding to image.

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        thresh: Threshold value.
        maxval: Value to set pixel if value <= thresh.
        beta: Sharpness, where larger beta approaches hard threshold.

    Shape:
        - Input: The input tensor to threshold with an arbitrary shape.
        - Output: Thresholded tensor of same shape/dtype/device as Input.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = ThreshBinaryInv(thresh=0.5, maxval=255.)(x)
        >>> torch.unique(out)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/maxval/beta have to be with shape broadcastable to Input shape.
    """

    def __init__(
        self, thresh: Union[float, Tensor], maxval: Union[float, Tensor], beta: Union[float, Tensor] = 50.0
    ) -> None:
        super().__init__()
        self.thresh: Union[float, Tensor] = thresh
        self.maxval: Union[float, Tensor] = maxval
        self.beta: Union[float, Tensor] = beta

    def forward(self, input: Tensor) -> Tensor:
        return thresh_binary_inv(input, self.thresh, self.maxval, self.beta)


class ThreshTrunc(Module):
    r"""Apply differentiable truncate thresholding to image.

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        thresh: Threshold value.
        beta: Sharpness, where larger beta approaches hard threshold.

    Shape:
        - Input: The input tensor to threshold with an arbitrary shape.
        - Output: Thresholded tensor of same shape/dtype/device as Input.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = ThreshTrunc(0.5)(x)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/beta have to be with shape broadcastable to Input shape.
    """

    def __init__(self, thresh: Union[float, Tensor], beta: Union[float, Tensor] = 50.0) -> None:
        super().__init__()
        self.thresh: Union[float, Tensor] = thresh
        self.beta: Union[float, Tensor] = beta

    def forward(self, input: Tensor) -> Tensor:
        return thresh_trunc(input, self.thresh, self.beta)


class ThreshToZero(Module):
    r"""Apply differentiable to-zero thresholding to image.

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        thresh: Threshold value.
        beta: Sharpness, where larger beta approaches hard threshold.

    Shape:
        - Input: The input tensor to threshold with an arbitrary shape.
        - Output: Thresholded tensor of same shape/dtype/device as Input.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = ThreshToZero(0.5)(x)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/beta have to be with shape broadcastable to Input shape.
    """

    def __init__(self, thresh: Union[float, Tensor], beta: Union[float, Tensor] = 50.0) -> None:
        super().__init__()
        self.thresh: Union[float, Tensor] = thresh
        self.beta: Union[float, Tensor] = beta

    def forward(self, input: Tensor) -> Tensor:
        return thresh_tozero(input, self.thresh, self.beta)


class ThreshToZeroInv(Module):
    r"""Apply differentiable inverse to-zero thresholding to image.

    Larger beta approaches hard threshold behavior, described in
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Args:
        thresh: Threshold value.
        beta: Sharpness, where larger beta approaches hard threshold.

    Shape:
        - Input: The input tensor to threshold with an arbitrary shape.
        - Output: Thresholded tensor of same shape/dtype/device as Input.

    Example:
        >>> x = torch.rand(1, 3, 3, 3)
        >>> out = ThreshToZeroInv(0.5)(x)
        >>> out.shape
        torch.Size([1, 3, 3, 3])

    Notes:
        Tensor thresh/beta have to be with shape broadcastable to Input shape.
    """

    def __init__(self, thresh: Union[float, Tensor], beta: Union[float, Tensor] = 50.0) -> None:
        super().__init__()
        self.thresh: Union[float, Tensor] = thresh
        self.beta: Union[float, Tensor] = beta

    def forward(self, input: Tensor) -> Tensor:
        return thresh_tozero_inv(input, self.thresh, self.beta)
