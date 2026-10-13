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


def histogram_matching(source: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Adjust the pixel values of an image to match its histogram towards a target image.

    `Histogram matching <https://en.wikipedia.org/wiki/Histogram_matching>`_ is the transformation
    of an image so that its histogram matches a specified histogram. In this implementation, the
    histogram is computed over the flattened image array. Code referred to
    `here <https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x>`_.

    Args:
        source: Image to transform.
        template: Template image. It can have different dimensions to source.

    Returns:
        The transformed output image as the same shape as the source image. A floating source keeps its dtype; an
        integer source returns ``float32`` on its own value scale. The quantiles are computed in ``float32``
        (``float64`` for a ``float64`` source), so ``uint8``, ``float16`` and ``bfloat16`` images of any size work.

    Note:
        This function does not matches histograms element-wisely if input a batched tensor.

    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts.
    _, bin_idx, s_counts = torch.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = torch.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile). The counts are summed in int64 and the quantiles computed
    # in float32 (float64 for float64 input): a uint8 sum wraps past 255, and float16/bfloat16 sums stop being exact
    # past 2048/256, which repeats quantiles and makes interp divide by zero.
    compute_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    s_quantiles = torch.cumsum(s_counts, dim=0).to(compute_dtype)
    s_quantiles = s_quantiles / s_quantiles[-1]
    t_quantiles = torch.cumsum(t_counts, dim=0).to(compute_dtype)
    t_quantiles = t_quantiles / t_quantiles[-1]
    t_values = t_values.to(compute_dtype)

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image.
    # Anchor the template CDF at quantile 0 so a source quantile below the template's lowest one maps to the
    # template minimum (as np.interp does) instead of being extrapolated below it.
    t_quantiles = torch.cat([t_quantiles.new_zeros(1), t_quantiles])
    t_values = torch.cat([t_values[:1], t_values])
    interp_t_values = interp(s_quantiles, t_quantiles, t_values)

    # A floating source keeps its dtype; an integer source returns float32 on its own scale.
    out_dtype = source.dtype if source.is_floating_point() else compute_dtype
    return interp_t_values[bin_idx].reshape(oldshape).to(out_dtype)


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    This is confirmed to be a correct implementation.
    See https://github.com/pytorch/pytorch/issues/1552#issuecomment-979998307

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.

    """
    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)

    # A zero-width segment (repeated xp) is only selected at the clipped upper end, x >= xp[-1]. Dividing by its width
    # gave NaN, e.g. when float32 rounds the last cumulative counts of a 2**24-pixel template to the same quantile.
    # Divide by a safe width there and return the right endpoint, as np.interp does for x >= xp[-1].
    dx = xp[i] - xp[i - 1]
    zero_width = dx == 0
    out = (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / torch.where(zero_width, torch.ones_like(dx), dx)
    return torch.where(zero_width, fp[i], out)
