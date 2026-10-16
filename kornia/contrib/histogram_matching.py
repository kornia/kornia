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
    of an image so that its histogram matches a specified histogram. Code referred to
    `here <https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x>`_.

    Convention:
        - One histogram of every element of ``source``, all batch items and channels together, is matched to one
          histogram of every element of ``template``, which can have any shape. This is
          ``skimage.exposure.match_histograms(source, template, channel_axis=None)``, which needs both inputs to have
          the same number of dimensions (flatten them otherwise); to match each sample or channel on its own, call the
          function on each slice.
        - A source value maps to the template value at its quantile, interpolated linearly between template values
          with :func:`interp`. The result stays inside the template's range once that range is rounded to the output
          dtype, so with a ``float64`` template and a ``float32`` source it can pass the unrounded ends by that
          rounding. It can also fall between template values: a ``uint8`` template gives fractional values, as in
          skimage.
        - A floating source keeps its dtype; an integer or ``bool`` source returns ``float32``. The quantiles are
          computed in ``float32`` (``float64`` for a ``float64`` source).
        - It is not differentiable: with a non-empty ``source``, no gradient reaches it and a backward pass into
          ``template`` raises.
        - An empty ``source`` with a non-empty ``template`` returns an empty tensor, still attached to its autograd
          graph; an empty ``template`` raises ``ValueError``, even when ``source`` is empty too.

    Args:
        source: Image to transform.
        template: Template image. It can have different dimensions to source.

    Returns:
        The transformed output image, with the shape of the source image.

    """
    oldshape = source.shape
    compute_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    out_dtype = source.dtype if source.is_floating_point() else compute_dtype
    if template.numel() == 0:
        raise ValueError("histogram_matching needs a non-empty template.")
    if source.numel() == 0:
        return source.to(out_dtype)
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

    # A floating source keeps its dtype; an integer source returns float32.
    return interp_t_values[bin_idx].reshape(oldshape).to(out_dtype)


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Convention:
        - Outside ``[xp[0], xp[-1]]`` the end segments are extrapolated linearly, where :func:`numpy.interp` clamps to
          ``fp[0]`` and ``fp[-1]``; SciPy's ``interp1d(xp, fp, fill_value="extrapolate")`` makes the same choice. At a
          knot the result is that knot's ``fp``.
        - ``xp`` and ``fp`` are 1-D; ``x`` can have any shape, which the output keeps. ``xp`` must be increasing, which
          is not checked. The output dtype is the promoted dtype of ``x``, ``xp`` and ``fp``, or the default float
          dtype when all three are integer.
        - Known defects: with integer ``x`` and ``xp`` and a ``float16`` ``fp``, any coordinate difference above 65504
          overflows, and the result can be NaN, inf or silently wrong, such as 0; in ``float16`` a segment whose ``fp``
          difference overflows can be about one ulp off (`#5967 <https://github.com/kornia/kornia/issues/5967>`_).

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.

    """
    if fp.is_complex():
        # The real and imaginary parts are interpolated separately, as np.interp does.
        return torch.complex(_interp_real(x, xp, fp.real), _interp_real(x, xp, fp.imag))
    return _interp_real(x, xp, fp)


def _interp_real(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    # interp for a real fp. A separate function, not a recursive call from interp: TorchScript rejects recursion.
    # Tensor.is_floating_point() rather than dtype.is_floating_point, which TorchScript does not support.
    float_coords = x.is_floating_point() or xp.is_floating_point()
    dtype = torch.promote_types(torch.promote_types(x.dtype, xp.dtype), fp.dtype)
    if not (float_coords or fp.is_floating_point()):
        # All-integer inputs interpolate in the default float dtype (read from a factory: TorchScript has no
        # torch.get_default_dtype).
        dtype = torch.empty(0, device=fp.device).dtype
    # Integer coordinates are searched, compared and subtracted in int64 (as float32 they round past 2**24, and uint8
    # differences wrap); only the two differences that form each weight are converted. Floating coordinates are
    # converted to the value dtype, which never narrows them.
    coord_dtype = dtype if float_coords else torch.int64
    x, xp, fp = x.to(coord_dtype), xp.to(coord_dtype), fp.to(dtype)
    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    dx = x1 - x0
    zero_width = dx == 0
    dx = torch.where(zero_width, torch.ones_like(dx), dx).to(dtype)
    w = (x - x0).to(dtype) / dx
    d = f1 - f0
    # f1 - f0 can overflow while both ends are finite. The value is then the weighted sum of the two ends, with each
    # weight divided by dx on its own, so that d/dx is the slope wherever the slope is representable. The other
    # formulas see zeros there, so their discarded backward does not multiply a zero gradient by the infinite f1 - f0.
    overflow = torch.isinf(d) & torch.isfinite(f0) & torch.isfinite(f1)
    zero = torch.zeros_like(d)
    f0s, f1s = torch.where(overflow, zero, f0), torch.where(overflow, zero, f1)
    ds = f1s - f0s
    # lerp is exact at w = 0 and w = 1, so a point on a knot returns fp at that knot; outside [xp[0], xp[-1]] it
    # extrapolates the end segment.
    out = torch.lerp(f0s, f1s, w)
    a = f0s + w * ds
    b = f1s - ds * (1 - w)
    # lerp's two halves meet an infinite end as inf - inf; try each half in turn, as np.interp does
    out = torch.where(torch.isnan(out), a, out)
    out = torch.where(torch.isnan(out), b, out)
    out = torch.where(torch.isnan(out) & (f0 == f1), f0, out)
    # A point on a knot returns fp there even when the neighbouring fp is infinite (np.interp). The segment formulas
    # keep d/dx = slope at the knot; the bare fp is used only where they are NaN.
    out = torch.where(
        x == x0,
        torch.where(torch.isnan(a), f0, a),
        torch.where(x == x1, torch.where(torch.isnan(b), f1, b), out),
    )
    # The weighted sum is exact at the knots too: its weights are 1 and 0 there.
    out = torch.where(overflow, f0 * ((x1 - x).to(dtype) / dx) + f1 * w, out)
    # A zero-width (repeated) segment is reached only at a clipped end: below a repeated first knot the value is
    # fp[0], at or above a repeated last knot fp[-1].
    out = torch.where(zero_width, torch.where(x < x0, f0, f1), out)
    # A NaN query is NaN, also on a flat or zero-width segment (np.interp).
    return torch.where(torch.isnan(x), torch.full_like(out, float("nan")), out)
