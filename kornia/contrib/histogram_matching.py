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
        The transformed output image as the same shape as the source image.

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
    # template images (maps pixel value --> quantile)

    s_quantiles = torch.cumsum(s_counts, dim=0, dtype=source.dtype)
    s_quantiles = s_quantiles / s_quantiles[-1]
    t_quantiles = torch.cumsum(t_counts, dim=0, dtype=source.dtype)
    t_quantiles = t_quantiles / t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Interpolate ``x`` tensor according to ``xp`` and ``fp`` as in ``np.interp``.

    This implementation cannot reproduce numpy results identically, but reasonable.
    Code referred to `here <https://github.com/pytorch/pytorch/issues/1552#issuecomment-926972915>`_.

    Args:
        x: the input tensor that needs to be interpolated.
        xp: the x-coordinates of the referred data points.
        fp: the y-coordinates of the referred data points, same length as ``xp``.

    Returns:
        The interpolated values, same shape as ``x``.
    """
    if x.dim() != xp.dim() != fp.dim() != 1:
        raise ValueError(
            f"Required 1D vector across ``x``, ``xp``, ``fp``. Got {x.dim()}, {xp.dim()}, {fp.dim()}."
        )

    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    locs = torch.searchsorted(xp, x)
    locs = locs.clip(1, len(xp) - 1) - 1
    return slopes[locs] * (x - xp[locs]) + xp[locs]
