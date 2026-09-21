Template matching
=================

.. currentmodule:: kornia.feature

Locating a reference patch
--------------------------

Use :func:`match_template_zncc` to find the translation of a known reference crop in a batch
of search images or regions of interest. For example, a local registration or patch-tracking
pipeline can use the score map to rank candidate positions while keeping the image and
template in PyTorch. Templates may be shared across the batch or supplied per image.
The search assumes a fixed scale and orientation; it is not a complete tracker.

Each channel is centered spatially, then the channel contributions are combined using the
`OpenCV TM_CCOEFF_NORMED definition
<https://docs.opencv.org/4.13.0/df/dfb/group__imgproc__object.html>`_. The classical correlation
formulation is described by Lewis :cite:`Lewis1995template`, also in Eq. (2) of the author's
`expanded article <https://www.scribblethink.org/Work/nvisionInterface/nip.pdf>`_. This
implementation directly centers spatial tiles; it does not implement the reference's FFT
or summed-area acceleration and makes no speed comparison with OpenCV.

The following synthetic example locates one color template in two search images with
different positive contrast gains and per-channel intensity offsets. A third image is
constant and has no valid location. It needs no external images or model weights.

.. code-block:: python

    import torch
    from kornia.feature import match_template_zncc

    generator = torch.Generator().manual_seed(4700)
    template = torch.rand(1, 3, 5, 5, generator=generator)
    images = torch.rand(3, 3, 24, 28, generator=generator)
    offsets = torch.tensor([0.2, -0.1, 0.3])[:, None, None]
    images[0, :, 6:11, 9:14] = 1.7 * template[0] + offsets
    images[1, :, 13:18, 3:8] = 0.6 * template[0] - offsets
    images[2] = 0
    images.requires_grad_()
    template.requires_grad_()

    scores, valid = match_template_zncc(images, template)
    best_score, index = scores.masked_fill(~valid, -torch.inf).flatten(1).max(1)
    has_valid = valid.flatten(1).any(1)
    out_width = scores.shape[-1]
    xy = torch.stack((index % out_width, index // out_width), dim=-1)
    xy = torch.where(has_valid[:, None], xy, -1)

    assert xy.tolist() == [[9, 6], [3, 13], [-1, -1]]
    assert has_valid.tolist() == [True, True, False]
    torch.testing.assert_close(best_score[:2], torch.ones(2), atol=2e-6, rtol=0)

    # The score map supports gradients; the integer peak coordinates above do not.
    loss = -scores[valid].mean()
    loss.backward()
    assert torch.isfinite(images.grad).all() and torch.isfinite(template.grad).all()

Coordinates identify the template's top-left corner. With real inputs, crop the template
from a reference image and supply the search images in ``(B,C,H,W)`` layout. If a search
image is an ROI, add its origin to recover full-image coordinates. Repeated patterns may
produce several peaks; applications must choose their own ambiguity handling and acceptance
thresholds on representative data. The example above validates usage on synthetic inputs,
not real-image detection accuracy.

Validity and related APIs
-------------------------

The mask expresses numerical validity, not detection confidence or visibility. Constant
templates or windows return zero and false. Masking before peak selection matters because
an invalid zero score can exceed a valid negative score. ``min_variance`` is a strict
threshold on average centered energy in squared intensity units, not an additive denominator
epsilon. Changing the intensity scale can therefore change validity even when the ideal
score is unchanged. Arbitrary independent gains in different channels are not invariant.

.. list-table:: Related operations
   :header-rows: 1
   :widths: 35 65

   * - Operation
     - Relationship to this API
   * - :func:`kornia.feature.match_nn` and descriptor matchers
     - Match descriptor sets; they do not return a sliding raw-patch score map.
   * - :func:`kornia.filters.filter2d`
     - Performs per-channel correlation. Its ``normalized=True`` option L1-normalizes the
       kernel, rather than centering and normalizing both operands in every image window.
   * - OpenCV ``matchTemplate(..., TM_CCOEFF_NORMED)``
     - Uses the same multichannel formula on nondegenerate inputs, but a single image pair
       and different constant-template handling. This API adds explicit batching,
       PyTorch gradients, and a numerical-validity mask.

Supported inputs are float32/float64 tensors. There is no padding, rotation or scale search,
or template weighting mask. Gradients describe nondegenerate regions of the score map;
the validity decision and hard peak selection are not differentiable.

.. autofunction:: match_template_zncc
