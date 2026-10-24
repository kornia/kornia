kornia.tracking
===============

.. meta::
   :description: Track sparse points with Lucas-Kanade or track planar targets with HomographyTracker using PyTorch tensors.

.. currentmodule:: kornia.tracking

Sparse point tracking
---------------------

Use :func:`track_points_lk` to propagate already-detected points between nearby grayscale
frames in a PyTorch pipeline. It estimates an independent local translation for each point
and returns coordinates, a boolean validity mask, and a mean squared patch error. The
algorithm is the translation-only, inverse-compositional variant of Lucas--Kanade
:cite:`LucasKanade1981,BakerMatthews2004`, with a single image scale.

The following synthetic example tracks two points in each of two image pairs with known
small translations. A third point lies at the image boundary and is rejected because its
window is not fully supported. No external images or pretrained models are required.

.. code-block:: python

    import torch
    from kornia.tracking import track_points_lk

    y, x = torch.meshgrid(torch.arange(64.0), torch.arange(72.0), indexing="ij")

    def frame(dx=0.0, dy=0.0):
        xx, yy = x - dx, y - dy
        return (
            0.5
            + 0.18 * torch.sin(0.42 * xx + 0.12 * yy)
            + 0.15 * torch.cos(0.16 * xx - 0.48 * yy)
            + 0.08 * torch.sin(0.29 * xx + 0.36 * yy)
        )[None, None]

    shifts = torch.tensor([[0.7, -0.4], [-0.8, 0.6]])
    previous = frame().expand(2, -1, -1, -1)
    following = torch.cat([frame(*shift) for shift in shifts])
    points = torch.tensor([[[20.2, 20.4], [39.3, 37.2], [0.0, 0.0]]]).expand(2, -1, -1)

    tracked, valid, mse = track_points_lk(previous, following, points, window_size=15)
    displacement = tracked - points
    expected = shifts[:, None].expand_as(displacement)
    assert valid.tolist() == [[True, True, False], [True, True, False]]
    torch.testing.assert_close(displacement[valid], expected[valid], atol=0.03, rtol=0)
    assert torch.isinf(mse[~valid]).all()

    # Correspondences from the first image pair, for downstream geometric estimation.
    source_points = points[0, valid[0]]
    target_points = tracked[0, valid[0]]

For video frames, supply floating grayscale tensors ``(B,1,H,W)`` and pixel coordinates
``(B,N,2)`` in ``(x,y)`` order, with matching dtype and device. Intensities in ``[0,1]``
are recommended: the texture threshold and MSE depend on the intensity scale. The optional
``points_next`` argument contains absolute next-frame initial estimates, not displacement
vectors. Larger motions need a suitable initial estimate; this function builds no pyramid.
Point detection, trajectory management, and reacquisition remain application responsibilities.

The mask requires sufficient two-dimensional texture, complete window support, and
convergence within the iteration budget. It is not a visibility or occlusion estimate.
Applications can add photometric or forward/backward consistency checks appropriate to
their data. Gradients through valid tracks are local to regions away from interpolation,
stopping, and validity boundaries.

Relationship to other tracking APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`HomographyTracker` maintains a planar target and uses feature matching and RANSAC
to estimate a homography. ``track_points_lk`` is stateless and returns individual point
correspondences; it does not implement the tracker's class interface or estimate a homography.
:class:`~kornia.geometry.transform.image_registrator.ImageRegistrator` instead optimizes a global image transform.
These tools share Kornia's tensor and coordinate conventions while serving different tasks.

OpenCV provides `calcOpticalFlowPyrLK
<https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html>`_. Its ``maxLevel=0`` mode
is the closest scope comparison, but it is not numerically or API-equivalent:

.. list-table:: Sparse LK conventions
   :header-rows: 1
   :widths: 20 40 40

   * - Convention
     - ``track_points_lk``
     - OpenCV CPU ``calcOpticalFlowPyrLK``
   * - Inputs
     - Batched floating grayscale images and points; float32 or float64
     - An 8-bit image pair or pyramid and float32 points
   * - Initial estimate
     - Optional absolute ``points_next`` coordinates
     - ``nextPts`` with ``OPTFLOW_USE_INITIAL_FLOW``
   * - Sampling and support
     - Central differences, bilinear sampling, complete in-image windows
     - Scharr derivatives and padded border support
   * - Status and error
     - Convergence-based boolean validity and MSE; invalid error is infinity
     - Byte status and default L1 error; invalid error is undefined

The reference window additionally needs a one-pixel gradient halo. Thresholds should not be
copied between implementations without accounting for intensity and derivative scaling.
The OpenCV details above refer to its CPU implementation; output equality is not promised.

.. autofunction:: track_points_lk

Planar target tracking
----------------------

.. autoclass:: HomographyTracker
   :members:
