kornia.tracking
===============

.. meta::
   :description: The `kornia.tracking` module provides tools for tracking objects across frames in computer vision tasks. It includes classes like `HomographyTracker` to estimate homography transformations and track objects over time.

.. currentmodule:: kornia.tracking

Track a planar target through video frames. :class:`HomographyTracker` matches each grayscale ``(1, 1, H, W)`` frame
against a target image with local features and returns the homography from target pixels to frame pixels, reusing
the previous frame's homography to guide the next match.

.. autoclass:: HomographyTracker
   :members:
