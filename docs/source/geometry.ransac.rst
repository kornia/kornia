kornia.geometry.ransac
======================

.. meta::
   :description: The kornia.geometry.ransac module implements the RANSAC (Random Sample Consensus) algorithm for robust fitting of models in the presence of outliers. The RANSAC class allows for efficient outlier rejection and model estimation, which is crucial in tasks such as stereo vision, homography estimation, and 3D reconstruction. This module is valuable for geometric computer vision problems.

.. currentmodule:: kornia.geometry.ransac

Robust model fitting with RANSAC (Random Sample Consensus), used to estimate homographies, fundamental and essential matrices from noisy correspondences.

.. autoclass:: RANSAC
   :members: forward
