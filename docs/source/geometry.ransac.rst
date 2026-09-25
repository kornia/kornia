kornia.geometry.ransac
======================

.. meta::
   :description: The kornia.geometry.ransac module implements the RANSAC (Random Sample Consensus) algorithm for robust fitting of models in the presence of outliers. The RANSAC class allows for efficient outlier rejection and model estimation, which is crucial in tasks such as stereo vision, homography estimation, and 3D reconstruction. This module is valuable for geometric computer vision problems.

.. currentmodule:: kornia.geometry.ransac

Robust model fitting with RANSAC (Random Sample Consensus), used to estimate homographies, fundamental and essential matrices from noisy correspondences.

Batches, scores, and refinement
-------------------------------

``batch_size`` counts minimal sample sets. The total budget is
``batch_size * max_iter``; seven-point fundamental and five-point essential
solvers may produce several candidate matrices from each set. Smaller batches
permit earlier stopping, while larger batches amortize accelerator overhead.
``confidence=1`` disables early stopping and runs the whole budget.
Measure end-to-end latency and pose accuracy when choosing a batch size.

``score_type="msac"`` minimizes the sum of squared residuals truncated at
``inl_th ** 2``. The returned internal score is normalized to increase with
quality. Confidence stopping always uses the number of inliers, not this score.
Fundamental and essential models use Sampson residuals. Essential estimation
expects camera-normalized coordinates, so its threshold is not in pixels.
Line-segment homographies use the squared mean distance from the transferred
endpoints to the target segment's line; zero-length target segments are outliers.
Their local optimization still weights segments by the length-scaled residual of
:func:`~kornia.geometry.homography.line_segment_transfer_error_one_way`, which
down-weights long segments (`#4867 <https://github.com/kornia/kornia/issues/4867>`_).

A model is accepted only with more inliers than its minimal sample (four
correspondences for homographies, five for essential and seven or eight for
fundamental matrices): a model fitted to a sample always fits that sample, so
only further inliers show a consensus. Without one, the estimator returns the
all-zero matrix and an empty mask.

``prosac_sampling=True`` enables the per-sample growth schedule of
`Chum and Matas (CVPR 2005) <https://cmp.felk.cvut.cz/~matas/papers/chum-prosac-cvpr05.pdf>`_.
Order the correspondences from best to worst before calling the estimator.
For example, when smaller descriptor ratios indicate better matches:

.. code-block:: python

    order = ratios.argsort()
    estimator = RANSAC("fundamental", score_type="msac", prosac_sampling=True, seed=0)
    F, sorted_inliers = estimator(points1[order], points2[order])
    inliers = sorted_inliers.new_empty(sorted_inliers.shape)
    inliers[order] = sorted_inliers

The prefix grows within a batch, and each growth sample includes the newest
correspondence. This implements PROSAC sampling, not its original prefix-based
termination test: it runs the full budget and does not apply the uniform-sampling
confidence formula to biased draws. Uninformative or reversed rankings can hurt
accuracy. The ``weights`` argument is not used to rank correspondences.

By default, local optimization repeatedly refits all current inliers. A refit
replaces the model when it raises the score, or when it ties it: a least-squares
fit on the same support is more precise than the minimal-sample model. A refit
that does not raise the score ends the refinement. This is iterative refitting,
not the full LO-RANSAC algorithm with inner resampling and a threshold schedule.
Set ``max_lo_iters=0`` to disable it. Optionally, ``lo_sample_size`` caps each
randomized non-minimal refit: ``max_lo_iters`` independent subsets are fit in one
solver batch, then the best accepted consensus gets a full-inlier refit.
Consensus sets no larger than the cap use ordinary iterative full-inlier
refitting. This bounded variant is inspired by
`Lebeda et al. (BMVC 2012) <https://cmp.felk.cvut.cz/software/LO-RANSAC/Lebeda-2012-Fixing_LORANSAC-BMVC_abstract.pdf>`_;
it does not implement the complete LO+ algorithm. The cap must be at least the
non-minimal solver's sample size (eight for fundamental/essential matrices).
Evaluate the accuracy/runtime trade-off on your data before enabling it.

.. autoclass:: RANSAC
   :members: forward
