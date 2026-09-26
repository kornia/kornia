kornia.geometry.ransac
======================

.. meta::
   :description: The kornia.geometry.ransac module implements the RANSAC (Random Sample Consensus) algorithm for robust fitting of models in the presence of outliers. The RANSAC class allows for efficient outlier rejection and model estimation, which is crucial in tasks such as stereo vision, homography estimation, and 3D reconstruction. This module is valuable for geometric computer vision problems.

.. currentmodule:: kornia.geometry.ransac

Robust model fitting with RANSAC (Random Sample Consensus), used to estimate homographies, fundamental and essential matrices from noisy correspondences.

Batches, scores, and refinement
-------------------------------

Hypotheses are generated and verified in batches. The sampling budget is
``max_samples`` minimal samples when given, otherwise ``batch_size * max_iter``
(``2048 * max_iter``, the historical default, with ``batch_size="auto"``); the
last batch is truncated to it. Seven-point fundamental and five-point essential
solvers may produce several candidate matrices from each set. Early stopping is
checked between batches, and ``confidence=1`` runs the whole budget.

The default ``batch_size="auto"`` picks the batch per call. On CUDA and MPS a
homography batch costs about the same from a few hundred up to 8192 hypotheses,
so the budget is drawn in batches of 8192; the epipolar solvers are
compute-bound past 2048 hypotheses, so their batches stop there. Verification
holds a ``batch x N`` residual matrix, so past ``2**27`` entries (about 1 GiB
at peak in float32) the batch shrinks with ``N``, down to 2048. A PROSAC batch
of either size spans most of the growth schedule, whereas certifying a model
after a small first batch drawn from a short prefix can stop on a poorly
conditioned fit. On CPU
the cost is linear in ``batch * N`` residuals and the eight-point solver costs
about ten DLTs, so the batch aims at a millisecond or so of work: 256 to 2048
hypotheses for homographies and 128 to 512 for the epipolar models, fewer for
more correspondences, which lets early stopping pay off. An integer
``batch_size`` fixes the batch on every device.

``score_type="msac"`` minimizes the sum of squared residuals truncated at
``inl_th ** 2``. The returned internal score is normalized to increase with
quality. Confidence stopping always uses the number of inliers, not this score.
Fundamental and essential models use Sampson residuals. Essential estimation
expects camera-normalized coordinates, so its threshold is not in pixels.
Line-segment homographies use the squared mean distance from the transferred
endpoints to the target segment's line; zero-length target segments are outliers.
Their local optimization re-weights segments by the same distance.

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
correspondence. Stopping follows the paper's termination-length test: a ranked
prefix certifies the incumbent when its support there is non-random (a Chernoff
bound on accidental inliers with probability 0.05, at significance 0.05) and the
draws it needs for ``confidence`` fit inside the prefix's growth interval, so
that only draws from that prefix count. As in OpenCV's USAC, a prefix shorter
than ``min(N / 2, 100)`` or supporting fewer than 20% of all correspondences
cannot terminate; the whole set is always a candidate, so PROSAC never runs
longer than the uniform bound. Stopping is checked between batches.
Uninformative or reversed rankings can hurt accuracy. The ``weights`` argument
is not used to rank correspondences.

PROSAC pays off when the ranking tracks inlier-ness and the inlier ratio is low,
where uniform sampling cannot stop early: on SIFT correspondences with a ratio
test (HEB), it adds about 0.1 mAA at equal time for homographies. On learned
matchers with 85% or more inliers, uniform sampling already stops after one
batch, and the confidence-ranked prefixes can be spatially clustered, so the
model PROSAC certifies first may fit them well and the pose poorly. Prefer
uniform sampling there, or run PROSAC with ``confidence=1`` when the extra
draws are affordable.

By default, local optimization repeatedly refits all current inliers. For the
homography models the refit is the iteratively re-weighted least squares of
:func:`~kornia.geometry.homography.find_homography_dlt_iterated`, whose Gaussian
weights use ``inl_th`` as their standard deviation, so a correspondence at the
threshold keeps weight ``exp(-1/2)``. A refit
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
   :members: forward, resolve_batch_size
