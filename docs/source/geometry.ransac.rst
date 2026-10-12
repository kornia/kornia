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

``model_type="fundamental"`` draws seven-point samples and completes each with
up to three rank-2 matrices, the roots of the cubic det F = 0, like
``"fundamental_7pt"``; ``"fundamental_8pt"`` draws eight-point samples. Seven
correspondences make an all-inlier sample more likely than eight, so fewer
samples reach the same confidence.

The default ``batch_size="auto"`` picks the batches per call. With the default
``local_optimization="lm"`` (see below), a CPU call starts with 256 samples, 512
for homographies, and doubles the batch after each one up to 2048, 4096 for
homographies: inputs with many inliers stop after a small first batch, and the
others pay the per-batch overhead only a few times. On CUDA and MPS the batch
is the whole budget up to 8192 samples. Essential matrices start smaller on
every device, with 64 samples up to 1024 on CPU and 256 up to 8192 on CUDA and
MPS: a five-point sample needs few draws at high inlier ratios, and its
eigenvalue solve runs on the host (on MPS, the whole solve). All shrink for many
correspondences, down to 64 samples, so that above that floor a batch scores at
most ``2**22`` (CPU) or ``2**25`` residuals. The rest of
this paragraph describes ``local_optimization="dlt"``: on CUDA and MPS a
homography batch costs about the same from a few hundred up to 8192 hypotheses,
so the budget is drawn in batches of 8192; the epipolar solvers are
compute-bound past 2048 hypotheses, so their batches stop there. Verification
holds a ``batch x N`` residual matrix, so past ``2**27`` entries (about 1 GiB
at peak in float32) the batch shrinks with ``N``, down to 2048. Other devices
retain the historical 2048-sample batch. A PROSAC batch
of either size spans most of the growth schedule, whereas certifying a model
after a small first batch drawn from a short prefix can stop on a poorly
conditioned fit. On CPU
the cost is linear in ``batch * N`` residuals and the eight-point solver costs
about ten DLTs, so the batch aims at a millisecond or so of work: 256 to 2048
hypotheses for homographies and 128 to 512 for the epipolar models, fewer for
more correspondences, which lets early stopping pay off. An integer
``batch_size`` fixes the batch on every device.

The default ``score_type="msac"`` minimizes the sum of squared residuals
truncated at ``inl_th ** 2``; ``"ransac"`` counts inliers instead. The returned
internal score is normalized to increase with quality. Confidence stopping
always uses the number of inliers, not this score.
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

Local optimization
------------------

Homographies, fundamental and essential matrices default to
``local_optimization="lm"``. The sampling loop keeps the eight best-scoring
minimal models of the whole run, in coordinates normalized once per call (the
caller's calibrated ones for essential matrices, see below). After sampling,
all eight are refined together with
``max_lo_iters`` Levenberg-Marquardt iterations that minimize the squared
Sampson distance or one-way transfer error of every correspondence, truncated
at ``inl_th``. The best-scoring model among the refits and the minimal models
is then refined on its own inliers with ``refine_iters`` iterations of a Cauchy
loss of scale ``inl_th / 3``, the noise level of a threshold at three standard
deviations, and the returned mask holds the inliers of the returned model after
conversion to the input dtype. If rounding removes sufficient support, the call
returns the all-zero model and empty inlier mask used for estimation failure.
Fundamental matrices keep rank two through the parametrization of
`Bartoli and Sturm (TPAMI 2004) <https://doi.org/10.1109/TPAMI.2004.1265873>`_;
the refinement follows PoseLib's ``refine_fundamental`` and
``refine_homography``. Refining several models once sampling ends, rather than
each new incumbent, is the batched form of PoseLib's rule of refining every
minimal model that improves on the best one so far. The homography and
fundamental-matrix solvers take their null spaces from a partial-pivoted LU
factorization, the five-point solver from Householder reflections, and scores come from
one matrix product per batch, which keeps the per-hypothesis cost low on every
device. The refinements are a few dozen small operations per iteration and run
on the CPU in float64 for every device, where launch latency would dominate.
Early stopping follows the support of the best minimal model.

On PhotoTourism pairs this raises the pose mAA of fundamental matrices by 0.04
to 0.1 over the subset refits below at the same sample budget. SIFT matches take
a third of the time on CPU and half on CUDA; ALIKED matches with LightGlue, on
which both stop early, take about 0.5 ms longer on CPU. On the HEB homographies
it adds about 0.04 mAA at a quarter to a half of the time.

Essential matrices are estimated in the caller's calibrated coordinates, without
the normalization, which would take them off the essential manifold, and are
returned with unit Frobenius norm and their largest entry positive. They are
refined as ``E = U diag(1, 1, 0) V^T`` with five parameters, as many as the
rotation and translation direction of PoseLib's ``refine_relpose``: rotations
of ``U`` about three axes and of ``V`` about its first two, which leaves out
the rotation of both about their third axes that does not change ``E``. The
five-point samples are solved with Nister's method in float64: a Householder
null space, the degree-ten polynomial from polynomial products, and its real
roots from the eigenvalues of its companion matrix on the host. MPS, which has
no float64, solves its samples on the host. On PhotoTourism pairs this raises
the pose mAA of essential matrices by 0.03 to 0.07 over the subset refits
below, in about half the time on CPU, and an eighth of it with 8000 SIFT
features per image.

``local_optimization="dlt"``, the only choice for line-segment homographies,
refits the incumbent on its inliers. For the
homography models the refit is the iteratively re-weighted least squares of
:func:`~kornia.geometry.homography.find_homography_dlt_iterated`, whose Gaussian
weights use ``inl_th`` as their standard deviation, so a correspondence at the
threshold keeps weight ``exp(-1/2)``. A refit
replaces the model when it raises the score, or when it ties it: a least-squares
fit on the same support is more precise than the minimal-sample model. A refit
that does not raise the score ends the refinement. This is iterative refitting,
not the full LO-RANSAC algorithm with inner resampling and a threshold schedule.
Set ``max_lo_iters=0`` to disable it. By default, ``lo_sample_size=32`` caps each
randomized non-minimal refit: ``max_lo_iters`` independent 32-inlier subsets are
fit in one solver batch, then the best accepted consensus gets a full-inlier
refit. Consensus sets no larger than the cap use ordinary iterative full-inlier
refitting, which ``lo_sample_size=None`` selects for every consensus set. This
bounded variant is inspired by
`Lebeda et al. (BMVC 2012) <https://cmp.felk.cvut.cz/software/LO-RANSAC/Lebeda-2012-Fixing_LORANSAC-BMVC_abstract.pdf>`_;
it does not implement the complete LO+ algorithm. The cap must be at least the
non-minimal solver's sample size (eight for fundamental/essential matrices).
On PhotoTourism fundamental matrices the subset refits match or beat full-inlier
refitting at a lower cost on both CPU and CUDA.

Compiled estimation
-------------------

``compile=True`` (torch 2.14 or later) runs the whole ``local_optimization="lm"``
estimation as one ``torch.compile`` graph on CPU and CUDA. The sampling loop is a
``torch.while_loop`` that carries the pool of the best minimal models, the
early-stopping bound is computed inside the graph, and the batch sizes and the
number of models that survive the degeneracy tests are dynamic. The graph does
not depend on the number of correspondences, the threshold, the confidence or
the sample budget: it is traced once per model type, score type, iteration
counts, dtype and device. The first call compiles it, in 30 s for homographies
to 60 s for essential matrices on an Apple M1, and saves the compiled function
next to inductor's cache (``TORCHINDUCTOR_CACHE_DIR``); later processes load it
in about a second. ``KORNIA_RANSAC_AOT=0`` disables that file, and each process
then compiles again, in 17 to 35 s once inductor's kernels are cached.
Ordinary calls and calls under ``torch.inference_mode()`` share the same artifact.
Compiled arithmetic also runs with autocast disabled, using the input's working
precision and float64 host refinement. The artifact cache distinguishes CPU
thread counts, default dtypes and default devices; changing these settings may require another
first-call compilation. On CPU, large explicit sampling batches are split so
scoring holds at most ``2**22`` residuals per batch (or one hypothesis when
``N`` alone exceeds that limit). This keeps the total sample budget and can
check confidence stopping earlier than the requested batch size would.
The sampling keys are a counter-based hash of the call's seed and each
sample's index, computed inside the graph, and Floyd's algorithm turns them
into subsets on every device, where compilation fuses its steps into one
kernel. A sample's keys therefore depend neither on batch boundaries nor on the
device, calls with different seeds do not share samples, and seeded calls
never touch a global generator, including during concurrent calls. An unseeded
call draws one seed from the input device's global generator and then hashes it
as a seeded call does.
On CUDA the loop counters and stopping bound stay on the host, with a transfer
of the leading score and inlier count after each batch; compilation therefore
does not remove every host-device synchronization.

For the initial implementation at revision ``ad91a621``, on that CPU
(4 threads, synthetic scenes of 500 to 5000 correspondences with
20% or 50% inliers, averaged over seeds), compiled calls were 1.2 to 1.7 times
faster for homographies, 1.6 to 2.4 times for fundamental matrices and 1.1 to
1.6 times for essential matrices, whose five-point solver gained nothing: its
time goes to LAPACK factorizations and the eigenvalues of the companion matrix.
The compiled call runs the same algorithm with its own random stream, so a
seeded call is reproducible but differs from the eager one. On synthetic
scenes of 500 and 2000 correspondences with 20% to 50% inliers its mean inlier
recall was within 0.015 of eager's for homographies and essential matrices and
within 0.03 for fundamental matrices, where eager runs with other seeds differ
from each other as much (60 seeds per case at 20% and 30% inliers). PROSAC
sampling, ``degensac=True`` and ``local_optimization="dlt"`` are not supported.
The current sampler draws a different seeded stream;
use ``benchmarks/geometry/ransac_compile_synthetic.py`` to measure current
latency and recovery, including ``--confidence 1`` for equal sample budgets.

Dominant planes
---------------

A seven-point sample with five correspondences on one plane yields a fundamental
matrix consistent with the whole plane, whatever its two other correspondences
are (Chum, Werner and Matas, CVPR 2005). When most inliers lie on one plane, such
a wrong model can have the most support and stop the sampling, and no refinement
recovers the off-plane inliers it never counted. ``"fundamental"`` and
``"fundamental_7pt"`` support opt-in DEGENSAC (``degensac=True``): record-setting samples
are tested for this degeneracy, and a degenerate one hands its plane homography
to a plane-and-parallax search over the correspondences off the plane, whose
best model, refined, competes with the minimal ones. Each plane is searched once
per call.
:func:`~kornia.geometry.homography.sampson_homography_distance` measures the
plane. The test's tolerance, taken from Chum's implementation, also flags samples
in scenes without a dominant plane, so the estimate for a given seed can change
there too. ``degensac=False`` (the default) and ``degensac=None`` leave it off,
retaining the plain seven-point behavior.

.. autoclass:: RANSAC
   :members: forward, resolve_batch_size
