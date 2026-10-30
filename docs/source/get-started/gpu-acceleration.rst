GPU-accelerated vision
======================

.. meta::
   :description: Every Kornia operator is a batched PyTorch function: move the input tensors to CUDA or Apple MPS and the whole pipeline runs on the GPU, with no per-image Python loops and no code changes.

Kornia operators are plain PyTorch functions, so they follow PyTorch's device model: an operation runs on
whatever device its input tensors live on. There is no ``use_gpu`` flag and nothing to configure; move the
data, and the pipeline moves with it.

.. code-block:: python

   import torch
   import kornia

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   images = torch.rand(64, 3, 224, 224, device=device)  # a whole batch on the GPU
   gray = kornia.color.rgb_to_grayscale(images)
   blurred = kornia.filters.gaussian_blur2d(gray, (5, 5), (1.5, 1.5))
   rotated = kornia.geometry.transform.rotate(blurred, torch.full((64,), 30.0, device=device))

Three habits make GPU pipelines fast:

- **Batch.** Every operator takes ``(B, C, H, W)`` tensors and processes the whole batch in one kernel launch;
  a Python loop over single images throws that away.
- **Load onto the device.** :func:`kornia.io.load_image` takes a ``device`` argument, so images can be decoded
  straight to GPU memory.
- **Stay on the device.** Avoid round trips through NumPy or PIL in the middle of a pipeline; Kornia covers
  color, filtering, geometry and augmentation, so the whole preprocessing chain can stay on the GPU.

Augmentations are the typical win: a :class:`~kornia.augmentation.AugmentationSequential` pipeline samples and
applies transforms for the entire batch on the GPU, inside the training loop, and most operators also run under
``torch.compile`` for kernel fusion. The :doc:`performance page <performance>` shows measured eager-mode
comparisons against torchvision, albumentations, OpenCV and PIL on the devices the committed benchmark runs
cover -- CPU and Apple silicon today, with more hardware being added -- and the
:doc:`precision page <precision>` documents ``float16``/``bfloat16`` support.

Apple silicon (``device="mps"``) is supported as well. PyTorch's MPS backend has no ``float64`` tensors, so keep
inputs in ``float32`` (or half precision) there, and be aware that autocast on MPS can change the effective
dtype of an operation.

Choosing a feature pipeline: measured wins and losses
-----------------------------------------------------

An accelerator-compatible implementation is not automatically faster than its reference.
For local features, choose the detector, descriptor and matcher together, and check both
accuracy and latency on your workload. The following **interim M1 audit** compares Kornia
at commit ``160e0f0f7`` with pinned author implementations and explicitly named alternatives.
These measurements do not predict CUDA or Intel performance.

.. list-table:: Baseline guidance from the feature audit
   :header-rows: 1
   :widths: 22 35 43

   * - Pipeline / reference
     - Measured result
     - When to choose it
   * - MKD / author PyTorch
     - Oxford speedup: **2.33–2.45× CPU**, **5.97–6.34× MPS** for polar/concatenated
       descriptors. Oxford @3 px accuracy is identical; PhotoTourism accuracy is slightly lower.
     - A strong measured speed advantage for these descriptor pipelines. Validate accuracy
       on your data; this is not a claim that MKD is the best descriptor overall.
   * - ALIKED n16 / author Python
     - **1.67× CPU** speedup on Oxford, with identical @3 px accuracy. Baseline MPS runs at
       **0.59×** the reference speed (about 1.69× the latency).
     - Prefer Kornia for this measured CPU workload. For MPS inference, the author Python
       implementation is faster at this revision; benchmark again after optimization.
   * - XFeat / author
     - **1.46× CPU** speedup on Oxford; @3 px accuracy **60.00% / 59.17%** (Kornia / author).
       MPS timing is approximately tied and noisy.
     - A measured CPU advantage. Do not assume the same speedup on an accelerator.
   * - SIFT pipelines / OpenCV RootSIFT
     - Kornia runs at **0.07–0.13×** OpenCV's CPU speed on Oxford (roughly 8–14× the latency).
       @3 px accuracy spans **69.17–72.50%**, versus OpenCV's **70.00%**.
     - Prefer OpenCV for latency-sensitive CPU SIFT inference. Choose Kornia when tensor
       composition or differentiable components matter, and measure the resulting pipeline.
   * - LoFTR / author
     - Oxford speed ratios **0.99× CPU**, approximately **1.06× MPS**; MPS timing is noisy.
       Oxford and HPatches aggregate @3 px accuracy is identical.
     - Treat these as effective ties on this machine. Existing optimizations do not establish
       a meaningful M1 speed advantage in this experiment.
   * - XFeat* / author, accuracy tradeoff
     - HPatches @3 px accuracy is **36.09% / 20.46%** (Kornia / author). However, an isolated
       comparison of Kornia's refinement change against its earlier revision reduced
       PhotoTourism mAA by **1.58 percentage points** while improving HPatches @3 px.
     - The planar improvement is substantial on this subset; it is not a universal accuracy
       improvement. Check the application metric before selecting a refinement policy.
   * - HardNet, SOSNet, HyNet, TFeat / authors, using shared SIFT frames
     - Near parity in full-pipeline latency; aggregate accuracy is identical on the measured cohorts.
     - Choose based on integration needs. These results do not establish a broad speed or
       accuracy advantage for the ports.

Speedup means **reference latency / Kornia latency** on the same device and timed pairs;
values below one favor the reference. These are full-pipeline measurements, including both
feature extractions, matching, output transfer and shared geometry/scoring. Image I/O and
input transfer are excluded. They are not isolated descriptor kernel timings.

The accuracy cohorts are Oxford's 40 pairs, **145 pairs from 29 viewpoint-only HPatches
sequences**, and **130 eligible PhotoTourism validation pairs from three scenes** (ten seeded
images per scene). This is neither the full HPatches benchmark nor an official IMC score.
Planar accuracy is the percentage with mean homography corner error below 3 px; PhotoTourism
uses scene-macro pose mAA over 1–10 degrees. A common OpenCV MAGSAC estimator and three
match-order permutations are used; standalone RANSAC performance is outside this audit.

Inputs use FP32, requested maximum side 640 (dimensions rounded to multiples of 32), and a
requested sparse budget of 2,048 features; native dense LoFTR is uncapped. The machine has
an 8 GB Apple M1, Python 3.11.14 and PyTorch 2.14.0. PyTorch uses four threads; OpenCV reports
eight. MPS fallback is enabled, so these are device-pipeline results, not a claim that every
operation executes on the GPU. Oxford timing usually covers three pairs (XFeat: eight), with
warmup, synchronization and repeated medians/IQRs. Small differences and noisy runs should
not drive a deployment decision. Accuracy covers all stated pairs, independently of timing.

Proposed improvements, measured separately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These candidate changes are **not included in the baseline above**. The audit motivates
separate implementation changes and further hardware validation:

.. list-table:: Candidate versus the same Kornia baseline
   :header-rows: 1
   :widths: 25 75

   * - Change
     - Measured result and limit
   * - ALIKED MPS inference
     - **1.59× Oxford / 1.71× HPatches** full-pipeline speedup, with unchanged aggregate accuracy.
       Patched Kornia is approximately tied with the author on Oxford and remains slightly
       slower on HPatches. This does not establish a CUDA speedup.
   * - MPS SNN / SMNN
     - **1.42–1.49× / 1.26–1.33×** matcher-only speedup across the three cohorts;
       identical indices and ratios across **315 pairs**. These gains are not whole-pipeline
       gains, and the candidate does not universally beat OpenCV matching.
   * - Affine-shape mirror fix
     - PhotoTourism mAA **+0.49 percentage points**; Oxford @3 px unchanged;
       HPatches @3 px **−0.46 percentage points**. Correctness improves; accuracy effects are
       mixed. No speedup is claimed.

Native recipes that change the matching policy, detector or sampling convention must be
reported separately from matched port comparisons. In particular, a higher native-pipeline
score alone does not identify a better descriptor implementation. Reuse the same images,
weights, feature budget and estimator when extending this audit to another machine, and
publish losses and missing measurements alongside wins.
