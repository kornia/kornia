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

A GPU does not make every port faster than the library it replaces. For local features, the
:ref:`image matching guide <feature-speed-by-device>` compares each of kornia's pipelines with its reference
implementation on CUDA, an x86 CPU and Apple silicon, including where kornia is slower.

Apple silicon (``device="mps"``) is supported as well. PyTorch's MPS backend has no ``float64`` tensors, so keep
inputs in ``float32`` (or half precision) there, and be aware that autocast on MPS can change the effective
dtype of an operation.

.. _mps-linear-algebra:

Several geometry solvers need linear-algebra kernels that PyTorch's MPS backend gained in torch 2.7, 2.10 and 2.14.
On an older torch they raise ``NotImplementedError`` on MPS. Each row gives the first torch that runs the call on
MPS (Apple M1, ``PYTORCH_ENABLE_MPS_FALLBACK`` unset):

.. list-table::
   :header-rows: 1
   :widths: 60 25 15

   * - Call
     - Missing kernel
     - Runs on MPS from
   * - :class:`~kornia.geometry.ransac.RANSAC` with ``"homography"``, ``"fundamental"``, ``"fundamental_7pt"``
       or ``"fundamental_8pt"``; :func:`~kornia.geometry.epipolar.find_fundamental` with ``method="7POINT"``,
       or with ``method="8POINT"`` on exactly eight points and no weights
     - ``linalg_lu_factor_ex``
     - torch 2.7
   * - :func:`~kornia.geometry.homography.find_homography_dlt` with the default ``solver="lu"``, and
       :func:`~kornia.geometry.homography.find_homography_dlt_iterated`
     - ``linalg_lu_factor_ex`` and ``linalg_lu_solve``
     - torch 2.10
   * - :func:`~kornia.geometry.epipolar.find_fundamental` with the default ``method="8POINT"`` on more than eight
       points or with weights
     - ``linalg_eigh``
     - torch 2.14

The other :class:`~kornia.geometry.ransac.RANSAC` models (``"affine"``, ``"fundamental_affine"``, ``"essential"`` and
``"homography_from_linesegments"``), :func:`~kornia.geometry.epipolar.find_essential` and
:func:`~kornia.geometry.homography.find_homography_dlt` with ``solver="svd"`` run on MPS from torch 2.5.1. On an older
torch, set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` in the environment that starts Python: PyTorch then runs the missing
kernels on the CPU and copies their results back to MPS.
