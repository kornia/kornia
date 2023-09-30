kornia.augmentation
===================

This module implements in a high level logic. The main features of this module, and similar to the rest of the
library, is that can it perform data augmentation routines in a batch mode, using any supported device,
and can be used for backpropagation. Some of the available functionalities which are worth to mention are the
following: random rotations; affine and perspective transformations; several random color intensities transformations,
image noise distortion, motion blurring, and many of the different differentiable data augmentation policies.
In addition, we include a novel feature which is not found in other augmentations frameworks,
which allows the user to retrieve the applied transformation or chained transformations after each
call e.g. the generated random rotation matrix which can be used later to undo the image transformation
itself, or to be applied to additional metadata such as the label images for semantic segmentation,
in bounding boxes or landmark keypoints for object detection tasks. It gives the user the flexibility to
perform complex data augmentations pipelines.

Try it out :

.. image:: https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Interactive%20Demo-blue
   :target: https://huggingface.co/spaces/kornia/kornia-augmentations-tester

Benchmark
---------

.. table:: Here is a benchmark performed on `Google Colab <https://colab.research.google.com/drive/1b-HpK4EsZR8uolztgH4roNBLaDwcMULx?usp=sharing>`_
   K80 GPU with different libraries and batch sizes. This benchmark shows
   strong GPU augmentation speed acceleration brought by Kornia data augmentations. The image size is fixed to 224x224 and the
   unit is milliseconds (ms).

   +--------------------------------+-----------------+-----------------+-----------------------------------------------------+
   |           Libraries            |   TorchVision   | Albumentations  |                 Kornia (GPU)                        |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |          Batch Size            |       1         |        1        |        1        |        32       |        128      |
   +================================+=================+=================+=================+=================+=================+
   |      RandomPerspective         |     4.88±1.82   |    4.68±3.60    |   4.74±2.84     |   0.37±2.67     |   0.20±27.00    |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |          ColorJiggle           |     4.40±2.88   |    3.58±3.66    |   4.14±3.85     |   0.90±24.68    |   0.83±12.96    |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |        RandomAffine            |     3.12±5.80   |    2.43±7.11    |   3.01±7.80     |   0.30±4.39     |   0.18±6.30     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |      RandomVerticalFlip        |     0.32±0.08   |    0.34±0.16    |   0.35±0.82     |   0.02±0.13     |   0.01±0.35     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |      RandomHorizontalFlip      |     0.32±0.08   |    0.34±0.18    |   0.31±0.59     |   0.01±0.26     |   0.01±0.37     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |           RandomRotate         |     1.82±4.70   |    1.59±4.33    |   1.58±4.44     |   0.25±2.09     |   0.17±5.69     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |           RandomCrop           |     4.09±3.41   |    4.03±4.94    |   3.84±3.07     |   0.16±1.17     |   0.08±9.42     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |           RandomErasing        |     2.31±1.47   |    1.89±1.08    |   2.32±3.31     |   0.44±2.82     |   0.57±9.74     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |          RandomGrayscale       |     0.41±0.18   |    0.43±0.60    |   0.45±1.20     |   0.03±0.11     |   0.03±7.10     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |         RandomResizedCrop      |     4.23±2.86   |    3.80±3.61    |   4.07±2.67     |   0.23±5.27     |   0.13±8.04     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
   |         CenterCrop             |     2.93±1.29   |    2.81±1.38    |   2.88±2.34     |   0.13±2.20     |   0.07±9.41     |
   +--------------------------------+-----------------+-----------------+-----------------+-----------------+-----------------+


.. currentmodule:: kornia.augmentation

.. toctree::

   augmentation.auto
   augmentation.base
   augmentation.container
   augmentation.module
