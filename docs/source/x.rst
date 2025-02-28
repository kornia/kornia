.. meta::
   :name: description
   :content: "The Kornia.x module provides utilities for training Kornia models, including domain-specific trainers for image classification, semantic segmentation, and object detection. It also offers training callbacks like ModelCheckpoint and EarlyStopping."
kornia.x
========

.. currentmodule:: kornia.x

Package with the utilities to train kornia models.

.. autoclass:: Trainer


Domain trainers
---------------

.. autoclass:: ImageClassifierTrainer
.. autoclass:: SemanticSegmentationTrainer
.. autoclass:: ObjectDetectionTrainer


Callbacks
---------

.. autoclass:: ModelCheckpoint
.. autoclass:: EarlyStopping
