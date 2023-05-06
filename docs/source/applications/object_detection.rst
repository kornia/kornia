Object detection
================

.. image:: https://production-media.paperswithcode.com/thumbnails/task/task-0000000004-7757802e.jpg
   :align: right
   :width: 40%

Object detection consists in detecting objects belonging to a certain category from an image,
determining the absolute location and also assigning each detected instance a predefined category.
In the last few years, several models have emerged based on deep learning. Being the state of the art
models being based on two stages. First, regions with higher recall values are located, so that all
objects in the image adhere to the proposed regions. The second stage consists of classification models,
usually CNNs, used to determine the category of each proposed region (instances).

Learn more: `https://paperswithcode.com/task/object-detection <https://paperswithcode.com/task/object-detection>`_

Finetuning
----------

In order to customize your model with your own data you can use our :ref:`training_api` to perform the
`fine-tuning <https://paperswithcode.com/methods/category/fine-tuning>`_ of your model.

We provide :py:class:`~kornia.x.ObjectDetectionTrainer`
with a default training structure to train object detection problems. However, one can leverage this is
API using the models provided by Kornia or use existing libraries from the PyTorch ecosystem such
as `torchvision <https://pytorch.org/vision/stable/models.html>`_.

Create the dataloaders and transforms:

.. literalinclude:: ../_static/scripts/object_detection.py
   :language: python
   :lines: 17-39

Define your model, losses, optimizers and schedulers:

.. literalinclude:: ../_static/scripts/object_detection.py
   :language: python
   :lines: 40-50

Create your preprocessing and augmentations pipeline:

.. literalinclude:: ../_static/scripts/object_detection.py
   :language: python
   :lines: 50-90

Finally, instantiate the :py:class:`~kornia.x.ObjectDetectionTrainer`
and execute your training pipeline.

.. literalinclude:: ../_static/scripts/object_detection.py
   :language: python
   :lines: 90-111

.. seealso::
   Play with the full example `here <https://github.com/kornia/kornia/tree/master/examples/train/object_detection>`_
