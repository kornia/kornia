Semantic segmentation
=====================

.. image:: https://production-media.paperswithcode.com/thumbnails/task/task-0000000885-bec5f079_K84qLCL.jpg
   :align: right
   :width: 40%

Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the
same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a
category. Some example benchmarks for this task are Cityscapes, PASCAL VOC and ADE20K. Models are usually evaluated with
the Mean Intersection-Over-Union (Mean IoU) and Pixel Accuracy metrics.

Learn more: `https://paperswithcode.com/task/semantic-segmentation <https://paperswithcode.com/task/semantic-segmentation>`_

Finetuning
----------

In order to customize your model with your own data you can use our :ref:`training_api` to perform the
`fine-tuning <https://paperswithcode.com/methods/category/fine-tuning>`_ of your model.

We provide :py:class:`~kornia.x.SemanticSegmentationTrainer` with a default training structure to train semantic
segmentation problems. However, one can leverage this is API using the models provided by Kornia or
use existing libraries from the PyTorch ecosystem such as `torchvision <https://pytorch.org/vision/stable/models.html>`_.

Create the dataloaders and transforms:

.. literalinclude:: ../../../examples/train/semantic_segmentation/main.py
   :language: python
   :lines: 21-47

Define your model, losses, optimizers and schedulers:

.. literalinclude:: ../../../examples/train/semantic_segmentation/main.py
   :language: python
   :lines: 49-61

Create your preprocessing and augmentations pipeline:

.. literalinclude:: ../../../examples/train/semantic_segmentation/main.py
   :language: python
   :lines: 63-82

Finally, instantiate the :py:class:`~kornia.x.SemanticSegmentationTrainer` and execute your training pipeline.

.. literalinclude:: ../../../examples/train/semantic_segmentation/main.py
   :language: python
   :lines: 101-111

.. seealso::
   Play with the full example `here <https://github.com/kornia/kornia/tree/master/examples/train/semantic_segmentation>`_
