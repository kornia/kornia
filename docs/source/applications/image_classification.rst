Image Classification
====================

.. image:: https://production-media.paperswithcode.com/thumbnails/task/task-0000000951-52325f45_O0tAMly.jpg
   :align: right
   :width: 20%

Image Classification is a fundamental task that attempts to comprehend an entire image as a whole.
The goal is to classify the image by assigning it to a specific label. Typically, Image Classification refers to images
in which only one object appears and is analyzed. In contrast, object detection involves both classification and
localization tasks, and is used to analyze more realistic cases in which multiple objects may exist in an image.

Learn more: `https://paperswithcode.com/task/image-classification <https://paperswithcode.com/task/image-classification>`_

Inference
---------

Kornia provides a couple of backbones based on `transformers <https://paperswithcode.com/methods/category/vision-transformer>`_
to perform image classification. Checkout the following apis :py:class:`~kornia.contrib.VisionTransformer`,
:py:class:`~kornia.contrib.ClassificationHead` and combine as follows to customize your own classifier:

.. code:: python

   import torch.nn as nn
   import kornia.contrib as K

   classifier = nn.Sequential(
      K.VisionTransformer(image_size=224, patch_size=16),
      K.ClassificationHead(num_classes=1000)
   )

   img = torch.rand(1, 3, 224, 224)
   out = classifier(img)     # BxN
   scores = out.argmax(-1)   # B

.. tip::
   Read more about our :ref:`kornia_vit`

Finetuning
----------

In order to customize your model with your own data you can use our :ref:`training_api` to perform the
`fine-tuning <https://paperswithcode.com/methods/category/fine-tuning>`_ of your model.

We provide :py:class:`~kornia.x.ImageClassifierTrainer` with a default training structure to train basic
image classification problems. However, one can leverage this is API using the models provided by Kornia or
use existing libraries from the PyTorch ecosystem such as `torchvision <https://pytorch.org/vision/stable/models.html>`_
or `timm <https://rwightman.github.io/pytorch-image-models/>`_.

.. literalinclude:: ../../../examples/train/image_classifier/main.py
   :language: python
   :lines: 20-46

Define your augmentations and callbacks:

.. literalinclude:: ../../../examples/train/image_classifier/main.py
   :language: python
   :lines: 49-66

Finally, instantiate the :py:class:`~kornia.x.ImageClassifierTrainer` and execute your training pipeline.

.. literalinclude:: ../../../examples/train/image_classifier/main.py
   :language: python
   :lines: 68-74

.. seealso::
   Play with the full example `here <https://github.com/kornia/kornia/tree/master/examples/train/image_classifier>`_
