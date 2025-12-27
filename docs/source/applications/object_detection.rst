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

In order to customize your model with your own data, you can use the models provided by Kornia or use existing libraries from the PyTorch ecosystem such
as `torchvision <https://pytorch.org/vision/stable/models.html>`_.

You can use standard PyTorch training loops with Kornia models and augmentations.
