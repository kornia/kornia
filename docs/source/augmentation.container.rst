kornia.augmentation.container
=============================

.. currentmodule:: kornia.augmentation.container

The classes in this section are containers for augmenting different data formats (e.g. videos).


Video Data Augmentation
-----------------------

.. autoclass:: VideoSequential

   .. automethod:: forward


Video data is a special case of 3D volumetric data that contains both spatial and temporal information, which can be referred as 2.5D than 3D.
In most applications, augmenting video data requires a static temporal dimension to have the same augmentations are performed for each frame.
Thus, `VideoSequential` can be used to do such trick as same as `nn.Sequential`.
Currently, `VideoSequential` supports data format like :math:`(B, C, T, H, W)` and :math:`(B, T, C, H, W)`.

.. code-block:: python

   import kornia.augmentation as K

   transform = K.VideoSequential(
      K.RandomAffine(360),
      K.ColorJitter(0.2, 0.3, 0.2, 0.3),
      data_format="BCTHW",
      same_on_frame=True
   )
