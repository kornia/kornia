Automatic Augmentation Methods
==============================

.. meta::
   :description: The Automatic Augmentation Methods module in Kornia provides common data augmentation policies like AutoAugment, RandAugment, and TrivialAugment to improve the accuracy of image classification models. It also includes methods for augmentation search.

.. currentmodule:: kornia.augmentation.auto

Augmentation Policy
-------------------

Choose a policy by how it selects operations:

- :class:`AutoAugment` selects a sub-policy from a searched dataset preset or a supplied list.
- :class:`RandAugment` selects ``n`` operations without replacement and sets their magnitudes using ``m``.
- :class:`TrivialAugment` selects one operation and samples its magnitude per image, if it has one.

Each call selects the operation sequence for the batch. Individual operations then use their own probability
gates and parameters. The policies accept floating image batches ``(B, C, H, W)``; channel and range requirements
come from the operations in the chosen policy. Use :class:`~kornia.augmentation.container.ImageSequential` for
a sequence of transforms you choose explicitly.

.. autoclass:: AutoAugment

   .. automethod:: get_transformation_matrix

   .. automethod:: forward_parameters

   .. automethod:: forward

   .. automethod:: inverse

.. autoclass:: RandAugment

   .. automethod:: get_transformation_matrix

   .. automethod:: forward_parameters

   .. automethod:: forward

   .. automethod:: inverse

.. autoclass:: TrivialAugment

   .. automethod:: get_transformation_matrix

   .. automethod:: forward_parameters

   .. automethod:: forward

   .. automethod:: inverse

.. autoclass:: PolicyAugmentBase

.. autoclass:: PolicySequential

.. autoclass:: kornia.augmentation.auto.operations.OperationBase

   .. automethod:: forward_parameters

   .. automethod:: forward

Policy operations
-----------------

The following building blocks are primarily useful when defining a custom policy. The policy classes above choose and
sequence them for the supplied AutoAugment, RandAugment and TrivialAugment policies.

.. currentmodule:: kornia.augmentation.auto.operations.ops

.. autoclass:: AutoContrast
.. autoclass:: Brightness
.. autoclass:: Contrast
.. autoclass:: Equalize
.. autoclass:: Gray
.. autoclass:: HorizontalFlip
.. autoclass:: Hue
.. autoclass:: Invert
.. autoclass:: Posterize
.. autoclass:: Rotate
.. autoclass:: Saturate
.. autoclass:: Sharpness
.. autoclass:: ShearX
.. autoclass:: ShearY
.. autoclass:: Solarize
.. autoclass:: SolarizeAdd
.. autoclass:: TranslateX
.. autoclass:: TranslateY
.. autoclass:: VerticalFlip

Augmentation Search Methods
---------------------------

WIP. This module contains common data augmentation search methods.
