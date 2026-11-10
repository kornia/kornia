Automatic Augmentation Methods
==============================

.. meta::
   :description: The Automatic Augmentation Methods module in Kornia provides common data augmentation policies like AutoAugment, RandAugment, and TrivialAugment to improve the accuracy of image classification models. It also includes methods for augmentation search.

.. currentmodule:: kornia.augmentation.auto

Augmentation Policy
-------------------

The three policies differ in how they pick operations:

- :class:`AutoAugment` picks one sub-policy from a searched dataset preset, or from a list you supply.
- :class:`RandAugment` picks ``n`` distinct operations and sets every magnitude from ``m``.
- :class:`TrivialAugment` picks one operation and draws its magnitude, if it has one, per image.

One call picks one operation sequence for the whole batch; each operation then applies its own probability and
parameters. To chain transforms of your own choice, use :class:`~kornia.augmentation.container.ImageSequential`.

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

The operations the policies above sample. Use them directly to define a custom policy.

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
