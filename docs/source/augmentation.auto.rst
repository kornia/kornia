Automatic Augmentation Methods
==============================

.. currentmodule:: kornia.augmentation.auto

Augmentation Policy
-------------------

This module contains common data augmentation policies that can improve the accuracy of image classification models.

.. autoclass:: AutoAugment

   .. automethod:: get_transformation_matrix

   .. automethod:: forward_parameters

   .. automethod:: forward

   .. automethod:: inverse

.. autoclass:: RandomAugment

   .. automethod:: get_transformation_matrix

   .. automethod:: forward_parameters

   .. automethod:: forward

   .. automethod:: inverse

.. autoclass:: TrivialAugment

   .. automethod:: get_transformation_matrix

   .. automethod:: forward_parameters

   .. automethod:: forward

   .. automethod:: inverse

Augmentation Search Methods
---------------------------

WIP. This module contains common data augmentation search methods.
