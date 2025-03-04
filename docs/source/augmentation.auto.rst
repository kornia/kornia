Automatic Augmentation Methods
==============================

.. meta::
   :name: description
   :content: "The Automatic Augmentation Methods module in Kornia provides common data augmentation policies like AutoAugment, RandAugment, and TrivialAugment to improve the accuracy of image classification models. It also includes methods for augmentation search."

.. currentmodule:: kornia.augmentation.auto

Augmentation Policy
-------------------

This module contains common data augmentation policies that can improve the accuracy of image classification models.

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

Augmentation Search Methods
---------------------------

WIP. This module contains common data augmentation search methods.
