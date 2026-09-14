2D mix transforms
=================

.. currentmodule:: kornia.augmentation

Transforms that combine samples or rearrange patches within an image. Supported label, box and mask keys
vary by class; mixing does not imply support for every annotation type.

.. autoclass:: PatchMix
.. autoclass:: RandomCutMixV2
.. autoclass:: RandomJigsaw
.. autoclass:: RandomMixUpV2
.. autoclass:: RandomMosaic
.. autoclass:: RandomTransplantation
