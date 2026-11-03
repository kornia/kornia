2D mix transforms
=================

.. currentmodule:: kornia.augmentation

Transforms that combine samples or rearrange patches within an image. Supported label, box and mask keys
vary by class; mixing does not imply support for every annotation type.

Use ``RandomMixUpV2`` to blend complete paired images and ``RandomCutMixV2`` to replace a sampled rectangle with its
paired image. ``RandomMosaic`` constructs a crop from several batch images, whereas ``RandomJigsaw`` rearranges tiles
within each image. ``RandomTransplantation`` requires image masks and copies a labelled region with its mask.

.. autoclass:: PatchMix
.. autoclass:: RandomCutMixV2
.. autoclass:: RandomJigsaw
.. autoclass:: RandomMixUpV2
.. autoclass:: RandomMosaic
.. autoclass:: RandomTransplantation
