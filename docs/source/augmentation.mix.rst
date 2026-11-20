2D mix transforms
=================

.. currentmodule:: kornia.augmentation

Transforms that combine samples or rearrange patches within an image. Supported label, box and mask keys
vary by class; mixing does not imply support for every annotation type.

``RandomMixUpV2`` blends each image with a paired batch image, and ``RandomCutMixV2`` pastes in a rectangle from
it. ``RandomMosaic`` replaces an image by a crop of a mosaic of batch images; ``RandomJigsaw`` shuffles the tiles of
each image. ``RandomTransplantation`` copies the region of one mask label, together with its mask, from the previous
image in the batch.

.. autoclass:: PatchMix
.. autoclass:: RandomCutMixV2
.. autoclass:: RandomJigsaw
.. autoclass:: RandomMixUpV2
.. autoclass:: RandomMosaic
.. autoclass:: RandomTransplantation
.. autoclass:: TokenMix
