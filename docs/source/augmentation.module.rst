.. currentmodule:: kornia.augmentation


Transforms2D
============

Set of operators to perform data augmentation on 2D image tensors.

.. autoclass:: CenterCrop
.. autoclass:: ColorJitter
.. autoclass:: GaussianBlur
.. autoclass:: RandomAffine
.. autoclass:: RandomCrop
.. autoclass:: RandomErasing
.. autoclass:: RandomGrayscale
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomVerticalFlip
.. autoclass:: RandomMotionBlur
.. autoclass:: RandomPerspective
.. autoclass:: RandomResizedCrop
.. autoclass:: RandomRotation
.. autoclass:: RandomSolarize
.. autoclass:: RandomPosterize
.. autoclass:: RandomSharpness
.. autoclass:: RandomEqualize
.. autoclass:: RandomMixUp
.. autoclass:: RandomCutMix
.. autoclass:: RandomInvert

.. automodule:: kornia.augmentation.functional.functional
    :members:


.. currentmodule:: kornia.augmentation

Transforms3D
============

Set of operators to perform data augmentation on 3D volumetric tensors.

.. autoclass:: RandomDepthicalFlip3D
.. autoclass:: RandomHorizontalFlip3D
.. autoclass:: RandomVerticalFlip3D
.. autoclass:: RandomRotation3D
.. autoclass:: RandomAffine3D
.. autoclass:: RandomCrop3D
.. autoclass:: CenterCrop3D
.. autoclass:: RandomMotionBlur3D
.. autoclass:: RandomEqualize3D

.. automodule:: kornia.augmentation.functional.functional3d
    :members:


.. currentmodule:: kornia.augmentation

Normalizations
==============

Normalization operations are shape-agnostic for both 2D and 3D tensors.

.. autoclass:: Denormalize
.. autoclass:: Normalize
