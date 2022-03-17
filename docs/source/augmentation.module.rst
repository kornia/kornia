.. currentmodule:: kornia.augmentation


Transforms2D
============

Set of operators to perform data augmentation on 2D image tensors.

.. autoclass:: CenterCrop
.. autoclass:: ColorJitter
.. autoclass:: RandomAffine
.. autoclass:: RandomBoxBlur
.. autoclass:: RandomCrop
.. autoclass:: RandomChannelShuffle
.. autoclass:: RandomCutMix
.. autoclass:: RandomErasing
.. autoclass:: RandomElasticTransform
.. autoclass:: RandomEqualize
.. autoclass:: RandomFisheye
.. autoclass:: RandomGrayscale
.. autoclass:: RandomGaussianBlur
.. autoclass:: RandomGaussianNoise
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomInvert
.. autoclass:: RandomMixUp
.. autoclass:: RandomMotionBlur
.. autoclass:: RandomPerspective
.. autoclass:: RandomPlanckianJitter
.. autoclass:: RandomPosterize
.. autoclass:: RandomResizedCrop
.. autoclass:: RandomRotation
.. autoclass:: RandomSharpness
.. autoclass:: RandomSolarize
.. autoclass:: RandomVerticalFlip
.. autoclass:: RandomThinPlateSpline


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


.. currentmodule:: kornia.augmentation

Normalizations
==============

Normalization operations are shape-agnostic for both 2D and 3D tensors.

.. autoclass:: Denormalize
.. autoclass:: Normalize

Image Resize
============

.. autoclass:: Resize
.. autoclass:: LongestMaxSize
.. autoclass:: SmallestMaxSize
