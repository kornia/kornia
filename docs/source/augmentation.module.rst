Image Augmentations
===================

.. currentmodule:: kornia.augmentation

Transforms2D
------------

Set of operators to perform data augmentation on 2D image tensors.

Intensity
~~~~~~~~~

.. autoclass:: RandomPlanckianJitter
.. autoclass:: RandomPlasmaShadow
.. autoclass:: RandomPlasmaBrightness
.. autoclass:: RandomPlasmaContrast
.. autoclass:: ColorJiggle
.. autoclass:: ColorJitter
.. autoclass:: RandomBoxBlur
.. autoclass:: RandomBrightness
.. autoclass:: RandomChannelShuffle
.. autoclass:: RandomContrast
.. autoclass:: RandomEqualize
.. autoclass:: RandomGamma
.. autoclass:: RandomGrayscale
.. autoclass:: RandomGaussianBlur
.. autoclass:: RandomGaussianNoise
.. autoclass:: RandomHue
.. autoclass:: RandomMotionBlur
.. autoclass:: RandomPosterize
.. autoclass:: RandomRGBShift
.. autoclass:: RnadomSaturation
.. autoclass:: RandomSharpness
.. autoclass:: RandomSolarize

Geometric
~~~~~~~~~

.. autoclass:: CenterCrop
.. autoclass:: RandomAffine
.. autoclass:: RandomCrop
.. autoclass:: RandomErasing
.. autoclass:: RandomElasticTransform
.. autoclass:: RandomFisheye
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomInvert
.. autoclass:: RandomPerspective
.. autoclass:: RandomResizedCrop
.. autoclass:: RandomRotation
.. autoclass:: RandomVerticalFlip
.. autoclass:: RandomThinPlateSpline

Mix
~~~

.. autoclass:: RandomCutMixV2
.. autoclass:: RandomMixUpV2
.. autoclass:: RandomMosaic
.. autoclass:: RandomJigsaw

Transforms3D
------------

Set of operators to perform data augmentation on 3D volumetric tensors.

Geometric
~~~~~~~~~

.. autoclass:: RandomDepthicalFlip3D
.. autoclass:: RandomHorizontalFlip3D
.. autoclass:: RandomVerticalFlip3D
.. autoclass:: RandomRotation3D
.. autoclass:: RandomAffine3D
.. autoclass:: RandomCrop3D
.. autoclass:: CenterCrop3D

Intensity
~~~~~~~~~

.. autoclass:: RandomMotionBlur3D
.. autoclass:: RandomEqualize3D

Normalizations
--------------

Normalization operations are shape-agnostic for both 2D and 3D tensors.

.. autoclass:: Denormalize
.. autoclass:: Normalize

Image Resize
------------

.. autoclass:: Resize
.. autoclass:: LongestMaxSize
.. autoclass:: SmallestMaxSize
