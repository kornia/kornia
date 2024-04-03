Image Augmentations
===================

.. currentmodule:: kornia.augmentation

Transforms2D
------------

Set of operators to perform data augmentation on 2D image tensors.

Intensity
~~~~~~~~~

.. autoclass:: ColorJiggle
.. autoclass:: ColorJitter
.. autoclass:: RandomAutoContrast
.. autoclass:: RandomBoxBlur
.. autoclass:: RandomBrightness
.. autoclass:: RandomChannelDropout
.. autoclass:: RandomChannelShuffle
.. autoclass:: RandomClahe
.. autoclass:: RandomContrast
.. autoclass:: RandomEqualize
.. autoclass:: RandomGamma
.. autoclass:: RandomGaussianBlur
.. autoclass:: RandomGaussianIllumination
.. autoclass:: RandomGaussianNoise
.. autoclass:: RandomGrayscale
.. autoclass:: RandomHue
.. autoclass:: RandomInvert
.. autoclass:: RandomJPEG
.. autoclass:: RandomLinearCornerIllumination
.. autoclass:: RandomLinearIllumination
.. autoclass:: RandomMedianBlur
.. autoclass:: RandomMotionBlur
.. autoclass:: RandomPlanckianJitter
.. autoclass:: RandomPlasmaBrightness
.. autoclass:: RandomPlasmaContrast
.. autoclass:: RandomPlasmaShadow
.. autoclass:: RandomPosterize
.. autoclass:: RandomRain
.. autoclass:: RandomRGBShift
.. autoclass:: RandomSaltAndPepperNoise
.. autoclass:: RandomSaturation
.. autoclass:: RandomSharpness
.. autoclass:: RandomSnow
.. autoclass:: RandomSolarize


Geometric
~~~~~~~~~

.. autoclass:: CenterCrop
.. autoclass:: PadTo
.. autoclass:: RandomAffine
.. autoclass:: RandomCrop
.. autoclass:: RandomElasticTransform
.. autoclass:: RandomErasing
.. autoclass:: RandomFisheye
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomPerspective
.. autoclass:: RandomResizedCrop
.. autoclass:: RandomRotation
.. autoclass:: RandomShear
.. autoclass:: RandomThinPlateSpline
.. autoclass:: RandomVerticalFlip


Mix
~~~

.. autoclass:: RandomCutMixV2
.. autoclass:: RandomJigsaw
.. autoclass:: RandomMixUpV2
.. autoclass:: RandomMosaic
.. autoclass:: RandomTransplantation

Transforms3D
------------

Set of operators to perform data augmentation on 3D volumetric tensors.

Geometric
~~~~~~~~~

.. autoclass:: CenterCrop3D
.. autoclass:: RandomAffine3D
.. autoclass:: RandomCrop3D
.. autoclass:: RandomDepthicalFlip3D
.. autoclass:: RandomHorizontalFlip3D
.. autoclass:: RandomRotation3D
.. autoclass:: RandomVerticalFlip3D

Intensity
~~~~~~~~~

.. autoclass:: RandomEqualize3D
.. autoclass:: RandomMotionBlur3D

Mix
~~~

.. autoclass:: RandomTransplantation3D

Normalizations
--------------

Normalization operations are shape-agnostic for both 2D and 3D tensors.

.. autoclass:: Denormalize
.. autoclass:: Normalize

Image Resize
------------

.. autoclass:: LongestMaxSize
.. autoclass:: Resize
.. autoclass:: SmallestMaxSize
