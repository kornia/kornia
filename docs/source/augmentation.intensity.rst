2D intensity transforms
=======================

.. currentmodule:: kornia.augmentation

Transforms that change pixel values but keep every pixel where it is, so masks, boxes and keypoints pass through
unchanged.

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
.. autoclass:: RandomDissolving
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

Normalization
-------------

Deterministic normalization operators, shape-agnostic for 2D and 3D tensors.

.. autoclass:: Denormalize
.. autoclass:: Normalize
