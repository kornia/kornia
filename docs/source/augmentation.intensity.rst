2D intensity transforms
=======================

.. currentmodule:: kornia.augmentation

Transforms that change pixel values but keep every pixel where it is, so masks, boxes and keypoints pass through
unchanged. With default arguments, 24 of the 37 transforms on this page clamp their output to ``[0, 1]``. These 13
do not: the four blurs, ``RandomGaussianNoise``, ``RandomSaltAndPepperNoise``, ``RandomGrayscale``, ``RandomInvert``,
``RandomChannelShuffle``, ``RandomChannelDropout``, ``RandomRain``, ``Normalize`` and ``Denormalize``.

``ColorJiggle`` and ``ColorJitter`` both jitter brightness, contrast, saturation and hue; ``ColorJitter`` uses
torchvision's formulas. ``RandomEqualize`` equalizes whole channels, ``RandomClahe`` local tiles.

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
