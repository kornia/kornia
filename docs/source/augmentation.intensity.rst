2D intensity transforms
=======================

.. currentmodule:: kornia.augmentation

Choose an intensity transform when pixel positions must stay fixed. The classes take image tensors of shape
``(B, C, H, W)`` (with some classes accepting an unbatched image), and their range and channel requirements
vary: color effects commonly clamp selected inputs to ``[0, 1]``, whereas filters and noise may retain arbitrary
values. Their geometric matrix is the identity, so masks, boxes and keypoints pass through unless a class documents
a specialized handler. Use a geometric augmentation to move pixels or change annotation coordinates.

Use ``ColorJiggle`` for Kornia's additive brightness and HSV color adjustments, or ``ColorJitter`` for
torchvision-compatible formulas. ``RandomEqualize`` equalizes whole channels; ``RandomClahe`` does so in local
tiles. ``Normalize`` and ``Denormalize`` use supplied channel statistics; they are not range-clamping transforms.

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
