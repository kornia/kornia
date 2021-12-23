kornia.nn
======================

These are the basic building blocks for graphs.


.. currentmodule:: kornia


Containers
----------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    kornia.nn.AugmentationSequential
    kornia.nn.ImageSequential
    kornia.nn.VideoSequential
    kornia.nn.PatchSequential


Augmentation Layers
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    kornia.nn.RandomHorizontalFlip
    kornia.nn.RandomVerticalFlip
    kornia.nn.CenterCrop
    kornia.nn.ColorJitter
    kornia.nn.RandomAffine
    kornia.nn.RandomBoxBlur
    kornia.nn.RandomChannelShuffle
    kornia.nn.RandomCrop
    kornia.nn.RandomElasticTransform
    kornia.nn.RandomEqualize
    kornia.nn.RandomErasing
    kornia.nn.RandomFisheye
    kornia.nn.RandomGaussianBlur
    kornia.nn.RandomGaussianNoise
    kornia.nn.RandomGrayscale
    kornia.nn.RandomHorizontalFlip
    kornia.nn.RandomInvert
    kornia.nn.RandomMotionBlur
    kornia.nn.RandomPerspective
    kornia.nn.RandomPosterize
    kornia.nn.RandomResizedCrop
    kornia.nn.RandomRotation
    kornia.nn.RandomSharpness
    kornia.nn.RandomSolarize
    kornia.nn.RandomThinPlateSpline
    kornia.nn.RandomVerticalFlip
    kornia.nn.PadTo
    kornia.nn.Normalize
    kornia.nn.Denormalize


3D Augmentation Layers
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    kornia.nn.RandomAffine3D
    kornia.nn.RandomRotation3D
    kornia.nn.RandomCrop3D
    kornia.nn.CenterCrop3D
    kornia.nn.RandomHorizontalFlip3D
    kornia.nn.RandomVerticalFlip3D
    kornia.nn.RandomDepthicalFlip3D
    kornia.nn.RandomEqualize3D
    kornia.nn.RandomMotionBlur3D
    kornia.nn.RandomPerspective3D
    kornia.nn.Normalize
    kornia.nn.Denormalize


Mix Augmentation Layers
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    kornia.nn.RandomCutMix
    kornia.nn.RandomMixUp


Color Conversion Layers
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    kornia.nn.RgbToBgr
    kornia.nn.RgbToHls
    kornia.nn.RgbToHsv
    kornia.nn.RgbToLab
    kornia.nn.RgbToLinearRgb
    kornia.nn.RgbToLuv
    kornia.nn.RgbToRaw
    kornia.nn.RgbToRgba
    kornia.nn.RgbToXyz
    kornia.nn.RgbToYcbcr
    kornia.nn.RgbToYuv
    kornia.nn.RgbToYuv420
    kornia.nn.RgbToYuv422
    kornia.nn.RgbToGrayscale
    kornia.nn.RgbaToBgr
    kornia.nn.RgbaToRgb
    kornia.nn.BgrToRgb
    kornia.nn.BgrToRgba
    kornia.nn.BgrToGrayscale
    kornia.nn.GrayscaleToRgb
    kornia.nn.RawToRgb
    kornia.nn.HlsToRgb
    kornia.nn.HsvToRgb
    kornia.nn.LabToRgb
    kornia.nn.LinearRgbToRgb
    kornia.nn.LuvToRgb
    kornia.nn.YcbcrToRgb
    kornia.nn.Yuv420ToRgb
    kornia.nn.Yuv422ToRgb
    kornia.nn.YuvToRgb
    kornia.nn.XyzToRgb
