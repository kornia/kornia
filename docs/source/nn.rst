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

    nn.AugmentationSequential
    nn.ImageSequential
    nn.VideoSequential
    nn.PatchSequential


Augmentation Layers
-------------------


2D Augmentation Layers
++++++++++++++++++++++

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RandomHorizontalFlip
    nn.RandomVerticalFlip
    nn.CenterCrop
    nn.ColorJitter
    nn.RandomAffine
    nn.RandomBoxBlur
    nn.RandomChannelShuffle
    nn.RandomCrop
    nn.RandomElasticTransform
    nn.RandomEqualize
    nn.RandomErasing
    nn.RandomFisheye
    nn.RandomGaussianBlur
    nn.RandomGaussianNoise
    nn.RandomGrayscale
    nn.RandomHorizontalFlip
    nn.RandomInvert
    nn.RandomMotionBlur
    nn.RandomPerspective
    nn.RandomPosterize
    nn.RandomResizedCrop
    nn.RandomRotation
    nn.RandomSharpness
    nn.RandomSolarize
    nn.RandomThinPlateSpline
    nn.RandomVerticalFlip
    nn.PadTo
    nn.Normalize
    nn.Denormalize


3D Augmentation Layers
++++++++++++++++++++++

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RandomAffine3D
    nn.RandomRotation3D
    nn.RandomCrop3D
    nn.CenterCrop3D
    nn.RandomHorizontalFlip3D
    nn.RandomVerticalFlip3D
    nn.RandomDepthicalFlip3D
    nn.RandomEqualize3D
    nn.RandomMotionBlur3D
    nn.RandomPerspective3D
    nn.Normalize
    nn.Denormalize


Mix Augmentation Layers
++++++++++++++++++++++++

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RandomCutMix
    nn.RandomMixUp


Color Conversion Layers
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RgbToBgr
    nn.RgbToHls
    nn.RgbToHsv
    nn.RgbToLab
    nn.RgbToLinearRgb
    nn.RgbToLuv
    nn.RgbToRaw
    nn.RgbToRgba
    nn.RgbToXyz
    nn.RgbToYcbcr
    nn.RgbToYuv
    nn.RgbToYuv420
    nn.RgbToYuv422
    nn.RgbToGrayscale
    nn.RgbaToBgr
    nn.RgbaToRgb
    nn.BgrToRgb
    nn.BgrToRgba
    nn.BgrToGrayscale
    nn.GrayscaleToRgb
    nn.RawToRgb
    nn.HlsToRgb
    nn.HsvToRgb
    nn.LabToRgb
    nn.LinearRgbToRgb
    nn.LuvToRgb
    nn.YcbcrToRgb
    nn.Yuv420ToRgb
    nn.Yuv422ToRgb
    nn.YuvToRgb
    nn.XyzToRgb


Image Enhance Layers
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.AddWeighted
    nn.AdjustBrightness
    nn.AdjustContrast
    nn.AdjustGamma
    nn.AdjustHue
    nn.AdjustSaturation
    nn.Denormalize
    nn.Invert
    nn.Normalize
    nn.ZCAWhitening


Filtering Layers
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BlurPool2D
    nn.BoxBlur
    nn.Canny
    nn.GaussianBlur2d
    nn.Laplacian
    nn.MaxBlurPool2D
    nn.MedianBlur
    nn.MotionBlur
    nn.MotionBlur3D
    nn.Sobel
    nn.SpatialGradient
    nn.SpatialGradient3d
    nn.UnsharpMask
