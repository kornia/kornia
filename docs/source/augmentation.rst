kornia.augmentation
-------------------

.. currentmodule:: kornia.augmentation

The classes in this section perform various data augmentation operations.

Kornia provides Torchvision-like augmentation APIs while may not reproduce Torchvision, because Kornia is a library aligns to OpenCV functionalities, not PIL. Besides, pure floating computation is used in Kornia which gaurentees a better precision without any float -> uint8 conversions. To be specified, the different functions are:

- AdjustContrast
- AdjustBrightness
- RandomRectangleErasing

For detailed comparision, please checkout the [Colab: Kornia vs. Torchvision](https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS).

.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomVerticalFlip
.. autoclass:: RandomRectangleErasing
.. autoclass:: RandomGrayscale
.. autoclass:: RandomAffine
.. autoclass:: RandomPerspective
.. autoclass:: RandomRotation
.. autoclass:: ColorJitter
.. autoclass:: CenterCrop
.. autoclass:: RandomCrop
.. autoclass:: RandomResizedCrop
.. autoclass:: AdjustBrightness
.. autoclass:: AdjustContrast
.. autoclass:: AdjustHue
.. autoclass:: AdjustGamma
.. autoclass:: Normalize
.. autoclass:: Denormalize

.. automodule:: kornia.augmentation.functional
    :members:
