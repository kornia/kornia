kornia.filters
==============

.. currentmodule:: kornia.filters

The functions in this sections perform various image filtering operations.

Blurring
--------

.. autofunction:: bilateral_blur
.. autofunction:: blur_pool2d
.. autofunction:: box_blur
.. autofunction:: gaussian_blur2d
.. autofunction:: guided_blur
.. autofunction:: joint_bilateral_blur
.. autofunction:: max_blur_pool2d
.. autofunction:: median_blur
.. autofunction:: motion_blur
.. autofunction:: unsharp_mask

Interactive Demo
~~~~~~~~~~~~~~~~
.. raw:: html

    <gradio-app space="kornia/kornia-image-filtering"></gradio-app>

Visit the `Kornia image filtering demo on the Hugging Face Spaces
<https://huggingface.co/spaces/kornia/kornia-image-filtering>`_.

Edge detection
--------------

.. autofunction:: canny
.. autofunction:: laplacian
.. autofunction:: sobel
.. autofunction:: spatial_gradient
.. autofunction:: spatial_gradient3d


.. autoclass:: Laplacian
.. autoclass:: Sobel
.. autoclass:: Canny
.. autoclass:: SpatialGradient
.. autoclass:: SpatialGradient3d
.. autoclass:: DexiNed


Interactive Demo
~~~~~~~~~~~~~~~~
.. raw:: html

    <gradio-app space="kornia/edge_detector"></gradio-app>

Visit the `Kornia edge detector demo on the Hugging Face Spaces
<https://huggingface.co/spaces/kornia/edge_detector>`_.


Filtering API
-------------

.. autofunction:: filter2d
.. autofunction:: filter2d_separable
.. autofunction:: filter3d

Kernels
-------

.. autofunction:: get_gaussian_kernel1d
.. autofunction:: get_gaussian_erf_kernel1d
.. autofunction:: get_gaussian_discrete_kernel1d
.. autofunction:: get_gaussian_kernel2d
.. autofunction:: get_hanning_kernel1d
.. autofunction:: get_hanning_kernel2d
.. autofunction:: get_laplacian_kernel1d
.. autofunction:: get_laplacian_kernel2d
.. autofunction:: get_motion_kernel2d

Module
------

.. autoclass:: BilateralBlur
.. autoclass:: BlurPool2D
.. autoclass:: BoxBlur
.. autoclass:: MaxBlurPool2D
.. autoclass:: MedianBlur
.. autoclass:: GaussianBlur2d
.. autoclass:: GuidedBlur
.. autoclass:: JointBilateralBlur
.. autoclass:: MotionBlur
.. autoclass:: UnsharpMask
