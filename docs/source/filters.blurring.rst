Blurring
========

.. currentmodule:: kornia.filters

Blur, sharpen or downsample images. Every function takes a ``(B, C, H, W)`` image (:func:`motion_blur3d` a
``(B, C, D, H, W)`` volume) and returns the same size, except :func:`blur_pool2d` and :func:`max_blur_pool2d`, which
downsample.

Which function to use:

- Average a window, equally with :func:`box_blur` or with Gaussian weights with :func:`gaussian_blur2d`. Both run as
  two 1D passes by default (``separable=True``).
- Keep edges. :func:`median_blur` takes the median of the window, which removes outliers such as salt-and-pepper
  noise. :func:`bilateral_blur` weights each neighbour by its distance and colour difference.
  :func:`joint_bilateral_blur` and :func:`guided_blur` take the edges from a second, guidance image.
- Blur along a line, as motion does: :func:`motion_blur` for images and :func:`motion_blur3d` for volumes.
- Sharpen by adding back what a Gaussian blur removes: :func:`unsharp_mask`.
- Downsample a feature map with less aliasing: :func:`blur_pool2d`, or :func:`max_blur_pool2d` to take a local maximum
  first. :func:`edge_aware_blur_pool2d` blurs at the input size and keeps the pixels on edges.

Each function has a module form that stores its arguments.

Functions
---------

.. autofunction:: bilateral_blur
.. autofunction:: blur_pool2d
.. autofunction:: box_blur
.. autofunction:: edge_aware_blur_pool2d
.. autofunction:: gaussian_blur2d
.. autofunction:: guided_blur
.. autofunction:: joint_bilateral_blur
.. autofunction:: max_blur_pool2d
.. autofunction:: median_blur
.. autofunction:: motion_blur
.. autofunction:: motion_blur3d
.. autofunction:: unsharp_mask

Modules
-------

.. autoclass:: BilateralBlur
.. autoclass:: BlurPool2D
.. autoclass:: BoxBlur
.. autoclass:: EdgeAwareBlurPool2D
.. autoclass:: MaxBlurPool2D
.. autoclass:: MedianBlur
.. autoclass:: GaussianBlur2d
.. autoclass:: GuidedBlur
.. autoclass:: JointBilateralBlur
.. autoclass:: MotionBlur
.. autoclass:: MotionBlur3D
.. autoclass:: UnsharpMask

Dissolving
----------

.. autoclass:: StableDiffusionDissolving
