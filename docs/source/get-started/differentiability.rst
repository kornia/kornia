Differentiability
=================

.. meta::
   :description: Use Kornia's differentiable color conversions, filters, warps and losses in training loops, and distinguish their gradients from discrete outputs such as thresholded edge maps.

Kornia provides differentiable PyTorch operations for color conversions, filtering, geometric warps,
augmentations and losses. These operations can sit *inside* a model or a loss function, not only in the data
loader. Differentiability depends on the selected operation, its inputs and which output you use; returning a
floating-point tensor does not by itself establish a gradient path.

That enables three families of use cases:

**Optimize an image directly.** Treat the image as the parameter. This is total-variation denoising in a few
lines (see the :doc:`denoising guide </applications/image_denoising>`):

.. code-block:: python

   import torch
   import kornia

   noisy = torch.rand(1, 3, 64, 64)
   estimate = noisy.clone().requires_grad_(True)
   optimizer = torch.optim.Adam([estimate], lr=0.01)

   for _ in range(50):
       optimizer.zero_grad()
       loss = torch.nn.functional.mse_loss(estimate, noisy) \
            + 1e-4 * kornia.losses.total_variation(estimate).mean()
       loss.backward()
       optimizer.step()

**Optimize transformation parameters.** Gradients flow through :func:`~kornia.geometry.transform.warp_perspective`
with respect to the homography, which is how :class:`~kornia.geometry.transform.ImageRegistrator` aligns two
images by direct gradient descent (see the :doc:`registration guide </applications/image_registration>`):

.. code-block:: python

   import torch
   from kornia.geometry import ImageRegistrator

   img_src = torch.rand(1, 1, 32, 32)
   img_dst = torch.rand(1, 1, 32, 32)
   homo = ImageRegistrator("similarity").register(img_src, img_dst)

**Train through vision ops.** Structural losses such as :func:`~kornia.losses.ssim_loss` backpropagate to the
network that produced the image. Differentiable filters and descriptors can also form part of a training
objective; use outputs that retain an autograd path to the quantity you want to optimize.

.. code-block:: python

   import torch
   import kornia

   prediction = torch.rand(2, 3, 64, 64, requires_grad=True)  # imagine a network output
   target = torch.rand(2, 3, 64, 64)
   loss = kornia.losses.ssim_loss(prediction, target, window_size=5)
   loss.backward()
   assert prediction.grad is not None

Discrete outputs and gradient paths
-----------------------------------

Thresholding, comparisons and index selection make discrete decisions. For example,
:func:`~kornia.filters.canny` returns a suppressed gradient magnitude and an edge map. The magnitude retains
an autograd path through the selected responses, but the suppression decisions are not differentiated.
The thresholded edge map has no autograd path to the image, even with ``hysteresis=False``.
For an edge-based training objective, :func:`~kornia.filters.sobel` provides a gradient magnitude without
Canny's suppression and thresholding:

.. code-block:: python

   import torch
   import kornia

   image = torch.linspace(0, 1, 64 * 64).reshape(1, 1, 64, 64).requires_grad_()
   magnitude, edges = kornia.filters.canny(image, hysteresis=False)
   assert magnitude.requires_grad
   assert not edges.requires_grad

   kornia.filters.sobel(image).mean().backward()
   assert image.grad is not None
   assert torch.isfinite(image.grad).all()

Setting ``requires_grad_()`` on a detached edge map does not reconnect it to the image. Also keep the
computation out of ``torch.no_grad()`` and ``torch.inference_mode()`` when gradients are needed.
Gradient checks with ``torch.autograd.gradcheck`` exercise differentiable paths in the test suite; they do
not imply that every output of a multi-output function is differentiable.
