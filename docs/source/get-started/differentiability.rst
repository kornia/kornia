Differentiability
=================

.. meta::
   :description: Every Kornia operator is differentiable: gradients flow through color conversions, filters, warps and augmentations, enabling image optimization, spatial transformer style networks and self-supervised losses.

Kornia's defining feature is that every operator is written with differentiable PyTorch ops, so autograd flows
through the whole library: color conversions, filters, geometric warps, augmentations and losses. Classical
vision operations can therefore sit *inside* a model or a loss function, not only in the data loader.

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
network that produced the image; differentiable augmentations make techniques like AutoAugment-style policy
search possible; and edge detectors or descriptors can be fine-tuned end to end.

.. code-block:: python

   import torch
   import kornia

   prediction = torch.rand(2, 3, 64, 64, requires_grad=True)  # imagine a network output
   target = torch.rand(2, 3, 64, 64)
   loss = kornia.losses.ssim_loss(prediction, target, window_size=5)
   loss.backward()
   assert prediction.grad is not None

Gradient correctness is enforced in CI with ``torch.autograd.gradcheck`` across the test suite.
