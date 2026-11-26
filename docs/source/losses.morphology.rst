Morphology
==========

.. currentmodule:: kornia.losses

:class:`HausdorffDTLoss` weights segmentation errors by distances computed from the target and the prediction
thresholded at 0.5. Use it alongside a region loss as an auxiliary term. It supports images and volumes, including
binary segmentation with a single sigmoid channel.

For binary segmentation, pass the model's logits to the region loss and the sigmoid probabilities to the distance loss.
Here ``logits`` and the int64 label map ``target`` both have shape ``(B, 1, H, W)``. Choose ``hd_weight`` on validation
data using both an overlap metric and a boundary metric.

.. code-block:: python

   import torch.nn.functional as F
   from kornia.losses import HausdorffDTLoss

   boundary_loss = HausdorffDTLoss()
   region = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))
   loss = region + hd_weight * boundary_loss(logits.sigmoid(), target)

:class:`HausdorffERLoss` and :class:`HausdorffERLoss3D` weight errors by their erosion depth for images and volumes,
respectively. All three losses take probabilities and a single-channel int64 label map.

.. autoclass:: HausdorffDTLoss
.. autoclass:: HausdorffERLoss
.. autoclass:: HausdorffERLoss3D
