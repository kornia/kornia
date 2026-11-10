Morphology
==========

.. currentmodule:: kornia.losses

The Hausdorff losses estimate the distance between the predicted and the labelled segmentation by eroding the error of
the pixels that thresholding the prediction at 0.5 gets wrong. They take probabilities, not logits: pass a softmax. :class:`HausdorffERLoss` takes images
``(B, C, H, W)`` and :class:`HausdorffERLoss3D` volumes ``(B, C, D, H, W)``, with an int64 label map of one channel.

.. autoclass:: HausdorffERLoss
.. autoclass:: HausdorffERLoss3D
