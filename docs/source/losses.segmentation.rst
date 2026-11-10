Semantic segmentation
=====================

.. currentmodule:: kornia.losses

The multi-class losses take class logits ``(B, C, H, W)`` and a label map ``(B, H, W)``, and apply the softmax
themselves. :func:`dice_loss` and :func:`tversky_loss` optimise the class overlap, :func:`lovasz_softmax_loss` the IoU
of every class, and :func:`lovasz_hinge_loss` the IoU of one foreground, from one logit channel.
:func:`focal_loss` is a cross entropy over logits ``(B, C, *)`` that down-weights easy pixels, and
:func:`binary_focal_loss_with_logits` is the same for independent sigmoid outputs, against targets of their shape.
:func:`one_hot` turns a label map into a one-hot tensor.

Functions
---------

.. autofunction:: one_hot
.. autofunction:: binary_focal_loss_with_logits
.. autofunction:: focal_loss
.. autofunction:: dice_loss
.. autofunction:: tversky_loss
.. autofunction:: lovasz_hinge_loss
.. autofunction:: lovasz_softmax_loss

Modules
-------

.. autoclass:: BinaryFocalLossWithLogits
.. autoclass:: DiceLoss
.. autoclass:: TverskyLoss
.. autoclass:: FocalLoss
.. autoclass:: LovaszHingeLoss
.. autoclass:: LovaszSoftmaxLoss
