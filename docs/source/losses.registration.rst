Registration
============

.. currentmodule:: kornia.losses

Similarity losses for image registration: the negative mutual information and the negative normalized mutual
information of two signals, estimated from a kernel (Parzen-window) joint histogram. The functions take flat signals,
2D images or 3D volumes; each module stores the target once and compares other signals with it. The Convention block
of :func:`mutual_information_loss` states the rules that all of them share.

.. _mutual-information-porting:

Porting from scikit-learn and scikit-image
------------------------------------------

The losses use the natural logarithm, as scikit-learn's ``mutual_info_score`` does.
:func:`normalized_mutual_information_loss` negates Studholme's :math:`\mathrm{NMI} = (H(X) + H(Y)) / H(X, Y)`, which
is scikit-image's ``normalized_mutual_information``. scikit-learn's ``normalized_mutual_info_score`` divides the mutual
information by the arithmetic mean of the two entropies, its default, and equals :math:`2 - 2 / \mathrm{NMI}`
for the same joint histogram with nonzero joint entropy. If both signals are constant, scikit-image returns NaN
(:math:`0 / 0`) and scikit-learn returns 1.0, while kornia's value is undefined.

Both libraries count a hard histogram. scikit-image's ``bins`` defaults to 100, kornia's ``num_bins`` to 64. With the
default kernel and ``window_radius``, kornia's histogram is a hard one when every value sits on a bin centre, as on
integer images whose values span ``0, ..., K`` (both ends present) with ``num_bins=K + 1``.

Functions
---------

.. autofunction:: mutual_information_loss
.. autofunction:: mutual_information_loss_2d
.. autofunction:: mutual_information_loss_3d
.. autofunction:: normalized_mutual_information_loss
.. autofunction:: normalized_mutual_information_loss_2d
.. autofunction:: normalized_mutual_information_loss_3d

Modules
-------

.. autoclass:: MILossFromRef
.. autoclass:: MILossFromRef2D
.. autoclass:: MILossFromRef3D
.. autoclass:: NMILossFromRef
.. autoclass:: NMILossFromRef2D
.. autoclass:: NMILossFromRef3D

Kernels
-------

.. autoclass:: MIKernel
