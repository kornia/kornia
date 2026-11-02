Detectors
=========

.. currentmodule:: kornia.feature

The response functions score every pixel and return a map of the input's size, ``(B, C, H, W)``, without non-maxima
suppression: :func:`harris_response` and :func:`gftt_response` for corners, :func:`hessian_response` and
:func:`dog_response_single` for blobs, and :func:`dog_response` for the levels of a Gaussian scale stack
``(B, C, D, H, W)``, which returns ``(B, C, D - 1, H, W)``. Each has a module form, which fixes the arguments other
than the image and ``sigmas`` at construction, to pass to a detector as the ``resp_module`` of
:class:`ScaleSpaceDetector` or the ``model`` of :class:`MultiResolutionDetector`: :class:`CornerHarris`,
:class:`CornerGFTT`, :class:`BlobHessian`, :class:`BlobDoGSingle`, and :class:`BlobDoG`, which takes a scale stack
and so serves only as a ``resp_module`` with ``scale_space_response=True``.

The detectors return local affine frames (LAFs), ``(B, N, 2, 3)`` in input pixels, with their responses ``(B, N)``,
strongest first, and pad the slots no detection filled with a zero LAF and a zero response, which
:func:`laf_is_filled` finds:

- :class:`ScaleSpaceDetector` takes a batch and finds the sub-pixel maxima of a response over a Gaussian scale space,
  the Hessian determinant by default.
- :class:`MultiResolutionDetector` takes one image and finds the maxima of a response at the integer pixels of each
  level of an image pyramid, without scale selection or a sub-pixel step.
- :class:`KeyNetDetector` is that detector on the learned :class:`KeyNet` response.

For a detector together with a descriptor, see :doc:`feature.local_features`.

Response functions
------------------

.. autofunction:: gftt_response
.. autofunction:: harris_response
.. autofunction:: hessian_response
.. autofunction:: dog_response
.. autofunction:: dog_response_single

.. autoclass:: BlobHessian
.. autoclass:: CornerGFTT
.. autoclass:: CornerHarris
.. autoclass:: BlobDoG
.. autoclass:: BlobDoGSingle

Detectors
---------

.. autoclass:: KeyNet
.. autoclass:: MultiResolutionDetector
   :members: forward, remove_borders, detect_features_on_single_level, detect

.. autoclass:: ScaleSpaceDetector
   :members: forward, detect

.. autoclass:: KeyNetDetector
   :members: forward
