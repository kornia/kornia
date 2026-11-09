Detectors
=========

.. currentmodule:: kornia.feature

Response functions score image locations; detectors select local maxima and return feature frames.
Use a response function when you need a dense score map, or a detector when you need sparse keypoints.

For an image ``(B, C, H, W)``, :func:`harris_response` and :func:`gftt_response` score corners;
:func:`hessian_response` and :func:`dog_response_single` score blobs. These functions return maps of the same
shape as the image, without non-maximum suppression. :func:`dog_response` instead takes a Gaussian scale stack
``(B, C, D, H, W)`` and returns differences between adjacent levels, ``(B, C, D - 1, H, W)``.

To use a response in a detector, choose its module form: :class:`CornerHarris`, :class:`CornerGFTT`,
:class:`BlobHessian`, :class:`BlobDoGSingle` or :class:`BlobDoG`. Pass it as ``resp_module`` to
:class:`ScaleSpaceDetector`, or as ``model`` to :class:`MultiResolutionDetector`.
:class:`BlobDoG` requires a scale stack, so use it only with :class:`ScaleSpaceDetector` and
``scale_space_response=True``.

The detectors return LAFs ``(B, N, 2, 3)`` in input pixels and responses ``(B, N)``, strongest first.
Unused slots contain zero LAFs and zero responses. Use :func:`laf_is_filled` to keep detected features and
discard padded slots; apply the same mask to LAFs and responses.

- :class:`ScaleSpaceDetector` takes a batch and refines maxima over a Gaussian scale space.
  Its default response is the Hessian determinant.
- :class:`MultiResolutionDetector` takes one image and finds maxima at integer pixels of each pyramid level,
  without scale selection or subpixel refinement.
- :class:`KeyNetDetector` uses :class:`MultiResolutionDetector` with the learned :class:`KeyNet` response.

For detection and description together, see :doc:`feature.local_features`.

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

:class:`ScaleSpaceDetector` applies border and refined-centre mask checks before each
octave's feature budget. Rejected candidates do not consume slots while weaker valid
detections remain; any slots
left after all valid candidates are exhausted contain zero responses and zero LAFs.

.. autoclass:: ScaleSpaceDetector
   :members: forward, detect

.. autoclass:: KeyNetDetector
   :members: forward
