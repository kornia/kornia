Detectors
=========

.. currentmodule:: kornia.feature

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
