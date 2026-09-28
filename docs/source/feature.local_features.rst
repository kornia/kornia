Local features (detector and descriptor together)
=================================================

.. currentmodule:: kornia.feature

.. autoclass:: LocalFeature
   :members: forward

.. autoclass:: SOLD2_detector
   :members: forward

.. autoclass:: ALIKED
   :members: forward, from_pretrained, forward_laf

.. autoclass:: ALIKEDFeatures
   :undoc-members:
   :members: n, to

.. autoclass:: DeDoDe
   :members: forward, from_pretrained, describe, detect

.. autoclass:: DISK
   :members: forward, from_pretrained, heatmap_and_dense_descriptors

.. autoclass:: XFeat
   :members: forward, from_pretrained, detectAndCompute, detectAndComputeDense, match_xfeat, match_xfeat_star

.. autoclass:: XFeatModel
   :members: forward

.. autoclass:: InterpolateSparse2d
   :members: forward

.. autoclass:: DISKFeatures
   :undoc-members:
   :members: x, y, to

.. autoclass:: SIFTFeature
   :members: forward

The scale-space SIFT preset supports ``descriptor_backend="pyramid"`` for
specialized sparse extraction from the detector's Gaussian pyramid::

    feature = kornia.feature.SIFTFeatureScaleSpace(
        num_features=4096, descriptor_backend="pyramid"
    )
    lafs, responses, descriptors = feature(grayscale_image)

This path uses a dedicated DoG detector, with strict sparse extrema, iterative
subpixel refinement, and top-K ranking by absolute response. It applies neither
contrast nor edge rejection. Precise doubling and integer octave decimation keep
pixel coordinates aligned. The converged integer Gaussian layer and refined
continuous scale are retained separately. Gradients are computed once per used
layer and shared between orientation and description. It returns one descriptor per detection and the same fixed feature
budget as the default patch backend. Its sampling and Gaussian support differ
from patch extraction, so descriptors are not numerically interchangeable.
No pyramid is retained between calls. For this backend, ``compile_modules``
accepts ``scale_pyr`` and ``subpix`` (or ``True`` for both); sparse orientation
and description run eagerly. The default ``patch`` backend retains the existing
generic detector and its compilation options.

.. autoclass:: SIFTFeatureScaleSpace
   :members: forward

.. autoclass:: GFTTAffNetHardNet
   :members: forward

.. autoclass:: HesAffNetHardNet
   :members: forward

.. autoclass:: KeyNetAffNetHardNet
   :members: forward

.. autoclass:: KeyNetHardNet
   :members: forward
