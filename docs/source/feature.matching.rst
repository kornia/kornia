Matching
========

.. currentmodule:: kornia.feature

The descriptor matchers take the descriptors of two images, and some of them their LAFs, and return
``(dists, idxs)``: a quality and an ``(index in desc1, index in desc2)`` pair per match, which
:func:`match_smnn_batched`, for padded batches of pairs, prefixes with the batch index. :func:`match_nn` keeps every
nearest neighbour, :func:`match_mnn` the mutual ones, :func:`match_snn` and :func:`match_smnn` add Lowe's ratio
test, and :func:`match_fginn` and :func:`match_adalam` also read the LAFs. :class:`DescriptorMatcher` and
:class:`GeometryAwareDescriptorMatcher` are their module forms, and :class:`LocalFeatureMatcher` runs a detector and
a descriptor on two images before a descriptor-only matcher.
The learned matchers are networks with pretrained weights: :class:`LightGlueMatcher`, :class:`LightGlue` and
:class:`OnnxLightGlue` match detected keypoints and their descriptors, and :class:`LoFTR` matches the images
themselves, without a detector. To keep the matches that one geometric model explains, pass them to
:class:`~kornia.geometry.ransac.RANSAC`.

Descriptor matching
-------------------

.. autofunction:: match_nn
.. autofunction:: match_mnn
.. autofunction:: match_snn
.. autofunction:: match_smnn
.. autofunction:: match_smnn_batched
.. autofunction:: match_fginn
.. autofunction:: match_adalam

.. autoclass:: DescriptorMatcher
   :members: forward

.. autoclass:: GeometryAwareDescriptorMatcher
   :members: forward

.. autoclass:: LocalFeatureMatcher
   :members: forward

Learned matchers
----------------

.. autoclass:: LightGlueMatcher
   :members: forward

.. autoclass:: LightGlue
   :members: forward

.. autoclass:: OnnxLightGlue
   :members: forward

.. autoclass:: LoFTR
   :members: forward, extract_features
