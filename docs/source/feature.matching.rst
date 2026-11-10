Matching
========

.. currentmodule:: kornia.feature

Descriptor matchers return ``(dists, idxs)``: one value and one index pair per match.
Each row of ``idxs`` is ``(index in desc1, index in desc2)``.
Choose the matcher by its acceptance rule, and interpret its first output as follows:

.. list-table:: Matcher selection and scores
   :header-rows: 1
   :widths: 28 42 30

   * - Matcher
     - Acceptance rule
     - First output
   * - :func:`match_nn`, :func:`match_mnn`
     - Nearest neighbour; MNN also requires the reverse match.
     - Distance; lower is better.
   * - :func:`match_snn`, :func:`match_smnn`
     - Nearest-to-second-nearest ratio test; SMNN requires mutual matches and tests both directions.
     - Ratio; lower is better. SMNN returns the larger ratio.
   * - :func:`match_fginn`
     - Ratio test against a geometrically inconsistent neighbour, using LAFs.
     - Ratio; lower is better.
   * - :func:`match_adalam`
     - Local affine consistency using LAFs.
     - Inverse confidence, or a squared-distance ratio in the no-seed fallback; lower is better.
   * - :class:`LightGlueMatcher`
     - Learned matching with pretrained weights for the selected feature.
     - Matching score; higher is better.

:class:`DescriptorMatcher` wraps NN, MNN, SNN and SMNN as a module.
:class:`GeometryAwareDescriptorMatcher` wraps FGINN and AdaLAM.
For padded batches of descriptor pairs, use :func:`match_smnn_batched`; its index rows include the batch index.
To detect, describe and match two images in one call, use :class:`LocalFeatureMatcher` with a descriptor-only matcher.

:class:`LightGlue` takes feature dictionaries and returns the full network outputs; :class:`OnnxLightGlue`
runs an exported model. :class:`LoFTR` matches images directly, without a detector.
To keep matches that agree with a geometric model, use :class:`~kornia.geometry.ransac.RANSAC`.

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
