.. _kornia_qwen3_vl:

Qwen3-VL
========

.. card::
    :link: https://github.com/QwenLM/Qwen3-VL

    **Qwen3-VL**
    ^^^
    **Abstract:** Qwen3-VL is the next iteration of Alibaba's open-source vision-language family.
    The dense releases (2B, 4B, and 8B parameters) couple a ViT vision tower with DeepStack feature
    fusion, a dynamic-resolution image preprocessor and a Qwen3 text decoder, supporting tasks such
    as visual question answering, optical character recognition, document understanding, grounded
    detection, and temporal video grounding.

    **Tasks:** Visual Question Answering, OCR, Visual Grounding, Video Understanding

    **Licence:** Apache-2.0

    +++
    **Authors:** Qwen Team

Configuration
-------------

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLConfig
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLVisionConfig
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLTextConfig
   :members:

Image preprocessor
------------------

The image preprocessor performs Qwen3-VL's dynamic-resolution resize policy, per-channel
normalisation, and patchification. The output is a flat
``(N_patches, in_channels * temporal_patch_size * patch_size**2)`` tensor paired with a
``grid_thw`` descriptor of shape ``(B, 3)`` that the vision tower consumes directly.

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLImageProcessor
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLImageProcessorConfig
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLPreprocessorOutput
   :members:

.. autofunction:: kornia.models.qwen3_vl.smart_resize

Vision tower
------------

The vision tower is a pre-norm ViT with Conv3d patch embedding, a learned absolute position
embedding (bilinearly interpolated to the input grid), 2D rotary positional embeddings, and a
DeepStack fusion mechanism that surfaces intermediate transformer-layer features through dedicated
patch mergers. The state-dict layout matches the official ``model.visual.*`` keys, so weights from
the published Qwen3-VL checkpoints can be loaded after the prefix is stripped.

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLVisionModel
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLVisionEncoderOutput
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLBlock
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLAttention
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLMLP
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLPatchEmbed
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLPatchMerger
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLRotaryEmbedding
   :members:

.. autofunction:: kornia.models.qwen3_vl.apply_rotary_pos_emb_vision

.. autofunction:: kornia.models.qwen3_vl.rotate_half

Loading official weights
------------------------

.. autofunction:: kornia.models.qwen3_vl.remap_hf_vision_state_dict
