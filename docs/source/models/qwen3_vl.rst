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

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLProjectorConfig
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLTextConfig
   :members:

Vision encoder
--------------

The vision tower is a pre-norm ViT with 2D rotary positional embeddings and a
DeepStack fusion mechanism that surfaces intermediate transformer-layer features
to downstream projectors. The core multimodal model class arrives in a follow-up
pull request tracked in
`kornia#3622 <https://github.com/kornia/kornia/issues/3622>`_.

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLVisionTransformer
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLVisionEncoderOutput
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLEncoder
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLLayer
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLAttention
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLMLP
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLPatchEmbed
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLRotaryEmbedding
   :members:

.. autofunction:: kornia.models.qwen3_vl.apply_rotary_pos_emb

Image preprocessor
------------------

The image preprocessor performs Qwen3-VL's dynamic-resolution resize policy
followed by per-channel normalization. Output dimensions are constrained to
multiples of ``patch_size * spatial_merge_size`` so each merged token covers a
whole patch grid cell, and the total pixel count is clamped to a configurable
``[min_pixels, max_pixels]`` band.

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLImageProcessor
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLImageProcessorConfig
   :members:

.. autofunction:: kornia.models.qwen3_vl.smart_resize
