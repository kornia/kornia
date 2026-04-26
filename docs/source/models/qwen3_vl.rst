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

Module reference
----------------

The configuration dataclasses are scaffolded first; encoder, preprocessor, and core
model classes will be filled in by follow-up pull requests tracked in
`kornia#3622 <https://github.com/kornia/kornia/issues/3622>`_.

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLConfig
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLVisionConfig
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLProjectorConfig
   :members:

.. autoclass:: kornia.models.qwen3_vl.Qwen3VLTextConfig
   :members:
