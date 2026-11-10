Codec
=====

.. currentmodule:: kornia.enhance

A differentiable JPEG codec, to simulate compression artifacts inside a training loop. An integer image raises
:exc:`~kornia.core.exceptions.TypeCheckError`; convert 8-bit data with ``image.float() / 255`` first.

.. autofunction:: jpeg_codec_differentiable

.. autoclass:: JPEGCodecDifferentiable
