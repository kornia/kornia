Codec
=====

.. currentmodule:: kornia.enhance

A differentiable JPEG codec, useful to simulate compression artifacts inside a training loop.
It accepts RGB ``(B, 3, H, W)`` tensors in ``[0, 1]`` and a quality tensor of
shape ``(1,)`` or ``(B,)``.

.. autofunction:: jpeg_codec_differentiable

.. autoclass:: JPEGCodecDifferentiable
