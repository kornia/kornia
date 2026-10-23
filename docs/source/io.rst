kornia.io
=========

.. meta::
   :description: The kornia.io package loads and saves images directly as PyTorch tensors. It is backed by kornia-rs, a low-level Rust computer vision library with wheels for Linux, macOS and Windows.

.. currentmodule:: kornia.io

Package to load and save image data directly as tensors.

The package is backed by `kornia-rs <https://github.com/kornia/kornia-rs>`_, a low-level computer vision
library written in `Rust <https://www.rust-lang.org/>`_ that is installed automatically as a dependency of
Kornia (wheels are available for Linux, macOS and Windows). kornia-rs decodes into a NumPy array that the tensor
can reuse: ``RGB8`` of an RGB PNG or JPEG loaded on the CPU, for example, is a view of that array, not a copy.

.. code-block:: python

    import torch
    import kornia as K
    from kornia.io import ImageLoadType

    file_path = "image.jpg"

    img = K.io.load_image(file_path, ImageLoadType.UNCHANGED, device="cuda")
    # (C, H, W) in the file's original dtype and channel count, on "cuda"

    img = K.io.load_image(file_path, ImageLoadType.RGB8, device="cpu")
    # (3, H, W) torch.uint8 in [0, 255], on "cpu"

    img = K.io.load_image(file_path, ImageLoadType.GRAY8, device="cuda")
    # (1, H, W) torch.uint8 in [0, 255], on "cuda"

    img = K.io.load_image(file_path, ImageLoadType.GRAY32, device="cpu")
    # (1, H, W) torch.float32 in [0, 1], on "cpu"

    img = K.io.load_image(file_path, ImageLoadType.RGB32, device="cuda")
    # (3, H, W) torch.float32 in [0, 1], on "cuda"

    K.io.write_image("copy.jpg", (img * 255).round().to(torch.uint8).cpu())  # JPEG takes uint8 (C, H, W)

.. tip::
    Most Kornia operators expect a batched ``(B, C, H, W)`` float tensor in ``[0, 1]``; load with
    ``ImageLoadType.RGB32`` and add the batch dimension with ``img[None]``.

:ref:`Files and arrays <image-io-conventions>` tabulates the load types and the formats :func:`write_image` takes,
and maps OpenCV's and Pillow's readers and writers onto them.

.. autofunction:: load_image
.. autofunction:: write_image
.. autofunction:: get_sample_images

.. autoclass:: ImageLoadType
    :members:
    :undoc-members:
