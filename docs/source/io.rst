kornia.io
=========

.. currentmodule:: kornia.io

Package to load and save image data.

The package internally implements `kornia_rs <https://github.com/kornia/kornia-rs>`_ which contains a low level implementation
for Computer Vision in the `Rust <https://www.rust-lang.org/>`_ language. In addition, we implement the `DLPack <https://github.com/dmlc/dlpack>`_ protocol
natively in Rust to reduce the memory footprint during the decoding and types conversion.

.. tip::
    You need to ``pip install kornia_rs`` to use this package. For now we only support Linux platforms.
    Contact us or sponsor the project for more support (mac, win, rust, c++, video and camera). See:
    `https://opencollective.com/kornia <https://opencollective.com/kornia>`_

.. note::
    The package needs at least PyTorch 1.10.0 installed.

.. code-block:: python

    import kornia as K
    from kornia.io import ImageLoadType
    from kornia.core import Tensor

    img: Tensor = K.io.load_image(file_path, ImageLoadType.UNCHANGED, device="cuda")
    # will load CxHxW / in the original format in "cuda"

    img: Tensor = K.io.load_image(file_path, ImageLoadType.RGB8, device="cpu")
    # will load 3xHxW / in torch.uint in range [0,255] in "cpu"

    img: Tensor = K.io.load_image(file_path, ImageLoadType.GRAY8, device="cuda")
    # will load 1xHxW / in torch.uint8 in range [0,255] in "cuda"

    img: Tensor = K.io.load_image(file_path, ImageLoadType.GRAY32, device="cpu")
    # will load 1xHxW / in torch.float32 in range [0,1] in "cpu"

    img: Tensor = K.io.load_image(file_path, ImageLoadType.RGB32, device="cuda")
    # will load 3xHxW / in torch.float32 in range [0,1] in "cuda"

.. autofunction:: load_image
.. autofunction:: write_image

.. autoclass:: ImageLoadType
    :members:
    :undoc-members:
