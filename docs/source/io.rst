kornia.io
=========

.. currentmodule:: kornia.io

Package to load and save image data.

The package internally implements `kornia_rs <https://github.com/kornia/kornia-rs>`_ which contains a low level implementation
for Computer Vision in the `Rust <https://www.rust-lang.org/>`_ language. In addition, we implement the `DLPack <https://github.com/dmlc/dlpack>`_ protocol
natively in Rust to reduce the memory footprint during the decoding and types conversion.

.. tip::
    You need to ``pip install kornia_rs`` to use this package.

.. code-block:: python

    import kornia as K
    from kornia.core import Tensor

    img: Tensor = K.io.load_image(file_path, ImageType.UNCHANGED, "cuda")
    # it will load a 3xHxW / in the original format in "cuda"

    img: Tensor = K.io.load_image(file_path, ImageType.RGB8, "cpu")
    # it will load a 3xHxW / in torch.uint in range [0,255] in "cpu"

    img: Tensor = K.io.load_image(file_path, ImageType.GRAY8, "cuda")
    # it will load a 1xHxW / in torch.uint8 in range [0,255] in "cuda"

    img: Tensor = K.io.load_image(file_path, ImageType.GRAY32, "cpu")
    # it will load a 1xHxW / in torch.float32 in range [0,1] in "cpu"

    img: Tensor = K.io.load_image(file_path, ImageType.RGB32, "cuda")
    # it will load a 3xHxW / in torch.float32 in range [0,1] in "cuda"

.. autofunction:: load_image

.. autoclass:: ImageType
    :members:
    :undoc-members:
