.. _kornia_vit:

Vision Transformer (ViT)
========================

.. rst-class:: kornia-badges

:bdg-primary:`Image classification` :bdg-secondary:`Apache-2.0`

:class:`~kornia.models.vit.VisionTransformer` is a plain ViT encoder: the image is cut into ``patch_size`` squares,
each patch becomes a token, a class token is prepended and a stack of transformer blocks mixes them. The module
ships as an **architecture only** (no pretrained weights, no classification head) and returns one embedding per token,
``(B, 1 + N, embed_dim)``, with the class token first.

Run it
------

.. code-block:: python

    import torch
    from kornia.models.vit import VisionTransformer

    vit = VisionTransformer(image_size=224, patch_size=16)  # ViT-B/16 layout, random init

    image = torch.rand(1, 3, 224, 224)
    tokens = vit(image)  # (1, 197, 768): class token + 14×14 patch tokens
    cls, patches = tokens[:, 0], tokens[:, 1:]  # (1, 768) and (1, 196, 768)

.. figure:: /_static/img/models/vit.jpg
   :align: center
   :alt: A panda photo with the 16-pixel patch grid drawn on it, a 14x14 heatmap of patch-token norms, and a heatmap of the output token embeddings.

   The 14×14 patch grid a 224 px image is split into, the norm of each output patch token on that grid, and the
   first 96 dimensions of all 197 output tokens. The weights are randomly initialised, so the values are not
   meaningful until you train the model or load a checkpoint.

Add a head
----------

A classifier reads the class token and adds a linear layer:

.. code-block:: python

    import torch.nn as nn


    class Classifier(nn.Module):
        def __init__(self, num_classes: int = 1000) -> None:
            super().__init__()
            self.backbone = VisionTransformer(image_size=224, patch_size=16)
            self.head = nn.Linear(768, num_classes)  # 768 is the default embed_dim

        def forward(self, x):
            return self.head(self.backbone(x)[:, 0])  # (B, num_classes) from the class token

The same pattern gives multi-task or dense heads: keep the class token for image-level outputs, or reshape the
196 patch tokens back to a ``14 × 14`` map for detection and segmentation necks. The constructor exposes
``embed_dim``, ``depth``, ``num_heads``, ``dropout_rate`` and ``attention_dropout_rate`` to size the model.

Paper
-----

.. card::
    :link: https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1

    **ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
    ^^^
    **Abstract:** While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc. ), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

    **Tasks:** Image Classification, Fine-Grained Image Classification, Document Image Classification

    **Datasets:** CIFAR-10, ImageNet, CIFAR-100

    **Conference:** ICLR 2021

    **Licence:** Apache-2.0

    +++
    **Authors:**  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

.. image:: https://github.com/google-research/vision_transformer/raw/main/vit_figure.png
   :align: center
