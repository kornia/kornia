.. _kornia_vit:

Vision Transformer (ViT)
========================

.. rst-class:: kornia-badges

:bdg-primary:`Image classification` :bdg-secondary:`Apache-2.0`

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


Kornia-ViT
----------

We provide the operator :py:class:`~kornia.models.vit.VisionTransformer` that is meant to be used across tasks.
One can use the *ViT* in Kornia as follows:

.. code:: python

    import torch
    from kornia.models.vit import VisionTransformer

    img = torch.rand(1, 3, 224, 224)
    vit = VisionTransformer(image_size=224, patch_size=16)
    out = vit(img)  # (1, 197, 768): the class token followed by 196 patch tokens

Usage
~~~~~

``kornia-vit`` does not include any classification head. The backbone returns one embedding per token,
``(B, 1 + N, hidden_dim)``, with the class token first; a classifier reads the class token and adds a linear
head using standard PyTorch modules:

.. code:: python

    import torch
    import torch.nn as nn
    from kornia.models.vit import VisionTransformer


    class Classifier(nn.Module):
        def __init__(self, num_classes: int = 1000) -> None:
            super().__init__()
            self.backbone = VisionTransformer(image_size=224, patch_size=16)
            self.head = nn.Linear(768, num_classes)  # 768 is the default hidden_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tokens = self.backbone(x)         # (B, 197, 768)
            return self.head(tokens[:, 0])    # (B, num_classes), from the class token


    img = torch.rand(1, 3, 224, 224)
    out = Classifier()(img)   # (1, 1000)
    scores = out.argmax(-1)   # (1,)

Beyond simple image classification, the API is flexible enough to design your own pipelines, e.g.
for multi-task learning, object detection or segmentation. We show an example of a multi-task
module with two different classification heads:

.. code:: python

    import torch
    import torch.nn as nn
    from kornia.models.vit import VisionTransformer

    class MultiTaskTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = VisionTransformer(
                image_size=224, patch_size=16)
            self.head1 = nn.Linear(768, 10)  # Example: 768 is the default hidden_dim
            self.head2 = nn.Linear(768, 50)

        def forward(self, x: torch.Tensor):
            cls_token = self.transformer(x)[:, 0]  # (B, 768)
            return {
                "head1": self.head1(cls_token),
                "head2": self.head2(cls_token),
            }

.. tip::
    More heads, examples and a training API are coming soon!
