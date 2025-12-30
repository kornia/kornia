.. _kornia_vit:

Vision Transformer (ViT)
........................

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

    img = torch.rand(1, 3, 224, 224)
    vit = VisionTransformer(image_size=224, patch_size=16)
    out = vit(img)

Usage
~~~~~

``kornia-vit`` does not include any classification head.
You can easily add your own classification head using standard PyTorch modules.

.. code:: python

    import torch.nn as nn
    from kornia.models import VisionTransformer

    classifier = nn.Sequential(
        VisionTransformer(image_size=224, patch_size=16),
        nn.Linear(768, 1000)  # Example: 768 is the default hidden_dim, 1000 is num_classes
    )

    img = torch.rand(1, 3, 224, 224)
    out = classifier(img)     # BxN
    scores = out.argmax(-1)   # B

In addition to create simple image classification, our API is flexible enough to design your pipelines e.g
to solve problems for multi-task, object detection, segmentation, etc. We show an example of a multi-task
class with two different classification heads:

.. code:: python

    from kornia.models import VisionTransformer

    class MultiTaskTransfornmer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = VisionTransformer(
                image_size=224, patch_size=16)
            self.head1 = nn.Linear(768, 10)  # Example: 768 is the default hidden_dim
            self.head2 = nn.Linear(768, 50)

        def forward(self, x: torch.Tensor):
            out = self.transformer(x)
            return {
                "head1": self.head1(out),
                "head2": self.head2(out),
            }

.. tip::
    More heads, examples and a training API soon !!
