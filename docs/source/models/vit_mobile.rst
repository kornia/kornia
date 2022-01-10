.. _kornia_vit_mobile:

MobileViT
.........

.. card::
    :link: https://arxiv.org/abs/2110.02178

    **MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer**
    ^^^
    **Abstract:** Light-weight convolutional neural networks (CNNs) are the de-facto for mobile vision tasks. Their spatial inductive biases allow them to learn representations with fewer parameters across different vision tasks. However, these networks are spatially local. To learn global representations, self-attention-based vision trans-formers (ViTs) have been adopted. Unlike CNNs, ViTs are heavy-weight. In this paper, we ask the following question: is it possible to combine the strengths of CNNs and ViTs to build a light-weight and low latency network for mobile vision tasks? Towards this end, we introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than Mo-bileNetv3 for a similar number of parameters.

    **Tasks:** Image Classification, Object Detection, Semantic Segmentation

    **Datasets:** ImageNet, MS-COCO, PASCAL VOC

    +++
    **Authors:**  Sachin Mehta, Mohammad Rastegari

.. image:: https://user-images.githubusercontent.com/67839539/136470152-2573529e-1a24-4494-821d-70eb4647a51d.png
   :align: center


Kornia-MobileViT
----------------

We provide :py:class:`~kornia.contrib.MobileViT` which can be used for many downstream tasks, e.g., classification, object detection and semantic segmentation.
One can use the *MobileViT* in Kornia as follows:

.. code:: python

    img = torch.rand(1, 3, 256, 256)
    mvit = MobileViT(mode='xxs')
    out = mvit(img)


Usage
~~~~~

Similar to ``Kornia-ViT``, ``Kornia-MobileViT`` does not include any classification head. But you can add it simply by doing:

.. code:: python

    import torch.nn as nn
    import kornia.contrib as K

    classifier = nn.Sequential(
        K.MobileViT(mode='xxs'),
        nn.AvgPool2d(256 // 32, 1),
        nn.Flatten(),
        nn.Linear(320, 1000)
    )

    img = torch.rand(1, 3, 256, 256)
    out = classifier(img)     # 1x1000
