EfficientViT
============

.. meta::
   :description: Use EfficientViT pretrained PyTorch vision backbones for image classification, detection and segmentation with Kornia.

.. rst-class:: kornia-badges

:bdg-primary:`Segmentation` :bdg-primary:`Classification` :bdg-primary:`Detection` :bdg-secondary:`Apache-2.0`

EfficientViT is a high-resolution vision backbone built on multi-scale linear attention. Kornia ships the ImageNet
pretrained backbones (``b1``, ``b2``, ``b3`` at 224/256/288 px) as :class:`~kornia.models.efficient_vit.EfficientViT`.
The model returns a dictionary with the input and the feature map after each stage, so it plugs into any dense
prediction or classification head.

Run it
------

.. code-block:: python

    import torch
    from kornia.io import load_image
    from kornia.geometry import resize
    from kornia.models.efficient_vit import EfficientViT, EfficientViTConfig

    image = resize(load_image("panda.jpg")[None], (224, 224))  # (1, 3, 224, 224) float in [0, 1]

    model = EfficientViT.from_config(EfficientViTConfig.from_pretrained("b1", 224)).eval()
    with torch.no_grad():
        features = model(image)  # dict: "input", "stage0" ... "stage4", "stage_final"

    for name, feat in features.items():
        print(name, tuple(feat.shape))  # stage0 (1, 16, 112, 112) ... stage_final (1, 256, 7, 7)

.. figure:: /_static/img/models/efficient_vit.jpg
   :align: center
   :alt: The input image followed by six feature-map heatmaps of decreasing resolution, from 112x112 down to 7x7.

   Mean absolute activation of every stage of ``b1`` at 224 px: the resolution halves and the channel count doubles
   from ``stage0`` (16 × 112 × 112) to ``stage_final`` (256 × 7 × 7).

The backbone carries no classification head; global-average-pool ``stage_final`` and add an ``nn.Linear`` for
classification, or feed the pyramid to a segmentation or detection neck.

Paper
-----

.. card::
    :link: https://arxiv.org/abs/2205.14756

    **EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction**
    ^^^
    **Abstract:** High-resolution dense prediction enables many appealing real-world applications, such as computational photography, autonomous driving, etc. However, the vast computational cost makes deploying state-of-the-art high-resolution dense prediction models on hardware devices difficult. This work presents EfficientViT, a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel convolution, or complicated topology structure to obtain good performances, our multi-scale linear attention achieves the global receptive field and multi-scale learning (two desirable features for high-resolution dense prediction) with only lightweight and hardware-efficient operations. As such, EfficientViT delivers remarkable performance gains over previous state-of-the-art models with significant speedup on diverse hardware platforms, including mobile CPU, edge GPU, and cloud GPU. Without performance loss on Cityscapes, our EfficientViT provides up to 13.9x and 6.2x GPU latency reduction over SegFormer and SegNeXt, respectively. For super-resolution, EfficientViT delivers up to 6.4x speedup over Restormer while providing 0.11dB gain in PSNR. For Segment Anything, EfficientViT delivers similar zero-shot image segmentation quality as ViT-Huge with 84x higher throughput on GPU. Code: this https URL.

    **Tasks:** Classification, Segmentation, Detection

    **Licence:** Apache 2.0

    +++
    **Authors:** Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
