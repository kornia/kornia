.. _kornia_tiny_vit:

TinyViT
=======

.. rst-class:: kornia-badges

:bdg-primary:`Image classification` :bdg-primary:`Detection`

TinyViT is a family of small vision transformers (5M, 11M and 21M parameters) distilled from large teachers on
ImageNet-22k. Kornia's :class:`~kornia.models.tiny_vit.TinyViT` ships the ImageNet-1k classification heads, so it
works out of the box as a classifier, and its backbone is the image encoder behind :doc:`MobileSAM <mobile_sam>`.

Run it
------

.. code-block:: python

    import torch
    from kornia.io import load_image
    from kornia.geometry import resize
    from kornia.models.tiny_vit import TinyViT

    image = resize(load_image("panda.jpg")[None], (224, 224))  # (1, 3, 224, 224) float in [0, 1]

    model = TinyViT.from_config("5m", pretrained=True).eval()  # '5m', '11m' or '21m'; ImageNet-1k head
    with torch.no_grad():
        logits = model(image)  # (1, 1000)

    top5 = logits.softmax(-1).topk(5)
    print(top5.indices, top5.values)  # ImageNet class ids and probabilities

.. figure:: /_static/img/models/tiny_vit.jpg
   :align: center
   :alt: A panda photo next to a bar chart of the five most probable ImageNet classes, with giant panda far ahead.

   Top-5 ImageNet classes predicted by TinyViT-5M for the input on the left.

``pretrained`` also accepts ``'in22k'`` and, for ``'21m'``, the higher-resolution ``'in1k_384'`` and
``'in1k_512'`` checkpoints; pass ``img_size=`` to run at another resolution and the attention biases are
interpolated. The ImageNet-22k checkpoint carries a 21841-class head, so load it with
``TinyViT.from_config("5m", pretrained="in22k", num_classes=21841)``: with the default ``num_classes=1000`` the
head does not match, is reset to zeros with a warning and the model returns all-zero logits. For feature extraction
replace the head instead: ``model.head = torch.nn.Identity()`` makes the model return the pooled embedding.

Paper
-----

.. card::
    :link: https://arxiv.org/abs/2207.10666

    **TinyViT: Fast Pretraining Distillation for Small Vision Transformers**
    ^^^
    **Abstract:** Vision transformer (ViT) recently has drawn great attention in computer vision due to its remarkable model capability. However, most prevailing ViT models suffer from huge number of parameters, restricting their applicability on devices with limited resources. To alleviate this issue, we propose TinyViT, a new family of tiny and efficient small vision transformers pretrained on large-scale datasets with our proposed fast distillation framework. The central idea is to transfer knowledge from large pretrained models to small ones, while enabling small models to get the dividends of massive pretraining data. More specifically, we apply distillation during pretraining for knowledge transfer. The logits of large teacher models are sparsified and stored in disk in advance to save the memory cost and computation overheads. The tiny student transformers are automatically scaled down from a large pretrained model with computation and parameter constraints. Comprehensive experiments demonstrate the efficacy of TinyViT. It achieves a top-1 accuracy of 84.8% on ImageNet-1k with only 21M parameters, being comparable to Swin-B pretrained on ImageNet-21k while using 4.2 times fewer parameters. Moreover, increasing image resolutions, TinyViT can reach 86.5% accuracy, being slightly better than Swin-L while using only 11% parameters. Last but not the least, we demonstrate a good transfer ability of TinyViT on various downstream tasks. Code and models are available at https://github.com/microsoft/Cream/tree/main/TinyViT.

    **Tasks:** Image Classification, Object Detection

    **Datasets:** ImageNet, MS-COCO

    +++
    **Authors:**  Kan Wu, Jinnian Zhang, Houwen Peng, Mengchen Liu, Bin Xiao, Jianlong Fu, Lu Yuan

.. image:: https://github.com/microsoft/Cream/blob/main/TinyViT/.figure/framework.png?raw=true
   :align: center
