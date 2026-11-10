.. _kornia_tiny_vit:

TinyViT
=======

.. meta::
   :description: Use TinyViT pretrained PyTorch vision transformers for image classification and as Kornia's MobileSAM encoder.

.. rst-class:: kornia-badges

:bdg-primary:`Image classification` :bdg-primary:`Detection`

TinyViT is a family of small vision transformers (5M, 11M and 21M parameters) distilled from large teachers on
ImageNet-22k. Kornia's :class:`~kornia.models.tiny_vit.TinyViT` ships the ImageNet-1k classification heads, so it
works out of the box as a classifier, and its backbone is the image encoder behind :doc:`MobileSAM <mobile_sam>`.

Run it
------

.. code-block:: python

    import torch
    from kornia.enhance import normalize
    from kornia.io import load_image
    from kornia.geometry import resize
    from kornia.models.tiny_vit import TinyViT

    image = resize(load_image("panda.jpg")[None], (224, 224))  # (1, 3, 224, 224) float in [0, 1]
    # the checkpoints were trained on images normalized with the ImageNet mean and standard deviation
    image = normalize(image, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))

    model = TinyViT.from_config("5m", pretrained=True).eval()  # '5m', '11m' or '21m'; ImageNet-1k head
    with torch.no_grad():
        logits = model(image)  # (1, 1000)

    top5 = logits.softmax(-1).topk(5)
    print(top5.indices, top5.values)  # ImageNet class ids and probabilities

.. figure:: /_static/img/models/tiny_vit.jpg
   :align: center
   :alt: A panda photo next to a bar chart of the five most probable ImageNet classes, with giant panda first.

   Top-5 ImageNet classes predicted by TinyViT-5M for the input on the left.

``pretrained`` also accepts ``'in22k'`` and, for ``'21m'``, the higher-resolution ``'in1k_384'`` and
``'in1k_512'`` checkpoints. ``pretrained=True`` picks them for an ``img_size`` of at least 384 or 512, and they are
built with the windows they were trained with, ``window_sizes=[12, 12, 24, 12]`` and ``[16, 16, 32, 16]``; named
explicitly, they build at the ``img_size`` you pass (224 by default), so pass ``img_size=384`` or ``512`` to run
them at their resolution. Every other model keeps the 224-px windows ``[7, 7, 14, 7]`` at any ``img_size``, unless
you pass ``window_sizes=``, and the attention biases are interpolated only when the windows differ from the
checkpoint's. ``img_size`` must be a multiple of 32 and the input ``img_size`` square (sides down to
``img_size - 7`` round to the same token grid; other sizes raise ``ValueError``), and accuracy is not preserved away
from a checkpoint's resolution: on the panda above, TinyViT-5M's probability of the right class drops from 0.37 at
224 px to 0.02 at ``img_size=256``. The ImageNet-22k checkpoint carries a 21841-class head, so load it with
``TinyViT.from_config("5m", pretrained="in22k", num_classes=21841)``: with the default ``num_classes=1000`` the
head does not match, is reset to zeros with a warning, the model returns all-zero logits, and it stays in training
mode as a backbone to fine-tune. For feature extraction replace the head and switch to evaluation mode:
``model.head = torch.nn.Identity()`` makes the model return the pooled embedding, and ``model.eval()`` makes it
independent of the rest of the batch.

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
