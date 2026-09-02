MobileSAM
=========

.. rst-class:: kornia-badges

:bdg-primary:`Segmentation` :bdg-primary:`Visual prompting` :bdg-secondary:`Apache-2.0`

MobileSAM replaces SAM's ViT-H image encoder with a distilled TinyViT, making the model about 60× smaller while
keeping the same prompt encoder and mask decoder. In Kornia it is one more ``model_type`` of
:class:`~kornia.models.sam.SamConfig`, so it is used exactly like :doc:`SAM <segment_anything>` through the
:class:`~kornia.contrib.visual_prompter.VisualPrompter`.

Run it
------

.. code-block:: python

    import torch
    from kornia.io import load_image
    from kornia.models.sam import SamConfig
    from kornia.contrib.visual_prompter import VisualPrompter
    from kornia.geometry.boxes import Boxes
    from kornia.geometry.keypoints import Keypoints

    image = load_image("simba.png")  # (3, H, W) float in [0, 1]

    prompter = VisualPrompter(SamConfig("mobile_sam", pretrained=True))  # ~40 MB checkpoint
    prompter.set_image(image)  # encode once, query many times

    keypoints = Keypoints(torch.tensor([[[300.0, 90.0]]]))  # (K, N, 2): K prompts of N (x, y) points, in pixels
    prediction = prompter.predict(keypoints=keypoints, keypoints_labels=torch.tensor([[1]]))  # 1 = foreground
    best = prediction.binary_masks[0, prediction.scores.argmax()]  # (H, W) bool

    box = Boxes.from_tensor(torch.tensor([[[180.0, 20.0, 380.0, 240.0]]]), mode="xyxy")
    mask = prompter.predict(boxes=box, multimask_output=False).binary_masks[0, 0]

.. figure:: /_static/img/models/mobile_sam.jpg
   :align: center
   :alt: A cartoon lion cub with a point and a box prompt drawn on it (left), and the MobileSAM masks for the point prompt and the box prompt overlaid in green (centre, right).

   MobileSAM masks for the same point and box prompts used on the :doc:`SAM page <segment_anything>`, with the
   predicted IoU scores in the titles.

For the full prompter API, checkpoint loading and prompter-free usage, refer to :doc:`segment_anything`.

Paper
-----

.. card::
    :link: https://arxiv.org/abs/2306.14289

    **Faster Segment Anything: Towards Lightweight SAM for Mobile Applications**
    ^^^
    **Abstract:** Segment Anything Model (SAM) has attracted significant attention due to its impressive zero-shot transfer performance and high versatility for numerous vision applications (like image editing with fine-grained control). Many of such applications need to be run on resource-constraint edge devices, like mobile phones. In this work, we aim to make SAM mobile-friendly by replacing the heavyweight image encoder with a lightweight one. A naive way to train such a new SAM as in the original SAM paper leads to unsatisfactory performance, especially when limited training sources are available. We find that this is mainly caused by the coupled optimization of the image encoder and mask decoder, motivated by which we propose decoupled distillation. Concretely, we distill the knowledge from the heavy image encoder (ViT-H in the original SAM) to a lightweight image encoder, which can be automatically compatible with the mask decoder in the original SAM. The training can be completed on a single GPU within less than one day, and the resulting lightweight SAM is termed MobileSAM which is more than 60 times smaller yet performs on par with the original SAM. For inference speed, With a single GPU, MobileSAM runs around 10ms per image: 8ms on the image encoder and 4ms on the mask decoder. With superior performance, our MobileSAM is around 5 times faster than the concurrent FastSAM and 7 times smaller, making it more suitable for mobile applications. Moreover, we show that MobileSAM can run relatively smoothly on CPU. The code for our project is provided at https://github.com/ChaoningZhang/MobileSAM, with a demo showing that MobileSAM can run relatively smoothly on CPU.

    **Tasks:** Segmentation

    **Datasets:** SA-1B

    **Licence:** Apache 2.0

    +++
    **Authors:** Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, Choong Seon Hong
