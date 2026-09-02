Kornia Models
=============

.. meta::
   :description: The deep learning models shipped with Kornia, grouped by task, with their papers, tasks and licences: RT-DETR, YuNet, Segment Anything, MobileSAM, EfficientViT, ViT, MobileViT, TinyViT, LoFTR, HardNet, AffNet, SOLD2, DexiNed and DeFMO.

Kornia ships a curated set of model architectures, each wrapped as a regular ``nn.Module``. Most of them come with
pretrained weights that are downloaded on first use; MobileViT ships as an architecture only, for training or for
loading your own checkpoint. The pages below summarize each model's paper, task and licence. The builders that
construct them are documented on the :doc:`kornia.models </models>` API page, and the local-feature models under
:doc:`kornia.feature </feature>`.

.. list-table::
   :header-rows: 1
   :widths: 22 28 30 20

   * - Model
     - Task
     - Venue
     - Licence
   * - :doc:`RT-DETR <rt_detr>`
     - Object detection
     - arXiv 2023
     - Apache-2.0
   * - :doc:`YuNet <yunet>`
     - Face detection
     - IEEE TIP 2021
     - Apache-2.0
   * - :doc:`Segment Anything (SAM) <segment_anything>`
     - Promptable segmentation
     - ICCV 2023
     - Apache-2.0
   * - :doc:`MobileSAM <mobile_sam>`
     - Promptable segmentation (lightweight)
     - arXiv 2023
     - Apache-2.0
   * - :doc:`EfficientViT <efficient_vit>`
     - Segmentation, classification, detection backbone
     - ICCV 2023
     - Apache-2.0
   * - :doc:`Vision Transformer (ViT) <vit>`
     - Image classification backbone
     - ICLR 2021
     - Apache-2.0
   * - :doc:`MobileViT <vit_mobile>`
     - Image classification backbone (mobile)
     - ICLR 2022
     - --
   * - :doc:`TinyViT <tiny_vit>`
     - Image classification backbone (small)
     - ECCV 2022
     - --
   * - :doc:`LoFTR <loftr>`
     - Image matching (detector-free)
     - CVPR 2021
     - Apache-2.0
   * - :doc:`HardNet <hardnet>`
     - Local feature descriptor
     - NeurIPS 2017
     - MIT
   * - :doc:`AffNet <affnet>`
     - Affine shape estimation for local features
     - ECCV 2018
     - MIT
   * - :doc:`SOLD2 <sold2>`
     - Line detection and matching
     - CVPR 2021
     - MIT
   * - :doc:`DexiNed <dexined>`
     - Edge detection
     - WACV 2020
     - MIT
   * - :doc:`DeFMO <defmo>`
     - Enhancement -- video deblurring of fast-moving objects
     - CVPR 2021
     - Apache-2.0

.. toctree::
   :caption: Object detection
   :hidden:

   RT-DETR — real-time detector <rt_detr>
   YuNet — face detector <yunet>

.. toctree::
   :caption: Segmentation
   :hidden:

   SAM — promptable masks <segment_anything>
   MobileSAM — lightweight SAM <mobile_sam>
   EfficientViT — fast backbone <efficient_vit>

.. toctree::
   :caption: Image classification
   :hidden:

   ViT — transformer backbone <vit>
   MobileViT — mobile backbone <vit_mobile>
   TinyViT — distilled backbone <tiny_vit>

.. toctree::
   :caption: Local features and matching
   :hidden:

   LoFTR — detector-free matcher <loftr>
   HardNet — patch descriptor <hardnet>
   AffNet — affine shape estimator <affnet>
   SOLD2 — line detector and matcher <sold2>

.. toctree::
   :caption: Edge detection
   :hidden:

   DexiNed — fine-grained edges <dexined>

.. toctree::
   :caption: Enhance
   :hidden:

   DeFMO — video deblur <defmo>
