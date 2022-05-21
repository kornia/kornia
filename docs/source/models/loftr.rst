LoFTR (matching)
................

.. card::
    :link: https://paperswithcode.com/paper/loftr-detector-free-local-feature-matching

    **LoFTR: Detector-Free Local Feature Matching with Transformers**
    ^^^
    **Abstract:** We present a novel method for local image feature matching. Instead of performing image feature detection, description, and matching sequentially, we propose to first establish pixel-wise dense matches at a coarse level and later refine the good matches at a fine level. In contrast to dense methods that use a cost volume to search correspondences, we use self and cross attention layers in Transformer to obtain feature descriptors that are conditioned on both images. The global receptive field provided by Transformer enables our method to produce dense matches in low-texture areas, where feature detectors usually struggle to produce repeatable interest points. The experiments on indoor and outdoor datasets show that LoFTR outperforms state-of-the-art methods by a large margin. LoFTR also ranks first on two public benchmarks of visual localization among the published methods.

    **Tasks:** Local Feature Matching, Visual Localisation

    **Datasets:** ScanNet, HPatches, MegaDepth, InLoc

    **Conference:** CVPR 2021

    **Licence:** Apache-2.0

    +++
    **Authors:** Jiaming Sun*, Zehong Shen*, Yu'ang Wang*, Hujun Bao, Xiaowei Zhou

.. image:: https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/loftr-github-demo.gif
   :align: center
