SOLD2 (Line detection and matching)
...................................

.. card::
    :link: https://arxiv.org/abs/2104.03362

    **SOLD²: Self-supervised Occlusion-aware Line Description and Detection**
    ^^^
    **Abstract:** Compared to feature point detection and description, detecting and matching line segments offer additional challenges. Yet, line features represent a promising complement to points for multi-view tasks. Lines are indeed well-defined by the image gradient, frequently appear even in poorly textured areas and offer robust structural cues. We thus hereby introduce the first joint detection and description of line segments in a single deep network. Thanks to a self-supervised training, our method does not require any annotated line labels and can therefore generalize to any dataset. Our detector offers repeatable and accurate localization of line segments in images, departing from the wireframe parsing approach. Leveraging the recent progresses in descriptor learning, our proposed line descriptor is highly discriminative, while remaining robust to viewpoint changes and occlusions. We evaluate our approach against previous line detection and description methods on several multi-view datasets created with homographic warps as well as real-world viewpoint changes. Our full pipeline yields higher repeatability, localization accuracy and matching metrics, and thus represents a first step to bridge the gap with learned feature points methods.

    **Tasks:** Line detection, Line description, Line matching

    **Datasets:** Wireframe, YorkUrban, ETH3D

    **Conference:** CVPR 2021

    **Licence:** MIT

    +++
    **Authors:** Rémi Pautrat*, Juan-Ting Lin*, Viktor Larsson, Martin R. Oswald, Marc Pollefeys

.. image:: https://github.com/cvg/SOLD2/raw/main/assets/videos/demo_moving_camera.gif
   :align: center
