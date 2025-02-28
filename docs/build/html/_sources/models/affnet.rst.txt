Affnet (detection)
..................

.. card::
    :link: https://paperswithcode.com/paper/repeatability-is-not-enough-learning-affine

    **Affnet: Repeatability Is Not Enough: Learning Affine Regions via Discriminability**
    ^^^
    **Abstract:** A method for learning local affine-covariant regions is presented. We show that maximizing geometric repeatability does not lead to local regions, a.k.a features,that are reliably matched and this necessitates descriptor-based learning. We explore factors that influence such learning and registration: the loss function, descriptor type, geometric parametrization and the trade-off between matchability and geometric accuracy and propose a novel hard negative-constant loss function for learning of affine regions. The affine shape estimator -- AffNet -- trained with the hard negative-constant loss outperforms the state-of-the-art in bag-of-words image retrieval and wide baseline stereo. The proposed training process does not require precisely geometrically aligned patches.

    **Tasks:** Image Retrieval

    **Datasets:** Oxford5k, HPatches

    **Conference:** ECCV 2018

    **Licence:** MIT

    +++
    **Authors:** Dmytro Mishkin, Filip Radenovic, Jiri Matas

.. image:: https://raw.githubusercontent.com/ducha-aiki/affnet/master/imgs/graf16HesAffNet.jpg
   :align: center
