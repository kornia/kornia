Hardnet (descriptor)
....................

.. card::
    :link: https://paperswithcode.com/paper/working-hard-to-know-your-neighbors-margins

    **Hardnet:  Working hard to know your neighbor's margins: Local descriptor learning loss**
    ^^^
    **Abstract:** We introduce a novel loss for learning local feature descriptors which is inspired by the Lowe's matching criterion for SIFT. We show that the proposed loss that maximizes the distance between the closest positive and closest negative patch in the batch is better than complex regularization methods; it works well for both shallow and deep convolution network architectures. Applying the novel loss to the L2Net CNN architecture results in a compact descriptor -- it has the same dimensionality as SIFT (128) that shows state-of-art performance in wide baseline stereo, patch verification and instance retrieval benchmarks. It is fast, computing a descriptor takes about 1 millisecond on a low-end GPU

    **Tasks:** Image Retrieval, Patch Matching

    **Datasets:** Oxford5k, HPatches, Oxford-Affine

    **Conference:** NeurIPS 2017

    **Licence:** MIT

    +++
    **Authors:**  Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas

.. image:: https://raw.githubusercontent.com/DagnyT/hardnet/master/img/hardnet_hpatches.png
   :align: center
