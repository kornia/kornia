HardNet
=======

.. rst-class:: kornia-badges

:bdg-primary:`Local feature descriptor` :bdg-secondary:`MIT`

HardNet turns a ``32×32`` grayscale patch into a 128-dimensional L2-normalised descriptor, the same size as SIFT
but trained with a hard-negative margin loss. :class:`~kornia.feature.HardNet` is the descriptor only: pair it with
a detector such as :class:`~kornia.feature.KeyNetDetector`, extract patches from the local affine frames and match
the descriptors with :func:`~kornia.feature.match_snn`.

Run it
------

.. code-block:: python

    import torch
    from kornia.color import rgb_to_grayscale
    from kornia.feature import HardNet, KeyNetDetector, extract_patches_from_pyramid, get_laf_center, match_snn
    from kornia.io import load_image

    gray1 = rgb_to_grayscale(load_image("church_1.png")[None])  # (1, 1, H, W)
    gray2 = rgb_to_grayscale(load_image("church_2.png")[None])

    detector = KeyNetDetector(pretrained=True, num_features=1500).eval()
    hardnet = HardNet(pretrained=True).eval()
    with torch.no_grad():
        lafs1, _ = detector(gray1)  # (1, N, 2, 3) local affine frames
        lafs2, _ = detector(gray2)
        patches1 = extract_patches_from_pyramid(gray1, lafs1)[0]  # (N, 1, 32, 32)
        patches2 = extract_patches_from_pyramid(gray2, lafs2)[0]
        desc1, desc2 = hardnet(patches1), hardnet(patches2)  # (N, 128), L2-normalised
        dists, idx = match_snn(desc1, desc2, th=0.9)  # (M, 1) distances, (M, 2) index pairs

    pts1 = get_laf_center(lafs1)[0, idx[:, 0]]  # (M, 2) matched (x, y) in image 1
    pts2 = get_laf_center(lafs2)[0, idx[:, 1]]

.. figure:: /_static/img/models/hardnet.jpg
   :align: center
   :alt: Eight 32 by 32 grayscale patches with their 128-dimensional descriptors shown as a heatmap, next to two church photographs joined by the matched keypoints.

   Eight of the KeyNet patches and their HardNet descriptors (left), and the second-nearest-neighbour ratio
   matches between the two views (right).

:class:`~kornia.feature.HardNet8` is the wider follow-up trained on more data, and
:class:`~kornia.feature.LocalFeature` bundles detector, orientation, shape and descriptor into one module
(see :class:`~kornia.feature.KeyNetHardNet`).

Paper
-----

.. card::
    :link: https://paperswithcode.com/paper/working-hard-to-know-your-neighbors-margins

    **HardNet: Working hard to know your neighbor's margins: Local descriptor learning loss**
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
