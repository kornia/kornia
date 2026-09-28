AffNet
======

.. meta::
   :description: Estimate affine-covariant local feature shapes with AffNet, Kornia's pretrained PyTorch model.

.. rst-class:: kornia-badges

:bdg-primary:`Affine shape estimation` :bdg-secondary:`MIT`

AffNet predicts the affine shape of a local region so that the same surface patch, seen from two viewpoints, is
normalised to the same square before it is described. :class:`~kornia.feature.LAFAffNetShapeEstimator` takes the
local affine frames (LAFs) of any detector and returns frames with the estimated ellipse in place of the circle.

Run it
------

.. code-block:: python

    import torch
    from kornia.color import rgb_to_grayscale
    from kornia.feature import KeyNetDetector, LAFAffNetShapeEstimator, laf_to_boundary_points
    from kornia.io import load_image

    gray = rgb_to_grayscale(load_image("church_1.png")[None])  # (1, 1, H, W)

    detector = KeyNetDetector(pretrained=True, num_features=32).eval()
    affnet = LAFAffNetShapeEstimator(pretrained=True).eval()
    with torch.no_grad():
        lafs, _ = detector(gray)  # (1, N, 2, 3) isotropic local affine frames
        lafs_affine = affnet(lafs, gray)  # (1, N, 2, 3) affine-covariant frames

    boundaries = laf_to_boundary_points(lafs_affine)  # (1, N, 50, 2) ellipse outlines in (x, y) pixels, for plotting

.. figure:: /_static/img/models/affnet.jpg
   :align: center
   :alt: The same church photograph twice: on the left with circular KeyNet regions, on the right with the elliptical regions AffNet estimated for them.

   The 32 strongest KeyNet regions before (circles) and after AffNet (ellipses that follow the local surface).

The estimated frames feed straight into :func:`~kornia.feature.extract_patches_from_pyramid` and a descriptor such
as :class:`~kornia.feature.HardNet`; :class:`~kornia.feature.KeyNetAffNetHardNet` wires the three together, and
:class:`~kornia.feature.PatchAffineShapeEstimator` is the raw patch-to-shape network.

Paper
-----

.. card::
    :link: https://paperswithcode.com/paper/repeatability-is-not-enough-learning-affine

    **AffNet: Repeatability Is Not Enough: Learning Affine Regions via Discriminability**
    ^^^
    **Abstract:** A method for learning local affine-covariant regions is presented. We show that maximizing geometric repeatability does not lead to local regions, a.k.a. features, that are reliably matched and this necessitates descriptor-based learning. We explore factors that influence such learning and registration: the loss function, descriptor type, geometric parametrization and the trade-off between matchability and geometric accuracy and propose a novel hard negative-constant loss function for learning of affine regions. The affine shape estimator -- AffNet -- trained with the hard negative-constant loss outperforms the state-of-the-art in bag-of-words image retrieval and wide baseline stereo. The proposed training process does not require precisely geometrically aligned patches.

    **Tasks:** Image Retrieval

    **Datasets:** Oxford5k, HPatches

    **Conference:** ECCV 2018

    **Licence:** MIT

    +++
    **Authors:** Dmytro Mishkin, Filip Radenovic, Jiri Matas

.. image:: https://raw.githubusercontent.com/ducha-aiki/affnet/master/imgs/graf16HesAffNet.jpg
   :align: center
