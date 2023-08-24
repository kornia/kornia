Image Matching
==============

Image matching is a process of finding pixel and region correspondences between two images of the same scene.
Such correspondences are useful for 3D reconstruction of the scene and relative camera pose estimation.
It is also known as "Wide baseline stereo" and you can read more about it at `Wide Baseline Stereo Blog <https://ducha-aiki.github.io/wide-baseline-stereo-blog/2021/01/09/wxbs-in-simple-terms.html>`_

We provide many modules and functions for the image matching: from building blocks like `local feature detectors <https://kornia.readthedocs.io/en/latest/feature.html#detectors>`_, `descriptors <https://kornia.readthedocs.io/en/latest/feature.html#descriptors>`_,
`descriptor matching <https://kornia.readthedocs.io/en/latest/feature.html#matching>`_, `geometric model estimation <https://kornia.readthedocs.io/en/latest/geometry.epipolar.html#fundamental>`_

However we recommend to start with high-level API, such as :py:class:`~kornia.feature.LoFTR` you can use to find correspondence between two images.

.. code:: python

    from kornia.feature import LoFTR

    matcher = LoFTR(pretrained="outdoor")
    input = {"image0": img1, "image1": img2}
    correspondences_dict = matcher(input)


.. image:: https://raw.githubusercontent.com/kornia/data/main/matching/matching_loftr.jpg

You also can go through or full tutorial using Colab found `here <https://kornia.github.io/tutorials/nbs/image_matching.html>`_.
