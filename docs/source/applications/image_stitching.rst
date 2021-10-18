Image Stitching
============================

Image stitching is the process of combining multiple images with overlapping fields of view to produce a segmented panorama. Here, we provide :py:class:`~kornia.contrib.ImageStitcher` to easily stitch a number of images.

.. image:: https://raw.githubusercontent.com/kornia/data/main/matching/stitch_before.png

.. code:: python

    from kornia.contrib import ImageStitcher

    matcher = KF.LoFTR(pretrained='outdoor')
    IS = ImageStitcher(matcher, estimator='ransac').cuda()
    # NOTE: it would require a large CPU memory if many images.
    with torch.no_grad():
        out = IS(*imgs)

.. image:: https://raw.githubusercontent.com/kornia/data/main/matching/stitch_after.png
