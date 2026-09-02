Image Stitching
===============

Image stitching is the process of combining multiple images with overlapping fields of view to produce a panorama. Here, we provide :py:class:`~kornia.contrib.image_stitching.ImageStitcher` to easily stitch a number of images.

.. image:: https://raw.githubusercontent.com/kornia/data/main/matching/stitch_before.png
   :alt: Input images to be stitched

Learn more: https://paperswithcode.com/task/image-stitching/

.. code:: python

    import torch
    import kornia.feature as KF
    from kornia.contrib import ImageStitcher
    from kornia.io import ImageLoadType, load_image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgs = [load_image(p, ImageLoadType.RGB32, device=device)[None] for p in ("left.jpg", "right.jpg")]

    matcher = KF.LoFTR(pretrained="outdoor")
    stitcher = ImageStitcher(matcher, estimator="ransac").to(device)
    # NOTE: stitching many images at once can require a lot of memory.
    with torch.no_grad():
        panorama = stitcher(*imgs)  # (1, 3, H, W)

.. image:: https://raw.githubusercontent.com/kornia/data/main/panorama/out_panorama.jpg
   :alt: Stitched panorama

Explore with your data: https://colab.research.google.com/github/kornia/tutorials/blob/master/source/image_stitching.ipynb
