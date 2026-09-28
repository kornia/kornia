Image Registration
==================

.. meta::
   :description: Align images with differentiable registration and PyTorch autograd using Kornia's ImageRegistrator.

Image registration is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, data from different sensors, times, depths, or viewpoints. It is used in computer vision, medical imaging, and compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from these different measurements.

Learn more: `https://paperswithcode.com/task/image-registration <https://paperswithcode.com/task/image-registration>`_

..  youtube:: Re1q6vRfZac

We provide the :py:class:`~kornia.geometry.transform.image_registrator.ImageRegistrator` API, which you can use to
automatically align two images by direct optimization, leveraging PyTorch autograd.

.. code:: python

    import torch
    from kornia.geometry import ImageRegistrator

    img_src = torch.rand(1, 1, 32, 32)
    img_dst = torch.rand(1, 1, 32, 32)
    registrator = ImageRegistrator("similarity")
    homo = registrator.register(img_src, img_dst)  # (1, 3, 3) transform that warps img_src onto img_dst

Then, if you want to perform a more sophisticated process:

.. literalinclude:: ../_static/image_registration.py

To reproduce the same results as in the video shown above, you can go through our full tutorial using Colab, found `here <https://www.kornia.org/tutorials/nbs/image_registration.html>`_.
