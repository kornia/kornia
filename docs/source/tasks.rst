Supported Applications
======================

Kornia provides from bottom to top granularity in order to implement Computer Vision related applications.

In this section, we showcase our high-level API in terms of abstraction for common Computer Vision algorithms
that can be used across different domains such as Robotics, Industrial applications or for the AR/VR industry.

Image Registration
------------------

Image registration is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, data from different sensors, times, depths, or viewpoints. It is used in computer vision, medical imaging, and compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from these different measurements.

Learn more @ PWC : `https://paperswithcode.com/task/image-registration <https://paperswithcode.com/task/image-registration>`_

..  youtube:: I-g3EhBIsDs

We provide the :py:class:`~kornia.geometry.transform.image_registrator.ImageRegistrator` API from which you can leverage to align automatically two images by a process of direct optimisation using the PyTorch Autograd differentiability.

.. code:: python

    from kornia.geometry import ImageRegistrator
    img_src = torch.rand(1, 1, 32, 32)
    img_dst = torch.rand(1, 1, 32, 32)
    registrator = ImageRegistrator('similarity')
    homo = registrator.register(img_src, img_dst)

Then, if you want to perform a more sophisticated process:

.. literalinclude:: _static/image_registration.py

To reproduce the same results as in the showed video you can go through or full tutorial using Colab found `here <https://kornia-tutorials.readthedocs.io/en/latest/image_registration.html>`_ .

Video Deblurring
----------------

**COMING SOON**

Image Matching
--------------

**COMING SOON**
